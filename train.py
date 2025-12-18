import argparse
import csv
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import RDeepONetH5
from models import RDeepONetV2
from training_utils import build_regression_loss, build_scheduler, resolve_gradient_clip, WarmupCosineScheduler, loss_tensor


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_norm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = cfg['data'].get('normalization', {})
    tl_min = float(n.get('tl_min', 40.0))
    tl_max = float(n.get('tl_max', 120.0))
    # Provide sensible defaults for freq/zs to match data factory ranges
    return {
        'tl_db': {'min': tl_min, 'max': tl_max},
        'freq': {'f_min': 20.0, 'f_max': 10000.0},
        'zs': {'denom': 5000.0},
    }


def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg['data']
    norm_cfg = build_norm_cfg(cfg)

    split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    mode = data_cfg.get('mode', 'coord')
    pts = int(data_cfg.get('pts_per_map', 4096))

    sampler_cfg = data_cfg.get('sampler', {})
    train_ds = RDeepONetH5(data_cfg['path'], 'train', split_ratio, mode, pts, norm_cfg, sampler_cfg)
    val_ds   = RDeepONetH5(data_cfg['path'], 'val',   split_ratio, mode, pts, norm_cfg, sampler_cfg)

    tcfg = cfg['training']
    pin_mem = (tcfg.get('accelerator', 'cuda') == 'cuda')

    loader_args = dict(batch_size=int(tcfg.get('batch_size', 4)),
                       num_workers=int(tcfg.get('num_workers', 4)),
                       pin_memory=pin_mem,
                       shuffle=True)

    return (DataLoader(train_ds, **loader_args),
            DataLoader(val_ds, **loader_args))


def build_model(cfg: Dict[str, Any]) -> RDeepONetV2:
    m = cfg['model']
    K = int(m.get('final_projection_dim', 256))
    trunk_hidden = int(m['trunk'].get('hidden_dim', 256))
    trunk_layers = int(m['trunk'].get('num_layers', 4))
    branch_pretrained = bool(m['branch_cnn'].get('pretrained', True))
    cond_hidden = int(m['branch_cond'].get('hidden_dim', 128))
    cond_out = int(m['branch_cond'].get('output_dim', 64))

    return RDeepONetV2(K=K,
                       pretrained=branch_pretrained,
                       dropout=0.1,
                       L=6,
                       hidden=trunk_hidden,
                       depth=trunk_layers,
                       cond_hidden=cond_hidden,
                       cond_out=cond_out)


def image_gradients(x: torch.Tensor):
    # x: [B,1,H,W] in [0,1]
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(x, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(x, sobel_y, padding=1)
    return gx, gy


def ssim_global(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> torch.Tensor:
    # Very lightweight global SSIM (not windowed), expects x,y in [0,1], shape [B,1,H,W]
    mu_x = x.mean(dim=[2, 3], keepdim=True)
    mu_y = y.mean(dim=[2, 3], keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=[2, 3], keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = (num / (den + 1e-8)).mean()
    return ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_train.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    outdir = Path(cfg.get('output_dir', 'experiments/rdeeponet_v2_run1'))
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if (cfg['training'].get('accelerator', 'cuda') == 'cuda' and torch.cuda.is_available()) else 'cpu')
    torch.manual_seed(42)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    # losses
    loss_weights = cfg.get('loss_weights', {'mse': 1.0, 'ssim': 0.0, 'grad': 0.0})
    primary_loss, primary_loss_name, primary_loss_params = build_regression_loss(cfg.get('loss', {}))
    primary_weight = float(loss_weights.get('mse', loss_weights.get('primary', 1.0)))
    ssim_weight = float(loss_weights.get('ssim', 0.0))
    grad_weight = float(loss_weights.get('grad', 0.0))

    # optim & scheduler
    ocfg = cfg['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(ocfg.get('lr', 1e-3)),
        weight_decay=float(ocfg.get('weight_decay', 1e-2)),
    )
    epochs = int(cfg['training'].get('epochs', 100))
    scheduler = build_scheduler(optimizer, cfg.get('scheduler', {}), epochs)
    if isinstance(scheduler, WarmupCosineScheduler):
        scheduler.step()  # prime warmup so first epoch starts with scaled lr

    use_amp = bool(cfg['training'].get('use_amp', (cfg['training'].get('accelerator', 'cuda') == 'cuda')))
    amp_enabled = use_amp and (cfg['training'].get('accelerator', 'cuda') == 'cuda')
    # Note: Modern PyTorch AMP (1.10+) handles Huber/SmoothL1 loss stably
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    clip_cfg = resolve_gradient_clip(cfg['training'])

    best_val = float('inf')
    early_patience = int(cfg['training'].get('early_stopping_patience', 0))
    mode = cfg['data'].get('mode', 'coord')

    tl_min = float(cfg['data']['normalization'].get('tl_min', 40.0))
    tl_max = float(cfg['data']['normalization'].get('tl_max', 120.0))
    tl_range = tl_max - tl_min

    hist_train, hist_val = [], []
    hist_mae_db_train, hist_mae_db_val = [], []
    metrics_rows = []

    for epoch in range(epochs):
        model.train()
        total = 0.0
        total_mae = 0.0
        for batch in train_loader:
            ray = batch['ray'].to(device)
            cond = batch['cond'].to(device)
            optimizer.zero_grad(set_to_none=True)

            if mode == 'coord':
                coords = batch['coords'].to(device)
                target = batch['tl'].to(device)
                weights = batch.get('edge_weight')
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    pred = model.forward_coord(ray, cond, coords)
                    loss_vals = loss_tensor(pred, target, primary_loss_name, primary_loss_params)
                    if weights is not None:
                        w = weights.to(device)
                        w = w / (w.mean() + 1e-8)
                        primary_term = (loss_vals * w).mean()
                    else:
                        primary_term = loss_vals.mean()
                    loss = primary_weight * primary_term
                scaler.scale(loss).backward()
            else:
                target = batch['tl'].to(device)
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    pred = model.forward_full(ray, cond).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    tgt = target.unsqueeze(0)  # [1,1,H,W]
                    primary_term = primary_loss(pred, tgt)
                    ssim_term = 0.0
                    grad_term = 0.0
                    if ssim_weight > 0.0:
                        ssim_val = ssim_global(pred.clamp(0, 1), tgt.clamp(0, 1))
                        ssim_term = (1.0 - ssim_val)
                    if grad_weight > 0.0:
                        gx_p, gy_p = image_gradients(pred)
                        gx_t, gy_t = image_gradients(tgt)
                        grad_term = (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()
                    loss = primary_weight * primary_term + ssim_weight * ssim_term + grad_weight * grad_term
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if clip_cfg.enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_cfg.max_norm, clip_cfg.norm_type)
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            # track train MAE in dB space (approx): inverse min-max then MAE
            if mode == 'coord':
                mae_norm = (pred - target).abs().mean().item()
                total_mae += mae_norm * tl_range

        # validation
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_mae = 0.0
            for batch in val_loader:
                ray = batch['ray'].to(device)
                cond = batch['cond'].to(device)
                if mode == 'coord':
                    coords = batch['coords'].to(device)
                    target = batch['tl'].to(device)
                    pred = model.forward_coord(ray, cond, coords)
                    loss = primary_loss(pred, target)
                    val_mae += (pred - target).abs().mean().item() * tl_range
                else:
                    target = batch['tl'].to(device)
                    pred = model.forward_full(ray, cond).unsqueeze(0).unsqueeze(0)
                    tgt = target.unsqueeze(0)
                    loss = primary_loss(pred, tgt)
                val_total += float(loss.item())

        avg_train = total / max(1, len(train_loader))
        avg_val = val_total / max(1, len(val_loader))
        avg_mae_db_train = (total_mae / max(1, len(train_loader))) if mode == 'coord' else float('nan')
        avg_mae_db_val = (val_mae / max(1, len(val_loader))) if mode == 'coord' else float('nan')

        hist_train.append(avg_train)
        hist_val.append(avg_val)
        if mode == 'coord':
            hist_mae_db_train.append(avg_mae_db_train)
            hist_mae_db_val.append(avg_mae_db_val)

        current_lr = optimizer.param_groups[0]['lr']
        metrics_rows.append({
            'epoch': epoch + 1,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'mae_db_train': avg_mae_db_train,
            'mae_db_val': avg_mae_db_val,
            'lr': current_lr,
            'loss': primary_loss_name,
        })

        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  mae_db_train={avg_mae_db_train:.2f}  mae_db_val={avg_mae_db_val:.2f}")

        # save best
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            wait = 0
            ckpt = {
                'epoch': best_epoch,
                'state_dict': model.state_dict(),
                'config': cfg,
                'best_val': best_val,
            }
            torch.save(ckpt, outdir / 'best.pt')
        else:
            # early stopping counter
            if early_patience > 0:
                wait = locals().get('wait', 0) + 1
                if wait >= early_patience:
                    print(f"Early stopping at epoch {epoch+1} (best@{locals().get('best_epoch', '?')}, best_val={best_val:.4f})")
                    break
                else:
                    locals()['wait'] = wait

        # save curves every epoch
        plt.figure(figsize=(6,4))
        plt.plot(hist_train, label='train_loss')
        plt.plot(hist_val, label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss (primary)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'loss_curve.png', dpi=150)
        plt.close()

        if mode == 'coord' and len(hist_mae_db_val) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(hist_mae_db_train, label='mae_db_train')
            plt.plot(hist_mae_db_val, label='mae_db_val')
            plt.xlabel('epoch')
            plt.ylabel('MAE (dB)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / 'mae_db_curve.png', dpi=150)
            plt.close()

    print('Training completed.')

    metrics_path = outdir / 'metrics.csv'
    if metrics_rows:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'mae_db_train', 'mae_db_val', 'lr', 'loss']
        with metrics_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(row)


if __name__ == '__main__':
    main()
