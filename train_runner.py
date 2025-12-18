"""
train_runner.py
- Training helper for Optuna trials and smoke tests
- Supports coord-mode training with full-map MAE(dB) validation
"""

from __future__ import annotations

from copy import deepcopy
import os
import argparse
import csv
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterable

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import optuna  # optional for pruning integration
except Exception:
    optuna = None
try:
    import mlflow  # optional; used for logging if available
except Exception:
    mlflow = None

from dataset import RDeepONetH5
from models import RDeepONetV2
from utils_eval import infer_full_map, mae_db_from_norm
from training_utils import (
    build_regression_loss, build_scheduler, resolve_gradient_clip, 
    WarmupCosineScheduler, loss_tensor,
    LossComposer, PhysicsLossConfig, build_physics_loss_config,
    compute_reciprocity_loss, compute_smoothness_fd, compute_gradient_map_fd
)

def build_ema_model(model: nn.Module) -> nn.Module:
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema(model: nn.Module, ema_model: Optional[nn.Module], decay: float):
    if ema_model is None:
        return
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
            ema_buf.copy_(buf)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_norm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = cfg['data'].get('normalization', {})
    tl_min = float(n.get('tl_min', 40.0))
    tl_max = float(n.get('tl_max', 120.0))
    return {
        'tl_db': {'min': tl_min, 'max': tl_max},
        'freq': {'f_min': 20.0, 'f_max': 10000.0},
        'zs': {'denom': 5000.0},
    }


def assert_preproc_match(train_ds: RDeepONetH5, val_ds: RDeepONetH5, outdir: Path) -> None:
    issues = []
    if train_ds.norm != val_ds.norm:
        issues.append({'type': 'norm_mismatch', 'train': train_ds.norm, 'val': val_ds.norm})
    keys = ['sampler_strategy', 'edge_ratio', 'grad_threshold', 'edge_weight_scale']
    for k in keys:
        if getattr(train_ds, k, None) != getattr(val_ds, k, None):
            issues.append({'type': 'sampler_mismatch', 'key': k,
                           'train': getattr(train_ds, k, None),
                           'val': getattr(val_ds, k, None)})
    if issues:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / 'failure_preproc.json', 'w') as f:
            json.dump({'issues': issues}, f, indent=2)
        raise RuntimeError(f"Train/val preprocessing mismatch: {issues}")


def save_debug_figures(model: nn.Module,
                       batch: Dict[str, torch.Tensor],
                       outdir: Path,
                       device: torch.device,
                       tl_min: float,
                       tl_max: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    ray = batch['ray'].to(device)
    cond = batch['cond'].to(device)
    tl_gt = batch['tl'].to(device)
    tl_pred = infer_full_map(model, ray, cond, device=device)
    if tl_pred.dim() == 3:
        tl_pred = tl_pred[0]
    tl_pred = tl_pred.detach().cpu()
    tl_gt = tl_gt[0].cpu() if tl_gt.dim() == 4 else tl_gt.cpu()
    diff = (tl_pred - tl_gt).abs()

    grad_map = compute_gradient_map_fd(model, ray.to(device), cond.to(device), device=device).cpu()
    # simple spectral high-freq energy
    pred_np = tl_pred.numpy()
    fft2 = np.fft.fftshift(np.fft.fft2(pred_np))
    mag = np.abs(fft2)
    h, w = mag.shape
    center = (h // 2, w // 2)
    radius = min(center)
    mask = np.ones_like(mag, dtype=bool)
    rr, cc = np.ogrid[:h, :w]
    mask[(rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= (0.25 * radius) ** 2] = False
    hf_energy = mag[mask].sum() / (mag.sum() + 1e-8)

    def _imshow(ax, data, title):
        im = ax.imshow(data, cmap='viridis', origin='lower', vmin=0, vmax=1)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _imshow(axes[0], tl_gt, 'GT')
    _imshow(axes[1], tl_pred, 'Pred')
    _imshow(axes[2], diff, 'Abs Diff')
    _imshow(axes[3], grad_map, 'Grad |∇|')
    fig.suptitle(f'High-freq energy: {hf_energy:.4f}')
    fig.tight_layout()
    fig.savefig(figures_dir / 'debug_maps.png', dpi=200)
    plt.close(fig)


def make_datasets(cfg: Dict[str, Any], split_ratio: Dict[str, float], overrides: Optional[Dict[str, Any]] = None) -> Tuple[RDeepONetH5, RDeepONetH5]:
    data_cfg = cfg['data']
    norm_cfg = build_norm_cfg(cfg)
    root = data_cfg['path']
    pts = int((overrides or {}).get('pts_per_map', data_cfg.get('pts_per_map', 1024)))

    sampler_cfg = dict(cfg['data'].get('sampler', {}))
    if overrides:
        if 'sampler_strategy' in overrides:
            sampler_cfg['strategy'] = overrides['sampler_strategy']
        if 'sampler_edge_ratio' in overrides:
            sampler_cfg['edge_ratio'] = overrides['sampler_edge_ratio']
        if 'sampler_grad_threshold' in overrides:
            sampler_cfg['grad_threshold'] = overrides['sampler_grad_threshold']
        if 'sampler_weight_scale' in overrides:
            sampler_cfg['weight_scale'] = overrides['sampler_weight_scale']
        if 'edge_weight_scale' in overrides:
            sampler_cfg['weight_scale'] = overrides['edge_weight_scale']
        if 'grad_threshold' in overrides:
            sampler_cfg['grad_threshold'] = overrides['grad_threshold']
        if overrides.get('sampler_enabled') is False:
            sampler_cfg['strategy'] = 'uniform'
            sampler_cfg['enabled'] = False
        elif overrides.get('sampler_enabled') is True and not sampler_cfg.get('strategy'):
            sampler_cfg['strategy'] = sampler_cfg.get('strategy', 'edge_focus')
            sampler_cfg['enabled'] = True
    if not bool(sampler_cfg.get('enabled', True)):
        sampler_cfg['strategy'] = 'uniform'
    sampler_cfg.pop('enabled', None)
    train_ds = RDeepONetH5(root, 'train', split_ratio, 'coord', pts, norm_cfg, sampler_cfg)
    val_ds_full = RDeepONetH5(root, 'val', split_ratio, 'full', pts, norm_cfg, sampler_cfg)

    limit = None
    if overrides and "limit_files" in overrides:
        limit = int(overrides["limit_files"])
    else:
        limit = data_cfg.get('limit_files')
        if limit is None:
            limit = cfg.get('limit_files')
    if limit is not None:
        limit = int(limit)
        if limit > 0:
            train_ds.files = train_ds.files[:limit]
            val_cap = min(len(val_ds_full.files), max(1, limit // 10))
            val_ds_full.files = val_ds_full.files[:val_cap]

    return train_ds, val_ds_full


def make_loaders(cfg: Dict[str, Any], train_ds, val_ds_full, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    pin_mem = (cfg['training'].get('accelerator', 'cuda') == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0))
    val_loader_full = DataLoader(val_ds_full, batch_size=1, shuffle=False,
                                 num_workers=0, pin_memory=pin_mem)
    return train_loader, val_loader_full


def build_model(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> RDeepONetV2:
    m = cfg['model']
    trunk_cfg = m.get('trunk', {})
    branch_cfg = m.get('branch_cnn', {})
    cond_cfg = m.get('branch_cond', {})

    ov = overrides or {}
    K = int(ov.get('final_projection_dim', m.get('final_projection_dim', 256)))
    trunk_hidden = int(ov.get('trunk_hidden', trunk_cfg.get('hidden_dim', 256)))
    trunk_layers = int(ov.get('trunk_depth', trunk_cfg.get('num_layers', 6)))
    L = int(ov.get('positional_L', m.get('positional_L', 6)))

    base_branch_dropout = branch_cfg.get('dropout', m.get('dropout', 0.1))
    branch_dropout = float(ov.get('branch_dropout', ov.get('dropout', base_branch_dropout)))
    pretrained = bool(ov.get('pretrained', branch_cfg.get('pretrained', True)))

    branch_variant = str(ov.get('branch_variant', branch_cfg.get('variant', 'resnet18'))).lower()

    branch_params_cfg = dict(branch_cfg.get('params', {}))
    branch_params = {k: v for k, v in branch_params_cfg.items() if k not in ('out_dim', 'dropout')}

    passthrough_keys = [
        'lora_rank', 'lora_alpha', 'lora_layers', 'lora_train_bn',
        'convnext_drop_path_rate', 'convnext_train_stages', 'train_stages',
        'drop_path_rate', 'proj_dropout'
    ]
    for key in passthrough_keys:
        if key in branch_cfg and key not in branch_params:
            branch_params[key] = branch_cfg[key]

    branch_out_dim = int(branch_params_cfg.get('out_dim', branch_cfg.get('out_dim', 512)))
    branch_out_dim = int(ov.get('branch_out_dim', branch_out_dim))

    def _parse_int_sequence(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [int(v) for v in value]
        if isinstance(value, str):
            tokens = [tok.strip() for tok in value.replace(';', ',').split(',')]
            return [int(tok) for tok in tokens if tok]
        return [int(value)]

    film_layers_cfg = ov.get('trunk_film_layers', trunk_cfg.get('film_layers'))
    film_layers = _parse_int_sequence(film_layers_cfg)
    film_gain = float(ov.get('trunk_film_gain', trunk_cfg.get('film_gain', 1.0)))
    trunk_type = str(ov.get('trunk_type', trunk_cfg.get('type', 'mlp'))).lower()
    trunk_cond_mode = str(ov.get('trunk_cond_mode', trunk_cfg.get('cond_mode', 'film'))).lower()
    trunk_fourier_dim = int(ov.get('trunk_fourier_dim', trunk_cfg.get('fourier_dim', 256)))
    trunk_fourier_sigma = float(ov.get('trunk_fourier_sigma', trunk_cfg.get('fourier_sigma', 1.0)))
    trunk_w0 = float(ov.get('trunk_w0', trunk_cfg.get('w0', 30.0)))

    for key, value in ov.items():
        if not key.startswith('branch_') or key == 'branch_variant':
            continue
        inner_key = key[len('branch_'):]
        if inner_key == 'out_dim':
            branch_out_dim = int(value)
        elif inner_key == 'dropout':
            branch_dropout = float(value)
        else:
            branch_params[inner_key] = value

    cond_hidden = int(cond_cfg.get('hidden_dim', 128))
    cond_out = int(cond_cfg.get('output_dim', 64))

    model = RDeepONetV2(
        K=K,
        pretrained=pretrained,
        dropout=branch_dropout,
        L=L,
        hidden=trunk_hidden,
        depth=trunk_layers,
        cond_hidden=cond_hidden,
        cond_out=cond_out,
        branch_variant=branch_variant,
        branch_params=branch_params,
        branch_out_dim=branch_out_dim,
        film_layers=film_layers or None,
        film_gain=film_gain,
        trunk_type=trunk_type,
        trunk_cond_mode=trunk_cond_mode,
        trunk_fourier_dim=trunk_fourier_dim,
        trunk_fourier_sigma=trunk_fourier_sigma,
        trunk_w0=trunk_w0
    )

    freeze_cfg = ov.get('freeze_layers', m.get('freeze_layers', 'none'))
    freeze_key = str(freeze_cfg).lower() if freeze_cfg is not None else 'none'
    if hasattr(model.branch_cnn, 'apply_freeze'):
        model.branch_cnn.apply_freeze(freeze_key)
    elif freeze_key != 'none':
        def _freeze(module: nn.Module):
            for param in module.parameters():
                param.requires_grad = False
        if hasattr(model.branch_cnn, 'stem'):
            if freeze_key in ['layer1', 'layer1-2', 'layer12', 'layer1_2']:
                _freeze(model.branch_cnn.stem[4])
            if freeze_key in ['layer1-2', 'layer12', 'layer1_2']:
                _freeze(model.branch_cnn.stem[5])

    return model


def make_optimizer(cfg: Dict[str, Any], model: nn.Module, overrides: Optional[Dict[str, Any]] = None):
    ocfg = cfg.get('optimizer', {})
    name = (overrides.get('optimizer') if overrides and 'optimizer' in overrides else ocfg.get('name', 'AdamW')).lower()
    lr = float(overrides.get('lr', ocfg.get('lr', 1e-3))) if overrides else float(ocfg.get('lr', 1e-3))
    wd = float(overrides.get('weight_decay', ocfg.get('weight_decay', 1e-2))) if overrides else float(ocfg.get('weight_decay', 1e-2))
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def make_scheduler(cfg: Dict[str, Any], optimizer, epochs: int, steps_per_epoch: int,
                   overrides: Optional[Dict[str, Any]] = None):
    scfg = dict(cfg.get('scheduler', {}))
    base_name = scfg.get('name', 'CosineAnnealingLR')
    if overrides and 'scheduler' in overrides:
        base_name = overrides['scheduler']
    scfg['name'] = base_name

    override_keys = [
        'warmup_epochs', 'eta_min', 'eta_min_ratio', 'T_max', 'pct_start', 'div_factor', 'final_div_factor',
        'anneal_strategy', 'max_lr', 'step_size', 'gamma', 'steps_per_epoch', 'epochs'
    ]
    if overrides:
        for key in override_keys:
            if key in overrides:
                scfg[key] = overrides[key]

    name = base_name.lower()
    if name == 'onecyclelr':
        scfg.setdefault('steps_per_epoch', max(1, steps_per_epoch))
        scfg.setdefault('epochs', epochs)
        scfg.setdefault('pct_start', 0.3)
        scfg.setdefault('div_factor', 25.0)
        scfg.setdefault('final_div_factor', 10000.0)
        scfg.setdefault('anneal_strategy', 'cos')
        scfg.setdefault('max_lr', max(group['lr'] for group in optimizer.param_groups))

    return build_scheduler(optimizer, scfg, epochs)


def evaluate_full_mae_db(model: nn.Module,
                          val_loader_full: DataLoader,
                          tl_min: float, tl_max: float,
                          device: torch.device,
                          check_consistency: bool = False) -> Tuple[float, Optional[float], float]:
    model.eval()
    maes = []
    maes_percent = []
    consistency_stats = {'diffs': []} if check_consistency else None
    db_range = max(1e-6, float(tl_max) - float(tl_min))
    with torch.no_grad():
        for batch in val_loader_full:
            ray = batch['ray'].to(device)
            cond = batch['cond'].to(device)
            tl_gt_norm = batch['tl'].cpu()
            tl_pred_norm = infer_full_map(model, ray, cond, device=device, consistency=consistency_stats)
            tl_pred_norm = tl_pred_norm.unsqueeze(0).unsqueeze(0)
            mae = mae_db_from_norm(tl_pred_norm, tl_gt_norm, tl_min, tl_max)
            mae_db = float(mae.item())
            maes.append(mae_db)
            maes_percent.append((mae_db / db_range) * 100.0)
    avg_mae = float(sum(maes) / max(1, len(maes)))
    avg_pct = float(sum(maes_percent) / max(1, len(maes_percent)))
    avg_diff = None
    if consistency_stats and consistency_stats['diffs']:
        avg_diff = float(sum(consistency_stats['diffs']) / len(consistency_stats['diffs']))
    return avg_mae, avg_diff, avg_pct


def fit_one_trial(cfg: Dict[str, Any],
                  overrides: Optional[Dict[str, Any]] = None,
                  max_train_steps: Optional[int] = None,
                  trial: Any = None,
                  prune_interval: int = 5,
                  grace_epochs: int = 40,
                  patience: int = 30,
                  force_no_physics: bool = False,
                  force_amp: bool = False) -> Dict[str, Any]:
    train_cfg = cfg['training']
    eval_cfg = cfg.get('evaluation', {})

    seed_value = int((overrides or {}).get('seed', cfg.get('seed', 42)))
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.manual_seed_all(seed_value)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    if overrides and 'patience' in overrides:
        patience = int(overrides['patience'])
    elif 'early_stopping_patience' in train_cfg:
        patience = int(train_cfg['early_stopping_patience'])
    if overrides and 'grace_epochs' in overrides:
        grace_epochs = int(overrides['grace_epochs'])
    elif 'grace_epochs' in train_cfg:
        grace_epochs = int(train_cfg['grace_epochs'])

    num_workers = int((overrides or {}).get('num_workers', train_cfg.get('num_workers', 4)))
    if num_workers > 0:
        torch.set_num_threads(max(4, num_workers * 2))
    else:
        torch.set_num_threads(4)

    outdir = Path((overrides or {}).get('outdir', cfg.get('output_dir', 'experiments/rdeeponet_v2_run1')))
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = outdir / 'config_resolved.yaml'
    try:
        with open(resolved_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        if overrides:
            with open(outdir / 'overrides.json', 'w', encoding='utf-8') as f:
                json.dump(overrides, f, indent=2)
    except Exception:
        pass

    accelerator = train_cfg.get('accelerator', 'cuda')
    device = torch.device('cuda' if (accelerator == 'cuda' and torch.cuda.is_available()) else 'cpu')

    split_ratio = {'train': 0.8, 'val': 0.2}
    train_ds, val_ds_full = make_datasets(cfg, split_ratio, overrides)
    assert_preproc_match(train_ds, val_ds_full, outdir)
    batch_size = int((overrides or {}).get('batch_size', train_cfg.get('batch_size', 8)))
    train_loader, val_loader_full = make_loaders(cfg, train_ds, val_ds_full, batch_size, num_workers)

    model = build_model(cfg, overrides).to(device)

    ema_cfg = dict(train_cfg.get('ema', {}))
    if overrides and 'ema' in overrides:
        ema_cfg.update(overrides['ema'])
    ema_enabled = bool(ema_cfg.get('enabled', False))
    ema_decay = float(ema_cfg.get('decay', 0.999))
    ema_model = build_ema_model(model) if ema_enabled else None
    if ema_model is not None:
        ema_model = ema_model.to(device)
        ema_model.eval()

    def init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    if hasattr(model, 'trunk'):
        model.trunk.apply(init_weights)
    if hasattr(model, 'final_projection'):
        model.final_projection.apply(init_weights)

    optimizer = make_optimizer(cfg, model, overrides)
    epochs = int((overrides or {}).get('epochs', train_cfg.get('epochs', 100)))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = make_scheduler(cfg, optimizer, epochs, steps_per_epoch, overrides)
    if isinstance(scheduler, WarmupCosineScheduler):
        scheduler.step()

    # Loss configuration
    loss_weights = cfg.get('loss_weights', {'mse': 1.0, 'ssim': 0.0, 'grad': 0.0, 'tv': 0.0, 'reciprocity': 0.0, 'smooth': 0.0})
    loss_cfg = dict(cfg.get('loss', {}))
    if overrides:
        if 'huber_delta' in overrides:
            loss_cfg['huber_delta'] = overrides['huber_delta']
        if 'loss_type' in overrides:
            loss_cfg['type'] = overrides['loss_type']
    primary_loss, primary_loss_name, primary_loss_params = build_regression_loss(loss_cfg)
    primary_weight = float(loss_weights.get('mse', loss_weights.get('primary', loss_weights.get('value', 1.0))))
    if overrides and 'primary_weight' in overrides:
        primary_weight = float(overrides['primary_weight'])
    reciprocity_weight = float(loss_weights.get('reciprocity', 0.0))
    smooth_weight = float(loss_weights.get('smooth', loss_weights.get('grad', 0.0)))
    tv_weight = float(loss_weights.get('tv', 0.0))

    if overrides and 'loss_reciprocity_weight' in overrides:
        reciprocity_weight = float(overrides['loss_reciprocity_weight'])
    if overrides and 'loss_smooth_weight' in overrides:
        smooth_weight = float(overrides['loss_smooth_weight'])
    if overrides and 'loss_tv_weight' in overrides:
        tv_weight = float(overrides['loss_tv_weight'])

    # Physics-informed loss configuration
    physics_cfg = build_physics_loss_config(cfg)
    if overrides:
        if 'physics_warmup_epochs' in overrides:
            physics_cfg.warmup_epochs = int(overrides['physics_warmup_epochs'])
        if 'physics_auto_scale' in overrides:
            physics_cfg.auto_scale = bool(overrides['physics_auto_scale'])
        if 'reciprocity_n_samples' in overrides:
            physics_cfg.reciprocity_n_samples = int(overrides['reciprocity_n_samples'])
        if 'smooth_n_samples' in overrides:
            physics_cfg.smooth_n_samples = int(overrides['smooth_n_samples'])
        if 'smooth_delta' in overrides:
            physics_cfg.smooth_delta = float(overrides['smooth_delta'])
    
    if force_no_physics:
        reciprocity_weight = 0.0
        smooth_weight = 0.0
        tv_weight = 0.0
        physics_cfg.enabled = False

    # Build LossComposer for physics-informed training
    loss_composer = LossComposer(
        lambda_val=primary_weight,
        lambda_rec=reciprocity_weight,
        lambda_smooth=smooth_weight,
        physics_cfg=physics_cfg
    )
    
    # Keep for backward compatibility logging
    loss_weights['reciprocity'] = reciprocity_weight
    loss_weights['smooth'] = smooth_weight
    loss_weights['tv'] = tv_weight

    use_amp_flag = train_cfg.get('use_amp', accelerator == 'cuda')
    if overrides and 'use_amp' in overrides:
        use_amp_flag = overrides['use_amp']
    amp_enabled = bool(use_amp_flag and device.type == 'cuda')
    if primary_loss_name != 'mse':
        amp_enabled = False
    if force_amp:
        amp_enabled = True
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    clip_cfg = resolve_gradient_clip(train_cfg)
    if overrides:
        if 'gradient_clip_val' in overrides:
            clip_cfg.max_norm = float(overrides['gradient_clip_val'])
            clip_cfg.enabled = True
        if 'gradient_clip_enabled' in overrides:
            clip_cfg.enabled = bool(overrides['gradient_clip_enabled'])
        if 'gradient_clip_norm_type' in overrides:
            clip_cfg.norm_type = float(overrides['gradient_clip_norm_type'])

    acc_steps = int((overrides or {}).get('accumulate_steps', train_cfg.get('accumulate_steps', 1)))

    tl_min = float(cfg['data']['normalization'].get('tl_min', 40.0))
    tl_max = float(cfg['data']['normalization'].get('tl_max', 120.0))
    db_range = max(1e-6, tl_max - tl_min)

    best_mae = float('inf')
    best_mae_percent = float('inf')
    metrics_rows: list[Dict[str, Any]] = []
    step_count = 0
    epochs_since_improve = 0

    # Enhanced logging setup
    training_log_path = outdir / 'training_log.jsonl'
    checkpoint_dir = outdir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Track training statistics
    grad_norm_history = []
    amp_scale_history = []
    lr_history = []

    for epoch in trange(epochs, desc='Epoch', position=0, leave=False):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_mae_norm = 0.0
        running_recip = 0.0
        running_smooth = 0.0
        running_tv = 0.0
        running_recip_valid_ratio = 0.0
        batches = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False, position=1)):
            ray = batch['ray'].to(device)
            cond = batch['cond'].to(device)
            coords = batch['coords'].to(device)
            target = batch['tl'].to(device)
            weights = batch.get('edge_weight')
            
            # Initialize loss terms
            raw_losses = {}
            loss_log = {}
            reciprocity_term = torch.zeros(1, device=device, dtype=ray.dtype)
            smooth_term = torch.zeros(1, device=device, dtype=ray.dtype)
            tv_term = torch.zeros(1, device=device, dtype=ray.dtype)
            recip_stats = {}

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # Primary forward pass
                pred = model.forward_coord(ray, cond, coords)
                loss_vals = loss_tensor(pred, target, primary_loss_name, primary_loss_params)
                if weights is not None:
                    w = weights.to(device)
                    w = w / (w.mean() + 1e-8)
                    primary_term = (loss_vals * w).mean()
                else:
                    primary_term = loss_vals.mean()
                
                raw_losses['value'] = primary_term

                # Reciprocity loss (correct physical implementation)
                # TL(r, z_r | z_s) ≈ TL(r, z_s | z_r) - swap source and receiver depths
                if loss_composer.should_compute_reciprocity(epoch) or reciprocity_weight > 0.0:
                    reciprocity_term, recip_stats = compute_reciprocity_loss(
                        model, ray, cond, coords,
                        n_samples=physics_cfg.reciprocity_n_samples,
                        skip_invalid=physics_cfg.reciprocity_skip_invalid
                    )
                    raw_losses['reciprocity'] = reciprocity_term

                # Smoothness loss (finite difference gradient penalty)
                if loss_composer.should_compute_smooth(epoch) or smooth_weight > 0.0:
                    smooth_term = compute_smoothness_fd(
                        model, ray, cond, coords,
                        delta=physics_cfg.smooth_delta,
                        n_samples=physics_cfg.smooth_n_samples
                    )
                    raw_losses['smooth'] = smooth_term

                # TV loss (backward compatibility - uses full grid, expensive)
                if tv_weight > 0.0:
                    full_pred = model.forward_full(ray, cond)
                    if full_pred.dim() == 2:
                        full_pred = full_pred.unsqueeze(0)
                    elif full_pred.dim() == 3 and full_pred.size(0) != ray.size(0):
                        full_pred = full_pred.view(ray.size(0), ray.shape[-2], ray.shape[-1])
                    diff_x = full_pred[:, :, 1:] - full_pred[:, :, :-1]
                    diff_y = full_pred[:, 1:, :] - full_pred[:, :-1, :]
                    tv_term = diff_x.abs().mean() + diff_y.abs().mean()

                # Compute total loss using LossComposer
                total_loss, loss_log = loss_composer.compute(raw_losses, epoch)
                
                # Add TV if enabled (not in LossComposer for backward compat)
                if tv_weight > 0.0:
                    total_loss = total_loss + tv_weight * tv_term

                loss = total_loss / max(1, acc_steps)

                # NaN detection with failure logging
                if not torch.isfinite(loss).all():
                    failure_info = {
                        'epoch': epoch, 'step': i, 'loss': float(loss.item()),
                        'raw_losses': {k: float(v.item()) if torch.is_tensor(v) else v for k, v in raw_losses.items()},
                        'timestamp': datetime.now().isoformat()
                    }
                    failure_path = outdir / 'failure_log.json'
                    with open(failure_path, 'w') as f:
                        json.dump(failure_info, f, indent=2)
                    print(f"Warning: Non-finite loss detected at epoch {epoch}, step {i}: {loss.item()}")
                    if trial is not None:
                        return {'best_mae_db': float('inf'), 'best_mae_percent': float('inf')}

            if tv_weight > 0.0:
                running_tv += float(tv_term.detach().item())
            if smooth_weight > 0.0 or loss_composer.should_compute_smooth(epoch):
                running_smooth += float(smooth_term.detach().item())

            scaler.scale(loss).backward()
            if ((i + 1) % acc_steps) == 0:
                scaler.unscale_(optimizer)
                grad_norm = 0.0
                if clip_cfg.enabled:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_cfg.max_norm, clip_cfg.norm_type).item()
                    grad_norm_history.append(grad_norm)

                scaler.step(optimizer)
                amp_scale = scaler.get_scale()
                amp_scale_history.append(amp_scale)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Log current learning rate
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)

                if ema_model is not None:
                    update_ema(model, ema_model, ema_decay)

                if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            step_count += 1
            running_loss += float(loss.item() * max(1, acc_steps))
            running_mae_norm += float((pred.detach() - target.detach()).abs().mean().item())
            if reciprocity_weight > 0.0 or loss_composer.should_compute_reciprocity(epoch):
                running_recip += float(reciprocity_term.detach().item())
                if recip_stats:
                    running_recip_valid_ratio += recip_stats.get('valid_ratio', 1.0)
            batches += 1
            if max_train_steps is not None and step_count >= max_train_steps:
                break

        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        eval_model = ema_model if ema_model is not None else model
        if ema_model is not None:
            eval_model.eval()
        val_mae, consistency_diff, val_mae_percent = evaluate_full_mae_db(
            eval_model,
            val_loader_full,
            tl_min,
            tl_max,
            device,
            check_consistency=bool(eval_cfg.get('check_consistency', False))
        )
        if not torch.isfinite(torch.tensor(val_mae)).all() or val_mae > 100.0:
            print(f"Warning: Invalid validation MAE at epoch {epoch}: {val_mae}")
            if trial is not None:
                return {'best_mae_db': float('inf'), 'best_mae_percent': float('inf')}

        avg_train_loss = running_loss / max(1, batches)
        avg_train_mae_db = (running_mae_norm / max(1, batches)) * (tl_max - tl_min)
        avg_train_mae_percent = (avg_train_mae_db / db_range) * 100.0
        avg_recip_loss = running_recip / max(1, batches) if (reciprocity_weight > 0.0 or loss_composer.should_compute_reciprocity(epoch)) else 0.0
        avg_recip_valid_ratio = running_recip_valid_ratio / max(1, batches) if (reciprocity_weight > 0.0 or loss_composer.should_compute_reciprocity(epoch)) else 1.0
        avg_smooth_loss = running_smooth / max(1, batches) if (smooth_weight > 0.0 or loss_composer.should_compute_smooth(epoch)) else 0.0
        avg_tv_loss = running_tv / max(1, batches) if tv_weight > 0.0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Get current lambda values from LossComposer
        current_lambdas = loss_composer.get_lambda(epoch)

        # Calculate training statistics
        avg_grad_norm = np.mean(grad_norm_history[-steps_per_epoch:]) if grad_norm_history else 0.0
        avg_amp_scale = np.mean(amp_scale_history[-steps_per_epoch:]) if amp_scale_history else 0.0

        # Get scheduler info
        eta_min_ratio = getattr(scheduler, 'eta_min_ratio', None) if hasattr(scheduler, 'eta_min_ratio') else None

        if mlflow is not None:
            try:
                mlflow.log_metric('train_loss', float(avg_train_loss), step=epoch)
                mlflow.log_metric('train_mae_db', float(avg_train_mae_db), step=epoch)
                mlflow.log_metric('val_mae_db', float(val_mae), step=epoch)
                if reciprocity_weight > 0.0 or loss_composer.should_compute_reciprocity(epoch):
                    mlflow.log_metric('train_recip_loss', float(avg_recip_loss), step=epoch)
                    mlflow.log_metric('recip_valid_ratio', float(avg_recip_valid_ratio), step=epoch)
                if smooth_weight > 0.0 or loss_composer.should_compute_smooth(epoch):
                    mlflow.log_metric('train_smooth_loss', float(avg_smooth_loss), step=epoch)
                mlflow.log_metric('lambda_reciprocity', float(current_lambdas.get('reciprocity', 0.0)), step=epoch)
                mlflow.log_metric('lambda_smooth', float(current_lambdas.get('smooth', 0.0)), step=epoch)
                if consistency_diff is not None:
                    mlflow.log_metric('forward_vs_sweep_mae', float(consistency_diff), step=epoch)
            except Exception:
                pass

        # Enhanced epoch logging with physics-informed loss details
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_mae_db': avg_train_mae_db,
            'train_mae_percent': avg_train_mae_percent,
            'val_mae_db': val_mae,
            'val_mae_percent': val_mae_percent,
            'lr': current_lr,
            'loss': primary_loss_name,
            'consistency_mae': consistency_diff,
            # Physics-informed loss terms
            'reciprocity_loss': avg_recip_loss if (reciprocity_weight > 0.0 or loss_composer.should_compute_reciprocity(epoch)) else None,
            'reciprocity_valid_ratio': avg_recip_valid_ratio if (reciprocity_weight > 0.0 or loss_composer.should_compute_reciprocity(epoch)) else None,
            'smooth_loss': avg_smooth_loss if (smooth_weight > 0.0 or loss_composer.should_compute_smooth(epoch)) else None,
            'tv_loss': avg_tv_loss if tv_weight > 0.0 else None,
            # Lambda values (from warmup schedule)
            'lambda_value': current_lambdas.get('value', primary_weight),
            'lambda_reciprocity': current_lambdas.get('reciprocity', 0.0),
            'lambda_smooth': current_lambdas.get('smooth', 0.0),
            # Training diagnostics
            'grad_norm': avg_grad_norm,
            'amp_scale': avg_amp_scale,
            'eta_min_ratio': eta_min_ratio,
            'timestamp': datetime.now().isoformat(),
            'seed': seed_value
        }
        metrics_rows.append(epoch_log)

        # Save training log as JSONL
        with open(training_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(epoch_log) + '\n')

        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_mae_percent = val_mae_percent
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_mae_db': best_mae,
                'best_mae_percent': best_mae_percent,
                'config': cfg,
                'scaler': scaler.state_dict(),
            }
            if ema_model is not None:
                checkpoint['ema_state_dict'] = ema_model.state_dict()
            torch.save(checkpoint, outdir / 'best.pt')
            try:
                first_val_batch = next(iter(val_loader_full))
                save_debug_figures(eval_model, first_val_batch, outdir, device, tl_min, tl_max)
            except Exception:
                pass
            epochs_since_improve = 0

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_mae_db': best_mae,
                'best_mae_percent': best_mae_percent,
                'scaler': scaler.state_dict(),
            }
            if ema_model is not None:
                periodic_checkpoint['ema_state_dict'] = ema_model.state_dict()
            torch.save(periodic_checkpoint, checkpoint_dir / f'epoch_{epoch+1}.pt')
        else:
            epochs_since_improve += 1

        if trial is not None:
            try:
                trial.report(val_mae, step=epoch)
                if epoch + 1 >= grace_epochs and (epoch + 1) % max(1, prune_interval) == 0 and trial.should_prune():
                    if optuna is not None:
                        raise optuna.TrialPruned()
                    raise Exception('__PRUNE__')
            except Exception:
                pass

        if epoch + 1 >= grace_epochs and epochs_since_improve >= patience:
            break
        if max_train_steps is not None:
            break

    metrics_path = outdir / 'metrics.csv'
    final_epoch = 0
    if metrics_rows:
        # Get all fieldnames from the first row
        fieldnames = list(metrics_rows[0].keys()) if metrics_rows else []
        with metrics_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(row)
                final_epoch = row['epoch']
    try:
        with open(outdir / 'metrics_best.json', 'w') as f:
            json.dump({'best_mae_db': best_mae, 'best_mae_percent': best_mae_percent,
                       'final_epoch': final_epoch}, f, indent=2)
    except Exception:
        pass

    return {'best_mae_db': best_mae, 'best_mae_percent': best_mae_percent, 'final_epoch': final_epoch}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_train_mini.yaml')
    parser.add_argument('--smoke', action='store_true', help='Run a quick smoke: few steps + one eval')
    parser.add_argument('--seed', type=int, help='Override random seed')
    parser.add_argument('--grad_clip', type=float, help='Override gradient clip max norm')
    parser.add_argument('--suffix', type=str, help='Suffix appended to output_dir')
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    if args.seed is not None:
        cfg['seed'] = int(args.seed)
    if args.suffix:
        base_outdir = Path(cfg.get('output_dir', 'experiments/rdeeponet_v2_run1'))
        cfg['output_dir'] = str(base_outdir.parent / f"{base_outdir.name}_{args.suffix}")
    
    overrides: Dict[str, Any] = {}
    extra_kwargs: Dict[str, Any] = {}
    if args.smoke:
        overrides.update({'batch_size': 4, 'pretrained': False, 'limit_files': 60})
        extra_kwargs['max_train_steps'] = 2
    if args.grad_clip is not None:
        overrides['gradient_clip_val'] = float(args.grad_clip)
        overrides['gradient_clip_enabled'] = True
    
    result = fit_one_trial(cfg, overrides=overrides or None, **extra_kwargs)
    print({'best_mae_db': result['best_mae_db'], 'best_mae_percent': result.get('best_mae_percent')})
    
    
if __name__ == '__main__':
    main()
