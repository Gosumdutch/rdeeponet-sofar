"""
run_comparison.py
- Automated comparison runner for value-only vs loss-guided training
- Generates results CSV and comparison figures
- Supports physics-informed evaluation metrics

Usage:
    python run_comparison.py --config config_train.yaml --output_dir experiments/comparison
    python run_comparison.py --config config_stage2_highres.yaml --quick  # Quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train_runner import (
    load_config, fit_one_trial, make_datasets, make_loaders, 
    build_model, build_norm_cfg
)
from utils_eval import (
    infer_full_map, mae_db_from_norm, compute_rmse_db,
    evaluate_physics_metrics, denorm_tl
)
from training_utils import compute_gradient_map_fd


def run_single_experiment(
    cfg: Dict[str, Any],
    experiment_name: str,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run a single training experiment and return results."""
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge overrides
    run_cfg = deepcopy(cfg)
    if overrides:
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                d = run_cfg
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = value
            else:
                run_cfg[key] = value
    
    run_cfg['output_dir'] = str(exp_dir)
    
    # Run training
    start_time = time.time()
    result = fit_one_trial(run_cfg, overrides=overrides)
    train_time = time.time() - start_time
    
    result['train_time_sec'] = train_time
    result['experiment_name'] = experiment_name
    result['output_dir'] = str(exp_dir)
    
    return result


def evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    n_samples: int = 50
) -> Dict[str, Any]:
    """Evaluate a trained checkpoint with physics metrics."""
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Get normalization config
    norm_cfg = build_norm_cfg(cfg)
    tl_min = norm_cfg['tl_db']['min']
    tl_max = norm_cfg['tl_db']['max']
    
    # Create test dataloader
    split_ratio = {'train': 0.8, 'val': 0.2}
    _, val_ds = make_datasets(cfg, split_ratio)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # Metrics accumulators
    maes_db = []
    rmses_db = []
    recip_violations = []
    grad_means = []
    high_freq_ratios = []
    inference_times = []
    
    # Evaluate on samples
    count = 0
    for batch in val_loader:
        if count >= n_samples:
            break
            
        ray = batch['ray'].to(device)
        cond = batch['cond'].to(device)
        tl_gt = batch['tl']
        
        # Inference timing
        start = time.time()
        with torch.no_grad():
            tl_pred = infer_full_map(model, ray, cond, device=device)
        inference_times.append(time.time() - start)
        
        # Basic metrics
        tl_pred = tl_pred.unsqueeze(0).unsqueeze(0)
        mae = mae_db_from_norm(tl_pred, tl_gt, tl_min, tl_max)
        rmse = compute_rmse_db(tl_pred, tl_gt, tl_min, tl_max)
        maes_db.append(float(mae.item()))
        rmses_db.append(rmse)
        
        # Physics metrics
        physics = evaluate_physics_metrics(model, ray, cond, device, tl_min, tl_max)
        recip_violations.append(physics['reciprocity_violation_db'])
        grad_means.append(physics['grad_mean'])
        high_freq_ratios.append(physics['high_freq_artifact_ratio'])
        
        count += 1
    
    return {
        'mae_db': float(np.mean(maes_db)),
        'mae_db_std': float(np.std(maes_db)),
        'rmse_db': float(np.mean(rmses_db)),
        'rmse_db_std': float(np.std(rmses_db)),
        'reciprocity_violation_db': float(np.mean(recip_violations)),
        'reciprocity_violation_std': float(np.std(recip_violations)),
        'grad_mean': float(np.mean(grad_means)),
        'high_freq_artifact_ratio': float(np.mean(high_freq_ratios)),
        'inference_time_ms': float(np.mean(inference_times) * 1000),
        'n_samples': count
    }


def generate_comparison_figures(
    model_value: torch.nn.Module,
    model_guided: torch.nn.Module,
    sample_batch: Dict[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    tl_min: float = 40.0,
    tl_max: float = 120.0
):
    """Generate comparison figures for a single sample."""
    ray = sample_batch['ray'].to(device)
    cond = sample_batch['cond'].to(device)
    tl_gt = sample_batch['tl'].numpy()[0]  # [H, W]
    
    # Get predictions
    with torch.no_grad():
        pred_value = infer_full_map(model_value, ray, cond, device=device).numpy()
        pred_guided = infer_full_map(model_guided, ray, cond, device=device).numpy()
    
    # Denormalize to dB
    tl_gt_db = tl_gt * (tl_max - tl_min) + tl_min
    pred_value_db = pred_value * (tl_max - tl_min) + tl_min
    pred_guided_db = pred_guided * (tl_max - tl_min) + tl_min
    
    # Compute differences
    diff_value = pred_value_db - tl_gt_db
    diff_guided = pred_guided_db - tl_gt_db
    
    # Compute gradient maps
    grad_value = compute_gradient_map_fd(model_value, ray, cond, device).numpy()
    grad_guided = compute_gradient_map_fd(model_guided, ray, cond, device).numpy()
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: GT, Value-Only Pred, Loss-Guided Pred, Ray Map
    vmin, vmax = tl_min, tl_max
    
    im0 = axes[0, 0].imshow(tl_gt_db, aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth TL (dB)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(pred_value_db, aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Value-Only Prediction (dB)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(pred_guided_db, aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Loss-Guided Prediction (dB)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    ray_map = ray[0, 0].cpu().numpy()
    im3 = axes[0, 3].imshow(ray_map, aspect='auto', cmap='hot')
    axes[0, 3].set_title('Ray Density Map')
    plt.colorbar(im3, ax=axes[0, 3])
    
    # Row 2: Difference maps
    diff_max = max(abs(diff_value).max(), abs(diff_guided).max(), 5.0)
    
    im4 = axes[1, 0].imshow(diff_value, aspect='auto', cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[1, 0].set_title(f'Value-Only Error (MAE={abs(diff_value).mean():.2f} dB)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(diff_guided, aspect='auto', cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[1, 1].set_title(f'Loss-Guided Error (MAE={abs(diff_guided).mean():.2f} dB)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Error histograms
    axes[1, 2].hist(diff_value.flatten(), bins=50, alpha=0.7, label='Value-Only', color='blue')
    axes[1, 2].hist(diff_guided.flatten(), bins=50, alpha=0.7, label='Loss-Guided', color='orange')
    axes[1, 2].set_xlabel('Error (dB)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Error Distribution')
    axes[1, 2].legend()
    
    # Absolute error comparison
    im7 = axes[1, 3].imshow(abs(diff_value) - abs(diff_guided), aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 3].set_title('|Error_value| - |Error_guided| (positive = guided better)')
    plt.colorbar(im7, ax=axes[1, 3])
    
    # Row 3: Gradient maps (artifact detection)
    grad_max_val = max(grad_value.max(), grad_guided.max())
    
    im8 = axes[2, 0].imshow(grad_value, aspect='auto', cmap='hot', vmin=0, vmax=grad_max_val)
    axes[2, 0].set_title(f'Value-Only |grad TL| (mean={grad_value.mean():.4f})')
    plt.colorbar(im8, ax=axes[2, 0])
    
    im9 = axes[2, 1].imshow(grad_guided, aspect='auto', cmap='hot', vmin=0, vmax=grad_max_val)
    axes[2, 1].set_title(f'Loss-Guided |grad TL| (mean={grad_guided.mean():.4f})')
    plt.colorbar(im9, ax=axes[2, 1])
    
    # Gradient difference
    im10 = axes[2, 2].imshow(grad_value - grad_guided, aspect='auto', cmap='RdBu_r')
    axes[2, 2].set_title('Gradient Diff (positive = guided smoother)')
    plt.colorbar(im10, ax=axes[2, 2])
    
    # Metadata
    f_norm = cond[0, 0].item()
    z_s = cond[0, 1].item()
    axes[2, 3].text(0.1, 0.7, f'Frequency (norm): {f_norm:.3f}', fontsize=12)
    axes[2, 3].text(0.1, 0.5, f'Source depth (norm): {z_s:.3f}', fontsize=12)
    axes[2, 3].text(0.1, 0.3, f'Value-Only MAE: {abs(diff_value).mean():.2f} dB', fontsize=12)
    axes[2, 3].text(0.1, 0.1, f'Loss-Guided MAE: {abs(diff_guided).mean():.2f} dB', fontsize=12)
    axes[2, 3].axis('off')
    axes[2, 3].set_title('Sample Info')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_figure.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_comparison(
    config_path: str,
    output_dir: str,
    quick: bool = False,
    n_eval_samples: int = 50
):
    """
    Run value-only vs loss-guided comparison.
    
    Args:
        config_path: Path to base config file
        output_dir: Output directory for results
        quick: If True, run quick test with fewer epochs
        n_eval_samples: Number of samples for evaluation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Quick mode overrides
    if quick:
        cfg['training']['epochs'] = 5
        cfg['training']['early_stopping_patience'] = 3
        cfg['data']['pts_per_map'] = 512
        n_eval_samples = 10
    
    print("=" * 60)
    print("Physics-Informed Loss Comparison")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Quick mode: {quick}")
    print("=" * 60)
    
    # Experiment 1: Value-Only (all physics λ = 0)
    print("\n[1/2] Running Value-Only experiment...")
    value_only_overrides = {
        'loss_reciprocity_weight': 0.0,
        'loss_smooth_weight': 0.0,
        'loss_tv_weight': 0.0,
    }
    result_value = run_single_experiment(
        cfg, 'value_only', output_path, value_only_overrides
    )
    print(f"  Best MAE: {result_value['best_mae_db']:.3f} dB")
    print(f"  Train time: {result_value['train_time_sec']:.1f} sec")
    
    # Experiment 2: Loss-Guided (physics λ from config)
    print("\n[2/2] Running Loss-Guided experiment...")
    result_guided = run_single_experiment(
        cfg, 'loss_guided', output_path, {}
    )
    print(f"  Best MAE: {result_guided['best_mae_db']:.3f} dB")
    print(f"  Train time: {result_guided['train_time_sec']:.1f} sec")
    
    # Evaluate both checkpoints
    print("\nEvaluating checkpoints...")
    
    ckpt_value = Path(result_value['output_dir']) / 'best.pt'
    ckpt_guided = Path(result_guided['output_dir']) / 'best.pt'
    
    eval_value = evaluate_checkpoint(ckpt_value, cfg, device, n_eval_samples)
    eval_guided = evaluate_checkpoint(ckpt_guided, cfg, device, n_eval_samples)
    
    # Combine results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config_path,
        'quick_mode': quick,
        'value_only': {**result_value, **eval_value},
        'loss_guided': {**result_guided, **eval_guided}
    }
    
    # Save results JSON
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save results CSV
    csv_path = output_path / 'results.csv'
    fieldnames = [
        'experiment', 'best_mae_db', 'rmse_db', 'reciprocity_violation_db',
        'grad_mean', 'high_freq_artifact_ratio', 'inference_time_ms', 'train_time_sec'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, data in [('value_only', results['value_only']), ('loss_guided', results['loss_guided'])]:
            row = {'experiment': name}
            for field in fieldnames[1:]:
                row[field] = data.get(field, data.get(f'{field}', 'N/A'))
            writer.writerow(row)
    
    # Generate comparison figures
    print("\nGenerating comparison figures...")
    
    # Load both models
    model_value = build_model(cfg).to(device)
    model_value.load_state_dict(torch.load(ckpt_value, map_location=device)['state_dict'])
    model_value.eval()
    
    model_guided = build_model(cfg).to(device)
    model_guided.load_state_dict(torch.load(ckpt_guided, map_location=device)['state_dict'])
    model_guided.eval()
    
    # Get a sample for visualization
    split_ratio = {'train': 0.8, 'val': 0.2}
    _, val_ds = make_datasets(cfg, split_ratio)
    sample = val_ds[0]
    sample_batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
    
    norm_cfg = build_norm_cfg(cfg)
    generate_comparison_figures(
        model_value, model_guided, sample_batch, output_path, device,
        norm_cfg['tl_db']['min'], norm_cfg['tl_db']['max']
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value-Only':<15} {'Loss-Guided':<15} {'Delta':<10}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('MAE (dB)', 'mae_db'),
        ('RMSE (dB)', 'rmse_db'),
        ('Reciprocity Violation (dB)', 'reciprocity_violation_db'),
        ('Gradient Mean', 'grad_mean'),
        ('High-Freq Artifact Ratio', 'high_freq_artifact_ratio'),
        ('Inference Time (ms)', 'inference_time_ms'),
    ]
    
    for label, key in metrics_to_compare:
        v1 = results['value_only'].get(key, 0)
        v2 = results['loss_guided'].get(key, 0)
        delta = v2 - v1
        sign = '+' if delta > 0 else ''
        print(f"{label:<30} {v1:<15.4f} {v2:<15.4f} {sign}{delta:<10.4f}")
    
    print("-" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Value-only vs Loss-guided comparison')
    parser.add_argument('--config', type=str, default='config_train.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='experiments/comparison',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode with fewer epochs')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of evaluation samples')
    
    args = parser.parse_args()
    
    run_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        quick=args.quick,
        n_eval_samples=args.n_samples
    )


if __name__ == '__main__':
    main()

