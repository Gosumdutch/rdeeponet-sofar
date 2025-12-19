"""
Error Decomposition Analysis for R-DeepONet
- r zone: near/mid/far range
- z zone: surface/channel/bottom
- f zone: low/high frequency
- special zones: shadow/caustic (high gradient)
- Top 5% worst samples analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

from models import RDeepONetV2
from dataset import RDeepONetH5
from utils_eval import denorm_tl, infer_full_map
from train_runner import load_config, build_norm_cfg


def load_model_from_checkpoint(ckpt_path: str, cfg: Dict[str, Any]) -> nn.Module:
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Extract model params from checkpoint or config
    m = cfg.get('model', {})
    trunk_cfg = m.get('trunk', {})
    
    # Check if overrides are in checkpoint
    overrides = ckpt.get('overrides', {})
    
    model = RDeepONetV2(
        K=int(overrides.get('final_projection_dim', m.get('final_projection_dim', 256))),
        pretrained=False,  # Don't load pretrained weights
        dropout=float(overrides.get('dropout', m.get('dropout', 0.1))),
        L=int(overrides.get('positional_L', m.get('positional_L', 6))),
        hidden=int(overrides.get('trunk_hidden', trunk_cfg.get('hidden_dim', 256))),
        depth=int(overrides.get('trunk_depth', trunk_cfg.get('num_layers', 6))),
        trunk_type=str(overrides.get('trunk_type', trunk_cfg.get('type', 'mlp'))),
        trunk_cond_mode=str(overrides.get('trunk_cond_mode', trunk_cfg.get('cond_mode', 'film'))),
        trunk_fourier_dim=int(overrides.get('trunk_fourier_dim', trunk_cfg.get('fourier_dim', 256))),
        trunk_fourier_sigma=float(overrides.get('trunk_fourier_sigma', trunk_cfg.get('fourier_sigma', 1.0))),
        trunk_w0=float(overrides.get('trunk_w0', trunk_cfg.get('w0', 30.0))),
    )
    
    # Load state dict
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    return model


def compute_zone_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    tl_min: float = 40.0,
    tl_max: float = 120.0
) -> Dict[str, Dict[str, float]]:
    """Compute errors for different spatial zones."""
    # Ensure 2D
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    
    if pred.shape != gt.shape:
        # Resize if needed
        if len(gt.shape) == 2 and len(pred.shape) == 2:
            scale = (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1])
            pred = zoom(pred, scale, order=1)
    
    H, W = pred.shape
    
    # Denormalize to dB
    pred_db = pred * (tl_max - tl_min) + tl_min
    gt_db = gt * (tl_max - tl_min) + tl_min
    error = np.abs(pred_db - gt_db)
    
    results = {}
    
    # r zones (columns): near/mid/far
    r_zones = {
        'near': (0, W // 3),
        'mid': (W // 3, 2 * W // 3),
        'far': (2 * W // 3, W)
    }
    for name, (start, end) in r_zones.items():
        zone_error = error[:, start:end]
        if zone_error.size > 0:
            results[f'r_{name}'] = {
                'mae': float(np.mean(zone_error)),
                'rmse': float(np.sqrt(np.mean(zone_error ** 2))),
                'max': float(np.max(zone_error)),
                'std': float(np.std(zone_error))
            }
    
    # z zones (rows): surface/channel/bottom
    z_zones = {
        'surface': (0, H // 3),
        'channel': (H // 3, 2 * H // 3),
        'bottom': (2 * H // 3, H)
    }
    for name, (start, end) in z_zones.items():
        zone_error = error[start:end, :]
        if zone_error.size > 0:
            results[f'z_{name}'] = {
                'mae': float(np.mean(zone_error)),
                'rmse': float(np.sqrt(np.mean(zone_error ** 2))),
                'max': float(np.max(zone_error)),
                'std': float(np.std(zone_error))
            }
    
    # Gradient-based zones (shadow/caustic)
    gy, gx = np.gradient(gt)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_threshold = np.percentile(grad_mag, 90)  # Top 10% gradient
    
    # High gradient (caustic/interference)
    high_grad_mask = grad_mag >= grad_threshold
    if np.sum(high_grad_mask) > 0:
        results['caustic'] = {
            'mae': float(np.mean(error[high_grad_mask])),
            'rmse': float(np.sqrt(np.mean(error[high_grad_mask] ** 2))),
            'max': float(np.max(error[high_grad_mask])),
            'count': int(np.sum(high_grad_mask))
        }
    
    # Low gradient (shadow zones)
    low_grad_threshold = np.percentile(grad_mag, 10)
    low_grad_mask = grad_mag <= low_grad_threshold
    if np.sum(low_grad_mask) > 0:
        results['shadow'] = {
            'mae': float(np.mean(error[low_grad_mask])),
            'rmse': float(np.sqrt(np.mean(error[low_grad_mask] ** 2))),
            'max': float(np.max(error[low_grad_mask])),
            'count': int(np.sum(low_grad_mask))
        }
    
    # Overall
    if error.size > 0:
        results['overall'] = {
            'mae': float(np.mean(error)),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'max': float(np.max(error)),
            'std': float(np.std(error))
        }
    else:
        results['overall'] = {'mae': 0.0, 'rmse': 0.0, 'max': 0.0, 'std': 0.0}
    
    return results


def analyze_dataset(
    model: nn.Module,
    dataset: RDeepONetH5,
    device: torch.device,
    tl_min: float = 40.0,
    tl_max: float = 120.0,
    max_samples: int = 500
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Analyze all samples and compute aggregated statistics."""
    model.eval()
    model = model.to(device)
    
    all_results = []
    n_samples = min(len(dataset), max_samples)
    
    # Aggregate zones
    zone_keys = ['r_near', 'r_mid', 'r_far', 'z_surface', 'z_channel', 'z_bottom', 
                 'caustic', 'shadow', 'overall']
    aggregate = {k: {'mae': [], 'rmse': [], 'max': []} for k in zone_keys}
    
    # Frequency bins
    freq_errors = {'low': [], 'high': []}
    
    for idx in tqdm(range(n_samples), desc='Analyzing samples'):
        sample = dataset[idx]
        ray = sample['ray'].unsqueeze(0).to(device)
        cond = sample['cond'].unsqueeze(0).to(device)
        gt = sample['tl'].squeeze().numpy()  # [1,H,W] -> [H,W]
        
        # Get metadata for frequency
        with h5py.File(dataset.files[idx], 'r') as f:
            meta = dict(f['metadata'].attrs)
        freq = float(meta.get('frequency', meta.get('frequency_hz', 100)))
        
        # Infer full map
        with torch.no_grad():
            pred = infer_full_map(model, ray, cond, device=device).numpy()
        
        # Compute zone errors
        zone_errors = compute_zone_errors(pred, gt, tl_min, tl_max)
        
        # Store sample result
        sample_result = {
            'idx': idx,
            'file': str(dataset.files[idx]),
            'frequency': freq,
            'overall_mae': zone_errors['overall']['mae'],
            'zones': zone_errors
        }
        all_results.append(sample_result)
        
        # Aggregate
        for k in zone_keys:
            if k in zone_errors:
                aggregate[k]['mae'].append(zone_errors[k]['mae'])
                aggregate[k]['rmse'].append(zone_errors[k].get('rmse', zone_errors[k]['mae']))
                aggregate[k]['max'].append(zone_errors[k].get('max', zone_errors[k]['mae']))
        
        # Frequency binning (50Hz threshold)
        if freq <= 100:
            freq_errors['low'].append(zone_errors['overall']['mae'])
        else:
            freq_errors['high'].append(zone_errors['overall']['mae'])
    
    # Compute aggregate statistics
    summary = {}
    for k in zone_keys:
        if aggregate[k]['mae']:
            summary[k] = {
                'mae_mean': float(np.mean(aggregate[k]['mae'])),
                'mae_std': float(np.std(aggregate[k]['mae'])),
                'rmse_mean': float(np.mean(aggregate[k]['rmse'])),
                'max_mean': float(np.mean(aggregate[k]['max']))
            }
    
    # Frequency summary
    summary['freq_low'] = {
        'mae_mean': float(np.mean(freq_errors['low'])) if freq_errors['low'] else 0,
        'mae_std': float(np.std(freq_errors['low'])) if freq_errors['low'] else 0,
        'count': len(freq_errors['low'])
    }
    summary['freq_high'] = {
        'mae_mean': float(np.mean(freq_errors['high'])) if freq_errors['high'] else 0,
        'mae_std': float(np.std(freq_errors['high'])) if freq_errors['high'] else 0,
        'count': len(freq_errors['high'])
    }
    
    return all_results, summary


def get_worst_samples(all_results: List[Dict], top_pct: float = 0.05) -> List[Dict]:
    """Get top worst samples by MAE."""
    sorted_results = sorted(all_results, key=lambda x: x['overall_mae'], reverse=True)
    n_worst = max(1, int(len(sorted_results) * top_pct))
    return sorted_results[:n_worst]


def plot_zone_summary(summary: Dict[str, Any], outdir: Path):
    """Plot zone-wise error summary."""
    outdir.mkdir(parents=True, exist_ok=True)
    
    # R zones
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # r-zone plot
    r_zones = ['r_near', 'r_mid', 'r_far']
    r_labels = ['Near (0-33%)', 'Mid (33-66%)', 'Far (66-100%)']
    r_mae = [summary[z]['mae_mean'] for z in r_zones if z in summary]
    r_std = [summary[z]['mae_std'] for z in r_zones if z in summary]
    
    ax = axes[0]
    bars = ax.bar(r_labels[:len(r_mae)], r_mae, yerr=r_std, capsize=5, color=['#2ecc71', '#f1c40f', '#e74c3c'])
    ax.set_ylabel('MAE (dB)')
    ax.set_title('Error by Range Zone')
    ax.axhline(y=summary['overall']['mae_mean'], color='gray', linestyle='--', label='Overall')
    ax.legend()
    
    # z-zone plot
    z_zones = ['z_surface', 'z_channel', 'z_bottom']
    z_labels = ['Surface', 'Channel', 'Bottom']
    z_mae = [summary[z]['mae_mean'] for z in z_zones if z in summary]
    z_std = [summary[z]['mae_std'] for z in z_zones if z in summary]
    
    ax = axes[1]
    bars = ax.bar(z_labels[:len(z_mae)], z_mae, yerr=z_std, capsize=5, color=['#3498db', '#9b59b6', '#1abc9c'])
    ax.set_ylabel('MAE (dB)')
    ax.set_title('Error by Depth Zone')
    ax.axhline(y=summary['overall']['mae_mean'], color='gray', linestyle='--', label='Overall')
    ax.legend()
    
    # Special zones
    special_zones = ['shadow', 'caustic']
    special_labels = ['Shadow Zone', 'Caustic/Interference']
    special_mae = [summary[z]['mae_mean'] for z in special_zones if z in summary]
    special_std = [summary[z]['mae_std'] for z in special_zones if z in summary]
    
    ax = axes[2]
    if special_mae:
        bars = ax.bar(special_labels[:len(special_mae)], special_mae, yerr=special_std, capsize=5, color=['#34495e', '#e67e22'])
        ax.set_ylabel('MAE (dB)')
        ax.set_title('Error by Special Zone')
        ax.axhline(y=summary['overall']['mae_mean'], color='gray', linestyle='--', label='Overall')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(outdir / 'zone_error_summary.png', dpi=150)
    plt.close(fig)
    
    # Frequency plot
    fig, ax = plt.subplots(figsize=(8, 5))
    freq_labels = ['Low Freq (<100Hz)', 'High Freq (>100Hz)']
    freq_mae = [summary['freq_low']['mae_mean'], summary['freq_high']['mae_mean']]
    freq_std = [summary['freq_low']['mae_std'], summary['freq_high']['mae_std']]
    freq_count = [summary['freq_low']['count'], summary['freq_high']['count']]
    
    bars = ax.bar(freq_labels, freq_mae, yerr=freq_std, capsize=5, color=['#2980b9', '#c0392b'])
    ax.set_ylabel('MAE (dB)')
    ax.set_title('Error by Frequency Band')
    for i, (bar, cnt) in enumerate(zip(bars, freq_count)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + freq_std[i] + 0.1, 
                f'n={cnt}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(outdir / 'frequency_error_summary.png', dpi=150)
    plt.close(fig)


def plot_worst_samples(
    model: nn.Module,
    dataset: RDeepONetH5,
    worst_samples: List[Dict],
    device: torch.device,
    outdir: Path,
    tl_min: float = 40.0,
    tl_max: float = 120.0,
    n_plots: int = 5
):
    """Plot worst samples with GT, Pred, Error maps."""
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    for i, sample_info in enumerate(worst_samples[:n_plots]):
        idx = sample_info['idx']
        sample = dataset[idx]
        ray = sample['ray'].unsqueeze(0).to(device)
        cond = sample['cond'].unsqueeze(0).to(device)
        gt = sample['tl'].squeeze().numpy()  # [1,H,W] -> [H,W]
        
        with torch.no_grad():
            pred = infer_full_map(model, ray, cond, device=device).squeeze().numpy()
        
        # Denormalize
        gt_db = np.squeeze(gt) * (tl_max - tl_min) + tl_min
        pred_db = np.squeeze(pred) * (tl_max - tl_min) + tl_min
        error = np.abs(pred_db - gt_db)
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        im0 = axes[0].imshow(gt_db, cmap='viridis', origin='lower', vmin=40, vmax=120)
        axes[0].set_title(f'GT (freq={sample_info["frequency"]:.0f}Hz)')
        plt.colorbar(im0, ax=axes[0], label='TL (dB)')
        
        im1 = axes[1].imshow(pred_db, cmap='viridis', origin='lower', vmin=40, vmax=120)
        axes[1].set_title('Prediction')
        plt.colorbar(im1, ax=axes[1], label='TL (dB)')
        
        im2 = axes[2].imshow(error, cmap='hot', origin='lower', vmin=0, vmax=15)
        axes[2].set_title(f'Abs Error (MAE={sample_info["overall_mae"]:.2f} dB)')
        plt.colorbar(im2, ax=axes[2], label='Error (dB)')
        
        # Gradient map of GT
        gy, gx = np.gradient(np.squeeze(gt))
        grad_mag = np.sqrt(gx**2 + gy**2)
        im3 = axes[3].imshow(grad_mag, cmap='plasma', origin='lower')
        axes[3].set_title('GT Gradient (structure)')
        plt.colorbar(im3, ax=axes[3], label='|grad|')
        
        for ax in axes:
            ax.set_xlabel('Range (r)')
            ax.set_ylabel('Depth (z)')
        
        fig.suptitle(f'Worst Sample #{i+1}: {Path(sample_info["file"]).name}', fontsize=12)
        plt.tight_layout()
        fig.savefig(outdir / f'worst_sample_{i+1:02d}.png', dpi=150)
        plt.close(fig)


def generate_report(summary: Dict, worst_samples: List[Dict], outdir: Path):
    """Generate text report."""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ERROR DECOMPOSITION ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Overall
    report_lines.append("## OVERALL PERFORMANCE")
    report_lines.append(f"  MAE: {summary['overall']['mae_mean']:.3f} +/- {summary['overall']['mae_std']:.3f} dB")
    report_lines.append(f"  RMSE: {summary['overall']['rmse_mean']:.3f} dB")
    report_lines.append("")
    
    # Range zones
    report_lines.append("## RANGE ZONES (r)")
    for zone in ['r_near', 'r_mid', 'r_far']:
        if zone in summary:
            label = zone.replace('r_', '').upper()
            report_lines.append(f"  {label}: MAE = {summary[zone]['mae_mean']:.3f} +/- {summary[zone]['mae_std']:.3f} dB")
    report_lines.append("")
    
    # Depth zones
    report_lines.append("## DEPTH ZONES (z)")
    for zone in ['z_surface', 'z_channel', 'z_bottom']:
        if zone in summary:
            label = zone.replace('z_', '').upper()
            report_lines.append(f"  {label}: MAE = {summary[zone]['mae_mean']:.3f} +/- {summary[zone]['mae_std']:.3f} dB")
    report_lines.append("")
    
    # Special zones
    report_lines.append("## SPECIAL ZONES")
    if 'shadow' in summary:
        report_lines.append(f"  SHADOW (low grad): MAE = {summary['shadow']['mae_mean']:.3f} +/- {summary['shadow']['mae_std']:.3f} dB")
    if 'caustic' in summary:
        report_lines.append(f"  CAUSTIC (high grad): MAE = {summary['caustic']['mae_mean']:.3f} +/- {summary['caustic']['mae_std']:.3f} dB")
    report_lines.append("")
    
    # Frequency
    report_lines.append("## FREQUENCY BANDS")
    report_lines.append(f"  LOW (<100Hz): MAE = {summary['freq_low']['mae_mean']:.3f} +/- {summary['freq_low']['mae_std']:.3f} dB (n={summary['freq_low']['count']})")
    report_lines.append(f"  HIGH (>100Hz): MAE = {summary['freq_high']['mae_mean']:.3f} +/- {summary['freq_high']['mae_std']:.3f} dB (n={summary['freq_high']['count']})")
    report_lines.append("")
    
    # Worst samples
    report_lines.append("## TOP 5% WORST SAMPLES")
    for i, ws in enumerate(worst_samples[:10]):
        report_lines.append(f"  {i+1}. MAE={ws['overall_mae']:.2f} dB, freq={ws['frequency']:.0f}Hz, file={Path(ws['file']).name}")
    report_lines.append("")
    
    # Analysis insights
    report_lines.append("## KEY INSIGHTS")
    
    # Find worst zone
    zone_maes = {k: v['mae_mean'] for k, v in summary.items() if 'mae_mean' in v and k != 'overall'}
    if zone_maes:
        worst_zone = max(zone_maes, key=zone_maes.get)
        best_zone = min(zone_maes, key=zone_maes.get)
        report_lines.append(f"  - Worst performing zone: {worst_zone} ({zone_maes[worst_zone]:.3f} dB)")
        report_lines.append(f"  - Best performing zone: {best_zone} ({zone_maes[best_zone]:.3f} dB)")
    
    # Check for pattern in worst samples
    if worst_samples:
        worst_freqs = [ws['frequency'] for ws in worst_samples[:10]]
        avg_worst_freq = np.mean(worst_freqs)
        report_lines.append(f"  - Avg frequency of worst samples: {avg_worst_freq:.0f} Hz")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    with open(outdir / 'error_decomposition_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Error Decomposition Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.pt')
    parser.add_argument('--config', type=str, default='config_train.yaml', help='Config file')
    parser.add_argument('--outdir', type=str, default='experiments/error_analysis', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=500, help='Max samples to analyze')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--params-json', type=str, default=None, help='Best params JSON (optional)')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    cfg = load_config(args.config)
    norm_cfg = build_norm_cfg(cfg)
    tl_min = norm_cfg['tl_db']['min']
    tl_max = norm_cfg['tl_db']['max']
    
    # Load params if provided
    overrides = {}
    if args.params_json and Path(args.params_json).exists():
        with open(args.params_json, 'r') as f:
            params = json.load(f)
            if 'best_params' in params:
                overrides = params['best_params']
            else:
                overrides = params
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    
    # Build model with overrides
    if overrides:
        cfg_model = cfg.get('model', {})
        cfg_model['trunk'] = cfg_model.get('trunk', {})
        cfg_model['trunk']['type'] = overrides.get('trunk_type', overrides.get('family', 'ff'))
        cfg_model['trunk']['hidden_dim'] = overrides.get('trunk_hidden', 256)
        cfg_model['trunk']['num_layers'] = overrides.get('trunk_depth', 6)
        cfg_model['trunk']['cond_mode'] = overrides.get('trunk_cond_mode', 'none')
        cfg_model['trunk']['fourier_dim'] = overrides.get('trunk_fourier_dim', 256)
        cfg_model['trunk']['fourier_sigma'] = overrides.get('trunk_fourier_sigma', 1.0)
    
    model = load_model_from_checkpoint(args.checkpoint, cfg)
    model = model.to(device)
    model.eval()
    print("Model loaded")
    
    # Load dataset
    data_cfg = cfg['data']
    split_ratio = {'train': 0.8, 'val': 0.2}
    sampler_cfg = data_cfg.get('sampler', {})
    
    # Use validation set
    dataset = RDeepONetH5(
        root=data_cfg['path'],
        split='val',
        split_ratio=split_ratio,
        mode='full',
        pts_per_map=8192,
        norm_cfg=norm_cfg,
        sampler_cfg=sampler_cfg
    )
    print(f"Loaded {len(dataset)} validation samples")
    
    # Analyze
    print("Analyzing errors...")
    all_results, summary = analyze_dataset(
        model, dataset, device, tl_min, tl_max, args.max_samples
    )
    
    # Get worst samples
    worst_samples = get_worst_samples(all_results, top_pct=0.05)
    print(f"Found {len(worst_samples)} worst samples (top 5%)")
    
    # Save results
    with open(outdir / 'all_results.json', 'w') as f:
        # Convert numpy types
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(outdir / 'worst_samples.json', 'w') as f:
        json.dump(worst_samples, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Generate plots
    print("Generating plots...")
    plot_zone_summary(summary, outdir)
    plot_worst_samples(model, dataset, worst_samples, device, outdir, tl_min, tl_max, n_plots=5)
    
    # Generate report
    print("Generating report...")
    generate_report(summary, worst_samples, outdir)
    
    print(f"\nResults saved to {outdir}")
    print("Files:")
    print("  - error_decomposition_report.txt")
    print("  - summary.json")
    print("  - worst_samples.json")
    print("  - zone_error_summary.png")
    print("  - frequency_error_summary.png")
    print("  - worst_sample_*.png")


if __name__ == '__main__':
    main()
