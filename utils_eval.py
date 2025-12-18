"""
utils_eval.py
- dB denormalization and MAE(dB) utilities
- Full map inference helper (forward_full preferred, fallback to forward_coord)
- Physics-informed evaluation metrics (reciprocity violation, gradient magnitude)

Note: TL normalization follows [40,120] dB -> [0,1].
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np


def denorm_tl(x: torch.Tensor, tl_min: float = 40.0, tl_max: float = 120.0) -> torch.Tensor:
    """Inverse of min-max normalization for TL.
    x: [..] in [0,1]
    return: TL in dB, clipped to [tl_min, tl_max]
    """
    x = torch.as_tensor(x)
    tl = x * (tl_max - tl_min) + tl_min
    return torch.clamp(tl, tl_min, tl_max)


def mae_db_from_norm(pred_norm: torch.Tensor,
                     gt_norm: torch.Tensor,
                     tl_min: float = 40.0,
                     tl_max: float = 120.0) -> torch.Tensor:
    """Compute MAE in dB given normalized predictions and targets in [0,1].
    Shapes of pred_norm and gt_norm must be broadcastable.
    """
    pred_db = denorm_tl(pred_norm, tl_min, tl_max)
    gt_db = denorm_tl(gt_norm, tl_min, tl_max)
    return (pred_db - gt_db).abs().mean()


@torch.no_grad()
def infer_full_map(model: nn.Module,
                   ray: torch.Tensor,
                   cond: torch.Tensor,
                   device: Optional[torch.device] = None,
                   H: int = 256,
                   W: int = 256,
                   consistency: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """Full-map inference helper returning normalized TL in [0,1].
    Tries model.forward_full first; if unavailable, falls back to forward_coord over a full grid.

    Inputs:
      - ray: [1,1,H,W] or [B,1,H,W] (first item used)
      - cond: [1,2] or [B,2] (first item used)

    Returns:
      - tl_norm_map: [H, W] in [0,1]
    """
    model_device = next(model.parameters()).device
    dev = device or model_device

    # ensure batch=1 tensors
    if ray.dim() == 3:  # [1,H,W]
        ray_b = ray.unsqueeze(0)  # [1,1,H,W]
    elif ray.dim() == 4:
        ray_b = ray[:1]
    else:
        raise ValueError(f"ray shape unsupported: {tuple(ray.shape)}")

    if cond.dim() == 1:  # [2]
        cond_b = cond.unsqueeze(0)
    elif cond.dim() == 2:
        cond_b = cond[:1]
    else:
        raise ValueError(f"cond shape unsupported: {tuple(cond.shape)}")

    ray_b = ray_b.to(dev, non_blocking=True)
    cond_b = cond_b.to(dev, non_blocking=True)
    model = model.to(dev)

    # Try forward_full API (fast path)
    def _sweep() -> torch.Tensor:
        if not (hasattr(model, 'forward_coord') and callable(getattr(model, 'forward_coord'))):
            raise AttributeError("Model has neither forward_full nor forward_coord for full-map inference.")
        r = torch.linspace(0.0, 1.0, W, device=dev)
        z = torch.linspace(0.0, 1.0, H, device=dev)
        Rg, Zg = torch.meshgrid(r, z, indexing='xy')
        coords = torch.stack([Rg.reshape(-1), Zg.reshape(-1)], dim=-1)
        N = coords.shape[0]
        chunk = 16384
        preds = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            coords_chunk = coords[s:e].unsqueeze(0)
            out = model.forward_coord(ray_b, cond_b, coords_chunk)
            preds.append(out[0].detach())
        tl_vec = torch.cat(preds, dim=0)
        tl_map = tl_vec.view(W, H).t().contiguous()
        return torch.clamp(tl_map, 0.0, 1.0).cpu()

    tl_map_full = None
    if hasattr(model, 'forward_full') and callable(getattr(model, 'forward_full')):
        try:
            out = model.forward_full(ray_b, cond_b)
            if out.dim() == 2:
                tl_map_full = out.detach().cpu().clamp(0.0, 1.0)
            elif out.dim() == 3:
                tl_map_full = out[0].detach().cpu().clamp(0.0, 1.0)
        except Exception:
            tl_map_full = None

    if tl_map_full is not None and consistency is None:
        return tl_map_full

    tl_map_sweep = _sweep()

    if tl_map_full is not None:
        if consistency is not None:
            diff = (tl_map_full - tl_map_sweep).abs().mean().item()
            consistency.setdefault('diffs', []).append(diff)
        return tl_map_full

    return tl_map_sweep


# =============================================================================
# Physics-Informed Evaluation Metrics
# =============================================================================

@torch.no_grad()
def compute_reciprocity_violation(
    model: nn.Module,
    ray: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    n_pairs: int = 100,
    n_ranges: int = 10
) -> Dict[str, float]:
    """
    Evaluate reciprocity violation on a single sample.
    
    Physical reciprocity: TL(r, z_r | z_s) = TL(r, z_s | z_r)
    
    Args:
        model: R-DeepONet model
        ray: [1, 1, H, W] ray density map
        cond: [1, 2] condition vector (f_norm, z_s)
        device: Compute device
        n_pairs: Number of depth pairs to test
        n_ranges: Number of range points per pair
        
    Returns:
        Dict with 'mean_violation', 'max_violation', 'std_violation' in normalized units
    """
    model.eval()
    
    z_s = cond[0, 1].item()  # Original source depth
    f_norm = cond[0, 0].item()  # Normalized frequency
    
    # Sample random receiver depths
    z_r_samples = torch.rand(n_pairs, device=device)
    
    # Sample range points
    r_samples = torch.linspace(0.1, 0.9, n_ranges, device=device)  # Avoid boundaries
    
    violations = []
    
    for z_r in z_r_samples:
        z_r_val = z_r.item()
        
        # Skip if z_r is too close to z_s
        if abs(z_r_val - z_s) < 0.05:
            continue
        
        # Create coordinate grids for this pair
        # Pass 1: source=z_s, query=(r, z_r)
        coords_1 = torch.stack([r_samples, torch.full_like(r_samples, z_r_val)], dim=-1).unsqueeze(0)
        pred_1 = model.forward_coord(ray, cond, coords_1)  # [1, n_ranges]
        
        # Pass 2: source=z_r, query=(r, z_s)
        cond_swapped = torch.tensor([[f_norm, z_r_val]], device=device, dtype=cond.dtype)
        coords_2 = torch.stack([r_samples, torch.full_like(r_samples, z_s)], dim=-1).unsqueeze(0)
        pred_2 = model.forward_coord(ray, cond_swapped, coords_2)  # [1, n_ranges]
        
        # Violation = |pred_1 - pred_2|
        violation = (pred_1 - pred_2).abs()
        violations.append(violation)
    
    if not violations:
        return {'mean_violation': 0.0, 'max_violation': 0.0, 'std_violation': 0.0}
    
    all_violations = torch.cat(violations, dim=1)
    
    return {
        'mean_violation': float(all_violations.mean().item()),
        'max_violation': float(all_violations.max().item()),
        'std_violation': float(all_violations.std().item())
    }


@torch.no_grad()
def compute_gradient_metrics(
    model: nn.Module,
    ray: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    H: int = 256,
    W: int = 256,
    delta: float = 1.0 / 256.0
) -> Dict[str, Any]:
    """
    Compute gradient-based metrics for artifact detection.
    
    Args:
        model: R-DeepONet model
        ray: [1, 1, H, W] ray density map
        cond: [1, 2] condition vector
        device: Compute device
        H, W: Grid size
        delta: Finite difference step
        
    Returns:
        Dict with gradient statistics and optional gradient map
    """
    model.eval()
    
    # Create full grid coordinates
    r = torch.linspace(0, 1, W, device=device)
    z = torch.linspace(0, 1, H, device=device)
    R, Z = torch.meshgrid(r, z, indexing='xy')
    coords = torch.stack([R.flatten(), Z.flatten()], dim=-1).unsqueeze(0)  # [1, H*W, 2]
    
    # Shifted coordinates
    coords_r_plus = coords.clone()
    coords_r_plus[:, :, 0] = (coords_r_plus[:, :, 0] + delta).clamp(0, 1)
    
    coords_z_plus = coords.clone()
    coords_z_plus[:, :, 1] = (coords_z_plus[:, :, 1] + delta).clamp(0, 1)
    
    # Forward passes
    pred_base = model.forward_coord(ray, cond, coords)
    pred_r = model.forward_coord(ray, cond, coords_r_plus)
    pred_z = model.forward_coord(ray, cond, coords_z_plus)
    
    # Compute gradients
    grad_r = (pred_r - pred_base) / delta
    grad_z = (pred_z - pred_base) / delta
    
    # Gradient magnitude
    grad_mag = torch.sqrt(grad_r.pow(2) + grad_z.pow(2))
    grad_mag_2d = grad_mag.reshape(H, W)
    
    # Statistics
    grad_mean = float(grad_mag.mean().item())
    grad_max = float(grad_mag.max().item())
    grad_std = float(grad_mag.std().item())
    
    # High-frequency artifact metric: ratio of high gradient regions
    threshold = grad_mean + 2 * grad_std
    high_grad_ratio = float((grad_mag > threshold).float().mean().item())
    
    return {
        'grad_mean': grad_mean,
        'grad_max': grad_max,
        'grad_std': grad_std,
        'high_grad_ratio': high_grad_ratio,
        'grad_map': grad_mag_2d.cpu().numpy()
    }


@torch.no_grad()
def evaluate_physics_metrics(
    model: nn.Module,
    ray: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    tl_min: float = 40.0,
    tl_max: float = 120.0
) -> Dict[str, Any]:
    """
    Comprehensive physics-informed evaluation for a single sample.
    
    Args:
        model: R-DeepONet model
        ray: [1, 1, H, W] ray density map
        cond: [1, 2] condition vector
        device: Compute device
        tl_min, tl_max: TL normalization range
        
    Returns:
        Dict with all physics metrics
    """
    model.eval()
    ray = ray.to(device)
    cond = cond.to(device)
    
    # Reciprocity violation
    recip_metrics = compute_reciprocity_violation(model, ray, cond, device)
    
    # Gradient metrics
    grad_metrics = compute_gradient_metrics(model, ray, cond, device)
    
    # Convert reciprocity violation to dB
    db_range = tl_max - tl_min
    recip_violation_db = recip_metrics['mean_violation'] * db_range
    
    return {
        'reciprocity_violation_norm': recip_metrics['mean_violation'],
        'reciprocity_violation_db': recip_violation_db,
        'reciprocity_max_violation_norm': recip_metrics['max_violation'],
        'grad_mean': grad_metrics['grad_mean'],
        'grad_max': grad_metrics['grad_max'],
        'high_freq_artifact_ratio': grad_metrics['high_grad_ratio'],
        'grad_map': grad_metrics['grad_map']
    }


def compute_rmse_db(
    pred_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    tl_min: float = 40.0,
    tl_max: float = 120.0
) -> float:
    """Compute RMSE in dB."""
    pred_db = denorm_tl(pred_norm, tl_min, tl_max)
    gt_db = denorm_tl(gt_norm, tl_min, tl_max)
    return float(torch.sqrt(((pred_db - gt_db) ** 2).mean()).item())


def _self_test():
    # quick unit tests for denorm and mae
    x = torch.tensor([0.0, 0.5, 1.0])
    y = torch.tensor([0.0, 0.6, 0.9])
    m = mae_db_from_norm(x, y, 40.0, 120.0)
    assert m.item() >= 0.0
    d = denorm_tl(torch.tensor([0.25]))
    assert torch.allclose(d, torch.tensor([60.0]))
    print("utils_eval self-test passed")


if __name__ == "__main__":
    _self_test()
