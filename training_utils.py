"""Shared training utilities for R-DeepONet experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class GradientClipConfig:
    enabled: bool
    max_norm: float
    norm_type: float


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.eta_min = float(eta_min)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        max_base_lr = max(self.base_lrs) if self.base_lrs else 0.0
        self.eta_min_ratio = self.eta_min / max_base_lr if max_base_lr > 0 else 0.0
        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
        self.cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=self.eta_min)
        self.last_epoch = -1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "eta_min": self.eta_min,
            "eta_min_ratio": self.eta_min_ratio,
            "last_epoch": self.last_epoch,
            "cosine": self.cosine.state_dict(),
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.warmup_epochs = state_dict.get("warmup_epochs", self.warmup_epochs)
        self.total_epochs = state_dict.get("total_epochs", self.total_epochs)
        self.eta_min = state_dict.get("eta_min", self.eta_min)
        self.eta_min_ratio = state_dict.get("eta_min_ratio", self.eta_min_ratio)
        self.last_epoch = state_dict.get("last_epoch", self.last_epoch)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)
        self.cosine.load_state_dict(state_dict.get("cosine", self.cosine.state_dict()))

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group["lr"] = base_lr * factor
            return
        if self.last_epoch == self.warmup_epochs:
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group["lr"] = base_lr
        self.cosine.step()


def resolve_gradient_clip(training_cfg: Dict[str, Any]) -> GradientClipConfig:
    clip_cfg = training_cfg.get("gradient_clip")
    if clip_cfg is None:
        max_norm = training_cfg.get("gradient_clip_val")
        if max_norm is None:
            return GradientClipConfig(False, 0.0, 2.0)
        return GradientClipConfig(True, float(max_norm), 2.0)
    enabled = bool(clip_cfg.get("enabled", True))
    max_norm = float(clip_cfg.get("max_norm", clip_cfg.get("value", 1.0)))
    norm_type = float(clip_cfg.get("norm_type", 2.0))
    return GradientClipConfig(enabled, max_norm, norm_type)


def build_regression_loss(loss_cfg: Dict[str, Any]) -> Tuple[nn.Module, str, Dict[str, Any]]:
    cfg = loss_cfg or {}
    loss_type = str(cfg.get("type", "mse")).lower()
    delta = float(cfg.get("huber_delta", 1.0))
    if loss_type in {"huber", "smooth_l1"}:
        try:
            loss = nn.SmoothL1Loss(beta=delta)
        except TypeError:
            loss = nn.SmoothL1Loss()
        return loss, "smooth_l1", {"delta": delta}
    if loss_type in {"l1", "mae"}:
        loss = nn.L1Loss()
        return loss, "l1", {}
    return nn.MSELoss(), "mse", {}


def build_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Dict[str, Any], total_epochs: int):
    cfg = dict(scheduler_cfg or {})
    name = str(cfg.get("name", "CosineAnnealingLR")).lower()
    eta_min_ratio = cfg.get("eta_min_ratio")
    eta_min_value = cfg.get("eta_min")
    if eta_min_ratio is not None:
        base_candidates = [group.get("lr", 0.0) for group in optimizer.param_groups if group.get("lr") is not None]
        base_lr = max(base_candidates) if base_candidates else 0.0
        if base_lr > 0:
            eta_min_value = float(eta_min_ratio) * base_lr
    if eta_min_value is None:
        eta_min_value = cfg.get("eta_min", 1e-5)
    eta_min = float(eta_min_value)
    cfg["eta_min"] = eta_min
    if name in {"warmupcosine", "warmup_cosine", "warmupcosineannealinglr"}:
        warmup_epochs = int(cfg.get("warmup_epochs", 3))
        return WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, eta_min)
    if name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("T_max", total_epochs)),
            eta_min=eta_min,
        )
    if name == "onecyclelr":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("max_lr", max(group["lr"] for group in optimizer.param_groups))),
            steps_per_epoch=int(cfg.get("steps_per_epoch", 1)),
            epochs=int(cfg.get("epochs", total_epochs)),
            pct_start=float(cfg.get("pct_start", 0.3)),
            div_factor=float(cfg.get("div_factor", 25.0)),
            final_div_factor=float(cfg.get("final_div_factor", 10000.0)),
            anneal_strategy=str(cfg.get("anneal_strategy", "cos")),
        )
    if name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 30)),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    if name == "exponentiallr":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(cfg.get("gamma", 0.95)),
        )
    # default fallback
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.get("T_max", total_epochs)),
        eta_min=float(cfg.get("eta_min", 1e-5)),
    )


def loss_tensor(pred: torch.Tensor, target: torch.Tensor, loss_name: str, params: Dict[str, Any]) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    if loss_name == "mse":
        return (pred - target) ** 2
    if loss_name == "smooth_l1":
        beta = float(params.get("delta", 1.0))
        try:
            return F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
        except TypeError:
            return F.smooth_l1_loss(pred, target, reduction='none')
    if loss_name == "l1":
        return (pred - target).abs()
    return (pred - target) ** 2


# =============================================================================
# Physics-Informed Loss Components
# =============================================================================

@dataclass
class PhysicsLossConfig:
    """Configuration for physics-informed loss terms."""
    enabled: bool = True
    warmup_epochs: int = 15
    warmup_type: str = "linear"  # linear | cosine
    auto_scale: bool = True
    scale_ema_decay: float = 0.99
    # Reciprocity settings
    reciprocity_n_samples: int = 64
    reciprocity_skip_invalid: bool = True
    # Smoothness settings
    smooth_delta: float = 1.0 / 256.0
    smooth_n_samples: int = 128


def build_physics_loss_config(cfg: Dict[str, Any]) -> PhysicsLossConfig:
    """Build PhysicsLossConfig from config dict."""
    physics_cfg = cfg.get("physics_loss", {})
    recip_cfg = physics_cfg.get("reciprocity", {})
    smooth_cfg = physics_cfg.get("smooth", {})
    
    return PhysicsLossConfig(
        enabled=bool(physics_cfg.get("enabled", True)),
        warmup_epochs=int(physics_cfg.get("warmup_epochs", 15)),
        warmup_type=str(physics_cfg.get("warmup_type", "linear")),
        auto_scale=bool(physics_cfg.get("auto_scale", True)),
        scale_ema_decay=float(physics_cfg.get("scale_ema_decay", 0.99)),
        reciprocity_n_samples=int(recip_cfg.get("n_samples", 64)),
        reciprocity_skip_invalid=bool(recip_cfg.get("skip_invalid", True)),
        smooth_delta=float(smooth_cfg.get("delta", 1.0 / 256.0)),
        smooth_n_samples=int(smooth_cfg.get("n_samples", 128)),
    )


class LossComposer:
    """
    Composes total loss from value loss + physics-informed terms.
    
    Features:
    - Independent on/off for each term (λ=0 skips computation)
    - EMA-based loss term scaling for stable λ tuning
    - Warmup schedule: physics terms activate after warmup_epochs
    - Logs raw and weighted values separately
    """
    
    def __init__(self, 
                 lambda_val: float = 1.0,
                 lambda_rec: float = 0.0,
                 lambda_smooth: float = 0.0,
                 physics_cfg: Optional[PhysicsLossConfig] = None):
        self.lambda_val = float(lambda_val)
        self.lambda_rec_target = float(lambda_rec)
        self.lambda_smooth_target = float(lambda_smooth)
        
        self.cfg = physics_cfg or PhysicsLossConfig()
        
        # EMA scale factors (initialized to 1.0, updated during training)
        self.ema_val = 1.0
        self.ema_rec = 1.0
        self.ema_smooth = 1.0
        self._step_count = 0
        
    def get_lambda(self, epoch: int) -> Dict[str, float]:
        """Get current lambda values with warmup applied."""
        lambdas = {"value": self.lambda_val}
        
        if not self.cfg.enabled:
            lambdas["reciprocity"] = 0.0
            lambdas["smooth"] = 0.0
            return lambdas
        
        # Physics terms warmup
        if epoch < self.cfg.warmup_epochs:
            factor = 0.0
        else:
            progress = (epoch - self.cfg.warmup_epochs) / max(1, self.cfg.warmup_epochs)
            progress = min(1.0, progress)
            
            if self.cfg.warmup_type == "cosine":
                factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            else:  # linear
                factor = progress
        
        lambdas["reciprocity"] = self.lambda_rec_target * factor
        lambdas["smooth"] = self.lambda_smooth_target * factor
        
        return lambdas
    
    def update_scale(self, raw_losses: Dict[str, float]) -> None:
        """Update EMA scale factors based on raw loss magnitudes."""
        if not self.cfg.auto_scale:
            return
            
        decay = self.cfg.scale_ema_decay
        self._step_count += 1
        
        # Use bias correction for early steps
        bias_correction = 1.0 - (decay ** self._step_count)
        
        if "value" in raw_losses and raw_losses["value"] > 1e-10:
            self.ema_val = decay * self.ema_val + (1 - decay) * raw_losses["value"]
        if "reciprocity" in raw_losses and raw_losses["reciprocity"] > 1e-10:
            self.ema_rec = decay * self.ema_rec + (1 - decay) * raw_losses["reciprocity"]
        if "smooth" in raw_losses and raw_losses["smooth"] > 1e-10:
            self.ema_smooth = decay * self.ema_smooth + (1 - decay) * raw_losses["smooth"]
    
    def get_scale_factors(self) -> Dict[str, float]:
        """Get scale factors to normalize loss terms to similar magnitudes."""
        if not self.cfg.auto_scale:
            return {"value": 1.0, "reciprocity": 1.0, "smooth": 1.0}
        
        # Normalize relative to value loss
        base = max(1e-10, self.ema_val)
        return {
            "value": 1.0,
            "reciprocity": base / max(1e-10, self.ema_rec),
            "smooth": base / max(1e-10, self.ema_smooth),
        }
    
    def compute(self, 
                raw_losses: Dict[str, torch.Tensor], 
                epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and return logging dict.
        
        Args:
            raw_losses: Dict with 'value', 'reciprocity', 'smooth' tensors
            epoch: Current epoch for warmup scheduling
            
        Returns:
            total_loss: Combined loss tensor
            log_dict: Dict with raw_*, weighted_*, lambda_* values
        """
        device = raw_losses.get("value", torch.tensor(0.0)).device
        dtype = raw_losses.get("value", torch.tensor(0.0)).dtype
        
        lambdas = self.get_lambda(epoch)
        scales = self.get_scale_factors()
        
        # Update EMA with raw values
        raw_floats = {k: float(v.detach().item()) if torch.is_tensor(v) else float(v) 
                      for k, v in raw_losses.items()}
        self.update_scale(raw_floats)
        
        # Compute total loss
        total = torch.zeros(1, device=device, dtype=dtype)
        log_dict: Dict[str, float] = {}
        
        # Value loss (always computed)
        if "value" in raw_losses:
            val_loss = raw_losses["value"]
            weighted_val = lambdas["value"] * val_loss
            total = total + weighted_val
            log_dict["raw_value"] = raw_floats.get("value", 0.0)
            log_dict["weighted_value"] = float(weighted_val.detach().item())
            log_dict["lambda_value"] = lambdas["value"]
        
        # Reciprocity loss (skip if λ=0)
        if lambdas["reciprocity"] > 0 and "reciprocity" in raw_losses:
            rec_loss = raw_losses["reciprocity"]
            scaled_rec = rec_loss * scales["reciprocity"]
            weighted_rec = lambdas["reciprocity"] * scaled_rec
            total = total + weighted_rec
            log_dict["raw_reciprocity"] = raw_floats.get("reciprocity", 0.0)
            log_dict["weighted_reciprocity"] = float(weighted_rec.detach().item())
            log_dict["lambda_reciprocity"] = lambdas["reciprocity"]
            log_dict["scale_reciprocity"] = scales["reciprocity"]
        
        # Smoothness loss (skip if λ=0)
        if lambdas["smooth"] > 0 and "smooth" in raw_losses:
            sm_loss = raw_losses["smooth"]
            scaled_sm = sm_loss * scales["smooth"]
            weighted_sm = lambdas["smooth"] * scaled_sm
            total = total + weighted_sm
            log_dict["raw_smooth"] = raw_floats.get("smooth", 0.0)
            log_dict["weighted_smooth"] = float(weighted_sm.detach().item())
            log_dict["lambda_smooth"] = lambdas["smooth"]
            log_dict["scale_smooth"] = scales["smooth"]
        
        log_dict["total_loss"] = float(total.detach().item())
        
        return total, log_dict
    
    def should_compute_reciprocity(self, epoch: int) -> bool:
        """Check if reciprocity should be computed this epoch."""
        if not self.cfg.enabled:
            return False
        return self.lambda_rec_target > 0 and epoch >= self.cfg.warmup_epochs
    
    def should_compute_smooth(self, epoch: int) -> bool:
        """Check if smoothness should be computed this epoch."""
        if not self.cfg.enabled:
            return False
        return self.lambda_smooth_target > 0 and epoch >= self.cfg.warmup_epochs


def compute_reciprocity_loss(
    model: nn.Module,
    ray: torch.Tensor,
    cond: torch.Tensor,
    coords: torch.Tensor,
    n_samples: int = 64,
    skip_invalid: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute physical reciprocity loss: TL(r, z_r | z_s) ≈ TL(r, z_s | z_r)
    
    In-batch self-swap: swap source depth (z_s in cond) with receiver depth (z_query).
    
    Args:
        model: R-DeepONet model with forward_coord method
        ray: [B, 1, H, W] ray density map
        cond: [B, 2] condition vector where cond[:, 1] = z_s (normalized source depth)
        coords: [B, N, 2] query coordinates where coords[:, :, 1] = z_query
        n_samples: Number of coordinate pairs to sample
        skip_invalid: Whether to skip out-of-domain swaps
        
    Returns:
        loss: Reciprocity loss tensor
        stats: Dict with 'valid_ratio', 'n_pairs' for logging
    """
    B = ray.shape[0]
    N = coords.shape[1]
    device = ray.device
    dtype = ray.dtype
    
    # Sample subset of query points
    n_samples = min(n_samples, N)
    idx = torch.randperm(N, device=device)[:n_samples]
    
    r_sampled = coords[:, idx, 0]  # [B, n_samples] - range
    z_r = coords[:, idx, 1]        # [B, n_samples] - receiver/query depth
    z_s = cond[:, 1:2]             # [B, 1] - source depth
    f_norm = cond[:, 0:1]          # [B, 1] - normalized frequency (keep same)
    
    # Domain check: ensure swapped depths are in valid range [0, 1]
    if skip_invalid:
        valid_mask = (z_r >= 0) & (z_r <= 1) & (z_s >= 0) & (z_s <= 1)
        valid_mask = valid_mask.expand(-1, n_samples)  # [B, n_samples]
    else:
        valid_mask = torch.ones(B, n_samples, device=device, dtype=torch.bool)
    
    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        return torch.zeros(1, device=device, dtype=dtype), {"valid_ratio": 0.0, "n_pairs": 0}
    
    # Pass 1: original configuration (source=z_s, query=(r, z_r))
    coords_1 = torch.stack([r_sampled, z_r], dim=-1)  # [B, n_samples, 2]
    pred_1 = model.forward_coord(ray, cond, coords_1)  # [B, n_samples]
    
    # Pass 2: swapped configuration (source=z_r, query=(r, z_s))
    # For each sample point, use that z_r as the new source depth
    # and query at the original source depth z_s
    z_s_expanded = z_s.expand(-1, n_samples)  # [B, n_samples]
    coords_2 = torch.stack([r_sampled, z_s_expanded], dim=-1)  # [B, n_samples, 2]
    
    # Create swapped condition: same frequency, but source depth = z_r
    cond_swapped = torch.zeros(B, n_samples, 2, device=device, dtype=dtype)
    cond_swapped[:, :, 0] = f_norm.expand(-1, n_samples)  # Same frequency
    cond_swapped[:, :, 1] = z_r  # Swapped: receiver becomes source
    
    # Forward for each swapped configuration
    # Reshape for batched forward
    ray_expanded = ray.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)  # [B, n_samples, 1, H, W]
    ray_flat = ray_expanded.reshape(B * n_samples, 1, ray.shape[-2], ray.shape[-1])
    cond_flat = cond_swapped.reshape(B * n_samples, 2)
    coords_2_flat = coords_2.reshape(B * n_samples, 1, 2)
    
    pred_2_flat = model.forward_coord(ray_flat, cond_flat, coords_2_flat)  # [B*n_samples, 1]
    pred_2 = pred_2_flat.reshape(B, n_samples)
    
    # Compute reciprocity violation
    diff = (pred_1 - pred_2).abs()
    
    # Apply valid mask
    masked_diff = diff * valid_mask.float()
    loss = masked_diff.sum() / (valid_mask.sum() + 1e-8)
    
    stats = {
        "valid_ratio": float(valid_mask.float().mean().item()),
        "n_pairs": int(valid_count),
    }
    
    return loss, stats


def compute_smoothness_fd(
    model: nn.Module,
    ray: torch.Tensor,
    cond: torch.Tensor,
    coords: torch.Tensor,
    delta: float = 1.0 / 256.0,
    n_samples: int = 128
) -> torch.Tensor:
    """
    Compute smoothness loss using finite difference gradient penalty.
    
    ||∇TL||² = (∂TL/∂r)² + (∂TL/∂z)²
    
    Uses forward difference: ∂TL/∂r ≈ (TL(r+δ, z) - TL(r, z)) / δ
    
    Args:
        model: R-DeepONet model with forward_coord method
        ray: [B, 1, H, W] ray density map
        cond: [B, 2] condition vector
        coords: [B, N, 2] query coordinates
        delta: Finite difference step size (default: 1/256 for 256x256 grid)
        n_samples: Number of points to sample for gradient computation
        
    Returns:
        loss: Mean squared gradient magnitude
    """
    B = ray.shape[0]
    N = coords.shape[1]
    device = ray.device
    dtype = ray.dtype
    
    # Sample subset of query points
    n_samples = min(n_samples, N)
    idx = torch.randperm(N, device=device)[:n_samples]
    base_coords = coords[:, idx, :]  # [B, n_samples, 2]
    
    # Create shifted coordinates
    coords_r_plus = base_coords.clone()
    coords_r_plus[:, :, 0] = (coords_r_plus[:, :, 0] + delta).clamp(0, 1)
    
    coords_z_plus = base_coords.clone()
    coords_z_plus[:, :, 1] = (coords_z_plus[:, :, 1] + delta).clamp(0, 1)
    
    # Forward passes (3 total)
    pred_base = model.forward_coord(ray, cond, base_coords)
    pred_r = model.forward_coord(ray, cond, coords_r_plus)
    pred_z = model.forward_coord(ray, cond, coords_z_plus)
    
    # Compute gradient approximation
    # Handle boundary: if coord was clamped, effective delta is smaller
    effective_delta_r = coords_r_plus[:, :, 0] - base_coords[:, :, 0]
    effective_delta_z = coords_z_plus[:, :, 1] - base_coords[:, :, 1]
    
    # Avoid division by zero at boundaries
    safe_delta_r = effective_delta_r.clamp(min=delta * 0.1)
    safe_delta_z = effective_delta_z.clamp(min=delta * 0.1)
    
    grad_r = (pred_r - pred_base) / safe_delta_r
    grad_z = (pred_z - pred_base) / safe_delta_z
    
    # L2 penalty: squared gradient magnitude
    grad_mag_sq = grad_r.pow(2) + grad_z.pow(2)
    
    return grad_mag_sq.mean()


def compute_gradient_map_fd(
    model: nn.Module,
    ray: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    H: int = 256,
    W: int = 256,
    delta: float = 1.0 / 256.0
) -> torch.Tensor:
    """
    Generate |∇TL| map for visualization (Fig.6).
    
    Args:
        model: R-DeepONet model
        ray: [1, 1, H, W] ray density map
        cond: [1, 2] condition vector
        device: Compute device
        H, W: Output grid size
        delta: Finite difference step
        
    Returns:
        grad_mag: [H, W] gradient magnitude map
    """
    model.eval()
    
    with torch.no_grad():
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
        
        # Gradients
        grad_r = (pred_r - pred_base) / delta
        grad_z = (pred_z - pred_base) / delta
        
        # Magnitude
        grad_mag = torch.sqrt(grad_r.pow(2) + grad_z.pow(2))
        grad_mag = grad_mag.reshape(H, W)
    
    return grad_mag.cpu()
