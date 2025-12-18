import os
import random
import h5py
import math
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _minmax(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return (x - xmin) / max(1e-12, (xmax - xmin))


def _log10_norm(f: float, f_min: float, f_max: float) -> float:
    f = np.clip(f, f_min, f_max)
    return (np.log10(f) - np.log10(f_min)) / (np.log10(f_max) - np.log10(f_min))


class RDeepONetH5(Dataset):
    """R-DeepONet dataset from H5 files.
    Supports two modes:
      - full: return full 256x256 TL field
      - coord: sample coordinates and return (ray, coords, targets)
    """

    def __init__(self,
                 root: str,
                 split: str,
                 split_ratio: Dict[str, float],
                 mode: str = "coord",
                 pts_per_map: int = 4096,
                 norm_cfg: Dict[str, Any] = None,
                 sampler_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.pts_per_map = pts_per_map
        self.norm = norm_cfg or {}
        self.sampler_cfg = sampler_cfg or {}
        self.sampler_strategy = str(self.sampler_cfg.get('strategy', 'uniform')).lower()
        self.edge_ratio = float(self.sampler_cfg.get('edge_ratio', 0.6))
        self.grad_threshold = float(self.sampler_cfg.get('grad_threshold', 0.05))
        self.edge_weight_scale = float(self.sampler_cfg.get('weight_scale', 1.0))

        # index files
        files = sorted([str(p) for p in self.root.glob('*.h5')])
        if len(files) == 0:
            raise FileNotFoundError(f"No H5 files in {root}. Expected generated data at 'R-DeepONet_Data/data/h5'.")

        # deterministic split
        rng = random.Random(42)
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * split_ratio.get('train', 0.8))
        n_val = int(n * split_ratio.get('val', 0.1))
        if split == 'train':
            self.files = files[:n_train]
        elif split == 'val':
            self.files = files[n_train:n_train + n_val]
        else:
            self.files = files[n_train + n_val:]

    def __len__(self):
        return len(self.files)

    def _load_h5(self, path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        with h5py.File(path, 'r') as f:
            X = f['X'][()]        # ray_map (256x256)
            Y = f['Y'][()]        # TL field (256x256)
            meta = dict(f['metadata'].attrs)
        return X.astype(np.float32), Y.astype(np.float32), meta

    def _norm_inputs(self, ray: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        # ray 0..1 already
        ray = np.clip(ray, 0.0, 1.0)
        # freq
        if 'freq' in self.norm.get('aliases', {}):
            pass
        f = float(meta.get('frequency', meta.get('frequency_hz', 0.0)))
        f_norm = _log10_norm(f, self.norm['freq']['f_min'], self.norm['freq']['f_max'])
        # zs
        zs = float(meta.get('source_depth', meta.get('source_depth_m', 0.0))) / float(self.norm['zs']['denom'])
        cond = np.array([f_norm, zs], dtype=np.float32)
        return ray, cond

    def _norm_targets(self, tl: np.ndarray) -> np.ndarray:
        # 40..120 dB → 0..1
        return _minmax(np.clip(tl, self.norm['tl_db']['min'], self.norm['tl_db']['max']),
                       self.norm['tl_db']['min'], self.norm['tl_db']['max']).astype(np.float32)

    def _grad_norm_map(self, tl_norm: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(tl_norm)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        max_val = float(grad_mag.max())
        if max_val < 1e-8:
            return np.zeros_like(grad_mag, dtype=np.float32)
        return (grad_mag / max_val).astype(np.float32)

    def _sample_indices(self, grad_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = grad_norm.shape
        if self.sampler_strategy != 'edge_focus':
            ys = np.random.randint(0, H, size=self.pts_per_map, dtype=np.int64)
            xs = np.random.randint(0, W, size=self.pts_per_map, dtype=np.int64)
            return ys, xs

        edge_candidates = np.argwhere(grad_norm >= self.grad_threshold)
        n_edge_target = int(self.pts_per_map * self.edge_ratio)
        n_edge = min(len(edge_candidates), n_edge_target)
        ys_edge = np.array([], dtype=np.int64)
        xs_edge = np.array([], dtype=np.int64)
        if n_edge > 0 and len(edge_candidates) > 0:
            replace = len(edge_candidates) < n_edge
            chosen = edge_candidates[np.random.choice(len(edge_candidates), size=n_edge, replace=replace)]
            ys_edge = chosen[:, 0]
            xs_edge = chosen[:, 1]
        n_uniform = self.pts_per_map - len(ys_edge)
        ys_uniform = np.random.randint(0, H, size=n_uniform, dtype=np.int64)
        xs_uniform = np.random.randint(0, W, size=n_uniform, dtype=np.int64)
        ys = np.concatenate([ys_edge, ys_uniform])
        xs = np.concatenate([xs_edge, xs_uniform])
        if len(ys) > 1:
            order = np.random.permutation(len(ys))
            ys = ys[order]
            xs = xs[order]
        return ys, xs

    def __getitem__(self, idx: int):
        X, Y, meta = self._load_h5(self.files[idx])
        ray, cond = self._norm_inputs(X, meta)
        tl = self._norm_targets(Y)
        grad_norm = self._grad_norm_map(tl)

        if self.mode == 'full':
            r = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            z = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            Rg, Zg = np.meshgrid(r, z)
            coords = np.stack([Rg, Zg], axis=-1)
            sample = {
                'ray': torch.from_numpy(ray)[None, ...],
                'cond': torch.from_numpy(cond),
                'coords': torch.from_numpy(coords).permute(2, 0, 1),
                'tl': torch.from_numpy(tl)[None, ...],
                'edge_map': torch.from_numpy(grad_norm)[None, ...],
            }
            return sample

        ys, xs = self._sample_indices(grad_norm)
        H, W = tl.shape
        r = xs.astype(np.float32) / (W - 1)
        z = ys.astype(np.float32) / (H - 1)
        coords = np.stack([r, z], axis=-1)
        targets = tl[ys, xs]
        weights = 1.0 + self.edge_weight_scale * grad_norm[ys, xs]

        sample = {
            'ray': torch.from_numpy(ray)[None, ...],
            'cond': torch.from_numpy(cond),
            'coords': torch.from_numpy(coords),
            'tl': torch.from_numpy(targets),
            'edge_weight': torch.from_numpy(weights.astype(np.float32)),
        }
        return sample
        


