"""
Optuna HPO for R-DeepONetV2
- coord-mode training, full-map MAE(dB) objective
- Sampler: multivariate TPE, seed=42, startup=20
- Pruner: ASHA (rf=3, interval=5, grace=40)
- Storage: SQLite at experiments/optuna/rdeeponet.db
- Study: rdeeponet_v2_full_mae_db
- OOM fallback: reduce batch_size and pts_per_map
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import optuna
import mlflow
from mlflow import log_params as mlflow_log_params
from mlflow import log_metric as mlflow_log_metric
from mlflow import log_artifact as mlflow_log_artifact
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from train_runner import fit_one_trial, load_config


def suggest_space(trial: optuna.Trial, smoke: bool = False) -> Dict[str, Any]:
    # batch/pts kept small in smoke
    batch = trial.suggest_categorical("batch_size", [4, 8, 12, 16] if not smoke else [4])
    pts = trial.suggest_categorical("pts_per_map", [1024, 2048, 4096, 8192] if not smoke else [1024])
    lr = trial.suggest_loguniform("lr", 5e-5, 5e-3) if not smoke else 1e-3
    wd = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2) if not smoke else 1e-2
    opt = trial.suggest_categorical("optimizer", ["Adam", "AdamW"]) if not smoke else "AdamW"
    sch = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "OneCycleLR"]) if not smoke else "CosineAnnealingLR"

    K = trial.suggest_categorical("K", [128, 192, 256, 384, 512] if not smoke else [256])
    trunk_hidden = trial.suggest_categorical("trunk_hidden", [256, 320, 384, 448, 512] if not smoke else [256])
    trunk_depth = trial.suggest_int("trunk_depth", 4, 10) if not smoke else 6
    L = trial.suggest_int("positional_L", 4, 10) if not smoke else 6
    dropout = trial.suggest_float("dropout", 0.0, 0.3) if not smoke else 0.1

    pretrained = trial.suggest_categorical("pretrained", [True, False]) if not smoke else False
    freeze = trial.suggest_categorical("freeze_layers", ["none", "layer1", "layer1-2"]) if not smoke else "none"

    grad_clip = trial.suggest_float("gradient_clip_val", 0.5, 2.0) if not smoke else 1.0
    acc = trial.suggest_categorical("accumulate_steps", [1, 2, 4]) if not smoke else 1
    num_workers = trial.suggest_categorical("num_workers", [0, 2, 4]) if not smoke else 0

    return {
        "batch_size": batch,
        "pts_per_map": pts,
        "lr": lr,
        "weight_decay": wd,
        "optimizer": opt,
        "scheduler": sch,
        "final_projection_dim": K,
        "trunk_hidden": trunk_hidden,
        "trunk_depth": trunk_depth,
        "positional_L": L,
        "dropout": dropout,
        "pretrained": pretrained,
        "freeze_layers": freeze,
        "gradient_clip_val": grad_clip,
        "accumulate_steps": acc,
        "num_workers": num_workers,
    }


def is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return isinstance(err, RuntimeError) and ("out of memory" in msg or "cuda" in msg and "alloc" in msg) or isinstance(err, torch.cuda.OutOfMemoryError)


def run_with_oom_fallback(cfg: Dict[str, Any], base_overrides: Dict[str, Any], trial: optuna.Trial, smoke: bool, outdir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # try combinations decreasing memory
    bs_candidates = [base_overrides.get("batch_size", 8)]
    if not smoke:
        bs_candidates += [16, 12, 8, 4, 2]
    else:
        bs_candidates += [4, 2]
    bs_candidates = sorted(set(int(b) for b in bs_candidates), reverse=True)

    pts_candidates = [base_overrides.get("pts_per_map", 4096)]
    if not smoke:
        pts_candidates += [8192, 4096, 2048, 1024]
    else:
        pts_candidates += [1024]
    pts_candidates = sorted(set(int(p) for p in pts_candidates), reverse=True)

    for bs in bs_candidates:
        for pts in pts_candidates:
            overrides = dict(base_overrides)
            overrides.update({"batch_size": max(2, bs), "pts_per_map": max(256, pts), "outdir": str(outdir)})
            try:
                max_steps = 2 if smoke else None
                result = fit_one_trial(cfg, overrides=overrides, max_train_steps=max_steps, trial=trial,
                                       prune_interval=5, grace_epochs=40, patience=30)
                return result, overrides
            except Exception as e:
                if is_oom(e):
                    torch.cuda.empty_cache()
                    continue
                # propagate prune signal or other errors
                raise
    # if all fail, raise
    raise RuntimeError("OOM fallback exhausted")


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def update_topk(study: optuna.Study, k: int, root: Path):
    # copy best.pt into topk dir with ranking
    ensure_dirs(root)
    trials = [t for t in study.best_trials]
    trials = sorted(trials, key=lambda t: t.value)
    summary = []
    # clear dir
    for p in root.glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass
    for rank, t in enumerate(trials[:k], 1):
        trial_dir = Path(t.user_attrs.get("trial_dir", ""))
        src = trial_dir / "best.pt"
        dst = root / f"rank{rank:02d}_trial{t.number}_mae{t.value:.4f}.pt"
        if src.exists():
            try:
                with open(trial_dir / "params.json", "w") as f:
                    json.dump(t.params, f, indent=2)
            except Exception:
                pass
            try:
                import shutil
                shutil.copy2(src, dst)
            except Exception:
                pass
        summary.append({"rank": rank, "trial": t.number, "value": t.value, "dir": str(trial_dir)})
    with open(root / "topk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_train.yaml")
    ap.add_argument("--storage", type=str, default="sqlite:///experiments/optuna/rdeeponet.db")
    ap.add_argument("--study", type=str, default="rdeeponet_v2_full_mae_db")
    ap.add_argument("--timeout-hours", type=float, default=12.0)
    ap.add_argument("--n-trials", type=int, default=0, help="0 for unlimited until timeout")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # storage/study
    ensure_dirs(Path("experiments/optuna"))
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=20)
    pruner = SuccessiveHalvingPruner(min_resource=40, reduction_factor=3, min_early_stopping_rate=0)
    study = optuna.create_study(direction="minimize", study_name=args.study, storage=args.storage,
                                load_if_exists=True, sampler=sampler, pruner=pruner)

    timeout_seconds = int(args.timeout_hours * 3600)

    def objective(trial: optuna.Trial):
        smoke = args.smoke
        params = suggest_space(trial, smoke)
        # trial outdir
        trial_dir = Path(f"experiments/optuna/{args.study}/trial_{trial.number:05d}")
        ensure_dirs(trial_dir)
        params_for_runner = dict(params)
        params_for_runner["outdir"] = str(trial_dir)
        # run with OOM fallback
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(args.study)
        run_name = f"trial_{trial.number:05d}"
        with mlflow.start_run(run_name=run_name, nested=False):
            # log parameters
            mlflow_log_params(params)
            mlflow_log_metric("trial_number", trial.number)
            try:
                result, used_overrides = run_with_oom_fallback(cfg, params_for_runner, trial, smoke, trial_dir)
            except Exception as e:
                # record dir for debugging
                trial.set_user_attr("trial_dir", str(trial_dir))
                raise
            # save params file
            with open(trial_dir / "params.json", "w") as f:
                json.dump(params, f, indent=2)
            # log artifacts and metrics
            mlflow_log_metric("best_mae_db", float(result["best_mae_db"]))
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("used_overrides", used_overrides)
            # artifacts: params.json and best.pt if exists
            try:
                if (trial_dir / "params.json").exists():
                    mlflow_log_artifact(str(trial_dir / "params.json"))
                if (trial_dir / "best.pt").exists():
                    mlflow_log_artifact(str(trial_dir / "best.pt"))
            except Exception:
                pass
            return float(result["best_mae_db"])

    n_trials = None if args.n_trials == 0 else args.n_trials
    optuna.study.max_trials = None  # placeholder: we rely on timeout
    optuna.logging.set_verbosity(optuna.logging.INFO)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True, callbacks=[
            lambda s, t: update_topk(s, k=5, root=Path(f"experiments/optuna/{args.study}/topk"))
        ])
    finally:
        # final topk update
        update_topk(study, k=5, root=Path(f"experiments/optuna/{args.study}/topk"))
        # write study best
        best = {"number": study.best_trial.number, "value": study.best_value, "params": study.best_params}
        ensure_dirs(Path(f"experiments/optuna/{args.study}"))
        with open(Path(f"experiments/optuna/{args.study}/best_summary.json"), "w") as f:
            json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
