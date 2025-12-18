"""
Trunk HPO pipeline (FF-MLP + SIREN only), physics off, anti-smoothing.

Stages:
- stage0: two quick stability checks (MSE/Huber) with AMP forced ON.
- stage1: global search across family={ff,siren} with TPE+ASHA on subset data.
- stage2: winner family refinement on longer epochs.
- stage3: best params, seed=3 replication.

Artifacts: run_dir per trial/seed with config_resolved.yaml, overrides.json,
metrics.csv, metrics_best.json, best.pt, figures/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from train_runner import fit_one_trial, load_config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_params(trial: optuna.Trial, family: str, stage: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params['family'] = family
    params['lr'] = trial.suggest_loguniform('lr', 2e-4, 5e-3)
    params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 5e-3)
    params['trunk_depth'] = trial.suggest_int('trunk_depth', 4, 10)
    params['trunk_hidden'] = trial.suggest_int('trunk_hidden', 256, 768, step=64)
    params['trunk_cond_mode'] = trial.suggest_categorical('trunk_cond_mode', ['none', 'film', 'concat'])
    params['edge_weight_scale'] = trial.suggest_float('edge_weight_scale', 1.0, 5.0)
    params['grad_threshold'] = trial.suggest_float('grad_threshold', 0.01, 0.06)

    if family == 'ff':
        params['trunk_type'] = 'ff'
        params['trunk_fourier_dim'] = trial.suggest_categorical('trunk_fourier_dim', [128, 256, 512, 768, 1024])
        params['trunk_fourier_sigma'] = trial.suggest_loguniform('trunk_fourier_sigma', 0.5, 32.0)
    elif family == 'siren':
        params['trunk_type'] = 'siren'
        params['trunk_w0'] = trial.suggest_uniform('trunk_w0', 10.0, 60.0)
    else:
        raise ValueError(f"Unsupported family {family}")

    # loss to keep AMP on
    params['loss_type'] = 'mse'
    params['use_amp'] = True
    return params


def get_study(study_name: str, storage: str, direction: str = 'minimize') -> optuna.Study:
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=10)
    pruner = SuccessiveHalvingPruner(min_resource=5, reduction_factor=3, min_early_stopping_rate=0)
    return optuna.create_study(study_name=study_name, storage=storage,
                               load_if_exists=True, sampler=sampler, pruner=pruner,
                               direction=direction)


def stage0(config: Dict[str, Any], root: Path) -> None:
    cases = [
        {'name': 'mse_amp', 'loss_type': 'mse'},
        {'name': 'huber_amp', 'loss_type': 'huber', 'huber_delta': 1.0},
    ]
    for case in cases:
        outdir = root / 'stage0' / case['name']
        overrides = {
            'outdir': str(outdir),
            'loss_type': case['loss_type'],
            'huber_delta': case.get('huber_delta', 1.0),
            'epochs': 2,
            'limit_files': 32,
            'pts_per_map': 2048,
            'batch_size': 8,
            'use_amp': True,
            'trunk_type': 'ff',
            'trunk_fourier_dim': 256,
            'trunk_fourier_sigma': 1.0,
            'trunk_hidden': 384,
            'trunk_depth': 5,
            'trunk_cond_mode': 'film',
        }
        result = fit_one_trial(config, overrides=overrides, max_train_steps=None,
                               force_no_physics=True, force_amp=True)
        with open(outdir / 'stage0_result.json', 'w') as f:
            json.dump({'case': case['name'], 'result': result}, f, indent=2)


def pick_winner_family(study: optuna.Study) -> str:
    # choose family with best median value
    fam_values: Dict[str, list] = {'ff': [], 'siren': []}
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        fam = t.params.get('family')
        if fam in fam_values:
            fam_values[fam].append(t.value)
    best_fam = 'ff'
    best_score = float('inf')
    for fam, vals in fam_values.items():
        if not vals:
            continue
        med = sorted(vals)[len(vals) // 2]
        if med < best_score:
            best_score = med
            best_fam = fam
    return best_fam


def stage1(args, cfg: Dict[str, Any]) -> None:
    study = get_study(args.study, args.storage)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage1')

    def objective(trial: optuna.Trial):
        family = trial.suggest_categorical('family', ['ff', 'siren'])
        params = sample_params(trial, family, 'stage1')
        run_dir = stage_root / f"trial_{trial.number:05d}"
        overrides = dict(params)
        overrides.update({
            'outdir': str(run_dir),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
        })
        result = fit_one_trial(cfg, overrides=overrides, trial=trial,
                               force_no_physics=True, force_amp=True)
        trial.set_user_attr('used_overrides', overrides)
        trial.set_user_attr('trial_dir', str(run_dir))
        return float(result['best_mae_db'])

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_seconds, gc_after_trial=True)
    with open(stage_root / 'stage1_summary.json', 'w') as f:
        json.dump({'best_value': study.best_value, 'best_params': study.best_params}, f, indent=2)


def stage2(args, cfg: Dict[str, Any]) -> None:
    stage1_root = Path(args.output_root) / args.study / 'stage1'
    study1 = get_study(args.study, args.storage)
    winner_family = pick_winner_family(study1)
    study2_name = f"{args.study}_stage2"
    study2 = get_study(study2_name, args.storage.replace('.db', '_stage2.db'))
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage2')

    def objective(trial: optuna.Trial):
        params = sample_params(trial, winner_family, 'stage2')
        run_dir = stage_root / f"trial_{trial.number:05d}"
        overrides = dict(params)
        overrides.update({
            'outdir': str(run_dir),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
        })
        result = fit_one_trial(cfg, overrides=overrides, trial=trial,
                               force_no_physics=True, force_amp=True)
        trial.set_user_attr('used_overrides', overrides)
        trial.set_user_attr('trial_dir', str(run_dir))
        return float(result['best_mae_db'])

    study2.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_seconds, gc_after_trial=True)
    with open(stage_root / 'stage2_summary.json', 'w') as f:
        json.dump({'best_value': study2.best_value, 'best_params': study2.best_params,
                   'winner_family': winner_family}, f, indent=2)


def stage3(args, cfg: Dict[str, Any]) -> None:
    study2_name = f"{args.study}_stage2"
    study2 = get_study(study2_name, args.storage.replace('.db', '_stage2.db'))
    best_params = dict(study2.best_params)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage3')
    seeds = [42, 43, 44]
    results = []
    for s in seeds:
        run_dir = stage_root / f"seed_{s}"
        overrides = dict(best_params)
        overrides.update({
            'outdir': str(run_dir),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
            'seed': s,
        })
        result = fit_one_trial(cfg, overrides=overrides, trial=None,
                               force_no_physics=True, force_amp=True)
        results.append({'seed': s, 'result': result})
    with open(stage_root / 'stage3_summary.json', 'w') as f:
        json.dump({'results': results, 'params': best_params}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_train.yaml')
    ap.add_argument('--study', type=str, default='trunk_global')
    ap.add_argument('--storage', type=str, default='sqlite:///experiments/optuna/trunk_global.db')
    ap.add_argument('--output-root', type=str, default='experiments/trunk_hpo')
    ap.add_argument('--stage', type=str, required=True,
                    choices=['stage0', 'stage1', 'stage2', 'stage3'])
    ap.add_argument('--n-trials', type=int, default=36)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--limit-files', type=int, default=200)
    ap.add_argument('--pts-per-map', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--timeout-hours', type=float, default=8.0)
    args = ap.parse_args()

    args.timeout_seconds = int(args.timeout_hours * 3600)
    cfg = load_config(args.config)
    ensure_dir(Path(args.output_root))

    if args.stage == 'stage0':
        stage0(cfg, Path(args.output_root) / args.study)
    elif args.stage == 'stage1':
        stage1(args, cfg)
    elif args.stage == 'stage2':
        # longer run defaults
        if args.n_trials == 36:
            args.n_trials = 12
        if args.epochs == 12:
            args.epochs = 60
        args.epochs = args.epochs if args.epochs else 60
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage2(args, cfg)
    elif args.stage == 'stage3':
        if args.epochs == 12:
            args.epochs = 60
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage3(args, cfg)


if __name__ == '__main__':
    main()
