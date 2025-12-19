"""
Trunk HPO pipeline (multiscale FF only), physics off, anti-smoothing.

Stages (names kept as stage1/2):
- stage1 (scout): DV 6개만, 저해상도/짧은 epoch, 강한 프루닝, 고정 eval 서브셋.
- stage2 (exploit): stage1 top-k=6 + sigma preset 상위 2개 유지, 3개 추가 DV, 풀 트레이닝.

Objective: overall + 0.5*mid + 0.5*caustic + penalty(mid/caustic >> overall).
Artifacts: run_dir per trial/seed with config_resolved.yaml, overrides.json,
metrics.csv, metrics_best.json, best.pt, figures/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from train_runner import fit_one_trial, load_config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


SIGMA_PRESETS = {
    "two_level": ([0.5, 2.0, 8.0, 32.0], "two_level"),
    "wide": ([0.25, 1.0, 16.0, 64.0], "wide"),
    "three_level": ([0.5, 2.0, 8.0, 32.0], "three_level"),
}


def split_dim(total: int, sigmas: list[float], ratio: float = 0.5) -> list[int]:
    n = len(sigmas)
    if n == 0:
        return []
    if n == 2:
        low = max(1, int(total * ratio))
        high = max(1, total - low)
        return [low, high]
    dims = [total // n] * n
    for i in range(total - sum(dims)):
        dims[i % n] += 1
    return dims


def sample_params(trial: optuna.Trial, stage: str, allowed_sigma_presets: Optional[list[str]] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    # DV set depends on stage
    params['trunk_type'] = 'ff'
    params['lr'] = trial.suggest_loguniform('lr', 5e-4, 3e-3)
    params['trunk_depth'] = trial.suggest_categorical('trunk_depth', [4, 5, 6])
    params['trunk_hidden'] = 768  # fixed for stage1
    params['trunk_cond_mode'] = trial.suggest_categorical('trunk_cond_mode', ['film', 'concat'])
    if params['trunk_cond_mode'] == 'film':
        params['trunk_film_gain'] = trial.suggest_uniform('trunk_film_gain', 0.5, 3.0)
    else:
        params['trunk_film_gain'] = 1.0
    sigma_choices = allowed_sigma_presets if allowed_sigma_presets else list(SIGMA_PRESETS.keys())
    sigma_key = trial.suggest_categorical('sigma_bank', sigma_choices)
    sigma_list, _ = SIGMA_PRESETS[sigma_key]
    params['trunk_fourier_sigmas'] = sigma_list
    params['trunk_fourier_dim_total'] = trial.suggest_categorical('trunk_fourier_dim_total', [1024, 1536])
    params['trunk_fourier_dims'] = split_dim(params['trunk_fourier_dim_total'], sigma_list, ratio=0.5)
    params['weight_decay'] = 1e-6
    params['edge_weight_scale'] = 8
    params['grad_threshold'] = 0.005

    if stage == 'stage2':
        params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 5e-3)
        dim_ratio = trial.suggest_categorical('low_high_dim_ratio', ['50_50', '70_30'])
        ratio = 0.5 if dim_ratio == '50_50' else 0.7
        params['trunk_fourier_dims'] = split_dim(params['trunk_fourier_dim_total'], sigma_list, ratio=ratio)
        params['edge_weight_scale'] = trial.suggest_categorical('edge_weight_scale', [5, 8])

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
        params = sample_params(trial, 'stage1')
        run_dir = stage_root / f"trial_{trial.number:05d}"
        overrides = dict(params)
        overrides.update({
            'outdir': str(run_dir),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
            'eval_subset': args.eval_subset,
        })
        result = fit_one_trial(cfg, overrides=overrides, trial=trial,
                               force_no_physics=True, force_amp=True, enable_gate=True)
        trial.set_user_attr('used_overrides', overrides)
        trial.set_user_attr('trial_dir', str(run_dir))
        trial.set_user_attr('val_mae_mid_db', result.get('val_mae_mid_db'))
        trial.set_user_attr('val_mae_caustic_db', result.get('val_mae_caustic_db'))
        # objective with penalty
        overall = float(result['best_mae_db'])
        mid = float(result.get('val_mae_mid_db', overall))
        caustic = float(result.get('val_mae_caustic_db', overall))
        combined = overall + 0.5 * mid + 0.5 * caustic
        delta = 0.15
        penalty = 0.3
        if mid > overall + delta:
            combined += penalty
        if caustic > overall + delta:
            combined += penalty
        return combined

    study.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    topk = trials_sorted[:6]
    sigma_counts = {}
    for t in trials_sorted:
        key = t.params.get('sigma_bank')
        sigma_counts[key] = sigma_counts.get(key, 0) + 1
    summary = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'topk': [
            {'number': t.number, 'value': t.value, 'sigma_bank': t.params.get('sigma_bank'),
             'params': t.params, 'mid': t.user_attrs.get('val_mae_mid_db'),
             'caustic': t.user_attrs.get('val_mae_caustic_db')}
            for t in topk
        ],
        'sigma_counts': sigma_counts
    }
    with open(stage_root / 'stage1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def stage2(args, cfg: Dict[str, Any]) -> None:
    stage1_root = Path(args.output_root) / args.study / 'stage1'
    study1 = get_study(args.study, args.storage)
    trials_sorted = [t for t in study1.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    topk_trials = trials_sorted[:6]
    # top-2 sigma presets
    sigma_order = []
    for t in topk_trials:
        key = t.params.get('sigma_bank')
        if key not in sigma_order:
            sigma_order.append(key)
        if len(sigma_order) >= 2:
            break
    allowed_sigma_presets = sigma_order if sigma_order else list(SIGMA_PRESETS.keys())
    study2_name = f"{args.study}_stage2"
    study2 = get_study(study2_name, args.storage.replace('.db', '_stage2.db'))
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage2')
    # Forced coverage queue: 12 combos (sigma × grad × ratio)
    forced_queue = []
    for sigma_key in allowed_sigma_presets:
        for grad_th in [0.005, 0.01, 0.02]:
            for ratio_key, ratio_val in [('50_50', 0.5), ('70_30', 0.7)]:
                forced_queue.append({'sigma_bank': sigma_key,
                                     'grad_threshold': grad_th,
                                     'low_high_dim_ratio': ratio_key})
                if len(forced_queue) >= 12:
                    break
            if len(forced_queue) >= 12:
                break
        if len(forced_queue) >= 12:
            break

    # Track Fourier dim coverage targets
    target_per_dim = {1024: 8, 1536: 8}
    dim_counts = {1024: 0, 1536: 0}

    def objective(trial: optuna.Trial):
        params = sample_params(trial, 'stage2', allowed_sigma_presets=allowed_sigma_presets)
        # Apply forced combos for first 12 trials
        if trial.number < len(forced_queue):
            forced = forced_queue[trial.number]
            params['sigma_bank'] = forced['sigma_bank']
            params['trunk_fourier_sigmas'] = SIGMA_PRESETS[params['sigma_bank']][0]
            params['grad_threshold'] = forced['grad_threshold']
            params['low_high_dim_ratio'] = forced['low_high_dim_ratio']
            ratio = 0.5 if forced['low_high_dim_ratio'] == '50_50' else 0.7
            params['trunk_fourier_dims'] = split_dim(params['trunk_fourier_dim_total'], params['trunk_fourier_sigmas'], ratio=ratio)

        # Enforce Fourier dim coverage
        if dim_counts[1024] < target_per_dim[1024] and dim_counts[1536] < target_per_dim[1536]:
            # choose the lesser-covered one
            target_dim = min(dim_counts, key=lambda k: dim_counts[k]/target_per_dim[k])
        else:
            target_dim = min(dim_counts, key=dim_counts.get)
        if dim_counts[target_dim] < target_per_dim[target_dim]:
            params['trunk_fourier_dim_total'] = target_dim
        dim_counts[params['trunk_fourier_dim_total']] += 1

        # Hard constraints for stage2
        params['trunk_cond_mode'] = 'concat'
        params['trunk_depth'] = trial.suggest_categorical('trunk_depth_stage2', [5, 6, 7])
        # First 4 trials: restrict lr to lower band
        if trial.number < 4:
            params['lr'] = trial.suggest_loguniform('lr_lower_band', 6e-4, 1.2e-3)
        else:
            params['lr'] = trial.suggest_loguniform('lr', 5e-4, 2.5e-3)

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
                               force_no_physics=True, force_amp=True, enable_gate=False)
        trial.set_user_attr('used_overrides', overrides)
        trial.set_user_attr('trial_dir', str(run_dir))
        trial.set_user_attr('val_mae_mid_db', result.get('val_mae_mid_db'))
        trial.set_user_attr('val_mae_caustic_db', result.get('val_mae_caustic_db'))
        overall = float(result['best_mae_db'])
        mid = float(result.get('val_mae_mid_db', overall))
        caustic = float(result.get('val_mae_caustic_db', overall))
        return overall + 0.5 * mid + 0.5 * caustic

    study2.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study2.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    topk = trials_sorted[:6]
    summary = {
        'best_value': study2.best_value,
        'best_params': study2.best_params,
        'topk': [
            {'number': t.number, 'value': t.value, 'sigma_bank': t.params.get('sigma_bank'),
             'params': t.params, 'mid': t.user_attrs.get('val_mae_mid_db'),
             'caustic': t.user_attrs.get('val_mae_caustic_db')}
            for t in topk
        ],
        'allowed_sigma_presets': allowed_sigma_presets
    }
    with open(stage_root / 'stage2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


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
    ap.add_argument('--eval-subset', type=int, default=256, help='Fixed eval subset size for stage1')
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
