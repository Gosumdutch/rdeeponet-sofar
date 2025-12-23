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
    if stage == 'stage1':
        params['trunk_depth'] = trial.suggest_categorical('trunk_depth', [4, 5, 6])
    else:
        params['trunk_depth'] = 6
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
    """Stage3: Reproducibility test with 3 seeds."""
    study2_name = f"{args.study}_stage2"
    study2 = get_study(study2_name, args.storage.replace('.db', '_stage2.db'))
    best_params = dict(study2.best_params)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage3')
    
    # Compute FF-MLP params from best_params
    sigma_bank = best_params.get('sigma_bank', 'three_level')
    sigma_list = SIGMA_PRESETS.get(sigma_bank, SIGMA_PRESETS['three_level'])[0]
    dim_total = best_params.get('trunk_fourier_dim_total', 1536)
    ratio_str = best_params.get('low_high_dim_ratio', '70_30')
    ratio = 0.7 if ratio_str == '70_30' else 0.5
    fourier_dims = split_dim(dim_total, sigma_list, ratio=ratio)
    
    seeds = [0, 1, 2]  # Changed from [42, 43, 44]
    results = []
    for s in seeds:
        run_dir = stage_root / f"seed_{s}"
        overrides = dict(best_params)
        # Add FF-MLP params
        overrides['trunk_type'] = 'ff'
        overrides['trunk_fourier_sigmas'] = sigma_list
        overrides['trunk_fourier_dims'] = fourier_dims
        overrides['trunk_hidden'] = best_params.get('trunk_hidden', 768)
        overrides['trunk_depth'] = best_params.get('trunk_depth_stage2', best_params.get('trunk_depth', 6))
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
        results.append({
            'seed': s,
            'mae_fullgrid': result.get('best_mae_db'),
            'mae_mid': result.get('val_mae_mid_db'),
            'mae_caustic': result.get('val_mae_caustic_db'),
            'mae_highfreq': result.get('val_mae_highfreq_db'),
            'result': result
        })
    # Summary statistics
    mae_values = [r['mae_fullgrid'] for r in results if r['mae_fullgrid'] is not None]
    summary = {
        'results': results,
        'params': best_params,
        'stats': {
            'mean_mae': float(sum(mae_values) / len(mae_values)) if mae_values else None,
            'std_mae': float((sum((x - sum(mae_values)/len(mae_values))**2 for x in mae_values) / len(mae_values))**0.5) if len(mae_values) > 1 else 0.0,
            'min_mae': min(mae_values) if mae_values else None,
            'max_mae': max(mae_values) if mae_values else None,
        }
    }
    with open(stage_root / 'stage3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def stage3_prime(args, cfg: Dict[str, Any]) -> None:
    """Stage3-Prime: Weight decay re-exploration with grid search."""
    study2_name = f"{args.study}_stage2"
    study2 = get_study(study2_name, args.storage.replace('.db', '_stage2.db'))
    best_params = dict(study2.best_params)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage3_prime')
    
    # Compute FF-MLP params from best_params
    sigma_bank = best_params.get('sigma_bank', 'three_level')
    sigma_list = SIGMA_PRESETS.get(sigma_bank, SIGMA_PRESETS['three_level'])[0]
    dim_total = best_params.get('trunk_fourier_dim_total', 1536)
    ratio_str = best_params.get('low_high_dim_ratio', '70_30')
    ratio = 0.7 if ratio_str == '70_30' else 0.5
    fourier_dims = split_dim(dim_total, sigma_list, ratio=ratio)
    
    # Fixed params from Stage2 best, only vary weight_decay
    wd_grid = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 4e-3]
    results = []
    
    for wd in wd_grid:
        run_dir = stage_root / f"wd_{wd:.0e}" if wd > 0 else stage_root / "wd_0"
        overrides = dict(best_params)
        overrides['weight_decay'] = wd
        # Add FF-MLP params
        overrides['trunk_type'] = 'ff'
        overrides['trunk_fourier_sigmas'] = sigma_list
        overrides['trunk_fourier_dims'] = fourier_dims
        overrides['trunk_hidden'] = best_params.get('trunk_hidden', 768)
        overrides['trunk_depth'] = best_params.get('trunk_depth_stage2', best_params.get('trunk_depth', 6))
        overrides.update({
            'outdir': str(run_dir),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
            'seed': 0,  # Fixed seed for fair comparison
        })
        result = fit_one_trial(cfg, overrides=overrides, trial=None,
                               force_no_physics=True, force_amp=True)
        results.append({
            'weight_decay': wd,
            'mae_fullgrid': result.get('best_mae_db'),
            'mae_mid': result.get('val_mae_mid_db'),
            'mae_caustic': result.get('val_mae_caustic_db'),
            'mae_highfreq': result.get('val_mae_highfreq_db'),
            'result': result
        })
    
    # Find best wd
    best_result = min(results, key=lambda x: x['mae_fullgrid'] if x['mae_fullgrid'] else float('inf'))
    summary = {
        'results': results,
        'base_params': best_params,
        'best_wd': best_result['weight_decay'],
        'best_mae': best_result['mae_fullgrid'],
        'wd_grid': wd_grid
    }
    with open(stage_root / 'stage3_prime_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def stage3_single(args, cfg: Dict[str, Any]) -> None:
    """Stage3-Single: Single run with wd=1e-4 using FF-MLP."""
    study2_name = f"{args.study}_stage2"
    study2 = get_study(study2_name, args.storage.replace('.db', '_stage2.db'))
    best_params = dict(study2.best_params)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage3_single')
    
    # Compute FF-MLP params from best_params
    sigma_bank = best_params.get('sigma_bank', 'three_level')
    sigma_list = SIGMA_PRESETS.get(sigma_bank, SIGMA_PRESETS['three_level'])[0]
    dim_total = best_params.get('trunk_fourier_dim_total', 1536)
    ratio_str = best_params.get('low_high_dim_ratio', '70_30')
    ratio = 0.7 if ratio_str == '70_30' else 0.5
    fourier_dims = split_dim(dim_total, sigma_list, ratio=ratio)
    
    run_dir = stage_root / "wd_1e-04"
    overrides = dict(best_params)
    overrides['weight_decay'] = 1e-4
    # Add FF-MLP params
    overrides['trunk_type'] = 'ff'
    overrides['trunk_fourier_sigmas'] = sigma_list
    overrides['trunk_fourier_dims'] = fourier_dims
    overrides['trunk_hidden'] = best_params.get('trunk_hidden', 768)
    overrides['trunk_depth'] = best_params.get('trunk_depth_stage2', best_params.get('trunk_depth', 6))
    overrides.update({
        'outdir': str(run_dir),
        'epochs': args.epochs,
        'limit_files': args.limit_files,
        'pts_per_map': args.pts_per_map,
        'batch_size': args.batch_size,
        'use_amp': True,
        'seed': 0,
    })
    
    print(f"Running stage3_single with FF-MLP:")
    print(f"  trunk_type: ff")
    print(f"  sigma_bank: {sigma_bank}")
    print(f"  fourier_sigmas: {sigma_list}")
    print(f"  fourier_dims: {fourier_dims}")
    print(f"  trunk_hidden: {overrides['trunk_hidden']}")
    print(f"  trunk_depth: {overrides['trunk_depth']}")
    print(f"  weight_decay: {overrides['weight_decay']}")
    
    result = fit_one_trial(cfg, overrides=overrides, trial=None,
                           force_no_physics=True, force_amp=True)
    summary = {
        'mae_fullgrid': result.get('best_mae_db'),
        'mae_mid': result.get('val_mae_mid_db'),
        'mae_caustic': result.get('val_mae_caustic_db'),
        'mae_highfreq': result.get('val_mae_highfreq_db'),
        'params': overrides,
        'result': result
    }
    with open(stage_root / 'stage3_single_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def load_best_params_for_eval(args) -> Dict[str, Any]:
    stage3_path = Path(args.output_root) / args.study / 'stage3' / 'stage3_summary.json'
    stage2_path = Path(args.output_root) / args.study / 'stage2' / 'stage2_summary.json'
    if stage3_path.exists():
        with open(stage3_path, 'r') as f:
            data = json.load(f)
            return dict(data.get('params', {}))
    if stage2_path.exists():
        with open(stage2_path, 'r') as f:
            data = json.load(f)
            return dict(data.get('best_params', {}))
    raise FileNotFoundError("No stage3 or stage2 summary found for best params.")


def stage4_eval_ood(args, cfg: Dict[str, Any]) -> None:
    """Stage4: OOD evaluation without training. Load best model and evaluate on different range zones."""
    import torch
    from dataset import RDeepONetH5
    from torch.utils.data import DataLoader
    from train_runner import evaluate_mae_zones, mae_db_from_norm
    from models import RDeepONetV2
    
    # Find best model from stage3_prime (wd=1e-4) or stage3
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage4_ood')
    best_model_candidates = [
        Path(args.output_root) / args.study / 'stage3_prime' / 'wd_1e-04' / 'best.pt',
        Path(args.output_root) / args.study / 'stage3' / 'seed_1' / 'best.pt',
        Path(args.output_root) / args.study / 'stage2' / 'trial_00000' / 'best.pt',
    ]
    best_model_path = None
    for p in best_model_candidates:
        if p.exists():
            best_model_path = p
            break
    if best_model_path is None:
        raise FileNotFoundError(f"No best model found. Checked: {best_model_candidates}")
    
    # Load params from stage3 summary to build correct model config
    best_params = load_best_params_for_eval(args)
    
    # Build model config from params
    sigma_bank = best_params.get('sigma_bank', 'three_level')
    sigma_list = SIGMA_PRESETS.get(sigma_bank, SIGMA_PRESETS['three_level'])[0]
    dim_total = best_params.get('trunk_fourier_dim_total', 1536)
    ratio_str = best_params.get('low_high_dim_ratio', '70_30')
    ratio = 0.7 if ratio_str == '70_30' else 0.5
    fourier_dims = split_dim(dim_total, sigma_list, ratio=ratio)
    depth = best_params.get('trunk_depth_stage2', best_params.get('trunk_depth', 6))
    
    model_cfg = {
        'K': cfg.get('model', {}).get('K', 256),
        'hidden': best_params.get('trunk_hidden', 768),
        'depth': depth,
        'trunk_type': 'ff',
        'trunk_cond_mode': best_params.get('trunk_cond_mode', 'concat'),
        'trunk_fourier_sigmas': sigma_list,
        'trunk_fourier_dims': fourier_dims,
        'film_gain': best_params.get('trunk_film_gain', 1.0),
    }
    print(f"Model config: {model_cfg}")
    
    print(f"Loading model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RDeepONetV2(**model_cfg).to(device)
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is the state_dict itself
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load validation dataset
    norm_cfg = cfg.get('norm', {})
    tl_min = norm_cfg.get('tl', {}).get('tl_min', 40.0)
    tl_max = norm_cfg.get('tl', {}).get('tl_max', 120.0)
    
    val_ds = RDeepONetH5(
        root=cfg['data']['root'],
        split='val',
        split_ratio=cfg['data'].get('split_ratio', {'train': 0.8, 'val': 0.1, 'test': 0.1}),
        mode='full',
        pts_per_map=args.pts_per_map,
        norm_cfg=norm_cfg
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # Evaluate on different range zones
    results = {'IID': [], 'OOD_range_70_30': [], 'OOD_range_far': []}
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if args.limit_files and idx >= args.limit_files:
                break
            ray = batch['ray'].to(device)
            cond = batch['cond'].to(device)
            tl_gt_norm = batch['tl'].cpu().squeeze()
            
            # Full map inference
            from train_runner import infer_full_map
            tl_pred_norm = infer_full_map(model, ray, cond, device=device)
            
            H, W = tl_gt_norm.shape
            r_coords = torch.linspace(0, 1, W)
            
            # IID (full range 0-70%)
            iid_mask = r_coords <= 0.7
            # OOD_range_70_30 (70-100%)
            ood_mask = r_coords > 0.7
            # OOD_far (last 20%)
            far_mask = r_coords > 0.8
            
            # Compute MAE for each zone
            def zone_mae(pred, gt, mask, tl_min, tl_max):
                mask_2d = mask.unsqueeze(0).expand(H, -1)
                if not mask_2d.any():
                    return None
                pred_zone = pred[mask_2d]
                gt_zone = gt[mask_2d]
                return mae_db_from_norm(pred_zone, gt_zone, tl_min, tl_max).item()
            
            iid_mae = zone_mae(tl_pred_norm, tl_gt_norm, iid_mask, tl_min, tl_max)
            ood_mae = zone_mae(tl_pred_norm, tl_gt_norm, ood_mask, tl_min, tl_max)
            far_mae = zone_mae(tl_pred_norm, tl_gt_norm, far_mask, tl_min, tl_max)
            
            # Mid/caustic/highfreq for each zone
            from train_runner import compute_zone_mae
            overall, mid, caustic, highfreq = compute_zone_mae(
                tl_pred_norm.unsqueeze(0).unsqueeze(0),
                tl_gt_norm.unsqueeze(0).unsqueeze(0),
                tl_min, tl_max
            )
            
            results['IID'].append({'mae': iid_mae, 'overall': overall, 'mid': mid, 'caustic': caustic, 'highfreq': highfreq})
            results['OOD_range_70_30'].append({'mae': ood_mae})
            results['OOD_range_far'].append({'mae': far_mae})
    
    # Aggregate
    def avg(lst, key='mae'):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None
    
    iid_fullgrid = avg(results['IID'], 'overall')
    iid_mid = avg(results['IID'], 'mid')
    iid_caustic = avg(results['IID'], 'caustic')
    iid_highfreq = avg(results['IID'], 'highfreq')
    ood_70_30 = avg(results['OOD_range_70_30'])
    ood_far = avg(results['OOD_range_far'])
    
    delta_tl = (ood_70_30 - iid_fullgrid) if (ood_70_30 and iid_fullgrid) else None
    
    summary = {
        'model_path': str(best_model_path),
        'n_samples': len(results['IID']),
        'IID_fullgrid': iid_fullgrid,
        'IID_mid': iid_mid,
        'IID_caustic': iid_caustic,
        'IID_highfreq': iid_highfreq,
        'OOD_range_70_30': ood_70_30,
        'OOD_range_far': ood_far,
        'delta_TL': delta_tl,
    }
    
    print("\n=== Stage4 OOD Eval Results ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f} dB")
        else:
            print(f"  {k}: {v}")
    
    with open(stage_root / 'stage4_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def stage5_epoch_tune(args, cfg: Dict[str, Any]) -> None:
    best_params = load_best_params_for_eval(args)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage5_epoch')
    study_name = f"{args.study}_stage5_epoch"
    # No ASHA (pruner=None) because epoch is budget variable
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=10)
    study = optuna.create_study(study_name=study_name, storage=args.storage.replace('.db', '_stage5.db'),
                                load_if_exists=True, sampler=sampler, direction='minimize')

    def objective(trial: optuna.Trial):
        overrides = dict(best_params)
        overrides.update({
            'epochs': trial.suggest_categorical('epochs_tune', [60, 90, 120, 180]),
            'outdir': str(stage_root / f"trial_{trial.number:05d}"),
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
        })
        # lr/wd + physics DVs
        overrides['lr'] = trial.suggest_loguniform('lr', 6e-4, 2.0e-3)
        overrides['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-7, 5e-3)
        overrides['loss_type'] = 'huber'
        overrides['huber_delta'] = trial.suggest_categorical('huber_delta', [0.5, 1.0, 1.5])
        lambda_rec = trial.suggest_loguniform('lambda_rec', 1e-4, 1e-2)
        reg_type = trial.suggest_categorical('reg_type', ['none', 'grad', 'tv'])
        overrides['loss_reciprocity_weight'] = lambda_rec
        overrides['loss_smooth_weight'] = 0.0
        overrides['loss_tv_weight'] = 0.0
        if reg_type == 'grad':
            overrides['loss_smooth_weight'] = trial.suggest_loguniform('lambda_grad', 1e-6, 3e-4)
        elif reg_type == 'tv':
            overrides['loss_tv_weight'] = trial.suggest_loguniform('lambda_tv', 1e-6, 1e-4)
        overrides['physics_warmup_epochs'] = trial.suggest_categorical('physics_warmup_epochs', [20, 30, 40])
        res = fit_one_trial(cfg, overrides=overrides, trial=trial,
                            force_no_physics=False, force_amp=True, enable_gate=False)
        return float(res['best_mae_db'])

    study.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    with open(stage_root / 'stage5_summary.json', 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'topk': [{'number': t.number, 'value': t.value, 'params': t.params} for t in trials_sorted[:6]]
        }, f, indent=2)


def stage6_physics(args, cfg: Dict[str, Any]) -> None:
    best_params = load_best_params_for_eval(args)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage6_physics')
    study_name = f"{args.study}_stage6_physics"
    study = get_study(study_name, args.storage.replace('.db', '_stage6.db'))

    def objective(trial: optuna.Trial):
        overrides = dict(best_params)
        overrides.update({
            'outdir': str(stage_root / f"trial_{trial.number:05d}"),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
            'loss_type': 'huber',
            'huber_delta': trial.suggest_uniform('huber_delta', 0.5, 1.5),
        })
        lambda_rec = trial.suggest_loguniform('lambda_rec', 1e-5, 5e-2)
        reg_type = trial.suggest_categorical('reg_type', ['none', 'grad', 'tv'])
        overrides['loss_reciprocity_weight'] = lambda_rec
        overrides['loss_smooth_weight'] = 0.0
        overrides['loss_tv_weight'] = 0.0
        if reg_type == 'grad':
            overrides['loss_smooth_weight'] = trial.suggest_loguniform('lambda_grad', 1e-6, 1e-3)
        elif reg_type == 'tv':
            overrides['loss_tv_weight'] = trial.suggest_loguniform('lambda_tv', 1e-6, 1e-3)
        res = fit_one_trial(cfg, overrides=overrides, trial=trial,
                            force_no_physics=False, force_amp=True, enable_gate=False)
        return float(res['best_mae_db'])

    study.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    with open(stage_root / 'stage6_summary.json', 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'topk': [{'number': t.number, 'value': t.value, 'params': t.params} for t in trials_sorted[:6]]
        }, f, indent=2)


def stage7_branch(args, cfg: Dict[str, Any]) -> None:
    best_params = load_best_params_for_eval(args)
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage7_branch')
    study_name = f"{args.study}_stage7_branch"
    study = get_study(study_name, args.storage.replace('.db', '_stage7.db'))

    def objective(trial: optuna.Trial):
        overrides = dict(best_params)
        overrides.update({
            'outdir': str(stage_root / f"trial_{trial.number:05d}"),
            'epochs': args.epochs,
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
        })
        overrides['branch_variant'] = trial.suggest_categorical('branch_variant', ['resnet18', 'se_resnet18'])
        res = fit_one_trial(cfg, overrides=overrides, trial=trial,
                            force_no_physics=True, force_amp=True, enable_gate=False)
        return float(res['best_mae_db'])

    study.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    with open(stage_root / 'stage7_summary.json', 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'topk': [{'number': t.number, 'value': t.value, 'params': t.params} for t in trials_sorted[:6]]
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_train.yaml')
    ap.add_argument('--study', type=str, default='trunk_global')
    ap.add_argument('--storage', type=str, default='sqlite:///experiments/optuna/trunk_global.db')
    ap.add_argument('--output-root', type=str, default='experiments/trunk_hpo')
    ap.add_argument('--stage', type=str, required=True,
                    choices=['stage0', 'stage1', 'stage2', 'stage3', 'stage3_prime', 'stage3_single', 'stage4', 'stage5', 'stage6', 'stage7'])
    ap.add_argument('--n-trials', type=int, default=36)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--limit-files', type=int, default=200)
    ap.add_argument('--pts-per-map', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--timeout-hours', type=float, default=8.0)
    ap.add_argument('--eval-subset', type=int, default=256, help='Fixed eval subset size for stage1')
    ap.add_argument('--split-ids', nargs='*', default=None, help='OOD split ids for stage4')
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
    elif args.stage == 'stage3_prime':
        if args.epochs == 12:
            args.epochs = 60
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage3_prime(args, cfg)
    elif args.stage == 'stage3_single':
        if args.epochs == 12:
            args.epochs = 60
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage3_single(args, cfg)
    elif args.stage == 'stage4':
        # No training, just eval - use all data
        args.limit_files = None
        args.pts_per_map = 8192
        stage4_eval_ood(args, cfg)
    elif args.stage == 'stage5':
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage5_epoch_tune(args, cfg)
    elif args.stage == 'stage6':
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage6_physics(args, cfg)
    elif args.stage == 'stage7':
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage7_branch(args, cfg)


if __name__ == '__main__':
    main()
