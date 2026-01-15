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
    from train_runner import evaluate_mae_zones, mae_db_from_norm, build_norm_cfg
    from models import RDeepONetV2
    
    # Find best model from stage3_prime (wd=1e-4) or stage3
    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage4_ood')
    best_model_candidates = [
        Path(args.output_root) / args.study / 'stage3_single' / 'wd_1e-04' / 'best.pt',
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
    norm_cfg = build_norm_cfg(cfg)
    tl_min = norm_cfg['tl_db']['min']
    tl_max = norm_cfg['tl_db']['max']
    
    data_root = cfg['data'].get('root', cfg['data'].get('path', 'R-DeepONet_Data/data/h5'))
    val_ds = RDeepONetH5(
        root=data_root,
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
    """Stage5: Epoch/LR/Warmup tuning with fixed FF-MLP architecture.
    
    Use --epoch-split to run on 3 machines (time-balanced):
      'short': Laptop (4070) - epochs {60, 75}, lr [5e-4, 1.2e-3]
      'mid':   Pro A (4500)  - epochs {90, 180}, lr conditional
      'long':  Pro B (4500)  - epochs {105, 120, 150}, lr conditional
      'all':   full range (default)
    """
    best_params = load_best_params_for_eval(args)
    
    # Determine epoch split mode
    epoch_split = getattr(args, 'epoch_split', 'all')
    stage_suffix = f"_stage5_{epoch_split}" if epoch_split != 'all' else '_stage5_epoch'
    stage_root = ensure_dir(Path(args.output_root) / args.study / f'stage5_{epoch_split}')
    study_name = f"{args.study}{stage_suffix}"
    
    # No pruner - epoch is budget variable
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=5)
    study = optuna.create_study(study_name=study_name, storage=args.storage.replace('.db', f'_stage5_{epoch_split}.db'),
                                load_if_exists=True, sampler=sampler, direction='minimize', pruner=None)
    
    # Fixed FF-MLP params from Stage3
    sigma_bank = best_params.get('sigma_bank', 'three_level')
    sigma_list = SIGMA_PRESETS.get(sigma_bank, SIGMA_PRESETS['three_level'])[0]
    dim_total = best_params.get('trunk_fourier_dim_total', 1536)
    ratio_str = best_params.get('low_high_dim_ratio', '70_30')
    ratio = 0.7 if ratio_str == '70_30' else 0.5
    fourier_dims = split_dim(dim_total, sigma_list, ratio=ratio)
    depth = best_params.get('trunk_depth_stage2', best_params.get('trunk_depth', 7))

    def objective(trial: optuna.Trial):
        # Epochs based on split mode (3-way split for 3 machines, time-balanced)
        if epoch_split == 'short':
            # Laptop (RTX 4070): shortest epochs
            epochs = trial.suggest_categorical('epochs', [60, 75])
            lr = trial.suggest_loguniform('lr', 5e-4, 1.2e-3)
        elif epoch_split == 'mid':
            # Pro A (RTX Pro 4500): short + longest mixed
            epochs = trial.suggest_categorical('epochs', [90, 180])
            if epochs == 90:
                lr = trial.suggest_loguniform('lr', 3e-4, 9e-4)
            else:  # 180
                lr = trial.suggest_loguniform('lr', 2e-4, 7e-4)
        elif epoch_split == 'long':
            # Pro B (RTX Pro 4500): medium-long range
            epochs = trial.suggest_categorical('epochs', [105, 120, 150])
            if epochs <= 120:
                lr = trial.suggest_loguniform('lr', 3e-4, 9e-4)
            else:  # 150
                lr = trial.suggest_loguniform('lr', 2e-4, 7e-4)
        else:  # 'all'
            epochs = trial.suggest_categorical('epochs', [60, 75, 90, 105, 120, 150, 180])
            # LR: epoch-conditional
            if epochs <= 75:
                lr = trial.suggest_loguniform('lr', 5e-4, 1.2e-3)
            elif epochs >= 150:
                lr = trial.suggest_loguniform('lr', 2e-4, 7e-4)
            elif epochs >= 90:
                lr = trial.suggest_loguniform('lr', 3e-4, 9e-4)
            else:
                lr = trial.suggest_loguniform('lr', 4e-4, 1.0e-3)
        
        # Warmup ratio
        warmup_ratio = trial.suggest_categorical('warmup_ratio', [0.03, 0.05, 0.08])
        warmup_epochs = max(1, int(epochs * warmup_ratio))
        
        overrides = dict(best_params)
        # Fixed FF-MLP params
        overrides['trunk_type'] = 'ff'
        overrides['trunk_fourier_sigmas'] = sigma_list
        overrides['trunk_fourier_dims'] = fourier_dims
        overrides['trunk_hidden'] = best_params.get('trunk_hidden', 768)
        overrides['trunk_depth'] = depth
        overrides['trunk_cond_mode'] = 'concat'
        overrides['weight_decay'] = 1e-4  # Fixed from Stage3-Prime winner
        
        overrides.update({
            'epochs': epochs,
            'lr': lr,
            'warmup_epochs': warmup_epochs,
            'outdir': str(stage_root / f"trial_{trial.number:05d}"),
            'limit_files': args.limit_files,
            'pts_per_map': args.pts_per_map,
            'batch_size': args.batch_size,
            'use_amp': True,
            'seed': 0,
        })
        
        res = fit_one_trial(cfg, overrides=overrides, trial=trial,
                            force_no_physics=True, force_amp=True, enable_gate=False)
        return float(res['best_mae_db'])

    study.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)
    
    summary = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'epoch_split': epoch_split,
        'fixed_params': {
            'trunk_type': 'ff',
            'sigma_bank': sigma_bank,
            'fourier_dim_total': dim_total,
            'dim_ratio': ratio_str,
            'depth': depth,
            'weight_decay': 1e-4,
        },
        'topk': [{'number': t.number, 'value': t.value, 'params': t.params} for t in trials_sorted[:6]]
    }
    with open(stage_root / 'stage5_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def build_stage6_fixed_overrides(args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sigma_bank = 'three_level'
    sigma_list = SIGMA_PRESETS[sigma_bank][0]
    dim_total = 1536
    ratio = 0.7
    fourier_dims = split_dim(dim_total, sigma_list, ratio=ratio)
    epochs = 180
    warmup_epochs = max(1, int(epochs * 0.05))
    physics_warmup_epochs = 80

    fixed_overrides = {
        'trunk_type': 'ff',
        'trunk_fourier_sigmas': sigma_list,
        'trunk_fourier_dims': fourier_dims,
        'trunk_hidden': 768,
        'trunk_depth': 7,
        'trunk_cond_mode': 'concat',
        'trunk_fourier_dim_total': dim_total,
        'low_high_dim_ratio': '70_30',
        'edge_weight_scale': 8,
        'epochs': epochs,
        'lr': 2.8e-4,
        'warmup_epochs': warmup_epochs,
        'weight_decay': 1e-4,
        'limit_files': None,
        'pts_per_map': args.pts_per_map,
        'batch_size': args.batch_size,
        'use_amp': True,
        'seed': 42,
        'grace_epochs': epochs + 1,
        'patience': epochs + 1,
        'loss_type': 'huber',
        'loss_reciprocity_weight': 0.0,
        'loss_smooth_weight': 0.0,
        'loss_tv_weight': 0.0,
        'physics_warmup_epochs': physics_warmup_epochs,
        'save_best_ckpt': True,
        'save_periodic_ckpt': False,
        'save_debug_figures': False,
        'save_final_ckpt': False,
        'mlflow_artifacts': False,
        'save_best_ckpt_mode': 'deferred',
        'val_every_epochs': args.stage6_val_every_epochs,
        'highfreq_every_epochs': args.stage6_highfreq_every_epochs,
        'val_limit_files': args.stage6_val_limit_files,
        'compute_highfreq': args.stage6_compute_highfreq,
        'tv_batch_limit': 2,
    }
    meta = {
        'epochs': epochs,
        'lr': 2.8e-4,
        'warmup_ratio': 0.05,
        'weight_decay': 1e-4,
        'trunk_type': 'ff',
        'sigma_bank': sigma_bank,
        'fourier_dim_total': dim_total,
        'dim_ratio': '70_30',
        'depth': 7,
        'cond_mode': 'concat',
        'physics_warmup_epochs': physics_warmup_epochs,
        'loss_reciprocity_weight': 0.0,
        'loss_tv_weight': 0.0,
        'loss_smooth_weight_range': [1e-7, 1e-5],
        'loss_tv_weight_range': [1e-7, 1e-5],
        'loss_reciprocity_weight_range': [1e-5, 1e-2],
        'huber_delta_range': [0.5, 1.2],
        'stage6_reg_mode': args.stage6_reg_mode,
    }
    return fixed_overrides, meta


def build_stage6_sampler(args) -> optuna.samplers.BaseSampler:
    backend = args.stage6_autosampler_backend
    seed = 42

    if backend in ('auto', 'optunahub'):
        try:
            import optunahub
            # Use load_module (correct API)
            mod = optunahub.load_module("samplers/auto_sampler")
            if hasattr(mod, 'AutoSampler'):
                return mod.AutoSampler(seed=seed)
        except Exception as e:
            if backend == 'optunahub':
                raise RuntimeError(f"Failed to load AutoSampler from optunahub: {e}")
            # Fall through to TPE for 'auto'

    if backend in ('auto', 'optuna'):
        # Try optuna built-in AutoSampler (if exists in future versions)
        auto_sampler_cls = getattr(optuna.samplers, 'AutoSampler', None)
        if auto_sampler_cls is not None:
            return auto_sampler_cls(seed=seed)
        raise RuntimeError("AutoSampler not available. Install optunahub: pip install optunahub")

    raise ValueError(f"Unknown stage6 autosampler backend: {backend}")


def stage6_physics(args, cfg: Dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader
    from train_runner import build_norm_cfg, compute_zone_mae, build_model, make_datasets
    from utils_eval import infer_full_map, mae_db_from_norm

    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage6_physics')
    study_name = f"{args.study}_stage6_physics"

    sampler = build_stage6_sampler(args)
    pruner = SuccessiveHalvingPruner(min_resource=10, reduction_factor=3, min_early_stopping_rate=0)
    study = optuna.create_study(study_name=study_name, storage=args.storage.replace('.db', '_stage6.db'),
                                load_if_exists=True, sampler=sampler, pruner=pruner, direction='minimize')

    fixed_overrides, fixed_meta = build_stage6_fixed_overrides(args)
    ood_eval_mode = args.stage6_ood_eval_mode
    ood_lite_count = max(1, int(args.stage6_ood_lite_count))
    reg_mode = args.stage6_reg_mode

    split_ratio = {'train': 0.8, 'val': 0.2}
    eval_overrides = dict(fixed_overrides)
    eval_overrides['pts_per_map'] = args.pts_per_map
    _, val_ds_full = make_datasets(cfg, split_ratio, eval_overrides)
    val_loader = DataLoader(val_ds_full, batch_size=1, shuffle=False, num_workers=0)
    ood_lite_count = min(ood_lite_count, len(val_ds_full))
    ood_lite_indices = set(range(ood_lite_count))

    def eval_iid_ood_metrics(eval_model: torch.nn.Module, outdir: Path) -> Dict[str, Any]:
        norm_cfg = build_norm_cfg(cfg)
        tl_min = norm_cfg['tl_db']['min']
        tl_max = norm_cfg['tl_db']['max']
        iid_limit = args.stage6_val_limit_files
        if iid_limit is not None:
            iid_limit = int(iid_limit)
            if iid_limit <= 0:
                iid_limit = None
        max_needed_idx = None
        if iid_limit is not None:
            max_needed_idx = iid_limit - 1
        if ood_eval_mode == 'lite':
            base_idx = max_needed_idx if max_needed_idx is not None else -1
            max_needed_idx = max(base_idx, ood_lite_count - 1)

        eval_model.eval()
        device = next(eval_model.parameters()).device

        results = {'IID': [], 'OOD_range_70_30': [], 'OOD_range_far': []}
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if max_needed_idx is not None and idx > max_needed_idx:
                    break
                need_iid = iid_limit is None or idx < iid_limit
                need_ood = (ood_eval_mode == 'full' or idx in ood_lite_indices)
                if not (need_iid or need_ood):
                    continue
                ray = batch['ray'].to(device)
                cond = batch['cond'].to(device)
                tl_gt_norm = batch['tl'].cpu().squeeze()

                tl_pred_norm = infer_full_map(eval_model, ray, cond, device=device, require_forward_full=True)

                H, W = tl_gt_norm.shape
                r_coords = torch.linspace(0, 1, W)
                iid_mask = r_coords <= 0.7
                ood_mask = r_coords > 0.7
                far_mask = r_coords > 0.8

                def zone_mae(pred, gt, mask):
                    mask_2d = mask.unsqueeze(0).expand(H, -1)
                    if not mask_2d.any():
                        return None
                    pred_zone = pred[mask_2d]
                    gt_zone = gt[mask_2d]
                    return mae_db_from_norm(pred_zone, gt_zone, tl_min, tl_max).item()

                if need_iid:
                    iid_mae = zone_mae(tl_pred_norm, tl_gt_norm, iid_mask)
                    overall, mid, caustic, highfreq = compute_zone_mae(
                        tl_pred_norm.unsqueeze(0).unsqueeze(0),
                        tl_gt_norm.unsqueeze(0).unsqueeze(0),
                        tl_min, tl_max
                    )
                    results['IID'].append({'mae': iid_mae, 'overall': overall, 'mid': mid,
                                           'caustic': caustic, 'highfreq': highfreq})
                if need_ood:
                    ood_mae = zone_mae(tl_pred_norm, tl_gt_norm, ood_mask)
                    far_mae = zone_mae(tl_pred_norm, tl_gt_norm, far_mask)
                    if ood_mae is not None:
                        results['OOD_range_70_30'].append({'mae': ood_mae})
                    if far_mae is not None:
                        results['OOD_range_far'].append({'mae': far_mae})

        def avg(lst, key='mae'):
            vals = [x[key] for x in lst if x.get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        iid_fullgrid = avg(results['IID'], 'overall')
        iid_mid = avg(results['IID'], 'mid')
        iid_caustic = avg(results['IID'], 'caustic')
        iid_highfreq = avg(results['IID'], 'highfreq')
        ood_70_30 = avg(results['OOD_range_70_30'])
        ood_far = avg(results['OOD_range_far'])
        delta_70_30 = (ood_70_30 - iid_fullgrid) if (ood_70_30 is not None and iid_fullgrid is not None) else None
        delta_far = (ood_far - iid_fullgrid) if (ood_far is not None and iid_fullgrid is not None) else None

        return {
            'model_path': str(outdir / 'final.pt'),
            'n_samples': len(results['IID']),
            'ood_samples': len(results['OOD_range_70_30']),
            'ood_eval_mode': ood_eval_mode,
            'ood_lite_count': ood_lite_count,
            'IID_fullgrid': iid_fullgrid,
            'IID_mid': iid_mid,
            'IID_caustic': iid_caustic,
            'IID_highfreq': iid_highfreq,
            'OOD_range_70_30': ood_70_30,
            'OOD_range_far': ood_far,
            'delta_70_30': delta_70_30,
            'delta_far': delta_far,
            'mae_full': iid_fullgrid,
            'mae_shadow': ood_70_30,
            'mae_far': ood_far,
        }

    def compute_objective(metrics: Dict[str, Any]) -> float:
        mae_full = metrics.get('IID_fullgrid')
        mae_shadow = metrics.get('OOD_range_70_30')
        mae_far = metrics.get('OOD_range_far')
        if None in (mae_full, mae_shadow, mae_far):
            return float('inf')
        return (float(mae_full) + float(mae_shadow) + float(mae_far)) / 3.0

    def objective(trial: optuna.Trial):
        huber_delta = trial.suggest_float('huber_delta', 0.5, 1.2)
        lambda_grad = 0.0
        lambda_tv = 0.0
        lambda_rec = 0.0
        reg_type = reg_mode

        if reg_mode == 'grad':
            lambda_grad = trial.suggest_float('loss_smooth_weight', 1e-7, 1e-5, log=True)
        elif reg_mode == 'tv':
            lambda_tv = trial.suggest_float('loss_tv_weight', 1e-7, 1e-5, log=True)
        elif reg_mode == 'grad_tv':
            lambda_grad = trial.suggest_float('loss_smooth_weight', 1e-7, 1e-5, log=True)
            lambda_tv = trial.suggest_float('loss_tv_weight', 1e-7, 1e-5, log=True)
        elif reg_mode == 'rec':
            lambda_rec = trial.suggest_float('loss_reciprocity_weight', 1e-5, 1e-2, log=True)
        elif reg_mode == 'auto':
            reg_type = trial.suggest_categorical('reg_type', ['grad', 'tv', 'rec'])
            if reg_type == 'grad':
                lambda_grad = trial.suggest_float('loss_smooth_weight', 1e-7, 1e-5, log=True)
            elif reg_type == 'tv':
                lambda_tv = trial.suggest_float('loss_tv_weight', 1e-7, 1e-5, log=True)
            else:
                lambda_rec = trial.suggest_float('loss_reciprocity_weight', 1e-5, 1e-2, log=True)
        else:
            raise ValueError(f"Unknown stage6 reg mode: {reg_mode}")

        overrides = dict(fixed_overrides)
        overrides.update({
            'outdir': str(stage_root / f"trial_{trial.number:05d}"),
            'huber_delta': huber_delta,
            'loss_reciprocity_weight': lambda_rec,
            'loss_smooth_weight': lambda_grad,
            'loss_tv_weight': lambda_tv,
        })

        res = fit_one_trial(cfg, overrides=overrides, trial=trial,
                            force_no_physics=False, force_amp=True, enable_gate=False, return_model=True)
        eval_model = res.pop('eval_model', None)
        res.pop('device', None)
        if eval_model is None:
            return float('inf')
        metrics = eval_iid_ood_metrics(eval_model, Path(overrides['outdir']))
        obj = compute_objective(metrics)

        trial_summary = {
            'trial_number': trial.number,
            'objective': obj,
            'params': {
                'reg_type': reg_type,
                'huber_delta': huber_delta,
                'loss_reciprocity_weight': lambda_rec,
                'loss_smooth_weight': lambda_grad,
                'loss_tv_weight': lambda_tv,
            },
            'fixed_params': fixed_meta,
            'metrics': metrics,
            'fit_result': res,
        }
        with open(Path(overrides['outdir']) / 'stage6_trial_summary.json', 'w') as f:
            json.dump(trial_summary, f, indent=2, default=str)

        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('mae_full', metrics.get('mae_full'))
        trial.set_user_attr('mae_shadow', metrics.get('mae_shadow'))
        trial.set_user_attr('mae_far', metrics.get('mae_far'))
        trial.set_user_attr('objective', obj)
        if eval_model is not None:
            del eval_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return obj

    study.optimize(objective, n_trials=args.n_trials, timeout=None, gc_after_trial=True)
    trials_sorted = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials_sorted, key=lambda t: t.value)

    summary = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'fixed_params': fixed_meta,
        'ood_eval_mode': ood_eval_mode,
        'ood_lite_count': ood_lite_count,
        'objective_def': 'mean(IID_fullgrid, OOD_range_70_30, OOD_range_far)',
        'reg_mode': reg_mode,
        'autosampler_backend': args.stage6_autosampler_backend,
        'topk': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'metrics': t.user_attrs.get('metrics')
            }
            for t in trials_sorted[:6]
        ]
    }
    with open(stage_root / 'stage6_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def stage6_eval_topk(args, cfg: Dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader
    from train_runner import build_norm_cfg, compute_zone_mae, build_model, make_datasets
    from utils_eval import infer_full_map, mae_db_from_norm

    stage_root = ensure_dir(Path(args.output_root) / args.study / 'stage6_physics')
    summary_path = stage_root / 'stage6_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing stage6 summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    topk = summary.get('topk', [])[:args.stage6_topk]

    fixed_overrides, fixed_meta = build_stage6_fixed_overrides(args)

    split_ratio = {'train': 0.8, 'val': 0.2}
    eval_overrides = dict(fixed_overrides)
    eval_overrides['pts_per_map'] = args.pts_per_map
    _, val_ds_full = make_datasets(cfg, split_ratio, eval_overrides)
    val_loader = DataLoader(val_ds_full, batch_size=1, shuffle=False, num_workers=0)

    def eval_full_ood(model_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
        norm_cfg = build_norm_cfg(cfg)
        tl_min = norm_cfg['tl_db']['min']
        tl_max = norm_cfg['tl_db']['max']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(cfg, overrides).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        results = {'IID': [], 'OOD_range_70_30': [], 'OOD_range_far': []}
        with torch.no_grad():
            for batch in val_loader:
                ray = batch['ray'].to(device)
                cond = batch['cond'].to(device)
                tl_gt_norm = batch['tl'].cpu().squeeze()

                tl_pred_norm = infer_full_map(model, ray, cond, device=device, require_forward_full=True)

                H, W = tl_gt_norm.shape
                r_coords = torch.linspace(0, 1, W)
                iid_mask = r_coords <= 0.7
                ood_mask = r_coords > 0.7
                far_mask = r_coords > 0.8

                def zone_mae(pred, gt, mask):
                    mask_2d = mask.unsqueeze(0).expand(H, -1)
                    if not mask_2d.any():
                        return None
                    pred_zone = pred[mask_2d]
                    gt_zone = gt[mask_2d]
                    return mae_db_from_norm(pred_zone, gt_zone, tl_min, tl_max).item()

                iid_mae = zone_mae(tl_pred_norm, tl_gt_norm, iid_mask)
                ood_mae = zone_mae(tl_pred_norm, tl_gt_norm, ood_mask)
                far_mae = zone_mae(tl_pred_norm, tl_gt_norm, far_mask)

                overall, mid, caustic, highfreq = compute_zone_mae(
                    tl_pred_norm.unsqueeze(0).unsqueeze(0),
                    tl_gt_norm.unsqueeze(0).unsqueeze(0),
                    tl_min, tl_max
                )
                results['IID'].append({'mae': iid_mae, 'overall': overall, 'mid': mid,
                                       'caustic': caustic, 'highfreq': highfreq})
                results['OOD_range_70_30'].append({'mae': ood_mae})
                results['OOD_range_far'].append({'mae': far_mae})

        def avg(lst, key='mae'):
            vals = [x[key] for x in lst if x.get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        iid_fullgrid = avg(results['IID'], 'overall')
        iid_mid = avg(results['IID'], 'mid')
        iid_caustic = avg(results['IID'], 'caustic')
        iid_highfreq = avg(results['IID'], 'highfreq')
        ood_70_30 = avg(results['OOD_range_70_30'])
        ood_far = avg(results['OOD_range_far'])
        delta_70_30 = (ood_70_30 - iid_fullgrid) if (ood_70_30 is not None and iid_fullgrid is not None) else None
        delta_far = (ood_far - iid_fullgrid) if (ood_far is not None and iid_fullgrid is not None) else None

        return {
            'model_path': str(model_path),
            'n_samples': len(results['IID']),
            'IID_fullgrid': iid_fullgrid,
            'IID_mid': iid_mid,
            'IID_caustic': iid_caustic,
            'IID_highfreq': iid_highfreq,
            'OOD_range_70_30': ood_70_30,
            'OOD_range_far': ood_far,
            'delta_70_30': delta_70_30,
            'delta_far': delta_far,
            'mae_full': iid_fullgrid,
            'mae_shadow': ood_70_30,
            'mae_far': ood_far,
        }

    top_root = ensure_dir(stage_root / 'topk_eval')
    for entry in topk:
        params = entry.get('params', {})
        huber_delta = params.get('huber_delta', 1.0)
        lambda_rec = params.get('loss_reciprocity_weight', 0.0)
        lambda_grad = params.get('loss_smooth_weight', 0.0)
        lambda_tv = params.get('loss_tv_weight', 0.0)

        trial_number = entry.get('number', 0)
        trial_root = ensure_dir(top_root / f"trial_{trial_number:05d}")
        seed_results = []
        for seed in [0, 1, 42]:
            overrides = dict(fixed_overrides)
            overrides.update({
                'outdir': str(trial_root / f"seed_{seed}"),
                'seed': seed,
                'huber_delta': huber_delta,
                'loss_reciprocity_weight': lambda_rec,
                'loss_smooth_weight': lambda_grad,
                'loss_tv_weight': lambda_tv,
            })
            res = fit_one_trial(cfg, overrides=overrides, trial=None,
                                force_no_physics=False, force_amp=True, enable_gate=False)
            model_path = Path(overrides['outdir']) / 'best.pt'
            if not model_path.exists():
                raise FileNotFoundError(f"Missing best.pt at {model_path}")
            metrics = eval_full_ood(model_path, overrides)
            metrics['seed'] = seed
            metrics['fit_result'] = res
            seed_results.append(metrics)
            with open(Path(overrides['outdir']) / 'stage6_seed_summary.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        with open(trial_root / 'stage6_topk_summary.json', 'w') as f:
            json.dump({
                'trial_number': trial_number,
                'params': params,
                'fixed_params': fixed_meta,
                'seeds': seed_results,
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
                    choices=['stage0', 'stage1', 'stage2', 'stage3', 'stage3_prime', 'stage3_single', 'stage4', 'stage5', 'stage6', 'stage6_eval_topk', 'stage7'])
    ap.add_argument('--n-trials', type=int, default=36)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--limit-files', type=int, default=200)
    ap.add_argument('--pts-per-map', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--timeout-hours', type=float, default=8.0)
    ap.add_argument('--eval-subset', type=int, default=256, help='Fixed eval subset size for stage1')
    ap.add_argument('--split-ids', nargs='*', default=None, help='OOD split ids for stage4')
    ap.add_argument('--epoch-split', type=str, default='all', choices=['short', 'mid', 'long', 'all'],
                    help='Stage5 epoch split: short={60,75}, mid={90,180}, long={105,120,150}, all=full')
    ap.add_argument('--stage6-ood-eval-mode', type=str, default='lite', choices=['lite', 'full'],
                    help='Stage6 OOD eval mode: lite uses fixed subset, full uses all')
    ap.add_argument('--stage6-ood-lite-count', type=int, default=64,
                    help='Stage6 lite OOD sample count (fixed subset size)')
    ap.add_argument('--stage6-reg-mode', type=str, default='grad',
                    choices=['grad', 'tv', 'grad_tv', 'rec', 'auto'],
                    help='Stage6 regularization search mode')
    ap.add_argument('--stage6-autosampler-backend', type=str, default='auto',
                    choices=['auto', 'optuna', 'optunahub'],
                    help='Stage6 AutoSampler backend')
    ap.add_argument('--stage6-val-every-epochs', type=int, default=5,
                    help='Stage6 validation frequency (epochs)')
    ap.add_argument('--stage6-highfreq-every-epochs', type=int, default=10,
                    help='Stage6 highfreq metric frequency (epochs)')
    ap.add_argument('--stage6-val-limit-files', type=int, default=64,
                    help='Stage6 val limit files (fixed leading subset)')
    ap.add_argument('--stage6-compute-highfreq', action='store_true',
                    help='Stage6 compute highfreq metric during val')
    ap.add_argument('--stage6-topk', type=int, default=3,
                    help='Stage6 top-k to re-evaluate in stage6_eval_topk')
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
        if args.n_trials == 36:
            args.n_trials = 12
        args.epochs = 180
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage6_physics(args, cfg)
    elif args.stage == 'stage6_eval_topk':
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage6_eval_topk(args, cfg)
    elif args.stage == 'stage7':
        args.limit_files = None
        args.pts_per_map = 8192
        args.batch_size = 8
        stage7_branch(args, cfg)


if __name__ == '__main__':
    main()
