"""Controlled 3-versus-13-threshold extension of the frozen CellFlow v1 run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd

import run_cellflow_v1 as base
from sciplex_eval import EvalSpec, evaluate_predictions, response_score, summarize
from sciplex_eval.metrics import weak_responder_fraction

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]
TRIAL = 'wrf_l0p1_h0p3'
METHOD = 'cellflow_wrf13'
METRICS = ['wrf_err_pp_q0.9', 'wrf_err_pp_q0.95', 'wrf_err_pp_q0.975', 'w1_r',
           'mmd2', 'rmse_mean_all', 'rmse_mean_deg50', 'rmse_mean_program', 'wrf_err_pp_grid13_mean']


def initialize(run, baseline):
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    (run / 'checks').mkdir()
    old = json.loads((baseline / 'config.json').read_text())
    selection = json.loads((baseline / 'selected_methods.json').read_text())
    assert selection['methods']['cellflow_wrf'] == TRIAL
    base.audit_panel(run)
    previous_panel = json.loads((baseline / 'panel_audit.json').read_text())
    current_panel = json.loads((run / 'panel_audit.json').read_text())
    assert current_panel['panel_sha256'] == previous_panel['panel_sha256'], 'The runtime panel must match the original run byte for byte.'
    spec = EvalSpec.load(base.REPO / 'artifacts/eval_spec_v1')
    panel = ad.read_h5ad(base.PANEL)
    controls = panel[panel.obs['split'].eq('control')]
    taus = np.quantile(response_score(controls.X, spec), QS)
    np.testing.assert_allclose(taus[-3:], spec.taus, atol=2e-6, rtol=0)
    taus[-3:] = spec.taus  # Preserve all three existing cutoffs exactly.
    assert np.isfinite(taus).all() and len(taus) == 13
    config = {**old, 'training_quantiles': QS, 'training_taus': taus.tolist()}
    assert {k: config[k] for k in old} == old
    base.save_json(run / 'config.json', config)
    base.save_json(run / 'selected_methods.json', {'methods': {METHOD: TRIAL},
        'selected_on': 'Fixed existing WRF setting; no hyperparameter search in this experiment.'})
    protocol = dict(baseline_run=str(baseline), run=str(run), seeds=old['seeds'],
        comparison='Only the supervised threshold set changes; loss still averages over thresholds.',
        weight=0.1, smoothing_h=0.3, train_steps=30000,
        validation_steps=old['validation_steps'], checkpoint_selection=old['checkpoint_selection'],
        training_quantiles=QS, training_taus=taus.tolist(), thresholds_from='Permitted vehicle controls only',
        unchanged_original_config_keys=sorted(old), baseline_config_sha256=base.digest(baseline / 'config.json'),
        baseline_prediction_policy=old['prediction_postprocessing'],
        exploratory=True, exploratory_reason='Requested after earlier CellFlow and chemCPA test results were reviewed.',
        evaluation='Frozen v1 eight metrics plus 13-threshold mean; errors computed per seed before averaging.',
        test_policy='Freeze all three selected checkpoints before scoring; no retuning from these test results.')
    base.save_json(run / 'ablation_protocol.json', protocol)
    shutil.copytree(base.ROOT / 'src', run / 'source', ignore=shutil.ignore_patterns('vc_wrf_1d8b173', '__pycache__'))
    shutil.copytree(base.REPO / 'artifacts/eval_spec_v1', run / 'eval_spec_v1')
    shutil.copyfile(base.REPO / 'configs/split_v1.json', run / 'split_v1.json')
    base.save_json(run / 'source_manifest.json', {str(p.relative_to(run)): base.digest(p)
        for p in run.rglob('*') if p.is_file()})
    print('INITIALIZED', str(run), 'taus', taus.tolist(), flush=True)


def preflight(run):
    import jax
    import jax.numpy as jnp
    from scipy.special import expit
    from cellflow_population_loss import response_score_jax, soft_fractions
    spec = EvalSpec.load(base.REPO / 'artifacts/eval_spec_v1')
    config = json.loads((run / 'config.json').read_text())
    taus = np.array(config['training_taus'])
    panel = ad.read_h5ad(base.PANEL)
    permitted = panel[panel.obs['split'].isin(['train', 'control'])]
    rng = np.random.default_rng(902)
    x = base.dense(permitted.X[rng.choice(permitted.n_obs, 128, replace=False)]).astype('float32')
    idx = jnp.asarray(spec.program_idx)
    got = np.asarray(soft_fractions(jnp.asarray(x), idx, spec.mu0, spec.s0, jnp.asarray(taus), 0.3))
    expected = expit((taus[None, :] - response_score(x, spec)[:, None]) / 0.3).mean(0)
    np.testing.assert_allclose(got, expected, atol=2e-6, rtol=2e-6)
    # Equal cutoff weighting must remain a mean, not grow the effective weight with Q.
    original = jnp.asarray(spec.taus)
    f3 = soft_fractions(jnp.asarray(x), idx, spec.mu0, spec.s0, original, 0.3)
    repeated = soft_fractions(jnp.asarray(x), idx, spec.mu0, spec.s0, jnp.tile(original, 3), 0.3)
    np.testing.assert_allclose(float(jnp.mean(f3 ** 2)), float(jnp.mean(repeated ** 2)), atol=1e-7)
    z = np.zeros((13, spec.n_genes), dtype=np.float32)
    z[:, spec.program_idx[0]] = np.maximum(spec.mu0 - (spec.s0 + 1e-6) * taus, 0.01)
    fn = lambda v: soft_fractions(v, idx, spec.mu0, spec.s0, jnp.asarray(taus), 0.3).mean()
    grad = np.asarray(jax.grad(fn)(jnp.asarray(z)))
    assert np.isfinite(grad).all() and np.linalg.norm(grad) > 0
    for value in [0., 100., 1000.]:
        extreme = np.zeros((2, spec.n_genes), dtype=np.float32)
        extreme[:, spec.program_idx[0]] = value
        assert np.isfinite(np.asarray(jax.grad(fn)(jnp.asarray(extreme)))).all()
    result = dict(passed=True, thresholds=13, hard_spec_thresholds_unchanged=spec.quantiles,
                  soft_fraction_max_error=float(np.max(np.abs(got - expected))),
                  finite_nonzero_gradients=True, loss_averages_over_cutoffs=True,
                  devices=[str(d) for d in jax.devices()])
    base.save_json(run / 'checks/threshold_loss_check.json', result)
    print('THRESHOLD_PREFLIGHT', json.dumps(result), flush=True)


def train(run, baseline, seed):
    directory = run / 'trials' / TRIAL / f'seed_{seed}'
    assert not directory.exists(), 'A fresh run is required; preserve partial attempts.'
    base.train_seed(run, seed, TRIAL)
    old = baseline / 'trials' / TRIAL / f'seed_{seed}'
    assert (directory / 'training_cell_ids.csv').read_bytes() == (old / 'training_cell_ids.csv').read_bytes()
    a = pd.read_csv(directory / 'training_loss.csv').query('step <= 3000')
    b = pd.read_csv(old / 'training_loss.csv').query('step <= 3000')
    assert len(a) == len(b) == 3000
    np.testing.assert_allclose(a.total_loss, b.total_loss, atol=5e-6, rtol=1e-5)
    assert not a.auxiliary_active.any()
    base.save_json(directory / 'controlled_comparison_check.json', dict(passed=True,
        training_cell_ids_identical=True, pre_auxiliary_updates=3000,
        maximum_pre_auxiliary_loss_difference=float(np.max(np.abs(a.total_loss - b.total_loss))),
        all_original_config_settings_unchanged=True))


def grid_records(spec, panel, pred, taus, method, seed, split):
    result = []
    for condition in sorted(pred.obs.condition.unique()):
        real = panel.X[panel.obs.condition.eq(condition).to_numpy()]
        generated = pred.X[pred.obs.condition.eq(condition).to_numpy()]
        fr = weak_responder_fraction(response_score(real, spec), taus)
        fp = weak_responder_fraction(response_score(generated, spec), taus)
        for q, tau, measured, predicted in zip(QS, taus, fr, fp):
            result.append(dict(method=method, seed=seed, split=split, condition=condition, q=q,
                tau=tau, wrf_real=measured, wrf_pred=predicted, err_pp=100 * abs(predicted - measured)))
    return result


def evaluate(run, baseline):
    from cellflow.model import CellFlow
    config = json.loads((run / 'config.json').read_text())
    protocol = json.loads((run / 'ablation_protocol.json').read_text())
    old_config = json.loads((baseline / 'config.json').read_text())
    assert {k: config[k] for k in old_config} == old_config
    outcomes = json.loads((run / 'training_outcomes.json').read_text())
    seeds = outcomes['completed_seeds']
    assert seeds, 'No completed seed can be evaluated.'
    choices = {}
    for seed in seeds:
        directory = run / 'trials' / TRIAL / f'seed_{seed}'
        assert json.loads((directory / 'training_complete.json').read_text())['steps'] == 30000
        assert json.loads((directory / 'controlled_comparison_check.json').read_text())['passed']
        choices[str(seed)] = dict(checkpoint=json.loads((directory / 'selected_checkpoint.json').read_text()),
                                 model_sha256=base.digest(directory / 'best_CellFlow.pkl'))
    assert json.loads((run / 'validation_integration_audit.json').read_text())['passed']
    frozen = dict(choices=choices, config_sha256=base.digest(run / 'config.json'),
                  protocol_sha256=base.digest(run / 'ablation_protocol.json'))
    freeze_path = run / 'test_evaluation_started.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen
    else:
        base.save_json(freeze_path, frozen)
    panel = ad.read_h5ad(base.PANEL)
    spec = EvalSpec.load(base.REPO / 'artifacts/eval_spec_v1')
    train_cells = base.model_data(panel[panel.obs['split'].isin(['train', 'control'])])
    for seed in seeds:
        directory = run / 'trials' / TRIAL / f'seed_{seed}'
        file = directory / 'preds_test.h5ad'
        if not file.exists():
            model = CellFlow.load(str(directory / 'best_CellFlow.pkl'))
            pred = base.predict(model, train_cells, config, 'test', seed)
            pred.write_h5ad(directory / 'preds_test.tmp.h5ad', compression='gzip')
            (directory / 'preds_test.tmp.h5ad').rename(file)
            del model, pred
        assert base.digest(directory / 'best_CellFlow.pkl') == frozen['choices'][str(seed)]['model_sha256']
    del train_cells
    comparison(run, baseline, panel, spec, config)


def comparison(run, baseline, panel, spec, config):
    outcomes = json.loads((run / 'training_outcomes.json').read_text())
    seeds = outcomes['completed_seeds']
    selected = json.loads((baseline / 'selected_methods.json').read_text())['methods']
    sources = {m: (baseline, trial) for m, trial in selected.items()}
    sources[METHOD] = (run, TRIAL)
    records, grids, verification, artifact_index = [], [], [], []
    taus = np.array(config['training_taus'])
    historical = pd.read_csv(baseline / 'all_per_condition_metrics.csv')
    for method, (root, trial) in sources.items():
        for seed in seeds:
            directory = root / 'trials' / trial / f'seed_{seed}'
            for split in ['val', 'test']:
                path = directory / f'preds_{split}.h5ad'
                pred = ad.read_h5ad(path)
                assert pred.n_vars == spec.n_genes
                assert (pred.obs.groupby('condition').size() == 512).all()
                if method == METHOD:
                    reference = ad.read_h5ad(baseline / 'trials' / TRIAL / f'seed_{seed}' / f'preds_{split}.h5ad')
                    assert list(pred.obs_names) == list(reference.obs_names)
                    assert list(pred.obs.source_control_id) == list(reference.obs.source_control_id)
                    del reference
                rec = evaluate_predictions(spec, panel, pred, split=split, seed=0)
                grid = grid_records(spec, panel, pred, taus, method, seed, split)
                grid_means = pd.DataFrame(grid).groupby('condition').err_pp.mean()
                for row in rec:
                    records.append(dict(method=method, seed=seed, split=split, **row,
                                        wrf_err_pp_grid13_mean=float(grid_means[row['condition']])))
                grids.extend(grid)
                if method != METHOD:
                    expected = historical[(historical.method == method) & (historical.seed == seed) & (historical.split == split)].set_index('condition')
                    actual = pd.DataFrame(rec).set_index('condition').loc[expected.index]
                    np.testing.assert_allclose(actual[METRICS[:-1]], expected[METRICS[:-1]], atol=1e-9, rtol=1e-9)
                    verification.append(dict(method=method, seed=seed, split=split, old_metrics_reproduced=True))
                else:
                    base.save_diagnostics(run, method, split, seed, panel, pred, spec)
                artifact_index.append(dict(method=method, seed=seed, split=split, path=str(path), sha256=base.digest(path)))
                print('SCORED', method, seed, split, flush=True)
                del pred
    # Reuse exactly the original sampled-reference populations for both arms.
    for split in ['val', 'test']:
        for method in ['unchanged_control', 'nearest_dose', 'adjacent_dose_mixture']:
            path = baseline / 'reference_predictions' / f'{method}_{split}.h5ad'
            pred = ad.read_h5ad(path)
            rec = evaluate_predictions(spec, panel, pred, split=split, seed=0)
            grid = grid_records(spec, panel, pred, taus, method, 0, split)
            grid_means = pd.DataFrame(grid).groupby('condition').err_pp.mean()
            records.extend(dict(method=method, seed=0, split=split, **row,
                wrf_err_pp_grid13_mean=float(grid_means[row['condition']])) for row in rec)
            grids.extend(grid)
        rec = base.replicate_reference_compatible(spec, panel, split, seed=0)
        for row in rec:
            condition = row['condition']
            fractions = []
            for rep in ['rep1', 'rep2']:
                mask = (panel.obs.condition.eq(condition) & panel.obs.replicate.eq(rep)).to_numpy()
                fractions.append(weak_responder_fraction(response_score(panel.X[mask], spec), taus))
            error = 100 * np.abs(fractions[0] - fractions[1])
            records.append(dict(method='replicate_reference', seed=0, split=split, **row,
                                wrf_err_pp_grid13_mean=float(error.mean())))
    frame = pd.DataFrame(records)
    assert np.isfinite(frame[METRICS].to_numpy()).all()
    out = run / 'report'
    out.mkdir(exist_ok=True)
    frame.to_csv(out / 'all_per_condition_metrics.csv', index=False)
    pd.DataFrame(grids).to_csv(out / 'threshold_grid_per_condition.csv', index=False)
    summaries = []
    for (method, split), part in frame.groupby(['method', 'split']):
        per_condition = part.groupby('condition')[METRICS].mean()
        summary = summarize(per_condition.reset_index().to_dict('records'))
        seed_means = part.groupby('seed')[METRICS].mean()
        for metric in METRICS:
            summaries.append(dict(method=method, split=split, metric=metric, mean=summary[metric]['mean'],
                seed_sd=float(seed_means[metric].std()) if len(seed_means) > 1 else float('nan'),
                ci95_low=summary[metric]['ci95'][0], ci95_high=summary[metric]['ci95'][1],
                n_seeds=len(seed_means), n_conditions=len(per_condition)))
    pd.DataFrame(summaries).to_csv(out / 'metric_summary.csv', index=False)
    paired = []
    rng = np.random.default_rng(42)
    test = frame[frame.split == 'test']
    for reference in ['cellflow_wrf', 'cellflow', 'cellflow_mmd']:
        a = test[test.method == reference].groupby('condition')[METRICS].mean()
        b = test[test.method == METHOD].groupby('condition')[METRICS].mean().loc[a.index]
        for metric in METRICS:
            delta = (b[metric] - a[metric]).to_numpy()
            boots = rng.choice(delta, (2000, len(delta)), replace=True).mean(1)
            paired.append(dict(reference=reference, metric=metric, difference=float(delta.mean()),
                ci95_low=float(np.quantile(boots, .025)), ci95_high=float(np.quantile(boots, .975)),
                conditions_improved=int((delta < 0).sum()), conditions=len(delta)))
    pd.DataFrame(paired).to_csv(out / 'paired_differences.csv', index=False)
    base.save_json(run / 'prediction_index.json', artifact_index)
    base.save_json(run / 'checks/comparison_verification.json', dict(passed=True, historical_checks=verification,
        old_prediction_files_read_only=True, new_source_control_ids_match_old=True,
        shared_metric_count=8, supplementary_metric='13-threshold absolute errors, averaged after per-seed errors',
        generated_cells_per_condition=512, exploratory=True))
    base.save_json(run / 'evaluation_complete.json', dict(status='complete', seeds=seeds,
        planned_seeds=config['seeds'], failures=outcomes['failures'],
        test_conditions=17, new_models=len(seeds), no_retuning=True, job_id=os.environ.get('SLURM_JOB_ID')))
    print('COMPARISON_EVALUATION_COMPLETE', str(run), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['initialize', 'preflight', 'train', 'evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    if args.stage == 'initialize':
        initialize(args.run, args.baseline)
    elif args.stage == 'preflight':
        preflight(args.run)
    elif args.stage == 'train':
        assert args.seed is not None
        train(args.run, args.baseline, args.seed)
    else:
        evaluate(args.run, args.baseline)
