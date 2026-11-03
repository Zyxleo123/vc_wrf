"""Validate the completed experiment and inventory its reusable artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import anndata as ad
import numpy as np
import pandas as pd

METRICS = ['wrf_err_pp_q0.9', 'wrf_err_pp_q0.95', 'wrf_err_pp_q0.975',
           'w1_r', 'mmd2', 'rmse_mean_all', 'rmse_mean_deg50', 'rmse_mean_program']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(run):
    run = Path(run)
    root = Path(os.environ['RESPONSE_FRACTION_ROOT'])
    data = Path(os.environ['RESPONSE_FRACTION_DATA'])
    (run / 'checks').mkdir(exist_ok=True)
    read = lambda p: json.loads(Path(p).read_text())
    config = read(run / 'config.json')
    selection = read(run / 'selected_methods.json')
    methods = selection['methods']
    frozen = read(run / 'test_evaluation_started.json')
    assert read(run / 'complete.json')['status'] == 'complete'
    assert read(run / 'validation_integration_audit.json')['passed']
    assert frozen['config_sha256'] == sha256(run / 'config.json')
    assert set(methods) == {'cellflow', 'cellflow_wrf', 'cellflow_mmd'}
    assert config['seeds'] == [0, 1, 2]
    assert len(selection['candidates']) == 9
    for method in ['wrf', 'mmd']:
        best = min((v for v in selection['candidates'] if v['method'] == method and v.get('status', 'complete') == 'complete'),
                   key=lambda v: tuple(v['selection_scores']))
        assert methods['cellflow_' + method] == best['trial']

    panel = ad.read_h5ad(data / 'processed/a549_hdaci_v1.h5ad', backed='r')
    controls = panel.obs[panel.obs['split'].eq('control')]
    training_ids = set(panel.obs_names[panel.obs['split'].isin(['train', 'control'])])
    expected_conditions = {s: set(panel.obs.loc[panel.obs['split'].eq(s), 'condition'])
                           for s in ['train', 'val', 'test']}
    checks = []
    hashes = {}
    summary = pd.read_csv(run / 'metric_summary.csv')
    all_records = pd.read_csv(run / 'all_per_condition_metrics.csv')
    for method, trial in methods.items():
        for seed in config['seeds']:
            directory = run / 'trials' / trial / f'seed_{seed}'
            done = read(directory / 'training_complete.json')
            assert done['steps'] == 30000 and done['all_losses_finite']
            losses = pd.read_csv(directory / 'training_loss.csv')
            assert np.array_equal(losses['step'], np.arange(1, 30001))
            assert np.isfinite(losses.select_dtypes('number').to_numpy()).all()
            ids = pd.read_csv(directory / 'training_cell_ids.csv', index_col=0)
            assert len(ids) == 19398 and set(ids.index) == training_ids
            assert ids['split'].isin(['train', 'control']).all()
            info = read(directory / 'model_info.json')
            assert info['n_parameters'] == 5258277
            model = directory / 'best_CellFlow.pkl'
            model_hash = sha256(model)
            assert model_hash == frozen['model_sha256'][f'{method}/s{seed}']
            hashes[str(model.relative_to(run))] = model_hash
            checkpoint = read(directory / 'selected_checkpoint.json')
            assert checkpoint == frozen['selected_models'][f'{method}/s{seed}']
            effective_source = run / 'trials' / checkpoint.get('source_trial', trial) / f'seed_{seed}'
            for origin in {directory, effective_source}:
                source_hashes = read(origin / 'source_hashes.json')
                assert sha256(origin / 'training_source.py') == source_hashes['training']
                assert sha256(origin / 'population_loss_source.py') == source_hashes['population_loss']
                for name in ['training_source.py', 'population_loss_source.py', 'source_hashes.json',
                             'trial_config.json', 'selected_checkpoint.json', 'validation_history.json',
                             'training_loss.csv', 'training_cell_ids.csv', 'training_complete.json', 'model_info.json']:
                    hashes[str((origin / name).relative_to(run))] = sha256(origin / name)
            prediction_checks = {}
            for split in (['val', 'test', 'train'] if seed == 0 else ['val', 'test']):
                path = directory / f'preds_{split}.h5ad'
                pred = ad.read_h5ad(path)
                conditions = expected_conditions[split]
                assert pred.shape == (512 * len(conditions), 2085)
                assert pred.obs_names.is_unique
                assert list(pred.var_names) == list(panel.var_names)
                assert set(pred.obs['condition']) == conditions
                assert pred.obs.groupby('condition', observed=True).size().eq(512).all()
                assert np.isfinite(pred.X).all() and (pred.X >= 0).all()
                assert pred.obs['source_control_id'].isin(controls.index).all()
                source_reps = controls.loc[pred.obs['source_control_id'], 'replicate'].astype(str).to_numpy()
                assert np.array_equal(source_reps, pred.obs['replicate'].astype(str).to_numpy())
                records = pd.read_json(run / 'metrics' / method / f'{split}_s{seed}' / 'per_condition.jsonl', lines=True)
                assert len(records) == len(conditions) and set(records['condition']) == conditions
                assert np.isfinite(records[METRICS].to_numpy()).all()
                wells = pd.read_csv(run / 'diagnostics' / method / f'{split}_s{seed}_well_metrics.csv')
                assert len(wells) == 2 * len(conditions)
                assert not wells.duplicated(['condition', 'replicate']).any()
                assert np.isfinite(wells[METRICS].to_numpy()).all()
                response = pd.read_csv(run / 'diagnostics' / method / f'{split}_s{seed}_response_scores.csv.gz')
                assert response['population'].eq('generated').sum() == len(pred)
                assert response['population'].eq('real').sum() == panel.obs['split'].eq(split).sum()
                assert np.isfinite(response['response_score']).all()
                prediction_checks[split] = {'shape': list(pred.shape), 'conditions': len(conditions),
                                            'source_controls_verified': True, 'metrics_finite': True}
                hashes[str(path.relative_to(run))] = sha256(path)
                del pred
            checks.append({'method': method, 'trial': trial, 'seed': seed,
                           'effective_source_trial': str(effective_source.relative_to(run)),
                           'training_updates': len(losses), 'predictions': prediction_checks})
        for split in ['val', 'test']:
            rows = all_records[all_records['method'].eq(method) & all_records['split'].eq(split)]
            assert len(rows) == 3 * len(expected_conditions[split])
            averaged = rows.groupby('condition', observed=True)[METRICS].mean().mean()
            reported = summary[summary['method'].eq(method) & summary['split'].eq(split)].set_index('metric')
            assert set(reported.index) == set(METRICS)
            np.testing.assert_allclose(averaged.loc[METRICS], reported.loc[METRICS, 'mean'], rtol=1e-10, atol=1e-10)
    panel.file.close()

    source = run / 'final_source'
    source.mkdir(exist_ok=True)
    for name in ['run_cellflow_v1.py', 'cellflow_population_loss.py', 'cellflow_worker.py',
                 'run_cellflow_workers.sh', 'resume_cellflow_driver.sh',
                 'check_validation_integration.py', 'report_cellflow_v1.py',
                 'verify_cellflow_artifacts.py', 'package_cellflow_results.py', 'complete_cellflow_evaluation.sh', 'check_population_loss.py']:
        shutil.copyfile(root / 'src' / name, source / name)
    shutil.copyfile(root / 'jobs/cellflow_v1_full.sbatch', source / 'cellflow_v1_full.sbatch')
    shutil.copyfile(run / 'config.json', source / 'cellflow_v1_config.json')
    for folder in ['final_source', 'report', 'eval_spec_v1', 'checks']:
        for path in sorted((run / folder).rglob('*')):
            if path.is_file() and path.name not in ['artifact_validation.json']:
                hashes[str(path.relative_to(run))] = sha256(path)
    for name in ['config.json', 'requirements.lock.txt', 'panel_audit.json', 'split_v1.json',
                 'selected_methods.json', 'test_evaluation_started.json', 'validation_integration_audit.json',
                 'combined_summary.json', 'metric_summary.csv', 'all_per_condition_metrics.csv',
                 'baseline_early_checkpoint_audit.json']:
        hashes[name] = sha256(run / name)
    for name in ['numerical_repair_record.json', 'candidate_failure_policy.json']:
        if (run / name).exists():
            hashes[name] = sha256(run / name)
    (run / 'final_artifact_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    inventory = [{'path': str(p.relative_to(run)), 'bytes': p.stat().st_size}
                 for p in sorted(run.rglob('*')) if p.is_file()]
    pd.DataFrame(inventory).to_csv(run / 'artifact_inventory.csv', index=False)
    result = {'passed': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'selected_models': len(checks), 'metrics_per_condition': len(METRICS),
              'all_test_model_hashes_unchanged': True, 'training_cell_ids_verified': True,
              'summary_aggregation_verified': True, 'runs': checks,
              'inventory_files': len(inventory), 'hashed_files': len(hashes)}
    (run / 'checks/artifact_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ARTIFACT_VALIDATION_PASSED', len(checks), 'models', len(hashes), 'hashed files', flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    verify(parser.parse_args().run)
