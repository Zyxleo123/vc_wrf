"""Validate the published result subset without GPU or scientific dependencies."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f'{label}: {actual} != {expected}')


def verify(repo):
    docs = repo / 'docs/cellflow/v1'
    manifest = json.loads((docs / 'publication_manifest.json').read_text())
    for entry in manifest['files']:
        path = repo / entry['path']
        assert path.is_file(), entry['path']
        assert sha256(path) == entry['sha256'], entry['path']
    frozen = json.loads((docs / 'final_artifact_sha256.json').read_text())
    source_records = json.loads((docs / 'published_source_provenance.json').read_text())
    initial = json.loads((docs / 'source_manifest.json').read_text())
    for entry in source_records:
        expected = (frozen if entry['verification'] == 'final_artifact_sha256.json' else initial)[entry['original_run_path']]
        assert entry['sha256'] == expected
        assert sha256(repo / entry['published_path']) == expected, entry['published_path']
    for name in ['arrays.npz', 'spec.json', 'pilot.json']:
        assert sha256(repo / 'artifacts/eval_spec_v1' / name) == frozen['eval_spec_v1/' + name]
    assert sha256(repo / 'configs/split_v1.json') == frozen['split_v1.json']

    spec_paths = manifest['shared_evaluator_git_blobs']
    for relative, expected in spec_paths.items():
        content = (repo / relative).read_bytes()
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        assert blob == expected, relative

    split = json.loads((repo / 'configs/split_v1.json').read_text())
    conditions = {c['condition']: c for c in split['conditions'] if c['split'] == 'test'}
    assert len(conditions) == 17
    models = ['cellflow', 'cellflow_wrf', 'cellflow_mmd']
    metrics = ['wrf_err_pp_q0.9', 'wrf_err_pp_q0.95', 'wrf_err_pp_q0.975',
               'w1_r', 'mmd2', 'rmse_mean_all', 'rmse_mean_deg50', 'rmse_mean_program']
    with (docs / 'report/all_per_condition_metrics.csv').open() as f:
        records = [r for r in csv.DictReader(f) if r['split'] == 'test' and r['method'] in models]
    assert len(records) == 3 * 3 * 17
    for method in models:
        for seed in range(3):
            part = [r for r in records if r['method'] == method and int(r['seed']) == seed]
            assert len(part) == 17
            assert {r['condition'] for r in part} == set(conditions)
            for row in part:
                assert row['evaluation_commit'] == manifest['evaluation_commit']
                assert int(float(row['n_real'])) == conditions[row['condition']]['n_cells']
                assert int(float(row['n_pred'])) == 512
                assert all(math.isfinite(float(row[k])) for k in metrics)
                for q in ['0.9', '0.95', '0.975']:
                    expected = 100 * abs(float(row['wrf_pred_q' + q]) - float(row['wrf_real_q' + q]))
                    close(float(row['wrf_err_pp_q' + q]), expected, (method, seed, row['condition'], q))
    with (docs / 'report/all_metric_summaries.csv').open() as f:
        summaries = {(r['method'], r['metric']): r for r in csv.DictReader(f)
                     if r['split'] == 'test' and r['method'] in models}
    assert len(summaries) == 24
    for method in models:
        for metric in metrics:
            means = [fmean(float(r[metric]) for r in records if r['method'] == method and int(r['seed']) == seed)
                     for seed in range(3)]
            summary = summaries[method, metric]
            close(fmean(means), float(summary['mean']), (method, metric, 'mean'))
            close(stdev(means), float(summary['training_seed_sd']), (method, metric, 'seed SD'))
            assert int(summary['n_conditions']) == 17 and int(summary['n_seeds']) == 3
    return {'passed': True, 'publication_files_checked': len(manifest['files']),
            'source_files_matched_to_archived_hashes': len(source_records),
            'identical_shared_evaluator_files': len(spec_paths),
            'selected_test_rows': len(records), 'metric_means_and_seed_sds_checked': 24,
            'scope': 'Saved metrics and provenance; no training or prediction re-scoring.'}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
