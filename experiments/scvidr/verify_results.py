"""Verify the published scVIDR metrics using the Python standard library."""
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, stdev

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / 'docs/scvidr'
MODELS = ['scvidr']
METRICS = ['wrf_err_pp_q0.9', 'wrf_err_pp_q0.95', 'wrf_err_pp_q0.975', 'w1_r',
           'mmd2', 'rmse_mean_all', 'rmse_mean_deg50', 'rmse_mean_program', 'wrf_err_pp_grid13_mean']


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, label):
    assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (label, a, b)


def verify():
    data = rows(RESULTS / 'per_condition.csv')
    summaries = rows(RESULTS / 'metrics.csv')
    per_seed = rows(RESULTS / 'per_seed.csv')
    grid = rows(RESULTS / 'thresholds.csv')
    specification = read(REPO / 'configs/eval_grid13_v1.json')
    grid_errors = defaultdict(list)
    for r in grid:
        close(float(r['err_pp']), 100 * abs(float(r['wrf_pred']) - float(r['wrf_real'])), 'threshold error')
        grid_errors[r['method'], r['seed'], r['split'], r['condition']].append(r)
    split_manifest = read(REPO / 'configs/split_v1.json')
    methods = set(r['method'] for r in data)
    assert set(MODELS).issubset(methods)
    for split, n_conditions in [('val', 9), ('test', 17)]:
        expected = {r['condition'] for r in split_manifest['conditions'] if r['split'] == split}
        assert len(expected) == n_conditions
        for method in methods:
            seeds = [0, 1, 2] if method in MODELS else [0]
            part = [r for r in data if r['method'] == method and r['split'] == split]
            assert {int(r['seed']) for r in part} == set(seeds)
            assert len(part) == n_conditions * len(seeds)
            for seed in seeds:
                selected = [r for r in part if int(r['seed']) == seed]
                assert {r['condition'] for r in selected} == expected
                for r in selected:
                    assert all(math.isfinite(float(r[m])) for m in METRICS)
                    if method == 'replicate_reference':
                        continue
                    assert int(float(r['n_pred'])) == 512
                    for q in ['0.9', '0.95', '0.975']:
                        close(float(r['wrf_err_pp_q' + q]),
                              100 * abs(float(r['wrf_pred_q' + q]) - float(r['wrf_real_q' + q])), q)
                    group = grid_errors[method, str(seed), split, r['condition']]
                    assert len(group) == 13
                    assert sorted(float(x['q']) for x in group) == specification['quantiles']
                    close(float(r['wrf_err_pp_grid13_mean']), fmean(float(x['err_pp']) for x in group), 'grid mean')
            for metric in METRICS:
                means = [fmean(float(r[metric]) for r in part if int(r['seed']) == seed) for seed in seeds]
                summary = [r for r in summaries if (r['method'], r['split'], r['metric']) == (method, split, metric)]
                assert len(summary) == 1
                s = summary[0]
                close(float(s['mean']), fmean(means), (method, metric))
                assert int(s['n_seeds']) == len(seeds) and int(s['n_conditions']) == n_conditions
                if len(seeds) > 1:
                    close(float(s['seed_sd']), stdev(means), 'sample SD')
                else:
                    assert not s['seed_sd']
                for seed, mean in zip(seeds, means):
                    saved = [r for r in per_seed if (r['method'], int(r['seed']), r['split']) == (method, seed, split)]
                    assert len(saved) == 1
                    close(float(saved[0][metric]), mean, 'per-seed mean')
    cellflow = rows(REPO / 'docs/cellflow/metrics.csv')
    for r in summaries:
        if r['method'] in ['unchanged_control', 'nearest_dose', 'adjacent_dose_mixture', 'replicate_reference']:
            match = [x for x in cellflow if (x['method'], x['split'], x['metric']) == (r['method'], r['split'], r['metric'])]
            assert len(match) == 1
            close(float(r['mean']), float(match[0]['mean']), ('reference', r['method'], r['metric']))
    config = read(REPO / 'experiments/scvidr/config.json')
    assert config['seeds'] == [0, 1, 2] and config['max_epochs'] == 100 and not config['early_stopping']
    return dict(passed=True, metrics=len(METRICS), model_test_rows=51, conditions=17)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
