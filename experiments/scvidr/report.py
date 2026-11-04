"""Copy scored scVIDR tables into docs/scvidr and compare them with CellFlow."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / 'docs/scvidr'
METRICS = ['wrf_err_pp_q0.9', 'wrf_err_pp_q0.95', 'wrf_err_pp_q0.975', 'wrf_err_pp_grid13_mean', 'w1_r', 'mmd2', 'rmse_mean_all', 'rmse_mean_deg50', 'rmse_mean_program']
MODELS = ['scvidr', 'scvidr_wrf', 'scvidr_wrf13', 'scvidr_mmd']
NAMES = {'scvidr': 'Standard scVIDR', 'scvidr_wrf': 'scVIDR + WRF, 3 thresholds', 'scvidr_wrf13': 'scVIDR + WRF, 13 thresholds', 'scvidr_mmd': 'scVIDR + MMD', 'cellflow': 'Standard CellFlow', 'unchanged_control': 'Vehicle cells', 'nearest_dose': 'Nearest dose', 'adjacent_dose_mixture': 'Adjacent-dose mixture', 'replicate_reference': 'Replicate disagreement'}


def spread(root, panel_path):
    import anndata as ad
    import sys
    sys.path.insert(0, str(REPO))
    from sciplex_eval import EvalSpec, response_score
    spec = EvalSpec.load(REPO / 'artifacts/eval_spec_v1')
    panel = ad.read_h5ad(panel_path)
    entries = json.loads((root / 'artifacts.json').read_text())['predictions'] + json.loads((REPO / 'docs/cellflow/artifacts.json').read_text())['predictions']
    rows = []
    for entry in entries:
        if entry['split'] != 'test' or entry['method'] not in MODELS + ['cellflow']:
            continue
        pred = ad.read_h5ad(entry['path'])
        for condition in sorted(pred.obs['condition'].unique()):
            real = panel.X[panel.obs['condition'].eq(condition).to_numpy()].toarray()
            generated = np.asarray(pred.X[pred.obs['condition'].eq(condition).to_numpy()])
            rows.append(dict(method=entry['method'], seed=entry['seed'], condition=condition, gene_variance_ratio=float(generated.var(0).sum() / real.var(0).sum()), response_sd_ratio=float(response_score(generated, spec).std() / response_score(real, spec).std()), zero_fraction_real=float((real == 0).mean()), zero_fraction_pred=float((generated == 0).mean())))
    frame = pd.DataFrame(rows)
    frame.to_csv(DOCS / 'spread.csv', index=False)
    print(frame.groupby('method')[['gene_variance_ratio', 'response_sd_ratio', 'zero_fraction_real', 'zero_fraction_pred']].median().round(3).to_string())


def settings(root, runs):
    models, validation, selection = [], {}, {}
    artifacts = json.loads((root / 'artifacts.json').read_text())
    for run in runs:
        methods = json.loads((run / 'selected_methods.json').read_text()) if (run / 'selected_methods.json').exists() else dict(scvidr='scvidr')
        if (run / 'hyperparameter_selection.json').exists():
            selection[run.name] = json.loads((run / 'hyperparameter_selection.json').read_text())
        for method, trial in methods.items():
            for seed in [0, 1, 2]:
                directory = run / f'seed_{seed}' if trial == 'scvidr' else run / 'trials' / trial / f'seed_{seed}'
                chosen = json.loads((directory / 'selected_checkpoint.json').read_text())
                complete = json.loads((directory / 'training_complete.json').read_text())
                models.append(dict(method=method, trial=trial, seed=seed, selected_epoch=chosen['epoch'], epochs_trained=chosen['epochs_trained'], path=str((directory / 'model/model_params.pt').resolve()), sha256=chosen['model_sha256'], elapsed_seconds=complete['elapsed_seconds'], gpu=complete['gpu']))
                validation[f'{method}/{seed}'] = json.loads((directory / 'validation_history.json').read_text())
    configs = {run.name: json.loads((run / 'config.json').read_text()) for run in runs}
    (root / 'settings.json').write_text(json.dumps(dict(runs=[str(r.resolve()) for r in runs], configs=configs, selected_models=models, hyperparameter_selection=selection, validation_history=validation, panel_sha256=artifacts['panel_sha256']), indent=2) + '\n')
    return models


def paired(a, b, model, reference, rng):
    rows = []
    for metric in METRICS:
        d = (a[metric] - b[metric]).to_numpy()
        boots = rng.choice(d, (2000, len(d)), replace=True).mean(axis=1)
        rows.append(dict(model=model, reference=reference, metric=metric, difference=float(d.mean()), ci95_low=float(np.quantile(boots, 0.025)), ci95_high=float(np.quantile(boots, 0.975)), conditions_improved=int((d < 0).sum()), conditions=len(d)))
    return rows


def report(root, runs):
    DOCS.mkdir(parents=True, exist_ok=True)
    for name in ['metrics.csv', 'per_seed.csv', 'per_condition.csv', 'thresholds.csv']:
        shutil.copyfile(root / 'metrics' / name, DOCS / name)
    models = settings(root, runs)
    ours = pd.read_csv(DOCS / 'metrics.csv')
    theirs = pd.read_csv(REPO / 'docs/cellflow/metrics.csv')
    table = pd.concat([ours, theirs[theirs.method.eq('cellflow')]])
    table = table[table.split.eq('test') & table.method.isin(NAMES)]
    wide = table.pivot(index='method', columns='metric', values='mean')[METRICS]
    wide['primary_seed_sd'] = table[table.metric.eq('wrf_err_pp_q0.95')].set_index('method')['seed_sd']
    wide['n_seeds'] = table[table.metric.eq('wrf_err_pp_q0.95')].set_index('method')['n_seeds']
    wide = wide.loc[[m for m in NAMES if m in wide.index]]
    wide.index = [NAMES[m] for m in wide.index]
    wide.index.name = 'method'
    wide.reset_index().to_csv(DOCS / 'comparison.csv', index=False)
    a = pd.read_csv(DOCS / 'per_condition.csv')
    b = pd.read_csv(REPO / 'docs/cellflow/per_condition.csv')
    test = {m: a[a.method.eq(m) & a.split.eq('test')].groupby('condition')[METRICS].mean() for m in MODELS}
    test['cellflow'] = b[b.method.eq('cellflow') & b.split.eq('test')].groupby('condition')[METRICS].mean()
    assert all(list(v.index) == list(test['scvidr'].index) and len(v) == 17 for v in test.values())
    rng = np.random.default_rng(0)
    rows = []
    for reference in ['scvidr_wrf', 'scvidr', 'scvidr_mmd']:
        rows += paired(test['scvidr_wrf13'], test[reference], 'scvidr_wrf13', reference, rng)
    rows += paired(test['scvidr'], test['cellflow'], 'scvidr', 'cellflow', rng)
    pd.DataFrame(rows).to_csv(DOCS / 'paired_differences.csv', index=False)
    print(wide.round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(models)[['method', 'trial', 'seed', 'selected_epoch']].to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.runs)
    spread(args.root, args.panel)


if __name__ == '__main__':
    main()
