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
NAMES = {'scvidr': 'scVIDR', 'cellflow': 'Standard CellFlow', 'unchanged_control': 'Vehicle cells', 'nearest_dose': 'Nearest dose', 'adjacent_dose_mixture': 'Adjacent-dose mixture', 'replicate_reference': 'Replicate-well disagreement'}


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
        if entry['split'] != 'test' or entry['method'] not in ['scvidr', 'cellflow']:
            continue
        pred = ad.read_h5ad(entry['path'])
        for condition in sorted(pred.obs['condition'].unique()):
            real = panel.X[panel.obs['condition'].eq(condition).to_numpy()].toarray()
            generated = np.asarray(pred.X[pred.obs['condition'].eq(condition).to_numpy()])
            rows.append(dict(method=entry['method'], seed=entry['seed'], condition=condition, gene_variance_ratio=float(generated.var(0).sum() / real.var(0).sum()), response_sd_ratio=float(response_score(generated, spec).std() / response_score(real, spec).std()), zero_fraction_real=float((real == 0).mean()), zero_fraction_pred=float((generated == 0).mean())))
    frame = pd.DataFrame(rows)
    frame.to_csv(DOCS / 'spread.csv', index=False)
    print(frame.groupby('method')[['gene_variance_ratio', 'response_sd_ratio', 'zero_fraction_real', 'zero_fraction_pred']].median().round(3).to_string())


def report(root):
    DOCS.mkdir(parents=True, exist_ok=True)
    for name in ['metrics.csv', 'per_seed.csv', 'per_condition.csv', 'thresholds.csv']:
        shutil.copyfile(root / 'metrics' / name, DOCS / name)
    selected = [json.loads((root / f'seed_{s}/selected_checkpoint.json').read_text()) for s in [0, 1, 2]]
    complete = [json.loads((root / f'seed_{s}/training_complete.json').read_text()) for s in [0, 1, 2]]
    validation = {s: json.loads((root / f'seed_{s}/validation_history.json').read_text()) for s in [0, 1, 2]}
    config = json.loads((root / 'config.json').read_text())
    legacy = json.loads((root / 'panel_legacy.json').read_text())
    models = [dict(seed=c['seed'], selected_epoch=c['epoch'], epochs_trained=c['epochs_trained'], path=str((root / f"seed_{c['seed']}/model/model_params.pt").resolve()), sha256=c['model_sha256'], elapsed_seconds=t['elapsed_seconds'], gpu=t['gpu']) for c, t in zip(selected, complete)]
    (root / 'settings.json').write_text(json.dumps(dict(config=config, results_root=str(root.resolve()), selected_models=models, validation_history=validation, panel_legacy_copy=legacy, versions=complete[0]['versions']), indent=2) + '\n')
    ours = pd.read_csv(DOCS / 'metrics.csv')
    theirs = pd.read_csv(REPO / 'docs/cellflow/metrics.csv')
    table = pd.concat([ours, theirs[theirs.method.eq('cellflow')]])
    table = table[table.split.eq('test') & table.method.isin(NAMES)]
    wide = table.pivot(index='method', columns='metric', values='mean')[METRICS]
    wide['primary_seed_sd'] = table[table.metric.eq('wrf_err_pp_q0.95')].set_index('method')['seed_sd']
    wide['n_seeds'] = table[table.metric.eq('wrf_err_pp_q0.95')].set_index('method')['n_seeds']
    wide = wide.loc[[m for m in NAMES if m in wide.index]]
    wide.insert(0, 'name', [NAMES[m] for m in wide.index])
    wide.reset_index().to_csv(DOCS / 'comparison.csv', index=False)
    a = pd.read_csv(DOCS / 'per_condition.csv')
    b = pd.read_csv(REPO / 'docs/cellflow/per_condition.csv')
    a = a[a.method.eq('scvidr') & a.split.eq('test')].groupby('condition')[METRICS].mean()
    b = b[b.method.eq('cellflow') & b.split.eq('test')].groupby('condition')[METRICS].mean()
    assert list(a.index) == list(b.index) and len(a) == 17
    rng = np.random.default_rng(0)
    rows = []
    for metric in METRICS:
        d = (a[metric] - b[metric]).to_numpy()
        boots = rng.choice(d, (2000, len(d)), replace=True).mean(axis=1)
        rows.append(dict(reference='cellflow', metric=metric, difference=float(d.mean()), ci95_low=float(np.quantile(boots, 0.025)), ci95_high=float(np.quantile(boots, 0.975)), conditions_improved=int((d < 0).sum()), conditions=len(d)))
    pd.DataFrame(rows).to_csv(DOCS / 'paired_differences.csv', index=False)
    print(wide.round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    args = parser.parse_args()
    report(args.root)
    spread(args.root, args.panel)


if __name__ == '__main__':
    main()
