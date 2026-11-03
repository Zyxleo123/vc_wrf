"""Score saved CellFlow populations with the shared evaluator and 13-threshold grid."""
import argparse
import json
from pathlib import Path

METRICS = ['wrf_err_pp_q0.9', 'wrf_err_pp_q0.95', 'wrf_err_pp_q0.975', 'w1_r',
           'mmd2', 'rmse_mean_all', 'rmse_mean_deg50', 'rmse_mean_program', 'wrf_err_pp_grid13_mean']


def evaluate(artifacts, out):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import run_cellflow_v1 as base
    from sciplex_eval import EvalSpec, evaluate_predictions, response_score, summarize
    from sciplex_eval.metrics import weak_responder_fraction

    index = json.loads(artifacts.read_text())
    assert base.digest(base.PANEL) == index['panel_sha256']
    spec = EvalSpec.load(base.REPO / 'artifacts/eval_spec_v1')
    panel = ad.read_h5ad(base.PANEL)
    grid = json.loads((base.REPO / 'configs/eval_grid13_v1.json').read_text())
    taus = np.asarray(grid['taus'])
    records, thresholds = [], []
    keys = [(p['method'], p['seed'], p['split']) for p in index['predictions']]
    assert len(keys) == len(set(keys)), 'Duplicate prediction entry'
    for entry in index['predictions']:
        path = Path(entry['path'])
        if 'sha256' in entry:
            assert base.digest(path) == entry['sha256'], path
        pred = ad.read_h5ad(path)
        assert list(pred.var_names) == spec.gene_ids
        assert (pred.obs.groupby('condition', observed=True).size() == 512).all()
        method, seed, split = entry['method'], entry['seed'], entry['split']
        scored = evaluate_predictions(spec, panel, pred, split=split, seed=0)
        for row in scored:
            condition = row['condition']
            real = panel.X[panel.obs.condition.eq(condition).to_numpy()]
            generated = pred.X[pred.obs.condition.eq(condition).to_numpy()]
            fr = weak_responder_fraction(response_score(real, spec), taus)
            fp = weak_responder_fraction(response_score(generated, spec), taus)
            errors = 100 * np.abs(fp - fr)
            records.append(dict(method=method, seed=seed, split=split, **row,
                                wrf_err_pp_grid13_mean=float(errors.mean())))
            for q, tau, measured, predicted, error in zip(grid['quantiles'], taus, fr, fp, errors):
                thresholds.append(dict(method=method, seed=seed, split=split, condition=condition,
                                       q=q, tau=tau, wrf_real=measured, wrf_pred=predicted, err_pp=error))
    for split in sorted({p['split'] for p in index['predictions']}):
        for row in base.replicate_reference_compatible(spec, panel, split, seed=0):
            fractions = []
            for rep in ['rep1', 'rep2']:
                mask = (panel.obs.condition.eq(row['condition']) & panel.obs.replicate.eq(rep)).to_numpy()
                fractions.append(weak_responder_fraction(response_score(panel.X[mask], spec), taus))
            errors = 100 * np.abs(fractions[0] - fractions[1])
            records.append(dict(method='replicate_reference', seed=0, split=split, **row,
                                wrf_err_pp_grid13_mean=float(errors.mean())))
    frame = pd.DataFrame(records)
    assert np.isfinite(frame[METRICS].to_numpy()).all()
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
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / 'per_condition.csv', index=False)
    pd.DataFrame(thresholds).to_csv(out / 'thresholds.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'metrics.csv', index=False)
    frame.groupby(['method', 'seed', 'split'])[METRICS].mean().reset_index().to_csv(out / 'per_seed.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.artifacts, args.out)


if __name__ == '__main__':
    main()
