"""Report the completed controlled comparison without changing any model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_threshold_comparison import METRICS, METHOD, TRIAL
from run_cellflow_v1 import digest, save_json

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
a = p.parse_args()
run = a.run
assert json.loads((run / 'evaluation_complete.json').read_text())['status'] == 'complete'
protocol = json.loads((run / 'ablation_protocol.json').read_text())
outcomes = json.loads((run / 'training_outcomes.json').read_text())
seeds = outcomes['completed_seeds']
freeze = json.loads((run / 'test_evaluation_started.json').read_text())
assert digest(run / 'config.json') == freeze['config_sha256']
assert digest(run / 'ablation_protocol.json') == freeze['protocol_sha256']
for seed, choice in freeze['choices'].items():
    assert digest(run / 'trials' / TRIAL / f'seed_{seed}' / 'best_CellFlow.pkl') == choice['model_sha256']
out = run / 'report'
metrics = pd.read_csv(out / 'metric_summary.csv')
records = pd.read_csv(out / 'all_per_condition_metrics.csv')
grid = pd.read_csv(out / 'threshold_grid_per_condition.csv')
paired = pd.read_csv(out / 'paired_differences.csv')
methods = ['cellflow', 'cellflow_wrf', METHOD, 'cellflow_mmd']
names = {'cellflow': 'Standard CellFlow', 'cellflow_wrf': 'WRF: 3 thresholds',
         METHOD: 'WRF: 13 thresholds', 'cellflow_mmd': 'CellFlow + MMD',
         'unchanged_control': 'Vehicle', 'replicate_reference': 'Replicate disagreement',
         'nearest_dose': 'Nearest dose', 'adjacent_dose_mixture': 'Adjacent-dose mixture'}
colors = ['#687386', '#2673b8', '#cf6c2c', '#2b917a']
test = metrics[metrics.split == 'test'].set_index(['method', 'metric'])
lines = ['CellFlow: three versus thirteen supervised thresholds',
         'Exploratory extension of the existing v1 benchmark.',
         f"TRAINING OUTCOME: {len(seeds)}/{len(protocol['seeds'])} seeds completed; failures: {json.dumps(outcomes['failures'])}",
         f'METRIC TABLE: matched completed seed(s) {seeds} only, for every model.',
         'Failed seeds are not replaced or omitted from the training-outcome report.', '',
         'Only the training cutoff set changes. Weight=0.1; smoothing h=0.3; 30,000 updates;',
         'same three planned seeds, initialization recipe, architecture, optimizer, source controls and split.',
         'The loss averages over cutoffs, so adding cutoffs does not multiply the loss weight.',
         'Checkpoints are selected at 5k/10k/20k/30k by validation q0.95 error, then W1.',
         'Final generation: 512 cells/condition, replicate-matched controls, 64 Heun steps.',
         'All three original high-quantile cutoffs are preserved exactly.', '',
         'TEST: 17 held-out-dose conditions, equal-weight means. Seed SD is unavailable with one completed seed.',
         'WRF units are percentage points; lower is better for every metric.',
         'Each seed is scored separately before averaging; the 13-threshold mean uses this same rule.', '']
for metric in METRICS:
    lines.append(metric)
    for method in methods:
        r = test.loc[method, metric]
        sd = f'{r.seed_sd:.6f}' if np.isfinite(r.seed_sd) else 'NA'
        lines.append(f"  {names[method]:24} {r['mean']:.6f}; seed SD {sd}; condition CI [{r.ci95_low:.6f}, {r.ci95_high:.6f}]")
lines += ['', 'PAIRED DIFFERENCES: 13 thresholds minus 3 thresholds; negative means improvement.']
for _, r in paired[paired.reference == 'cellflow_wrf'].iterrows():
    lines.append(f"  {r.metric}: {r.difference:+.6f} pp".replace(' pp', ' pp' if r.metric.startswith('wrf') else '')
                 + f"; 95% CI [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]; {r.conditions_improved}/{r.conditions} conditions improve")
lines += ['', 'REFERENCE PRIMARY WRF ERRORS']
for method in ['unchanged_control', 'nearest_dose', 'adjacent_dose_mixture', 'replicate_reference']:
    lines.append(f"  {names[method]}: {test.loc[(method, 'wrf_err_pp_q0.95'), 'mean']:.6f} pp")
training = []
for seed in seeds:
    directory = run / 'trials' / TRIAL / f'seed_{seed}'
    selected = json.loads((directory / 'selected_checkpoint.json').read_text())
    complete = json.loads((directory / 'training_complete.json').read_text())
    check = json.loads((directory / 'controlled_comparison_check.json').read_text())
    assert check['passed']
    training.append({**check, 'seed': seed, 'selected_step': selected['step'], 'seconds': complete['elapsed_seconds']})
pd.DataFrame(training).to_csv(out / 'training_summary.csv', index=False)
lines += ['', 'SELECTED 13-THRESHOLD CHECKPOINTS', json.dumps(training, indent=2), '',
          'INTERPRETATION LIMITS',
          'The extension was requested after prior CellFlow and chemCPA test results were reviewed.',
          'No new hyperparameter search was performed, and the test did not select checkpoints.',
          'This is exploratory; confirmation requires an independently fixed evaluation.',
          'If any seed failed, the metric table is conditional on training success and must not',
          'be presented as a successful three-seed improvement. The original three-threshold',
          'setting completed all three seeds. Stability is part of the experimental outcome.',
          'Seed SD includes training and source-control sampling variation. Condition-bootstrap',
          'intervals do not represent new independent biological experiments.',
          'The pre-auxiliary training histories were not numerically identical to the archived runs.',
          'The original strict trajectory gate failed; its logs are retained. A separate audit',
          'verified identical training functions, cell IDs, settings, environment and first-step losses.',
          'The only population-loss code change is the threshold list. The cause of subsequent',
          'numerical divergence was not isolated. This is matched settings, not a bitwise replay.',
          'No model or checkpoint was changed when that verification gate was revised.',
          'The v1 q0.95 endpoint is weakly sensitive, and q0.975 lies at the zero-program boundary.',
          'Replicate disagreement is a reference, not an absolute lower bound.',
          'Existing baseline and three-threshold runs were reused without modification.',
          'Their original eight metrics were independently reproduced from saved predictions.',
          'The chemCPA screenshot used a different seed-aggregation rule for its grid column;',
          'do not compare that grid number directly with the consistently scored numbers here.', '',
          'SOURCE BASELINE', protocol['baseline_run'], 'THIS RUN', str(run)]
(out / 'comparison_report.txt').write_text('\n'.join(lines) + '\n')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
selected_metrics = ['wrf_err_pp_q0.95', 'wrf_err_pp_grid13_mean', 'w1_r', 'mmd2']
titles = ['Primary WRF error (pp)', '13-threshold mean error (pp)', 'Response-score Wasserstein-1', 'Multigene MMD²']
for ax, metric, title in zip(axes.flat, selected_metrics, titles):
    rows = test.loc[[(m, metric) for m in methods]]
    ax.barh(range(4), rows['mean'], color=colors)
    errors = rows.seed_sd.to_numpy() if len(seeds) > 1 else np.maximum(0, np.vstack([rows['mean'] - rows.ci95_low, rows.ci95_high - rows['mean']]))
    ax.errorbar(rows['mean'], range(4), xerr=errors, color='#333333', fmt='none', capsize=3)
    ax.set(yticks=range(4), yticklabels=[names[m] for m in methods], title=title, xlabel='Lower is better')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
error_label = 'seed SD' if len(seeds) > 1 else 'condition-bootstrap 95% CI'
fig.suptitle(f'CellFlow: 3 versus 13 supervised thresholds\nMatched seed(s) {seeds}; {len(outcomes["failures"])} failed runs; bars = {error_label}')
fig.savefig(out / 'comparison.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
for method, color in zip(methods, colors):
    curve = grid[(grid.split == 'test') & (grid.method == method)].groupby('q').err_pp.mean()
    ax.plot(curve.index, curve, marker='o', markersize=4, color=color, label=names[method])
ax.axvline(.95, color='#888888', linewidth=1, linestyle='--')
ax.set(xlabel='Control response-score quantile q', ylabel='WRF absolute error (percentage points)',
       title='Threshold sensitivity: per-seed errors, then equal averaging')
ax.legend(frameon=False)
ax.grid(alpha=.2)
fig.savefig(out / 'threshold_sensitivity.png', dpi=180)
plt.close(fig)

# Verify all reported means directly from the independent per-condition table.
for _, row in metrics.iterrows():
    part = records[(records.method == row.method) & (records.split == row.split)]
    expected = part.groupby('condition')[row.metric].mean().mean()
    np.testing.assert_allclose(expected, row['mean'], atol=1e-10, rtol=1e-10)
final = dict(status='completed_with_training_failures' if outcomes['failures'] else 'complete',
             completed_models_verified=len(seeds), planned_seeds=protocol['seeds'], completed_seeds=seeds,
             failures=outcomes['failures'], source_controls_matched=True,
             historical_metrics_reproduced=True, summary_aggregation_verified=True,
             numerical_trajectories_identical=False,
             no_github_push=True, selected_steps=[v['selected_step'] for v in training])
save_json(run / 'complete.json', final)
save_json(run / 'artifact_sha256.json', {str(p.relative_to(run)): digest(p)
    for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'artifact_sha256.json'})
print(json.dumps(final), flush=True)
