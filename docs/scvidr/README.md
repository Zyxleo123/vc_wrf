# scVIDR results

A549 HDAC-inhibitor dose prediction, 17 held-out test conditions. Results average
three seeds; ± is sample SD. WRF errors are percentage points. Lower is better.

| Method | WRF error, q=.95 | Mean error, 13 thresholds | Response W₁ | MMD² | DEG50 RMSE | Program RMSE |
|---|---:|---:|---:|---:|---:|---:|
| scVIDR | 6.99 ± 0.13 | 22.13 | 0.769 | 0.0924 | 0.292 | 0.0682 |
| + WRF loss, 3 thresholds | 5.67 ± 1.11 | 16.26 | 0.571 | 0.0794 | 0.323 | 0.0955 |
| + WRF loss, 13 thresholds | 5.94 ± 0.40 | 10.85 | 0.442 | 0.0938 | 0.368 | 0.1436 |
| + MMD loss | 12.47 ± 2.82 | 19.61 | 0.738 | 0.0673 | 0.401 | 0.0907 |
| CellFlow | 5.30 ± 1.02 | 11.36 | 0.419 | 0.0538 | 0.284 | 0.0815 |
| Vehicle cells | 4.79 | 14.98 | 0.471 | 0.0594 | 0.447 | 0.1012 |
| Nearest dose | 4.34 | 6.64 | 0.253 | 0.0275 | 0.269 | 0.0653 |
| Adjacent-dose mixture | 3.49 | 5.42 | 0.213 | 0.0161 | 0.198 | 0.0605 |
| Replicate-well disagreement | 3.81 | 7.98 | 0.273 | 0.0076 | 0.245 | 0.0860 |

[Code and configuration](../../experiments/scvidr/README.md) ·
[All metrics](metrics.csv) · [Per-seed results](per_seed.csv) ·
[Per-condition results](per_condition.csv) · [Comparison](comparison.csv) ·
[Paired differences](paired_differences.csv) · [Spread diagnostics](spread.csv)

All scVIDR models train for 100 epochs. The WRF variant uses λ=5 and h=0.1 and the MMD
variant λ=50, chosen on seed 0 validation from λ∈{5, 15, 50, 150} × h∈{0.1, 0.3} and
λ∈{50, 500}; the 13-threshold run reuses λ=5 and h=0.1. These are CellFlow's grids
scaled by 486, the ratio of the two base losses. Each condition uses 512 generated
cells. The CellFlow row is copied from [CellFlow results](../cellflow/README.md), and the
reference rows were rescored in this run and match those results exactly.

Paired over the 17 test conditions, 13-threshold WRF minus standard scVIDR (bootstrap
95% CI): 13-threshold mean −11.28 (−14.59 to −7.30, 16 of 17 conditions better),
response W₁ −0.327 (−0.415 to −0.222, 16 of 17), WRF q=.95 −1.06 (−3.20 to 1.06, 11 of
17), DEG50 RMSE +0.076 (0.055 to 0.096) and program RMSE +0.075 (0.052 to 0.100). Against
the 3-threshold WRF model it lowers the 13-threshold mean by 5.41 (3.41 to 7.43) and W₁
by 0.129 (0.077 to 0.180). Standard scVIDR minus CellFlow: WRF q=.95 +1.69 (−1.26 to
4.62), 13-threshold mean +10.77 (3.28 to 17.06), W₁ +0.350 (0.149 to 0.527).

The WRF loss moves scVIDR's response distribution toward the measured one but raises its
mean-expression error, as for CellFlow. With 13 thresholds, scVIDR reaches a lower
13-threshold mean than CellFlow (10.85 against 11.36) and a similar W₁ (0.442 against
0.419). The MMD loss lowers MMD² from 0.0924 to 0.0673 but raises the primary WRF error
from 6.99 to 12.47. scVIDR decodes latent means, so all variants generate narrow
populations: median total gene variance is 0.225 to 0.264 of the measured condition,
against 0.924 for CellFlow. Every scVIDR model is behind nearest dose and the
adjacent-dose mixture on every metric.

Seed-0 validation WRF q=.95 across the grid does not change monotonically with λ (4.05
at λ=5, 7.67 at λ=50 and 4.78 at λ=150, all with h=0.1), so the selection is noisy. At
q=.975 every scVIDR and CellFlow model predicts a fraction of 1.0 for every condition
and has the same error, 7.41. These are exploratory results.

To verify the saved metrics locally without a GPU or dataset:

```bash
python experiments/scvidr/verify_results.py
```
