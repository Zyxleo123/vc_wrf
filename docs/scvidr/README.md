# scVIDR results

A549 HDAC-inhibitor dose prediction, 17 held-out test conditions. Results average
three seeds; ± is sample SD. WRF errors are percentage points. Lower is better.

| Method | WRF error, q=.95 | Mean error, 13 thresholds | Response W₁ | MMD² | DEG50 RMSE | Program RMSE |
|---|---:|---:|---:|---:|---:|---:|
| scVIDR | 6.99 ± 0.13 | 22.13 | 0.769 | 0.0924 | 0.292 | 0.0682 |
| CellFlow | 5.30 ± 1.02 | 11.36 | 0.419 | 0.0538 | 0.284 | 0.0815 |
| Vehicle cells | 4.79 | 14.98 | 0.471 | 0.0594 | 0.447 | 0.1012 |
| Nearest dose | 4.34 | 6.64 | 0.253 | 0.0275 | 0.269 | 0.0653 |
| Adjacent-dose mixture | 3.49 | 5.42 | 0.213 | 0.0161 | 0.198 | 0.0605 |
| Replicate-well disagreement | 3.81 | 7.98 | 0.273 | 0.0076 | 0.245 | 0.0860 |

[Code and configuration](../../experiments/scvidr/README.md) ·
[All metrics](metrics.csv) · [Per-seed results](per_seed.csv) ·
[Per-condition results](per_condition.csv) · [Comparison](comparison.csv) ·
[Paired differences with CellFlow](paired_differences.csv) · [Spread diagnostics](spread.csv)

scVIDR trains for 100 epochs per seed. The checkpoint is chosen on validation WRF
q=.95, then response W₁, among epochs 10, 25, 50, 75 and 100; the selected epochs are
100, 75 and 100 for seeds 0, 1 and 2. Each condition uses 512 generated cells. The
CellFlow row is copied from [CellFlow results](../cellflow/README.md). The reference
rows were rescored in this run and match those results exactly.

Against CellFlow on the 17 test conditions (scVIDR minus CellFlow, bootstrap 95% CI):
WRF q=.95 +1.69 (−1.10 to 4.59, 6 of 17 conditions better), 13-threshold mean +10.77
(4.23 to 17.01), response W₁ +0.350 (0.144 to 0.533), MMD² +0.039 (0.004 to 0.084),
DEG50 RMSE +0.008 (−0.040 to 0.059) and program RMSE −0.013 (−0.025 to −0.002, 13 of 17
better).

scVIDR decodes the latent mean of each control cell, so the generated populations are
narrow: median total gene variance is 0.225 of the measured condition (CellFlow 0.924)
and median response score SD is 0.599 (CellFlow 0.725). Its WRF error peaks at 37.6 pp
at q=.8, where CellFlow's is 18.1. Its mean expression is close to CellFlow's. It is
behind nearest dose and the adjacent-dose mixture on every metric, since its dose rule
scales the 10 µM shift by log1p(dose) and does not use the observed 10 nM or other
intermediate dose.

Validation WRF q=.95 at the selected checkpoints is 4.94, 4.50 and 4.87, against 7.03,
6.84 and 7.10 on test. From epoch 25 on, validation error moves by up to 5.3 pp between
checkpoints of one seed, so the selection is noisy. At q=.975 both scVIDR and CellFlow
predict a fraction of 1.0 for every condition and have the same error, 7.41. These are
exploratory results.

To verify the saved metrics locally without a GPU or dataset:

```bash
python experiments/scvidr/verify_results.py
```
