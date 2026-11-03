# CellFlow results

A549 HDAC-inhibitor dose prediction, 17 held-out test conditions. Results average
three seeds; ± is sample SD. WRF errors are percentage points. Lower is better.

| Method | WRF error, q=.95 | Mean error, 13 thresholds | Response W₁ | MMD² | DEG50 RMSE | Program RMSE |
|---|---:|---:|---:|---:|---:|---:|
| CellFlow | 5.30 ± 1.02 | 11.36 | 0.419 | 0.0538 | 0.284 | 0.0815 |
| + WRF loss, 3 thresholds | 4.93 ± 0.99 | 11.43 | 0.387 | 0.0627 | 0.337 | 0.0822 |
| + WRF loss, 13 thresholds | 4.91 ± 0.55 | 11.38 | 0.387 | 0.0542 | 0.371 | 0.0849 |
| + MMD loss | 5.10 ± 0.82 | 12.42 | 0.411 | 0.0512 | 0.374 | 0.0888 |
| Vehicle cells | 4.79 | 14.98 | 0.471 | 0.0594 | 0.447 | 0.1012 |
| Nearest dose | 4.34 | 6.64 | 0.253 | 0.0275 | 0.269 | 0.0653 |
| Adjacent-dose mixture | 3.49 | 5.42 | 0.213 | 0.0161 | 0.198 | 0.0605 |
| Replicate-well disagreement | 3.81 | 7.98 | 0.273 | 0.0076 | 0.245 | 0.0860 |

[Code and configuration](../../experiments/cellflow/README.md) ·
[All metrics](metrics.csv) · [Per-seed results](per_seed.csv) ·
[Per-condition results](per_condition.csv) · [Models and predictions](artifacts.json)

WRF variants use λ=0.1, h=0.3 and float64 auxiliary calculations. Each condition
uses 512 generated cells. Errors are computed before averaging seeds. References
are fixed evaluations. These are exploratory results; chemCPA uses different
generation, selection and 13-threshold aggregation settings.
