# chemCPA on sci-Plex3: results record

Snapshots of the comparison reports from the 2026-10-30 runs, copied from the
gitignored `results/chemcpa/`. Each one is redrawn from saved predictions by
`slurm/chemcpa/report.sbatch results/chemcpa/submissions/<submission>.env`.
See the main [README](../../README.md#chemcpa-baseline) for the method and conclusions,
and [`wrf_sweep_log.md`](wrf_sweep_log.md) for the response-fraction-loss protocol,
which was fixed before any sweep result, and every configuration tried.

| folder | split | compares | submission |
|---|---|---|---|
| [`likelihoods_test`](likelihoods_test/comparison.md) | test | chemCPA-NB vs chemCPA-Gauss | `20261030-165655` |
| [`wrf_round1_val`](wrf_round1_val/comparison.md) | val | round 1: 6 loss settings (spec τs) and chemCPA-NB | `20261030-172753` |
| [`wrf_round1_test`](wrf_round1_test/comparison.md) | test | round-1 pick, chemCPA-NB, chemCPA-Gauss | `20261030-173429` |
| [`wrf_round2_val`](wrf_round2_val/comparison.md) | val | round 2: 6 loss settings (13 τs), chemCPA-NB, round-1 pick | `20261030-175952` |
| [`wrf_round2_test`](wrf_round2_test/comparison.md) | test | round-2 pick, chemCPA-NB, round-1 pick | `20261030-180637` |

Each folder holds `comparison.md` (the frozen v1 metrics, plus WRF error at 13 control-quantile
thresholds), `comparison.csv`, and `threshold_sensitivity.png`. Rows are means over 3 seeds;
vehicle and replicate rows are the shared references. Figures plot at most three models.
