# CellFlow and chemCPA: comparison status

Reviewed against chemCPA branch `chemcpa-wrf-loss` at
`186875e3d4834f3e311b1c7a3ce6b3f39716d8f3` on October 30, 2026.

## What is matched

The split manifest, all frozen evaluation-spec files (including the numeric arrays),
and every file in `sciplex_eval/` have identical Git blob hashes between the CellFlow
evaluation commit and the reviewed chemCPA commit. Both experiments target the same
A549 HDAC-inhibitor panel and the same 17 held-out drug-dose test conditions.

Shared metrics: WRF absolute errors at q=0.90, 0.95 and 0.975; response-score W1;
PCA-space multiscale RBF MMD²; population-mean RMSE for all genes, top-50 DEGs and
the response-program genes. The response score, normalization, gene order, PCA,
thresholds, DEG sets and equal-condition weighting are defined by the same spec.
Both main result tables average seed-specific errors over three seeds.

CellFlow's panel SHA256 is
`8d07db08aa0323b35b65f2e2742492ac1c71d291e27f6bdce733cacb701cf8ff`.
The chemCPA adapter validates the frozen split and records panel/spec hashes in its
run metadata, but those run-level hashes were not available in the public comparison
table reviewed here. Byte-for-byte equality of the two runtime H5AD panel files has
not been independently checked; compare those run metadata before final consolidation.

## Differences that must accompany a combined table

| Setting | CellFlow v1 | Reviewed chemCPA runs |
|---|---|---|
| Generated cells per condition | 512 | At least 1,024 |
| Source controls | Sampled within replicate; target replicate proportions from frozen metadata | Sampled from the pooled control population |
| Checkpoint selection | Validation primary WRF, then response W1 | Validation MMD² |
| Loss-setting selection | Validation search on seed 0; selected settings repeated with seeds 1 and 2 | Validation summaries over three seeds |
| Drug conditioning | One-hot drug identity | Fixed Morgan fingerprints with a learned encoder |
| Generated expression | Continuous log-expression, negative entries clipped to zero | NB count samples, normalized by generated totals |
| Vehicle reference | 512 sampled, replicate-matched controls; 4.7928 pp primary error | Full pooled controls; 4.8904 pp primary error |
| 13-threshold mean | Now reported in the precision-rerun comparison; per-seed absolute errors averaged afterward | Reported after averaging fractions across seeds, unlike the primary metric |

The eight metrics can be displayed side by side as results of the stated pipelines.
They are not a controlled comparison of architecture alone, and small numerical
differences do not establish superiority. These generation and selection choices
must remain visible; do not label the existing runs as a fully matched benchmark.

The [completed CellFlow precision reruns](q13_repair_20261030/README.md) add the
13-threshold column without changing the frozen eight metrics. Both WRF arms were
rerun using float64 auxiliary trajectories and scoring, while the ordinary flow
loss, parameters and optimizer remain float32. The original v1 and the incomplete
seed-0-only comparison remain separate records. The chemCPA review above remains
anchored to its stated commit; no teammate code or results were modified here.

## Next evaluation, without retraining

1. Confirm panel/spec hashes from both archived runs.
2. Use one evaluator to score each saved prediction set independently by seed,
   then average metrics equally across conditions and seeds. For the 13-threshold
   grid, take absolute errors **before** averaging seeds. Use paired condition
   uncertainty for between-method comparisons.
3. Include the same vehicle and replicate reference for every model. State the
   source-control and generated-sample policies. A common subsample of saved
   predictions can diagnose sample-size sensitivity, but does not remove the
   original control-source or checkpoint-selection differences.
4. Treat a harmonized checkpoint-selection rule as a future protocol, fixed using
   validation data before a fresh confirmatory test. Do not retune to the current
   test results.

The chemCPA 13-threshold run also changes smoothing h from 0.5 to 0.25, so it is not
an isolated threshold-count ablation. It was designed after the first test round,
as disclosed in the teammate's sweep log. Retain that exploratory label.

The v1 primary endpoint is weakly sensitive to drug effects and q=0.975 falls on a
zero-program-count boundary. Keep v1 unchanged; validate any revised score or
threshold specification separately before freezing a new evaluation.

Sources: [chemCPA comparison](https://github.com/Zyxleo123/vc_wrf/blob/186875e3d4834f3e311b1c7a3ce6b3f39716d8f3/docs/chemcpa/wrf_round2_test/comparison.md),
[report aggregation](https://github.com/Zyxleo123/vc_wrf/blob/186875e3d4834f3e311b1c7a3ce6b3f39716d8f3/sciplex_baselines/report_chemcpa.py),
[sweep protocol](https://github.com/Zyxleo123/vc_wrf/blob/186875e3d4834f3e311b1c7a3ce6b3f39716d8f3/docs/chemcpa/wrf_sweep_log.md).
