CellFlow WRF: three versus thirteen thresholds after numerical repair

All six runs completed 30,000 updates: seeds 0, 1 and 2 for each threshold set.
These are new matched WRF reruns. The original v1 source and results are preserved.
Standard CellFlow and MMD references are rescored archived predictions, not new training.

The original 13-threshold attempt failed at updates 3550 (seed 1) and 3150 (seed 2).
A diagnostic replay of seed 1 captured float32 overflow during the 8-step auxiliary
Heun trajectory, before response scoring. Inputs, OT coupling and ordinary flow loss
were finite. Seed 2 did not reproduce its original failure within a 4500-update replay;
the exact first failing operation for that original seed is therefore not established.

Repair: use float64 for the differentiable auxiliary trajectory and soft response-score
calculation, then cast the bounded scalar penalty back to float32. The ordinary flow
calculation, model parameters and optimizer retain their original precision.
Weight=.1, h=.3, optimizer, learning rate, clipping, architecture, data, splits, seeds,
30,000 updates, 8 auxiliary steps and 64 prediction steps are unchanged. The original
validation checkpoint rule is retained. Both threshold arms use the same repaired code.
No skipped batches, replaced NaNs, added trajectory clipping or hyperparameter search.
The helper in source/ is the archived WRF experiment implementation; the MMD reference
uses its original implementation in experiments/cellflow_v1, not this repaired helper.

Verification: captured failing update and raw gradients are finite; original optimizer
norm agrees with an independent float64 calculation. All 180,000 training losses and
32,400 auxiliary parameter/optimizer checks are finite. Training cell IDs match v1.
The planned seed-0 validation check at 64 versus 128 solver steps passed for both arms.
All six selected checkpoints were frozen before new test scoring. Archived baseline
metrics were reproduced and output file checksums verified.

Results are in ../../docs/cellflow/q13_repair_20261030/report/.
The report includes all nine metrics, per-seed/per-condition records, paired condition
bootstrap intervals, figures, and vehicle/dose-interpolation/replicate references.
For 13 versus 3 thresholds, primary WRF error changes by -0.0260 percentage points
(condition-bootstrap 95% interval -0.4710 to +0.4150); this is not clear evidence of a gain.
W1 is essentially unchanged. MMD improves in the mean, while DEG50 and program RMSE worsen.

Limits: this is an exploratory repair after previous test results were seen. Higher
precision prevents overflow but does not establish accurate 8-step auxiliary trajectories;
the captured repaired trajectory still reached about 1.6e43. The q=.975 cutoff sits at
the zero-program-count boundary. These checks do not establish biological correctness.
Seed SD is not a confidence interval across independent biological experiments.
The existing chemCPA 13-threshold table uses a different seed-aggregation convention;
harmonize it before a direct final comparison. No GitHub push was made.

PSC archive (models, predictions, source, logs and checks):
/ocean/projects/cis260266p/shared/response-fraction/results/cellflow_q13_precision_repair_20261030
Access is set for ywang179 and amotwani. Original failed attempts remain preserved.

Reproduction: use the recorded Linux/CUDA environment in requirements.lock.txt and
pinned evaluator 1d8b173f8d4120c8ba4487d7e03511b26e7a71ad with the frozen panel/split.
The source scripts are exact archived PSC scripts, with historical deployment paths.
Adapt those paths and use fresh output directories for a new run. The stage order in
source/production_driver.sh is initialize -> six training runs -> validation integration
checks -> freeze/evaluate -> report -> share. Set RESPONSE_FRACTION_ROOT and
RESPONSE_FRACTION_DATA as in the v1 reproduction guide. Do not run the PSC launcher on
macOS. Diagnostic scripts require their original captured state retained on PSC.

precision_fix.patch records the exact numerical code change. The JSON manifest below
fingerprints this local review package. Large models and predictions remain on PSC.
