# CellFlow condition-response embedding length ablation

Date: 2026-11-06. The condition-response embedding is the `teacher` covariate. This ablation changes
only the number of response coordinates supplied to the selected CellFlow arm
`z70_lr3e4_mv10_t`; all other model and training settings are held fixed. Length 20 is the existing
arm, while lengths 10, 40, and 78 are new runs with three training seeds each. Length 78 is the full
basis supported by the 78 training-condition means; no validation or test condition is used to form
the basis.

Training and validation jobs: `101704`–`101706`. Frozen test evaluation of the new lengths: `101842`.
Each test value uses 17 held-out-dose conditions, 512 generated cells per condition, sampling offset
0, and the checkpoint with minimum validation MMD² for that training seed. Values are mean ± sample
SD over the three training seeds. Lower is better for every metric. Here “RMSD” means the frozen
evaluator's mean-expression RMSE.

## Validation and test

| Embedding length | Validation MMD² | Validation RMSE all | Validation RMSE DEG50 | Test MMD² | Test RMSE all | Test RMSE DEG50 | Test RMSE program |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | **0.0063** | **0.0437** | **0.166** | 0.0161 ± 0.0007 | 0.0539 ± 0.0007 | 0.196 ± 0.003 | 0.077 ± 0.003 |
| **20** | 0.0070 | 0.0445 | 0.174 | **0.0115 ± 0.0008** | **0.0504 ± 0.0002** | **0.184 ± 0.003** | 0.077 ± 0.002 |
| 40 | 0.0075 | 0.0450 | 0.179 | 0.0150 ± 0.0016 | 0.0527 ± 0.0006 | 0.194 ± 0.005 | **0.076 ± 0.002** |
| 78 | 0.0075 | 0.0450 | 0.178 | 0.0149 ± 0.0018 | 0.0528 ± 0.0009 | 0.200 ± 0.007 | 0.079 ± 0.005 |

Length 10 was best on validation but did not transfer to test. Length 20 has the best test MMD²,
all-gene RMSE, and DEG50 RMSE. Length 40 has a slightly lower program RMSE.

Paired condition-bootstrap differences below are length 20 minus the other length; negative means
length 20 is better. The intervals use seed-averaged per-condition values and 10,000 bootstrap
samples.

| Comparison | MMD² | RMSE all | RMSE DEG50 | RMSE program |
|---|---:|---:|---:|---:|
| 20 − 10 | −0.0046 [−0.0118, +0.0002] | −0.0035 [−0.0088, +0.0003] | −0.011 [−0.042, +0.008] | −0.000 [−0.001, +0.001] |
| 20 − 40 | −0.0034 [−0.0097, +0.0008] | −0.0023 [−0.0067, +0.0011] | −0.010 [−0.036, +0.006] | +0.001 [−0.000, +0.003] |
| 20 − 78 | −0.0034 [−0.0088, +0.0004] | −0.0024 [−0.0067, +0.0007] | −0.015 [−0.042, +0.002] | **−0.002 [−0.003, −0.001]** |

The core-metric intervals between lengths include zero. Thus, 20 has the strongest aggregate test
result, but the available 17 test conditions do not establish a decisive difference in MMD²,
all-gene RMSE, or DEG50 RMSE between embedding lengths. Length 20 is significantly better than 78
on program RMSE.

## Comparison with other methods

| Method | Test MMD² | RMSE all | RMSE DEG50 | RMSE program |
|---|---:|---:|---:|---:|
| **Condition-response embedding, length 20** | **0.0115** | **0.0504** | **0.184** | 0.077 |
| Adjacent-dose mixture | 0.0161 | 0.0577 | 0.198 | **0.060** |
| Matched latent backbone (`z70_k8_res_lr3e4`) | 0.0169 | 0.0543 | 0.211 | 0.080 |
| Grouped velocity matching (`meanv_l10`) | 0.0175 | 0.0574 | 0.214 | 0.083 |
| chemCPA-Gauss + mean consistency | 0.0223 | 0.0612 | 0.238 | **0.059** |
| CellFlow retrain (`base`) | 0.0240 | 0.0635 | 0.224 | 0.085 |
| Original chemCPA-Gauss | 0.0334 | 0.0713 | 0.308 | 0.072 |
| Published CellFlow | 0.0538 | 0.0917 | 0.284 | 0.082 |

Paired test differences for length 20 minus selected comparisons:

| Comparison | MMD² | RMSE all | RMSE DEG50 | RMSE program |
|---|---:|---:|---:|---:|
| 20 − matched latent backbone | **−0.0054 [−0.0111, −0.0004]** | −0.0039 [−0.0085, +0.0007] | −0.027 [−0.060, +0.003] | **−0.003 [−0.005, −0.001]** |
| 20 − CellFlow retrain | **−0.0125 [−0.0234, −0.0052]** | **−0.0131 [−0.0210, −0.0071]** | **−0.040 [−0.058, −0.023]** | **−0.008 [−0.012, −0.004]** |
| 20 − chemCPA-Gauss + mean consistency | **−0.0108 [−0.0178, −0.0051]** | **−0.0108 [−0.0164, −0.0059]** | **−0.054 [−0.091, −0.024]** | **+0.018 [+0.004, +0.032]** |
| 20 − adjacent-dose mixture | −0.0045 [−0.0130, +0.0025] | −0.0073 [−0.0170, +0.0014] | −0.013 [−0.039, +0.016] | **+0.017 [+0.001, +0.033]** |

Length 20 is significantly better than the matched latent backbone on MMD² and program RMSE, and
better than the CellFlow retrain on every metric. It is significantly better than chemCPA-Gauss +
mean consistency on MMD² and both general expression errors, but worse on the proliferation-program
subset. Against the adjacent-dose mixture, the MMD² and expression-error point estimates favor
length 20 but their paired intervals include zero; the mixture is better on program RMSE.

## Decision

Retain embedding length **20**. Shortening to 10 improves validation but not test, and widening to 40
or 78 does not improve test MMD² or expression RMSE. Because every width has now been inspected on
test, these test results are an exploratory ablation and should not be used for another tuning round.

