# Held-out-dose prediction from adjacent doses: study log

Scope: population metrics only (MMD², mean-expression RMSEs, W1(r), plus supplementary energy
distance and Pearson Δ). WRF is out of scope here.

Names (changed 2026-11-06, numbers untouched): the *condition response embedding* is the `teacher` covariate and *grouped velocity matching* is
`aux.kind: mean_velocity` (formerly "average-arrow" or "average-velocity"). Configs and variant names keep the old identifiers.

## Protocol (fixed 2026-11-05, before any statistics model or population generator was scored)

- Development uses **inner tasks** (each drug's training intermediate dose, predicted from its
  10 nM and 10 µM conditions; training cells only) and **validation** (9 conditions).
- Test (17 conditions) is scored **once**, for the methods chosen on validation, and compared
  with the published adjacent-dose mixture by a paired condition bootstrap.
- Mean predictors ("can a simple model infer a condition's mean from other conditions?") are
  ranked on validation DEG50 RMSE of the predicted mean; all-gene and program RMSE and Pearson Δ
  are reported alongside. Noise floors are reported with them.
- Population generators are ranked on validation MMD² (frozen evaluator), and a generator only
  counts as an improvement if its validation DEG50 RMSE is also no worse than the published
  adjacent mixture's.

## Diagnostics (job 97605, submission 20261105-004321; inner + val only)

- Shifting the published log-dose mixture onto the measured target mean (oracle) cuts val MMD²
  0.0201 → 0.0048 and energy distance 0.51 → 0.06 (inner 0.0337 → 0.0074): **the mixture's shape is
  nearly right; its error is its mean.** OT displacement interpolation and shifted single doses are
  no better than the mixture (val MMD² 0.019–0.028).
- The published upper-dose share (≈ 0.49) is too high: the best fixed share is ≈ 0.2–0.3 (val MMD²
  0.0139 at 0.2). A per-condition oracle share reaches only 0.0126 MMD² and 0.191 DEG50 RMSE (vs
  0.2095 for log-dose interpolation of the means), so the held-out mean lies off the segment between
  its neighbours; re-weighting alone cannot reach the oracle-mean level.
- Noise floor of the measured mean (val): DEG50 RMSE 0.069 (0.082 with a 512-cell prediction),
  all-gene 0.025, program 0.034. Interpolating the means reaches DEG50 0.21: about 3× the floor.
- Train-condition mean shifts are nearly rank 2 (PC1 82%, PC2 9%). Along PC1 the dose responses
  are sigmoidal with drug-specific potency and a common saturation level.
- chemCPA does not fit its own **training** conditions: in-sample MMD² 0.043 (Gaussian) / 0.066 (NB),
  DEG50 0.35 / 0.45, worse than the real-cell mixture on unseen val doses (0.020 / 0.22). A log-dose
  mixture of chemCPA's own adjacent-dose outputs scores the same as its direct prediction, so its
  gap is fit, not interpolation.

## Mean predictors (stats; jobs 97636 shards + 97637 merge, submission 20261105-005934; inner + val)

Ranked on val DEG50 RMSE of the predicted mean (9 conditions); inner (26 training-internal tasks) in brackets.

| predictor | val DEG50 | val all-gene | val program | val Pearson Δ DEG50 | [inner DEG50] |
|---|---|---|---|---|---|
| autodec_e4_wd0.0001 (MLP auto-decoder) | **0.153** | 0.038 | 0.039 | 0.70 | [0.184] |
| traj_l1.5_r0.1_hill (shared trajectory, Hill progress) | 0.160 | 0.042 | 0.054 | 0.68 | [0.232] |
| lowrank_r4 | 0.180 | 0.040 | 0.042 | 0.61 | [0.246] |
| linear_w0.4 / linear_w0.3 (best fixed weight) | 0.203 | 0.049–0.051 | 0.052–0.054 | 0.64 | [0.260–0.264] |
| hill (progress-weighted interpolation) | 0.204 | 0.050 | 0.060 | 0.64 | [0.235] |
| linear (log-dose; the published mixture's mean) | 0.210 | 0.055 | 0.052 | 0.63 | [0.244] |
| metanet | 0.230 | 0.058 | 0.068 | 0.61 | [0.208] |
| vehicle | 0.332 | 0.062 | 0.088 | 0.02 | [0.426] |
| noise floor (measured mean; with a 512-cell prediction) | 0.069 (0.082) | 0.025 | 0.034 | | [0.074] |

Chosen mean predictor: autodec_e4_wd0.0001 (lowest val DEG50). The population stage also carries the
best of the other families and one ensemble (autodec_e4_wd0.0001 + traj_l1.5_r0.1_hill), added after
these validation numbers were seen.

## chemCPA: dose encoding and population-mean consistency (training job 97662, submission 20261105-010422)

New, opt-in config keys (defaults reproduce every existing run; unit-tested):
- `dose_transform: "log10_nM"`: doser input log10(dose)/4 (10/100/1000/10000 nM → 0.25/0.5/0.75/1) instead of
  log1p(dose/1000) (→ 0.010/0.095/0.69/2.40, which nearly merges 10 and 100 nM).
- `mean_loss`: weight × mean squared difference between the counterfactual population's mean (permitted
  controls pushed through a training condition, exactly as at prediction time) and that training
  condition's measured mean (NB: log1p(1e4 × mean count proportion); Gaussian: mean log1p(CP10K)), over
  8 sampled training conditions × 64 control cells per autoencoder step. Only training conditions and
  vehicle are read.

Validation at the selected checkpoint (min val MMD², the baselines' rule), 3 seeds:

| variant | val MMD² | val DEG50 RMSE |
|---|---|---|
| gauss_v1 (original) | 0.0180 | 0.259 |
| gausslog_v1 | 0.0177 | 0.250 |
| gaussmean_l10 | 0.0134 | 0.239 |
| **gaussmean_l100** | **0.0129** | **0.201** |
| gausslogmean_l10 / l100 | 0.0169 / 0.0169 | 0.235 / 0.216 |
| nb_v1 (original) | 0.0272 | 0.322 |
| nblog_v1 | 0.0240 | 0.300 |
| **nbmean_l10** | **0.0200** | **0.259** |
| nbmean_l100 | 0.0393 | 0.326 |
| nblogmean_l10 / l100 | 0.0221 / 0.0375 | 0.259 / 0.319 |
| CellFlow / CellFlow + MMD (frozen rows) | 0.0432 / 0.0284 | 0.265 / 0.292 |

**Test decision (2026-11-05, before any test number of these variants was computed):** score test once for
`gaussmean_l100` and `nbmean_l10` (lowest val MMD² within each likelihood; both also lower val DEG50).
Pre-declared comparisons, paired condition bootstrap on the frozen evaluator's per-condition rows
(MMD² primary; DEG50, all-gene and program RMSE, W1(r) secondary): each against the original chemCPA of
its likelihood (primary), against CellFlow and CellFlow + MMD loss, and against the adjacent-dose mixture
and vehicle references (context). Test uses the original pipeline's generation (1024 cells per
condition, sampling offsets 0 and 1); offset 0 is the comparison, offset 1 a robustness check.

### Test result (scored once; jobs 97716 evaluate, 97727 compare; tables in docs/interp/chemcpa_test.md)

17 conditions, 3 seeds, frozen evaluator; Δ = ref − new (> 0: new better), condition-bootstrap 95% CI.

| test | MMD² | RMSE DEG50 | RMSE all | RMSE program | W1(r) |
|---|---|---|---|---|---|
| chemCPA-Gauss + mean consistency (λ=100) | 0.0223 | 0.238 | 0.061 | 0.059 | 0.574 |
| chemCPA-Gauss (original) | 0.0334 | 0.308 | 0.071 | 0.072 | 0.601 |
| chemCPA-NB + mean consistency (λ=10) | 0.0300 | 0.290 | 0.068 | 0.068 | 0.385 |
| chemCPA-NB (original) | 0.0494 | 0.393 | 0.080 | 0.102 | 0.589 |
| CellFlow | 0.0538 | 0.284 | 0.092 | 0.082 | 0.419 |
| adjacent-dose mixture (real neighbour cells) | 0.0161 | 0.198 | 0.058 | 0.060 | 0.213 |

- Gauss variant vs original chemCPA-Gauss: DEG50 +0.069 [+0.013, +0.151] (14/17), program +0.013 [+0.004, +0.022];
  MMD² +0.0111 [−0.0037, +0.0312] (10/17, not conclusive); all-gene +0.010 [−0.000, +0.023].
- Gauss variant vs CellFlow: MMD² +0.0315 [+0.0163, +0.0486] (16/17), DEG50 +0.046 [+0.018, +0.072] (16/17),
  all-gene +0.031, program +0.023 (all CIs > 0); W1(r) worse, −0.155 [−0.239, −0.065] (dense Gaussian samples).
- NB variant vs original chemCPA-NB: DEG50 +0.103 [+0.016, +0.207], program +0.034, W1(r) +0.204 [+0.041, +0.378];
  MMD² +0.0195 [−0.0039, +0.0471] (not conclusive). vs CellFlow: MMD² +0.0238 [+0.0077, +0.0404], DEG50 tie.
- Neither variant beats the adjacent-dose mixture on test (Gauss: MMD² −0.0063 [−0.0168, +0.0049], DEG50
  −0.041 [−0.080, −0.005]); the gap narrowed from 0.0334 → 0.0223 MMD² and 0.308 → 0.238 DEG50.
- A second sampling seed reproduces the test numbers (0.0222 / 0.237 Gauss; 0.0299 / 0.288 NB).
- Caveat: CellFlow's checkpoints were chosen by a WRF-based rule, chemCPA's by validation MMD², which favours
  chemCPA on MMD². Original chemCPA-Gauss already beat CellFlow on test MMD² (0.033 vs 0.054), not on DEG50.
- Mechanism: in-sample fit on the 78 training conditions improves 3.7× in MMD² (Gauss 0.043 → 0.012, NB
  0.066 → 0.018). The term fits each training condition separately; it does not link a drug's doses.

## CellFlow (protocol fixed 2026-11-05, before any CellFlow run on this cluster)

Starting point: the published CellFlow (cellflow-tools 0.0.9; `experiments/cellflow/` at 2c478b8:
one-hot drug, log1p(dose)/log1p(10⁴), all 2,085 genes without PCA, 30,000 updates, EMA 0.999). From
its saved summaries (`docs/cellflow/v1/combined_summary.json` at bde8c6c), CellFlow shows chemCPA's
failure: generated from controls, its own **training** conditions score MMD² 0.065 and DEG50 RMSE 0.305
(seed 0), worse than the real-cell mixture on unseen doses. Its base checkpoints were chosen by a WRF
rule at steps 1,000–2,000 (the end of the learning-rate warm-up), and per-seed test MMD² ranges
0.035–0.086. The longer-trained published variants (10k–30k steps) converge to test DEG50 ≈ 0.37.
(cellflow-tools' `ema=0.999` keeps 99.9% of the *new* parameters each step, i.e. almost no averaging.)

- Re-run here with the published configuration, 3 seeds, validated at 500, 1k, 2k, 3k, 5k, 10k, 20k and
  30k steps on validation (9 conditions) and in-sample fit (78 training conditions, generated from
  controls). Generation as published: 512 cells per condition from permitted controls, replicate-matched,
  64 Heun steps, negatives clipped.
- **Checkpoint rule for every CellFlow run here: minimum validation MMD²** (the chemCPA rule). The re-run
  base under this rule is the matched baseline; the published CellFlow rows are reported alongside.
- A variant counts as an improvement if its validation MMD² is below the matched base's and its
  validation DEG50 RMSE is no worse. Test (17 conditions) is scored once, for the chosen variant(s) and
  the matched base, with paired condition bootstraps against the matched base (primary), published
  CellFlow, chemCPA-Gauss + mean consistency and the adjacent-dose mixture.

### What goes wrong in CellFlow (jobs 97773, 97792; submissions 20261105-022815, -023431)

- **The published long-trained variants ignore the condition.** Over the 17 test conditions, the SD of the
  predicted mean response score is 0.010 / 0.018 / 0.010 for `cellflow_mmd` and `cellflow_wrf13` (seeds 0/1/2)
  and 0.019 / 0.010 for `cellflow_wrf` seeds 1/2, with correlation to the measured values −0.14 to +0.27.
  Vehicle cells give 0.011 (sampling noise); the measured SD is 0.414. Base CellFlow is 0.03–0.07. They
  generate about one population for every drug and dose. This also bears on the WRF comparison: those WRF
  rows come from condition-blind models.
- The same state appears in the re-runs: from about step 5,000, metrics freeze at val MMD² ≈ 0.033 and in-sample
  MMD² ≈ 0.070, DEG50 ≈ 0.44, and never recover in the remaining 25,000 updates (`meanv_l10` seed 1,
  `ema1e3` seed 0, `mean_l10` seed 1, `mean_l10_ema` seed 0).
- **Identical models score very differently.** `base`, `mean_l1` and `mean_l10` share every update up to step 3,000
  (their terms are off; first active step checked in training_loss.csv), so they differ only by floating-point
  noise. At step 2,000 their in-sample MMD² is 0.068 / 0.033 / 0.019. Early checkpoint scores are close to a lottery.
- **Generation can diverge.** The published 64-step Heun rollout gave non-finite cells at step 3,000 (`mean_l1` seed 2,
  before its term started) and at step 30,000 (`base` seed 0, condition MC1568 10 µM, a training condition);
  `base` seed 2 at step 20,000 has finite but exploded cells (in-sample DEG50 RMSE 1.8e28).
- **Variance can collapse.** `base` seed 1 fits means best (in-sample DEG50 0.149 at 30k) while cell-to-cell variance
  falls to 0.32 of real, so val MMD² rises to 0.079.
- `ema=0.999` in cellflow-tools keeps 99.9% of the *new* parameters each step (no real averaging). A true slow
  average (`ema=0.001`) does not prevent the collapse above.
- Data: 96.2% of control-cell entries are exact zeros, and only 27% of cell variance lies in the spec PCA(50), so the
  flow-matching loss is dominated by per-cell noise. About 47% of generated values are negative before clipping.
- The published config differs from CellFlow's default optimizer, `MultiSteps(adam(5e-5), 20)`: it uses AdamW at 1e-3
  with no accumulation, and each batch holds one condition. Within a batch the condition is then a constant.
- The aux term in float32 overflowed in the first aux updates near peak learning rate (`aux inf`, steps 3,010–3,060),
  the published `nonfinite_training_loss`. The rollout state is now bounded and non-finite updates drop the term.

### Validation rounds (jobs 97773, 97792, 97836, 97871, 97874; test not touched)

Decision (2026-11-05): WRF-chosen checkpoints are not a goal here, because the aim is a good distribution in general. CellFlow's
comparison row is `base` at its lowest validation MMD²; the published 0.054 stays only as a reference labelled WRF-chosen. Both
chemCPA rows were checked: in all six gauss_v1 / gaussmean_l100 runs the chosen epoch is the validation MMD² minimum.

Validation, 9 conditions, mean ± SD over 3 seeds, each seed at its lowest validation MMD² of 8 checkpoints. "End" columns are the
30,000-step model: validation MMD², in-sample variance ratio (median over the 78 training conditions), and runs whose generated
effect size no longer follows the condition (SD of generated / measured effect size across training conditions below 0.05).

| variant | MMD² | DEG50 | program | W1(r) | selected steps | end MMD² | end variance | end ignores condition |
|---|---|---|---|---|---|---|---|---|
| grouped velocity matching λ=10 (`meanv_l10`) | 0.0111 ± 0.0039 | 0.201 | 0.076 | 0.409 | 30k/2k/30k | 0.0172 | 0.98 | 1/3 |
| latent 70, K=8, residual, peak lr 3e-4 (`z70_k8_res_lr3e4`) | 0.0123 ± 0.0007 | 0.207 | 0.077 | 0.424 | 2k/3k/2k | 0.0371 | 0.81 | 0/3 |
| latent 50, K=8, residual (`z50_k8_res`) | 0.0124 ± 0.0014 | 0.218 | 0.074 | 0.400 | 500/500/500 | 0.0242 | 0.84 | 0/3 |
| latent 70, K=8, inverse (`z70_k8_inv`) | 0.0124 ± 0.0008 | 0.203 | 0.049 | 0.816 | 500/1k/500 | 0.0242 | 0.14 | 0/3 |
| latent 70, K=8, residual (`z70_k8_res`) | 0.0130 ± 0.0003 | 0.212 | 0.074 | 0.407 | 500/1k/500 | 0.0274 | 0.82 | 0/3 |
| latent 70, K=8, residual, grouped velocity matching λ=10 (`z70_k8_res_mv10`) | 0.0131 ± 0.0012 | 0.202 | 0.076 | 0.421 | 3k/3k/2k | 0.0350 | 0.81 | 0/3 |
| latent 50, inverse decode (`z50_inv`) | 0.0155 ± 0.0020 | 0.198 | 0.048 | 0.847 | 3k/3k/20k | 0.0251 | 0.13 | 0/3 |
| true EMA 0.01 (`ema1e2`) | 0.0160 ± 0.0031 | 0.224 | 0.078 | 0.435 | 10k/2k/5k | 0.0552 | 0.58 | 0/3 |
| 8 × 128 + grouped velocity matching λ=10 (`k8n128_mv10`) | 0.0164 ± 0.0047 | 0.226 | 0.081 | 0.432 | 2k/1k/500 | 0.0863 | 0.34 | 0/3 |
| latent 50, residual decode (`z50_res`) | 0.0164 ± 0.0024 | 0.214 | 0.077 | 0.428 | 3k/3k/5k | 0.0273 | 0.84 | 0/3 |
| grouped velocity matching λ=100 (`meanv_l100`) | 0.0166 ± 0.0010 | 0.235 | 0.087 | 0.444 | 3k/30k/10k | 0.0178 | 1.00 | 0/3 |
| published recipe (`base`) | 0.0193 ± 0.0030 | 0.229 | 0.076 | 0.432 | 5k/2k/5k | 0.0525 | 0.66 | 1/3 |
| true EMA 0.001 (`ema1e3`) | 0.0195 ± 0.0064 | 0.265 | 0.081 | 0.434 | 1k/2k/5k | 0.0581 | 0.60 | 1/3 |
| 8 conditions × 128 cells (`k8n128`) | 0.0205 ± 0.0047 | 0.234 | 0.075 | 0.422 | 500/1k/500 | 0.0667 | 0.33 | 0/3 |
| 8 conditions × 32 cells (`k8n32`) | 0.0206 ± 0.0076 | 0.244 | 0.076 | 0.418 | 500/500/500 | 0.0696 | 0.32 | 0/3 |
| rollout λ=1 every update + EMA (`mean1_l1_ema`) | 0.0216 ± 0.0051 | 0.286 | 0.086 | 0.450 | 1k/5k/1k | 0.0339 | 0.96 | 3/3 |
| rollout λ=10 + EMA (`mean_l10_ema`) | 0.0216 ± 0.0031 | 0.278 | 0.084 | 0.445 | 1k/2k/3k | 0.0354 | 0.92 | 3/3 |
| rollout mean term λ=1, every 5 (`mean_l1`) | 0.0220 ± 0.0021 | 0.262 | 0.083 | 0.443 | 500/2k/2k | 0.0344 | 0.96 | 3/3 |
| rollout mean term λ=10, every 5 (`mean_l10`) | 0.0245 ± 0.0031 | 0.263 | 0.083 | 0.446 | 500/2k/500 | 0.0354 | 0.92 | 3/3 |
| rollout mean term λ=100, every 5 (`mean_l100`) | 0.0245 ± 0.0031 | 0.263 | 0.083 | 0.446 | 500/2k/500 | 0.0373 | 0.85 | 3/3 |
| rollout λ=10 every update + EMA (`mean1_l10_ema`) | 0.0248 ± 0.0005 | 0.308 | 0.089 | 0.463 | 1k/1k/1k | 0.0353 | 0.92 | 3/3 |

References, same split: published CellFlow (WRF-chosen) 0.0432 / 0.265 / 0.076 / 0.433; adjacent mixture 0.0195 / 0.213 / 0.058 / 0.242;
chemCPA-Gauss + mean consistency 0.0129 / 0.201 / 0.051 / 0.586; replicate floor 0.0032 / 0.196 / 0.078 / 0.220.

- 22 variants have now been looked at on 9 conditions, so the best rows are optimistic. chemCPA + mean consistency went from 0.0129
  on validation to 0.022 on test.
- **Ending up ignoring the condition:** 18 of 18 runs with the rollout term; 3 of 15 gene-space runs with one condition per update and no rollout
  term (`base` seed 0, `ema1e3` seed 0, `meanv_l10` seed 1); 0 of 9 gene-space runs with 8 conditions per update (none had a rollout term). 8 of the 9
  published runs with a rollout term (MMD, WRF, WRF-13) are also condition-blind. So the rollout term is the strongest association,
  and having 8 conditions per update may also protect, but the two have not been separated.
- The rollout rows' selected checkpoints mostly predate or barely overlap the term (it starts at step 3,000, or 1,000 for `mean1_*`), so
  their validation numbers say little about the term itself; the end columns do.
- 8 conditions per update in gene space fits means best in-sample, then loses variance (about 0.33 of real at the end). Why is untested.
  One candidate: the network (512 wide) is narrower than the data (2,085 genes), so it may not carry the re-expansion that flow
  matching needs when pairings are random in noise dimensions.
- Latent runs agree across seeds, but their lowest validation MMD² comes at steps 500–3,000 and worsens with training.
- Residual decoding keeps gene-level variance (about 0.82–0.84 at the end vs 0.13–0.15 for inverse decoding); W1(r) is 0.40–0.43 vs 0.82–0.85.
- Stopped early, not informative: `k8n128_mv100` and `k8n128_mv10_ema` (cancelled to free GPUs). The first `mean_*` round crashed or was
  cancelled and was rerun with the bounded rollout. `meanv_*` ran on the first version of the step function (same arithmetic at K = 1).
  `base` seed 0 is a rerun: the first run hit non-finite cells at its final validation.
- The published panel audit was not run here (the file hash differs from the published one; shape and gene order are asserted, and
  `build_panel.py` checks the frozen spec and split).
- Nothing above uses the relationship between a drug's doses yet; these are repairs to training.

**Test decision (2026-11-05, before any CellFlow test number here; scored now at the user's request):** `meanv_l10` (lowest validation
MMD²), `z70_k8_res_lr3e4` (lowest validation MMD² among variants where no seed ended ignoring the condition) and the matched `base`,
each seed at its selected checkpoint. Published generation (512 cells per condition); sampling offset 0 is the comparison, offset 1 a
robustness check. Paired condition bootstraps as declared in the protocol, primary against `base`. This comes before the dose-relationship
step, so that step is developed with these test numbers known: its choices are made on validation only, and any later test score for
it is a second look at the test set and will be reported as such.

### CellFlow test result (scored once for the picks above; jobs 97928 evaluate, 97955 compare; tables in docs/interp/cellflow_test.md)

17 conditions, 3 seeds, frozen evaluator, 512 generated cells per condition, each seed at its lowest validation MMD² checkpoint.

| test | MMD² | RMSE DEG50 | RMSE all | RMSE program | W1(r) |
|---|---|---|---|---|---|
| CellFlow latent, 8 conditions per update (`z70_k8_res_lr3e4`) | 0.0169 ± 0.0023 | 0.211 | 0.054 | 0.080 | 0.413 |
| CellFlow + grouped velocity matching (`meanv_l10`) | 0.0175 ± 0.0053 | 0.214 | 0.057 | 0.083 | 0.410 |
| CellFlow retrain, lowest val MMD² (`base`) | 0.0240 ± 0.0083 | 0.224 | 0.064 | 0.085 | 0.443 |
| chemCPA-Gauss + mean consistency | 0.0223 ± 0.0016 | 0.238 | 0.061 | 0.059 | 0.574 |
| CellFlow (published, WRF-chosen) | 0.0538 ± 0.0276 | 0.284 | 0.092 | 0.082 | 0.419 |
| adjacent-dose mixture (real neighbour cells) | 0.0161 | 0.198 | 0.058 | 0.060 | 0.213 |

Δ = ref − new (> 0: new better), condition-bootstrap 95% CI, conditions improved.

- **Checkpoint rule:** `base` vs published CellFlow, same recipe: MMD² +0.0297 [+0.0209, +0.0389] (17/17), DEG50 +0.060 [+0.037, +0.085]. Most of the
  distance from 0.054 comes from choosing the checkpoint on a distribution metric (the runs also differ).
- **Fixes vs `base` (primary):** `meanv_l10` MMD² +0.0065 [−0.0002, +0.0143] (16/17), DEG50 +0.010 [−0.021, +0.030], all-gene +0.006 [+0.001, +0.011],
  W1(r) +0.033 [+0.020, +0.046]. `z70_k8_res_lr3e4` MMD² +0.0071 [−0.0012, +0.0162] (14/17), DEG50 +0.013 [−0.025, +0.046], all-gene +0.009 [+0.003, +0.016],
  program +0.005 [+0.001, +0.009]. Neither MMD² CI excludes zero.
- **vs the mixture:** MMD² −0.0015 [−0.0080, +0.0050] and −0.0009 [−0.0079, +0.0058]; DEG50 −0.017 [−0.045, +0.009] and −0.013 [−0.036, +0.007]; all-gene +0.000 and
  +0.003, CIs include zero. Clearly worse on program (−0.023, −0.019) and W1(r) (−0.197, −0.200). `base` is worse on MMD² (−0.0080 [−0.0141, −0.0026]).
- **The earlier "chemCPA + mean consistency beats CellFlow" does not hold against the fair retrain.** `base` vs chemCPA-Gauss + mean consistency: MMD² −0.0017
  [−0.0140, +0.0091], DEG50 +0.014 [−0.017, +0.056], all-gene −0.002 [−0.012, +0.006]; program −0.026 [−0.041, −0.010] (chemCPA better); W1(r) +0.131 [+0.025, +0.231]
  (CellFlow better). The fixes vs chemCPA-Gauss: DEG50 better for both (+0.024 [+0.006, +0.043], +0.027 [+0.004, +0.051]), W1(r) better, program worse; MMD² +0.0048
  [−0.0012, +0.0103] and +0.0054 [+0.0000, +0.0109].
- A second sampling seed reproduces the test numbers (0.0169 / 0.210, 0.0173 / 0.212, 0.0238 / 0.222).
- No test evaluation produced non-finite cells. `meanv_l10` seed 1 is scored at step 2,000, before that run ended ignoring the condition.
- Neither fix uses the relationship between a drug's doses.

### Dose gain per drug: one number and two representative drugs (2026-11-06; jobs 101614 collect, 101615 draw; validation drugs only, no test cell read)

`sciplex_interp/dose_gain.py`, stages `dose_gain_collect` and `dose_gain_draw`. Figures `results/interp/figures/dose/6_dose_gain_cases.png` and `7_dose_gain_cells.png`.
Number: `gain` of `dose_sweep.py` on each drug's own direction (top-50 DEGs, measured 10 µM change), per drug rather than pooled. 0: ignores dose. 1: the full measured response.
The response score plays no part, at the user's request; 0–5 of each drug's top-50 DEGs are program genes (0 for both plotted drugs).

| 9 validation drugs | median gain | 95% CI (drugs) | per-seed medians | pooled | median, unseen-dose steps |
|---|---|---|---|---|---|
| Oracle: replicate well 2 vs 1 | 1.02 | 0.64, 1.16 | | 1.09 | 0.76 |
| `z70_lr3e4_mv10_t` (arm) | 0.85 | 0.47, 0.90 | 0.82–0.86 | 0.88 | 0.68 |
| `gaussmean_l100` (chemCPA + mean consistency) | 0.84 | 0.01, 1.00 | 0.71–0.91 | 0.95 | 0.66 |
| `z70_k8_res_lr3e4` (same backbone, no embedding or grouped loss) | 0.58 | 0.23, 0.73 | 0.55–0.60 | 0.64 | 0.70 |
| `base` (CellFlow retrain) | 0.50 | 0.12, 0.66 | 0.31–0.61 | 0.60 | 0.47 |
| `gauss_v1` (chemCPA-Gauss, original) | 0.34 | −0.01, 0.49 | 0.01–0.66 | 0.37 | 0.49 |

Arm minus model, difference of medians, paired drug bootstrap, drugs where the arm is higher:

- `base`: gain +0.34 [+0.17, +0.39], 9/9; unseen +0.22 [−1.45, +0.35], 7/9.
- `gauss_v1`: gain +0.50 [+0.26, +0.61], 9/9; unseen +0.19 [+0.06, +0.49], 8/9.
- `gaussmean_l100`: gain +0.00 [−0.09, +0.50], 5/9; unseen +0.03 [−0.35, +0.42], 6/9.
- `z70_k8_res_lr3e4`: gain +0.27 [+0.17, +0.31], 9/9; unseen −0.01 [−0.83, +0.24], 7/9.

- **Cases:** drugs ranked by the mean gain of `base` and `gauss_v1`, so the arm plays no part in the pick. Median: Givinostat (arm 0.90, chemCPA + mean consistency 1.00, CellFlow retrain 0.51, original chemCPA 0.28). 75th percentile, where the baselines do better than usual: Belinostat (0.89, 1.05, 0.65, 0.38). The opposite direction would land on Droxinostat, which has almost no dose response.
- **Dose, not condition:** at 10 µM every model orders the drugs on PC1 (across-drug r 0.98–1.00). CellFlow retrain flattens the curve rather than under-responding throughout: on the drug direction it reaches 1.80× the measured change at 1 µM and 0.67× at 10 µM. Original chemCPA is short at both (0.58×, 0.41×; across-drug slope about 0.4).
- **Limits:** three of the four doses are training doses. On the unseen steps the arm is not separated from `base` (which overshoots there on Givinostat and Belinostat, 2.14 and 2.63) and does not beat its own backbone. chemCPA + mean consistency ties the median and is closer to 1 on the strong drugs. The arm and its embedding were chosen on these drugs, so its numbers are optimistic. The response score goes the other way (pooled gain on r: arm 0.20, original chemCPA 0.25, chemCPA + mean consistency 0.33).
- **Axis:** the drug direction was fixed before any sweep was read, but it was preferred over PC1 for the per-drug number after both had been seen: three drugs have under 0.35 of dose range on PC1, so their ratios are noise (as low as −3.49). PC1 agrees in direction (median 0.81 / 0.63 / 0.37 for arm / `base` / `gauss_v1`, pooled 0.86 / 0.61 / 0.40), but the arm vs `base` CI on the PC1 median includes zero (+0.18 [−0.40, +1.22]). Only the Gaussian chemCPA was swept.
