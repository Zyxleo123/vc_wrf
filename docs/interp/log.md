# Held-out-dose prediction from adjacent doses: study log

Scope: population metrics only (MMD², mean-expression RMSEs, W1(r), plus supplementary energy
distance and Pearson Δ). WRF is out of scope here.

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
