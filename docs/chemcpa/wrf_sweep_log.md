# chemCPA-NB + response-fraction loss: sweep log

Protocol (fixed 2026-10-30, before round 1):
- Iterate on validation only (9 conditions). Test is scored once, for the single chosen config.
- Every run keeps the checkpoint with the lowest validation MMD² (same rule as the chemCPA-NB baseline).
- Hyperparameters (weight λ, smoothness h, and later loss sizes) are chosen on mean validation WRF error at q0.95 over 3 seeds.
- Positive result: validation WRF err q0.95 at least 1 pp below chemCPA-NB (7.44 → ≤ 6.44),
  with validation MMD² and DEG50 RMSE no more than 10% worse than chemCPA-NB (MMD² ≤ 0.0299, DEG50 ≤ 0.355).
  Stretch: below vehicle-as-prediction (5.69).
- Budget: at most 3 sweep rounds. All rounds are reported.

Baseline (chemCPA-NB, val, 3 seeds): WRF q0.95 7.44 ± 0.27, MMD² 0.0272, DEG50 0.322.

## Round 1
λ ∈ {10, 100, 1000} × h ∈ {0.1, 0.5}; 8 conditions × 64 cells per autoencoder step; spec taus (q 0.9/0.95/0.975).
Jobs: train 91163, evaluate 91164, report 91165 (submission 20261030-172753). Validation, mean ± seed SD:

| variant | WRF q0.95 | MMD² | DEG50 | criterion |
|---|---|---|---|---|
| λ=10, h=0.1 | 5.36 ± 0.28 | 0.0336 | 0.459 | fails MMD², DEG50 |
| **λ=10, h=0.5** | **4.95 ± 0.04** | **0.0252** | **0.339** | **passes (also below vehicle 5.69)** |
| λ=100, h=0.1 | 4.78 ± 0.88 | 0.0332 | 0.474 | fails MMD², DEG50 |
| λ=100, h=0.5 | 4.60 ± 0.34 | 0.0289 | 0.367 | fails DEG50 |
| λ=1000, h=0.1 | 5.40 ± 1.34 | 0.0622 | 0.519 | fails MMD², DEG50 |
| λ=1000, h=0.5 | 4.98 ± 0.84 | 0.0555 | 0.504 | fails MMD², DEG50 |
| chemCPA-NB | 7.44 ± 0.27 | 0.0272 | 0.322 | baseline |

Decision: positive on validation after round 1; stop iterating. Chosen config: nbwrf_l10_h0.5 (the only one meeting all criteria).
Not in the criterion but reported: program-gene mean RMSE 0.144 ± 0.053 vs 0.095 baseline (the loss distorts the
targeted program's means); all-gene mean RMSE 0.0735 vs 0.0658. Val predicted-vs-measured WRF Pearson 0.50 (baseline -0.39);
predicted range 87.5-97.9% (baseline 97.2-98.3%). Control reconstruction WRF per seed 0.986 / 0.911 / 0.925 (real 0.950).

## Test (single run of the chosen config)
Jobs: evaluate 91207, report 91208 (submission 20261030-173429). Test, 17 conditions, mean ± seed SD:

| | WRF q0.9 | q0.95 | q0.975 | W1(r) | MMD² | DEG50 | program RMSE |
|---|---|---|---|---|---|---|---|
| chemCPA-NB + WRF loss (λ=10, h=0.5) | 7.9 ± 0.9 | **5.27 ± 0.50** | 4.1 ± 0.2 | 0.51 | 0.049 | 0.420 | 0.135 |
| chemCPA-NB | 14.3 ± 1.0 | 7.13 ± 0.45 | 5.6 ± 0.4 | 0.59 | 0.049 | 0.393 | 0.102 |
| vehicle | 9.9 | 4.89 | 3.5 | 0.48 | 0.059 | 0.436 | 0.099 |
| replicate | 6.9 | 3.81 | 2.6 | 0.27 | 0.008 | 0.245 | 0.086 |

- vs chemCPA-NB: every seed better (4.71/5.47/5.64 vs 6.85-7.66); paired condition bootstrap −1.86 pp, 95% CI [−4.69, +0.65]; 12/17 conditions improved.
- vs vehicle: +0.38 pp, 95% CI [−2.40, +2.75]; 5/17 conditions improved. The validation stretch result did not hold on test.
- Predicted-vs-measured WRF Pearson 0.52 (baseline 0.02); predicted range 76.0–98.7% (baseline 95.9–98.5%).
- Threshold sensitivity: near-noise-floor error at q ≥ 0.85, but 16–17 pp error at q ≤ 0.4 (baseline/vehicle 5–18):
  the loss matched the supervised fractions partly by distorting the rest of the response distribution.

## Round 2 (requested after the round-1 test run)
Motivation, from validation only: the round-1 pick's validation WRF error is 21 pp at q=0.1 (chemCPA-NB 6.8),
so the loss distorts unsupervised thresholds. Round 2 supervises a dense set of control quantiles
q ∈ {0.1, 0.2, …, 0.9, 0.95, 0.975} (tau_q from control r(x), the spec's rule).
Grid: λ ∈ {10, 30, 100} × h ∈ {0.25, 0.5}; other settings as round 1.
Criterion (round 1's, plus one): val WRF q0.95 ≤ 6.44; MMD² ≤ 0.0299; DEG50 ≤ 0.355; and mean val WRF error
over the 13-threshold sensitivity grid ≤ chemCPA-NB's 15.85. Among passing configs, lowest val WRF q0.95.
Caveat for the eventual test number: the decision to run this round came after the round-1 test result was seen.
Jobs: train 91237, evaluate 91238, report 91239 (submission 20261030-175952). Validation, mean ± seed SD
("grid" = mean WRF error over the 13 sensitivity thresholds; chemCPA-NB 15.85, vehicle 13.88):

| variant | WRF q0.95 | grid | MMD² | DEG50 | program RMSE | criterion |
|---|---|---|---|---|---|---|
| **grid λ=10, h=0.25** | **4.78 ± 0.75** | **9.27** | **0.0173** | **0.302** | 0.145 | **passes** |
| grid λ=10, h=0.5 | 5.64 ± 0.14 | 12.72 | 0.0203 | 0.325 | 0.139 | passes |
| grid λ=30, h=0.25 | 5.74 ± 0.43 | 12.12 | 0.0212 | 0.383 | 0.211 | fails DEG50 |
| grid λ=30, h=0.5 | 6.22 ± 0.76 | 14.31 | 0.0226 | 0.388 | 0.180 | fails DEG50 |
| grid λ=100, h=0.25 | 5.01 ± 0.66 | 8.63 | 0.0413 | 0.471 | 0.233 | fails MMD², DEG50 |
| grid λ=100, h=0.5 | 5.68 ± 0.27 | 11.12 | 0.0358 | 0.443 | 0.237 | fails MMD², DEG50 |
| round-1 pick (3 taus) λ=10, h=0.5 | 4.95 ± 0.04 | 14.76 | 0.0252 | 0.339 | 0.144 | (round 1) |
| chemCPA-NB | 7.44 ± 0.27 | 15.85 | 0.0272 | 0.322 | 0.095 | baseline |

Decision: nbwrfgrid_l10_h0.25 (lowest val WRF q0.95 among passing). Stop; score test once.

## Round 2 test (single run)
Jobs: evaluate 91294, report 91295 (submission 20261030-180637). Test, 17 conditions, mean ± seed SD
("grid" = mean WRF error over the 13 sensitivity thresholds):

| | WRF q0.95 | grid | W1(r) | MMD² | DEG50 | program RMSE |
|---|---|---|---|---|---|---|
| round 2: grid λ=10, h=0.25 | 6.58 ± 0.17 | **9.92** | **0.44** | **0.039** | 0.404 | 0.145 |
| round 1: λ=10, h=0.5 | **5.27 ± 0.50** | 12.92 | 0.51 | 0.049 | 0.420 | 0.135 |
| chemCPA-NB | 7.13 ± 0.45 | 16.21 | 0.59 | 0.049 | 0.393 | 0.102 |
| vehicle | 4.89 | 15.10 | 0.48 | 0.059 | 0.436 | 0.099 |
| replicate | 3.81 | 7.98 | 0.27 | 0.008 | 0.245 | 0.086 |

- Primary (q0.95) vs chemCPA-NB: −0.56 pp, 95% CI [−2.70, +1.56], 7/17 improved: no clear gain at the frozen primary threshold.
  vs round-1 pick: +1.30 pp, CI [+0.41, +2.38] (worse). vs vehicle: +1.69 pp, CI [−0.32, +3.62].
- Whole response distribution: best model on every distribution-level measure (grid 9.9 vs 16.2 NB, 15.1 vehicle,
  8.0 replicate floor; mid thresholds q 0.5–0.8 at 10–11.5 pp vs vehicle 20–24, about at the replicate floor).
  Predicted-vs-measured WRF Pearson at q0.95: 0.63 (round 1 0.52, NB 0.02).
- Validation overstated the primary gain (val 4.78 ± 0.75 → test 6.58 ± 0.17).
Stopped after round 2: further rounds would be tuned with knowledge of two test results.
