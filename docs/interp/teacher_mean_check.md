# Direct mean-shift prediction from the condition response embedding

2026-11-06. Quick evaluation of the existing `results/interp/teacher/autodec.npz`, with no refitting or tuning.

The proposed operation works directly:

`predicted_shift = coefficients @ basis`

`predicted_mean = control_mean + predicted_shift`

The 20 numbers are signed coefficients (including magnitude), not a unit direction. Each gene's predicted shift is the inner product of those coefficients with its 20 loadings. The saved basis has shape 20 × 2,085 and consists of orthonormal right singular vectors of the 78 training-condition mean shifts relative to vehicle. There is no additional eigenvalue multiplication, centering offset, or scale correction: the saved coefficients are already unscaled. If using the normalized covariate supplied to CellFlow instead, undo its normalization first.

Both the predictor and basis were fit on training conditions only. These held-out conditions are unseen doses of drugs seen at other doses, not unseen drugs. Held-out measurements are used only for scoring and the explicitly labeled oracle diagnostic below. Test results are exploratory because this test set has been inspected in earlier experiments.

Values below are means of per-condition errors; lower is better. Adjacent interpolation uses the two bracketing training-dose means and the existing log1p-dose weight, without cell-sampling noise.

| Split | Conditions | Method | All-gene RMSE | Top-50 DEG RMSE |
|---|---:|---|---:|---:|
| Validation | 9 | Direct embedding reconstruction | **0.03779** | **0.15318** |
| Validation | 9 | Adjacent-dose interpolation | 0.05522 | 0.20955 |
| Validation | 9 | Zero shift (vehicle) | 0.06202 | 0.33222 |
| Validation | 9 | Oracle projection into the same 20 directions | 0.02806 | 0.09838 |
| Test | 17 | Direct embedding reconstruction | **0.04468** | **0.16563** |
| Test | 17 | Adjacent-dose interpolation | 0.05533 | 0.19149 |
| Test | 17 | Zero shift (vehicle) | 0.08166 | 0.43621 |
| Test | 17 | Oracle projection into the same 20 directions | 0.02959 | 0.09931 |

On test, direct reconstruction reduces mean all-gene RMSE by **19.2%** versus adjacent interpolation and wins on **16/17 conditions**. The paired mean difference is −0.01065 (95% condition-bootstrap interval −0.01961 to −0.00212; 10,000 draws). Top-50 DEG RMSE improves by 13.5%, winning on 14/17 conditions, but its difference interval includes zero (−0.05529 to +0.00950).

Test mean per-condition Pearson correlation of gene shifts is **0.624**, versus 0.539 for adjacent interpolation. Pooled shift skill, `1 − sum(prediction_error²) / sum(measured_shift²)`, is **0.798**, versus 0.668 for interpolation. This weights strong responses more heavily: equal-condition mean shift skill is only 0.333. Thus the 80% figure describes pooled shift energy, not uniformly accurate prediction of every condition.

The oracle uses measured held-out coefficients, `(measured_shift @ basis.T) @ basis`, and is not a deployable prediction. It retains **92.5%** of pooled test shift energy. Of the direct predictor's residual squared error, about **37%** is outside the 20-direction basis and **63%** is coefficient prediction error within it. The basis is sufficient to describe most of the shift; predicting the coefficients remains the larger source of error.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m sciplex_interp.teacher_mean_check
```

Code: [`teacher_mean_check.py`](../../sciplex_interp/teacher_mean_check.py). Outputs: `results/interp/teacher/mean_check/{per_condition,summary,paired}.csv`. The run checks basis orthonormality, the saved control mean, and that interpolation uses training conditions only.

Add `--plot` to write `mean_response_histograms_{val,test}.{png,pdf}` (one panel per condition) and `mean_response_histogram_pooled_{val,test}.{png,pdf}` (pooled overview) in the same output directory. Each histogram contains one value per gene: its mean over all treated cells, pooling replicates with cell-count weighting, minus its mean over all vehicle cells. The three overlays share bin edges within each panel and include all 2,085 genes without trimming tails. The pooled overview weights conditions equally. Histograms compare marginal distributions across genes; they do not assess whether the same genes receive the correct predicted shifts.

For a compact view of individual conditions, use `mean_response_histogram_examples_test.{png,pdf}`. Its two conditions are selected at the median and upper quartile of measured shift norm among the 17 test conditions (selection independent of prediction error). The density axis is logarithmic so the tails remain visible alongside the peak near zero; each panel contains only one condition.
