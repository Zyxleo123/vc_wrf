# CellFlow gene-space condition response embedding × grouped velocity matching comparison

Validation split (9 held-out-dose conditions), 512 generated cells per condition, 3 training seeds. Each seed uses its minimum-validation-MMD² checkpoint. Values are mean ± sample SD over seeds. Response-score and weak-responder quantities are excluded.

| Arm | Selected steps | MMD² | RMSE DEG50 | RMSE all genes | RMSE program |
|---|---|---|---|---|---|
| Matched retrained baseline | 2000/1000/1000 | 0.0191 ± 0.0006 | 0.224 ± 0.015 | 0.056 ± 0.002 | 0.076 ± 0.008 |
| + condition response embedding | 500/500/500 | 0.0091 ± 0.0007 | 0.193 ± 0.004 | 0.048 ± 0.001 | 0.072 ± 0.001 |
| + grouped velocity matching | 5000/3000/3000 | 0.0169 ± 0.0021 | 0.215 ± 0.009 | 0.057 ± 0.001 | 0.082 ± 0.006 |
| + condition response embedding + grouped velocity matching | 500/2000/1000 | 0.0079 ± 0.0003 | 0.185 ± 0.011 | 0.046 ± 0.000 | 0.077 ± 0.002 |
| Published-recipe retrain (context) | 5000/2000/5000 | 0.0193 ± 0.0030 | 0.229 ± 0.001 | 0.058 ± 0.003 | 0.076 ± 0.008 |

Paired differences versus **Matched retrained baseline**. Δ = arm − reference, so positive values are worse; intervals are condition-bootstrap 95% CIs.

| Arm | MMD² | RMSE DEG50 | RMSE all genes | RMSE program |
|---|---|---|---|---|
| + condition response embedding | -0.0100 [-0.0172, -0.0047] | -0.031 [-0.072, +0.002] | -0.008 [-0.013, -0.004] | -0.005 [-0.008, -0.001] |
| + grouped velocity matching | -0.0022 [-0.0036, -0.0012] | -0.008 [-0.018, -0.000] | +0.001 [-0.001, +0.002] | +0.006 [+0.004, +0.007] |
| + condition response embedding + grouped velocity matching | -0.0112 [-0.0190, -0.0053] | -0.039 [-0.073, -0.009] | -0.010 [-0.015, -0.005] | +0.000 [-0.004, +0.005] |
| Published-recipe retrain (context) | +0.0002 [-0.0018, +0.0022] | +0.005 [-0.011, +0.022] | +0.002 [-0.002, +0.005] | -0.000 [-0.004, +0.004] |

Additional paired contrasts. Δ = first arm − second arm; positive values are worse.

| Contrast | MMD² | RMSE DEG50 | RMSE all genes | RMSE program |
|---|---|---|---|---|
| + condition response embedding + grouped velocity matching − + condition response embedding | -0.0012 [-0.0021, -0.0004] | -0.008 [-0.015, -0.000] | -0.002 [-0.003, -0.001] | +0.005 [+0.003, +0.007] |
| + condition response embedding + grouped velocity matching − + grouped velocity matching | -0.0090 [-0.0161, -0.0035] | -0.030 [-0.058, -0.007] | -0.011 [-0.017, -0.006] | -0.005 [-0.009, -0.002] |
