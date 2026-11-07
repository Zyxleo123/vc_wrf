# CellFlow strongest-arm ablations

Validation split (9 held-out-dose conditions), 512 generated cells per condition, 3 training seeds. Each seed uses its minimum-validation-MMD² checkpoint. Values are mean ± sample SD over seeds. Response-score and weak-responder quantities are excluded.

| Arm | Selected steps | MMD² | RMSE DEG50 | RMSE all genes | RMSE program |
|---|---|---|---|---|---|
| Full arm | 3000/5000/2000 | 0.0070 ± 0.0002 | 0.174 ± 0.004 | 0.044 ± 0.001 | 0.072 ± 0.002 |
| − grouped velocity matching | 2000/1000/1000 | 0.0072 ± 0.0003 | 0.177 ± 0.002 | 0.044 ± 0.000 | 0.074 ± 0.003 |
| − condition response embedding | 3000/3000/3000 | 0.0107 ± 0.0017 | 0.198 ± 0.008 | 0.049 ± 0.002 | 0.075 ± 0.004 |
| − latent projection | 500/2000/1000 | 0.0079 ± 0.0003 | 0.185 ± 0.011 | 0.046 ± 0.000 | 0.077 ± 0.002 |
| − drug and dose covariates | 3000/5000/2000 | 0.0069 ± 0.0003 | 0.174 ± 0.005 | 0.044 ± 0.001 | 0.071 ± 0.002 |

Paired differences versus **Full arm**. Δ = ablation − full arm, so positive values are worse; intervals are condition-bootstrap 95% CIs.

| Ablation | MMD² | RMSE DEG50 | RMSE all genes | RMSE program |
|---|---|---|---|---|
| − grouped velocity matching | +0.0002 [-0.0004, +0.0009] | +0.003 [-0.001, +0.008] | -0.000 [-0.001, +0.001] | +0.002 [+0.001, +0.003] |
| − condition response embedding | +0.0037 [+0.0003, +0.0079] | +0.025 [+0.001, +0.052] | +0.005 [+0.001, +0.010] | +0.003 [+0.000, +0.007] |
| − latent projection | +0.0010 [-0.0001, +0.0023] | +0.011 [-0.000, +0.024] | +0.002 [+0.000, +0.004] | +0.005 [+0.002, +0.008] |
| − drug and dose covariates | -0.0001 [-0.0005, +0.0003] | +0.001 [-0.004, +0.007] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.000] |
