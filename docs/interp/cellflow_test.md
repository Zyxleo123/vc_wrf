# CellFlow variants vs the matched base, published CellFlow and references

Frozen evaluator, 3 training seeds (± = SD over seeds). Every CellFlow run here uses the minimum validation MMD² checkpoint; the published CellFlow rows used a WRF rule. Δ = ref − new on condition means; > 0 means the new variant is better. CI: condition bootstrap (two-level condition × seed CI in the CSV).

## val (9 conditions)

| | MMD² | RMSE DEG50 | RMSE all genes | RMSE program | W1(r) |
|---|---|---|---|---|---|
| CellFlow + grouped velocity matching | 0.0111 ± 0.0039 | 0.201 ± 0.019 | 0.050 ± 0.006 | 0.076 ± 0.002 | 0.409 ± 0.036 |
| CellFlow latent, 8 conditions per update | 0.0123 ± 0.0007 | 0.207 ± 0.006 | 0.050 ± 0.000 | 0.077 ± 0.008 | 0.424 ± 0.024 |
| CellFlow retrain (lowest val MMD²) | 0.0193 ± 0.0030 | 0.229 ± 0.001 | 0.058 ± 0.003 | 0.076 ± 0.008 | 0.432 ± 0.035 |
| chemCPA-Gauss + mean consistency | 0.0129 ± 0.0030 | 0.201 ± 0.013 | 0.052 ± 0.005 | 0.051 ± 0.002 | 0.586 ± 0.018 |
| chemCPA-NB + mean consistency | 0.0200 ± 0.0018 | 0.259 ± 0.013 | 0.060 ± 0.003 | 0.062 ± 0.008 | 0.402 ± 0.139 |
| CellFlow (published) | 0.0432 ± 0.0383 | 0.265 ± 0.053 | 0.081 ± 0.029 | 0.076 ± 0.003 | 0.433 ± 0.019 |
| CellFlow + MMD loss (published) | 0.0284 ± 0.0015 | 0.292 ± 0.005 | 0.073 ± 0.002 | 0.082 ± 0.001 | 0.433 ± 0.025 |
| Adjacent-dose mixture (real neighbour cells) | 0.0195 | 0.213 | 0.057 | 0.058 | 0.242 |
| Vehicle cells (no drug effect) | 0.0293 | 0.338 | 0.065 | 0.092 | 0.461 |
| Replicate well (noise floor) | 0.0032 | 0.196 | 0.057 | 0.078 | 0.220 |

### val: paired differences

| new | ref | MMD² | RMSE DEG50 | RMSE all genes | RMSE program | W1(r) |
|---|---|---|---|---|---|---|
| CellFlow + grouped velocity matching | CellFlow retrain (lowest val MMD²) | +0.0082 [+0.0031, +0.0139] 8/9 | +0.028 [+0.009, +0.048] 7/9 | +0.008 [+0.003, +0.014] 7/9 | +0.000 [-0.002, +0.003] 5/9 | +0.023 [+0.008, +0.037] 7/9 |
| CellFlow + grouped velocity matching | chemCPA-Gauss + mean consistency | +0.0018 [+0.0003, +0.0032] 7/9 | +0.000 [-0.017, +0.018] 6/9 | +0.002 [-0.000, +0.004] 6/9 | -0.025 [-0.050, -0.003] 4/9 | +0.177 [+0.056, +0.286] 7/9 |
| CellFlow + grouped velocity matching | chemCPA-NB + mean consistency | +0.0088 [+0.0042, +0.0152] 9/9 | +0.058 [+0.035, +0.088] 9/9 | +0.010 [+0.005, +0.016] 8/9 | -0.014 [-0.047, +0.016] 4/9 | -0.007 [-0.236, +0.196] 6/9 |
| CellFlow + grouped velocity matching | CellFlow (published) | +0.0321 [+0.0190, +0.0458] 9/9 | +0.064 [+0.044, +0.083] 9/9 | +0.032 [+0.024, +0.040] 9/9 | -0.000 [-0.005, +0.005] 4/9 | +0.024 [-0.000, +0.050] 6/9 |
| CellFlow + grouped velocity matching | CellFlow + MMD loss (published) | +0.0173 [+0.0103, +0.0280] 9/9 | +0.091 [+0.055, +0.148] 9/9 | +0.024 [+0.017, +0.030] 9/9 | +0.006 [+0.003, +0.010] 9/9 | +0.025 [+0.008, +0.043] 9/9 |
| CellFlow + grouped velocity matching | Adjacent-dose mixture (real neighbour cells) | +0.0083 [+0.0001, +0.0187] 5/9 | +0.012 [-0.010, +0.036] 5/9 | +0.007 [+0.001, +0.016] 7/9 | -0.018 [-0.041, +0.001] 4/9 | -0.167 [-0.318, -0.036] 1/9 |
| CellFlow + grouped velocity matching | Vehicle cells (no drug effect) | +0.0181 [+0.0023, +0.0441] 5/9 | +0.137 [+0.040, +0.258] 7/9 | +0.015 [+0.002, +0.034] 6/9 | +0.016 [+0.006, +0.026] 7/9 | +0.053 [+0.010, +0.097] 7/9 |
| CellFlow + grouped velocity matching | Replicate well (noise floor) | -0.0079 [-0.0168, -0.0006] 2/9 | -0.005 [-0.075, +0.053] 6/9 | +0.007 [-0.005, +0.017] 6/9 | +0.002 [-0.031, +0.031] 5/9 | -0.189 [-0.408, +0.021] 2/9 |
| CellFlow latent, 8 conditions per update | CellFlow retrain (lowest val MMD²) | +0.0070 [+0.0022, +0.0124] 7/9 | +0.022 [+0.002, +0.043] 6/9 | +0.008 [+0.002, +0.014] 7/9 | -0.001 [-0.003, +0.003] 3/9 | +0.008 [-0.001, +0.020] 6/9 |
| CellFlow latent, 8 conditions per update | chemCPA-Gauss + mean consistency | +0.0007 [-0.0008, +0.0021] 5/9 | -0.006 [-0.031, +0.019] 3/9 | +0.002 [-0.000, +0.004] 6/9 | -0.026 [-0.047, -0.006] 1/9 | +0.162 [+0.043, +0.268] 7/9 |
| CellFlow latent, 8 conditions per update | chemCPA-NB + mean consistency | +0.0077 [+0.0036, +0.0137] 9/9 | +0.051 [+0.029, +0.081] 9/9 | +0.010 [+0.005, +0.015] 8/9 | -0.015 [-0.044, +0.012] 4/9 | -0.022 [-0.254, +0.180] 6/9 |
| CellFlow latent, 8 conditions per update | CellFlow (published) | +0.0310 [+0.0179, +0.0445] 9/9 | +0.057 [+0.036, +0.080] 9/9 | +0.031 [+0.023, +0.040] 9/9 | -0.001 [-0.005, +0.003] 4/9 | +0.009 [-0.019, +0.036] 5/9 |
| CellFlow latent, 8 conditions per update | CellFlow + MMD loss (published) | +0.0162 [+0.0085, +0.0270] 8/9 | +0.084 [+0.043, +0.145] 9/9 | +0.023 [+0.017, +0.030] 9/9 | +0.005 [-0.001, +0.014] 6/9 | +0.010 [-0.021, +0.043] 3/9 |
| CellFlow latent, 8 conditions per update | Adjacent-dose mixture (real neighbour cells) | +0.0072 [-0.0004, +0.0169] 4/9 | +0.005 [-0.012, +0.024] 5/9 | +0.007 [+0.001, +0.016] 7/9 | -0.019 [-0.038, -0.003] 3/9 | -0.182 [-0.333, -0.049] 3/9 |
| CellFlow latent, 8 conditions per update | Vehicle cells (no drug effect) | +0.0170 [+0.0013, +0.0428] 5/9 | +0.130 [+0.037, +0.251] 7/9 | +0.015 [+0.001, +0.034] 5/9 | +0.015 [+0.002, +0.029] 6/9 | +0.038 [-0.010, +0.085] 7/9 |
| CellFlow latent, 8 conditions per update | Replicate well (noise floor) | -0.0091 [-0.0185, -0.0014] 2/9 | -0.011 [-0.083, +0.052] 6/9 | +0.007 [-0.005, +0.017] 6/9 | +0.001 [-0.028, +0.027] 5/9 | -0.204 [-0.419, +0.003] 2/9 |
| CellFlow retrain (lowest val MMD²) | chemCPA-Gauss + mean consistency | -0.0064 [-0.0127, -0.0008] 4/9 | -0.028 [-0.054, -0.003] 3/9 | -0.006 [-0.013, +0.000] 4/9 | -0.025 [-0.048, -0.004] 3/9 | +0.155 [+0.031, +0.261] 7/9 |
| CellFlow retrain (lowest val MMD²) | chemCPA-NB + mean consistency | +0.0007 [-0.0036, +0.0052] 4/9 | +0.029 [+0.002, +0.058] 6/9 | +0.002 [-0.006, +0.010] 5/9 | -0.014 [-0.045, +0.014] 4/9 | -0.030 [-0.264, +0.173] 6/9 |
| CellFlow retrain (lowest val MMD²) | CellFlow (published) | +0.0239 [+0.0134, +0.0353] 9/9 | +0.035 [+0.026, +0.045] 9/9 | +0.023 [+0.018, +0.029] 9/9 | -0.000 [-0.004, +0.004] 3/9 | +0.001 [-0.026, +0.029] 4/9 |
| CellFlow retrain (lowest val MMD²) | CellFlow + MMD loss (published) | +0.0091 [+0.0007, +0.0169] 8/9 | +0.063 [+0.030, +0.115] 9/9 | +0.015 [+0.006, +0.023] 8/9 | +0.006 [+0.000, +0.012] 6/9 | +0.002 [-0.023, +0.030] 3/9 |
| CellFlow retrain (lowest val MMD²) | Adjacent-dose mixture (real neighbour cells) | +0.0002 [-0.0048, +0.0066] 4/9 | -0.016 [-0.041, +0.009] 3/9 | -0.001 [-0.007, +0.006] 3/9 | -0.018 [-0.039, -0.001] 5/9 | -0.190 [-0.347, -0.054] 3/9 |
| CellFlow retrain (lowest val MMD²) | Vehicle cells (no drug effect) | +0.0100 [-0.0023, +0.0315] 4/9 | +0.108 [+0.030, +0.218] 7/9 | +0.007 [-0.002, +0.022] 4/9 | +0.016 [+0.004, +0.027] 6/9 | +0.030 [-0.016, +0.071] 7/9 |
| CellFlow retrain (lowest val MMD²) | Replicate well (noise floor) | -0.0161 [-0.0305, -0.0045] 1/9 | -0.033 [-0.112, +0.034] 4/9 | -0.001 [-0.017, +0.013] 4/9 | +0.001 [-0.030, +0.029] 5/9 | -0.211 [-0.433, -0.006] 2/9 |

## test (17 conditions)

| | MMD² | RMSE DEG50 | RMSE all genes | RMSE program | W1(r) |
|---|---|---|---|---|---|
| CellFlow + grouped velocity matching | 0.0175 ± 0.0053 | 0.214 ± 0.020 | 0.057 ± 0.008 | 0.083 ± 0.002 | 0.410 ± 0.043 |
| CellFlow latent, 8 conditions per update | 0.0169 ± 0.0023 | 0.211 ± 0.006 | 0.054 ± 0.001 | 0.080 ± 0.008 | 0.413 ± 0.034 |
| CellFlow retrain (lowest val MMD²) | 0.0240 ± 0.0083 | 0.224 ± 0.010 | 0.064 ± 0.008 | 0.085 ± 0.011 | 0.443 ± 0.052 |
| chemCPA-Gauss + mean consistency | 0.0223 ± 0.0016 | 0.238 ± 0.010 | 0.061 ± 0.002 | 0.059 ± 0.003 | 0.574 ± 0.014 |
| chemCPA-NB + mean consistency | 0.0300 ± 0.0050 | 0.290 ± 0.027 | 0.068 ± 0.003 | 0.068 ± 0.004 | 0.385 ± 0.072 |
| CellFlow (published) | 0.0538 ± 0.0276 | 0.284 ± 0.038 | 0.092 ± 0.021 | 0.082 ± 0.004 | 0.419 ± 0.029 |
| CellFlow + MMD loss (published) | 0.0512 ± 0.0007 | 0.374 ± 0.002 | 0.088 ± 0.001 | 0.089 ± 0.002 | 0.411 ± 0.033 |
| Adjacent-dose mixture (real neighbour cells) | 0.0161 | 0.198 | 0.058 | 0.060 | 0.213 |
| Vehicle cells (no drug effect) | 0.0594 | 0.447 | 0.085 | 0.101 | 0.471 |
| Replicate well (noise floor) | 0.0076 | 0.245 | 0.068 | 0.086 | 0.273 |

Second sampling seed (offset 1), MMD² / RMSE DEG50: CellFlow + grouped velocity matching 0.0173 / 0.212; CellFlow latent, 8 conditions per update 0.0169 / 0.210; CellFlow retrain (lowest val MMD²) 0.0238 / 0.222

### test: paired differences

| new | ref | MMD² | RMSE DEG50 | RMSE all genes | RMSE program | W1(r) |
|---|---|---|---|---|---|---|
| CellFlow + grouped velocity matching | CellFlow retrain (lowest val MMD²) | +0.0065 [-0.0002, +0.0142] 16/17 | +0.010 [-0.021, +0.031] 16/17 | +0.006 [+0.001, +0.011] 15/17 | +0.001 [-0.002, +0.005] 9/17 | +0.033 [+0.020, +0.046] 15/17 |
| CellFlow + grouped velocity matching | chemCPA-Gauss + mean consistency | +0.0048 [-0.0010, +0.0103] 14/17 | +0.024 [+0.006, +0.042] 14/17 | +0.004 [-0.002, +0.010] 14/17 | -0.024 [-0.041, -0.008] 3/17 | +0.165 [+0.064, +0.262] 12/17 |
| CellFlow + grouped velocity matching | chemCPA-NB + mean consistency | +0.0124 [+0.0056, +0.0220] 16/17 | +0.075 [+0.042, +0.126] 16/17 | +0.010 [+0.006, +0.016] 15/17 | -0.016 [-0.034, +0.002] 10/17 | -0.024 [-0.168, +0.107] 10/17 |
| CellFlow + grouped velocity matching | CellFlow (published) | +0.0362 [+0.0235, +0.0498] 17/17 | +0.070 [+0.043, +0.099] 15/17 | +0.034 [+0.026, +0.043] 17/17 | -0.002 [-0.005, +0.002] 5/17 | +0.009 [-0.010, +0.029] 10/17 |
| CellFlow + grouped velocity matching | CellFlow + MMD loss (published) | +0.0337 [+0.0111, +0.0621] 16/17 | +0.160 [+0.064, +0.277] 16/17 | +0.031 [+0.016, +0.047] 16/17 | +0.006 [-0.001, +0.012] 8/17 | +0.001 [-0.023, +0.027] 8/17 |
| CellFlow + grouped velocity matching | Adjacent-dose mixture (real neighbour cells) | -0.0015 [-0.0080, +0.0049] 8/17 | -0.017 [-0.045, +0.010] 5/17 | +0.000 [-0.008, +0.008] 9/17 | -0.023 [-0.043, -0.004] 8/17 | -0.197 [-0.370, -0.042] 5/17 |
| CellFlow + grouped velocity matching | Vehicle cells (no drug effect) | +0.0419 [+0.0065, +0.0846] 12/17 | +0.233 [+0.093, +0.395] 16/17 | +0.028 [+0.005, +0.055] 11/17 | +0.018 [+0.009, +0.028] 12/17 | +0.061 [+0.026, +0.098] 13/17 |
| CellFlow + grouped velocity matching | Replicate well (noise floor) | -0.0099 [-0.0203, -0.0007] 9/17 | +0.030 [-0.020, +0.076] 11/17 | +0.010 [-0.001, +0.021] 11/17 | +0.003 [-0.021, +0.026] 11/17 | -0.137 [-0.316, +0.033] 7/17 |
| CellFlow latent, 8 conditions per update | CellFlow retrain (lowest val MMD²) | +0.0071 [-0.0011, +0.0162] 14/17 | +0.013 [-0.026, +0.046] 13/17 | +0.009 [+0.002, +0.016] 15/17 | +0.005 [+0.001, +0.009] 12/17 | +0.030 [+0.011, +0.050] 15/17 |
| CellFlow latent, 8 conditions per update | chemCPA-Gauss + mean consistency | +0.0054 [-0.0000, +0.0111] 14/17 | +0.027 [+0.004, +0.052] 11/17 | +0.007 [+0.002, +0.013] 14/17 | -0.021 [-0.034, -0.007] 3/17 | +0.161 [+0.071, +0.246] 12/17 |
| CellFlow latent, 8 conditions per update | chemCPA-NB + mean consistency | +0.0130 [+0.0073, +0.0206] 17/17 | +0.079 [+0.048, +0.117] 16/17 | +0.014 [+0.009, +0.018] 17/17 | -0.012 [-0.028, +0.002] 10/17 | -0.028 [-0.166, +0.100] 11/17 |
| CellFlow latent, 8 conditions per update | CellFlow (published) | +0.0368 [+0.0227, +0.0520] 17/17 | +0.073 [+0.039, +0.114] 16/17 | +0.037 [+0.027, +0.048] 17/17 | +0.002 [-0.003, +0.006] 8/17 | +0.006 [-0.010, +0.023] 8/17 |
| CellFlow latent, 8 conditions per update | CellFlow + MMD loss (published) | +0.0343 [+0.0121, +0.0611] 16/17 | +0.163 [+0.067, +0.283] 16/17 | +0.034 [+0.020, +0.050] 16/17 | +0.009 [-0.001, +0.020] 9/17 | -0.002 [-0.034, +0.033] 7/17 |
| CellFlow latent, 8 conditions per update | Adjacent-dose mixture (real neighbour cells) | -0.0009 [-0.0078, +0.0060] 6/17 | -0.013 [-0.035, +0.007] 6/17 | +0.003 [-0.004, +0.012] 11/17 | -0.019 [-0.036, -0.004] 6/17 | -0.200 [-0.365, -0.051] 4/17 |
| CellFlow latent, 8 conditions per update | Vehicle cells (no drug effect) | +0.0425 [+0.0076, +0.0844] 10/17 | +0.236 [+0.098, +0.397] 16/17 | +0.031 [+0.008, +0.058] 11/17 | +0.021 [+0.008, +0.035] 11/17 | +0.058 [+0.013, +0.104] 11/17 |
| CellFlow latent, 8 conditions per update | Replicate well (noise floor) | -0.0093 [-0.0203, +0.0002] 9/17 | +0.034 [-0.022, +0.084] 12/17 | +0.013 [+0.003, +0.023] 14/17 | +0.006 [-0.014, +0.027] 11/17 | -0.140 [-0.312, +0.027] 6/17 |
| CellFlow retrain (lowest val MMD²) | chemCPA-Gauss + mean consistency | -0.0017 [-0.0140, +0.0091] 10/17 | +0.014 [-0.017, +0.055] 9/17 | -0.002 [-0.012, +0.006] 8/17 | -0.026 [-0.041, -0.010] 2/17 | +0.131 [+0.025, +0.231] 12/17 |
| CellFlow retrain (lowest val MMD²) | chemCPA-NB + mean consistency | +0.0059 [-0.0060, +0.0184] 12/17 | +0.066 [+0.023, +0.122] 14/17 | +0.004 [-0.004, +0.012] 12/17 | -0.017 [-0.035, -0.001] 9/17 | -0.058 [-0.208, +0.080] 10/17 |
| CellFlow retrain (lowest val MMD²) | CellFlow (published) | +0.0297 [+0.0209, +0.0389] 17/17 | +0.060 [+0.037, +0.085] 16/17 | +0.028 [+0.022, +0.034] 17/17 | -0.003 [-0.006, +0.000] 5/17 | -0.024 [-0.046, -0.004] 5/17 |
| CellFlow retrain (lowest val MMD²) | CellFlow + MMD loss (published) | +0.0272 [-0.0004, +0.0567] 15/17 | +0.150 [+0.043, +0.277] 16/17 | +0.025 [+0.007, +0.043] 15/17 | +0.004 [-0.004, +0.012] 9/17 | -0.032 [-0.053, -0.010] 4/17 |
| CellFlow retrain (lowest val MMD²) | Adjacent-dose mixture (real neighbour cells) | -0.0080 [-0.0141, -0.0026] 3/17 | -0.026 [-0.060, +0.003] 6/17 | -0.006 [-0.013, +0.001] 7/17 | -0.024 [-0.043, -0.007] 6/17 | -0.230 [-0.411, -0.069] 4/17 |
| CellFlow retrain (lowest val MMD²) | Vehicle cells (no drug effect) | +0.0353 [-0.0033, +0.0800] 7/17 | +0.223 [+0.079, +0.397] 15/17 | +0.021 [-0.003, +0.050] 8/17 | +0.016 [+0.005, +0.028] 11/17 | +0.028 [-0.005, +0.061] 10/17 |
| CellFlow retrain (lowest val MMD²) | Replicate well (noise floor) | -0.0165 [-0.0318, -0.0047] 6/17 | +0.021 [-0.025, +0.064] 11/17 | +0.004 [-0.010, +0.017] 10/17 | +0.001 [-0.021, +0.024] 11/17 | -0.170 [-0.353, +0.009] 5/17 |

