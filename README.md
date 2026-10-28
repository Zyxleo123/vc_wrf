# sci-Plex3 held-out-dose prediction: data + shared evaluation

Shared data preparation and metrics for the response-fraction regularization project.
Every baseline (nearest-dose, adjacent-dose mixtures, WRF regression, scVIDR, FM, FM+MMD,
FM+fraction loss, ...) is scored by the same code against the same frozen spec:
`sciplex_eval` + `artifacts/eval_spec_v1`.

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt   # Babel: /data/user_data/yixiz/envs/sciplex
```

Data (Babel paths; pass your own `DATA_DIR` elsewhere):

| step | command | output |
|---|---|---|
| download (3.2 GB, GEO GSE139944 / GSM4150378) | `scripts/download_sciplex3.sh DATA_DIR` | `DATA_DIR/raw/GSE139944/` |
| all cells, raw counts | `python -I scripts/build_sciplex3_h5ad.py DATA_DIR` | `processed/sciplex3_raw_counts.h5ad` (799,317 cells x 110,983 genes, 2.4 GB) |
| A549 HDACi panel | `python -I scripts/build_panel.py --data-dir DATA_DIR` | `processed/a549_hdaci_v1.h5ad` (24,786 x 2,085, 12 MB) |

On Babel everything is already built under `/data/user_data/yixiz/sciplex3/processed/`.
The panel is small enough to copy. If you rebuild it, `build_panel.py` checks that it
reproduces the frozen spec and split in this repo, and then writes only the panel.

## Panel and split (v1, *proposed*: confirm as a team before any test run)

- **Cells:** A549, 24 h, hash-assigned. 3,773 vehicle controls plus 26 HDAC inhibitors (sci-Plex
  annotation `pathway_level_2 == "Histone deacetylation"` and "HDAC" in `target`) at
  10/100/1000/10000 nM, with 2 replicate wells per condition and about 100–300 cells per condition. Only human genes are used.
- **`X`:** `log1p(UMI / total human UMIs * 1e4)` over the 2,085 eval genes (2,000 HVGs
  fitted on control + train cells, plus the 97 program genes). `layers["counts"]` has the raw UMIs.
- **`obs`:** `condition` (`"A549|<drug>|<dose_nM>"`, e.g. `"A549|Belinostat (PXD101)|1000"`), `drug`, `dose`,
  `replicate`, `culture_plate`, `well_oligo`, `split`, …
- **`obs["split"]`:** each drug holds out **one intermediate dose** (100 or 1000 nM, both
  wells). The other three doses are `train`. 9 drugs are `val`, 17 are `test`; drugs are
  stratified by 10 µM response so both splits span potent and inactive drugs. Vehicle cells
  (`control`) may be used by every method. Details are in `configs/split_v1.json`.

Use `val` for every choice you make (λ, h, epochs, architecture). Do not touch `test`
cells for anything except the final scored runs.

## Evaluation spec (`artifacts/eval_spec_v1`, fitted on control + train cells only)

- **Program activity a(x):** the Tirosh et al. 2016 S + G2/M genes (97), computed as
  `log1p(sum_g expm1(x_g))`, i.e. the log of the program's total normalized expression (sci-Plex's proliferation
  index). The `mean_log` variant was worse in the pilot (`pilot.json`).
- **r(x) = (μ0 − a(x)) / (s0 + ε):** μ0 and s0 come from all A549 vehicle cells.
- **Thresholds:** τ_q is the q-quantile of r over controls, for q ∈ {0.90, **0.95** (primary), 0.975}, so a
  cell is a weak responder if it is no more suppressed than a fraction q of controls.
- **MMD space:** PCA(50) of the eval genes, with RBF bandwidths of 0.5/1/2 × the median pairwise distance.

If you implement the fraction loss (eq. 2), compute r(x) from `spec.program_gene_ids`,
`spec.mu0`, `spec.s0`, `spec.taus` and `spec.score_type`. This keeps the loss and the metric on the same score.

## Metrics (per held-out condition, then an equal-weight mean over conditions with a bootstrap 95% CI)

| key | compares | checks |
|---|---|---|
| `wrf_err_pp_q0.95` (+ `q0.9`, `q0.975`) | \|predicted − measured weak-responder fraction\|, in pp | the proportion at the chosen threshold (**primary**) |
| `w1_r` | 1-D Wasserstein-1 between the r(x) of real and generated cells (control-SD units) | response heterogeneity across all thresholds |
| `mmd2` | unbiased multi-scale RBF MMD² in spec PCA space | broader multigene distribution |
| `rmse_mean_all` / `rmse_mean_deg50` / `rmse_mean_program` | RMSE between real and predicted per-gene means: all eval genes / the condition's top-50 DEGs vs controls / program genes | average expression change |

## Scoring your baseline

Write one prediction file per method, seed and split:

- **Generated cells:** a `.h5ad` whose `var_names` are the spec gene ids (any order), whose `X` is in the
  same log1p(CP10K) space, and with `obs["condition"]` set for each held-out condition. Generate at least
  as many cells as were measured (`n_cells` in the split file).
- **Fractions only** (e.g. direct WRF regression): a `.csv` with columns `condition,q,wrf`.
  Only the WRF errors are reported.

```bash
python -m sciplex_eval.evaluate --pred preds_val.h5ad --method nearest_dose --seed 0 --split val --out results/nearest_dose/val_s0
python -m sciplex_eval.evaluate --reference replicate --split val --out results/replicate/val   # noise floor
```

This writes `per_condition.jsonl` and `summary.json`. In Python (e.g. for validation inside a training loop):

```python
from sciplex_eval import EvalSpec, load_panel, evaluate_predictions, summarize
spec, panel = EvalSpec.load("artifacts/eval_spec_v1"), load_panel()   # or load_panel(path)
summarize(evaluate_predictions(spec, panel, pred_adata, split="val"))["wrf_err_pp_q0.95"]
```

`python -m pytest tests` runs the unit tests, plus a check against the real panel when it is mounted.

## Reference numbers (val, 9 conditions)

| | WRF err q0.9 | q0.95 | q0.975 | W1(r) | MMD² | RMSE DEG50 |
|---|---|---|---|---|---|---|
| replicate well vs other well (noise floor) | 5.4 | 3.5 | 4.5 | 0.22 | 0.0032 | 0.20 |
| "no drug effect" (vehicle cells as the prediction) | 10.9 | 5.6 | 4.2 | 0.47 | 0.029 | 0.34 |

## Open issue: the WRF thresholds are almost insensitive with this score

At sci-Plex depth a control cell has a median of 8 UMIs in the 97 program genes, and 4.5% have 0.
Controls also mix G1 cells (low program) with cycling cells (high program). As a result:

- **τ₀.₉₇₅ is degenerate.** It falls on the point mass of cells with zero program UMIs, so even
  controls score f = 0.955, not 0.975.
- **At q = 0.95 a "responder" is any cell with ≤3 program UMIs.** For the 11 most potent HDACi at
  10 µM (train doses), WRF drops only 4 pp from controls, which equals the replicate noise (3.8 pp).
  The drug effect sits in the middle of the control distribution: the drop is 44 pp at q = 0.5 and 47 pp at q = 0.75.
- On val, the no-effect prediction is within about 2 pp of the noise floor at q = 0.95 and *beats* it at q = 0.975.

Options for the team: move the thresholds to the informative range (q ≈ 0.5–0.75), and/or use
a larger or denoised proliferation program. To try one, run
`build_panel.py --quantiles 0.5 0.75 0.9 --primary-q 0.75 --version v2`; v1 stays untouched.
