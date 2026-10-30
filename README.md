# sci-Plex3 held-out-dose prediction: data + shared evaluation

Shared data preparation and metrics for the response-fraction regularization project.
Every baseline (nearest-dose, adjacent-dose mixtures, WRF regression, scVIDR, FM, FM+MMD,
FM+fraction loss, ...) is scored by the same code against the same frozen spec:
`sciplex_eval` + `artifacts/eval_spec_v1`.

## chemCPA baseline

`sciplex_baselines` integrates the [paper-era chemCPA model](https://github.com/theislab/chemCPA/tree/a4a4ded0c3b949c64ff1ea51033be1b7c301c36b)
with the frozen panel and evaluation code. This **chemCPA-Morgan** adaptation uses
2048-bit radius-2 chemical fingerprints, a learned perturbation encoder and amortized
dose network, and adversarial basal-state disentanglement. It trains from scratch on
train/control cells. The baseline has no WRF auxiliary loss; a variant trained with the
proposal's response-fraction loss is [below](#chemcpa-nb--response-fraction-loss).
See [source provenance and adaptations](sciplex_baselines/vendor/chemCPA_NOTICE.md).
Result tables, figures and the sweep log are in [`docs/chemcpa/`](docs/chemcpa/README.md).

Two likelihoods are trained with otherwise identical [settings](configs/chemcpa_nb_v1.json):

- **`gauss_v1`**: the paper-era Gaussian decoder over log1p(CP10K). Its samples are dense,
  with no zero or low-UMI program cells. The v1 thresholds fall exactly on those cells, so
  this variant predicts WRF = 100% for every condition. It is reported as the upstream
  likelihood, not as a usable WRF predictor.
- **`nb_v1`** (primary): CPA's negative binomial over raw UMIs. The decoder predicts gene
  proportions, which are multiplied by the source control cell's library size. One extra
  feature holds all other human UMIs, so generated counts are normalized by their
  generated total exactly as real cells are. Treatment-induced library-size shifts are
  not modelled (treated medians are 0 to 20% above controls).

Everything runs as SLURM jobs. The submitter only calls `sbatch`. From `lov4`:

```bash
uv pip install -r requirements-chemcpa.txt          # once
slurm/chemcpa/submit.sh                             # data -> train (GPU array) -> evaluate (CPU array) -> report (CPU)
slurm/chemcpa/submit.sh REBUILD_DATA=1              # also set processed/ aside and rebuild it from the GEO download
slurm/chemcpa/submit.sh SKIP_DATA=1 VARIANTS=nb_v1 SEEDS="0 1 2 3 4"
# a validation-only sweep, compared in results/chemcpa/sweep1_val_s0/
slurm/chemcpa/submit.sh SKIP_DATA=1 EVALS="val 0" COMPARE_NAME=sweep1 VARIANTS="nbwrf_l10_h0.5 nbwrf_l100_h0.5 nb_v1"
```

Defaults (data directory, variants, seeds, evaluations) live in
[`slurm/chemcpa/defaults.sh`](slurm/chemcpa/defaults.sh). Override them as `VAR=value`
arguments. Each submission writes its resolved configuration to
`results/chemcpa/submissions/<timestamp>.env` and passes that file to every job.
Logs go to `results/chemcpa/logs/`, and every log starts with the host, GPU and
resource request.

| stage | partition | does |
|---|---|---|
| `data.sbatch` | cpu | verifies the GEO download checksums, builds the raw-count h5ad and panel (skipping outputs that exist); `build_panel.py` refuses to write unless it reproduces the frozen spec and split |
| `train.sbatch` | legacy, RTX 2080 Ti (gpu15 excluded: tasks hung there in CUDA start-up) | one task per (variant, seed): 100 epochs; keeps the checkpoint with the lowest validation MMD² |
| `evaluate.sbatch` | cpu | generates held-out-dose cells from permitted controls; scores val, test, and test with a second sampling seed |
| `report.sbatch` | cpu | per-variant reports and a cross-variant comparison, from saved artefacts only |

Training and evaluation skip finished work, so a job that hits its time limit is
simply resubmitted. A finished seed trained with a different config is refused, not
reused. Each seed saves its selected checkpoint, history, configuration, versions,
input hashes and chemical structures. Each evaluation saves the generated `.h5ad`
cells, shared per-condition and per-well metrics, and control-reconstruction
diagnostics. Reports add seed variability, a condition bootstrap, vehicle and
replicate references, CSV tables and figures. They also add a **supplementary
threshold-sensitivity** table: WRF error at control quantiles q = 0.1 … 0.975, using
the same r(x) and τ rule as the frozen spec. It is not used for model selection.
The bootstrap does not represent uncertainty across independent biological experiments.

`--generation mean` (evaluation and reporting) gives a separately labelled decoder-mean
diagnostic from the same checkpoints. No posthoc response calibration is performed.
The model verifies the frozen condition split and well isolation, uses only permitted
controls for counterfactual generation, and refuses a mismatched panel/spec at reload.
`python -m pytest tests` covers both likelihoods and the fraction loss (training updates,
checkpoints, resume, NB normalization, the analytic NB expectation against count
sampling, molecular conditioning, held-out-expression isolation).

### chemCPA results (spec v1, 3 seeds, submission `20261030-165655`)

Test split (17 withheld-dose conditions). Model rows are mean ± seed SD; lower is better.

| | WRF err q0.9 | **q0.95** | q0.975 | W1(r) | MMD² | RMSE DEG50 | RMSE program |
|---|---|---|---|---|---|---|---|
| chemCPA-NB | 14.3 ± 1.0 | **7.1 ± 0.5** | 5.6 ± 0.4 | 0.59 | 0.049 | 0.393 | 0.102 |
| chemCPA-Gauss | 19.4 ± 0.1 | **9.3 ± 0.0** | 7.4 ± 0.0 | 0.60 | 0.033 | 0.308 | 0.072 |
| vehicle cells as the prediction | 9.9 | **4.9** | 3.5 | 0.48 | 0.059 | 0.436 | 0.099 |
| replicate well (noise floor) | 6.9 | **3.8** | 2.6 | 0.27 | 0.008 | 0.245 | 0.086 |

Validation (9 conditions) agrees: NB 7.4 ± 0.3, Gaussian 9.5, vehicle 5.7, replicate 3.5 pp at q0.95.
A second sampling seed changes the test numbers by ≤ 0.05 pp. Reports, per-condition and
per-well tables and figures are in `results/chemcpa/{nb_v1,gauss_v1}/report_*` and
`results/chemcpa/compare_*`.

- **Neither variant beats the no-drug-effect prediction on WRF.** NB chemCPA's predicted
  fractions span only 95.9–98.5% while the measured fractions span 74.3–97.2%. Across conditions,
  predicted and measured WRF have Pearson r = 0.02.
- **Gaussian chemCPA: good average expression, wrong fractions.** It beats vehicle on
  DEG50 RMSE (0.31 vs 0.44) and MMD², yet every generated population is 100% weak
  responders.
- **NB chemCPA: the control tail is lost.** Reconstructing controls with no drug gives
  WRF 97.8–98.9% instead of 95.0%, with W1(r) 0.18–0.25. Per-gene means are fine (program
  RMSE 0.02–0.03) and MMD² is 0.001, below the replicate floor. The autoencoder smooths
  away the low-program (G1-like) cells that the v1 threshold counts.
- **Threshold sensitivity** (supplementary, from the same predictions): at mid thresholds
  (q 0.5–0.7) Gaussian chemCPA tracks the shift better than vehicle (12–15 vs 22–24 pp
  error). NB stays close to vehicle at every q. Both remain far above the replicate floor.

### chemCPA-NB + response-fraction loss

[`sciplex_baselines/fraction_loss.py`](sciplex_baselines/fraction_loss.py) implements the
proposal's loss (eqs. 2–3) for count generators. A config's `fraction_loss` block adds
λ · mean over sampled training conditions and spec τs of (f̃(generated) − f̃(measured))² to
chemCPA's autoencoder objective. Here f̃ is the mean of σ((τ − r)/h). Measured targets come
from training doses and vehicle only. For generated cells, the indicator is averaged
exactly over each cell's negative-binomial program-UMI distribution (program total
moment-matched to one NB, library = source cell's), so the loss is differentiable
without sampling counts.

The protocol was fixed before any sweep result
([sweep log](docs/chemcpa/wrf_sweep_log.md), with every configuration tried):

- Iterate on validation only, keeping the baseline's checkpoint rule (minimum validation MMD²).
- Count it as positive if validation WRF error at q0.95 falls ≥ 1 pp below chemCPA-NB, with
  MMD² and DEG50 RMSE no more than 10% worse.
- Score test once, for the single chosen configuration.

Round 1 (λ ∈ {10, 100, 1000} × h ∈ {0.1, 0.5}, 3 seeds each) produced exactly one
configuration meeting all criteria: **λ = 10, h = 0.5**
([config](configs/chemcpa_nbwrf_l10_h0.5.json)). Validation WRF was 4.95 ± 0.04 vs
7.44 ± 0.27, MMD² 0.025 vs 0.027, DEG50 0.339 vs 0.322. Iteration stopped there.

Test (17 conditions, scored once):

| | WRF err q0.9 | **q0.95** | q0.975 | W1(r) | MMD² | RMSE DEG50 | RMSE program |
|---|---|---|---|---|---|---|---|
| chemCPA-NB + WRF loss | 7.9 ± 0.9 | **5.3 ± 0.5** | 4.1 ± 0.2 | 0.51 | 0.049 | 0.420 | 0.135 |
| chemCPA-NB | 14.3 ± 1.0 | **7.1 ± 0.5** | 5.6 ± 0.4 | 0.59 | 0.049 | 0.393 | 0.102 |
| vehicle cells as the prediction | 9.9 | **4.9** | 3.5 | 0.48 | 0.059 | 0.436 | 0.099 |

- **The loss helps chemCPA on held-out doses.** All three seeds beat all three baseline
  seeds (4.7–5.6 vs 6.9–7.7 pp), and 12 of 17 conditions improve. Predicted fractions now
  track measured ones (Pearson r 0.52 vs 0.02). With 17 conditions the paired condition
  bootstrap is not conclusive: −1.9 pp, 95% CI [−4.7, +0.7].
- **It does not beat the no-drug-effect prediction on test** (+0.4 pp, CI [−2.4, +2.8]),
  although it did on validation.
- **Costs:** program-gene mean RMSE 0.135 vs 0.102, and DEG50 RMSE 7% worse. In the
  threshold-sensitivity curve, error is near the replicate floor at q ≥ 0.85 but 16–17 pp
  at q ≤ 0.4, worse than every other method. The loss matched the supervised fractions
  partly by distorting the rest of the response-score distribution.

**Round 2** was run after seeing the round-1 test result, and was motivated by the same
distortion on validation (21 pp at q = 0.1). It supervised 13 control quantiles,
q = 0.1 … 0.975, instead of the three spec τs (`"quantiles"` in the config). One criterion
was added: mean validation WRF error over those 13 thresholds no worse than chemCPA-NB's.
The best passing configuration was **λ = 10, h = 0.25**
([config](configs/chemcpa_nbwrfgrid_l10_h0.25.json)), with validation q0.95 4.78 ± 0.75 and
a 13-threshold mean of 9.3 vs 15.9. Its single test run, against the same baselines
("13-τ mean" = mean WRF error over the 13 thresholds):

| test | WRF err q0.95 | 13-τ mean | W1(r) | MMD² | RMSE DEG50 | RMSE program |
|---|---|---|---|---|---|---|
| round 2: 13 τs, λ=10, h=0.25 | 6.6 ± 0.2 | **9.9** | **0.44** | **0.039** | 0.404 | 0.145 |
| round 1: 3 τs, λ=10, h=0.5 | **5.3 ± 0.5** | 12.9 | 0.51 | 0.049 | 0.420 | 0.135 |
| chemCPA-NB | 7.1 ± 0.5 | 16.2 | 0.59 | 0.049 | 0.393 | 0.102 |
| vehicle cells as the prediction | 4.9 | 15.1 | 0.48 | 0.059 | 0.436 | 0.099 |
| replicate well (noise floor) | 3.8 | 8.0 | 0.27 | 0.008 | 0.245 | 0.086 |

- **At the frozen primary threshold, round 2 gives no clear gain** over chemCPA-NB (−0.6 pp,
  95% CI [−2.7, +1.6]). It is worse than round 1 (+1.3 pp, CI [+0.4, +2.4]). Validation
  overstated it (4.8 → 6.6).
- **Across the whole response distribution it is the best model:** best 13-τ mean, W1(r) and
  MMD² of all models and both references except the replicate floor. Error at q 0.5–0.8 is
  10–11.5 pp, against 20–24 for vehicle, about at the replicate floor. Predicted-vs-measured
  WRF correlation at q0.95 is 0.63.
- **The trade-off is real:** three supervised τs do best at the primary threshold, and many
  τs do best everywhere else. Because the v1 primary threshold is nearly insensitive (see
  the open issue below), this argues for settling the threshold question before the FM
  experiments.
- Iteration stopped after two rounds. Any further round would be tuned with knowledge of
  two test results.


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
On the autonlab cluster it is under `/zfsauton/scratch/yixiz/sciplex3/processed/`, rebuilt
from GEO by `slurm/chemcpa/data.sbatch`. The full rebuild takes about 10 minutes on a CPU node
(job 91130 reproduced the earlier build exactly), which is quicker than a transfer. If you rebuild it, `build_panel.py` checks that it
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
