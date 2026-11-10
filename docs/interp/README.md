# Held-out doses from adjacent doses: why the trivial baseline wins, and a fix for chemCPA

Current comparisons use MMD² and mean-expression errors; response-score and WRF metrics are obsolete.
Decisions and every number as it was produced: [`log.md`](log.md). Test tables with paired
bootstraps: [`chemcpa_test.md`](chemcpa_test.md). Code: [`sciplex_interp/`](../../sciplex_interp/)
and the opt-in chemCPA options in [`sciplex_baselines/chemcpa.py`](../../sciplex_baselines/chemcpa.py).

Names: the *condition response embedding* is the `teacher` covariate and *grouped velocity matching* is `aux.kind: mean_velocity` (formerly "average-arrow" or
"average-velocity"). Configs and variant names keep the old identifiers.

Latest clean ablations of the strongest CellFlow arm, using only MMD² and expression-space errors:
[`cellflow_ablation.md`](cellflow_ablation.md).

Condition-response embedding length ablation (10, 20, 40, and the full 78-coordinate training
basis), including frozen test comparisons: [`cellflow_embedding_length.md`](cellflow_embedding_length.md).

Matched non-latent CellFlow comparison of the retrained baseline, condition response embedding, grouped velocity matching and both:
[`cellflow_gene_factorial.md`](cellflow_gene_factorial.md).

Dose response in one number (dose gain, validation drugs, with limits): see the last entry of [`log.md`](log.md); code in
[`sciplex_interp/dose_gain.py`](../../sciplex_interp/dose_gain.py).

What the condition response embedding's 20 numbers mean, with two reportable results (the dose response is one curved path; class I
benzamides leave it in their own direction): [`condition_embedding.md`](condition_embedding.md).

## 1. The puzzle

The task: A549 cells treated with 26 HDAC inhibitors at 10, 100, 1,000 and 10,000 nM, about
100–300 cells per drug-dose condition. Each drug's one intermediate dose (100 or 1,000 nM) is
withheld: 9 drugs for validation, 17 for test. Predict the *population* of cells at the withheld
dose. The frozen evaluator scores it with:

- **MMD²**: distance between the predicted and real cloud of cells (spec PCA space, RBF kernels).
- **RMSE DEG50 / all genes / program**: error in average expression over the condition's 50 most
  changed genes, over all 2,085 genes, and over the 97 proliferation genes.
- **W1(r)**: distance between the predicted and real distributions of the proliferation score.

A trivial baseline beats every model on all of these: take the real measured cells of the same drug
at the two neighbouring *training* doses and mix them about half and half (CellFlow's
adjacent-dose mixture; test MMD² 0.016, against 0.054 for CellFlow and 0.033 / 0.049 for
chemCPA-Gauss / NB). The motivation was to use this relationship between a drug's doses inside the
models: consistency between the model's outputs for a drug's conditions, or a better dose encoding.

## 2. Why the trivial baseline wins

From [`sciplex_interp/diagnose.py`](../../sciplex_interp/diagnose.py), validation and training-internal
("inner") tasks only:

- **Its shape is right; its average is wrong.** Shifting the mixture onto the measured average of
  the withheld condition (a diagnostic cheat, never a method) cuts val MMD² from 0.020 to 0.005 and
  energy distance from 0.51 to 0.06. Cell-to-cell variety is nearly the same at neighbouring doses;
  what changes is the average expression. With the spec's kernel bandwidths, MMD² here mostly
  measures the average.
- **Re-weighting cannot fix it.** The best fixed share of the higher dose is 0.2–0.3, not ≈ 0.5
  (val MMD² 0.014). Even the best share chosen per condition reaches only MMD² 0.013 and DEG50 RMSE
  0.191, because the real average is not on the line between the neighbours: genes switch on at
  different doses. Optimal-transport interpolation and shifted single doses do no better than mixing.
- **Room left:** the noise floor of a measured average (about 220 cells) is DEG50 RMSE 0.07–0.08;
  averaging the neighbours reaches 0.21.

## 3. Can a simple network infer a condition's average from the other conditions?

Yes ([`sciplex_interp/stats.py`](../../sciplex_interp/stats.py)). The best is a small MLP "auto-decoder":
each drug gets a learned 4-number code, and the MLP maps (code, log10 dose) to the average expression
(in the top 20 principal directions of the training mean shifts). It is fit on the averages of the
78 training conditions only, and at prediction time sees only the drug's code and the withheld dose.

| predicted average, validation (9 conditions) | DEG50 | all genes | program | Pearson Δ DEG50 | inner DEG50 (26) |
|---|---|---|---|---|---|
| **MLP auto-decoder** | **0.153** | **0.038** | **0.039** | **0.70** | **0.184** |
| shared response trajectory + Hill dose curve | 0.160 | 0.042 | 0.054 | 0.68 | 0.232 |
| low-rank drug × dose completion | 0.180 | 0.040 | 0.042 | 0.61 | 0.246 |
| neighbours, best fixed weight | 0.203 | 0.051 | 0.052 | 0.64 | 0.260 |
| neighbours, log-dose weight (the mixture's average) | 0.210 | 0.055 | 0.052 | 0.63 | 0.244 |
| no drug effect (vehicle) | 0.332 | 0.062 | 0.088 | 0.02 | 0.426 |
| noise floor of the measured average | 0.069 | 0.025 | 0.034 | | 0.074 |

## 4. What chemCPA is

chemCPA is an autoencoder for single cells with a drug-and-dose "knob".

- **Encoder.** It squeezes a cell's gene values into 64 numbers meant to describe the cell's state
  *without* the drug. An adversary network tries to guess the drug from those 64 numbers, and the
  encoder is trained to defeat it.
- **Drug and dose.** Each drug is a 64-number direction computed from its chemical fingerprint
  (2,048-bit Morgan, frozen). A small "doser" network turns (drug, dose) into a scalar that scales
  that direction; vehicle gets zero. The scaled direction is added to the cell's 64 numbers.
- **Decoder.** It turns the 64 numbers into, for every gene, a *probability distribution* over that
  gene's value in the cell, not a single value (below).
- **Training.** Every training cell is encoded, its own drug-dose vector is added, and it is decoded.
  The loss rewards the decoder for giving the cell's observed values a high probability, i.e. each
  cell is rebuilt from itself. Nothing else is trained, apart from the adversary.
- **Prediction.** Take untreated control cells, encode them, add the target drug-dose vector, decode,
  and draw one value per gene. The result is a predicted population: "what these control cells would
  look like under drug X at dose d". So chemCPA is a conditional generator. Its randomness comes from
  which control cells it starts from and from the per-gene noise; it has no latent prior to sample
  from (unlike a VAE, diffusion or flow model).

### Counts versus distributions

The *data* are one number per gene per cell: its UMI count, the number of that gene's mRNA molecules
captured. The decoder outputs a *distribution* per gene because counts are noisy: two cells in the
same state can show 0 or 3 UMIs of a gene by capture and sequencing chance. Training asks that the
observed count be likely under the predicted distribution; generation draws one count from it, so a
generated cell is again one number per gene.

- **NB version** (`likelihood: nb`). For each generated cell the decoder outputs
  1. **fractions**: a softmax over 2,086 features, the 2,085 evaluated genes plus one "rest" feature
     for all other human genes' UMIs, summing to 1;
  2. **expected counts**: those fractions × the total UMI count of the control cell the prediction
     started from (treatment-induced changes in total UMIs are not modelled);
  3. **a dispersion** per feature: how noisy its count is around the expected value.

  One count per feature is drawn from the negative binomial, and the cell is normalised exactly like
  real cells: count ÷ total drawn counts (rest feature included) × 10⁴, then log1p. The rest feature
  is what makes this normalisation match the data, since real cells are normalised by their
  whole-transcriptome total, not by the 2,085 genes alone.
- **Gaussian version** (`likelihood: gauss`): a mean and a variance per gene of the normalised
  log-expression. Samples are continuous and dense, with almost no exact zeros, which is why it
  scores badly on W1(r) (and on WRF, see the main README).

## 5. Why chemCPA lost

- **The prediction path is never what is trained.** Prediction starts from *control* cells plus the
  drug vector; training only rebuilds each *treated* cell from its own encoding. If the encoder leaves
  some drug information in the 64 numbers (the adversary is not perfect), the decoder can rebuild
  treated cells from that leftover, and the drug vector never has to carry the full drug effect.
  Starting from control cells, the effect then comes out wrong.
- **Evidence: chemCPA does not fit its own training doses.** Generated from controls, its training
  conditions score MMD² 0.043 (Gaussian) / 0.066 (NB) and DEG50 RMSE 0.35 / 0.45, worse than the
  real-cell mixture on *withheld* doses (0.020 / 0.22).
- **The consistency idea was already satisfied.** chemCPA's withheld-dose prediction equals a
  log-dose mix of its own outputs at the two neighbouring doses (val MMD² 0.018 vs 0.019 Gaussian).
  It interpolates consistently, between wrong endpoints, so a consistency term between its outputs
  has nothing to fix.
- **Dose encoding was not the bottleneck.** chemCPA's dose input log1p(dose in µM) maps 10, 100,
  1,000 and 10,000 nM to 0.010, 0.095, 0.69 and 2.40, almost merging the two low doses. Evenly spaced
  log10 dose (`dose_transform: log10_nM`) changed little (val MMD² 0.0177 vs 0.0180).

## 6. The fix: population-mean consistency

`mean_loss` (opt-in) trains the prediction path directly. On every autoencoder update:

1. pick 8 training conditions (training drug-doses and vehicle);
2. push 64 control cells through chemCPA as if treated with each condition, as at prediction time;
3. compare the generated population's average with the measured average of that condition's real
   cells, and add weight × mean squared difference to the loss.

The averages are taken over expected values, so no counts are drawn and the term is differentiable.
NB compares log1p(10⁴ × average fraction) over the 2,086 features (real fraction = count ÷ total
UMIs); Gaussian compares the average normalised log-expression. Only training cells are read. The
weight was chosen on validation: 100 for Gaussian, 10 for NB (100 is too strong for NB).

**In-sample fit** (the 78 training conditions, generated from controls) improves about 3.7×:

| | MMD² | DEG50 RMSE | energy distance |
|---|---|---|---|
| chemCPA-Gauss: original → + mean consistency | 0.043 → **0.012** | 0.346 → **0.193** | 1.16 → **0.29** |
| chemCPA-NB: original → + mean consistency | 0.066 → **0.018** | 0.448 → **0.225** | 1.91 → **0.46** |

**Test** (17 conditions, 3 seeds, scored once for the validation picks; frozen evaluator, same
generation as the original runs):

| test | MMD² | DEG50 | all genes | program | W1(r) |
|---|---|---|---|---|---|
| **chemCPA-Gauss + mean consistency** | **0.0223** | **0.238** | **0.061** | **0.059** | 0.574 |
| chemCPA-Gauss (original) | 0.0334 | 0.308 | 0.071 | 0.072 | 0.601 |
| **chemCPA-NB + mean consistency** | 0.0300 | 0.290 | 0.068 | 0.068 | **0.385** |
| chemCPA-NB (original) | 0.0494 | 0.393 | 0.080 | 0.102 | 0.589 |
| CellFlow | 0.0538 | 0.284 | 0.092 | 0.082 | 0.419 |
| adjacent-dose mixture (real neighbour cells; not a model) | 0.0161 | 0.198 | 0.058 | 0.060 | 0.213 |

Paired condition bootstrap, Δ = ref − new (> 0: new better), 95% CI, conditions improved:

- **Gaussian + mean consistency vs CellFlow:** MMD² +0.032 [+0.016, +0.049] (16/17), DEG50
  +0.046 [+0.018, +0.072] (16/17), all genes +0.031 [+0.020, +0.042], program +0.023 [+0.008, +0.038];
  W1(r) worse, −0.155 [−0.239, −0.065].
- **vs original chemCPA:** DEG50 better for both (Gaussian +0.069 [+0.013, +0.151], 14/17; NB +0.103
  [+0.016, +0.207]); program better for both; NB also W1(r) +0.204 [+0.041, +0.378]. MMD² is lower by
  about a third for both, but neither CI excludes zero.
- **vs the adjacent mixture:** still worse (Gaussian DEG50 −0.041 [−0.080, −0.005]). On validation the
  Gaussian variant had edged ahead; validation was optimistic, as for the original (0.018 → 0.033).
- A second sampling seed reproduces the test numbers.
- CellFlow's checkpoints were chosen by a WRF-based rule and chemCPA's by validation MMD², which
  favours chemCPA on MMD². Original chemCPA-Gauss already beat CellFlow on test MMD², but not on DEG50.

## 7. How this relates to the motivation, and what is generative

The fix came from testing the motivation, but it does not itself link a drug's doses: it fits each
training condition at the population level. The adjacency is used only implicitly: once the
training doses are right, chemCPA's smooth dose function interpolates between correct endpoints.

| method | the neighbouring doses' measurements | generative |
|---|---|---|
| adjacent-dose mixture | resampled as the prediction | no |
| mixture shifted to a predicted average (val MMD² 0.009; not scored on test) | resampled as the prediction | no |
| MLP auto-decoder (average only) | training data only | predicts an average, not cells |
| chemCPA + mean consistency | training data only | yes: generates from control cells |

The withheld dose's own cells are never used, except in the diagnostic "oracle" shifts in §2.

The natural next step is to use the drug-dose relationship explicitly: add the MLP auto-decoder's
predicted average at each withheld dose as a second mean-consistency target. The MLP is learned from
the drug's other doses on training data only, so the generator would still use no neighbour cells at
prediction time.

## Reproduce

Everything runs as SLURM jobs from `lov4`; the submitter only calls `sbatch`, writes one env file per
submission to `results/interp/submissions/`, and prints every `sbatch` line. Defaults are in
[`slurm/interp/defaults.sh`](../../slurm/interp/defaults.sh).

```bash
slurm/interp/submit.sh STAGES=diagnose                                        # §2, §5 diagnostics
slurm/interp/submit.sh STAGES="stats populations" TASK_SETS="inner val"       # §3 and the hybrids
slurm/interp/submit.sh STAGES="chemcpa_train chemcpa_eval" CHEMCPA_TASK_SETS=val \
  VARIANTS="gausslog_v1 gaussmean_l10 gausslogmean_l10 gaussmean_l100 gausslogmean_l100 \
            nblog_v1 nbmean_l10 nblogmean_l10 nbmean_l100 nblogmean_l100"     # §6 sweep (val only)
slurm/interp/submit.sh STAGES="chemcpa_test chemcpa_compare" EVALS="val 0,test 0,test 1" \
  VARIANTS="gaussmean_l100 nbmean_l10"                                        # §6 test, once
```

New chemCPA configs live in [`configs/interp/`](../../configs/interp/) and train into
`results/interp/chemcpa/`, apart from `configs/chemcpa_*.json`, so the WRF analysis registry is
unaffected. The original runs are read from `results/chemcpa/`.
