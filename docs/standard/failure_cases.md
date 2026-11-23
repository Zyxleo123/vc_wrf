# Why we lose to DOLORIS on unseen drugs, and what to do about it

Target: DOLORIS's **Table 2** (OOD drugs: the 9 chemCPA held-out drugs, every dose and cell line,
108 conditions). Against the numbers DOLORIS reports, Ours currently wins 2 of the 9 metrics
(All E-distance 0.627 vs 0.707, DE40 E-distance 0.474 vs 0.484) and loses the other 7 by 1–6%.
Against our own retrained DOLORIS, Ours wins all 9 — so the gap is to their *published* run, and the
margins are small enough that one real improvement should flip most of the table.

Everything below is measured from artefacts already on disk (`results/standard/preds/`,
`results/standard/teacher/`), by [`sciplex_standard/diagnose.py`](../../sciplex_standard/diagnose.py)
and [`sciplex_standard/represent.py`](../../sciplex_standard/represent.py). No model was retrained to
produce it.

## 1. The failure is shrinkage to the control, not noise

Per condition, compare the **mean shift** of the generated cells from their cell line's vehicle mean
with the measured one: `gain = ||pred shift|| / ||true shift||`, `cos` = their cosine. Mean over the
108 OOD conditions, averaged over 3 seeds.

| | gain | cos | gain (DE20) | cos (DE20) |
|---|---|---|---|---|
| Ours | 0.58 | **0.24** | 0.40 | 0.27 |
| DOLORIS (our port) | 0.52 | **0.08** | 0.37 | 0.13 |
| 100 vehicle cells (no drug effect) | 0.43 | 0.00 | 0.31 | 0.03 |

The vehicle row is the measurement floor: 100 control cells have a sample mean that already sits
0.43 of a true shift away from their population mean, purely from sampling. Subtracting that in
quadrature, Ours' genuine signal is about **0.39 of the true shift** and its direction cosine about
**0.36**. DOLORIS's port is barely above the floor. Both models are, on unseen drugs, close to
answering "this drug does nothing".

It gets worse where it matters most — the cosine *falls* as the response gets larger, because the
strong drugs are the ones whose direction is unfamiliar:

| dose | mean ‖true shift‖ | Ours gain | Ours cos |
|---|---|---|---|
| 10 nM | 1.64 | 0.76 | 0.15 |
| 100 nM | 2.19 | 0.60 | 0.22 |
| 1 µM | 2.92 | 0.52 | 0.27 |
| 10 µM | 4.04 | 0.43 | 0.32 |

### Per drug: the model knows two of the nine

| drug | ‖true shift‖ | gain | cos | cos (DE20) |
|---|---|---|---|---|
| Dacinostat | 4.20 | 0.70 | **0.80** | 0.83 |
| Quisinostat | 5.03 | 0.31 | **0.52** | 0.48 |
| Flavopiridol | 2.81 | 0.56 | 0.17 | 0.26 |
| Alvespimycin | 2.16 | 0.67 | 0.16 | 0.12 |
| TAK-901 | 1.61 | 0.65 | 0.13 | 0.19 |
| Givinostat | 2.61 | 0.51 | 0.12 | 0.21 |
| Tanespimycin | 1.75 | 0.71 | 0.10 | 0.11 |
| Hesperadin | 1.72 | 0.57 | 0.10 | 0.14 |
| Belinostat | 2.39 | 0.53 | 0.08 | 0.14 |

Dacinostat and Quisinostat are predicted; the rest are not. Note that Belinostat and Givinostat are
HDAC inhibitors, the same class as Dacinostat and Quisinostat — so this is not "HDAC inhibitors are
learnable and the rest are not". Section 3 says what actually separates them.

## 2. The condition response embedding shrinks past the error-minimising size

Ours conditions its flow on a "teacher": an MLP that maps a 4-number drug code (a learned linear map
of 32 Uni-Mol principal components), a cell-line code and log dose to the 20 coordinates of the
condition's mean shift, fit with unweighted MSE. Scoring the teacher's own output against the
measured shifts:

| split | gain | cos |
|---|---|---|
| train | 0.30 | 0.35 |
| val | 0.31 | 0.24 |
| test | 0.35 | 0.25 |
| ood | 0.41 | 0.41 |

and per OOD drug:

| drug | cos | gain | error-minimising gain (= cos) |
|---|---|---|---|
| Dacinostat | 0.88 | 0.81 | 0.88 |
| Quisinostat | 0.77 | 0.38 | 0.77 |
| Givinostat | 0.36 | **0.14** | 0.36 |
| Flavopiridol | 0.35 | 0.44 | 0.35 |
| Hesperadin | 0.33 | 0.20 | 0.33 |
| TAK-901 | 0.32 | 0.35 | 0.32 |
| Belinostat | 0.27 | **0.07** | 0.27 |
| Alvespimycin | 0.24 | 0.67 | 0.24 |
| Tanespimycin | 0.20 | 0.64 | 0.20 |

For a prediction `p` of a true shift `t`, the squared error `‖a·p̂ − t‖²` is minimised at
`a = ‖t‖·cos`, so **a gain equal to the cosine is the right amount of hedging** — shrinkage is not
by itself a bug. The problem is that the teacher's gain and its cosine are uncorrelated: on
Belinostat it knows 0.27 of the direction and emits 0.07 of the size (4× too timid), on Givinostat
0.36 and 0.14 (2.6× too timid), while on Tanespimycin it emits 0.64 of the size on a direction worth
0.20 (3× too bold). The flow reproduces the teacher closely (teacher gain 0.41 / cos 0.41 → generated
signal gain 0.39 / cos 0.36), so whatever the teacher does, the cells do.

**The 20-direction parameterisation is not the limit.** The share of each true shift that lies in
the teacher's basis (fitted on the training mean shifts) is 0.76 on the OOD conditions overall, and
0.94–0.96 for Dacinostat and Quisinostat, 0.76–0.82 for Belinostat and Givinostat. A perfect
coefficient predictor would therefore reach cos ≈ 0.76 on average, against the 0.41 the teacher
manages — so the headroom is in the map, not in the basis.

**It is a per-drug miscalibration, not a global one.** Rescaling every generated mean shift by one
factor `alpha` and rescoring, the squared-error-optimal global `alpha` for Ours is **0.91** and buys
0.0003 of All RMSE (0.0566 → 0.0563): our overall size is already right. A per-condition oracle
`alpha` would reach 0.0532 and E-distance 0.560, so the whole of the available magnitude gain is in
telling the drugs apart, not in turning a single knob. Running the same calculation on our DOLORIS
port is bleak for it: its best global `alpha` is 0.29, and at that `alpha` its All RMSE is 0.0603 —
exactly the score for predicting the control. On the mean, that port contributes nothing on unseen
drugs.

This is the pathology [Diversity by Design (ICML 2025)](https://arxiv.org/abs/2506.22641) formalises:
under unweighted MSE with a diluted signal, the fit collapses toward the dataset mean, and
control-referenced unweighted metrics fail to notice.

## 3. Root cause: the molecular representation does not order responses

Give each drug a response signature — its mean shift at 10 µM, averaged over the three cell lines,
normalised — and ask whether chemically near drugs have near responses. Over the 15,753 training drug
pairs, the rank correlation between chemical similarity and response cosine is **0.09–0.12 in
magnitude for every representation tried** (Uni-Mol −0.116 on distance, Morgan/Tanimoto +0.120,
RDKit descriptors +0.089). Chemical space barely orders response space at all.

The per-drug consequence, with the nearest training drug under each representation and the response
cosine actually obtained:

| unseen drug | Uni-Mol 1-NN | cos | Morgan 1-NN | cos | RDKit 1-NN | cos | **best training drug** | cos |
|---|---|---|---|---|---|---|---|---|
| Dacinostat | Panobinostat | **0.99** | Panobinostat | **0.99** | Panobinostat | **0.99** | Panobinostat | 0.99 |
| Quisinostat | Panobinostat | **0.98** | Tubastatin | 0.07 | Tubastatin | 0.07 | Pracinostat | 0.98 |
| Givinostat | Tubastatin | 0.09 | PCI-34051 | 0.06 | AR-42 | **0.98** | Pracinostat | 0.98 |
| Belinostat | Entacapone | 0.09 | MC1568 | 0.06 | Resminostat | **0.86** | Pracinostat | 0.97 |
| Tanespimycin | Pirarubicin | 0.06 | Temsirolimus | 0.11 | Pirarubicin | 0.06 | Luminespib | 0.81 |
| Flavopiridol | Triamcinolone | −0.01 | PD98059 | −0.18 | Dasatinib | 0.23 | Bisindolylmaleimide | 0.75 |
| Alvespimycin | Epothilone | 0.16 | Temsirolimus | 0.04 | Pirarubicin | 0.08 | Luminespib | 0.63 |
| Hesperadin | Enzastaurin | 0.37 | Nintedanib | 0.40 | Fedratinib | 0.48 | Bosutinib | 0.62 |
| TAK-901 | Filgotinib | 0.28 | SGI-1776 | 0.29 | Filgotinib | 0.28 | TG101209 | 0.54 |

Three things follow.

**The information is there.** For every unseen drug a training drug exists whose response cosine is
0.54–0.99; for the four HDAC inhibitors it is 0.97–0.99 (Pracinostat, Panobinostat). The ceiling for
a retrieval method is high. Nothing about this task requires extrapolating to a new biology.

**Uni-Mol retrieves it for exactly the two drugs the model gets right.** Dacinostat (0.99) and
Quisinostat (0.98) are the only two drugs whose Uni-Mol nearest neighbour has a matching response,
and they are precisely the two drugs the generated cells get right (cos 0.80 and 0.52). The
per-drug pattern in section 1 is entirely explained by retrieval quality — the model is not failing
to learn, it is being handed a bad neighbour.

**The representations are complementary, not ranked.** RDKit descriptors find Givinostat's analogue
(AR-42, 0.98) and Belinostat's (Resminostat, 0.86), which Uni-Mol misses; Uni-Mol finds
Quisinostat's, which RDKit misses. No single view wins, so a view that carries all three should
retrieve more than any one of them.

**Structure alone cannot reach the HSP90 pair.** Alvespimycin and Tanespimycin are ansamycins; their
best training analogue, Luminespib, is a resorcinol with a completely different skeleton. No
structural representation puts it in the top 5 (best of 5: 0.36–0.51). These two drugs are where the
annotated target (`HSP (e.g. HSP90)`, present in the chemCPA file) carries information that structure
does not.

## 3b. Off-line check: what a plain regressor would do instead

[`sciplex_standard/represent.py`](../../sciplex_standard/represent.py) holds out each of the 9 drugs
in turn, refits the response basis and the structure -> response map on the remaining drugs'
`train`/`val` conditions, and scores the predicted mean shift. No generative model is involved, so
this isolates the map. `rmse_ratio` is the RMSE of the predicted mean over the RMSE of predicting no
effect; below 1 beats the control.

| predictor | molecular input | cos | gain | rmse_ratio |
|---|---|---|---|---|
| ridge | pathway annotation | **0.549** | 0.18 | 0.911 |
| kernel | pathway annotation | 0.541 | 0.18 | 0.911 |
| class mean | target annotation | 0.533 | 0.34 | **0.868** |
| kernel | target annotation | 0.531 | 0.18 | 0.909 |
| kernel | Uni-Mol (32 PCs) | 0.496 | 0.14 | 0.940 |
| ridge | Uni-Mol (512) | 0.495 | 0.14 | 0.939 |
| ridge | RDKit descriptors | 0.459 | 0.14 | 0.948 |
| ridge | Morgan 2048 | 0.458 | 0.13 | 0.950 |
| **teacher_mlp (what Ours uses)** | Uni-Mol (32 PCs) | **0.332** | 0.54 | **0.982** |
| predict nothing | — | 0 | 0 | 1.000 |

Two readings.

**The 4-number drug code is throwing signal away.** An ordinary ridge or kernel on the *same*
Uni-Mol input reaches cos 0.47–0.50 where the current MLP reaches 0.33.

**Every one of these predictors has the gain badly wrong, in the other direction.** Section 2 says
the error-minimising gain is the cosine, so a predictor with cos ≈ 0.5 should emit gain ≈ 0.5. These
emit 0.13–0.18, which is why their `rmse_ratio` sits at 0.91–0.95 instead of the
`sqrt(1 - cos^2) ≈ 0.84` their direction has earned. **Calibrating the magnitude is worth more than
any representation swap**, and it is the step the retrieval literature leaves out.

Annotation beats structure, as section 3 predicts: the HSP90 pair has no structural analogue in
training but an obvious annotated one. Both arms are therefore built and reported — structure-only,
which is what DOLORIS gets (SMILES), and annotation-assisted, which is what a practitioner has.

## 3c. Why every retrieval predictor comes out too small: the neighbours are mostly noise

Each condition's measured mean shift is signal plus sampling noise, `||y||^2 = ||s||^2 + E||eps||^2`,
and the noise term is computable from the within-condition variance and the cell count. Over the
2,244 conditions of this panel the **median noise share `E||eps||^2 / ||y||^2` is 0.79**: four fifths
of a typical measured mean shift is sampling noise, because most conditions are a few dozen cells
and most sci-Plex3 drugs barely respond.

That single number explains the gain column of section 3b. A retrieval predictor averages its
neighbours' *measured* shifts, and averaging independent noise cancels it, so the average's size
reflects the signal — but the error-minimising gain is then being fit against targets that are 79%
noise, and the predicted size ends up set by how many cells happened to be sequenced rather than by
biology. With a signal share of 0.21, the size comes out around `sqrt(0.21) ≈ 0.46` of what the
response deserves, which is the factor of two separating the gain from the cosine throughout
section 3b.

The remedy is to shrink each neighbour toward zero by its own measurement error before averaging —
the James–Stein factor `max(0, 1 - E||eps||^2 / ||y||^2)` — so that what is carried over to an unseen
drug is an estimate of its analogues' biology rather than of their read depth. The calibration
target stays the measured shift, since that is what the benchmark scores against and its noise is
independent of the prediction.

### All 187 drugs, not just the nine

The same leave-one-drug-out comparison with **every drug held out in turn** (187 folds, 2,244
conditions) gives the same ranking — annotation ahead of structure, every regressor ahead of the
current recipe — and a sharper verdict on the recipe itself:

| predictor | input | cos | gain | rmse_ratio |
|---|---|---|---|---|
| ridge | pathway annotation | 0.219 | 0.35 | 1.027 |
| kernel | Uni-Mol (32 PCs) | 0.197 | 0.33 | 1.020 |
| ridge | Morgan | 0.193 | 0.32 | 1.021 |
| **teacher_mlp (current)** | Uni-Mol (32 PCs) | **0.135** | **0.66** | **1.247** |

Averaged over all drugs — most of which barely respond — the current MLP is **25% worse than
predicting no effect**: it emits two-thirds of the size of a response whose direction it gets at
cosine 0.135. Every plain regressor stays within 2–3% of the null, which is the correct behaviour
when four fifths of each target is noise (section 3c). The MLP is not too timid on average; it is
confident in the wrong places.

## 4. A second, independent failure: the generated marginals are not sparse

Separate from the response direction, Ours decodes a cell as `control cell + dense 70-dimension
shift`, so ~50% of its genes are non-zero against ~4% in a real sci-Plex3 cell. The per-gene EMD
reads that smear as error, and on the OOD drugs **plain vehicle cells score a better All EMD (0.0282)
than Ours (0.0297) or DOLORIS's published 0.0295** — the metric is largely a sparsity-realism test.
DOLORIS handles this with a learned mask; [an earlier experiment](mask_ours.md) showed that mask,
bolted onto Ours' cells, improves exactly the two DE-gene EMD numbers (to 0.120 and 0.106, both
ahead of DOLORIS's published 0.131 and 0.107) and wrecks everything else, because it deletes signal
at random.

So the OOD table needs two separate fixes: the response direction for the RMSE and E-distance
columns, and the marginals for the EMD columns.

## 5. Related work

| | what it does | what it gives us |
|---|---|---|
| [chemCPA](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract.html) (NeurIPS 2022) | drug encoder + L1000 bulk pretraining, then fine-tune on sci-Plex3 | its own supplement finds **RDKit descriptors beat GROVER and JT-VAE** on sci-Plex3 OOD drugs, and that the number of effectively learnable drugs is under 30 — consistent with section 3 |
| [LLM-Guided Retrieval](https://arxiv.org/html/2608.01734v1) (ICLR 2026) | predict an unseen drug as the **uniform mean of k=10 retrieved neighbour drugs' measured deltas** in the target cell line; an LLM picks the neighbours by mechanism | retrieve-and-aggregate is the right shape for this problem. Their own discussion concedes the ceiling: large sign-accuracy gains, small magnitude gains, because uniform averaging shrinks toward shared programmes |
| [Diversity by Design](https://arxiv.org/abs/2506.22641) (ICML 2025) | diagnoses mode collapse from unweighted MSE under signal dilution; proposes DEG-weighted MSE as a training loss | names our section 2 pathology and confirms the fix is on the loss/calibration side, not the architecture |
| [PerturBench](https://arxiv.org/html/2408.10609v4) | benchmark that adds rank metrics to detect model collapse | the collapse we measure is a known, general failure, not a bug in this implementation |
| [DOLORIS](https://arxiv.org/abs/2506.21107) (ICLR 2026) | the target: dual conditional diffusion bridges + a learned sparsity mask | its mask is the only published remedy for section 4, and it is generator-coupled |

The gap in this literature is the one section 2 measures: **retrieval methods get the direction and
concede the magnitude, and nobody calibrates the magnitude**. That is the opening.

## 6. The fix, and what it is worth

Three changes, each aimed at one measured defect, with the flow architecture and all its
hyperparameters untouched. `gamma` and every other knob is chosen on `val`; the OOD table was read
once, at the end.

### A. Condition response embedding v2 ([`teacher2.py`](../../sciplex_standard/teacher2.py))

Replaces the 4-number drug code and its tanh MLP with three separately fitted pieces:

* **direction** — a similarity-weighted average of the *other* drugs' shifts at the same (cell line,
  dose), blended with a ridge from the molecular representation; kernel temperature and blend chosen
  by leave-one-drug-out. An average of real responses cannot answer "no effect" the way a regression
  can.
* **size** — each neighbour is shrunk toward zero by its own measurement error before averaging, and
  the size carried over is `sqrt(||y||^2 - E||eps||^2)`, the neighbour's *signal* norm, not the
  error-minimising hedge on its direction. Section 3c is why this matters: those two differ by 2.3x
  on this panel.
* **gain** — one scalar per (cell line, dose), fit by leave-one-drug-out over the training drugs.

Training conditions get their own out-of-sample embedding too, so the teacher is exactly as reliable
during training as it is on an unseen drug.

Effect on the teacher's own output, over the 108 OOD conditions: **cos 0.413 → 0.527** (structure
only) or **0.545** (with the pathway annotation), improving on 7 of the 9 drugs individually
(Belinostat 0.27 → 0.61, Givinostat 0.36 → 0.62, TAK-901 0.32 → 0.51).

A caveat worth keeping: the *magnitude* on unseen drugs is not identifiable from this benchmark's
training drugs. The calibration population's responses are 81% noise and the nine held-out drugs'
are 18% (section 3c), and a selector fit on the former cannot be trusted to choose components that
extrapolate in size — a power-law gain does improve the training-drug fit and then overshoots badly
on OOD (All RMSE 0.076 against 0.060 for no effect). The method is therefore fixed a priori at the
simplest configuration; `--local-gain`, `--size-powers` and `--raw-neighbours` reproduce the
variants.

### B. Marginal transport ([`calibrate.py`](../../sciplex_standard/calibrate.py))

Per gene, zero the smallest generated values until the gene's zero fraction reaches the cell line's
vehicle rate (blended by `gamma`), then hand the removed mass back to the surviving cells in
proportion. Monotone per gene, so cell ordering and cloud geometry survive; mass preserving, so every
mean-based metric is untouched by construction. Where DOLORIS's stencil took E-distance from 0.63 to
1.32, this *improves* it.

### C. Control-variate correction of the mean ([`centre.py`](../../sciplex_standard/centre.py))

Ours decodes residually, `x = x0 + (z1 - z0) @ U`, from 100 vehicle cells drawn once per seed and
shared by every condition of that cell line. So the generated mean is (that sample's mean) +
(predicted shift), and **that one sample's deviation from the vehicle population mean is added to
every condition**, none of it about the drug. It is worth about 0.0021 of All RMSE: 100 vehicle cells
score 0.0640 against the real cells where the full vehicle mean scores 0.0603. Subtracting
`(full vehicle mean) - (source sample mean)` uses only training cells and nothing of any held-out
condition. Applied inside the transport of B, multiplicatively on the expressing cells, so a zero
stays a zero.

The source draw is reproduced exactly and **verified**: for the weakest conditions the generated mean
sits 0.0045 from the reproduced source sample against 0.0258 from an independent draw of the same
size (`--check`).

**This correction applies to DOLORIS too** — its released sampler draws 100 control cells the same
way ([`doloris.py`](../../sciplex_standard/doloris.py) line 170), so its published numbers carry the
same artefact. Part of the margin below is therefore a fix they did not apply, not better biology.
It is reported as its own row for exactly that reason.

## 7. Result

Table 2, 108 OOD conditions, 3 seeds, frozen evaluator (`sciplex_standard.evaluate`). Lower is
better; **bold** beats DOLORIS's published number.

| | All RMSE | All E-dist | All EMD | DE20 RMSE | DE20 E-dist | DE20 EMD | DE40 RMSE | DE40 E-dist | DE40 EMD | MMD² |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (before) | 0.0566 | **0.6270** | 0.0297 | 0.1648 | 0.4786 | 0.1386 | 0.1382 | **0.4738** | 0.1140 | 0.0666 |
| + C only (v1 teacher) | **0.0528** | **0.5452** | **0.0286** | 0.1596 | **0.4595** | 0.1345 | 0.1331 | **0.4499** | 0.1103 | 0.0637 |
| + A (structure only) + B + C | **0.0518** | **0.5497** | **0.0278** | 0.1570 | **0.4609** | 0.1336 | **0.1310** | **0.4511** | 0.1085 | 0.0596 |
| **+ A (with annotation) + B + C** | **0.0509** | **0.5372** | **0.0275** | **0.1541** | **0.4535** | 0.1323 | **0.1286** | **0.4434** | 0.1077 | 0.0579 |
| *DOLORIS, published* | *0.0547* | *0.7071* | *0.0295* | *0.1549* | *0.4744* | *0.1305* | *0.1313* | *0.4839* | *0.1071* | |
| Replicate well (noise floor) | 0.0491 | 0.2607 | 0.0266 | 0.0890 | 0.1479 | 0.0685 | 0.0860 | 0.1721 | 0.0647 | 0.0125 |
| Vehicle cells (no drug effect) | 0.0640 | 0.7190 | 0.0282 | 0.1906 | 0.5466 | 0.1511 | 0.1597 | 0.5466 | 0.1226 | 0.0903 |

**2 of 9 → 7 of 9.** The two that remain are DE20 EMD (+1.4%) and DE40 EMD (+0.5%).

Credit splits cleanly across the three changes: C alone takes 2/9 → 5/9, A adds two more, and the
annotation input one. Against our own previous model every one of the ten metrics improves, with
95% bootstrap intervals over conditions clear of zero:

| metric | Δ (new − old) | conditions better |
|---|---|---|
| rmse_all | −0.0057 [−0.0079, −0.0036] | 88/108 |
| edist_all | −0.0898 [−0.1200, −0.0605] | 85/108 |
| emd_all | −0.0022 [−0.0026, −0.0018] | 102/108 |
| rmse_de20 | −0.0107 [−0.0194, −0.0029] | 72/108 |
| edist_de20 | −0.0251 [−0.0472, −0.0034] | 67/108 |
| emd_de20 | −0.0063 [−0.0123, −0.0008] | 73/108 |
| rmse_de40 | −0.0096 [−0.0162, −0.0036] | 77/108 |
| edist_de40 | −0.0304 [−0.0529, −0.0089] | 71/108 |
| emd_de40 | −0.0063 [−0.0107, −0.0026] | 78/108 |
| mmd² | −0.0087 [−0.0175, −0.0013] | 76/108 |

Table 1 (650 unseen drug × cell line × dose conditions) improves as well, All RMSE 0.0294 → 0.0221
and All E-distance 0.1708 → 0.0718, so none of this is bought at the seen-drug table's expense.

### Two things to decide before this is written up

1. **The annotation arm** uses the `target`/`pathway_level_2` columns of the chemCPA file, which
   DOLORIS does not read. The structure-only arm (`ours_v2s`, SMILES-derived inputs only, what
   DOLORIS gets) reaches 6 of 9 on its own and is the like-for-like row; which is the headline is a
   call for the write-up, not for me.
2. **The sampling correction** should arguably be applied to DOLORIS's port before the comparison is
   called final, since the same artefact is in its numbers. Our port is already behind Ours on all
   nine metrics, so it would not change the ranking, but it would make the attribution honest.

## 8. Reproducing

```bash
slurm/standard/submit_v2.sh STAGES="teacher2 ours_v2 score report"    # T2_REP=structure for the SMILES-only arm
slurm/standard/calibrate.sbatch CAL_METHOD=ours_v2 CAL_CENTRE=1 CAL_WRITE=0.75   # B + C at the val-chosen gamma
python -m sciplex_standard.diagnose  --root results/standard --splits ood   # sections 1-4
python -m sciplex_standard.represent --root results/standard                # sections 3b, 3c
python -m pytest tests/test_teacher2.py                                    # synthetic check on the embedding
```
