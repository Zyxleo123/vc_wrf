# What the condition response embedding's 20 numbers mean

2026-11-06. Code: [`sciplex_interp/teacher_viz.py`](../../sciplex_interp/teacher_viz.py). Figures and tables:
`results/interp/figures/teacher/`. Run log: the last entry of [`log.md`](log.md).

**Two results are marked reportable** (sections 2 and 3). Everything in section 4 is for internal use: real but weak, single-condition,
technical, or unexplained.

## 1. What was asked, and what the embedding is

The *condition response embedding* (code name `teacher`, file `results/interp/teacher/autodec.npz`) is the extra input that CellFlow arm
`z70_lr3e4_mv10_t` receives for each condition. A small network (the auto-decoder of [`stats.py`](../../sciplex_interp/stats.py)) gives each drug a
learned 4-number code. From that code and the log dose, it predicts how far the condition's *average* expression moves from vehicle along each of
the 20 main directions of change among the 78 training conditions. Those 20 numbers are the embedding. Each one is a weighted sum of genes with fixed
weights, so it can be read directly in genes.

The questions: (a) do related conditions get related vectors, whether the same drug at another dose or a biologically similar drug? (b) does each
coordinate mean something?

What was used: the saved embedding (a refit reproduces it exactly) and the measured means and cells of the training and validation conditions.
Test conditions appear only as the embedding's prediction from their labels, so no test measurement is read. At training doses the embedding equals
the measured means (r ≥ 0.97 on every coordinate). Most of what follows is therefore a property of the data, and every claim is checked on the
measured means as well.

## 2. Reportable: the dose response is one-dimensional, a single curved path whose bend is made of mid-dose genes

**Claim.** The 26 HDAC inhibitors do not form clusters. Every drug's average response moves along the same curved path out of vehicle, and drugs
differ mainly in how far along it they travel, not in direction. One smooth curve (each coordinate as a degree-4 polynomial of coordinate 1) holds
**93%** of the measured condition means' variance across the 20 coordinates, and 92% for the embedding.

**What the coordinates mean.**
- **Coordinate 1** is how far along the path a condition is: overall response strength. It holds 84% of the measured squared distance from vehicle.
  Proliferation genes fall along it (Hallmark G2-M checkpoint z = −8.0, E2F targets z = −7.1), while sterol genes and MALAT1 rise.
- **Coordinate 2** is the bend, not a second independent effect: 93% of it is a smooth function of coordinate 1. In genes, it separates two kinds of
  response along the path.

Measured change from vehicle (log expression), median over the training and validation conditions in each stretch of the path:

| position on the path (coordinate 1) | ≤ 0.5 | 0.5–1.5 | 1.5–3 | 3–6 | 6–10 | 10–14 |
|---|---|---|---|---|---|---|
| conditions | 51 | 8 | 5 | 7 | 3 | 13 |
| CYP24A1 | +0.07 | +0.84 | +0.89 | +0.32 | −0.26 | **−1.28** |
| IGFL2-AS1 | +0.03 | +0.38 | +1.16 | +1.01 | +1.18 | +0.30 |
| TFPI2 | −0.01 | +0.24 | +1.20 | +1.33 | +1.33 | +0.74 |
| MKI67 | −0.09 | −0.49 | **−0.79** | −0.79 | −0.76 | −0.76 |
| KRT81 | −0.07 | −0.59 | −0.85 | −1.02 | −1.02 | −1.22 |

The first three genes switch on part way and switch off again at full response; CYP24A1 ends below vehicle. Proliferation genes (MKI67, KRT81) drop
early and stay down. The bend in the path is the trace of those mid-dose genes.

**Evidence that it is real.**
- A condition's two replicate wells agree on both coordinates. On axes refit without the condition's group, r = 0.98 (95% CI 0.96–0.99) and
  0.93 (0.89–0.96). With condition labels shuffled, the largest of 400 values (20 shuffles x 20 coordinates) is 0.23.
- The curve and the gene table come from measured means, with no model involved.

**Why it matters for this study.**
- *Closeness is about stage, not drug.* At 1 µM a condition's nearest neighbour is the same drug for only 1 of 17 conditions. A drug part way
  through its response sits beside other drugs at the same stage.
- *It pictures why the adjacent-dose mixture gets the average wrong.* Mixing the cells of the two neighbouring doses lands on the straight line
  between them, which cuts across the bend and misses genes that peak in between. This matches the earlier diagnosis in [README](README.md)
  section 2 ("genes switch on at different doses"). It was not measured again here.

**What not to claim.** Any one-dimensional process whose genes peak at different points produces an arch like this in a PCA. Report "a
one-dimensional response with mid-dose-peaking genes", not "two independent programs".

**Figures:** `1_map.png` (the path, with the curve dashed) and `10_path_genes.png` (the coordinate 2 genes against position on the path).

**Suggested wording.** "Across 26 HDAC inhibitors, average responses lie on one curved path out of vehicle: a single smooth curve holds 93% of the
variance of the measured condition means. Drugs differ in how far they travel along it. The bend is carried by genes that rise at intermediate
response and fall at full response (CYP24A1: +0.9, then −1.3), while proliferation genes fall early and stay down (MKI67 −0.8)."

## 3. Reportable: class I-selective benzamides leave the shared path in their own direction

**Claim.** Take each drug's vector at 10 µM and subtract the shared curve at the same point along the path ("off-path"). The four drugs pushed
furthest along coordinate 5 are exactly the four class I-selective benzamides. No broad inhibitor comes close.

| drug (10 µM) | selectivity | off-path, coordinate 5 (measured) | embedding |
|---|---|---|---|
| Entinostat | class I | **−2.04** | −2.20 |
| Mocetinostat | class I | **−2.02** | −2.18 |
| Tacedinaline | class I | **−1.05** | −1.13 |
| Tucidinostat | class I | **−0.91** | −1.03 |
| Abexinostat | broad (most extreme of 13) | −0.57 | −0.57 |

Measurement noise on this coordinate is about 0.1 (standard error of a condition mean, 0.08–0.12).

**What it means in genes.** At matched overall strength (broad-inhibitor conditions of any dose at the same position on the path), measured change
from vehicle:

| gene | Entinostat + Mocetinostat | 8 broad conditions, same strength | Tacedinaline + Tucidinostat | 6 broad conditions, same strength |
|---|---|---|---|---|
| AGR2 | **+0.54** | −0.41 (range −0.60 to −0.26) | **+0.50** | −0.12 (−0.27 to +0.16) |
| PCDH9 | **−0.62** | −0.21 (−0.50 to −0.09) | **−0.35** | +0.25 (0.00 to +0.45) |
| KRT8 | **+0.62** | −0.10 (−0.54 to +0.07) | +0.56 | +0.40 (+0.07 to +0.71) |
| KRT18 | **+0.50** | −0.44 (−0.85 to −0.22) | +0.42 | +0.26 (−0.03 to +0.60) |

All four class I drugs raise AGR2 and lower PCDH9 beyond every strength-matched broad inhibitor. The two stronger ones (Entinostat, Mocetinostat)
also raise the epithelial keratins KRT8 and KRT18, which broad inhibitors at the same strength lower. In plain words: class I inhibitors push A549
cells toward an epithelial/secretory program that broad inhibitors suppress.

**Evidence that it is real.**
- Coordinate 5 replicates across wells: r = 0.73 (95% CI 0.52–0.84; with shuffled labels the 99th percentile is 0.18).
- It does not follow sequencing depth (r = −0.10 between a condition's wells).
- The embedding predicts it well at held-out validation doses (r = 0.92).
- The embedding never sees drug class or structure. The separation comes from expression alone, and the measured means show it independently of
  the embedding.

**Caveats.**
- Four drugs. Coordinate 5 was noticed *because* the class I drugs stood out, so "the top 4 of 26 are the 4 class I drugs" (about 1 in 15,000 by
  chance) is a post-hoc figure, not a p-value. The pre-specified test among the 15 responders, with the strength difference removed, gives
  rho = −0.36 (p = 0.09) for measured means and −0.37 (p = 0.11) for the embedding.
- Class I selectivity and benzamide chemistry coincide in these drugs. Binding kinetics also differ between benzamides and hydroxamates. The data
  cannot say which of these is the cause.
- Coordinate 5 is largely built by these drugs: three conditions hold 58% of its squared length. It also carries some well-to-well plate variation
  (p = 0.003), as do coordinates 1–3, but the class I shift is present in both wells of each condition.
- Tacedinaline and Tucidinostat respond weakly overall (coordinate 1 ≈ 1.4 at 10 µM), so their part rests on smaller changes.

**Figure:** `11_off_path.png`: every drug's off-path distance at 10 µM, with the two largest coordinates named.

**Suggested wording.** "After removing how far each drug travels along the shared path, the four drugs that deviate most are exactly the four class
I-selective benzamides (off-path −0.9 to −2.0 at 10 µM against at most −0.6 for 13 broad inhibitors; noise ≈ 0.1). At matched overall strength they
raise AGR2 and lower PCDH9, opposite to broad inhibitors, and the two stronger ones raise the epithelial keratins KRT8 and KRT18. The model input that
encodes this never sees drug class or structure."

## 4. Other findings (not for the report)

**Which coordinates carry something real.** Test: a condition's two replicate wells, on axes refit without its group of conditions (5 groups x 5
repeats). Coordinates where they agree (r ≥ 0.5): 1 (0.98), 2 (0.93), 3 (0.74), 5 (0.73), 9 (0.60), 11 (0.57), 10 (0.54). The other 13 fall below,
with 95% CIs from 0.02–0.31 (coordinate 14) to 0.22–0.63 (coordinate 6). So the input carries about five real numbers, not twenty.

| coordinate | reading | status |
|---|---|---|
| 1 | response strength | real (section 2) |
| 2 | bend of the path | real (section 2) |
| 3 | off-path; 3 of the 4 most negative drugs at 10 µM are class I; 55% a function of coordinate 1 | real |
| 4 | CUDC-907 (HDAC + PI3K) at 10 µM alone: lipid-synthesis genes about twice as high as with other strong inhibitors (FASN +2.37 vs +0.5 to +1.5) | built on 1 condition (76% of its squared length), so the test cannot check it |
| 5 | class I direction | real (section 3) |
| 6, 14 | sequencing depth: r 0.69 and 0.65 between a condition's wells, 0.64 and 0.52 across vehicle cells | technical; below 0.5 |
| 7 | mostly a per-drug offset (drug explains 83% of it on the dose grid): PCI-34051 (HDAC8-selective) and ITSA-1 | r 0.42, below 0.5 |
| 9, 11 | drug-specific (CUDC-907; Mocetinostat) | real, not interpreted |
| 10 | plate-linked (plate explains 44% of within-condition well variation, p < 0.001; mitochondrial genes 17% of its weight) | real but technical |
| others | — | do not replicate |

**Close conditions are close, but dose dominates.** Nearest neighbour is the same drug for 20% of measured means (labels shuffled within dose: at
most 7%), and 24% once position on the path is removed. Adjacent doses 10 → 100 nM sit about as far apart as two wells of one condition (median 0.81
vs 0.69–0.80). Going from 1 to 10 µM moves 4.9. The embedding exaggerates drug identity (30%; 44% off the path), partly by construction, because
it shares one code across a drug's doses.

**Drugs with no response still keep a fingerprint.** Among the 9 drugs that barely leave vehicle (3 fatty acids, 5 selective inhibitors, ITSA-1),
the nearest neighbour is still the same drug for 43% of measured means (chance at most 27%, p < 0.001). Shared culture plates do not explain it
(same drug, no shared plate: median distance 0.81, against 1.13 for different drugs). Cause untested.

**Drug class mostly reduces to strength.** Across all 26 drugs, same-class drugs are closer (rho = −0.52). After the gap in coordinate 1 is removed
it becomes +0.14 (p = 0.08). Broad inhibitors respond strongly, class I partly, and selective inhibitors and fatty acids not at all.

**Chemical similarity is a weak signal.** Among the 13 broad inhibitors, more similar structures (Tanimoto, Morgan fingerprints) are slightly closer:
rho −0.24 to −0.29, p 0.02–0.05, in both the embedding and the measured means. These come from about 90 tests with no correction. The embedding's
own off-path class separation (rho −0.25, p 0.03) is absent from the measured means (0.00), so it is not a property of the data.

**UMAP adds little.** UMAP of the embedding draws the same path. It keeps neighbours well (trustworthiness 0.97), but so does a plain plot of
coordinates 1 and 2 (0.93). Its small islands of weak drugs are about as far from vehicle as two wells of one condition.

## 5. Methods

- **Real coordinate test.** For each condition, its two wells' mean shifts (each against its own replicate's vehicle) are projected on axes fit
  without that condition's group of conditions (5 groups, 5 random splits, r averaged). Each refit axis takes its sign from the original axis.
  Calibrated on pure noise at this size (78 conditions, 20 axes): mean r 0.00, 99th percentile 0.16–0.21, no axis at 0.5. Two rejected versions:
  axes fit on the same cells give r 0.74–0.94 on noise. Refitting without one condition at a time lets 1–3% of noise axes pass 0.5, because the late
  axes are nearly tied and dropping one condition reorders them. A first "signal share" measure (variance beyond cell sampling) passed 18 of 20. It
  is kept in `axes.csv` for the record only, because it also counts technical variation between wells.
- **Path and off-path.** Each coordinate is fit as a degree-4 polynomial of coordinate 1, separately for the embedding (its 7-dose grid plus
  vehicle) and for the measured means plus vehicle. Off-path = coordinates minus the curve at the same coordinate 1.
- **Nearest neighbours.** Euclidean distance over the 20 coordinates (or the off-path coordinates). The null shuffles drug labels within each dose,
  5,000 times.
- **Drug pairs.** Spearman correlation between pair distance at 10 µM and same class, same structure or Tanimoto similarity, with drugs permuted
  5,000 times. "Strength removed" residualises the distance ranks on the ranks of the coordinate 1 gap.
- **Classes.** Selectivity is hand-assigned from vendor datasheets and general pharmacology (`SELECTIVITY` in the code), since the panel lists
  "HDAC" for 23 of 26 drugs. Sources include [Droxinostat](https://www.caymanchem.com/product/23869), [ITSA-1](https://www.abcam.com/en-us/products/biochemicals/itsa-1-hdac-activator-ab144535),
  [M344](https://www.caymanchem.com/product/13174), [TMP195](https://dcchemicals.com/product_show-TMP195.html),
  [Tubastatin A](https://dcchemicals.com/product_show-Tubastatin_A.html) and [PCI-34051](https://www.selleckchem.com/subunits/HDAC8_HDAC_selpan.html).
  Structure class (hydroxamate, benzamide, carboxylate) comes from GEO's SMILES by substructure.
- **Gene sets.** MSigDB Hallmark 2020 and GO Biological Process 2023, Enrichr text downloads of 2026-11-06 (`/zfsauton/scratch/yixiz/sciplex3/raw/genesets`).
  Rank test of each set's loadings within the panel's 2,085 genes, Benjamini-Hochberg within each coordinate and library.

## 6. Reproduce

```
slurm/interp/submit.sh STAGES="teacher_viz_tests teacher_viz_collect teacher_viz_draw"   # tests -> tables -> figures (CPU)
slurm/interp/submit.sh STAGES=teacher_viz_draw                                          # redraw from the saved tables only
```

Tests: [`tests/test_teacher_viz.py`](../../tests/test_teacher_viz.py) (synthetic data with known answers, including the noise calibration).
Tables written by `collect`: `axes.csv` (one row per coordinate, every check above), `vectors.csv` / `measured.csv` (coordinates and off-path
values), `gene_means.csv`, `neighbours.csv`, `similarity.csv`, `pairs.csv`, `genesets.csv`, `genes.csv`, `meta.json`.
