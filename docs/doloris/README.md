# DOLORIS on the A549 HDACi held-out-dose panel

DOLORIS (Chi et al., *Dual Conditional Diffusion Implicit Bridges with Sparsity Masking Strategy for
Unpaired Single-Cell Perturbation Estimation*, ICLR 2026; arXiv 2506.21107, earlier titled "Unlasting")
is retrained here on this project's panel, split and evaluator. All methods are also scored with
DOLORIS's own metrics (RMSE, E-distance, EMD on all genes, DE20 and DE40). Code: [`sciplex_doloris/`](../../sciplex_doloris/),
jobs: [`slurm/doloris/`](../../slurm/doloris/), results: `results/doloris/` (report tables in `results/doloris/report/`).

## Why a port, not the released pipeline

The released code ([github.com/ChangxiChi/Doloris](https://github.com/ChangxiChi/Doloris) at `1774472`)
contains the model, training and scoring code, but its sci-Plex3 run cannot be executed from the release:

- the data file it loads (`Dataset/molecular/sciplex_complete_unlasting_ver.h5ad`, which carries the split,
  SMILES, `dose_val` and chemCPA's DEG lists) is not released ("We will provide the pre-split data and the
  corresponding code in the following days");
- the molecular branch of `test.py` points to a `test_sciplex3.py` that is not in the repository;
- the gene graph is built from pretrained scGPT embeddings for every dataset, so the code needs the scGPT
  checkpoint and package even for drugs.

The paper's sci-Plex3 setting also differs from ours (three cell lines, 188 drugs, a random 30% of
(drug, dose, cell line) conditions held out, or whole drugs for the OOD table, chemCPA's 2,000 genes and
normalisation), so its absolute numbers are not comparable to the ones here. (A comment in `Preprocess.py`
mentions a double log1p; in the code that extra log1p is applied to the genetic datasets only.)

The model, diffusion process, mask model and inference were therefore transcribed into
`sciplex_doloris/` and run on our data. **Parity with the released code is checked**, not assumed:
[`tests/parity_doloris.py`](../../tests/parity_doloris.py) loads the released `model.py`, `diffusion.py`
and `respace.py` (with torch_geometric 2.6.1) and the metric/mask helpers of `test_util.py`, copies
identical weights into both implementations, and compares forward passes (train and eval mode),
the training loss and gradients, DDIM inversion and sampling, the mask model, the mask sampler and the
three metric functions. All 23 checks pass on an RTX 2080 Ti (`results/doloris/parity.json`; largest
differences ~1e-5 on outputs of order 1, metrics identical).

## What is run

| | released code / paper | here |
|---|---|---|
| data | chemCPA sci-Plex3, 3 cell lines | A549, 26 HDACi, 2,085 genes, log1p(CP10K) (frozen panel v1) |
| split | random 30% of (drug, dose, cell line) | one held-out intermediate dose per drug: 9 val, 17 test (frozen split v1) |
| scaling | ÷ max of training values; × max and negatives → 0 after generation | same |
| drug input | Uni-Mol CLS embedding (512) × sigmoid(dose network) | same (unimolv1, all-H weights); dose_val = dose / 10 µM as in chemCPA |
| diffusion | linear β, 500 steps respaced to 50 (DDIM), x0-prediction, MSE on non-zero entries | same |
| training | 60k steps (config; paper text says 100k), batch 32, AdamW lr 1e-4 (code default; paper text says 1e-3), linear decay | same |
| mask model | GAT over a scGPT-similarity + co-expression gene graph, BCE, 2 epochs, AdamW 1e-2 | **co-expression edges only** (scGPT checkpoint not used) |
| inference | DDIM-invert 100 control cells, DDIM-generate, mask from nearest training condition, thresholds 0.99/0.01 | same, **512 cells** (as for Ours and CellFlow) |
| checkpoint | last weights | last weights (`final`) **and** lowest validation RMSE (All) (`bestval`, every 5k steps) |
| seeds | not stated | 3 |

DE20/DE40 follow chemCPA's `lincs_DEGs`, which DOLORIS reads: scanpy `rank_genes_groups` (t-test)
of the condition's measured cells against all vehicle cells, ranked by |score|, first 20 / 40 genes
(`results/doloris/de_genes.json`). This differs from the project's DEG50 (top 50 by |mean shift|).

## Running

```bash
slurm/doloris/submit.sh                                   # tests, parity, embeddings, 3 seeds, scoring, report
slurm/doloris/submit.sh STAGES=report                     # redraw the tables from saved scores
```

GPU stages use `debug`/RTX 2080 Ti (the debug QoS allows 2 submitted jobs per user, so seeds are grouped:
`FIT_GROUPS="0,1 2"`); a seed trains in about 30 minutes. Everything else runs on `cpu`.
