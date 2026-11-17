# DOLORIS's sparsity mask on Ours (post-hoc): does it help?

Question: Ours (CellFlow latent flow + condition response embedding + grouped velocity matching)
decodes cells from a 70-dim latent, so its cells are dense — about **50% of genes non-zero per
cell**, against the ~4% of real sci-Plex3 cells. DOLORIS produces realistic sparsity with a separate
masking module. Can we borrow that module to sparsify Ours' cells, and does it improve the metrics?

**Verdict: no.** Bolting DOLORIS's mask onto Ours' saved cells post-hoc hurts almost every metric,
badly on the cell-geometry ones (E-distance, MMD²). The only wins are DE-gene EMD on the OOD drugs.
The mask is co-designed with DOLORIS's own generator; it does not transfer to a generator that was
never trained to be masked.

Frozen evaluator (`sciplex_standard.evaluate`, DOLORIS's 9 metrics + MMD²), 3 training seeds, 100
generated cells per condition. `ours_masked` / `ours_final_masked` reuse the **same** seed's trained
DOLORIS mask model. `ours_final` ≈ `ours` and `ours_final_masked` ≈ `ours_masked` throughout, so
only `ours` / `ours_masked` are shown below; the full tables with both rules are in
`results/standard/report.md`.

## What DOLORIS's mask is

DOLORIS imposes sparsity with a module **separate from its generator**, not a loss term inside it:

- **MaskModel** (`sciplex_doloris.model.MaskModel`): a small network — a gene-graph GAT over a
  co-expression graph — that, given `(drug, dose, cell line)` and a control prior, predicts a
  **per-gene probability of being non-zero**. It is trained on its own with binary cross-entropy
  against the real cells' zero/non-zero labels (`sciplex_standard.doloris.train`, 2 epochs).
- **Masking at generation** (`sciplex_doloris.generate.sample_mask`): the dense generator output is
  multiplied by a 0/1 stencil built as follows — (1) read the mask model's per-gene probabilities;
  (2) find the **training condition whose empirical non-zero rates are nearest** over the 300 most
  likely genes; (3) each generated cell inherits a **random real cell of that condition's** zero
  pattern; (4) genes the model is near-certain about are forced on (prob ≥ 0.99, or 0.95 on OOD) or
  off (≤ 0.01 / 0.05). The stencil keeps ~4% of genes, so a dense cell becomes sparse.

The stencil in step 3 is model-agnostic — it depends only on the condition — so it drops onto any
generator's cells. That is what makes this cheap to try on Ours.

## How it was applied to Ours

`sciplex_standard.mask_ours` reuses, per seed, that seed's already-trained mask model
(`results/standard/runs/doloris/seed_*/mask_model.pt`), its co-expression graph, the per-cell-line
control draw, and the training-condition zero-pattern pool — exactly as `sciplex_standard.doloris`
builds them — and masks the saved Ours cells condition by condition. Ours' cells are already in the
panel's log1p space and non-negative, so the 0/1 stencil multiplies them directly (no `x_max`; that
factor only undoes DOLORIS's internal 1/`x_max` scaling, which Ours never applied). No retraining.

Reproduce (CPU; needs the DOLORIS runs and the Ours preds on disk):

```
slurm/standard/submit_mask.sh            # mask -> re-score each split -> rebuild report
# or directly:
python -m sciplex_standard.mask_ours --root results/standard --seeds 0 1 2 --splits test ood val
```

Masking took Ours from **non-zero fraction 0.50 → 0.04**, i.e. it did make the cells realistically
sparse. The cells just become worse predictions.

## Results

Mean over conditions ± SD over seeds; **lower is better**. ✗ = masking worse, ✓ = masking better.

### Table 1 — unseen drug × cell line × dose (test, 650 conditions)

| metric | ours | ours_masked | |
|---|---|---|---|
| All EMD     | 0.0198 | 0.0351 | ✗ |
| All E-dist  | 0.1708 | 1.1601 | ✗ 6.8× |
| All RMSE    | 0.0294 | 0.0687 | ✗ |
| MMD²        | 0.0057 | 0.1137 | ✗ 20× |
| DE20 EMD    | 0.0479 | 0.0711 | ✗ |
| DE40 EMD    | 0.0435 | 0.0674 | ✗ |

Paired (Δ = masked − ours, 95% bootstrap CI over conditions): every metric worse with a CI clear of
0, e.g. edist_all +0.99 [+0.97, +1.00] (650/650 conditions favour unmasked), mmd2 +0.108
[+0.104, +0.111] (648/650).

### Table 2 — OOD drugs (ood, 108 conditions)

| metric | ours | ours_masked | |
|---|---|---|---|
| All EMD     | 0.0297 | 0.0363 | ✗ |
| All E-dist  | 0.6270 | 1.3236 | ✗ 2.1× |
| All RMSE    | 0.0566 | 0.0818 | ✗ |
| MMD²        | 0.0666 | 0.1513 | ✗ |
| DE20 EMD    | 0.1386 | 0.1202 | ✓ |
| DE40 EMD    | 0.1140 | 0.1056 | ✓ |

The two **DE-gene EMD** metrics are the only wins, and only on OOD: paired emd_de20 −0.018
[−0.028, −0.009], emd_de40 −0.008 [−0.016, −0.001]. Trimming the dense smear off the non-DE genes
helps the per-gene EMD once it is restricted to the DE genes, where the mask keeps the right genes.
Everything that sees the full gene vector or cell geometry is much worse: edist_all +0.70
[+0.63, +0.76] (104/108), mmd2 +0.085 [+0.071, +0.098] (97/108).

For reference, unmasked Ours already beats DOLORIS (`doloris_bestval`/`doloris_final`) on every
metric of both tables; masking Ours pushes it below DOLORIS and, on E-distance, near biolord — the
paper's weakest sparse-but-misplaced baseline.

## Why it doesn't transfer

The decisive evidence is the **identical mask on DOLORIS's own cells vs Ours' cells**:

| All E-dist | test | ood |
|---|---|---|
| DOLORIS (its generator + mask) | 0.28–0.35 | 0.73–0.77 |
| Ours + the same mask            | 1.16     | 1.32 |

Same masking rule, far worse on us. So the mask is not broken — Ours' cells are incompatible with it:

- DOLORIS's diffusion generator is **trained alongside the masking** (it adds the control prior and
  learns outputs that survive the stencil), so the ~4% of genes the stencil keeps are the ones
  DOLORIS put signal on.
- Ours spreads its signal densely through the PCA-latent decode with no notion that a hard 4%
  stencil is coming. Multiplying by a random real cell's on/off pattern keeps a near-random 4% slice
  of our signal and zeros the rest — including genes Ours predicted correctly — which destroys the
  per-cell geometry (hence the E-distance and MMD² blow-up). Ours' dense smear, by contrast, costs
  almost nothing on EMD-all or MMD² in the 50-PC space, which is why unmasked Ours scores so well.

Analogy: a detailed painting (Ours) sprayed through a stencil cut from a *different* painting — you
keep whatever bits line up with the holes and lose the rest.

## Takeaways / next steps

- "DOLORIS's sparse loss stuff" is a **generator-coupled module**, not a drop-in loss. A post-hoc
  bolt-on does not help Ours.
- If sparsity is worth pursuing, the destructive step is #3 (inheriting a whole foreign cell's zero
  pattern). Cheaper, gentler variants reuse the same job with a different masking rule:
  - **confident-off only** — zero only genes the mask model is near-certain are off (prob ≤ low),
    keep Ours' values everywhere else;
  - **soft mask** — multiply by the predicted P(non-zero) rather than a hard 0/1.
- The faithful version is architectural: a sparsity / zero-inflation head trained **jointly** with
  the flow, so Ours learns to place its signal where the mask will keep it — a project, not a
  bolt-on.
