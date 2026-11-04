# scVIDR

[Results](../../docs/scvidr/README.md) · [Base configuration](config.json) ·
[WRF and MMD, 3 thresholds](config_3.json) · [WRF, 13 thresholds](config_13.json)

`run_scvidr.py` trains the unmodified scVIDR VAE
([BhattacharyaLab/scVIDR](https://github.com/BhattacharyaLab/scVIDR) at
`bcb6a96c85cddc4eef052be739c122efb737fe0e`) on the train and vehicle cells of the frozen
v1 panel and generates the held-out doses with scVIDR's continuous dose rule:

z(d) = z_control + δ · log1p(d) / log1p(10000),  δ = mean z(drug, 10000 nM) − mean z(vehicle)

with z the posterior mean, d in nM and the decoder mean as the generated cell. This is
`VIDR.predict(continuous=True, regression=False)`. The panel has one cell type, and the
held-out drug's 10 µM cells are in training, so the cross-cell-type δ regression that
scVIDR uses for unseen cell types does not apply.

Differences from the scVIDR code path:

- δ uses all vehicle and 10 µM cells instead of scVIDR's random balanced subsample.
- Each condition decodes 512 vehicle cells drawn with `default_rng(10000 + seed)`, so the
  evaluator receives 512 cells per condition as for CellFlow. scVIDR decodes one balanced
  subsample, which is sometimes smaller than the measured condition.
- Negative decoder outputs are clipped to zero, as for CellFlow, so the response score
  `log1p(sum expm1(x))` is defined.
- Early stopping is off. In scvi-tools 0.13 it monitors `elbo_validation`, which includes
  the unweighted KL. With scVIDR's KL weight of 5e-05 that value is lowest at epoch 1, so
  patience 25 always stops at epoch 26 while the training loss is still falling
  (validation loss 186 at epoch 1, 117 at epoch 26). Training therefore runs all 100 epochs.
- The checkpoint is chosen among epochs 10, 25, 50, 75 and 100 by validation WRF q=.95,
  then response W₁, as for CellFlow.

Architecture and training otherwise use the scVIDR defaults from its sci-Plex notebook:
2 hidden layers of 800, 100 latent dimensions, dropout 0.2, KL weight 5e-05, nonlinear
decoder, batch size 256, train size 0.8, 100 epochs, Adam with learning rate 1e-3.

## WRF and MMD variants

`population_loss.py` adds CellFlow's generated-population losses to the scVIDR objective
without changing the model or the dose rule. Every 5th training step from epoch 10, one of
the 78 training conditions is sampled. 128 vehicle cells are encoded, shifted by
δ · log1p(d) / log1p(10000), with δ from 128 of the drug's 10 µM cells and 128 other
vehicle cells under the current encoder, decoded in eval mode with gradients and clipped
at zero. These generated cells are compared with up to 128 measured cells of that
condition, which are held fixed:

- WRF: mean squared difference of soft fractions sigmoid((τ − r(x)) / h) over the
  training thresholds, in float64, with CellFlow's overflow-safe r(x). `config_3.json`
  uses the three frozen thresholds and `config_13.json` the 13-threshold grid; the two
  differ only in their thresholds.
- MMD: unbiased three-bandwidth RBF MMD² in the frozen evaluation PCA space.

The weight is multiplied by 5 so its expected coefficient is λ, as for CellFlow. scVIDR's
base loss is about 106 per cell against 0.22 for CellFlow flow matching, so the CellFlow
grids are multiplied by about 486: λ∈{5, 15, 50, 150} × h∈{0.1, 0.3} for WRF and
λ∈{50, 500} for MMD. Settings are chosen on seed 0 validation, then repeated with seeds 1
and 2, and the 13-threshold run reuses the selected WRF setting. Checkpoints are chosen
among epochs 25, 50, 75 and 100, after the loss starts.

## Setup

scVIDR needs its 2023 stack (Python 3.8, torch 1.8.1+cu111, scvi-tools 0.13.0,
pytorch-lightning 1.3.8, anndata 0.8.0); the [lock file](requirements.lock.txt) is the
environment that produced the results. Scoring uses a second environment with the CellFlow
evaluator pins (Python 3.12, numpy 2.4.6, pandas 3.1.0, anndata 0.13.4, scipy 1.15.3).
anndata 0.8 cannot read the panel's string encoding, so `convert` writes a copy with
legacy string columns, checks that `X`, counts and metadata are unchanged and records both
checksums.

Run from the repository root on a CUDA GPU:

```bash
git clone https://github.com/BhattacharyaLab/scVIDR.git /path/to/scVIDR
export RESPONSE_FRACTION_DATA=/path/to/sciplex3 SCVIDR_SOURCE=/path/to/scVIDR
EVAL=/path/to/envs/sciplex_eval/bin/python; SCVIDR=/path/to/envs/scvidr/bin/python; ROOT=results/scvidr
$EVAL experiments/scvidr/run_scvidr.py convert --root $ROOT
for seed in 0 1 2; do $SCVIDR experiments/scvidr/run_scvidr.py train --root $ROOT --seed $seed; done
$SCVIDR experiments/scvidr/run_scvidr.py index --root $ROOT
```

For the variants, with `R3=results/scvidr_aux3` and `R13=results/scvidr_aux13`:

```bash
$EVAL experiments/scvidr/run_scvidr.py convert --root $R3 --config experiments/scvidr/config_3.json
$EVAL experiments/scvidr/run_scvidr.py convert --root $R13 --config experiments/scvidr/config_13.json
for trial in wrf_l5_h0p1 wrf_l5_h0p3 wrf_l15_h0p1 wrf_l15_h0p3 wrf_l50_h0p1 wrf_l50_h0p3 wrf_l150_h0p1 wrf_l150_h0p3 mmd_l50 mmd_l500; do
  $SCVIDR experiments/scvidr/run_scvidr.py train --root $R3 --config experiments/scvidr/config_3.json --trial $trial --seed 0
done
$SCVIDR experiments/scvidr/run_scvidr.py select --root $R3 --config experiments/scvidr/config_3.json
for seed in 1 2; do for trial in wrf_l5_h0p1 mmd_l50; do
  $SCVIDR experiments/scvidr/run_scvidr.py train --root $R3 --config experiments/scvidr/config_3.json --trial $trial --seed $seed
done; done
$SCVIDR experiments/scvidr/run_scvidr.py select --root $R13 --config experiments/scvidr/config_13.json --source $R3
for seed in 0 1 2; do $SCVIDR experiments/scvidr/run_scvidr.py train --root $R13 --config experiments/scvidr/config_13.json --trial wrf_l5_h0p1 --seed $seed; done
$SCVIDR experiments/scvidr/run_scvidr.py index --root $R3 --config experiments/scvidr/config_3.json
$SCVIDR experiments/scvidr/run_scvidr.py index --root $R13 --config experiments/scvidr/config_13.json
$SCVIDR experiments/scvidr/run_scvidr.py merge --root results/scvidr_all --sources $ROOT $R3 $R13
$EVAL experiments/cellflow/source/evaluate.py --artifacts results/scvidr_all/artifacts.json --out results/scvidr_all/metrics
$EVAL experiments/scvidr/report.py --root results/scvidr_all --runs $ROOT $R3 $R13 --panel "$RESPONSE_FRACTION_DATA/processed/a549_hdaci_v1.h5ad"
```

`select` writes the chosen trials (here `wrf_l5_h0p1` and `mmd_l50`) to
`selected_methods.json`; replace the trial names in the later loops if it picks others.

Training saves each seed's model, validation history and predictions under `$ROOT`, and
`index` writes a local `artifacts.json` of the predictions. Scoring uses
`experiments/cellflow/source/evaluate.py`, the same evaluator and 13-threshold grid as the
CellFlow results. On one V100 each training takes about 1.5 minutes, and all 20 runs
(3 standard, 10 grid, 7 repeats) finish in about 35 minutes. `report.py` writes the
shareable tables to `docs/scvidr/`: the evaluator tables, the comparison table, paired
differences and spread diagnostics.

To verify the saved metrics without a GPU or dataset:

```bash
python experiments/scvidr/verify_results.py
```
