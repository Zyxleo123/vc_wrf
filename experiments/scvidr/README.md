# scVIDR

[Results](../../docs/scvidr/README.md) · [Configuration](config.json)

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
$EVAL experiments/cellflow/source/evaluate.py --artifacts $ROOT/artifacts.json --out $ROOT/metrics
$EVAL experiments/scvidr/report.py --root $ROOT --panel "$RESPONSE_FRACTION_DATA/processed/a549_hdaci_v1.h5ad"
```

Training saves each seed's model, validation history and predictions under `$ROOT`, and
`index` writes a local `artifacts.json` of the predictions. Scoring uses
`experiments/cellflow/source/evaluate.py`, the same evaluator and 13-threshold grid as the
CellFlow results. One V100 trains, predicts and scores all three seeds in about ten
minutes. `report.py` writes the shareable tables to `docs/scvidr/`: the evaluator tables,
the comparison table, paired differences with CellFlow and spread diagnostics.

To verify the saved metrics without a GPU or dataset:

```bash
python experiments/scvidr/verify_results.py
```
