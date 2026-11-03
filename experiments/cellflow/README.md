# CellFlow

[Results](../../docs/cellflow/README.md) · [Settings](../../docs/cellflow/settings.json) ·
[Saved models and predictions](../../docs/cellflow/artifacts.json)

`source/run_cellflow_v1.py` contains the training and prediction functions.
`source/cellflow_population_loss.py` implements WRF supervision with float64 auxiliary
trajectories. `source/baseline_population_loss.py` contains the float32 MMD implementation.
The model, ordinary flow loss and optimizer use float32.

## Setup

Use Linux with a CUDA GPU and the pinned [environment](requirements.lock.txt).
Set `RESPONSE_FRACTION_DATA` to the data directory containing
`processed/a549_hdaci_v1.h5ad`. Run commands from the repository root.

```bash
pip install -r experiments/cellflow/requirements.lock.txt
export RESPONSE_FRACTION_DATA=/path/to/sciplex3
```

## Train the WRF comparison

```bash
python experiments/cellflow/source/run.py initialize --root results/cellflow
for q in 3 13; do
  for seed in 0 1 2; do
    python experiments/cellflow/source/run.py train --root results/cellflow --thresholds "$q" --seed "$seed"
  done
  python experiments/cellflow/source/check_validation_integration.py --run "results/cellflow/run$q"
done
python experiments/cellflow/source/run.py predict --root results/cellflow
python experiments/cellflow/source/evaluate.py --artifacts results/cellflow/artifacts.json --out results/cellflow/metrics
```

Both configurations use 30,000 updates, λ=0.1, h=0.3, 8 auxiliary Heun steps and
64 prediction steps. Checkpoints minimize validation WRF q=.95, then response W₁.
The configurations differ only in their supervised thresholds.

## Reproduce the published table

On PSC, use the saved predictions listed in the artifact index:

```bash
python experiments/cellflow/source/evaluate.py --artifacts docs/cellflow/artifacts.json --out results/cellflow/rescored
```

To verify the saved metrics locally without a GPU or dataset:

```bash
python experiments/cellflow/verify_results.py
```
