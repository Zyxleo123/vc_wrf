# Working on different baselines

Use one shared evaluator and split, separate model code and run records, and one
branch/pull request per baseline change. The [baseline index](docs/baselines/README.md)
links the code, results and completion status already available.

## Repository layout

| Location | Purpose |
|---|---|
| `sciplex_eval/` | Shared metric implementation |
| `artifacts/eval_spec_v1/`, `configs/split_v1.json` | Frozen response score, genes, PCA and split |
| `sciplex_baselines/<baseline>.py` | Reusable model adapters, as used on the chemCPA branch |
| `experiments/<baseline>_<run_id>/` | Exact run implementation, environment lock and reproduction instructions |
| `docs/<baseline>/<run_id>/` | Small result tables, per-condition/seed records, figures, configuration and provenance |
| `docs/baselines/registry.json` | Index of run status and artifact locations |
| PSC/Babel storage | Data panels, trained checkpoints and generated `.h5ad` files |

CellFlow's existing run snapshots retain their original paths and code so their
checksums remain useful. New reusable adapters can follow the common package layout
without rewriting an archived experiment. `results/` stays ignored for local outputs.

## Branch and review workflow

1. Start a branch such as `<username>/<baseline>-<change>` from current `main`.
   Contributors without write access can use a fork and open a pull request upstream.
2. Keep edits inside your baseline directories. Propose changes to the shared
   evaluator or split in a separate pull request; never overwrite a frozen spec.
3. Record configuration and selection rules before the scored run. Choose settings
   on validation data. Disclose exploratory work done after test results were seen.
4. Commit code, locked dependencies, small numeric results, figures and an artifact
   index. Keep large datasets, predictions and checkpoints on shared storage and
   include their paths and checksums. Do not commit credentials, SSH files or caches.
5. Open a pull request with the result, validation performed, failed attempts and
   limitations. Merge after teammate review; do not merge someone else's baseline
   branch as part of publishing your own results.

## What every result submission needs

- A unique run ID and contributor, code revision, evaluator revision, exact command,
  environment/dependency lock, panel checksum and spec/split checksums.
- Planned, completed and failed seeds; loss settings; numerical precision; training
  budget; checkpoint selection rule; prediction count, source-control sampling,
  replicate handling and expression postprocessing.
- Per-condition metrics with `method`, `seed`, `split`, `condition`, `n_real`, `n_pred`
  and the shared metric keys. Preserve individual seed results, not just a screenshot.
- Metric means, seed SD, number of seeds and conditions, and separately labeled
  condition-bootstrap intervals. Preserve failed runs and superseded versions.
- A short reproduction guide and pointers to larger artifacts. The optional
  [metadata template](docs/baselines/run_metadata.template.json) is a starting point.

## How to combine results fairly

Compute each condition's error separately for each seed; average conditions equally
within a seed, then average the seed means. Report sample SD across seed means.
For one completed seed, report SD as unavailable. Do not select the best test seed.

For WRF, compute `100 * abs(predicted_fraction - measured_fraction)` before averaging
seeds. Averaging fractions first can cancel errors and gives a different number.
The supplementary 13-threshold mean is the mean of those per-threshold errors,
again before seed averaging. It is not a three-threshold mean and is not one of the
eight frozen v1 metrics. [The recorded grid](configs/eval_grid13_v1.json)
preserves the thresholds used in the CellFlow comparison.

Match the data/spec hashes and disclose generation and checkpoint-selection policies
before putting models in a common table. The present CellFlow and chemCPA pipelines
have [documented differences](docs/cellflow/COMPARABILITY.md). Their numbers can be
shown as pipeline results, but they are not a fully controlled architecture benchmark.
Agree on any new common protocol using validation data before the next test round.

## Verify the CellFlow publication locally

Python 3.10+ and the standard library are sufficient; no cluster, GPU or dataset is needed:

```bash
python experiments/cellflow/verify_all_results.py
```

This verifies saved results and provenance. It does not retrain models or substitute
for evaluating the large prediction files retained on PSC.
