# CellFlow v1 source and reproduction

This directory publishes the saved implementation of the completed CellFlow,
CellFlow + WRF, and CellFlow + MMD experiment. The files in `source/` are byte-for-byte
copies of the archived final source, plus the original GPU integration helper needed
by `initialize`. The hash mapping is in
`../../docs/cellflow/v1/published_source_provenance.json`.

- `source/run_cellflow_v1.py`: data audit, training, validation selection, generation,
  references and aggregation.
- `source/cellflow_population_loss.py`: differentiable generated-population penalties.
- `source/cellflow_v1_config.json`: architecture, budget, loss grid and selection rules.
- `source/cellflow_v1_full.sbatch`: original PSC launch sequence, including the earlier
  baseline-checkpoint audit, selection, repeated seeds and final evaluation.
- `source/check_population_loss.py`, `check_cellflow_v1_gpu.py`,
  `check_validation_integration.py`: original numerical/integration checks.
- `source/verify_cellflow_artifacts.py`: full-run validation requiring PSC artifacts.
- `source/report_cellflow_v1.py`: report generation requiring full run diagnostics.
- `requirements.lock.txt`: recorded Linux/CUDA environment, not a portable macOS lock.

## Verify the published numbers locally

From the repository root, using Python 3.10 or later (standard library only):

```bash
python experiments/cellflow_v1/verify_published_results.py
```

This verifies publication checksums, source provenance, frozen spec/split hashes,
all 153 selected model/seed/condition test rows, WRF arithmetic, and the 24 model
metric means and seed SDs. It does not re-score large prediction files or run GPU tests.

## Reproduce training on a compatible GPU environment

Use a fresh working directory, the frozen panel, and the recorded CellFlow/JAX
versions. The saved scripts expect this layout:

```text
$RESPONSE_FRACTION_ROOT/
  src/                         # copy source/* here
    vc_wrf_1d8b173/             # checkout of evaluator commit 1d8b173...
  envs/cellflow/                # recorded Linux/CUDA environment
  results/                     # a fresh directory per run
  logs/
  cache/
$RESPONSE_FRACTION_DATA/
  processed/a549_hdaci_v1.h5ad
```

Set `RESPONSE_FRACTION_ROOT` and `RESPONSE_FRACTION_DATA` to those locations. The
original PSC shell launchers additionally source `project.env` at the historical
project path and contain that allocation's scheduler account, GPU type and log paths.
Adapt those deployment paths to your own allocation before submitting; `project.env`
is environment configuration, not included here. Do not run the scheduler launcher
locally. A minimal project.env sets the two variables above.

The source is deliberately preserved rather than silently rewriting the experiment.
The recorded launch order is initialize → initial candidates → early baseline audit →
merge baseline audit → select settings → repeat selected variants → validation ODE
check → evaluate → report. `cellflow_worker.py` partitions initial/repeat work. A
fresh run preserves failed candidates and uses validation-only selection. The
finalized rerun incorporates repairs; original failed/interrupted attempts remain
in the PSC original-run archive described in the report.

For report-only work, use the shared complete run at
`/ocean/projects/cis260266p/shared/response-fraction/results/cellflow_vc_wrf_v1_50003709`.
Copy it to a separate working location before regenerating reports. The published
summary subset lacks large predictions, models and detailed diagnostics required by
the original report script. `model_index.json` and `final_artifact_sha256.json` in
the documentation map and fingerprint those archived artifacts.

See [results](../../docs/cellflow/README.md) and
[comparison limitations](../../docs/cellflow/COMPARABILITY.md).
