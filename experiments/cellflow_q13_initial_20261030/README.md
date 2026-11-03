# Original 13-threshold attempt: incomplete

This preserves the exact October 30 implementation and report-finalization scripts.
Seed 0 completed 30,000 updates; seeds 1 and 2 failed with nonfinite losses at updates
3550 and 3150. The original report is a **matched seed-0-only comparison**, not a
three-seed result and not a selection of the best-performing seed.

The later [precision reruns](../cellflow_q13_repair_20261030/README.txt) completed both
threshold variants for all three seeds. Do not overwrite or pool these attempts.

See [outcomes and report](../../docs/cellflow/q13_initial_20261030/README.md).
`source/` preserves the initial code; `finalization_source/` preserves the later
audit/report scripts used to finalize the incomplete experiment. Script paths are
the original PSC deployment paths. Adapt them and use a fresh output directory
before reproducing an attempt. The recorded environment is Linux/CUDA, not macOS.
