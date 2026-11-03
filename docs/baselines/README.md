# Baseline and run index

Keep one row per experiment version. Completion status and seed counts matter as
much as the reported metric. The machine-readable index is [registry.json](registry.json).

| Baseline/run | Status | Scored seeds | Code | Results |
|---|---|---|---|---|
| CellFlow v1: standard, WRF-3 and MMD | Complete, original implementation | 0, 1, 2 for each method | [Source](../../experiments/cellflow_v1/README.md) | [Original report](../cellflow/v1/report/cellflow-v1-report.txt) |
| CellFlow initial WRF-13 attempt | Incomplete: seeds 1 and 2 failed | Seed 0 only, matched across comparison rows | [Source](../../experiments/cellflow_q13_initial_20261030/README.md) | [Outcomes](../cellflow/q13_initial_20261030/README.md) |
| CellFlow WRF-3 versus WRF-13 precision reruns | Complete, exploratory numerical change | 0, 1, 2 for both arms | [Source](../../experiments/cellflow_q13_repair_20261030/README.txt) | [Three-seed table](../cellflow/q13_repair_20261030/README.md) |
| chemCPA | Existing teammate branch; not merged into main as of 2026-11-02 | See teammate report | [Pinned source](https://github.com/Zyxleo123/vc_wrf/tree/186875e3d4834f3e311b1c7a3ce6b3f39716d8f3/sciplex_baselines) | [Pinned report index](https://github.com/Zyxleo123/vc_wrf/blob/186875e3d4834f3e311b1c7a3ce6b3f39716d8f3/docs/chemcpa/README.md) |

The chemCPA entry is a discovery link, not a claim that its runtime data hashes or
aggregation are fully harmonized with CellFlow. See [comparison status](../cellflow/COMPARABILITY.md)
and the [team workflow](../../CONTRIBUTING.md) before building a combined leaderboard.

Raw panels, model checkpoints and generated populations stay on shared storage.
Each experiment's archive record identifies its location and hashes. The current
PSC CellFlow archives are available to `ywang179` and `amotwani`; sharing with another
group member requires their cluster username and appropriate storage access.
