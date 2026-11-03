# Contributing

- Use one branch and pull request per baseline.
- Put implementation and configuration in `experiments/<model>/` and results in `docs/<model>/`.
- Use the shared `sciplex_eval` evaluator, frozen spec and split. Select checkpoints on validation data.
- Report per-seed errors before averaging, with sample SD across seeds.
- Keep large datasets, models and predictions on shared storage; record their paths and checksums.
- Add one row to the [baseline index](docs/baselines/README.md).
