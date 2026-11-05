"""Development tables for CellFlow runs, from saved history.jsonl rows only (no model, no test cells).

For each variant: validation and in-sample fit at every validation step (mean over seeds), and the
seeds' selected checkpoints (lowest validation MMD²).

usage: python -m sciplex_interp.cellflow_report [--variants base mean_l10 ...] [--steps]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DEFAULT_ROOT, read_jsonl

COLS = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "w1_r", "var_ratio"]


def condition_signal(part: pd.DataFrame) -> pd.Series:
    """Does the generated effect size follow the condition? spread 0 / corr 0 = condition ignored."""
    real, pred = part.effect_real_all, part.effect_pred_all
    return pd.Series({"spread": pred.std() / real.std(), "spread_corr": np.corrcoef(real, pred)[0, 1]})


def load(root: Path, variants: list[str] | None) -> pd.DataFrame:
    base = root / "cellflow"
    names = variants or sorted(p.name for p in base.iterdir() if p.is_dir())
    frames = []
    for v in names:
        for path in sorted((base / v).glob("seed_*/history.jsonl")):
            frames.append(pd.DataFrame(read_jsonl(path)))
    if not frames:
        raise SystemExit(f"no history under {base}")
    return pd.concat(frames, ignore_index=True)


def selected_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every variant x seed at its checkpoint of lowest mean validation MMD² (first on ties)."""
    val = df[df.task == "val"].groupby(["variant", "seed", "step"]).mmd2.mean().reset_index()
    best = val.loc[val.groupby(["variant", "seed"]).mmd2.idxmin(), ["variant", "seed", "step"]]
    return df.merge(best, on=["variant", "seed", "step"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="*")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--steps", action="store_true", help="also print every validation step")
    a = p.parse_args(argv)
    df = load(Path(a.root), a.variants)
    fmt = lambda v: f"{v:.4f}"
    if a.steps:
        for v, part in df.groupby("variant", sort=False):
            for task in ("val", "train_fit"):
                rows = part[part.task == task]
                per_seed = rows.groupby(["seed", "step"])[COLS].mean()
                per_seed = per_seed.join(rows.groupby(["seed", "step"])[["effect_real_all", "effect_pred_all"]].apply(condition_signal))
                table = per_seed.groupby("step").mean()
                table.insert(0, "seeds", per_seed.groupby("step").size())
                print(f"\n{v} {task} (mean over seeds)\n" + table.to_string(float_format=fmt))
    sel = selected_rows(df)
    keys = ["variant", "seed", "task"]
    per_seed = sel.groupby(keys)[COLS].mean().join(
        sel.groupby(keys)[["effect_real_all", "effect_pred_all"]].apply(condition_signal)).reset_index()
    steps = sel.groupby(["variant", "seed"]).step.first().groupby("variant").apply(lambda s: list(s))
    for task in ("val", "train_fit"):
        part = per_seed[per_seed.task == task]
        table = part.groupby("variant")[[*COLS, "spread", "spread_corr"]].mean()
        table.insert(1, "mmd2_seed_sd", part.groupby("variant").mmd2.std(ddof=1))
        print(f"\nselected checkpoint, {task} (mean over seeds)\n" + table.sort_values("mmd2").to_string(float_format=fmt))
    print("\nselected steps\n" + steps.to_string())


if __name__ == "__main__":
    main()
