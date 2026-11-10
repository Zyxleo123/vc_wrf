"""Tables from saved score rows only (score.py) and the DOLORIS selection records (generate.py).

Per method, each condition's value is first averaged over seeds; tables give the mean over
conditions ± SD over conditions (DOLORIS's convention) and the SD of the condition mean across
seeds. Paired comparisons: Δ = other − Ours per condition (lower is better for every metric, so
Δ > 0 favours Ours), 95% bootstrap CI over conditions, and conditions where Ours is lower.

usage: python -m sciplex_doloris.report --split test --root results/doloris --name v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_analysis.analyze import paired_stats
from sciplex_interp.core import REPO, read_jsonl

from .metrics import COLUMNS, GENE_SETS, KINDS

FROZEN_COLS = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "w1_r"]
FROZEN_LABELS = {"mmd2": "MMD²", "rmse_mean_deg50": "RMSE DEG50", "rmse_mean_all": "RMSE all genes",
                 "rmse_mean_program": "RMSE program", "w1_r": "W1(r)"}
KIND_LABELS = {"rmse": "RMSE", "edist": "E-dist", "emd": "EMD"}
SET_LABELS = {"all": "All", "de20": "DE20", "de40": "DE40"}
N_BOOT = 10000


def labels(name: str) -> dict:
    return {f"doloris_{name}_final": "DOLORIS (final checkpoint)", f"doloris_{name}_bestval": "DOLORIS (best val RMSE)",
            "ours": "**Ours**", "cellflow": "CellFlow", "chemcpa": "chemCPA", "adjacent": "Adjacent-dose mixture (reference)",
            "vehicle": "Vehicle cells (no drug effect)", "replicate": "Replicate well (noise floor)"}


def matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index="condition", columns="seed", values=metric)


def summary(df: pd.DataFrame, methods: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        part = df[df.method == m]
        for k in metrics:
            mat = matrix(part, k)
            per_cond = mat.mean(1)
            rows.append({"method": m, "metric": k, "mean": per_cond.mean(), "sd_conditions": per_cond.std(ddof=1),
                         "seed_sd": mat.mean(0).std(ddof=1) if mat.shape[1] > 1 else np.nan,
                         "n_seeds": mat.shape[1], "n_conditions": len(per_cond)})
    return pd.DataFrame(rows)


def fmt_table(summ: pd.DataFrame, methods: list[str], metrics: list[str], heads: list[str], lab: dict, sd: str) -> str:
    out = ["| | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
    for m in methods:
        cells = []
        for k in metrics:
            r = summ[(summ.method == m) & (summ.metric == k)].iloc[0]
            cells.append(f"{r['mean']:.4f} ± {r[sd]:.4f}" if np.isfinite(r[sd]) else f"{r['mean']:.4f}")
        out.append(f"| {lab[m]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--root", default=str(REPO / "results" / "doloris"))
    p.add_argument("--name", default="v1")
    a = p.parse_args(argv)

    root = Path(a.root)
    df = pd.DataFrame(read_jsonl(root / "scores" / f"{a.split}.jsonl"))
    lab = labels(a.name)
    methods = [m for m in lab if m in set(df.method)]
    clip_cols = [f"{c}_clip0" for c in COLUMNS]
    summ = summary(df, methods, FROZEN_COLS + COLUMNS + clip_cols)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / f"summary_{a.split}.csv", index=False)

    heads = [f"{SET_LABELS[g]} {KIND_LABELS[k]}" for g in GENE_SETS for k in KINDS]
    n_cond = int(df.condition.nunique())
    neg = df.groupby("method").negative_fraction.mean()
    with_neg = ", ".join(f"{lab[m].strip('*')} {100 * neg[m]:.0f}%" for m in methods if neg[m] > 0) or "none"
    md = [f"# DOLORIS on the A549 HDACi held-out-dose panel: {a.split} ({n_cond} conditions)", "",
          "Lower is better everywhere. Mean over conditions ± SD over conditions (DOLORIS's convention), each condition "
          "first averaged over seeds. Generated methods: 3 seeds. Written by `python -m sciplex_doloris.report`.", "",
          "## DOLORIS's metrics, cells as each method generated them", "",
          fmt_table(summ, methods, COLUMNS, heads, lab, "sd_conditions"), "",
          "## DOLORIS's metrics, negative values set to 0 for every method", "",
          "DOLORIS sets its own negative outputs to 0 before scoring (test.py); this applies the same step to every "
          "method. Ours and CellFlow already do so when generating. Saved cells with negative entries (share of all "
          f"values): {with_neg}; only those rows differ from the table above.", "",
          fmt_table(summ, methods, clip_cols, heads, lab, "sd_conditions"), "",
          "## Frozen project metrics (mean over conditions ± seed SD of that mean)", "",
          fmt_table(summ, methods, FROZEN_COLS, [FROZEN_LABELS[k] for k in FROZEN_COLS], lab, "seed_sd"), ""]

    rng = np.random.default_rng(0)
    paired = []
    ours = df[df.method == "ours"]
    for m in methods:
        if m == "ours" or ours.empty:
            continue
        for k in FROZEN_COLS + COLUMNS + clip_cols:
            s = paired_stats(matrix(df[df.method == m], k), matrix(ours, k), rng, N_BOOT)
            paired.append({"other": m, "metric": k, **s})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / f"paired_{a.split}.csv", index=False)
    if len(paired):
        others = [m for m in methods if m.startswith("doloris_")] + [m for m in ("cellflow", "chemcpa", "adjacent") if m in methods]
        md += ["## Paired against Ours", "",
               "Δ = other − Ours, averaged over conditions (Δ > 0 favours Ours), [95% bootstrap CI over conditions], "
               "conditions where Ours is lower.", "",
               "| metric | " + " | ".join(lab[o] for o in others) + " |", "|---" * (len(others) + 1) + "|"]
        names = {**FROZEN_LABELS, **{c: f"{SET_LABELS[c.split('_')[1]]} {KIND_LABELS[c.split('_')[0]]}" for c in COLUMNS},
                 **{f"{c}_clip0": f"{SET_LABELS[c.split('_')[1]]} {KIND_LABELS[c.split('_')[0]]}, negatives→0" for c in COLUMNS}}
        for k in FROZEN_COLS + COLUMNS + clip_cols:
            cells = []
            for o in others:
                r = paired[(paired.other == o) & (paired.metric == k)].iloc[0]
                cells.append(f"{r.delta_mean:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] {r.n_win}/{r.n_conditions}")
            md.append(f"| {names[k]} | " + " | ".join(cells) + " |")
        md.append("")

    sel = []
    for path in sorted((root / "runs" / a.name).glob("seed_*/selected_s0.json")):
        s = json.loads(path.read_text())
        curve = {r["step"]: r["val_rmse_all"] for r in s["curve"]}
        sel.append(f"| {path.parent.name} | {s['final']} ({curve[s['final']]:.5f}) | {s['bestval']} ({curve[s['bestval']]:.5f}) |")
    if sel:
        md += ["## DOLORIS checkpoint rules", "", "Validation RMSE (All) in parentheses.", "",
               "| run | final | best validation RMSE |", "|---|---|---|", *sel, ""]
    (out / f"{a.split}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
