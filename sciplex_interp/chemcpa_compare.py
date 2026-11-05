"""Paired comparison of chemCPA variants with the original models, from frozen per-condition rows.

New variants: <run root>/<variant>/seed_*/<split>_sample_s<offset>/per_condition.jsonl, written by
`python -m sciplex_baselines.chemcpa evaluate` (the frozen evaluator, 1024 generated cells per
condition, the same generation as the original runs). Original chemCPA, CellFlow and the reference
rows: results/analysis/tables/per_condition.csv. Population metrics only (no WRF).

Δ = ref − new on per-condition, seed-averaged errors (> 0: the new variant is better), with
sciplex_analysis.analyze.paired_stats: condition bootstrap 95% CI, a two-level (condition x seed)
CI, and all seed-pair differences.

usage: python -m sciplex_interp.chemcpa_compare --variants gaussmean_l100 nbmean_l10 \
         --primary gaussmean_l100=chemcpa/gauss_v1 nbmean_l10=chemcpa/nb_v1 --out docs/interp
Writes <out>/chemcpa_test.md, chemcpa_test_summary.csv and chemcpa_test_paired.csv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_analysis.analyze import paired_stats

from .core import read_jsonl

REPO = Path(__file__).resolve().parents[1]
MODEL_TABLE = REPO / "results" / "analysis" / "tables" / "per_condition.csv"
METRICS = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "w1_r"]
LABELS = {"mmd2": "MMD²", "rmse_mean_deg50": "RMSE DEG50", "rmse_mean_all": "RMSE all genes",
          "rmse_mean_program": "RMSE program", "w1_r": "W1(r)"}
REFS = {"chemcpa/gauss_v1": "chemCPA-Gauss (original)", "chemcpa/nb_v1": "chemCPA-NB (original)",
        "cellflow/cellflow": "CellFlow", "cellflow/cellflow_mmd": "CellFlow + MMD loss",
        "reference/adjacent_dose_mixture": "Adjacent-dose mixture (real neighbour cells)",
        "reference/vehicle": "Vehicle cells (no drug effect)",
        "reference/replicate": "Replicate well (noise floor)"}
VARIANT_LABELS = {"gaussmean_l100": "chemCPA-Gauss + mean consistency (λ=100)",
                  "nbmean_l10": "chemCPA-NB + mean consistency (λ=10)"}


def run_root(variant: str) -> Path:
    for root in (os.environ.get("CHEMCPA_ROOT", REPO / "results" / "interp" / "chemcpa"),
                 os.environ.get("BASE_CHEMCPA_ROOT", REPO / "results" / "chemcpa")):
        if (Path(root) / variant).is_dir():
            return Path(root) / variant
    raise SystemExit(f"no run directory for {variant}")


def variant_rows(variant: str, split: str, offset: int) -> pd.DataFrame:
    paths = sorted(run_root(variant).glob(f"seed_*/{split}_sample_s{offset}/per_condition.jsonl"))
    if not paths:
        raise SystemExit(f"no {split} offset-{offset} evaluations for {variant}")
    rows = [{**r, "seed": int(p.parent.parent.name.split("_")[1])} for p in paths for r in read_jsonl(p)]
    return pd.DataFrame(rows)[["seed", "condition", *METRICS]]


def model_rows(table: pd.DataFrame, key: str, split: str) -> pd.DataFrame:
    model, method = key.split("/")
    part = table[(table.model == model) & (table.method == method) & (table.split == split)].copy()
    if part.empty:
        raise SystemExit(f"no {split} rows for {key}")
    part["seed"] = part.seed.fillna(0).astype(int)
    return part[["seed", "condition", *METRICS]]


def matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index="condition", columns="seed", values=metric, aggfunc="mean")


def summary_cell(df: pd.DataFrame, metric: str) -> str:
    per_seed = df.groupby("seed")[metric].mean()
    v = df.groupby("condition")[metric].mean().mean()
    f = "{:.4f}" if metric == "mmd2" else "{:.3f}"
    s = f.format(v)
    return s + (f" ± {f.format(per_seed.std(ddof=1))}" if len(per_seed) > 1 else "")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--primary", nargs="+", default=[], help="variant=model/method pairs shown first")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--robustness-offset", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default=str(REPO / "docs" / "interp"))
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(MODEL_TABLE)
    rng = np.random.default_rng(0)
    md = ["# chemCPA with population-mean consistency vs the original models", "",
          "Frozen evaluator, 1024 generated cells per condition, 3 training seeds (± = SD over "
          "seeds). Δ = ref − new on condition means; > 0 means the new variant is better. "
          "CI: condition bootstrap (two-level condition × seed CI in the CSV).", ""]
    summary, paired = [], []
    for split in a.splits:
        ents = {f"variant:{v}": variant_rows(v, split, a.offset) for v in a.variants}
        ents.update({k: model_rows(table, k, split) for k in REFS})
        label = lambda k: VARIANT_LABELS.get(k.removeprefix("variant:"), k) if k.startswith("variant:") else REFS[k]
        n_cond = ents[f"variant:{a.variants[0]}"].condition.nunique()
        md += [f"## {split} ({n_cond} conditions)", "",
               "| | " + " | ".join(LABELS[m] for m in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for k, df in ents.items():
            md.append(f"| {label(k)} | " + " | ".join(summary_cell(df, m) for m in METRICS) + " |")
            summary.append({"split": split, "entity": k, "label": label(k),
                            **{m: df.groupby("condition")[m].mean().mean() for m in METRICS}})
        if split == "test" and a.robustness_offset is not None:
            md.append("")
            md.append(f"Second sampling seed (offset {a.robustness_offset}), MMD² / RMSE DEG50: " + "; ".join(
                f"{label('variant:' + v)} {variant_rows(v, split, a.robustness_offset).groupby('condition').mmd2.mean().mean():.4f}"
                f" / {variant_rows(v, split, a.robustness_offset).groupby('condition').rmse_mean_deg50.mean().mean():.3f}"
                for v in a.variants))
        md.append("")
        pairs = []
        primary = dict(x.split("=") for x in a.primary)
        for v in a.variants:
            refs = ([primary[v]] if v in primary else []) + [k for k in REFS if k != primary.get(v)]
            for ref in refs:
                pairs.append((ref, f"variant:{v}"))
        md += [f"### {split}: paired differences", "",
               "| new | ref | " + " | ".join(LABELS[m] for m in METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
        for ref, new in pairs:
            cells = []
            for m in METRICS:
                s = paired_stats(matrix(ents[ref], m), matrix(ents[new], m), rng, a.n_boot)
                paired.append({"split": split, "ref": ref, "new": new, "metric": m, **s})
                f = (lambda x: f"{x:+.4f}") if m == "mmd2" else (lambda x: f"{x:+.3f}")
                cells.append(f"{f(s['delta_mean'])} [{f(s['ci95_low'])}, {f(s['ci95_high'])}] "
                             f"{s['n_win']}/{s['n_conditions']}")
            md.append(f"| {label(new)} | {label(ref)} | " + " | ".join(cells) + " |")
        md.append("")
    pd.DataFrame(summary).to_csv(out / "chemcpa_test_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "chemcpa_test_paired.csv", index=False)
    (out / "chemcpa_test.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
