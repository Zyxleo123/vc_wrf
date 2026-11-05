"""Paired comparison of CellFlow variants with the matched base, published CellFlow and references.

New variants and the matched base: <root>/cellflow/<variant>/seed_*/<split>_sample_s<offset>/
per_condition.jsonl (sciplex_interp.cellflow evaluate: frozen evaluator, 512 generated cells per
condition, the published generation). chemCPA + mean consistency: its evaluation rows. Published
CellFlow, original chemCPA and the reference rows: results/analysis/tables/per_condition.csv.
Population metrics only (no WRF).

Δ = ref − new on per-condition, seed-averaged errors (> 0: the new variant is better), with
sciplex_analysis.analyze.paired_stats: condition bootstrap 95% CI, a two-level (condition x seed)
CI, and all seed-pair differences.

usage: python -m sciplex_interp.cellflow_compare --variants mean_l10 --base base --out docs/interp
Writes <out>/cellflow_test.md, cellflow_test_summary.csv and cellflow_test_paired.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_analysis.analyze import paired_stats

from .chemcpa_compare import LABELS, METRICS, MODEL_TABLE, matrix, model_rows, summary_cell, variant_rows
from .core import DEFAULT_ROOT, read_jsonl

REPO = Path(__file__).resolve().parents[1]
REFS = {"cellflow/cellflow": "CellFlow (published)", "cellflow/cellflow_mmd": "CellFlow + MMD loss (published)",
        "reference/adjacent_dose_mixture": "Adjacent-dose mixture (real neighbour cells)",
        "reference/vehicle": "Vehicle cells (no drug effect)", "reference/replicate": "Replicate well (noise floor)"}
CHEMCPA = {"gaussmean_l100": "chemCPA-Gauss + mean consistency", "nbmean_l10": "chemCPA-NB + mean consistency"}
VARIANT_LABELS = {"base": "CellFlow retrain (lowest val MMD²)", "meanv_l10": "CellFlow + average-arrow term",
                  "z70_k8_res_lr3e4": "CellFlow latent, 8 conditions per update"}


def cellflow_rows(root: Path, variant: str, split: str, offset: int) -> pd.DataFrame:
    paths = sorted((root / "cellflow" / variant).glob(f"seed_*/{split}_sample_s{offset}/per_condition.jsonl"))
    if not paths:
        raise SystemExit(f"no {split} offset-{offset} evaluations for CellFlow variant {variant}")
    rows = [{**r, "seed": int(p.parent.parent.name.split("_")[1])} for p in paths for r in read_jsonl(p)]
    return pd.DataFrame(rows)[["seed", "condition", *METRICS]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--base", default="base", help="the matched baseline (same selection rule)")
    p.add_argument("--labels", nargs="*", default=[], help="variant=label pairs for the tables")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--robustness-offset", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=str(REPO / "docs" / "interp"))
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(MODEL_TABLE)
    names = {**VARIANT_LABELS, **dict(x.split("=", 1) for x in a.labels)}
    a.variants = [v for v in a.variants if v != a.base]
    label = lambda k: {**REFS, **{f"chemcpa:{v}": t for v, t in CHEMCPA.items()}}.get(
        k, names.get(k.removeprefix("cellflow:"), k.removeprefix("cellflow:")))
    rng = np.random.default_rng(0)
    md = ["# CellFlow variants vs the matched base, published CellFlow and references", "",
          "Frozen evaluator, 3 training seeds (± = SD over seeds). Every CellFlow run here uses the minimum "
          "validation MMD² checkpoint; the published CellFlow rows used a WRF rule. Δ = ref − new on condition "
          "means; > 0 means the new variant is better. CI: condition bootstrap (two-level condition × seed CI "
          "in the CSV).", ""]
    summary, paired = [], []
    for split in a.splits:
        ents = {f"cellflow:{v}": cellflow_rows(root, v, split, a.offset) for v in [*a.variants, a.base]}
        ents.update({f"chemcpa:{v}": variant_rows(v, split, 0) for v in CHEMCPA})
        ents.update({k: model_rows(table, k, split) for k in REFS})
        n_cond = ents[f"cellflow:{a.base}"].condition.nunique()
        md += [f"## {split} ({n_cond} conditions)", "",
               "| | " + " | ".join(LABELS[m] for m in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for k, df in ents.items():
            md.append(f"| {label(k)} | " + " | ".join(summary_cell(df, m) for m in METRICS) + " |")
            summary.append({"split": split, "entity": k, "label": label(k),
                            **{m: df.groupby("condition")[m].mean().mean() for m in METRICS}})
        if split == "test" and a.robustness_offset is not None:
            again = {v: cellflow_rows(root, v, split, a.robustness_offset) for v in [*a.variants, a.base]}
            md += ["", f"Second sampling seed (offset {a.robustness_offset}), MMD² / RMSE DEG50: " + "; ".join(
                f"{label('cellflow:' + v)} {df.groupby('condition').mmd2.mean().mean():.4f} / "
                f"{df.groupby('condition').rmse_mean_deg50.mean().mean():.3f}" for v, df in again.items())]
        md.append("")
        refs = [f"cellflow:{a.base}", *[f"chemcpa:{v}" for v in CHEMCPA], *REFS]
        md += [f"### {split}: paired differences", "",
               "| new | ref | " + " | ".join(LABELS[m] for m in METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
        for v in a.variants:
            new = f"cellflow:{v}"
            for ref in refs:
                cells = []
                for m in METRICS:
                    s = paired_stats(matrix(ents[ref], m), matrix(ents[new], m), rng, a.n_boot)
                    paired.append({"split": split, "ref": ref, "new": new, "metric": m, **s})
                    f = (lambda x: f"{x:+.4f}") if m == "mmd2" else (lambda x: f"{x:+.3f}")
                    cells.append(f"{f(s['delta_mean'])} [{f(s['ci95_low'])}, {f(s['ci95_high'])}] "
                                 f"{s['n_win']}/{s['n_conditions']}")
                md.append(f"| {label(new)} | {label(ref)} | " + " | ".join(cells) + " |")
        md.append("")
    pd.DataFrame(summary).to_csv(out / "cellflow_test_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "cellflow_test_paired.csv", index=False)
    (out / "cellflow_test.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
