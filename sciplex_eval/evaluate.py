"""Evaluate a method's predictions on the held-out conditions of one split.

Prediction formats (one file per method x seed x split):
  generated cells  .h5ad, var_names = spec gene ids (any order, extra genes ignored),
                   X = log1p(CP10K) (dense or sparse), obs["condition"] = condition id
                   exactly as in the panel's obs["condition"], e.g. "A549|Belinostat (PXD101)|1000".
                   Generate at least as many cells as were measured (n_cells in the split file).
  fractions only   .csv with columns condition,q,wrf (q in spec.quantiles), for methods
                   such as direct WRF regression; only WRF errors are reported.

usage:
  python -m sciplex_eval.evaluate --pred preds.h5ad --method fm_base --seed 0 --split val --out results/fm_base/val_s0
  python -m sciplex_eval.evaluate --reference replicate --split val --out results/replicate/val
Writes <out>/per_condition.jsonl and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .metrics import evaluate_condition, summarize
from .spec import EvalSpec

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO / "artifacts" / "eval_spec_v1"
DEFAULT_PANEL = os.environ.get("SCIPLEX_PANEL", "/data/user_data/yixiz/sciplex3/processed/a549_hdaci_v1.h5ad")


def load_panel(path: str | Path = DEFAULT_PANEL) -> ad.AnnData:
    return ad.read_h5ad(path)


def split_conditions(panel: ad.AnnData, split: str) -> list[str]:
    obs = panel.obs
    return sorted(obs.loc[obs["split"] == split, "condition"].unique())


def _real(panel: ad.AnnData, spec: EvalSpec, cond: str, replicate: str | None = None):
    m = (panel.obs["condition"] == cond).values
    if replicate is not None:
        m &= (panel.obs["replicate"] == replicate).values
    return spec.align(panel.X[m], panel.var_names)


def evaluate_predictions(spec: EvalSpec, panel: ad.AnnData, pred, split: str, seed: int = 0) -> list[dict]:
    """pred: AnnData of generated cells, or DataFrame with columns condition,q,wrf."""
    conds = split_conditions(panel, split)
    records = []
    if isinstance(pred, ad.AnnData):
        Xp = spec.align(pred.X, pred.var_names)
        pc = pred.obs["condition"].astype(str).values
        missing = sorted(set(conds) - set(pc))
        if missing:
            raise ValueError(f"no predicted cells for {len(missing)} {split} conditions, e.g. {missing[:3]}")
        for c in conds:
            rec = evaluate_condition(spec, _real(panel, spec, c), X_pred=Xp[pc == c], seed=seed)
            records.append({"condition": c, **rec})
    else:
        table = pred.pivot_table(index="condition", columns="q", values="wrf")
        table.columns = np.round(table.columns.astype(float), 6)
        for c in conds:
            if c not in table.index:
                raise ValueError(f"no predicted fraction for {c}")
            wrf = {q: float(table.loc[c, round(q, 6)]) for q in spec.quantiles}
            records.append({"condition": c, **evaluate_condition(spec, _real(panel, spec, c), wrf_pred=wrf)})
    return records


def replicate_reference(spec: EvalSpec, panel: ad.AnnData, split: str, seed: int = 0) -> list[dict]:
    """Noise floor: score one replicate well's cells against the other's (both directions, averaged)."""
    records = []
    for c in split_conditions(panel, split):
        a, b = (_real(panel, spec, c, r) for r in ("rep1", "rep2"))
        r1 = evaluate_condition(spec, a, X_pred=b, seed=seed)
        r2 = evaluate_condition(spec, b, X_pred=a, seed=seed)
        records.append({"condition": c, **{k: (r1[k] + r2[k]) / 2 for k in r1}})
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", help=".h5ad of generated cells or .csv of fractions")
    p.add_argument("--reference", choices=["replicate"], help="score a built-in reference instead of --pred")
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--method", default=None)
    p.add_argument("--seed", type=int, default=0, help="the method's training seed (recorded only)")
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if (a.pred is None) == (a.reference is None):
        p.error("pass exactly one of --pred / --reference")

    spec = EvalSpec.load(a.spec)
    panel = load_panel(a.data)
    if a.reference:
        records, method = replicate_reference(spec, panel, a.split), f"reference_{a.reference}"
    else:
        pred = ad.read_h5ad(a.pred) if a.pred.endswith(".h5ad") else pd.read_csv(a.pred)
        records, method = evaluate_predictions(spec, panel, pred, a.split), a.method or Path(a.pred).stem

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"method": method, "seed": a.seed, "split": a.split, "spec_version": spec.version,
            "pred": a.pred, "data": str(a.data)}
    with open(out / "per_condition.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({**meta, **r}) + "\n")
    summary = {**meta, **summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{method} | {a.split} | {summary['n_conditions']} conditions (mean [95% CI over conditions])")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:24s} {v['mean']:9.4f}  [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
