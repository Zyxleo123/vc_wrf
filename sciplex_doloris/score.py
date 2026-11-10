"""Score every method on one split with the frozen evaluator and with DOLORIS's metrics.

Methods (cells as each method saved them; nothing is regenerated except the two references
that are defined by resampling measured cells):

  doloris_<name>_final / _bestval   this port, two checkpoint rules (generate.py), 3 seeds
  ours        CellFlow + condition response embedding + grouped velocity matching
              (results/interp/cellflow/z70_lr3e4_mv10_t), 3 seeds, 512 cells per condition
  cellflow    the retrained CellFlow baseline (results/interp/cellflow/base)
  chemcpa     original chemCPA-Gauss (results/chemcpa/gauss_v1), 1024 cells per condition
  adjacent    the adjacent-dose mixture, rebuilt with the recipe and random streams of
              sciplex_interp.cellflow_figures.collect (3 streams of 512 measured cells)
  vehicle     all vehicle cells as the prediction (no drug effect)
  replicate   one replicate well against the other, both directions averaged (noise floor;
              "real" is then one well, not the whole condition)

For each condition the row holds the frozen metrics (sciplex_eval.evaluate_condition: MMD²,
mean RMSE over all genes / DEG50 / program genes, W1(r)) and DOLORIS's nine metrics twice:
on the cells as saved, and with negative values set to 0 (DOLORIS applies that to its own
output before scoring, test.py; its own rows are identical both ways). Frozen metrics are
checked against each method's saved evaluator rows, so a mismatch in which cells were read
stops the job. DOLORIS predictions are first scored by the frozen evaluator CLI, which writes
per_condition.jsonl and summary.json next to them, as for every other method.

usage: python -m sciplex_doloris.score --split test --root results/doloris --name v1
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from sciplex_eval import evaluate as frozen_cli
from sciplex_eval.metrics import evaluate_condition
from sciplex_interp.core import DEFAULT_PANEL, FROZEN, REPO, Data, dense, read_jsonl, write_jsonl

from .generate import RULES
from .metrics import COLUMNS, de_genes, doloris_metrics

CHECKED = ("mmd2", "rmse_mean_all", "rmse_mean_deg50", "rmse_mean_program")


def file_methods(a, split: str) -> dict[str, list[tuple[int, Path]]]:
    roots = {f"doloris_{a.name}_{rule}": (Path(a.root) / "preds" / f"doloris_{a.name}_{rule}", "preds.h5ad") for rule in RULES}
    roots["ours"] = (Path(a.interp_root) / "cellflow" / a.ours, "preds.h5ad")
    roots["cellflow"] = (Path(a.interp_root) / "cellflow" / a.cellflow, "preds.h5ad")
    roots["chemcpa"] = (Path(a.chemcpa_root) / a.chemcpa, "predictions.h5ad")
    out = {}
    for method, (base, fname) in roots.items():
        paths = sorted(base.glob(f"seed_*/{split}_sample_s0/{fname}"))
        out[method] = [(int(p.parent.parent.name.split("_")[1]), p) for p in paths]
    return out


def read_cells(path: Path, spec) -> dict[str, np.ndarray]:
    a = ad.read_h5ad(path)
    X = dense(spec.align(a.X, a.var_names))
    labels = a.obs["condition"].astype(str).to_numpy()
    return {c: X[labels == c] for c in pd.unique(labels)}


def load_de(data: Data, root: Path) -> dict[str, list[int]]:
    path = root / "de_genes.json"
    targets = [t.target for s in ("val", "test") for t in data.tasks(s)]
    if path.exists():
        saved = json.loads(path.read_text())
        if set(targets) <= set(saved["conditions"]):
            ids = data.spec.gene_ids
            return {c: [ids.index(g) for g in v["gene_ids"]] for c, v in saved["conditions"].items()}
    vehicle = data.split_cfg["controls"]["condition"]
    de = de_genes(data.X, data.obs.condition.to_numpy(), data.spec.gene_ids, vehicle, targets)
    ids, sym = data.spec.gene_ids, data.spec.gene_symbols
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": "scanpy rank_genes_groups t-test vs all vehicle cells, rankby_abs, 50 genes "
                                          "(chemCPA rank_genes_groups_by_cov / lincs_DEGs)", "reference": vehicle,
                                "conditions": {c: {"gene_ids": [ids[i] for i in v], "symbols": [sym[i] for i in v]}
                                               for c, v in de.items()}}, indent=1) + "\n")
    return de


def score_pair(spec, real: np.ndarray, pred: np.ndarray, de: list[int]) -> dict:
    rec = evaluate_condition(spec, real, X_pred=pred, seed=0)
    out = {"n_real": len(real), "n_pred": len(pred), **{k: rec[k] for k in FROZEN},
           "negative_fraction": float((pred < 0).mean())}
    raw = doloris_metrics(real, pred, de)
    out.update({f"{k}": v for k, v in raw.items()})
    clipped = raw if out["negative_fraction"] == 0 else doloris_metrics(real, np.clip(pred, 0, None), de)
    out.update({f"{k}_clip0": v for k, v in clipped.items()})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--root", default=str(REPO / "results" / "doloris"))
    p.add_argument("--name", default="v1")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--interp-root", default=str(REPO / "results" / "interp"))
    p.add_argument("--chemcpa-root", default=str(REPO / "results" / "chemcpa"))
    p.add_argument("--ours", default="z70_lr3e4_mv10_t")
    p.add_argument("--cellflow", default="base")
    p.add_argument("--chemcpa", default="gauss_v1")
    a = p.parse_args(argv)

    root = Path(a.root)
    data = Data(a.data)
    spec, split = data.spec, a.split
    tasks = data.tasks(split)
    de = load_de(data, root)
    rows = []
    files = file_methods(a, split)
    for method, items in files.items():
        if not items:
            print(f"!! {method}: no {split} predictions found; left out")
            continue
        for seed, path in items:
            saved = path.parent / "per_condition.jsonl"
            if method.startswith("doloris_") and not saved.exists():
                frozen_cli.main(["--pred", str(path), "--method", method, "--seed", str(seed), "--split", split,
                                 "--data", a.data, "--out", str(path.parent)])
            ref = {r["condition"]: r for r in read_jsonl(saved)} if saved.exists() else {}
            cells = read_cells(path, spec)
            for t in tasks:
                rec = score_pair(spec, data.X[data.idx(t.target)], cells[t.target], de[t.target])
                if t.target in ref:  # float32 dense vs sparse panel means differ by ~1e-8; wrong cells by >> 1e-4
                    for k in CHECKED:
                        if abs(rec[k] - ref[t.target][k]) > 1e-7 + 1e-5 * abs(ref[t.target][k]):
                            raise SystemExit(f"{path}: {t.target} {k} {rec[k]:.10f} != saved {ref[t.target][k]:.10f}")
                rows.append({"method": method, "seed": seed, "condition": t.target, "source": str(path), **rec})
            print(f"{method} seed {seed}: {len(tasks)} conditions{' (matches saved rows)' if ref else ''}")

    for t in tasks:  # references built from measured cells
        real = data.X[data.idx(t.target)]
        for seed in range(3):  # cellflow_figures.collect's recipe and random stream for `adjacent`
            rng = np.random.default_rng([seed, zlib.crc32(t.target.encode()), zlib.crc32(split.encode())])
            pred = data.X[data.mixture(t, {t.low: 1 - t.w_logdose, t.high: t.w_logdose}, rng)]
            rows.append({"method": "adjacent", "seed": seed, "condition": t.target, **score_pair(spec, real, pred, de[t.target])})
        rows.append({"method": "vehicle", "seed": 0, "condition": t.target, **score_pair(spec, real, data.X[data.ctrl], de[t.target])})
        r1, r2 = data.X[data.idx(t.target, "rep1")], data.X[data.idx(t.target, "rep2")]
        one, two = score_pair(spec, r1, r2, de[t.target]), score_pair(spec, r2, r1, de[t.target])
        rows.append({"method": "replicate", "seed": 0, "condition": t.target,
                     **{k: (one[k] + two[k]) / 2 for k in one}})
    print(f"references: adjacent (3 streams), vehicle, replicate on {len(tasks)} conditions")

    out = root / "scores" / f"{split}.jsonl"
    write_jsonl(out, rows)
    df = pd.DataFrame(rows)
    cond_means = df.groupby(["method", "condition"])[["mmd2", "rmse_mean_deg50", *COLUMNS]].mean().groupby("method").mean()
    print(cond_means.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
