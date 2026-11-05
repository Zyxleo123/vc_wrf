"""Score a chemCPA run on population metrics: in-sample fit and held-out doses.

For one (variant, seed) checkpoint, cells are generated from permitted controls for every
training condition and every target of the requested task sets (512 cells each), and scored by
sciplex_interp.core.score:

  train_fit      generated vs measured cells of each *training* condition (in-sample)
  direct         the held-out-dose prediction itself
  own_mixture    log-dose mixture of chemCPA's own generated adjacent-dose cells
  oracle_shift   `direct` shifted onto the measured target mean (diagnostic: is the shape right?)
  shift:<pred>   `direct` shifted onto a statistics predictor's mean (sciplex_interp.stats)

usage: python -m sciplex_interp.chemcpa_eval --variant nb_v1 --seed 0 --task-sets val
Writes <root>/chemcpa_eval/<variant>/seed_<seed>_<task set>.jsonl; a file whose checkpoint hash
differs from the current checkpoint is recomputed, a matching one is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .core import DEFAULT_PANEL, DEFAULT_ROOT, METRICS, N_PRED, Data, read_jsonl, score, write_jsonl

REPO = Path(__file__).resolve().parents[1]


def run_dir(variant: str, seed: int) -> Path:
    roots = [Path(os.environ.get("CHEMCPA_ROOT", REPO / "results" / "interp" / "chemcpa")),
             Path(os.environ.get("BASE_CHEMCPA_ROOT", REPO / "results" / "chemcpa"))]
    for root in roots:
        if (root / variant / f"seed_{seed}" / "model.pt").is_file():
            return root / variant / f"seed_{seed}"
    raise SystemExit(f"no checkpoint for {variant} seed {seed} under {[str(r) for r in roots]}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_all(data: Data, run: Path, conditions: list[dict], seed: int) -> dict:
    import torch

    from sciplex_baselines.chemcpa import generate, make_model, n_features, n_outputs, output_gene_ids, prediction_inputs
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    config = ck["config"]
    if config.get("output_genes", "all") != "all":
        raise SystemExit("program-output runs have no full-gene cells to score")
    model = make_model(n_features(data.spec, config["likelihood"]), ck["fingerprints"], config, ck["seed"], "cpu",
                       n_outputs(data.spec, config))
    model.load_state_dict(ck["model"])
    controls, _ = prediction_inputs(data.adata, data.spec, "val", config["likelihood"])
    pred = generate(model, controls, conditions, ck["drugs"], output_gene_ids(data.spec, config), n_cells=N_PRED,
                    seed=seed + 50000)
    X = np.asarray(data.spec.align(pred.X, pred.var_names), dtype=np.float32)
    pc = pred.obs.condition.to_numpy()
    return {c["condition"]: X[pc == c["condition"]] for c in conditions}, int(ck["epoch"])


def evaluate(data: Data, root: Path, variant: str, seed: int, task_sets, shift_predictors) -> None:
    run = run_dir(variant, seed)
    digest = sha256(run / "model.pt")
    out = root / "chemcpa_eval" / variant
    todo = []
    for ts in task_sets:
        path = out / f"seed_{seed}_{ts}.jsonl"
        have = read_jsonl(path) if path.exists() else []
        if (have and have[0].get("checkpoint_sha256") == digest
                and {f"shift:{p}" for p in shift_predictors} <= {r["kind"] for r in have}):
            print(f"{variant} seed {seed} {ts}: up to date", flush=True)
            continue
        todo.append(ts)
    if not todo:
        return
    train = data.cond[data.cond.split == "train"]
    conds = [{"condition": c, "drug": r.drug, "dose_nM": float(r.dose_nM), "n_cells": 1} for c, r in train.iterrows()]
    tasks = {ts: data.tasks(ts) for ts in todo}
    conds += [{"condition": t.target, "drug": t.drug, "dose_nM": t.dose, "n_cells": 1}
              for ts in todo for t in tasks[ts] if t.target not in train.index]
    gen, epoch = generate_all(data, run, conds, seed)
    from .populations import load_means
    means = {p: load_means(root, p) for p in shift_predictors}
    base = {"variant": variant, "seed": seed, "epoch": epoch, "checkpoint_sha256": digest, "run": str(run)}
    fit_rows = [{**base, "kind": "train_fit", "condition": c, "dose": float(r.dose_nM),
                 **score(data, data.X[data.idx(c)], gen[c], seed=seed)} for c, r in train.iterrows()]
    for ts in todo:
        rows = [{**r, "task_set": ts} for r in fit_rows]
        rng = np.random.default_rng(seed)
        for t in tasks[ts]:
            X_real = data.X[data.idx(t.target)]
            Xg = gen[t.target]
            add = lambda kind, Xp: rows.append({**base, "task_set": ts, "kind": kind, "condition": t.target,
                                                 **score(data, X_real, Xp, seed=seed)})
            add("direct", Xg)
            w = t.w_logdose
            lo, hi = gen[data.condition_of(t.drug, t.low)], gen[data.condition_of(t.drug, t.high)]
            take_hi = rng.random(N_PRED) < w
            add("own_mixture", np.vstack([lo[rng.integers(len(lo), size=(~take_hi).sum())],
                                          hi[rng.integers(len(hi), size=take_hi.sum())]]))
            add("oracle_shift", Xg + (data.mean(t.target) - Xg.mean(0)).astype(np.float32))
            for p, m in means.items():
                add(f"shift:{p}", Xg + (m[(ts, t.target)] - Xg.mean(0)).astype(np.float32))
        write_jsonl(out / f"seed_{seed}_{ts}.jsonl", rows)
        import pandas as pd
        df = pd.DataFrame(rows)
        print(f"{variant} seed {seed} epoch {epoch} {ts}")
        print(df.groupby("kind")[METRICS].mean().to_string(float_format=lambda v: f"{v:.4f}"), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--task-sets", nargs="+", default=["val"])
    p.add_argument("--shift-predictors", nargs="*", default=[])
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    if "inner" in a.task_sets:
        p.error("inner targets are training conditions chemCPA was fit on; use val/test")
    evaluate(Data(a.data), Path(a.root), a.variant, a.seed, a.task_sets, a.shift_predictors)


if __name__ == "__main__":
    main()
