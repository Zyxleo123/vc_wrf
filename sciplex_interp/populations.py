"""Population generators built from the adjacent doses' real cells, scored on population metrics.

The diagnostics showed that the adjacent-dose mixture has nearly the right shape and the wrong
mean. These generators keep the mixture's cells and correct the mean with a statistics predictor
(sciplex_interp.stats), whose predictions are read from <root>/stats/means/.

  adjacent           the published recipe: log1p-dose weight, replicate-matched sampling
  adjacent_w{w}      fixed upper-dose share w
  mixw:<pred>        mixture with the neighbour weight <pred> implies (e.g. the Hill progress weight)
  shift:<pred>       log-dose mixture, every cell shifted by m̂ − (sample mean): the population mean
                     is exactly the predicted mean m̂
  shiftw:<pred>      the same, starting from the mixture with <pred>'s own weight
  shift0:<pred>      `shift`, then negative values set to 0 (keeps expression values valid)
  tilt:<pred>        the log-dose mixture's two pools exponentially tilted (max-entropy reweighting)
                     so that their weighted mean matches m̂ in the top-10 principal directions of
                     the training mean shifts; cells are resampled unchanged

usage: python -m sciplex_interp.populations --task-sets inner val --predictors linear hill ...
Writes <root>/populations/rows{tag}.jsonl. Finished (task set, generator) files are skipped.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .core import DEFAULT_PANEL, DEFAULT_ROOT, METRICS, N_PRED, Data, Task, read_jsonl, score, write_jsonl

FIXED_W = (0.2, 0.3)
TILT_K = 10


def load_means(root: Path, predictor: str) -> dict:
    """Predicted means of a stats predictor; "a+b" averages predictors a and b (an ensemble)."""
    parts = []
    for name in predictor.split("+"):
        z = np.load(root / "stats" / "means" / f"{name}.npz")
        parts.append({tuple(json.loads(k)): m for k, m in zip(z["keys"], z["m_hat"])})
    return {k: np.mean([part[k] for part in parts], 0) for k in parts[0]}


def load_weights(root: Path) -> dict:
    out = {}
    for r in read_jsonl(root / "stats" / "rows.jsonl"):
        if "w" in r and r["w"] is not None:
            out[(r["predictor"], r["task_set"], r["condition"])] = float(r["w"])
    return out


def basis(data: Data, task: Task, k: int) -> np.ndarray:
    """Top-k principal directions of the available training mean shifts (inner: target dropped)."""
    train = data.cond[data.cond.split == "train"].index
    D = np.array([data.mean(c) - data.ctrl_mean for c in train if not (task.task_set == "inner" and c == task.target)])
    return np.linalg.svd(D, full_matrices=False)[2][:k]


def tilt(data: Data, task: Task, m_hat: np.ndarray, w: float, rng: np.random.Generator, V: np.ndarray):
    """Max-entropy reweighting of the replicate-matched adjacent pools toward mean m̂ in span(V)."""
    n1 = int(round(N_PRED * task.n_rep1 / (task.n_rep1 + task.n_rep2)))
    picks = []
    for rep, take in (("rep1", n1), ("rep2", N_PRED - n1)):
        lo = data.idx(data.condition_of(task.drug, task.low), rep)
        hi = data.idx(data.condition_of(task.drug, task.high), rep)
        cells = np.r_[lo, hi]
        base = np.r_[np.full(len(lo), (1 - w) / len(lo)), np.full(len(hi), w / len(hi))]
        phi = (data.X[cells].astype(np.float64) - data.ctrl_mean) @ V.T
        target = (m_hat - data.ctrl_mean) @ V.T
        scale = phi.std(0) + 1e-9
        phi_s, target_s = phi / scale, target / scale
        lb = np.log(base)

        def dual(lam):
            a = lb + phi_s @ lam
            amax = a.max()
            q = np.exp(a - amax)
            Z = q.sum()
            return amax + np.log(Z) - lam @ target_s, (q / Z) @ phi_s - target_s

        res = minimize(dual, np.zeros(len(target_s)), jac=True, method="L-BFGS-B")
        a = lb + phi_s @ res.x
        q = np.exp(a - a.max())
        q /= q.sum()
        picks.append(rng.choice(cells, take, p=q))
    return np.concatenate(picks)


def run(data: Data, root: Path, task_sets, predictors, seeds: int, tag: str, kinds) -> None:
    out = root / "populations"
    means = {p: load_means(root, p) for p in predictors}
    weights = load_weights(root)
    plan = ["adjacent", *(f"adjacent_w{w}" for w in FIXED_W)]
    weighted = {k[0] for k in weights}
    for p in predictors:
        for kind in kinds:
            if kind in ("mixw", "shiftw") and p not in weighted:
                continue
            plan.append(f"{kind}:{p}")
    for ts in task_sets:
        path = out / f"{ts}{tag}.jsonl"
        done = {(r["generator"], r["condition"], r["seed"]) for r in read_jsonl(path)} if path.exists() else set()
        rows = read_jsonl(path) if path.exists() else []
        tasks = data.tasks(ts)
        V = {t.target: basis(data, t, TILT_K) for t in tasks} if any(g.startswith("tilt:") for g in plan) else {}
        for g in plan:
            kind, _, pred = g.partition(":")
            for t in tasks:
                X_real = data.X[data.idx(t.target)]
                for seed in range(seeds):
                    if (g, t.target, seed) in done:
                        continue
                    # common random numbers: every generator draws its mixture from the same stream
                    rng = np.random.default_rng([seed, zlib.crc32(t.target.encode()), zlib.crc32(ts.encode())])
                    w = t.w_logdose
                    if kind.startswith("adjacent_w"):
                        w = float(kind.removeprefix("adjacent_w"))
                    elif kind in ("mixw", "shiftw"):
                        w = weights[(pred, ts, t.target)]
                    m_hat = means[pred][(ts, t.target)] if pred else None
                    if kind == "tilt":
                        Xp = data.X[tilt(data, t, m_hat, w, rng, V[t.target])]
                    else:
                        Xp = data.X[data.mixture(t, {t.low: 1 - w, t.high: w}, rng)]
                        if kind.startswith("shift"):
                            Xp = Xp + (m_hat - Xp.mean(0)).astype(np.float32)
                            if kind == "shift0":
                                Xp = np.maximum(Xp, 0)
                    rows.append({"task_set": ts, "condition": t.target, "drug": t.drug, "generator": g,
                                 "seed": seed, "w": w, **score(data, X_real, Xp, seed=seed)})
            write_jsonl(path, rows)
            print(f"populations {ts} {g}: done", flush=True)
        import pandas as pd
        df = pd.DataFrame(rows)
        print(df.groupby("generator")[METRICS].mean().sort_values("mmd2").to_string(float_format=lambda v: f"{v:.4f}"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-sets", nargs="+", default=["inner", "val"])
    p.add_argument("--task-index", type=int, help="run only --task-sets[i] (array jobs)")
    p.add_argument("--predictors", nargs="*", default=[])
    p.add_argument("--kinds", nargs="+", default=["mixw", "shift", "shiftw", "shift0", "tilt"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--tag", default="")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    task_sets = a.task_sets if a.task_index is None else [a.task_sets[a.task_index]]
    run(Data(a.data), Path(a.root), task_sets, a.predictors, a.seeds, a.tag, a.kinds)


if __name__ == "__main__":
    main()
