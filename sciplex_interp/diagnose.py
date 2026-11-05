"""Where does the adjacent-dose mixture's advantage come from, and where do models lose?

  structure   train-condition mean shifts: SVD spectrum, per-drug dose norms and cosines
  floors      sampling-noise floor of each target's measured mean (no method can beat it)
  shapes      population candidates built from real cells of the adjacent doses, on the inner
              and val tasks: mixture weight grid, the published log-dose mixture, the same
              mixture shifted to the *measured* target mean (oracle: how much of the error is
              the mean), single doses shifted to the interpolated mean, and OT displacement
              interpolation between the two adjacent doses
  chemcpa     existing chemCPA checkpoints: in-sample fit on every training condition, the
              direct val prediction, and a log-dose mixture of chemCPA's *own* generated
              adjacent-dose cells (fit error vs interpolation error)

usage: python -m sciplex_interp.diagnose {real,chemcpa} --root results/interp
Writes <root>/diagnose/*.jsonl|json. Validation and inner tasks only; test is not read.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import DEFAULT_PANEL, DEFAULT_ROOT, METRICS, N_PRED, Data, mean_errors, score, write_jsonl

W_GRID = [round(0.1 * k, 1) for k in range(11)]
SEEDS = 5


def structure(data: Data) -> dict:
    rows, D = [], []
    for c, r in data.cond[data.cond.split == "train"].iterrows():
        d = data.mean(c) - data.ctrl_mean
        D.append(d)
        rows.append({"condition": c, "drug": r.drug, "dose": float(r.dose_nM), "norm": float(np.linalg.norm(d)),
                     "norm_pca": float(np.linalg.norm(data.Z[data.idx(c)].mean(0) - data.Z[data.ctrl].mean(0)))})
    D = np.array(D)
    s = np.linalg.svd(D, compute_uv=False)
    ev = s ** 2 / (s ** 2).sum()
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    for row, d in zip(rows, D):
        row["pc1_coord"] = float(d @ Vt[0])
        row["cos_pc1"] = float(abs(d @ Vt[0]) / (np.linalg.norm(d) + 1e-12))
    cos = []
    for drug in data.drugs:
        doses = data.train_doses(drug)
        vecs = {dd: data.mean(data.condition_of(drug, dd)) - data.ctrl_mean for dd in doses}
        for i, a in enumerate(doses):
            for b in doses[i + 1:]:
                va, vb = vecs[a], vecs[b]
                cos.append({"drug": drug, "a": a, "b": b,
                            "cos": float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))})
    return {"explained_variance": ev[:20].tolist(), "conditions": rows, "dose_cosines": cos}


def floors(data: Data, task_sets) -> list[dict]:
    """E||m_real - mu||^2 = sum_g var_g / n: the RMSE a perfect mean predictor still has."""
    out = []
    for ts in task_sets:
        for t in data.tasks(ts):
            X = data.X[data.idx(t.target)].astype(np.float64)
            n = len(X)
            var = X.var(0, ddof=1)
            d_real = X.mean(0) - data.spec.control_mean
            deg = np.argsort(-np.abs(d_real), kind="stable")[: data.spec.deg_top_k]
            out.append({"task_set": ts, "condition": t.target, "n_real": n,
                        "floor_rmse_mean_all": float(np.sqrt(var.mean() / n)),
                        "floor_rmse_mean_deg50": float(np.sqrt(var[deg].mean() / n)),
                        "floor_rmse_mean_program": float(np.sqrt(var[data.spec.program_idx].mean() / n)),
                        # a perfect predictor's own N_PRED-cell sample adds its own noise
                        "floor512_rmse_mean_deg50": float(np.sqrt(var[deg].mean() * (1 / n + 1 / N_PRED)))})
    return out


def ot_pairs(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    d2 = (Za * Za).sum(1)[:, None] + (Zb * Zb).sum(1)[None, :] - 2 * Za @ Zb.T
    _, cols = linear_sum_assignment(d2)
    return cols


def shapes(data: Data, task_sets) -> list[dict]:
    rows = []
    for ts in task_sets:
        for t in data.tasks(ts):
            X_real = data.X[data.idx(t.target)]
            w0 = t.w_logdose
            m_lo, m_hi = data.mean(data.condition_of(t.drug, t.low)), data.mean(data.condition_of(t.drug, t.high))
            m_interp = (1 - w0) * m_lo + w0 * m_hi
            base = {"task_set": ts, "condition": t.target, "drug": t.drug, "dose": t.dose, "low": t.low,
                    "high": t.high, "w_logdose": w0}
            rows.append({**base, "method": "mean:linear_logdose", "seed": 0, **mean_errors(data, t.target, m_interp)})
            rows.append({**base, "method": "mean:vehicle", "seed": 0, **mean_errors(data, t.target, data.ctrl_mean)})
            for w in W_GRID:
                rows.append({**base, "method": f"mean:linear_w{w}", "seed": 0,
                             **mean_errors(data, t.target, (1 - w) * m_lo + w * m_hi)})
            for seed in range(SEEDS):
                rng = np.random.default_rng([seed, zlib.crc32(t.target.encode()), zlib.crc32(ts.encode())])

                def add(method, X_pred, **extra):
                    rows.append({**base, "method": method, "seed": seed, **extra,
                                 **score(data, X_real, X_pred, seed=seed)})

                vehicle = rng.choice(data.ctrl, N_PRED)
                add("vehicle", data.X[vehicle])
                for w in W_GRID:
                    add(f"mix_w{w}", data.X[data.mixture(t, {t.low: 1 - w, t.high: w}, rng)])
                idx = data.mixture(t, {t.low: 1 - w0, t.high: w0}, rng)
                X_mix = data.X[idx]
                add("mix_logdose", X_mix)
                add("mix_logdose+oracle_mean", X_mix + (data.mean(t.target) - X_mix.mean(0)).astype(np.float32))
                add("mix_logdose+interp_mean", X_mix + (m_interp - X_mix.mean(0)).astype(np.float32))
                lo = data.mixture(t, {t.low: 1.0}, rng)
                hi = data.mixture(t, {t.high: 1.0}, rng)
                X_lo_shift = data.X[lo] + (m_interp - m_lo).astype(np.float32)
                X_hi_shift = data.X[hi] + (m_interp - m_hi).astype(np.float32)
                add("shift_low", X_lo_shift)
                add("shift_high", X_hi_shift)
                half = N_PRED // 2
                add("shift_both", np.vstack([X_lo_shift[:half], X_hi_shift[half:]]))
                cols = ot_pairs(data.Z[lo], data.Z[hi])
                add("ot_displacement", ((1 - w0) * data.X[lo] + w0 * data.X[hi][cols]).astype(np.float32))
            print(f"shapes {ts} {t.target}", flush=True)
    return rows


# ---- chemCPA: fit vs interpolation --------------------------------------------------------------
def chemcpa(data: Data, root: Path, variants=("nb_v1", "gauss_v1"), seeds=(0, 1, 2)) -> list[dict]:
    import torch

    from sciplex_baselines.chemcpa import (generate, make_model, n_features, n_outputs, output_gene_ids,
                                           prediction_inputs)
    runs = Path(__file__).resolve().parents[1] / "results" / "chemcpa"
    rows = []
    train = data.cond[data.cond.split == "train"]
    for variant in variants:
        for seed in seeds:
            ck = torch.load(runs / variant / f"seed_{seed}" / "model.pt", map_location="cpu", weights_only=True)
            config = ck["config"]
            model = make_model(n_features(data.spec, config["likelihood"]), ck["fingerprints"], config, ck["seed"],
                               "cpu", n_outputs(data.spec, config))
            model.load_state_dict(ck["model"])
            controls, _ = prediction_inputs(data.adata, data.spec, "val", config["likelihood"])
            gene_ids = output_gene_ids(data.spec, config)
            conds = [{"condition": c, "drug": r.drug, "dose_nM": float(r.dose_nM), "n_cells": 1}
                     for c, r in train.iterrows()]
            conds += [{"condition": t.target, "drug": t.drug, "dose_nM": t.dose, "n_cells": 1} for t in data.tasks("val")]
            pred = generate(model, controls, conds, ck["drugs"], gene_ids, n_cells=N_PRED, seed=seed + 50000)
            Xp = data.spec.align(pred.X, pred.var_names)
            pc = pred.obs.condition.to_numpy()
            gen = {c["condition"]: np.asarray(Xp[pc == c["condition"]], dtype=np.float32) for c in conds}
            base = {"variant": variant, "seed": seed, "epoch": ck["epoch"]}
            for c, r in train.iterrows():
                rows.append({**base, "kind": "train_fit", "condition": c, "dose": float(r.dose_nM),
                             **score(data, data.X[data.idx(c)], gen[c], seed=seed)})
            rng = np.random.default_rng(seed)
            for t in data.tasks("val"):
                X_real = data.X[data.idx(t.target)]
                rows.append({**base, "kind": "val_direct", "condition": t.target,
                             **score(data, X_real, gen[t.target], seed=seed)})
                w = t.w_logdose
                lo, hi = gen[data.condition_of(t.drug, t.low)], gen[data.condition_of(t.drug, t.high)]
                take_hi = rng.random(N_PRED) < w
                own = np.vstack([lo[rng.integers(len(lo), size=(~take_hi).sum())],
                                 hi[rng.integers(len(hi), size=take_hi.sum())]])
                rows.append({**base, "kind": "val_own_mixture", "condition": t.target,
                             **score(data, X_real, own, seed=seed)})
            print(f"chemcpa {variant} seed {seed} epoch {ck['epoch']}", flush=True)
    return rows


def summarize(rows: list[dict], keys: list[str]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = [m for m in METRICS if m in df]
    return df.groupby(keys)[cols].mean().to_string(float_format=lambda v: f"{v:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["real", "chemcpa"])
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    out = Path(a.root) / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    data = Data(a.data)
    if a.what == "real":
        (out / "structure.json").write_text(json.dumps(structure(data), indent=1))
        write_jsonl(out / "floors.jsonl", floors(data, ("inner", "val")))
        rows = shapes(data, ("inner", "val"))
        write_jsonl(out / "shapes.jsonl", rows)
        print(summarize(rows, ["task_set", "method"]))
    else:
        rows = chemcpa(data, Path(a.root))
        write_jsonl(out / "chemcpa.jsonl", rows)
        print(summarize(rows, ["variant", "kind"]))


if __name__ == "__main__":
    main()
