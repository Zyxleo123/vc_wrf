"""Mean-level dose-response "teacher": the MLP auto-decoder of sciplex_interp.stats, for CellFlow.

Fit on the 78 training-condition means only (the auto-decoder's own recipe: a learned 4-number code
per drug + log10 dose -> top-20 principal directions of the training mean shifts, 5 seeds averaged).
It then predicts every condition of the panel, and optionally a dose grid, from labels alone: no
held-out cell or held-out mean is read. Runs in the main environment (it needs torch, CellFlow's
environment does not have it) and leaves a file for the CellFlow jobs.

usage: python -m sciplex_interp.teacher
Writes <root>/teacher/autodec.npz: keys (JSON [drug, dose]), coef (n x 20, unscaled coordinates of the
mean shift in `basis`), basis (20 x genes), control_mean, scale (SD of the training coordinates).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, mean_errors
from .stats import K_BASIS, Context, autodecoder

GRID_NM = (10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0)


def fit(data: Data, k_e: int, wd: float, grid=GRID_NM) -> dict:
    ctx = Context(data, data.tasks("val")[0])  # val and test tasks share one context: all training means
    assert len(ctx.conds) == 78
    queries = sorted({(d, float(g)) for d in data.drugs for g in grid} |
                     {(r.drug, float(r.dose_nM)) for r in data.cond.itertuples()})
    predicted = np.array(autodecoder(ctx, queries, k_e, wd))
    basis = ctx.V[:K_BASIS]
    coef = (predicted - ctx.m0) @ basis.T
    return {"keys": np.array([json.dumps(q) for q in queries]), "coef": coef.astype(np.float32),
            "basis": basis.astype(np.float32), "control_mean": ctx.m0.astype(np.float32),
            "scale": np.float32((ctx.D @ basis.T).std())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--name", default="autodec")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    data = Data(a.data)
    out = fit(data, a.embedding, a.weight_decay)
    path = Path(a.root) / "teacher" / f"{a.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    # Sanity on validation only (test means are not read here).
    lookup = {tuple(json.loads(k)): c for k, c in zip(out["keys"], out["coef"])}
    errors = []
    for t in data.tasks("val"):
        m = out["control_mean"].astype(np.float64) + lookup[(t.drug, t.dose)].astype(np.float64) @ out["basis"]
        errors.append(mean_errors(data, t.target, m)["rmse_mean_deg50"])
    print(f"{path}: {len(out['keys'])} (drug, dose) points; validation DEG50 RMSE of the teacher mean {np.mean(errors):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
