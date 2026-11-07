"""Mean-level dose-response "teacher": the MLP auto-decoder of sciplex_interp.stats, for CellFlow.

Fit on the 78 training-condition means only (the auto-decoder's own recipe: a learned 4-number code
per drug + log10 dose -> principal directions of the training mean shifts, 5 seeds averaged).
It then predicts every condition of the panel, and optionally a dose grid, from labels alone: no
held-out cell or held-out mean is read. Runs in the main environment (it needs torch, CellFlow's
environment does not have it) and leaves a file for the CellFlow jobs.

usage: python -m sciplex_interp.teacher
Writes <root>/teacher/autodec.npz: keys (JSON [drug, dose]), coef (n x components, unscaled
coordinates of the mean shift in `basis`), basis (components x genes), control_mean, scale (SD of
the training coordinates), and the component count. There are 78 training-condition means, so 78
is the largest supported response basis without using held-out conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, mean_errors
from .stats import K_BASIS, Context, autodecoder_seeds

GRID_NM = (10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0)


def fit(data: Data, k_e: int, wd: float, components: int = K_BASIS, grid=GRID_NM) -> dict:
    ctx = Context(data, data.tasks("val")[0])  # val and test tasks share one context: all training means
    assert len(ctx.conds) == 78
    if not 1 <= components <= len(ctx.V):
        raise ValueError(f"components must be in [1, {len(ctx.V)}], got {components}")
    queries = sorted({(d, float(g)) for d in data.drugs for g in grid} |
                     {(r.drug, float(r.dose_nM)) for r in data.cond.itertuples()})
    coef = np.mean(autodecoder_seeds(ctx, queries, k_e, wd, n_basis=components)[0], axis=0)
    basis = ctx.V[:components]
    return {"keys": np.array([json.dumps(q) for q in queries]), "coef": coef.astype(np.float32),
            "basis": basis.astype(np.float32), "control_mean": ctx.m0.astype(np.float32),
            "scale": np.float32((ctx.D @ basis.T).std()),
            "components": np.int64(components)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding", type=int, default=4)
    p.add_argument("--components", type=int, default=K_BASIS)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--name", default="autodec")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    data = Data(a.data)
    out = fit(data, a.embedding, a.weight_decay, a.components)
    path = Path(a.root) / "teacher" / f"{a.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    # Sanity on validation only (test means are not read here).
    lookup = {tuple(json.loads(k)): c for k, c in zip(out["keys"], out["coef"])}
    errors = []
    for t in data.tasks("val"):
        m = out["control_mean"].astype(np.float64) + lookup[(t.drug, t.dose)].astype(np.float64) @ out["basis"]
        errors.append(mean_errors(data, t.target, m)["rmse_mean_deg50"])
    print(f"{path}: {len(out['keys'])} (drug, dose) points; {int(out['components'])} components; "
          f"validation DEG50 RMSE of the teacher mean {np.mean(errors):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
