"""Marginal calibration of generated cells: give them the zero-inflation of real cells, gently.

Ours decodes a cell as `control cell + (dense 70-dim shift)`, so about half of its genes are
non-zero against the ~4% of a real sci-Plex3 cell. The per-gene EMD and, to a lesser degree, the
E-distance read that smear as error: on the unseen drugs plain vehicle cells score a better All EMD
(0.0282) than either Ours (0.0297) or DOLORIS. `sciplex_standard.mask_ours` showed that DOLORIS's
own remedy -- multiplying by the zero pattern of a random real cell of a nearby training condition
-- destroys Ours' cell geometry (E-distance 6.8x worse), because it deletes signal at random.

This calibration keeps the ordering instead. For each gene independently:

  1. the target zero fraction is the gene's zero fraction among the cell line's vehicle cells
     (available for any condition), blended with the generated one by `gamma`;
  2. the smallest generated values of that gene are set to zero until that fraction is reached;
  3. the mass removed is given back to the surviving cells in proportion, so the gene's mean over
     the generated cells -- the model's actual prediction -- is unchanged.

Step 2 is monotone per gene, so which cells are high and which are low is preserved, and step 3
leaves every mean-based metric (RMSE, and the mean part of the EMD) exactly where it was. gamma = 0
is the current model; gamma = 1 matches the vehicle zero rate. Chosen on `val`, not on the tables.

usage: python -m sciplex_standard.calibrate --root results/standard --method ours --splits val ood test
Writes <root>/preds/<method>_cal/seed_*/<split>.h5ad for the chosen gamma and
<root>/diagnose/calibrate.jsonl with the whole gamma grid scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .md import md_table
from .data import DEFAULT_FILE, REPO, StdData
from .evaluate import read_cells, score_condition
from .data import load_spec

ROOT = REPO / "results" / "standard"
GAMMAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def calibrate(G: np.ndarray, zero_rate: np.ndarray, gamma: float, delta: np.ndarray | None = None) -> np.ndarray:
    """Zero the smallest values of each gene up to the blended target rate; give the mass back.

    With `delta`, the mass given back is not the mass removed but whatever puts the gene's mean at
    (current mean + delta): the control-variate correction of `sciplex_standard.centre`, applied
    multiplicatively to the cells that are expressing instead of added to every cell, so that a
    zero stays a zero and the zero-inflated shape survives the correction.
    """
    if gamma <= 0 and delta is None:
        return G
    G = np.asarray(G, dtype=np.float64)
    n, n_genes = G.shape
    own = (G <= 0).mean(0)
    target = max(gamma, 0.0) * zero_rate + (1.0 - max(gamma, 0.0)) * own
    k = np.clip(np.rint(target * n).astype(int), 0, n - 1)          # cells to zero per gene
    order = np.argsort(G, axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[:, None].repeat(n_genes, 1), axis=0)
    drop = rank < k[None, :]
    total = G.sum(0) if delta is None else np.maximum(G.sum(0) + n * np.asarray(delta, np.float64), 0.0)
    out = np.where(drop, 0.0, G)
    kept = out.sum(0)
    scale = np.divide(total, kept, out=np.ones_like(total), where=kept > 0)
    return (out * scale).astype(np.float32)


def control_zero_rate(data: StdData) -> dict:
    return {c: (data.X[data.ctrl_idx(c)] <= 0).mean(0).astype(np.float64) for c in data.cell_types}


def zero_rate_curves(data: StdData, n_bins: int = 60) -> dict:
    """Per cell line, how often a gene is zero as a function of its mean, from real training cells.

    The vehicle zero rate is the wrong target for a gene the drug switches on: an induced gene is
    expressed in more cells than at baseline, not fewer. In log-normalised UMI data the fraction of
    cells in which a gene is zero is close to a fixed function of its mean (the dropout curve), so
    it can be read off the training conditions once and then applied to any predicted mean. Only
    `train` conditions are used.
    """
    curves = {}
    for c in data.cell_types:
        means, zeros = [], []
        for k in data.keys("train", min_cells=10):
            if data.cond.cell_type[k] != c:
                continue
            Xk = data.X[data.idx(k)]
            means.append(Xk.mean(0, dtype=np.float64))
            zeros.append((Xk <= 0).mean(0))
        x = np.log1p(np.concatenate(means))
        y = np.concatenate(zeros)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        xs = np.array([x[b == i].mean() for i in range(len(edges) - 1) if (b == i).any()])
        ys = np.array([y[b == i].mean() for i in range(len(edges) - 1) if (b == i).any()])
        ys = np.minimum.accumulate(ys)                      # a zero rate that falls as the mean rises
        curves[c] = (xs, ys)
    return curves


def mean_zero_rate(G: np.ndarray, curve: tuple) -> np.ndarray:
    xs, ys = curve
    return np.interp(np.log1p(np.asarray(G, np.float64).mean(0)), xs, ys)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--method", default="ours")
    p.add_argument("--splits", nargs="+", default=["val", "ood", "test"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    p.add_argument("--write-gamma", type=float, default=None, help="also write calibrated cells at this gamma")
    p.add_argument("--centre", action="store_true",
                   help="also remove the 100 source cells' deviation from the vehicle mean (control variate), "
                        "inside the transport so zeros stay zeros; Ours' source draw only")
    p.add_argument("--zero-rate", choices=["vehicle", "mean"], default="vehicle",
                   help="target zero fraction per gene: the vehicle's, or read off the training dropout curve "
                        "at the gene's predicted mean")
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.root) / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    data, spec = StdData(a.file), load_spec()
    zr = control_zero_rate(data)
    curves = zero_rate_curves(data) if a.zero_rate == "mean" else None
    target = (lambda G, ct: mean_zero_rate(G, curves[ct])) if curves else (lambda G, ct: zr[ct])
    tag = ("" if a.zero_rate == "vehicle" else "_meanzr") + ("_cv" if a.centre else "")
    if a.centre:
        from .centre import deltas
        dl = {seed: deltas(data, seed) for seed in a.seeds}
    delta_of = (lambda seed, k: dl[seed][data.cond.cell_type[k]]) if a.centre else (lambda seed, k: None)
    rows = []
    dest = out / f"calibrate_{a.method}{tag}.jsonl"
    with open(dest, "w", buffering=1) as fh:
        for split in a.splits:
            keys = data.keys(split)
            for seed in a.seeds:
                cells = read_cells(root / "preds" / a.method / f"seed_{seed}" / f"{split}.h5ad", data.genes)
                for gamma in a.gammas:
                    for k in keys:
                        Y = calibrate(cells[k], target(cells[k], data.cond.cell_type[k]), gamma, delta_of(seed, k))
                        rec = {"split": split, "method": a.method, "seed": seed, "gamma": float(gamma),
                               "condition": k, "drug": data.cond.drug[k],
                               "nonzero_fraction": float((Y > 0).mean())}
                        rec.update(score_condition(data, spec, k, np.asarray(Y, np.float32)))
                        fh.write(json.dumps(rec) + "\n")
                        rows.append(rec)
                    print(f"{split} seed {seed} gamma {gamma}: {len(keys)} conditions", flush=True)
    df = pd.DataFrame(rows)
    cols = ["rmse_all", "edist_all", "emd_all", "rmse_de20", "edist_de20", "emd_de20", "rmse_de40",
            "edist_de40", "emd_de40", "mmd2", "nonzero_fraction"]
    tab = df.groupby(["split", "gamma"])[cols].mean().round(4)
    (out / f"calibrate_{a.method}{tag}.md").write_text(
        f"# Marginal calibration of {a.method} (gamma = share of the vehicle zero rate imposed)\n\n"
        + md_table(tab) + "\n")
    print(tab.to_string())
    if a.write_gamma is not None:
        import anndata as ad
        for split in a.splits:
            keys = data.keys(split)
            for seed in a.seeds:
                cells = read_cells(root / "preds" / a.method / f"seed_{seed}" / f"{split}.h5ad", data.genes)
                X = np.vstack([calibrate(cells[k], target(cells[k], data.cond.cell_type[k]), a.write_gamma,
                                         delta_of(seed, k)) for k in keys])
                labels = np.concatenate([[k] * len(cells[k]) for k in keys])
                A = ad.AnnData(np.asarray(X, np.float32),
                               obs=pd.DataFrame({"condition": labels}, index=[f"g{i}" for i in range(len(labels))]),
                               var=pd.DataFrame(index=data.genes))
                d = root / "preds" / f"{a.method}_cal{tag}" / f"seed_{seed}" / f"{split}.h5ad"
                d.parent.mkdir(parents=True, exist_ok=True)
                A.write_h5ad(d, compression="gzip")
                print(f"wrote {d}", flush=True)


if __name__ == "__main__":
    main()
