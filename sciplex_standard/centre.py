"""Control-variate correction of the generated mean: remove the error of the 100 source cells.

Ours generates a condition's 100 cells from 100 vehicle cells of its cell line, drawn once per seed
(`sciplex_standard.ours.sources`) and shared by every condition of that cell line. Decoding is
residual -- `x = x0 + (z1 - z0) @ U` -- so the generated mean is (mean of the 100 source cells) +
(predicted shift), and the first term carries that particular sample's deviation from the vehicle
population mean into every condition. On the unseen drugs that deviation is worth about 0.021 of
All RMSE (100 vehicle cells score 0.0640 against the real cells, the full vehicle mean 0.0603), none
of it about the drug.

The correction is a standard control variate: move every generated cell of a cell line by
delta = (mean of all vehicle cells of that line) - (mean of its 100 source cells). It uses only
vehicle cells, which are training data, and nothing of any held-out condition; it leaves the shape
of each generated cloud untouched (a constant shift, then the usual clip at zero). The same deviation
sits in any generator that starts from a fixed sample of controls, DOLORIS included.

The source draw is reproduced exactly (same pool, same order, same generator); `--check` verifies
it by testing that, for the weakest conditions, the generated mean is far closer to the reproduced
source mean than to an independent draw.

usage: python -m sciplex_standard.centre --method ours_v2 --splits val ood
Writes <root>/preds/<method>_ctr/seed_*/<split>.h5ad.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_FILE, REPO, StdData
from .evaluate import read_cells

ROOT = REPO / "results" / "standard"
N_GEN = 100


def source_rows(data: StdData, seed: int, offset: int = 0) -> dict:
    """The rows `ours.sources` drew: same pool order (training rows), same generator, same sequence."""
    rows = data.train_rows()
    ctrl = data.obs.control.to_numpy()[rows]
    ct = data.obs.cell_type.to_numpy()[rows]
    rng = np.random.default_rng(20000 + seed + 1000 * offset)
    out = {}
    for c in data.cell_types:                      # one generator, cell lines in this order
        pool = rows[ctrl & (ct == c)]
        out[c] = rng.choice(pool, N_GEN, replace=False)
    return out


def deltas(data: StdData, seed: int) -> dict:
    src = source_rows(data, seed)
    return {c: (data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) - data.X[src[c]].mean(0, dtype=np.float64))
            for c in data.cell_types}


def check(data: StdData, root: Path, method: str, seed: int) -> None:
    """The generated mean of the weakest conditions should sit on the reproduced source mean."""
    src = source_rows(data, seed)
    rng = np.random.default_rng(999)
    cells = read_cells(root / "preds" / method / f"seed_{seed}" / "val.h5ad", data.genes)
    keys = [k for k in cells if data.cond.dose_nM[k] <= 10]
    for c in data.cell_types:
        ks = [k for k in keys if data.cond.cell_type[k] == c][:15]
        if not ks:
            continue
        gen = np.mean([cells[k].mean(0) for k in ks], 0)
        mine = data.X[src[c]].mean(0)
        other = data.X[rng.choice(data.ctrl_idx(c), N_GEN, replace=False)].mean(0)
        full = data.X[data.ctrl_idx(c)].mean(0)
        d = lambda a: float(np.sqrt(((gen - a) ** 2).mean()))
        print(f"check seed {seed} {c}: |gen - reproduced sources| {d(mine):.4f}  |gen - another draw| {d(other):.4f}  "
              f"|gen - full vehicle| {d(full):.4f}  (delta rms {float(np.sqrt(((full - mine) ** 2).mean())):.4f})", flush=True)


def main(argv=None):
    import anndata as ad
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--method", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "ood"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args(argv)
    root = Path(a.root)
    data = StdData(a.file)
    for seed in a.seeds:
        check(data, root, a.method, seed)
        dl = deltas(data, seed)
        for split in a.splits:
            cells = read_cells(root / "preds" / a.method / f"seed_{seed}" / f"{split}.h5ad", data.genes)
            keys = data.keys(split)
            X = np.vstack([np.maximum(cells[k] + dl[data.cond.cell_type[k]][None, :], 0.0) for k in keys])
            labels = np.concatenate([[k] * len(cells[k]) for k in keys])
            A = ad.AnnData(X.astype(np.float32), obs=pd.DataFrame({"condition": labels},
                           index=[f"g{i}" for i in range(len(labels))]), var=pd.DataFrame(index=data.genes))
            d = root / "preds" / f"{a.method}_ctr" / f"seed_{seed}" / f"{split}.h5ad"
            d.parent.mkdir(parents=True, exist_ok=True)
            A.write_h5ad(d, compression="gzip")
            print(f"wrote {d}", flush=True)


if __name__ == "__main__":
    main()
