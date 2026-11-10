"""The standard sci-Plex3 benchmark: chemCPA's processed data with DOLORIS's two held-out tasks.

Data: chemCPA's `sciplex_complete_middle_subset.h5ad` (public, chemCPA-datasets bucket; the file
biolord loads, and the one whose shape and keys DOLORIS reads: 2,000 genes, `lincs_DEGs`,
`cov_drug_dose_name`, `dose_val`, `SMILES`). 354,640 cells, A549 / K562 / MCF7, 187 drugs at 4 doses,
13,004 vehicle cells, log1p of chemCPA's size-factor normalisation. Expression is used as stored.

Split (one model per method serves both tables, as DOLORIS's test.py scores one model on both):
  ood    every cell of the 9 unseen drugs of the chemCPA paper (also biolord's `ood_drugs`):
         DOLORIS Table 2, "OOD drug perturbations"
  test   each remaining (cell line, drug, dose) condition with probability 0.3 (DOLORIS App. A.2):
         Table 1, "unseen drug - cell line - dosage"
  val    10% of the remaining conditions, held out of training for checkpoint selection only
  train  the rest; all vehicle cells are training controls
DOLORIS's own random draw is not published, so this draw (seed 0) is ours; it is frozen in
configs/standard_split_v1.json.

usage: python -m sciplex_standard.data split   # writes the split file (once)
       python -m sciplex_standard.data spec    # PCA + MMD bandwidths from control + train cells
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FILE = "/zfsauton/scratch/yixiz/sciplex3/chemcpa/sciplex_complete_middle_subset.h5ad"
SPLIT_FILE = REPO / "configs" / "standard_split_v1.json"
SPEC_DIR = REPO / "artifacts" / "standard_spec_v1"
OOD_DRUGS = ["Dacinostat", "Givinostat", "Belinostat", "Hesperadin", "Quisinostat", "Alvespimycin", "Tanespimycin",
             "TAK-901", "Flavopiridol"]
TEST_PROB, VAL_FRAC, SEED = 0.3, 0.1, 0
DOSE_NM = 10000.0  # dose_val = dose / 10 uM
MIN_TEST_CELLS = 10  # test.py skips unseen-covariate conditions with fewer cells; OOD conditions are all scored


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_obs(path: str) -> tuple[pd.DataFrame, list[str], dict, object]:
    import anndata as ad
    a = ad.read_h5ad(path)
    o = a.obs
    ctrl = o["control"].astype(int).to_numpy() == 1
    obs = pd.DataFrame({
        "cell_type": o["cell_type"].astype(str).to_numpy(),
        "drug": np.where(ctrl, "control", o["condition"].astype(str).to_numpy()),
        "dose_val": np.where(ctrl, 0.0, o["dose_val"].astype(float).to_numpy()),
        "key": o["cov_drug_dose_name"].astype(str).to_numpy(),
        "replicate": o["replicate"].astype(str).to_numpy(),
        "smiles": o["SMILES"].astype(str).to_numpy(),
        "control": ctrl,
    })
    obs["dose_nM"] = obs["dose_val"] * DOSE_NM
    return obs, [str(g) for g in a.var_names], a.uns["lincs_DEGs"], a


class StdData:
    """Expression (dense float32), cell table, condition table with splits, DEG lists."""

    def __init__(self, path: str = DEFAULT_FILE, split_path: Path = SPLIT_FILE, load_X: bool = True):
        self.obs, self.genes, degs, a = read_obs(path)
        self.X = np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X, dtype=np.float32) if load_X else None
        del a
        pos = {g: i for i, g in enumerate(self.genes)}
        self.de = {k: [pos[str(g)] for g in v] for k, v in degs.items()}
        split = json.loads(Path(split_path).read_text())
        if split["source_sha256"] != sha256(path):
            raise SystemExit(f"{path} is not the file the split was drawn from")
        self.split_of = split["conditions"]
        self.obs["split"] = np.where(self.obs.control, "control", self.obs.key.map(self.split_of).fillna("missing"))
        assert not (self.obs.split == "missing").any()
        g = self.obs[~self.obs.control].groupby("key")
        self.cond = pd.DataFrame({"cell_type": g.cell_type.first(), "drug": g.drug.first(), "dose_val": g.dose_val.first(),
                                  "dose_nM": g.dose_nM.first(), "n_cells": g.size(), "split": g.split.first()})
        self.cell_types = sorted(self.obs.cell_type.unique())
        self.drugs = sorted(self.obs.loc[~self.obs.control, "drug"].unique())
        self._idx = {k: np.flatnonzero(m) for k, m in ((k, (self.obs.key == k).to_numpy()) for k in self.cond.index)}
        self._ctrl = {c: np.flatnonzero((self.obs.control & (self.obs.cell_type == c)).to_numpy()) for c in self.cell_types}

    def idx(self, key: str) -> np.ndarray:
        return self._idx[key]

    def ctrl_idx(self, cell_type: str) -> np.ndarray:
        return self._ctrl[cell_type]

    def keys(self, split: str, min_cells: int | None = None) -> list[str]:
        c = self.cond[self.cond.split == split]
        if min_cells is None:
            min_cells = MIN_TEST_CELLS if split == "test" else 0
        return sorted(c.index[c.n_cells >= min_cells])

    def train_rows(self) -> np.ndarray:
        return np.flatnonzero(self.obs.split.isin(["train", "control"]).to_numpy())


def make_split(path: str) -> dict:
    obs, *_ = read_obs(path)
    conds = sorted(obs.loc[~obs.control, "key"].unique())
    drug = obs[~obs.control].groupby("key").drug.first()
    missing = sorted(set(OOD_DRUGS) - set(drug))
    if missing:
        raise SystemExit(f"OOD drugs missing from the file: {missing}")
    rng = np.random.default_rng(SEED)
    out = {}
    for k in conds:  # sorted order, one draw per condition: test with probability 0.3
        out[k] = "ood" if drug[k] in OOD_DRUGS else ("test" if rng.random() < TEST_PROB else "train")
    for k in conds:  # then 10% of the remaining conditions for checkpoint selection
        if out[k] == "train" and rng.random() < VAL_FRAC:
            out[k] = "val"
    counts = pd.Series(out).value_counts().to_dict()
    return {"version": "standard_split_v1", "source": path, "source_sha256": sha256(path),
            "rule": f"ood = all cells of {len(OOD_DRUGS)} chemCPA-paper OOD drugs; each other (cell line, drug, dose) "
                    f"condition -> test with probability {TEST_PROB} (sorted keys, numpy default_rng({SEED})); then "
                    f"{VAL_FRAC} of the remaining conditions -> val; the rest train; all vehicle cells are controls",
            "ood_drugs": OOD_DRUGS, "counts": counts, "conditions": out}


def fit_spec(data: StdData, n_fit: int = 50000, n_pcs: int = 50, n_median: int = 2000) -> dict:
    """PCA and RBF bandwidths (0.5/1/2 x median pairwise distance) from control + train cells only."""
    rng = np.random.default_rng(0)
    rows = data.train_rows()
    sub = data.X[np.sort(rng.choice(rows, min(n_fit, len(rows)), replace=False))].astype(np.float64)
    mean = sub.mean(0)
    _, s, vt = np.linalg.svd(sub - mean, full_matrices=False)
    comps = vt[:n_pcs]
    z = (data.X[np.sort(rng.choice(rows, n_median, replace=False))] - mean) @ comps.T
    d = np.sqrt(np.maximum((z * z).sum(1)[:, None] + (z * z).sum(1)[None] - 2 * z @ z.T, 0))
    med = float(np.median(d[np.triu_indices(len(z), 1)]))
    return {"pca_mean": mean.astype(np.float32), "pca_components": comps.astype(np.float32),
            "explained_variance_ratio": (s[:n_pcs] ** 2 / (s ** 2).sum()).astype(np.float32),
            "mmd_sigmas": np.array([0.5 * med, med, 2 * med], dtype=np.float64), "mmd_n_max": np.int64(2000)}


def load_spec(spec_dir: Path = SPEC_DIR) -> dict:
    z = np.load(Path(spec_dir) / "spec.npz")
    return {k: z[k] for k in z.files}


def project(spec: dict, X: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, np.float64) - spec["pca_mean"]) @ spec["pca_components"].T.astype(np.float64))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["split", "spec", "smiles"])
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.stage == "split":
        if SPLIT_FILE.exists():
            print(f"{SPLIT_FILE} exists (frozen); not redrawn")
            return
        s = make_split(a.file)
        SPLIT_FILE.write_text(json.dumps(s, indent=1) + "\n")
        print(f"wrote {SPLIT_FILE}: {s['counts']}")
    elif a.stage == "spec":
        data = StdData(a.file)
        spec = fit_spec(data)
        SPEC_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(SPEC_DIR / "spec.npz", **spec)
        (SPEC_DIR / "spec.json").write_text(json.dumps({"fit_on": "control + train cells (50,000 random)", "n_pcs": 50,
                                                         "pc_variance_explained": float(spec["explained_variance_ratio"].sum()),
                                                         "mmd_sigmas": spec["mmd_sigmas"].tolist(), "split": str(SPLIT_FILE)},
                                                        indent=1) + "\n")
        n = {s: len(data.keys(s)) for s in ("train", "val", "test", "ood")}
        print(f"wrote {SPEC_DIR}; scored conditions {n}; sigmas {spec['mmd_sigmas']}")
    else:  # drug -> SMILES for the Uni-Mol job (DOLORIS keeps the part before '|')
        obs, *_ = read_obs(a.file)
        t = obs[~obs.control].groupby("drug").smiles.agg(lambda s: sorted(set(s)))
        bad = {d: v for d, v in t.items() if len(v) != 1}
        if bad:
            raise SystemExit(f"drugs with several SMILES: {bad}")
        df = pd.DataFrame({"drug": t.index, "smiles": [v[0].split("|")[0] for v in t]})
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}: {len(df)} drugs")


if __name__ == "__main__":
    main()
