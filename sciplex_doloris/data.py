"""Inputs DOLORIS sees, built from the frozen panel and split (no held-out cell is read).

  * training cells: the panel's `train` cells (78 drug-dose conditions) plus every vehicle cell
    (all vehicle cells are permitted controls, configs/split_v1.json), as the released
    PertData.train_cell holds the training conditions and all controls
  * expression: the panel's log1p(CP10K) over the 2,085 eval genes, divided by x_max, the largest
    value among training cells (PertData.normalization); predictions are multiplied back and
    clipped at 0 (test.py), so they are scored in the panel's own space
  * dose: chemCPA's dose_val = dose / 10 uM (0.001 ... 1), vehicle 0
  * drug: the Uni-Mol CLS embedding (512) of the drug's SMILES; vehicle gets zeros
  * control prior: per-gene mean and unbiased variance of the scaled training controls
  * gene graph: co-expression edges, |Pearson r| > threshold_co over 500 random control cells
    (Dataset/GRN.py co_expression_GRN). The released graph also adds edges between genes whose
    pretrained scGPT embeddings have cosine similarity > threshold; those are not included here
    (see docs/doloris/README.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sciplex_interp.core import Data

DOSE_SCALE_NM = 10000.0


@dataclass
class TrainSet:
    X: np.ndarray            # scaled expression, cells x genes (float32)
    dosage: np.ndarray       # (cells, 1) dose_val, 0 for vehicle
    drug_idx: np.ndarray     # index into Molecules.emb, -1 for vehicle
    condition: np.ndarray    # condition id per cell
    is_ctrl: np.ndarray
    x_max: float
    ctrl_mean: np.ndarray
    ctrl_var: np.ndarray


@dataclass
class Molecules:
    drugs: list[str]
    emb: np.ndarray          # (n_drugs, 512)

    def of(self, drug: str) -> np.ndarray:
        return self.emb[self.drugs.index(drug)]

    def rows(self, drug_idx: np.ndarray) -> np.ndarray:
        out = np.zeros((len(drug_idx), self.emb.shape[1]), dtype=np.float32)
        hit = drug_idx >= 0
        out[hit] = self.emb[drug_idx[hit]]
        return out


def load_molecules(path: str | Path) -> Molecules:
    z = np.load(path, allow_pickle=False)
    return Molecules([str(d) for d in z["drugs"]], z["emb"].astype(np.float32))


def dose_val(dose_nM) -> np.ndarray:
    return np.asarray(dose_nM, dtype=np.float32) / DOSE_SCALE_NM


def train_set(data: Data, mols: Molecules) -> TrainSet:
    split = data.obs.split.to_numpy()
    keep = np.flatnonzero((split == "train") | (split == "control"))
    X = data.X[keep]
    x_max = float(X.max())
    Xs = (X / x_max).astype(np.float32)
    is_ctrl = split[keep] == "control"
    drugs = data.obs.drug.to_numpy()[keep]
    drug_idx = np.array([-1 if c else mols.drugs.index(d) for d, c in zip(drugs, is_ctrl)], dtype=np.int64)
    dosage = np.where(is_ctrl, 0.0, dose_val(data.obs.dose.to_numpy()[keep])).astype(np.float32)[:, None]
    ctrl = Xs[is_ctrl].astype(np.float64)
    return TrainSet(Xs, dosage, drug_idx, data.obs.condition.to_numpy()[keep], is_ctrl, x_max,
                    ctrl.mean(0).astype(np.float32), ctrl.var(0, ddof=1).astype(np.float32))


def coexpression_adjacency(Xs_ctrl: np.ndarray, threshold: float, n_cells: int, seed: int) -> np.ndarray:
    """|Pearson r| > threshold between genes over n_cells random controls; genes without variance get
    no edges (pandas corr gives NaN for them)."""
    rng = np.random.default_rng(seed)
    sub = Xs_ctrl[rng.choice(len(Xs_ctrl), n_cells, replace=False)].astype(np.float64)
    sd = sub.std(0)
    z = np.zeros_like(sub)
    ok = sd > 0
    z[:, ok] = (sub[:, ok] - sub[:, ok].mean(0)) / sd[ok]
    r = z.T @ z / n_cells
    adj = np.abs(r) > threshold
    adj[~ok, :] = False
    adj[:, ~ok] = False
    np.fill_diagonal(adj, True)
    return adj


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
