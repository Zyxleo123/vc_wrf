"""Shared data access, interpolation tasks and population metrics for the dose-interpolation study.

Tasks. Every task asks for the cells of one target condition (drug D, dose d*) from other
conditions only:
  val / test   the frozen held-out condition of D; context = D's three training doses + vehicle.
  inner        D's *training* intermediate dose, predicted from D's 10 nM and 10 µM conditions
               + vehicle (the target and the held-out dose are both excluded). Uses training
               cells only, so it can be used freely for development.
The adjacent doses of a task are its nearest context doses below and above d*.

Metrics. The frozen evaluator (sciplex_eval.evaluate_condition) gives MMD², W1(r) and the
mean-expression RMSEs; WRF fields are computed but not used here. Supplementary metrics, same
cells: energy distance in the spec PCA space, and Pearson correlation of the predicted vs
measured mean change from controls (all genes; the condition's top-50 DEGs).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sciplex_eval import EvalSpec
from sciplex_eval.metrics import evaluate_condition

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "artifacts" / "eval_spec_v1"
SPLIT_CONFIG = REPO / "configs" / "split_v1.json"
DEFAULT_PANEL = os.environ.get("SCIPLEX_PANEL", "/zfsauton/scratch/yixiz/sciplex3/processed/a549_hdaci_v1.h5ad")
DEFAULT_ROOT = Path(os.environ.get("INTERP_ROOT", REPO / "results" / "interp"))
N_PRED = 512  # cells per predicted condition, as in CellFlow's published references
TASK_SETS = ("inner", "val", "test")
FROZEN = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "w1_r"]
EXTRA = ["edist", "pearson_delta_deg50", "pearson_delta_all"]
METRICS = FROZEN + EXTRA
HIGHER_BETTER = {"pearson_delta_deg50", "pearson_delta_all"}


def dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def log_dose(dose_nM) -> np.ndarray:
    """The published adjacent-mixture weight uses log1p(dose in nM)."""
    return np.log1p(np.asarray(dose_nM, dtype=np.float64))


@dataclass
class Task:
    task_set: str
    drug: str
    target: str            # condition id of the cells to predict
    dose: float            # d*
    context: list          # doses of D whose cells may be used (vehicle = 0 is always allowed)
    low: float
    high: float
    n_rep1: int
    n_rep2: int
    meta: dict = field(default_factory=dict)

    @property
    def w_logdose(self) -> float:
        """Share of the upper adjacent dose in the published mixture (log1p-dose position)."""
        a, b, t = log_dose([self.low, self.high, self.dose])
        return float((t - a) / (b - a))


class Data:
    """Panel expression (spec gene order), PCA coordinates, pools and tasks."""

    def __init__(self, panel_path: str = DEFAULT_PANEL, spec_dir: Path = SPEC_DIR):
        self.spec = EvalSpec.load(spec_dir)
        self.adata = ad.read_h5ad(panel_path)
        assert self.adata.obs_names.is_unique
        self.X = dense(self.spec.align(self.adata.X, self.adata.var_names))
        obs = self.adata.obs
        self.obs = pd.DataFrame({k: obs[k].astype(str).to_numpy() for k in
                                 ("condition", "drug", "replicate", "split", "culture_plate")})
        self.obs["dose"] = obs["dose"].astype(float).to_numpy()
        self.Z = self.spec.project(self.X)
        self.split_cfg = json.loads(SPLIT_CONFIG.read_text())
        self.cond = pd.DataFrame(self.split_cfg["conditions"]).set_index("condition")
        self.ctrl = np.flatnonzero((self.obs.split == "control").to_numpy())
        self.ctrl_mean = self.X[self.ctrl].mean(0).astype(np.float64)
        self.drugs = sorted(self.cond.drug.unique())
        self._idx = {c: np.flatnonzero(g) for c, g in
                     ((c, (self.obs.condition == c).to_numpy()) for c in self.obs.condition.unique())}
        self._mean = {}

    # ---- pools ------------------------------------------------------------------------------
    def condition_of(self, drug: str, dose: float) -> str:
        if dose == 0:
            return self.split_cfg["controls"]["condition"]
        hit = self.cond[(self.cond.drug == drug) & (self.cond.dose_nM == float(dose))]
        assert len(hit) == 1, (drug, dose)
        return hit.index[0]

    def idx(self, condition: str, replicate: str | None = None) -> np.ndarray:
        i = self.ctrl if condition == self.split_cfg["controls"]["condition"] else self._idx[condition]
        if replicate is not None:
            i = i[(self.obs.replicate.to_numpy()[i] == replicate)]
        return i

    def mean(self, condition: str) -> np.ndarray:
        if condition not in self._mean:
            self._mean[condition] = self.X[self.idx(condition)].mean(0).astype(np.float64)
        return self._mean[condition]

    def rep_mean(self, condition: str, replicate: str) -> np.ndarray:
        key = (condition, replicate)
        if key not in self._mean:
            self._mean[key] = self.X[self.idx(condition, replicate)].mean(0).astype(np.float64)
        return self._mean[key]

    def train_doses(self, drug: str) -> list[float]:
        c = self.cond[(self.cond.drug == drug) & (self.cond.split == "train")]
        return sorted(float(d) for d in c.dose_nM)

    # ---- tasks ------------------------------------------------------------------------------
    def tasks(self, task_set: str) -> list[Task]:
        out = []
        for drug in self.drugs:
            train = self.train_doses(drug)
            if task_set in ("val", "test"):
                c = self.cond[(self.cond.drug == drug) & (self.cond.split == task_set)]
                if c.empty:
                    continue
                target, dose, context = c.index[0], float(c.dose_nM.iloc[0]), train
            elif task_set == "inner":
                mids = [d for d in train if d not in (min(train), max(train))]
                assert len(mids) == 1, (drug, train)
                dose = mids[0]
                target, context = self.condition_of(drug, dose), [d for d in train if d != dose]
            else:
                raise ValueError(task_set)
            low = max(d for d in context if d < dose)
            high = min(d for d in context if d > dose)
            row = self.cond.loc[target]
            out.append(Task(task_set, drug, target, dose, context, low, high, int(row.n_rep1), int(row.n_rep2)))
        return out

    # ---- sampling ---------------------------------------------------------------------------
    def mixture(self, task: Task, components: dict, rng: np.random.Generator, n: int = N_PRED) -> np.ndarray:
        """Cell positions drawn from {dose: weight} components of task.drug, replicate-matched
        in the target's rep1/rep2 proportion (the published recipe)."""
        n1 = int(round(n * task.n_rep1 / (task.n_rep1 + task.n_rep2)))
        doses = [d for d, w in components.items() if w > 0]
        weights = np.array([components[d] for d in doses], dtype=np.float64)
        weights /= weights.sum()
        picks = []
        for rep, take in (("rep1", n1), ("rep2", n - n1)):
            pools = [self.idx(self.condition_of(task.drug, d), rep) for d in doses]
            comp = rng.choice(len(pools), take, p=weights)
            picks += [rng.choice(pools[k], int((comp == k).sum())) for k in range(len(pools))]
        return np.concatenate(picks).astype(np.int64)


# ---- metrics ----------------------------------------------------------------------------------
def _subsample(n: int, n_max: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if n <= n_max else np.sort(rng.choice(n, n_max, replace=False))


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """2E|a-b| - E|a-a'| - E|b-b'| with the within-sample terms excluding the diagonal."""
    def mean_dist(X, Y, drop_diag):
        d = np.sqrt(np.maximum((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T, 0))
        if drop_diag:
            return (d.sum() - np.trace(d)) / (len(X) * (len(X) - 1))
        return d.mean()
    return float(2 * mean_dist(A, B, False) - mean_dist(A, A, True) - mean_dist(B, B, True))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def score(data: Data, X_real: np.ndarray, X_pred: np.ndarray, seed: int = 0) -> dict:
    """Frozen population metrics plus the supplementary ones, for one condition."""
    spec = data.spec
    rec = evaluate_condition(spec, X_real, X_pred=X_pred, seed=seed)
    rec = {k: rec[k] for k in ("n_real", "n_pred", *FROZEN)}
    rng = np.random.default_rng(seed + 1)
    A = spec.project(X_real[_subsample(len(X_real), spec.mmd_n_max, rng)])
    B = spec.project(X_pred[_subsample(len(X_pred), spec.mmd_n_max, rng)])
    rec["edist"] = energy_distance(A, B)
    d_real = X_real.mean(0).astype(np.float64) - spec.control_mean
    d_pred = X_pred.mean(0).astype(np.float64) - spec.control_mean
    deg = np.argsort(-np.abs(d_real), kind="stable")[: spec.deg_top_k]
    rec["pearson_delta_all"] = _pearson(d_real, d_pred)
    rec["pearson_delta_deg50"] = _pearson(d_real[deg], d_pred[deg])
    return rec


def mean_errors(data: Data, target: str, m_pred: np.ndarray) -> dict:
    """Mean-only metrics of a predicted mean vector (the frozen RMSE definitions + Pearson Δ)."""
    spec = data.spec
    m_real = data.mean(target)
    d_real, d_pred = m_real - spec.control_mean, m_pred - spec.control_mean
    deg = np.argsort(-np.abs(d_real), kind="stable")[: spec.deg_top_k]
    rmse = lambda i: float(np.sqrt(np.mean((m_real[i] - m_pred[i]) ** 2)))
    return {"rmse_mean_all": rmse(slice(None)), "rmse_mean_deg50": rmse(deg),
            "rmse_mean_program": rmse(spec.program_idx),
            "pearson_delta_all": _pearson(d_real, d_pred), "pearson_delta_deg50": _pearson(d_real[deg], d_pred[deg])}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    tmp.rename(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open()]
