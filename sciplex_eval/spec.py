"""Frozen evaluation spec shared by every baseline.

Everything a metric needs that must NOT vary between methods lives here and is fitted
once (scripts/build_panel.py) on training + control cells only:

  - the evaluation gene space (Ensembl ids, order matters) and its representation,
    log1p(counts / human-UMI total * 1e4)
  - the proliferation program genes and how program activity a(x) is computed
  - control statistics mu0, s0 and the thresholds tau_q (q-quantiles of control r(x))
  - the PCA basis and RBF bandwidths used for MMD
  - the control mean expression used to pick per-condition DEGs

Load it with EvalSpec.load("artifacts/eval_spec_v1").
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

EPS = 1e-6


@dataclass
class EvalSpec:
    version: str
    cell_line: str
    gene_ids: list[str]
    gene_symbols: list[str]
    program_name: str
    program_gene_ids: list[str]
    score_type: str  # "log_sum" | "mean_log", see program_activity()
    mu0: float
    s0: float
    quantiles: list[float]  # q levels; tau_q = q-quantile of r(x) over controls
    taus: list[float]
    primary_q: float
    deg_top_k: int
    mmd_n_max: int
    mmd_sigmas: list[float]
    notes: dict = field(default_factory=dict)
    # arrays (saved to arrays.npz)
    pca_mean: np.ndarray | None = None  # (G,)
    pca_components: np.ndarray | None = None  # (n_pcs, G)
    control_mean: np.ndarray | None = None  # (G,) mean log-expression of controls

    def __post_init__(self):
        pos = {g: i for i, g in enumerate(self.gene_ids)}
        self.program_idx = np.array([pos[g] for g in self.program_gene_ids], dtype=np.int64)

    @property
    def n_genes(self) -> int:
        return len(self.gene_ids)

    @property
    def tau_by_q(self) -> dict[float, float]:
        return dict(zip(self.quantiles, self.taus))

    # ---- io -------------------------------------------------------------------------
    _ARRAYS = ("pca_mean", "pca_components", "control_mean")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        d = {k: v for k, v in asdict(self).items() if k not in self._ARRAYS}
        (path / "spec.json").write_text(json.dumps(d, indent=1))
        np.savez(path / "arrays.npz", **{k: getattr(self, k) for k in self._ARRAYS})

    @classmethod
    def load(cls, path: str | Path) -> "EvalSpec":
        path = Path(path)
        d = json.loads((path / "spec.json").read_text())
        arr = np.load(path / "arrays.npz")
        return cls(**d, **{k: arr[k] for k in cls._ARRAYS})

    # ---- gene-space helpers ---------------------------------------------------------
    def align(self, X, var_names) -> np.ndarray | sp.spmatrix:
        """Reorder columns of X (cells x genes, named by var_names) into spec gene order."""
        var_names = list(map(str, var_names))
        if var_names == self.gene_ids:
            return X
        pos = {g: i for i, g in enumerate(var_names)}
        missing = [g for g in self.gene_ids if g not in pos]
        if missing:
            raise ValueError(f"{len(missing)} spec genes missing from predictions, e.g. {missing[:5]}")
        return X[:, [pos[g] for g in self.gene_ids]]

    def project(self, X) -> np.ndarray:
        """PCA coordinates used for MMD."""
        Y = X @ self.pca_components.T
        Y = np.asarray(Y) - self.pca_mean @ self.pca_components.T
        return Y.astype(np.float64)


def _dense(X) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def activity_from_program_genes(Xp, score_type: str) -> np.ndarray:
    """a(x) from the program-gene columns only (log1p CP10K).

    log_sum : log1p(sum_g expm1(x_g)), the log of the program's total normalized
              expression (like sci-Plex's proliferation index; robust to sparse counts)
    mean_log: mean_g x_g
    Negative values (possible from continuous generators) are clipped to 0 first.
    """
    Xp = np.clip(_dense(Xp).astype(np.float64), 0, None)
    if score_type == "log_sum":
        return np.log1p(np.expm1(Xp).sum(axis=1))
    if score_type == "mean_log":
        return Xp.mean(axis=1)
    raise ValueError(score_type)


def program_activity(X, spec: EvalSpec) -> np.ndarray:
    """a(x) for cells X in spec gene order."""
    return activity_from_program_genes(X[:, spec.program_idx], spec.score_type)


def response_score(X, spec: EvalSpec) -> np.ndarray:
    """r(x) = (mu0 - a(x)) / (s0 + eps); larger = stronger suppression vs. controls (eq. 1)."""
    return (spec.mu0 - program_activity(X, spec)) / (spec.s0 + EPS)
