"""Population-level metrics comparing real and predicted cells of one condition.

All expression inputs are cells x genes in the spec gene order, log1p(CP10K).

  WRF absolute error   |f_tau(pred) - f_tau(real)| in percentage points, tau = tau_q for
                       q in spec.quantiles (primary q = spec.primary_q)
  response-score W1    1-D Wasserstein-1 between r(x) of real and predicted cells
                       (units: control SDs of program activity)
  MMD^2                unbiased, multi-scale RBF kernel, in the spec's PCA space
  mean-expression RMSE between per-gene population means: all spec genes, the top-k
                       DEGs of the real condition vs. controls, and the program genes
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from .spec import EvalSpec, response_score


def weak_responder_fraction(r: np.ndarray, taus) -> np.ndarray:
    """f_tau = Pr[r(x) < tau] for each tau (eq. 1, hard indicator)."""
    return (np.asarray(r)[:, None] < np.asarray(taus, dtype=np.float64)[None, :]).mean(axis=0)


def response_w1(r_real: np.ndarray, r_pred: np.ndarray) -> float:
    return float(wasserstein_distance(r_real, r_pred))


def mmd2_rbf(A: np.ndarray, B: np.ndarray, sigmas) -> float:
    """Unbiased MMD^2 with k(x, y) = mean_s exp(-|x-y|^2 / (2 s^2)).

    Unequal sample sizes are fine. Values near 0 (including small negatives) mean
    the two samples are indistinguishable at these bandwidths.
    """
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    m, n = len(A), len(B)
    if m < 2 or n < 2:
        return float("nan")
    inv = [1.0 / (2.0 * s * s) for s in sigmas]

    def kmean(X, Y, drop_diag):
        d2 = np.maximum((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T, 0)
        K = sum(np.exp(-g * d2) for g in inv) / len(inv)
        if drop_diag:
            return (K.sum() - np.trace(K)) / (len(X) * (len(X) - 1))
        return K.mean()

    return float(kmean(A, A, True) + kmean(B, B, True) - 2 * kmean(A, B, False))


def _mean(X) -> np.ndarray:
    return np.asarray(X.mean(axis=0)).ravel().astype(np.float64)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _subsample(X, n_max: int, rng: np.random.Generator):
    if X.shape[0] <= n_max:
        return X
    return X[np.sort(rng.choice(X.shape[0], n_max, replace=False))]


def qkey(q: float) -> str:
    return f"q{q:g}"


def evaluate_condition(spec: EvalSpec, X_real, X_pred=None, wrf_pred: dict | None = None,
                       seed: int = 0) -> dict:
    """Metrics for one condition.

    Give X_pred (generated cells) for the full metric set, or wrf_pred ({q: fraction})
    for methods that only predict the weak-responder fraction (WRF error only).
    Use all generated cells (>= the number of real cells is recommended); MMD caps
    both samples at spec.mmd_n_max cells.
    """
    if (X_pred is None) == (wrf_pred is None):
        raise ValueError("pass exactly one of X_pred / wrf_pred")
    r_real = response_score(X_real, spec)
    f_real = weak_responder_fraction(r_real, spec.taus)
    out = {"n_real": int(X_real.shape[0])}

    if X_pred is not None:
        r_pred = response_score(X_pred, spec)
        f_pred = weak_responder_fraction(r_pred, spec.taus)
        rng = np.random.default_rng(seed)
        m_real, m_pred = _mean(X_real), _mean(X_pred)
        deg = np.argsort(-np.abs(m_real - spec.control_mean), kind="stable")[: spec.deg_top_k]
        out.update({
            "n_pred": int(X_pred.shape[0]),
            "w1_r": response_w1(r_real, r_pred),
            "mmd2": mmd2_rbf(spec.project(_subsample(X_real, spec.mmd_n_max, rng)),
                             spec.project(_subsample(X_pred, spec.mmd_n_max, rng)), spec.mmd_sigmas),
            "rmse_mean_all": _rmse(m_real, m_pred),
            f"rmse_mean_deg{spec.deg_top_k}": _rmse(m_real[deg], m_pred[deg]),
            "rmse_mean_program": _rmse(m_real[spec.program_idx], m_pred[spec.program_idx]),
            "mean_r_real": float(r_real.mean()),
            "mean_r_pred": float(r_pred.mean()),
        })
    else:
        f_pred = np.array([wrf_pred[q] for q in spec.quantiles], dtype=np.float64)

    for q, fr, fp in zip(spec.quantiles, f_real, f_pred):
        out[f"wrf_real_{qkey(q)}"] = float(fr)
        out[f"wrf_pred_{qkey(q)}"] = float(fp)
        out[f"wrf_err_pp_{qkey(q)}"] = float(100 * abs(fp - fr))
    return out


def summarize(records: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Equal-weight mean over conditions with a bootstrap 95% CI (resampling conditions)."""
    rng = np.random.default_rng(seed)
    keys = [k for k in records[0] if k.startswith(("wrf_err", "w1", "mmd", "rmse"))]
    out = {"n_conditions": len(records)}
    for k in keys:
        v = np.array([r[k] for r in records], dtype=np.float64)
        v = v[np.isfinite(v)]
        boots = rng.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                  "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out
