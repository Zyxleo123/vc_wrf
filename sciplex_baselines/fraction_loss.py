"""Response-fraction regularization (proposal eqs. 2-3) for count-likelihood generators.

The smooth weak-responder indicator is sigma((tau - r(x)) / h), applied identically
to real and generated cells, with r(x) the frozen spec score. Real targets are
averages over measured cells. For a generated cell with a negative binomial over
its UMIs, the indicator's *expectation* over that cell's program count is used:
sampling counts is not differentiable, but the expectation is, and its average
over cells is an unbiased, lower-variance estimate of the sampled-population value.

r(x) depends on a cell only through S, its total program UMIs, and its library T:
a(x) = log1p(S / T * 1e4) for the "log_sum" score. The program total of a generated
cell is approximated by one NB with matched mean and variance, and T by the source
cell's library size (the generated total differs from it by sampling noise only).
"""
from __future__ import annotations

import numpy as np
import torch

from sciplex_eval.spec import EPS, EvalSpec, response_score


def soft_fraction(r, taus, h):
    """Mean over cells of sigma((tau - r) / h), one value per tau (numpy)."""
    z = (np.asarray(taus, dtype=np.float64)[None, :] - np.asarray(r, dtype=np.float64)[:, None]) / h
    return (1 / (1 + np.exp(-z))).mean(axis=0)


def supervised_taus(spec: EvalSpec, control_expression, quantiles=None):
    """The spec taus, or tau_q = q-quantile of control r(x) (the spec's own rule) for
    other quantiles; controls are permitted for every method."""
    if quantiles is None:
        return np.asarray(spec.taus, dtype=np.float64)
    return np.quantile(response_score(control_expression, spec), quantiles)


def real_targets(spec: EvalSpec, expression, h, taus):
    """Smoothed weak-responder fractions of measured cells (log1p(CP10K), spec order)."""
    return soft_fraction(response_score(expression, spec), taus, h)


def expected_soft_indicator(mean, theta, library, spec: EvalSpec, h, taus, max_count=256):
    """E_S[sigma((tau - r(S, T)) / h)] per cell and tau, for NB-decoded cells.

    mean, theta: (cells, features) NB means and inverse dispersions in spec gene
    order (extra trailing features are ignored); library: (cells,) T.
    Returns (cells, len(taus)). Mass above max_count takes the indicator at
    max_count, where cells are on the weak-responder side of every tau.
    """
    if spec.score_type != "log_sum":
        raise ValueError("the analytic NB expectation needs the log_sum program score")
    idx = torch.as_tensor(spec.program_idx, device=mean.device)
    mu, th = mean[:, idx].double(), theta[:, idx].double()
    m = mu.sum(1).clamp_min(1e-8)
    excess = (mu * mu / th).sum(1).clamp_min(1e-8)  # variance - mean of the program total
    k = (m * m / excess).clamp(max=1e4)  # moment-matched NB inverse dispersion
    s = torch.arange(max_count + 1, device=mean.device, dtype=torch.float64)
    log_pmf = (torch.lgamma(s[None] + k[:, None]) - torch.lgamma(k)[:, None] - torch.lgamma(s + 1)[None]
               + (k * torch.log(k / (k + m)))[:, None] + s[None] * torch.log(m / (k + m))[:, None])
    pmf = log_pmf.exp()
    a = torch.log1p(s[None] / library.double()[:, None] * 1e4)
    r = (spec.mu0 - a) / (spec.s0 + EPS)
    taus = torch.as_tensor(taus, device=mean.device, dtype=torch.float64)
    indicator = torch.sigmoid((taus[None, None] - r[..., None]) / h)
    tail = (1 - pmf.sum(1)).clamp_min(0)
    return ((pmf[..., None] * indicator).sum(1) + tail[:, None] * indicator[:, -1]).float()
