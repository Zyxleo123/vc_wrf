"""Can a simple model infer one condition's mean expression from the other conditions?

Every predictor returns a mean vector m̂ (log1p CP10K, spec gene order) for a task's target and
sees only the means a task allows: every training condition (for inner tasks, minus the target)
and vehicle. Mean shifts Δ = m − m0 are taken from the vehicle mean m0. "Progress" p is the
projection of Δ on the first principal direction of the available training mean shifts, oriented
so that responses are positive.

  vehicle, nearest     no change; the nearest context dose (log1p-dose distance)
  linear               (1-w) m_low + w m_high, w = log1p-dose position: the published mixture's mean
  linear_w{w}          the same with a fixed global weight
  linear_k{K}          `linear` projected onto the top-K principal directions (denoising control)
  hill                 per-drug Hill curve of progress, p(d) = E·d^n/(d^n + EC50^n), posterior over
                       (EC50, n, E) on a grid, with the (n, E) prior learned across all drugs
                       (empirical Bayes); m̂ = progress-weighted interpolation of the neighbours
  traj_l{ℓ}_r{λ}_{p}   shared response trajectory: m̂ = m0 + F(p*) + R*, F = kernel ridge regression
                       of Δ on p over every available training condition, R* = the neighbours'
                       residuals Δ − F(p) interpolated; p* from the log-dose ("lin") or Hill ("hill")
                       weight; `_nores` drops R*
  autodec_e{k}_wd{w}   MLP auto-decoder: learned per-drug embedding + log10 dose -> top-20 mean
                       shift, fit on every available training condition (a mean-level chemCPA)
  metanet              MLP(neighbour means, dose positions) -> correction to `linear` in the top-5
                       directions, trained on interpolation tasks built from training conditions
  lowrank_r{r}         drug x (dose, top-10 coordinates) matrix completion by ridge ALS

usage: python -m sciplex_interp.stats --task-sets inner val [--only REGEX]
Writes <root>/stats/rows.jsonl (one row per task set x condition x predictor; mean metrics and
the noise floor) and <root>/stats/means/<predictor>.npz (predicted means, for the population stage).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, Task, log_dose, mean_errors, read_jsonl, write_jsonl

K_BASIS = 20
FIXED_W = (0.2, 0.3, 0.4)


class Context:
    """Means a task may use: every available training condition, plus vehicle (Δ = 0)."""

    def __init__(self, data: Data, task: Task):
        train = data.cond[data.cond.split == "train"]
        keep = [c for c in train.index if not (task.task_set == "inner" and c == task.target)]
        self.data, self.task, self.conds = data, task, keep
        self.drug = train.loc[keep, "drug"].to_numpy()
        self.dose = train.loc[keep, "dose_nM"].to_numpy(float)
        self.m0 = data.ctrl_mean
        self.D = np.array([data.mean(c) - self.m0 for c in keep])
        _, _, Vt = np.linalg.svd(self.D, full_matrices=False)
        sign = 1.0 if (self.D[self.dose == 10000] @ Vt[0]).mean() >= 0 else -1.0
        Vt[0] *= sign
        self.V = Vt
        self.p = self.D @ Vt[0]
        # noise SD of a condition's progress, from its replicate halves
        diffs = []
        for c in keep:
            halves = [data.rep_mean(c, rep) for rep in ("rep1", "rep2")]
            diffs.append((halves[0] - halves[1]) @ Vt[0])
        self.sigma_p = float(np.sqrt(np.mean(np.square(diffs)) / 4))

    def delta(self, drug: str, dose: float) -> np.ndarray:
        if dose == 0:
            return np.zeros_like(self.m0)
        i = np.flatnonzero((self.drug == drug) & (self.dose == dose))
        assert len(i) == 1, (drug, dose)
        return self.D[i[0]]

    def prog(self, drug: str, dose: float) -> float:
        return 0.0 if dose == 0 else float(self.delta(drug, dose) @ self.V[0])

    def drug_points(self, drug: str) -> tuple[np.ndarray, np.ndarray]:
        m = self.drug == drug
        return np.r_[0.0, self.dose[m]], np.r_[0.0, self.p[m]]


# ---- simple references -------------------------------------------------------------------------
def neighbours(ctx: Context):
    t = ctx.task
    return ctx.delta(t.drug, t.low), ctx.delta(t.drug, t.high)


def linear(ctx: Context, w: float | None = None) -> np.ndarray:
    lo, hi = neighbours(ctx)
    w = ctx.task.w_logdose if w is None else w
    return ctx.m0 + (1 - w) * lo + w * hi


def nearest(ctx: Context) -> np.ndarray:
    t = ctx.task
    d = min(t.context, key=lambda x: (abs(log_dose(x) - log_dose(t.dose)), x))
    return ctx.m0 + ctx.delta(t.drug, d)


def project(ctx: Context, delta: np.ndarray, k: int) -> np.ndarray:
    V = ctx.V[:k]
    return (delta @ V.T) @ V


# ---- Hill progress model (empirical Bayes over slope and saturation) -------------------------------
LOG_EC50 = np.arange(-1.0, 7.0001, 0.05)                   # log10 nM
SLOPES = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
EMAX_RATIO = np.array([0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])


def _hill(dose: np.ndarray) -> np.ndarray:
    """h[d, ec50, n] = d^n / (d^n + EC50^n); 0 at d = 0."""
    d = np.asarray(dose, dtype=np.float64)[:, None, None]
    with np.errstate(divide="ignore"):
        z = SLOPES[None, None, :] * (np.log10(np.maximum(d, 1e-300)) - LOG_EC50[None, :, None])
    h = 1.0 / (1.0 + np.exp(-z))
    h[np.asarray(dose) == 0] = 0.0
    return h


class HillModel:
    def __init__(self, ctx: Context, n_em: int = 20):
        self.ctx = ctx
        self.emax = float(np.quantile(ctx.p[ctx.dose == 10000], 0.9)) * EMAX_RATIO
        self.sigma = float(np.sqrt(ctx.sigma_p ** 2 + (0.03 * self.emax.max()) ** 2))
        drugs = sorted(set(ctx.drug))
        self.loglik = {d: self._loglik(*ctx.drug_points(d)) for d in drugs}
        prior = np.full((len(SLOPES), len(EMAX_RATIO)), 1.0 / (len(SLOPES) * len(EMAX_RATIO)))
        for _ in range(n_em):  # hyperprior over (n, E): average of the drugs' posteriors
            post = [self._posterior(ll, prior).sum(0) for ll in self.loglik.values()]
            prior = np.mean(post, 0) + 1e-6
            prior /= prior.sum()
        self.prior = prior

    def _loglik(self, doses, ps) -> np.ndarray:
        """[ec50, n, E] Gaussian log-likelihood of the drug's progress points."""
        h = _hill(doses)                                         # d, ec50, n
        pred = h[..., None] * self.emax[None, None, None, :]     # d, ec50, n, E
        return -0.5 * (((ps[:, None, None, None] - pred) / self.sigma) ** 2).sum(0)

    @staticmethod
    def _posterior(ll, prior) -> np.ndarray:
        lp = ll + np.log(prior)[None]
        lp -= lp.max()
        post = np.exp(lp)
        return post / post.sum()

    def predict(self, drug: str, dose: float) -> float:
        post = self._posterior(self.loglik[drug], self.prior)
        h = _hill([dose])[0]                                     # ec50, n
        return float((post * h[..., None] * self.emax[None, None, :]).sum())


def progress_weight(ctx: Context, hill: HillModel) -> tuple[float, float]:
    """(p*, w): Hill-predicted progress and the neighbour weight that matches it; the log-dose
    weight when the neighbours' progress cannot be told apart from noise."""
    t = ctx.task
    p_lo, p_hi, p_star = ctx.prog(t.drug, t.low), ctx.prog(t.drug, t.high), hill.predict(t.drug, t.dose)
    if p_hi - p_lo < 3 * hill.sigma:
        return (1 - t.w_logdose) * p_lo + t.w_logdose * p_hi, t.w_logdose
    return p_star, float(np.clip((p_star - p_lo) / (p_hi - p_lo), 0.0, 1.0))


# ---- shared response trajectory --------------------------------------------------------------------
class Trajectory:
    """F(p) = kernel ridge regression of Δ on progress (RBF, bandwidth ell), F(0) = 0."""

    def __init__(self, ctx: Context, ell: float, lam: float):
        self.P = np.r_[0.0, ctx.p]
        Y = np.vstack([np.zeros_like(ctx.m0), ctx.D])
        self.ell = ell
        self.alpha = np.linalg.solve(self._k(self.P) + lam * np.eye(len(self.P)), Y)
        self.f0 = self._k(np.array([0.0])) @ self.alpha

    def _k(self, q):
        return np.exp(-0.5 * ((np.asarray(q)[:, None] - self.P[None, :]) / self.ell) ** 2)

    def __call__(self, q: float) -> np.ndarray:
        return (self._k(np.array([q])) @ self.alpha - self.f0)[0]


def trajectory(ctx: Context, traj: Trajectory, p_star: float, w_res: float, residual: bool) -> np.ndarray:
    t = ctx.task
    out = ctx.m0 + traj(p_star)
    if residual:
        r = lambda d: ctx.delta(t.drug, d) - traj(ctx.prog(t.drug, d))
        out = out + (1 - w_res) * r(t.low) + w_res * r(t.high)
    return out


# ---- learned models (torch) -------------------------------------------------------------------------
def _torch():
    import torch
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "4")))
    return torch


def autodecoder(ctx: Context, queries: list[tuple[str, float]], k_e: int, wd: float, seeds: int = 5,
                steps: int = 3000, width: int = 64) -> list[np.ndarray]:
    torch = _torch()
    drugs = sorted(set(ctx.drug))
    lookup = {d: i for i, d in enumerate(drugs)}
    V = ctx.V[:K_BASIS]
    Y = ctx.D @ V.T
    scale = float(Y.std())
    Yt = torch.tensor(Y / scale, dtype=torch.float32)
    idx = torch.tensor([lookup[d] for d in ctx.drug])
    x = torch.tensor(np.log10(ctx.dose)[:, None] / 4, dtype=torch.float32)
    q_idx = torch.tensor([lookup[d] for d, _ in queries])
    q_x = torch.tensor(np.log10([dd for _, dd in queries])[:, None] / 4, dtype=torch.float32)
    preds = []
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        E = torch.nn.Parameter(0.1 * torch.randn(len(drugs), k_e, generator=g))
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(k_e + 1, width), torch.nn.Tanh(), torch.nn.Linear(width, width),
                                  torch.nn.Tanh(), torch.nn.Linear(width, K_BASIS))
        opt = torch.optim.AdamW([{"params": net.parameters(), "weight_decay": wd},
                                 {"params": [E], "weight_decay": wd}], lr=3e-3)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((net(torch.cat([E[idx], x], 1)) - Yt) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds.append(net(torch.cat([E[q_idx], q_x], 1)).numpy() * scale)
    c = np.mean(preds, 0)
    return [ctx.m0 + ci @ V for ci in c]


def metanet_tasks(ctx: Context):
    """(low, high, query) interpolation tasks among each drug's available training doses + vehicle."""
    out = []
    for drug in sorted(set(ctx.drug)):
        doses = sorted(ctx.dose[ctx.drug == drug])
        if len(doses) < 3:
            continue
        full = [0.0] + doses
        for i in range(1, len(full) - 1):
            out.append((drug, full[i - 1], full[i + 1], full[i]))
    return out


def metanet(ctx: Context, queries: list[Task], k: int = 5, seeds: int = 5, steps: int = 2000,
            wd: float = 1e-2) -> list[np.ndarray]:
    torch = _torch()
    V = ctx.V[:k]

    def feats(drug, lo, hi, q):
        a, b, t = log_dose([lo, hi, q])
        c_lo, c_hi = ctx.delta(drug, lo) @ V.T, ctx.delta(drug, hi) @ V.T
        w = (t - a) / (b - a)
        return np.r_[c_lo, c_hi, w, a / 10, b / 10, t / 10], (1 - w) * c_lo + w * c_hi

    rows = metanet_tasks(ctx)
    F, Y = [], []
    for drug, lo, hi, q in rows:
        f, base = feats(drug, lo, hi, q)
        F.append(f)
        Y.append(ctx.delta(drug, q) @ V.T - base)
    F, Y = np.array(F), np.array(Y)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    scale = float(Y.std()) + 1e-12
    Ft, Yt = torch.tensor((F - mu) / sd, dtype=torch.float32), torch.tensor(Y / scale, dtype=torch.float32)
    QF, QB = zip(*(feats(t.drug, t.low, t.high, t.dose) for t in queries))
    Qt = torch.tensor((np.array(QF) - mu) / sd, dtype=torch.float32)
    preds = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, k))
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            ((net(Ft) - Yt) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():
            preds.append(net(Qt).numpy() * scale)
    corr = np.mean(preds, 0)
    out = []
    for t, cq in zip(queries, corr):
        lo, hi = ctx.delta(t.drug, t.low), ctx.delta(t.drug, t.high)
        w = t.w_logdose
        out.append(ctx.m0 + (1 - w) * lo + w * hi + cq @ V)
    return out


def lowrank(ctx: Context, queries: list[tuple[str, float]], rank: int, lam: float = 1.0, k: int = 10,
            iters: int = 200) -> list[np.ndarray]:
    doses = [10.0, 100.0, 1000.0, 10000.0]
    drugs = sorted(set(ctx.drug))
    V = ctx.V[:k]
    M = np.zeros((len(drugs), len(doses), k))
    obs = np.zeros((len(drugs), len(doses)), bool)
    for i, d in enumerate(drugs):
        for j, dd in enumerate(doses):
            m = (ctx.drug == d) & (ctx.dose == dd)
            if m.any():
                M[i, j], obs[i, j] = ctx.D[np.flatnonzero(m)[0]] @ V.T, True
    filled = M.copy()
    for i in range(len(drugs)):  # initial fill: log-dose interpolation from the drug's own doses
        for j in np.flatnonzero(~obs[i]):
            lo = max(jj for jj in range(len(doses)) if obs[i, jj] and jj < j)
            hi = min(jj for jj in range(len(doses)) if obs[i, jj] and jj > j)
            a, b, t = log_dose([doses[lo], doses[hi], doses[j]])
            w = (t - a) / (b - a)
            filled[i, j] = (1 - w) * M[i, lo] + w * M[i, hi]
    flat = filled.reshape(len(drugs), -1)
    U, s, Vt = np.linalg.svd(flat, full_matrices=False)
    A, B = U[:, :rank] * s[:rank], Vt[:rank].T                 # flat ≈ A @ B.T
    mask = np.repeat(obs, k, axis=1)
    Mf = M.reshape(len(drugs), -1)
    for _ in range(iters):
        for i in range(len(drugs)):
            Bo = B[mask[i]]
            A[i] = np.linalg.solve(Bo.T @ Bo + lam * np.eye(rank), Bo.T @ Mf[i, mask[i]])
        for j in range(Mf.shape[1]):
            Ao = A[mask[:, j]]
            B[j] = np.linalg.solve(Ao.T @ Ao + lam * np.eye(rank), Ao.T @ Mf[mask[:, j], j])
    full = (A @ B.T).reshape(len(drugs), len(doses), k)
    lookup = {d: i for i, d in enumerate(drugs)}
    return [ctx.m0 + full[lookup[d], doses.index(dd)] @ V for d, dd in queries]


# ---- driver -----------------------------------------------------------------------------------------
TRAJ_GRID = [(ell, lam) for ell in (0.75, 1.5, 3.0) for lam in (0.1, 1.0)]
AUTODEC_GRID = [(k_e, wd) for k_e in (2, 4) for wd in (1e-4, 1e-2)]


def run(data: Data, task_sets, only: str | None, shard: int = 0, n_shards: int = 1) -> tuple[list[dict], dict]:
    keep = (lambda name: True) if only is None else (lambda name: re.search(only, name) is not None)
    preds: dict[str, dict[tuple, np.ndarray]] = {}
    extra: dict[str, dict[tuple, dict]] = {}

    def put(name, task, m, **info):
        if keep(name):
            preds.setdefault(name, {})[(task.task_set, task.target)] = m
            extra.setdefault(name, {})[(task.task_set, task.target)] = info

    # one context per task for inner (the target is dropped); one shared context for val/test
    groups = []
    for ts in task_sets:
        tasks = data.tasks(ts)
        if ts == "inner":
            groups += [(Context(data, t), [t]) for t in tasks]
        else:
            groups.append((None, tasks))
    shared = None
    groups = [g for i, g in enumerate(groups) if i % n_shards == shard]
    for ctx, tasks in groups:
        if ctx is None:
            shared = shared or Context(data, tasks[0])
            ctx = shared
        hill = HillModel(ctx)
        trajs = {(ell, lam): Trajectory(ctx, ell, lam) for ell, lam in TRAJ_GRID}
        for t in tasks:
            ctx.task = t
            put("vehicle", t, ctx.m0)
            put("nearest", t, nearest(ctx))
            put("linear", t, linear(ctx), w=t.w_logdose)
            for w in FIXED_W:
                put(f"linear_w{w}", t, linear(ctx, w), w=w)
            for k in (5, 10, 20):
                put(f"linear_k{k}", t, ctx.m0 + project(ctx, linear(ctx) - ctx.m0, k))
            p_hill, w_hill = progress_weight(ctx, hill)
            p_true = float((data.mean(t.target) - ctx.m0) @ ctx.V[0])  # reported only, never used to predict
            put("hill", t, linear(ctx, w_hill), w=w_hill, p_star=p_hill, p_true=p_true,
                p_low=ctx.prog(t.drug, t.low), p_high=ctx.prog(t.drug, t.high))
            p_lin = (1 - t.w_logdose) * ctx.prog(t.drug, t.low) + t.w_logdose * ctx.prog(t.drug, t.high)
            for (ell, lam), traj in trajs.items():
                for tag, p_star, w_res in (("lin", p_lin, t.w_logdose), ("hill", p_hill, w_hill)):
                    name = f"traj_l{ell}_r{lam}_{tag}"
                    put(name, t, trajectory(ctx, traj, p_star, w_res, True), w=w_res, p_star=p_star)
                    put(name + "_nores", t, trajectory(ctx, traj, p_star, w_res, False), w=w_res, p_star=p_star)
        queries = [(t.drug, t.dose) for t in tasks]
        for k_e, wd in AUTODEC_GRID:
            name = f"autodec_e{k_e}_wd{wd:g}"
            if keep(name):
                for t, m in zip(tasks, autodecoder(ctx, queries, k_e, wd)):
                    put(name, t, m)
        if keep("metanet"):
            for t, m in zip(tasks, metanet(ctx, tasks)):
                put("metanet", t, m)
        for r in (2, 3, 4):
            if keep(f"lowrank_r{r}"):
                for t, m in zip(tasks, lowrank(ctx, queries, r)):
                    put(f"lowrank_r{r}", t, m)
        print(f"stats: {tasks[0].task_set} {len(tasks)} task(s) done", flush=True)
    rows = []
    for name, by_task in preds.items():
        for (ts, target), m in by_task.items():
            rows.append({"task_set": ts, "condition": target, "predictor": name, **extra[name][(ts, target)],
                         **mean_errors(data, target, m)})
    return rows, preds


def merge(out: Path, n_shards: int) -> list[dict]:
    """Combine shard outputs into rows.jsonl and means/<predictor>.npz."""
    rows = [r for i in range(n_shards) for r in read_jsonl(out / f"rows.shard{i}.jsonl")]
    write_jsonl(out / "rows.jsonl", rows)
    names = sorted({p.name.split(".shard")[0] for p in (out / "means").glob("*.shard*.npz")})
    for name in names:
        keys, ms = [], []
        for i in range(n_shards):
            path = out / "means" / f"{name}.shard{i}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            keys += list(z["keys"])
            ms += list(z["m_hat"])
        order = np.argsort(keys)
        np.savez_compressed(out / "means" / f"{name}.npz", m_hat=np.array(ms)[order], keys=np.array(keys)[order])
    return rows


def print_summary(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "pearson_delta_deg50", "pearson_delta_all"]
    summary = df.groupby(["task_set", "predictor"])[cols].mean().sort_values(["task_set", "rmse_mean_deg50"])
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merge", type=int, metavar="N_SHARDS", help="only merge N shard outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--task-sets", nargs="+", default=["inner", "val"])
    p.add_argument("--only", help="regex: run only predictors whose name matches")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--tag", default="", help="suffix for the output files (partial runs)")
    a = p.parse_args(argv)
    out = Path(a.root) / "stats"
    if a.merge:
        print_summary(merge(out, a.merge))
        return
    data = Data(a.data)
    rows, preds = run(data, a.task_sets, a.only, a.shard, a.n_shards)
    if a.n_shards > 1:
        a.tag = f"{a.tag}.shard{a.shard}"
    write_jsonl(out / f"rows{a.tag}.jsonl", rows)
    (out / "means").mkdir(parents=True, exist_ok=True)
    for name, by_task in preds.items():
        keys = sorted(by_task)
        np.savez_compressed(out / "means" / f"{name}{a.tag}.npz", m_hat=np.array([by_task[k] for k in keys]),
                            keys=np.array([json.dumps(k) for k in keys]))
    print_summary(rows)


if __name__ == "__main__":
    main()
