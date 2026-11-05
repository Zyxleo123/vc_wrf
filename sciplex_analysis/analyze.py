"""Tables for the six WRF analyses, for chemCPA and CellFlow, from saved artefacts only.

Inputs: extracts (sciplex_analysis.extract), chemCPA per_condition.jsonl, and CellFlow's
committed docs/cellflow/{per_condition,thresholds}.csv. Where a CellFlow extract has not
been copied from PSC yet, CellFlow falls back to its CSVs: frozen metrics, the 13-threshold
grid and an algebraically exact non-program RMSE. Analyses needing generated cells
(full r distributions, fine threshold grid, SD RMSE, non-program MMD) are then chemCPA-only.

Conventions
  e_{m,c}   per-condition error, averaged over a method's training seeds
  Δ_c       e_ref,c − e_new,c (positive: the new method is better)
  bootstrap resamples conditions (= drugs: each drug has one held-out dose per split);
            "two-level" also resamples whole training seeds of each method
Writes <root>/tables/*.csv and <root>/tables/summary.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_eval import EvalSpec
from sciplex_eval.spec import EPS

from .common import (CELLFLOW_DOCS, CELLFLOW_REFERENCE_NAMES, CHEMCPA_RUNS, DEFAULT_ROOT, FINE_QS, GRID13,
                     GRID13_QS, MODELS, N_GENES, N_PROGRAM, PRIMARY_Q, SPEC_DIR, SPEC_QS, SPLITS, extract_path,
                     fractions_below)

FROZEN = ["n_real", "n_pred", "w1_r", "mmd2", "rmse_mean_all", "rmse_mean_deg50", "rmse_mean_program",
          "mean_r_real", "mean_r_pred"] + [f"wrf_{k}_q{q:g}" for q in SPEC_QS for k in ("real", "pred", "err_pp")]
GENE_SETS = ("all", "deg50", "program", "nonprogram")
PAIR_METRICS = (["wrf_err_pp_q0.9", "wrf_err_pp_q0.95", "wrf_err_pp_q0.975", "wrf_err_pp_grid13_mean",
                 "wrf_err_pp_fine_mean", "w1_r", "mmd2", "mmd2_nonprogram"]
                + [f"rmse_mean_{s}" for s in GENE_SETS] + ["rmse_sd_all", "rmse_sd_nonprogram", "rmse_sd_deg50"])
WRF_ONLY = ["wrf_err_pp_q0.9", "wrf_err_pp_q0.95", "wrf_err_pp_q0.975", "wrf_err_pp_grid13_mean",
            "wrf_err_pp_fine_mean"]
N_RANDOM_SPLITS = 500


# ---------------------------------------------------------------------------------------------
class Measured:
    def __init__(self, root: Path, spec: EvalSpec):
        z = np.load(Path(root) / "extracts" / "measured.npz")
        self.r, self.condition, self.drug = z["r"], z["condition"].astype(str), z["drug"].astype(str)
        self.dose, self.replicate, self.split = z["dose"], z["replicate"].astype(str), z["split"].astype(str)
        self.heldout = [str(c) for c in z["heldout"]]
        self.mean, self.sd = z["mean"].astype(np.float64), z["sd"].astype(np.float64)
        self.control_mean = z["control_mean"]
        self.program_idx = z["program_idx"]
        self.nonprogram_idx = np.setdiff1d(np.arange(len(self.control_mean)), self.program_idx)
        self.row = {c: i for i, c in enumerate(self.heldout)}
        self.index = pd.Series(np.arange(len(self.r))).groupby(self.condition).indices
        self.control_r = self.r[self.split == "control"]
        self.r_zero_program = spec.mu0 / (spec.s0 + EPS)  # r of a cell with no program UMIs
        self.taus = {"grid13": np.quantile(self.control_r, GRID13_QS), "fine": np.quantile(self.control_r, FINE_QS)}
        self.qs = {"grid13": GRID13_QS, "fine": FINE_QS}
        self.primary_tau = float(np.quantile(self.control_r, PRIMARY_Q))
        self.mid_tau = float(np.quantile(self.control_r, 0.5))

    def r_of(self, condition: str, replicate: str | None = None) -> np.ndarray:
        idx = self.index[condition]
        if replicate is not None:
            idx = idx[self.replicate[idx] == replicate]
        return self.r[idx]

    def conditions(self, split: str) -> list[str]:
        return [c for c in self.heldout if self.split[self.index[c][0]] == split]

    def attributes(self, condition: str) -> dict:
        i = self.index[condition][0]
        return {"split": self.split[i], "drug": self.drug[i], "dose": float(self.dose[i])}


class Extract:
    def __init__(self, path: Path):
        z = np.load(path)
        self.conditions = [str(c) for c in z["conditions"]]
        offsets = np.concatenate([[0], np.cumsum(z["n_pred"])])
        self._r = {c: z["r"][offsets[i]:offsets[i + 1]] for i, c in enumerate(self.conditions)}
        self.mean, self.sd = z["mean"].astype(np.float64), z["sd"].astype(np.float64)
        self.mmd2_nonprogram = z["mmd2_nonprogram"]
        self.row = {c: i for i, c in enumerate(self.conditions)}

    def r_of(self, condition: str) -> np.ndarray:
        return self._r[condition]


# ---------------------------------------------------------------------------------------------
def run_index() -> list[dict]:
    """Every (model, method, seed, split) with frozen per-condition scores (sampling offset 0)."""
    runs = []
    for method in MODELS["chemcpa"]["methods"]:
        for path in sorted((CHEMCPA_RUNS / method).glob("seed_*/*_sample_s0/per_condition.jsonl")):
            runs.append({"model": "chemcpa", "method": method, "seed": int(path.parent.parent.name.split("_")[1]),
                         "split": path.parent.name.split("_")[0], "frozen": path})
    cf = pd.read_csv(CELLFLOW_DOCS / "per_condition.csv")
    for (method, seed, split), _ in cf[cf.method.isin(MODELS["cellflow"]["methods"])].groupby(["method", "seed", "split"]):
        runs.append({"model": "cellflow", "method": method, "seed": int(seed), "split": split, "frozen": None})
    return runs


def frozen_rows(run: dict, cellflow_pc: pd.DataFrame) -> pd.DataFrame:
    if run["model"] == "chemcpa":
        frame = pd.read_json(run["frozen"], lines=True)
    else:
        frame = cellflow_pc[(cellflow_pc.method == run["method"]) & (cellflow_pc.seed == run["seed"])
                            & (cellflow_pc.split == run["split"])]
    return frame[["condition"] + FROZEN].reset_index(drop=True)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def nonprogram_rmse_from_frozen(all_, program) -> np.ndarray:
    """Exact: G·RMSE_all² = P·RMSE_program² + (G−P)·RMSE_nonprogram²."""
    return np.sqrt(np.maximum(N_GENES * np.square(all_) - N_PROGRAM * np.square(program), 0) / (N_GENES - N_PROGRAM))


def decomposition(r_real, r_pred, tau) -> dict:
    """Signed WRF error at tau split into location, scale and remaining shape (pp).

    location: shift the measured scores to the predicted mean;
    scale:    additionally rescale them to the predicted SD;
    shape:    the rest (subpopulation sizes, point masses, skew)."""
    mu_m, mu_p = r_real.mean(), r_pred.mean()
    sd_m, sd_p = r_real.std(), r_pred.std()
    f_m = fractions_below(r_real, [tau])[0]
    f_loc = fractions_below(r_real + (mu_p - mu_m), [tau])[0]
    f_ls = fractions_below(mu_p + (r_real - mu_m) * (sd_p / sd_m if sd_m > 0 else 1.0), [tau])[0]
    f_p = fractions_below(r_pred, [tau])[0]
    return {"loc": 100 * (f_loc - f_m), "scale": 100 * (f_ls - f_loc), "shape": 100 * (f_p - f_ls),
            "total": 100 * (f_p - f_m)}


def distribution_stats(r: np.ndarray, r_zero: float) -> dict:
    q10, q25, q50, q75, q90 = np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"mu": float(r.mean()), "var": float(r.var()), "sd": float(r.std()), "q10": q10, "q25": q25,
            "q50": q50, "q75": q75, "q90": q90, "frac_zero_program": float(np.mean(r >= r_zero - 1e-9))}


def is_trained(q: float, trained_qs) -> bool:
    return any(abs(q - t) < 1e-9 for t in trained_qs)


# ---------------------------------------------------------------------------------------------
def build_model_tables(M: Measured, root: Path):
    cellflow_pc = pd.read_csv(CELLFLOW_DOCS / "per_condition.csv")
    cellflow_thr = pd.read_csv(CELLFLOW_DOCS / "thresholds.csv")
    pc_rows, thr_rows, dist_rows, availability = [], [], [], []
    for run in run_index():
        model, method, seed, split = run["model"], run["method"], run["seed"], run["split"]
        entry = MODELS[model]["methods"][method]
        meta = {"model": model, "method": method, "role": entry["role"], "seed": seed, "split": split}
        frozen = frozen_rows(run, cellflow_pc)
        path = extract_path(root, model, method, seed, split, 0)
        ex = Extract(path) if path.exists() else None
        availability.append({**meta, "extract": ex is not None, "extract_path": str(path)})
        if ex is None and model == "chemcpa":
            warnings.warn(f"missing chemCPA extract {path}; distribution-level metrics skipped for this run")
        for _, fr in frozen.iterrows():
            c = fr["condition"]
            rec = {**meta, **M.attributes(c), "condition": c, **{k: fr[k] for k in FROZEN}}
            for q in SPEC_QS:
                rec[f"bias_pp_q{q:g}"] = 100 * (fr[f"wrf_pred_q{q:g}"] - fr[f"wrf_real_q{q:g}"])
            rec["rmse_mean_nonprogram"] = float(nonprogram_rmse_from_frozen(fr["rmse_mean_all"], fr["rmse_mean_program"]))
            rec["source"] = "csv" if ex is None else "extract"
            r_real = M.r_of(c)
            if ex is not None:
                r_pred = ex.r_of(c)
                i, j = ex.row[c], M.row[c]
                m_real, m_pred, sd_real, sd_pred = M.mean[j], ex.mean[i], M.sd[j], ex.sd[i]
                deg = np.argsort(-np.abs(m_real - M.control_mean), kind="stable")[:50]
                sets = {"all": slice(None), "deg50": deg, "program": M.program_idx, "nonprogram": M.nonprogram_idx}
                for name, s in sets.items():
                    rec[f"rmse_mean_{name}_check" if name != "nonprogram" else "rmse_mean_nonprogram"] = rmse(m_real[s], m_pred[s])
                    rec[f"rmse_sd_{name}"] = rmse(sd_real[s], sd_pred[s])
                    rec[f"effect_corr_{name}"] = pearson(m_pred[s] - M.control_mean[s], m_real[s] - M.control_mean[s])
                rec["mmd2_nonprogram"] = float(ex.mmd2_nonprogram[i])
                stats_p = distribution_stats(r_pred, M.r_zero_program)
                rec.update({f"{k}_r_pred": v for k, v in stats_p.items()})
                for tag, tau in (("q0.95", M.primary_tau), ("q0.5", M.mid_tau)):
                    rec.update({f"dec_{k}_{tag}": v for k, v in decomposition(r_real, r_pred, tau).items()})
                dist_rows.append({**meta, "condition": c, "population": "generated", **stats_p})
                for grid in ("grid13", "fine"):
                    taus = M.taus[grid]
                    f_real, f_pred = fractions_below(r_real, taus), fractions_below(r_pred, taus)
                    errors = 100 * np.abs(f_pred - f_real)
                    rec[f"wrf_err_pp_{grid}_mean"] = float(errors.mean())
                    for q, t, a, b in zip(M.qs[grid], taus, f_real, f_pred):
                        thr_rows.append({**meta, "condition": c, "grid": grid, "q": q, "tau": t, "f_real": a,
                                         "f_pred": b, "err_pp": 100 * abs(b - a), "bias_pp": 100 * (b - a),
                                         "trained": is_trained(q, entry["train_qs"])})
            else:  # CellFlow without a PSC extract: its 13-threshold rows
                part = cellflow_thr[(cellflow_thr.method == method) & (cellflow_thr.seed == seed)
                                    & (cellflow_thr.split == split) & (cellflow_thr.condition == c)]
                rec["wrf_err_pp_grid13_mean"] = float(part.err_pp.mean())
                for _, t in part.iterrows():
                    thr_rows.append({**meta, "condition": c, "grid": "grid13", "q": float(t.q), "tau": float(t.tau),
                                     "f_real": t.wrf_real, "f_pred": t.wrf_pred, "err_pp": t.err_pp,
                                     "bias_pp": 100 * (t.wrf_pred - t.wrf_real),
                                     "trained": is_trained(float(t.q), entry["train_qs"])})
            stats_m = distribution_stats(r_real, M.r_zero_program)
            rec.update({f"{k}_r_real": v for k, v in stats_m.items()})
            pc_rows.append(rec)
    return pd.DataFrame(pc_rows), pd.DataFrame(thr_rows), pd.DataFrame(dist_rows), pd.DataFrame(availability)


def interpolation_neighbors(M: Measured, condition: str):
    a = M.attributes(condition)
    train = (M.drug == a["drug"]) & (M.split == "train")
    doses = sorted(set(M.dose[train]))
    low, high = max(d for d in doses if d < a["dose"]), min(d for d in doses if d > a["dose"])
    w = (np.log(a["dose"]) - np.log(low)) / (np.log(high) - np.log(low))
    r_low = M.r[train & (M.dose == low)]
    r_high = M.r[train & (M.dose == high)]
    return low, high, w, r_low, r_high


def build_reference_tables(M: Measured):
    """Vehicle, replicate and log-dose interpolation from measured cells; nearest-dose and
    adjacent-dose-mixture rows (and frozen expression metrics of every reference) from
    CellFlow's CSVs, which used the same evaluator, spec and panel."""
    cf_pc = pd.read_csv(CELLFLOW_DOCS / "per_condition.csv")
    cf_thr = pd.read_csv(CELLFLOW_DOCS / "thresholds.csv")
    pc_rows, thr_rows = [], []
    for split in SPLITS:
        for c in M.conditions(split):
            r_real = M.r_of(c)
            reps = [M.r_of(c, rep) for rep in ("rep1", "rep2")]
            low, high, w, r_low, r_high = interpolation_neighbors(M, c)
            per_grid = {}
            for grid in ("grid13", "fine"):
                taus = M.taus[grid]
                f_real = fractions_below(r_real, taus)
                f = {"vehicle": fractions_below(M.control_r, taus),
                     "interpolation": (1 - w) * fractions_below(r_low, taus) + w * fractions_below(r_high, taus)}
                f1, f2 = (fractions_below(rr, taus) for rr in reps)
                per_grid[grid] = (f_real, f, f1, f2)
                for name, fp in f.items():
                    for q, t, a, b in zip(M.qs[grid], taus, f_real, fp):
                        thr_rows.append({"model": "reference", "method": name, "role": "reference", "seed": 0,
                                         "split": split, "condition": c, "grid": grid, "q": q, "tau": t, "f_real": a,
                                         "f_pred": b, "err_pp": 100 * abs(b - a), "bias_pp": 100 * (b - a),
                                         "trained": False})
                for q, t, a, x, y in zip(M.qs[grid], taus, f_real, f1, f2):
                    thr_rows.append({"model": "reference", "method": "replicate", "role": "reference", "seed": 0,
                                     "split": split, "condition": c, "grid": grid, "q": q, "tau": t, "f_real": a,
                                     "f_pred": np.nan, "err_pp": 100 * abs(x - y), "bias_pp": np.nan, "trained": False})
            for name in ("vehicle", "interpolation", "replicate"):
                rec = {"model": "reference", "method": name, "role": "reference", "seed": 0, **M.attributes(c),
                       "condition": c, "source": "measured"}
                for grid in ("grid13", "fine"):
                    f_real, f, f1, f2 = per_grid[grid]
                    err = 100 * np.abs(f1 - f2) if name == "replicate" else 100 * np.abs(f[name] - f_real)
                    rec[f"wrf_err_pp_{grid}_mean"] = float(err.mean())
                spec_taus = np.quantile(M.control_r, SPEC_QS)
                f_real = fractions_below(r_real, spec_taus)
                if name == "replicate":
                    e = 100 * np.abs(fractions_below(reps[0], spec_taus) - fractions_below(reps[1], spec_taus))
                    fp = np.full(len(SPEC_QS), np.nan)
                else:
                    fp = (fractions_below(M.control_r, spec_taus) if name == "vehicle" else
                          (1 - w) * fractions_below(r_low, spec_taus) + w * fractions_below(r_high, spec_taus))
                    e = 100 * np.abs(fp - f_real)
                for q, a, b, err in zip(SPEC_QS, f_real, fp, e):
                    rec.update({f"wrf_real_q{q:g}": a, f"wrf_pred_q{q:g}": b, f"wrf_err_pp_q{q:g}": err,
                                f"bias_pp_q{q:g}": 100 * (b - a)})
                if name == "interpolation":
                    rec.update({"dose_low": low, "dose_high": high, "interp_w": w})
                pc_rows.append(rec)
    measured_pc = pd.DataFrame(pc_rows)
    # Frozen expression metrics of vehicle/replicate, and all of nearest-dose / adjacent mixture.
    cf_ref = cf_pc[cf_pc.method.isin(CELLFLOW_REFERENCE_NAMES)].copy()
    cf_ref["method"] = cf_ref.method.map(CELLFLOW_REFERENCE_NAMES)
    cf_ref["rmse_mean_nonprogram"] = nonprogram_rmse_from_frozen(cf_ref.rmse_mean_all, cf_ref.rmse_mean_program)
    expr_cols = ["w1_r", "mmd2", "rmse_mean_all", "rmse_mean_deg50", "rmse_mean_program", "rmse_mean_nonprogram",
                 "n_pred", "mean_r_pred"]
    merged = measured_pc.merge(cf_ref[["method", "split", "condition"] + expr_cols], on=["method", "split", "condition"],
                               how="left")
    extra = cf_ref[cf_ref.method.isin(["nearest_dose", "adjacent_dose_mixture"])].copy()
    extra = extra.assign(model="reference", role="reference", seed=0, source="csv")
    for q in SPEC_QS:
        extra[f"bias_pp_q{q:g}"] = 100 * (extra[f"wrf_pred_q{q:g}"] - extra[f"wrf_real_q{q:g}"])
    extra = extra.drop(columns=[c for c in ("n_real",) if c in extra])
    attrs = pd.DataFrame([{"condition": c, **M.attributes(c)} for c in M.heldout])
    extra = extra.drop(columns=[c for c in ("split",) if c in extra]).merge(attrs, on="condition")
    ref_pc = pd.concat([merged, extra], ignore_index=True)
    cf_ref_thr = cf_thr[cf_thr.method.isin(["nearest_dose", "adjacent_dose_mixture"])]
    extra_thr = pd.DataFrame({"model": "reference", "method": cf_ref_thr.method, "role": "reference", "seed": 0,
                              "split": cf_ref_thr.split, "condition": cf_ref_thr.condition, "grid": "grid13",
                              "q": cf_ref_thr.q.astype(float), "tau": cf_ref_thr.tau, "f_real": cf_ref_thr.wrf_real,
                              "f_pred": cf_ref_thr.wrf_pred, "err_pp": cf_ref_thr.err_pp,
                              "bias_pp": 100 * (cf_ref_thr.wrf_pred - cf_ref_thr.wrf_real), "trained": False})
    return ref_pc, pd.concat([pd.DataFrame(thr_rows), extra_thr], ignore_index=True)


# ---------------------------------------------------------------------------------------------
def condition_table(M: Measured, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per held-out condition attributes, and measured-vs-measured variability per threshold."""
    rows, var_rows = [], []
    taus = M.taus["grid13"]
    for c in M.heldout:
        a = M.attributes(c)
        r_real = M.r_of(c)
        reps = [M.r_of(c, rep) for rep in ("rep1", "rep2")]
        drug_train = (M.drug == a["drug"]) & (M.split == "train")
        r10 = M.r[drug_train & (M.dose == 10000)]
        low, high, w, _, _ = interpolation_neighbors(M, c)
        f_real = fractions_below(r_real, taus)
        f1, f2 = (fractions_below(rr, taus) for rr in reps)
        n1, n = len(reps[0]), len(r_real)
        split_err = np.empty((N_RANDOM_SPLITS, len(taus)))
        for b in range(N_RANDOM_SPLITS):
            perm = rng.permutation(n)
            split_err[b] = 100 * np.abs(fractions_below(r_real[perm[:n1]], taus) - fractions_below(r_real[perm[n1:]], taus))
        for k, q in enumerate(GRID13_QS):
            p = f_real[k]
            row = {"condition": c, **a, "q": q, "f_real": p, "replicate_err_pp": 100 * abs(f1[k] - f2[k]),
                   "random_split_err_pp": split_err[:, k].mean(),
                   "random_split_err_pp_p95": np.percentile(split_err[:, k], 95)}
            for n_pred in (512, 1024):
                # E|f_hat - f| for a perfect generator: both fractions binomial around p.
                row[f"perfect_model_err_pp_n{n_pred}"] = 100 * np.sqrt(2 / np.pi) * np.sqrt(p * (1 - p) * (1 / n + 1 / n_pred))
            var_rows.append(row)
        primary = GRID13_QS.index(PRIMARY_Q)
        rows.append({"condition": c, **a, "n_real": n, "n_rep1": n1, "n_rep2": len(reps[1]),
                     "wrf_real_q0.95": f_real[primary], "wrf_real_q0.5": f_real[GRID13_QS.index(0.5)],
                     "mean_r_real": float(r_real.mean()),
                     "potency_mean_r_10uM": float(r10.mean()) if len(r10) else np.nan,
                     "potency_wrf_q0.5_10uM": float(fractions_below(r10, [M.mid_tau])[0]) if len(r10) else np.nan,
                     "n_train_cells_drug": int(drug_train.sum()), "dose_low": low, "dose_high": high, "interp_w": w,
                     "replicate_err_pp_q0.95": 100 * abs(f1[primary] - f2[primary]),
                     "random_split_err_pp_q0.95": split_err[:, primary].mean()})
    return pd.DataFrame(rows), pd.DataFrame(var_rows)


def generation_noise(M: Measured, root: Path) -> pd.DataFrame:
    """|f(sampling offset 0) − f(offset 1)| for chemCPA runs with two sampling draws."""
    rows = []
    for method, entry in MODELS["chemcpa"]["methods"].items():
        for seed in range(3):
            p0, p1 = (extract_path(root, "chemcpa", method, seed, "test", s) for s in (0, 1))
            if not (p0.exists() and p1.exists()):
                continue
            e0, e1 = Extract(p0), Extract(p1)
            for c in e0.conditions:
                d = 100 * np.abs(fractions_below(e0.r_of(c), M.taus["grid13"]) - fractions_below(e1.r_of(c), M.taus["grid13"]))
                rows += [{"model": "chemcpa", "method": method, "role": entry["role"], "seed": seed, "condition": c,
                          "q": q, "resample_err_pp": v} for q, v in zip(GRID13_QS, d)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def error_matrix(pc: pd.DataFrame, model: str, method: str, split: str, metric: str) -> pd.DataFrame | None:
    part = pc[(pc.model == model) & (pc.method == method) & (pc.split == split)]
    if part.empty or metric not in part or part[metric].isna().all():
        return None
    return part.pivot_table(index="condition", columns="seed", values=metric, aggfunc="mean")


def paired_stats(ref: pd.DataFrame, new: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> dict:
    conds = ref.index.intersection(new.index)
    A, B = ref.loc[conds].to_numpy(np.float64), new.loc[conds].to_numpy(np.float64)
    d = A.mean(1) - B.mean(1)
    C = len(d)
    idx = rng.integers(0, C, (n_boot, C))
    boots = d[idx].mean(1)
    sa = rng.integers(0, A.shape[1], (n_boot, A.shape[1]))
    sb = rng.integers(0, B.shape[1], (n_boot, B.shape[1]))
    a_draw = np.take_along_axis(A[None].repeat(n_boot, 0), sa[:, None, :].repeat(C, 1), 2).mean(2)
    b_draw = np.take_along_axis(B[None].repeat(n_boot, 0), sb[:, None, :].repeat(C, 1), 2).mean(2)
    two = (np.take_along_axis(a_draw, idx, 1) - np.take_along_axis(b_draw, idx, 1)).mean(1)
    pairs = np.array([(A[:, i] - B[:, j]).mean() for i in range(A.shape[1]) for j in range(B.shape[1])])
    return {"n_conditions": C, "n_seeds_ref": A.shape[1], "n_seeds_new": B.shape[1], "mean_ref": A.mean(),
            "mean_new": B.mean(), "delta_mean": d.mean(), "delta_median": np.median(d),
            "win_fraction": float(np.mean(d > 0)), "n_win": int((d > 0).sum()), "n_loss": int((d < 0).sum()),
            "n_tie": int((d == 0).sum()), "largest_gain": d.max(), "largest_loss": d.min(),
            "delta_mean_without_largest_gain": float(np.sort(d)[:-1].mean()) if C > 1 else np.nan,
            "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
            "p_boot_le0": float(np.mean(boots <= 0)),
            "ci95_two_level_low": np.percentile(two, 2.5), "ci95_two_level_high": np.percentile(two, 97.5),
            "seed_pair_delta_mean": pairs.mean(), "seed_pair_delta_sd": pairs.std(ddof=1) if len(pairs) > 1 else np.nan,
            "seed_pair_delta_min": pairs.min(), "seed_pair_delta_max": pairs.max(),
            "seed_pairs_positive": float(np.mean(pairs > 0))}


def contrasts() -> list[dict]:
    out = []
    for model, spec in MODELS.items():
        roles = {e["role"]: m for m, e in spec["methods"].items()}
        base, wrf3, wrf13 = roles["base"], roles["wrf3"], roles["wrf13"]
        items = [("base_vs_wrf3", base, wrf3, "λ=0 → WRF loss, 3 τ"),
                 ("base_vs_wrf13", base, wrf13, "λ=0 → WRF loss, 13 τ"),
                 ("wrf3_vs_wrf13", wrf3, wrf13, "3 τ → 13 τ")]
        if "mmd" in roles:
            items += [("base_vs_mmd", base, roles["mmd"], "λ=0 → MMD loss"),
                      ("mmd_vs_wrf3", roles["mmd"], wrf3, "MMD loss → WRF loss, 3 τ"),
                      ("mmd_vs_wrf13", roles["mmd"], wrf13, "MMD loss → WRF loss, 13 τ")]
        for role in ("base", "wrf3", "wrf13") + (("mmd",) if "mmd" in roles else ()):
            for ref in ("vehicle", "interpolation", "adjacent_dose_mixture", "nearest_dose"):
                items.append((f"{ref}_vs_{role}", f"reference:{ref}", roles[role], f"{ref} → {role}"))
        if model == "chemcpa":  # matched λ and h, 3 vs 13 τ (validation only)
            items += [("matched_l10_h0.5_3vs13", "nbwrf_l10_h0.5", "nbwrfgrid_l10_h0.5", "3 τ → 13 τ (λ=10, h=0.5)"),
                      ("matched_l100_h0.5_3vs13", "nbwrf_l100_h0.5", "nbwrfgrid_l100_h0.5", "3 τ → 13 τ (λ=100, h=0.5)")]
        for name, ref, new, label in items:
            out.append({"model": model, "contrast": name, "ref": ref, "new": new, "label": label})
    return out


def resolve(pc: pd.DataFrame, model: str, method: str, split: str, metric: str):
    if method.startswith("reference:"):
        return error_matrix(pc, "reference", method.split(":", 1)[1], split, metric)
    return error_matrix(pc, model, method, split, metric)


def build_paired(pc: pd.DataFrame, thr: pd.DataFrame, rng, n_boot: int):
    rows, thr_rows = [], []
    for k in contrasts():
        for split in SPLITS:
            for metric in PAIR_METRICS:
                ref = resolve(pc, k["model"], k["ref"], split, metric)
                new = resolve(pc, k["model"], k["new"], split, metric)
                if ref is None or new is None:
                    continue
                rows.append({**k, "split": split, "metric": metric, **paired_stats(ref, new, rng, n_boot)})
            for grid in ("grid13", "fine"):
                part = thr[(thr.grid == grid) & (thr.split == split)]
                for q in sorted(part.q.unique()):
                    sub = part[np.isclose(part.q, q)]
                    def mat(method):
                        model, m = ("reference", method.split(":", 1)[1]) if method.startswith("reference:") else (k["model"], method)
                        s = sub[(sub.model == model) & (sub.method == m)]
                        return None if s.empty else s.pivot_table(index="condition", columns="seed", values="err_pp")
                    ref, new = mat(k["ref"]), mat(k["new"])
                    if ref is None or new is None:
                        continue
                    st = paired_stats(ref, new, rng, min(n_boot, 2000))
                    thr_rows.append({**k, "split": split, "grid": grid, "q": q,
                                     **{key: st[key] for key in ("delta_mean", "ci95_low", "ci95_high", "win_fraction",
                                                                 "mean_ref", "mean_new")}})
    return pd.DataFrame(rows), pd.DataFrame(thr_rows)


# ---------------------------------------------------------------------------------------------
def method_summary(pc: pd.DataFrame) -> pd.DataFrame:
    """Per (model, method, split): mean over conditions per seed, then mean ± SD over seeds."""
    metrics = [c for c in PAIR_METRICS + ["bias_pp_q0.95", "effect_corr_all", "effect_corr_deg50",
                                          "effect_corr_nonprogram", "rmse_sd_program"] if c in pc]
    rows = []
    for (model, method, split), part in pc.groupby(["model", "method", "split"]):
        per_seed = part.groupby("seed")[metrics].mean()
        entry = MODELS.get(model, {}).get("methods", {}).get(method, {})
        row = {"model": model, "method": method, "split": split, "role": part.role.iloc[0],
               "lam": entry.get("lam"), "h": entry.get("h"), "n_taus": entry.get("n_taus"),
               "checkpoint_rule": entry.get("checkpoint_rule"), "n_seeds": len(per_seed),
               "n_conditions": part.condition.nunique()}
        for m in metrics:
            row[f"{m}_mean"] = per_seed[m].mean()
            row[f"{m}_seed_sd"] = per_seed[m].std(ddof=1) if len(per_seed) > 1 else np.nan
        cm = part.groupby("condition")[["wrf_pred_q0.95", "wrf_real_q0.95"]].mean()
        row["pred_vs_measured_pearson_q0.95"] = pearson(cm["wrf_pred_q0.95"], cm["wrf_real_q0.95"]) if cm["wrf_pred_q0.95"].notna().all() else np.nan
        row["pred_range_q0.95"] = f"{100 * cm['wrf_pred_q0.95'].min():.1f}–{100 * cm['wrf_pred_q0.95'].max():.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def checks(M: Measured, pc: pd.DataFrame, ref_thr: pd.DataFrame) -> dict:
    out = {}
    out["grid13_taus_max_abs_diff_vs_config"] = float(np.max(np.abs(M.taus["grid13"] - np.array(GRID13["taus"]))))
    cf_thr = pd.read_csv(CELLFLOW_DOCS / "thresholds.csv")
    mine = ref_thr[(ref_thr.method == "vehicle") & (ref_thr.grid == "grid13")]
    theirs = cf_thr[cf_thr.method == "unchanged_control"].rename(columns={"err_pp": "err_cf", "wrf_real": "real_cf"})
    j = mine.merge(theirs[["split", "condition", "q", "err_cf", "real_cf"]], on=["split", "condition", "q"])
    out["vehicle_grid13_err_max_abs_diff_vs_cellflow_csv_pp"] = float(np.abs(j.err_pp - j.err_cf).max())
    out["measured_grid13_fraction_max_abs_diff_vs_cellflow_csv"] = float(np.abs(j.f_real - j.real_cf).max())
    out["vehicle_rows_compared"] = len(j)
    ex = pc[pc.source.eq("extract") & pc.model.ne("reference")]
    if len(ex):
        for name in ("all", "deg50", "program"):
            out[f"extract_vs_frozen_rmse_mean_{name}_max_abs_diff"] = float(np.abs(ex[f"rmse_mean_{name}_check"] - ex[f"rmse_mean_{name}"]).max())
        out["extract_vs_frozen_mean_r_pred_max_abs_diff"] = float(np.abs(ex["mu_r_pred"] - ex["mean_r_pred"]).max())
        out["extract_vs_frozen_wrf_err_q0.95_max_abs_diff_pp"] = float(np.abs(
            ex["wrf_err_pp_q0.95"] - 100 * np.abs(ex["wrf_pred_q0.95"] - ex["wrf_real_q0.95"])).max())
        derived = nonprogram_rmse_from_frozen(ex.rmse_mean_all, ex.rmse_mean_program)
        out["nonprogram_rmse_derived_vs_extract_max_abs_diff"] = float(np.abs(derived - ex.rmse_mean_nonprogram).max())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root = Path(a.root)
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    spec = EvalSpec.load(SPEC_DIR)
    M = Measured(root, spec)

    model_pc, model_thr, dist, availability = build_model_tables(M, root)
    ref_pc, ref_thr = build_reference_tables(M)
    pc = pd.concat([model_pc, ref_pc], ignore_index=True)
    thr = pd.concat([model_thr, ref_thr], ignore_index=True)
    for split in SPLITS:
        for c in M.conditions(split):
            dist = pd.concat([dist, pd.DataFrame([{"model": "measured", "method": "measured", "role": "measured",
                                                   "seed": 0, "split": split, "condition": c,
                                                   "population": "measured",
                                                   **distribution_stats(M.r_of(c), M.r_zero_program)}])])
    dist = pd.concat([dist, pd.DataFrame([{"model": "measured", "method": "vehicle", "role": "reference", "seed": 0,
                                           "split": "control", "condition": "A549|Vehicle|0", "population": "measured",
                                           **distribution_stats(M.control_r, M.r_zero_program)}])])
    conditions, variability = condition_table(M, rng)
    noise = generation_noise(M, root)
    paired, paired_thr = build_paired(pc, thr, rng, a.n_boot)
    methods = method_summary(pc)

    pc.to_csv(out / "per_condition.csv", index=False)
    thr.to_csv(out / "thresholds.csv", index=False)
    dist.to_csv(out / "distribution_summary.csv", index=False)
    conditions.to_csv(out / "conditions.csv", index=False)
    variability.to_csv(out / "measured_variability.csv", index=False)
    noise.to_csv(out / "generation_noise.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    paired_thr.to_csv(out / "paired_thresholds.csv", index=False)
    methods.to_csv(out / "methods.csv", index=False)
    availability.to_csv(out / "availability.csv", index=False)

    summary = {
        "checks": checks(M, pc, ref_thr),
        "taus": {"primary": M.primary_tau, "q0.5": M.mid_tau, "grid13": list(M.taus["grid13"])},
        "r_zero_program": M.r_zero_program,
        "control_frac_zero_program": float(np.mean(M.control_r >= M.r_zero_program - 1e-9)),
        "cellflow_extracts_available": bool(availability[availability.model == "cellflow"].extract.all()),
        "chemcpa_extracts_missing": availability[(availability.model == "chemcpa") & ~availability.extract].extract_path.tolist(),
        "loss_aggregation": {m: MODELS[m]["loss_aggregation"] for m in MODELS},
        "n_boot": a.n_boot,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["checks"], indent=1))
    key = paired[(paired.split == "test") & (paired.metric == "wrf_err_pp_q0.95")
                 & paired.contrast.isin(["base_vs_wrf3", "base_vs_wrf13", "wrf3_vs_wrf13", "base_vs_mmd",
                                         "vehicle_vs_wrf3", "interpolation_vs_wrf3"])]
    print(key[["model", "contrast", "mean_ref", "mean_new", "delta_mean", "ci95_low", "ci95_high", "win_fraction"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"tables -> {out}")


if __name__ == "__main__":
    main()
