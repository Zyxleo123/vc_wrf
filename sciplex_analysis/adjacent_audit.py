"""Does the adjacent-dose mixture beat CellFlow and chemCPA on merit, or on information the
models are not given?

The reference (CellFlow, git bde8c6c:experiments/cellflow_v1/source/run_cellflow_v1.py,
lines 422-446) predicts held-out dose d of drug D by resampling real *training* cells of D
at the two neighbouring training doses, weighted by log1p dose (≈ 0.49 / 0.50). Each
replicate's cells come from that replicate's wells, in the held-out condition's rep1/rep2
proportion. Every held-out well shares its culture plate with one of its two
adjacent-dose wells, so about half the source cells are batch-matched to the target.
Neither model sees plates; CellFlow sees the replicate, chemCPA does not.

  build --task 0   reproduce the published reference with its own RNG stream and compare
                   it with docs/cellflow/per_condition.csv; audit every source cell; score
                   the program-output chemCPA runs on the 13-threshold grid; write the
                   finite-sample floor of a perfect predictor (WRF only).
  build --task i   ablation VARIANTS[i-1], over --seeds sampling seeds and both splits,
                   scored by the frozen evaluator plus the 13-threshold grid.
  compare          paired condition bootstrap of every ablation against every model and
                   against the full mixture, from saved rows only; writes report.md.

Outputs go under <root>/adjacent_audit/. Finished row files are skipped, and a file
written under a different variant definition is refused, not reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sciplex_eval import EvalSpec, evaluate_predictions, response_score, response_score_from_program
from sciplex_eval.metrics import weak_responder_fraction

from .common import CELLFLOW_DOCS, CHEMCPA_RUNS, DEFAULT_PANEL, DEFAULT_ROOT, GRID13, REPO, SPEC_DIR, SPLITS

N_CELLS = 512  # CellFlow's prediction_cells_per_condition, used by every reference it published
GRID_TAUS = np.asarray(GRID13["taus"], dtype=np.float64)
METRICS = ["wrf_err_pp_q0.95", "wrf_err_pp_grid13_mean", "w1_r", "mmd2", "rmse_mean_deg50", "rmse_mean_program",
           "wrf_err_pp_q0.9", "wrf_err_pp_q0.975", "rmse_mean_all"]
REPORT_METRICS = ["wrf_err_pp_q0.95", "wrf_err_pp_grid13_mean", "w1_r", "mmd2", "rmse_mean_deg50"]
METRIC_LABELS = {"wrf_err_pp_q0.95": "WRF err q0.95 (pp)", "wrf_err_pp_grid13_mean": "13-τ mean (pp)",
                 "w1_r": "W1(r)", "mmd2": "MMD²", "rmse_mean_deg50": "RMSE DEG50",
                 "rmse_mean_program": "RMSE program"}

# One ingredient changed at a time, relative to "adjacent" (the published recipe).
#   weight: share of cells from the higher dose (None: log1p-dose weight, as published)
#   rep:    matched (target rep's wells), cross (the other replicate's wells: no plate is
#           shared with the target), pooled (both replicates, no rep1/rep2 split)
#   drug:   same, other (adjacent doses of one random other drug), class (all other drugs)
#   plate:  vehicle only; target = controls from the held-out well's own culture plate
VARIANTS = {
    "adjacent": dict(source="adjacent", weight=None, rep="matched", drug="same",
                     label="Adjacent-dose mixture (published recipe)"),
    "adjacent_w0": dict(source="adjacent", weight=0.0, rep="matched", drug="same",
                        label="Lower adjacent dose only"),
    "adjacent_w0.25": dict(source="adjacent", weight=0.25, rep="matched", drug="same",
                           label="Mixture, 25% upper dose"),
    "adjacent_w0.75": dict(source="adjacent", weight=0.75, rep="matched", drug="same",
                           label="Mixture, 75% upper dose"),
    "adjacent_w1": dict(source="adjacent", weight=1.0, rep="matched", drug="same",
                        label="Upper adjacent dose only"),
    "adjacent_cross_rep": dict(source="adjacent", weight=None, rep="cross", drug="same",
                               label="Mixture from the other replicate's wells (no shared plate)"),
    "adjacent_pooled_rep": dict(source="adjacent", weight=None, rep="pooled", drug="same",
                                label="Mixture, replicates pooled"),
    "other_drug": dict(source="adjacent", weight=None, rep="matched", drug="other",
                       label="Mixture of a random other HDACi"),
    "class_pool": dict(source="adjacent", weight=None, rep="matched", drug="class",
                       label="Mixture of all other HDACi pooled"),
    "vehicle_rep": dict(source="vehicle", rep="matched", plate="any", label="Vehicle, replicate-matched"),
    "vehicle_plate": dict(source="vehicle", rep="matched", plate="target", label="Vehicle, same plate as target"),
    "vehicle_pooled": dict(source="vehicle", rep="pooled", plate="any", label="Vehicle, any plate"),
}
N_TASKS = 1 + len(VARIANTS)
PROG_CHEMCPA = {"nbprog_v1": "chemCPA-NB, program-gene output",
                "nbprogwrf_l10_h0.5": "chemCPA-NB, program-gene output + WRF loss"}
MODELS = {  # (model, method) in results/analysis/tables/per_condition.csv -> label
    ("cellflow", "cellflow"): "CellFlow",
    ("cellflow", "cellflow_wrf"): "CellFlow + WRF loss, 3 τ",
    ("cellflow", "cellflow_wrf13"): "CellFlow + WRF loss, 13 τ",
    ("cellflow", "cellflow_mmd"): "CellFlow + MMD loss",
    ("chemcpa", "nb_v1"): "chemCPA-NB",
    ("chemcpa", "nbwrf_l10_h0.5"): "chemCPA-NB + WRF loss, 3 τ",
    ("chemcpa", "nbwrfgrid_l10_h0.25"): "chemCPA-NB + WRF loss, 13 τ",
    ("chemcpa", "gauss_v1"): "chemCPA-Gauss",
    **{("chemcpa_prog", k): v for k, v in PROG_CHEMCPA.items()},
}
CONTEXT = {("reference", "replicate"): "Replicate well vs other well",
           ("reference", "interpolation"): "Log-dose WRF interpolation (fractions only)"}


def dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    tmp.rename(path)


class Panel:
    """Panel expression, metadata and the frozen held-out conditions."""

    def __init__(self, path: str, spec: EvalSpec):
        self.path, self.spec = path, spec
        self.adata = ad.read_h5ad(path)
        assert self.adata.obs_names.is_unique, "provenance looks source cells up by obs name"
        self.X = dense(self.adata.X)
        obs = self.adata.obs
        self.obs = pd.DataFrame({k: obs[k].astype(str).to_numpy() for k in
                                 ("condition", "drug", "replicate", "split", "culture_plate")}, index=obs.index)
        self.obs["dose"] = obs["dose"].astype(float).to_numpy()
        self.split_cfg = json.loads((REPO / "configs" / "split_v1.json").read_text())
        self.meta = {c["condition"]: c for c in self.split_cfg["conditions"]}
        self.r = response_score(spec.align(self.X, self.adata.var_names), spec)
        self._pool = {}

    def heldout(self, split: str) -> list[dict]:
        return [c for c in self.split_cfg["conditions"] if c["split"] == split]

    def pool(self, **where) -> np.ndarray:
        """Positions of panel cells matching every column=value (None: any value)."""
        key = tuple(sorted(where.items()))
        if key not in self._pool:
            m = np.ones(len(self.obs), bool)
            for col, val in where.items():
                if val is not None:
                    m &= (self.obs[col] == val).to_numpy()
            self._pool[key] = np.flatnonzero(m)
        return self._pool[key]

    def adjacent(self, c: dict) -> tuple[float, float, float]:
        doses = np.sort(np.unique(self.obs.dose[(self.obs.drug == c["drug"]) & (self.obs.split == "train")]))
        target = float(c["dose_nM"])
        low, high = float(doses[doses < target].max()), float(doses[doses > target].min())
        upper = (np.log1p(target) - np.log1p(low)) / (np.log1p(high) - np.log1p(low))
        return low, high, float(upper)

    def target_plate(self, c: dict, rep: str) -> str:
        plates = set(self.obs.culture_plate[(self.obs.condition == c["condition"]) & (self.obs.replicate == rep)])
        assert len(plates) == 1, (c["condition"], rep, plates)
        return plates.pop()

    def other_drugs(self, c: dict, low: float, high: float) -> list[str]:
        train = self.obs[self.obs.split == "train"]
        doses = train.groupby("drug").dose.agg(set)
        return sorted(d for d, have in doses.items() if {low, high} <= have and d != c["drug"])

    def prediction(self, positions: list[np.ndarray], conditions: list[str], extra: list[dict]) -> ad.AnnData:
        idx = np.concatenate(positions)
        obs = pd.DataFrame([{"condition": c, **e} for c, e, p in zip(conditions, extra, positions) for _ in p])
        obs["source_cell"] = self.adata.obs_names[idx].to_numpy()
        obs.index = [f"pred_{i}" for i in range(len(obs))]
        return ad.AnnData(self.X[idx], obs=obs, var=pd.DataFrame(index=self.adata.var_names))


def score(P: Panel, pred: ad.AnnData, split: str, base: dict) -> list[dict]:
    """Frozen evaluator plus the 13-threshold grid (as in experiments/cellflow/source/evaluate.py)."""
    rows = []
    pc = pred.obs.condition.to_numpy()
    for rec in evaluate_predictions(P.spec, P.adata, pred, split=split, seed=0):
        c = rec["condition"]
        r_real = P.r[P.pool(condition=c)]
        r_pred = response_score(P.spec.align(pred.X[pc == c], pred.var_names), P.spec)
        err = 100 * np.abs(weak_responder_fraction(r_pred, GRID_TAUS) - weak_responder_fraction(r_real, GRID_TAUS))
        rows.append({**base, "split": split, **rec, "wrf_err_pp_grid13_mean": float(err.mean()),
                     "grid13_err_pp": err.tolist()})
    return rows


# ---------------------------------------------------------------------------------------------
def published(P: Panel, split: str):
    """CellFlow's nearest_dose and adjacent_dose_mixture, line for line (same RNG stream).

    Source: bde8c6c:experiments/cellflow_v1/source/run_cellflow_v1.py:422-446. `train` there
    is the panel's train + control cells in panel order, so pools here are identical arrays.
    """
    rng = np.random.default_rng(2026)
    out = {}
    for method in ["nearest_dose", "adjacent_dose_mixture"]:
        positions, conds, extra = [], [], []
        for c in P.heldout(split):
            in_train = P.obs.split.isin(["train", "control"])
            doses = np.sort(P.obs.dose[in_train & (P.obs.drug == c["drug"])].unique())
            target, n = c["dose_nM"], N_CELLS
            n1 = int(round(n * c["n_rep1"] / c["n_cells"]))
            if method == "nearest_dose":
                chosen = [float(doses[np.argmin(np.abs(np.log1p(doses) - np.log1p(target)))])]
                weights = [1.]
            else:
                low, high = float(doses[doses < target].max()), float(doses[doses > target].min())
                upper = (np.log1p(target) - np.log1p(low)) / (np.log1p(high) - np.log1p(low))
                chosen, weights = [low, high], [1 - upper, upper]
            for rep, take in [("rep1", n1), ("rep2", n - n1)]:
                dose_samples = rng.choice(chosen, take, p=weights)
                ids = [rng.choice(P.pool(drug=c["drug"], dose=float(d), replicate=rep, split="train"))
                       for d in dose_samples]
                positions.append(np.asarray(ids, dtype=np.int64))
                conds.append(c["condition"])
                extra.append({"replicate": rep})
        out[method] = P.prediction(positions, conds, extra)
    return out


def provenance(P: Panel, pred: ad.AnnData, split: str) -> dict:
    """Every source cell of an adjacent-dose prediction: allowed split, drug, dose, replicate."""
    src = P.obs.loc[pred.obs.source_cell.to_numpy()].reset_index(drop=True)
    tgt = pred.obs.reset_index(drop=True)
    heldout = set(P.obs.condition[P.obs.split.isin(["val", "test"])])
    checks = {"n_source_cells": int(len(src)),
              "all_split_train": bool((src.split == "train").all()),
              "no_heldout_condition": bool(not src.condition.isin(heldout).any()),
              "never_the_target_condition": bool((src.condition.to_numpy() != tgt.condition.to_numpy()).all()),
              "same_drug": bool((src.drug.to_numpy() == tgt.condition.str.split("|").str[1].to_numpy()).all()),
              "same_replicate": bool((src.replicate.to_numpy() == tgt.replicate.to_numpy()).all())}
    per = []
    for (cond, rep), part in tgt.groupby(["condition", "replicate"]):
        c = P.meta[cond]
        low, high, _ = P.adjacent(c)
        s = src.loc[part.index]
        plate = P.target_plate(c, rep)
        per.append({"condition": cond, "replicate": rep, "n": len(s), "target_plate": plate,
                    "doses_ok": bool(s.dose.isin([low, high]).all()), "frac_upper": float((s.dose == high).mean()),
                    "frac_on_target_plate": float((s.culture_plate == plate).mean()),
                    "dose_on_target_plate": sorted({float(d) for d in s.dose[s.culture_plate == plate]})})
    per = pd.DataFrame(per)
    checks.update({"all_doses_adjacent": bool(per.doses_ok.all()),
                   "frac_on_target_plate_mean": float(per.frac_on_target_plate.mean()),
                   "frac_on_target_plate_range": [float(per.frac_on_target_plate.min()),
                                                  float(per.frac_on_target_plate.max())],
                   "split": split, "per_well": per.to_dict("records")})
    return checks


def floor_rows(P: Panel, split: str, rng: np.random.Generator, n_mc: int = 20000) -> list[dict]:
    """WRF error of a perfect predictor that samples N_CELLS cells from the true population,
    scored against n_real measured cells: E|Bin(n_real, p)/n_real − Bin(N_CELLS, p)/N_CELLS|,
    with p plugged in as the measured fraction. Not a method: a finite-sample floor."""
    spec_taus = np.asarray(P.spec.taus, dtype=np.float64)
    rows = []
    for c in P.heldout(split):
        r = P.r[P.pool(condition=c["condition"])]
        n = len(r)

        def err(taus):
            p = weak_responder_fraction(r, taus)
            a = rng.binomial(n, p[None, :], (n_mc, len(p))) / n
            b = rng.binomial(N_CELLS, p[None, :], (n_mc, len(p))) / N_CELLS
            return 100 * np.abs(a - b).mean(0)

        e_spec, e_grid = err(spec_taus), err(GRID_TAUS)
        rows.append({"entity": "floor", "variant": "perfect_predictor", "sampling_seed": 0, "split": split,
                     "condition": c["condition"], "n_real": n,
                     **{f"wrf_err_pp_q{q:g}": float(e) for q, e in zip(P.spec.quantiles, e_spec)},
                     "wrf_err_pp_grid13_mean": float(e_grid.mean())})
    return rows


def chemcpa_prog_rows(P: Panel) -> list[dict]:
    """Program-output chemCPA runs (not in the analysis tables): frozen rows + the 13-τ grid."""
    rows = []
    for method in PROG_CHEMCPA:
        for path in sorted((CHEMCPA_RUNS / method).glob("seed_*/*_sample_s0/predictions.h5ad")):
            split = path.parent.name.split("_")[0]
            seed = int(path.parent.parent.name.split("_")[1])
            frozen = {r["condition"]: r for r in map(json.loads, (path.parent / "per_condition.jsonl").open())}
            pred = ad.read_h5ad(path)
            r_pred_all = response_score_from_program(P.spec.align_program(pred.X, pred.var_names), P.spec)
            pc = pred.obs.condition.astype(str).to_numpy()
            for c in P.heldout(split):
                cond = c["condition"]
                r_real = P.r[P.pool(condition=cond)]
                err = 100 * np.abs(weak_responder_fraction(r_pred_all[pc == cond], GRID_TAUS)
                                   - weak_responder_fraction(r_real, GRID_TAUS))
                f = frozen[cond]
                assert abs(f["wrf_err_pp_q0.95"] - 100 * abs(
                    weak_responder_fraction(r_pred_all[pc == cond], [P.spec.tau_by_q[0.95]])[0]
                    - weak_responder_fraction(r_real, [P.spec.tau_by_q[0.95]])[0])) < 1e-9, cond
                rows.append({"model": "chemcpa_prog", "method": method, "seed": seed, "split": split,
                             "condition": cond, **{k: f.get(k, np.nan) for k in METRICS},
                             "wrf_err_pp_grid13_mean": float(err.mean())})
    return rows


def sample_variant(P: Panel, name: str, split: str, seed: int) -> ad.AnnData:
    v = VARIANTS[name]
    rng = np.random.default_rng([zlib.crc32(name.encode()), seed, SPLITS.index(split)])
    positions, conds, extra = [], [], []
    for c in P.heldout(split):
        n1 = int(round(N_CELLS * c["n_rep1"] / c["n_cells"]))
        if v["rep"] == "pooled":
            plan = [(None, N_CELLS, None)]
        else:
            other = {"rep1": "rep2", "rep2": "rep1"}
            plan = [(rep, take, rep if v["rep"] == "matched" else other[rep])
                    for rep, take in [("rep1", n1), ("rep2", N_CELLS - n1)]]
        if v["source"] == "adjacent":
            low, high, upper = P.adjacent(c)
            w = upper if v["weight"] is None else v["weight"]
            if v["drug"] == "same":
                drugs = [c["drug"]]
            elif v["drug"] == "other":
                drugs = [str(rng.choice(P.other_drugs(c, low, high)))]
            else:
                drugs = P.other_drugs(c, low, high)
        for rep, take, src_rep in plan:
            if v["source"] == "adjacent":
                pools = [P.pool(drug=d, dose=dose, replicate=src_rep, split="train") for d in drugs for dose in (low, high)]
                weights = np.array([share for _ in drugs for share in (1 - w, w)]) / len(drugs)
            else:
                plate = P.target_plate(c, rep) if v["plate"] == "target" else None
                pools, weights = [P.pool(split="control", replicate=src_rep, culture_plate=plate)], np.array([1.0])
            keep = weights > 0
            pools, weights = [p for p, k in zip(pools, keep) if k], weights[keep] / weights[keep].sum()
            assert all(len(p) for p in pools), (name, c["condition"], rep)
            comp = rng.choice(len(pools), take, p=weights)
            ids = np.concatenate([rng.choice(pools[k], int((comp == k).sum())) for k in range(len(pools))])
            positions.append(ids.astype(np.int64))
            conds.append(c["condition"])
            extra.append({"replicate": rep or "pooled"})
    return P.prediction(positions, conds, extra)


def variant_config(name: str) -> str:
    return json.dumps({"variant": VARIANTS[name], "n_cells": N_CELLS}, sort_keys=True)


def build(task: int, data: str, root: Path, seeds: int) -> None:
    count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if count is not None and int(count) != N_TASKS:
        raise SystemExit(f"array has {count} tasks but the audit defines {N_TASKS}; fix --array in adjacent.sbatch")
    spec = EvalSpec.load(SPEC_DIR)
    P = Panel(data, spec)
    out = root / "adjacent_audit"
    if task == 0:
        panel_sha = sha256(data)
        expected = json.loads((CELLFLOW_DOCS / "artifacts.json").read_text())["panel_sha256"]
        frozen = pd.read_csv(CELLFLOW_DOCS / "per_condition.csv")
        repro, audit, rows = {"panel_sha256": panel_sha, "panel_matches_cellflow": panel_sha == expected}, [], []
        for split in SPLITS:
            for method, pred in published(P, split).items():
                scored = score(P, pred, split, {"entity": "published", "variant": method, "sampling_seed": 0})
                rows += scored
                ref = frozen[(frozen.method == method) & (frozen.split == split)].set_index("condition")
                mine = pd.DataFrame(scored).set_index("condition").loc[ref.index]
                cols = [k for k in METRICS + ["n_pred", "mean_r_pred"] if k in ref]
                diff = (mine[cols] - ref[cols]).abs().max()
                repro[f"{method}/{split}"] = {"max_abs_diff": diff.to_dict(), "max": float(diff.max()),
                                              "n_conditions": int(len(ref))}
                if method == "adjacent_dose_mixture":
                    audit.append(provenance(P, pred, split))
        repro["reproduced"] = all(v["max"] < 1e-6 for k, v in repro.items() if isinstance(v, dict))
        write_jsonl(out / "rows" / "published.jsonl", rows)
        rng = np.random.default_rng(7)
        write_jsonl(out / "rows" / "floor.jsonl", [r for s in SPLITS for r in floor_rows(P, s, rng)])
        write_jsonl(out / "model_rows" / "chemcpa_prog.jsonl", chemcpa_prog_rows(P))
        (out / "audit").mkdir(parents=True, exist_ok=True)
        (out / "audit" / "reproduction.json").write_text(json.dumps(repro, indent=1, default=float))
        (out / "audit" / "provenance.json").write_text(json.dumps(audit, indent=1, default=float))
        print(json.dumps({k: (v if not isinstance(v, dict) else v["max"]) for k, v in repro.items()}, default=float))
        for a in audit:
            print({k: v for k, v in a.items() if k != "per_well"})
        return
    name = list(VARIANTS)[task - 1]
    config = variant_config(name)
    for split in SPLITS:
        for seed in range(seeds):
            path = out / "rows" / name / f"{split}_s{seed}.jsonl"
            if path.exists():
                first = json.loads(path.open().readline())
                if first["config"] != config:
                    raise SystemExit(f"{path} was written under a different definition of {name}; move it aside")
                continue
            pred = sample_variant(P, name, split, seed)
            rows = score(P, pred, split, {"entity": "ablation", "variant": name, "sampling_seed": seed,
                                          "config": config})
            write_jsonl(path, rows)
        print(f"{name} {split}: {seeds} sampling seeds", flush=True)


# ---------------------------------------------------------------------------------------------
def load_entities(root: Path) -> pd.DataFrame:
    """One frame: (kind, name, seed, split, condition, metrics) for ablations and models."""
    out = root / "adjacent_audit"
    frames = []
    rows = [json.loads(line) for p in sorted((out / "rows").rglob("*.jsonl")) for line in p.open()]
    a = pd.DataFrame(rows)
    a["kind"] = np.where(a.entity == "ablation", "ablation", a.entity)
    a["name"] = a.variant
    a["seed"] = a.sampling_seed
    frames.append(a)
    pc = pd.read_csv(root / "tables" / "per_condition.csv")
    prog = pd.DataFrame([json.loads(line) for line in (out / "model_rows" / "chemcpa_prog.jsonl").open()])
    pc = pd.concat([pc, prog], ignore_index=True)
    for keys, kind in ((MODELS, "model"), (CONTEXT, "context")):
        for (model, method) in keys:
            part = pc[(pc.model == model) & (pc.method == method)].copy()
            if part.empty:
                raise SystemExit(f"no rows for {model}/{method} in the analysis tables")
            part["kind"], part["name"] = kind, f"{model}/{method}"
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    return df[["kind", "name", "seed", "split", "condition"] + METRICS]


def label(name: str) -> str:
    if name in VARIANTS:
        return VARIANTS[name]["label"]
    model, _, method = name.partition("/")
    if (model, method) in MODELS:
        return MODELS[(model, method)]
    if (model, method) in CONTEXT:
        return CONTEXT[(model, method)]
    return {"adjacent_dose_mixture": "Adjacent-dose mixture (published draw, reproduced)",
            "nearest_dose": "Nearest dose (published draw, reproduced)",
            "perfect_predictor": "Perfect predictor, finite-sample floor (WRF only)"}.get(name, name)


def matrix(df: pd.DataFrame, name: str, split: str, metric: str) -> pd.DataFrame | None:
    part = df[(df.name == name) & (df.split == split)]
    if part.empty or part[metric].isna().all():
        return None
    return part.pivot_table(index="condition", columns="seed", values=metric, aggfunc="mean")


def compare(root: Path, n_boot: int) -> None:
    from .analyze import paired_stats

    out = root / "adjacent_audit"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = load_entities(root)
    missing = [n for n in VARIANTS if n not in set(df.name)]
    if missing:
        raise SystemExit(f"ablations missing rows (did their build tasks run?): {missing}")
    rng = np.random.default_rng(0)

    summary = []
    for (kind, name, split), part in df.groupby(["kind", "name", "split"]):
        per_seed = part.groupby("seed")[METRICS].mean()
        cond_mean = part.groupby("condition")[METRICS].mean()
        for m in METRICS:
            if cond_mean[m].isna().all():
                continue
            summary.append({"kind": kind, "name": name, "label": label(name), "split": split, "metric": m,
                            "mean": cond_mean[m].mean(), "seed_sd": per_seed[m].std(ddof=1) if len(per_seed) > 1
                            else np.nan, "n_seeds": len(per_seed), "n_conditions": len(cond_mean)})
    summary = pd.DataFrame(summary)
    summary.to_csv(tables / "summary.csv", index=False)

    paired = []
    models = sorted(df.name[df.kind == "model"].unique())
    contrasts = [(m, v) for m in models for v in [*VARIANTS, "adjacent_dose_mixture"]]
    # Each ingredient on its own: ref = mixture with the ingredient removed, new = full mixture.
    contrasts += [(v, "adjacent") for v in VARIANTS if v != "adjacent"]
    contrasts += [("vehicle_rep", "vehicle_plate"), ("vehicle_pooled", "vehicle_rep")]
    for split in SPLITS:
        for metric in METRICS:
            for ref, new in contrasts:
                A, B = matrix(df, ref, split, metric), matrix(df, new, split, metric)
                if A is None or B is None:
                    continue
                paired.append({"split": split, "metric": metric, "ref": ref, "new": new, "ref_label": label(ref),
                               "new_label": label(new), **paired_stats(A, B, rng, n_boot)})
    paired = pd.DataFrame(paired)
    paired.to_csv(tables / "paired.csv", index=False)
    (out / "report.md").write_text(report(out, summary, paired))
    print(f"wrote {tables}/summary.csv, {tables}/paired.csv, {out}/report.md")


def report(out: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    repro = json.loads((out / "audit" / "reproduction.json").read_text())
    prov = json.loads((out / "audit" / "provenance.json").read_text())

    def cell(name, split, metric):
        r = summary[(summary.name == name) & (summary.split == split) & (summary.metric == metric)]
        if r.empty:
            return "–"
        r = r.iloc[0]
        fmt = "{:.4f}" if metric == "mmd2" else "{:.3f}" if metric in ("w1_r", "rmse_mean_deg50") else "{:.2f}"
        s = fmt.format(r["mean"])
        return s + (f" ± {fmt.format(r.seed_sd)}" if r.n_seeds > 1 and np.isfinite(r.seed_sd) else "")

    def table(split, names):
        head = "| | " + " | ".join(METRIC_LABELS[m] for m in REPORT_METRICS) + " |\n"
        head += "|---" * (len(REPORT_METRICS) + 1) + "|\n"
        return head + "".join(f"| {label(n)} | " + " | ".join(cell(n, split, m) for m in REPORT_METRICS) + " |\n"
                              for n in names)

    def ptable(split, pairs):
        p = paired[(paired.split == split)]
        lines = "| ref → new | metric | Δ (ref − new) | 95% CI | new better |\n|---|---|---|---|---|\n"
        for ref, new in pairs:
            for m in REPORT_METRICS:
                r = p[(p.ref == ref) & (p.new == new) & (p.metric == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines += (f"| {label(ref)} → {label(new)} | {METRIC_LABELS[m]} | {r.delta_mean:+.4g} | "
                          f"[{r.ci95_low:+.4g}, {r.ci95_high:+.4g}] | {r.n_win}/{r.n_conditions} |\n")
        return lines

    models = [f"{k[0]}/{k[1]}" for k in MODELS]
    abl = list(VARIANTS)
    ok = all(a["all_split_train"] and a["no_heldout_condition"] and a["same_drug"] and a["all_doses_adjacent"]
             and a["same_replicate"] for a in prov)
    text = ["# Adjacent-dose mixture audit", "",
            "Generated by `python -m sciplex_analysis.adjacent_audit compare`. Δ = ref − new, so a positive Δ "
            "means `new` has the lower error. The CI is a paired condition bootstrap. Ablations average "
            f"{int(summary[summary.kind == 'ablation'].n_seeds.max())} sampling seeds, and ± is their SD. "
            "Models average 3 training seeds.", "",
            "## Reproduction and provenance", "",
            f"- Panel sha256 matches CellFlow's: **{repro['panel_matches_cellflow']}**. Published reference "
            f"reproduced exactly (max abs diff < 1e-6 on every metric): **{repro['reproduced']}**.",
            *[f"  - `{k}`: max abs diff {v['max']:.2e} over {v['n_conditions']} conditions"
              for k, v in repro.items() if isinstance(v, dict)],
            f"- Every source cell is a train cell of the same drug at an adjacent training dose and the "
            f"same replicate; no held-out cell is used: **{ok}**.",
            *[f"  - {a['split']}: {a['n_source_cells']} cells, {100 * a['frac_on_target_plate_mean']:.0f}% on the "
              f"held-out well's own culture plate (range {100 * a['frac_on_target_plate_range'][0]:.0f}–"
              f"{100 * a['frac_on_target_plate_range'][1]:.0f}%)" for a in prov], ""]
    for split in ("test", "val"):
        text += [f"## {split}", "", table(split, models + ["adjacent_dose_mixture", *abl]
                                          + [f"{k[0]}/{k[1]}" for k in CONTEXT] + ["perfect_predictor"]), "",
                 "### Each ingredient removed from the mixture", "",
                 ptable(split, [(v, "adjacent") for v in abl if v != "adjacent"]
                        + [("vehicle_rep", "vehicle_plate"), ("vehicle_pooled", "vehicle_rep")]), "",
                 "### Ablations against each model", "",
                 ptable(split, [(m, v) for m in models for v in ("adjacent", "adjacent_cross_rep", "other_drug",
                                                                  "class_pool", "vehicle_rep")]), ""]
    return "\n".join(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--task", type=int, required=True, help=f"0 … {N_TASKS - 1}")
    b.add_argument("--data", default=DEFAULT_PANEL)
    b.add_argument("--root", default=str(DEFAULT_ROOT))
    b.add_argument("--seeds", type=int, default=20, help="sampling seeds per ablation")
    c = sub.add_parser("compare")
    c.add_argument("--root", default=str(DEFAULT_ROOT))
    c.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)
    if a.cmd == "build":
        if not 0 <= a.task < N_TASKS:
            raise SystemExit(f"--task must be in 0 … {N_TASKS - 1}")
        build(a.task, a.data, Path(a.root), a.seeds)
    else:
        compare(Path(a.root), a.n_boot)


if __name__ == "__main__":
    main()
