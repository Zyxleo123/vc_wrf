"""CellFlow for the dose-interpolation study: the published model, plus opt-in population terms.

What defines the published CellFlow baseline is reproduced from
experiments/cellflow/source/run_cellflow_v1.py at commit 2c478b8 (cellflow-tools 0.0.9): all 2,085
genes in log1p(CP10K) without PCA, one-hot drug + log1p(dose)/log1p(1e4), controls matched within
replicate, network sizes, OT coupling, optimizer and schedule, and generation (512 cells per
condition from permitted controls, replicate-matched, 64 Heun steps, negatives clipped).

New, from configs/interp/cellflow_<variant>.json (absent keys reproduce the published model):

  aux.kind "mean"           population-mean consistency. Every `every` updates from `start` on, the
                            update's first `batch` control cells are pushed through its training
                            condition by an `ode_steps`-step Heun rollout with gradients through
                            every velocity evaluation (as in the published WRF and MMD terms),
                            negatives are clipped as at prediction, and
                            weight × every × mean_genes (generated mean − measured mean)² is added.
                            The measured mean is over all cells of that condition and replicate.
                            The rollout state is kept in [BOUND_LOW, BOUND_HIGH] (an unstable field
                            otherwise overflows float32, the published `nonfinite_training_loss`), and
                            an update whose term or gradient is still non-finite drops the term.
  aux.kind "mean_velocity"  rollout-free: weight × mean_genes (batch mean of the predicted velocity
                            − batch mean of the target velocity)², every update: flow matching's
                            own population-mean component, up-weighted.
  conditions_per_batch      K training targets per update instead of the published one, each with
                            `batch_size` cells and its own OT coupling, so the condition varies within
                            an update. No K = 8 run ended ignoring the condition, but the cause of that
                            failure is not isolated (docs/interp/log.md).
  latent                    run the flow on coordinates in a fixed orthonormal basis instead of on the
                            genes: `dims` spec PCA components (fit on control + train cells) plus
                            `mean_dirs` principal directions of the training mean shifts outside them.
                            decode "inverse": cell = centre + z1 U. decode "residual": cell = x0 + (z1 − z0) U,
                            so a generated cell is its source control cell moved by the flow's
                            displacement, and keeps that cell's gene-level detail outside the basis.
  covariates                which condition inputs the encoder sees, from "drug" (one-hot), "dose" and
                            "teacher"; the published model uses the first two. "teacher" is the
                            mean-level dose-response prediction for the condition (sciplex_interp.teacher:
                            fit on training means, predicted from labels), so at a held-out dose the
                            flow is told the predicted mean shift.

Checkpoints are scored at `validation_steps` on validation (9 held-out conditions) and on the
in-sample fit (78 training conditions generated from controls); the one with the lowest mean
validation MMD² is kept (the chemCPA rule). Only training and control cells enter training.

usage (in the CellFlow environment, on a GPU):
  python -m sciplex_interp.cellflow train --variant base --seed 0
  python -m sciplex_interp.cellflow evaluate --variant base --seed 0 --split test --sampling-seed 0
Writes <root>/cellflow/<variant>/seed_<seed>/: config.json, history.jsonl (validation step x task x
condition), training_loss.csv, best_params.msgpack, selected.json, training_complete.json, and
<split>_sample_s<offset>/{preds.h5ad, per_condition.jsonl, summary.json} (the frozen evaluator; the
layout of sciplex_baselines.chemcpa evaluate, so chemcpa_compare-style tables can read it).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, dense, score, write_jsonl

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "interp"
N_CELLS = 512          # generated cells per condition, as published
ODE_STEPS = 64         # Heun steps at generation, as published
LOG_DOSE_MAX = np.log1p(10000)
BOUND_LOW, BOUND_HIGH = -2.0, 12.0   # rollout state; measured log1p(CP10K) is at most log1p(1e4) = 9.2
EXPLODED = 12.0                      # a generated cell with a value above this has left the data range


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_dir(root, variant: str, seed: int) -> Path:
    return Path(root) / "cellflow" / variant / f"seed_{seed}"


def read_config(variant: str, path: str | None = None) -> tuple[dict, Path]:
    path = Path(path) if path else CONFIGS / f"cellflow_{variant}.json"
    if not path.is_file():
        raise SystemExit(f"missing config {path}")
    return json.loads(path.read_text()), path


# ---- data and model, as published ---------------------------------------------------------------
def latent_basis(data: Data, latent: dict) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal rows U (k x genes) and the centre, from control + training information only."""
    U = np.asarray(data.spec.pca_components, dtype=np.float64)[: int(latent.get("dims", 50))]
    extra = int(latent.get("mean_dirs", 0))
    if extra:
        train = data.cond[data.cond.split == "train"].index
        shifts = np.array([data.mean(c) - data.ctrl_mean for c in train])
        V = np.linalg.svd(shifts, full_matrices=False)[2][:extra]
        V = V - (V @ U.T) @ U  # the part of each direction outside the PCA
        U = np.vstack([U, np.linalg.qr(V.T)[0].T])
    return U.astype(np.float32), np.asarray(data.spec.pca_mean, dtype=np.float32)


def decode(z1: np.ndarray, z0: np.ndarray, x0: np.ndarray, U: np.ndarray, centre: np.ndarray, kind: str) -> np.ndarray:
    """Gene-space cells from flow coordinates z1 that started at z0 (source cells x0)."""
    if kind == "inverse":
        return centre + z1 @ U
    if kind == "residual":
        return x0 + (z1 - z0) @ U
    raise SystemExit(f"unknown latent.decode {kind}")


def teacher_key(drug: str, dose: float) -> str:
    return f"{drug}|{float(dose):g}"


def model_data(data: Data, cfg: dict | None = None, root=None):
    """Training + control cells with the published covariates (no held-out expression)."""
    a = data.adata[data.adata.obs["split"].isin(["train", "control"])].copy()
    a.obs["is_control"] = a.obs["split"].eq("control")
    a.obs["log_dose"] = (np.log1p(a.obs["dose"].astype(float)) / LOG_DOSE_MAX).astype(np.float32)
    a.obs["model_condition"] = a.obs["condition"].astype(str) + "|" + a.obs["replicate"].astype(str)
    a.X = dense(a.X)
    drugs = sorted(a.obs.loc[~a.obs["is_control"], "drug"].astype(str).unique())
    a.uns["drug_embeddings"] = {d: np.eye(len(drugs), dtype=np.float32)[i] for i, d in enumerate(drugs)}
    a.uns["drug_embeddings"]["Vehicle"] = np.zeros(len(drugs), dtype=np.float32)
    if (cfg or {}).get("latent"):
        U, centre = latent_basis(data, cfg["latent"])
        a.obsm["X_latent"] = ((a.X - centre) @ U.T).astype(np.float32)
        a.uns["latent"] = {"basis": U, "centre": centre, "decode": cfg["latent"].get("decode", "inverse")}
    if "teacher" in (cfg or {}).get("covariates", ()):
        z = np.load(Path(root) / "teacher" / f"{cfg.get('teacher', 'autodec')}.npz")
        points = [json.loads(k) for k in z["keys"]]
        table = {teacher_key(d, x): (c / z["scale"]).astype(np.float32) for (d, x), c in zip(points, z["coef"])}
        table[teacher_key("Vehicle", 0.0)] = np.zeros(z["coef"].shape[1], dtype=np.float32)
        a.uns["teacher_embeddings"] = table
        a.obs["teacher_key"] = [teacher_key(d, x) for d, x in zip(a.obs["drug"].astype(str), a.obs["dose"].astype(float))]
    return a


def build(train, cfg: dict, seed: int):
    import optax
    from cellflow.model import CellFlow
    from cellflow.utils import match_linear

    cf = CellFlow(train, solver="otfm")
    known = {"drug": ("drug",), "dose": ("log_dose",), "teacher": ("teacher_key",)}
    covariates = {k: known[k] for k in cfg.get("covariates", ("drug", "dose"))}
    reps = {k: v for k, v in {"drug": "drug_embeddings", "teacher": "teacher_embeddings"}.items() if k in covariates}
    cf.prepare_data(sample_rep="X_latent" if cfg.get("latent") else "X", control_key="is_control",
                    perturbation_covariates=covariates, perturbation_covariate_reps=reps,
                    split_covariates=["replicate"], max_combination_length=1)
    schedule = optax.warmup_cosine_decay_schedule(init_value=1e-5, peak_value=cfg["learning_rate_peak"],
                                                  warmup_steps=cfg["warmup_steps"], decay_steps=cfg["training_steps"],
                                                  end_value=cfg["learning_rate_end"])
    optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adamw(schedule, weight_decay=cfg["weight_decay"]))
    mlp = lambda dims: {"layer_type": "mlp", "dims": list(dims), "dropout_rate": 0.0}
    cf.prepare_model(condition_mode="deterministic", pooling="mean", regularization=0.0,
                     layers_before_pool={k: mlp(cfg["condition_mlp_dims"]) for k in covariates},
                     layers_after_pool=mlp([128]), condition_embedding_dim=cfg["condition_embedding_dim"],
                     cond_output_dropout=cfg["condition_output_dropout"], time_freqs=cfg["time_freqs"],
                     time_encoder_dims=tuple(cfg["time_encoder_dims"]), hidden_dims=tuple(cfg["hidden_dims"]),
                     decoder_dims=tuple(cfg["decoder_dims"]), probability_path={"constant_noise": cfg["constant_noise"]},
                     match_fn=functools.partial(match_linear, epsilon=cfg["ot_epsilon"], tau_a=1.0, tau_b=1.0),
                     optimizer=optimizer, solver_kwargs={"ema": cfg["ema"]}, seed=seed)
    return cf, schedule


def make_sampler(cf, batch_size: int):
    """The published TrainSampler (same random stream), also reporting the target's index."""
    from cellflow.data._dataloader import TrainSampler

    class IndexedSampler(TrainSampler):
        def sample(self, rng):
            source_dist = rng.integers(0, self.n_source_dists)
            source = self._sample_from_mask(rng, self._data.split_covariates_mask == source_dist)
            target_dist = self._sample_target_dist_idx(source_dist, rng)
            target = self._sample_from_mask(rng, self._data.perturbation_covariates_mask == target_dist)
            return {"src_cell_data": self._data.cell_data[source], "tgt_cell_data": self._data.cell_data[target],
                    "condition": self._get_embeddings(target_dist, self._data.condition_data),
                    "target_idx": int(target_dist)}

        def sample_many(self, rng, k: int):
            """`k` single-condition batches, stacked: cells (k, n, genes), conditions (k, 1, dim)."""
            parts = [self.sample(rng) for _ in range(k)]
            return {"src_cell_data": np.stack([b["src_cell_data"] for b in parts]),
                    "tgt_cell_data": np.stack([b["tgt_cell_data"] for b in parts]),
                    "condition": {c: np.concatenate([b["condition"][c] for b in parts]) for c in parts[0]["condition"]},
                    "target_idx": np.array([b["target_idx"] for b in parts])}

    return IndexedSampler(data=cf.train_data, batch_size=batch_size)


def target_means(cf) -> np.ndarray:
    """Measured mean of every training target (condition x replicate), in target-index order."""
    d = cf.train_data
    means = [np.asarray(d.cell_data[d.perturbation_covariates_mask == i]).mean(0) for i in range(d.n_perturbations)]
    return np.stack(means).astype(np.float32)


def make_step(solver, aux: dict | None, means: np.ndarray, n_conditions: int = 1):
    """One optimizer update of flow matching over `n_conditions` targets (+ `aux`).

    Batches are stacked, (K, n, genes). With K = 1 and no term this is OTFlowMatching.step_fn.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from cellflow.solvers.utils import ema_update
    from ott.solvers import utils as solver_utils

    aux = aux or {"kind": "none", "weight": 0.0}
    kind, weight = aux["kind"], float(aux["weight"])
    if kind not in ("none", "mean", "mean_velocity"):
        raise SystemExit(f"unknown aux.kind {kind}")
    every, begin = int(aux.get("every", 1)), int(aux.get("start", 0))
    n_roll, n_steps = int(aux.get("batch", 128)), int(aux.get("ode_steps", 8))
    cond_dim = solver.vf.condition_embedding_dim
    means = jnp.asarray(means)
    K = n_conditions

    def rollout(apply_fn, params, x, condition):
        dt = 1.0 / n_steps
        zeros = jnp.zeros((x.shape[0], cond_dim), x.dtype)

        def velocity(t, y):
            return apply_fn({"params": params}, jnp.full((y.shape[0], 1), t, y.dtype), y, condition, zeros,
                            train=False)[0]

        def body(y, i):
            t = i * dt
            k1 = velocity(t, y)
            y_next = y + 0.5 * dt * (k1 + velocity(t + dt, jnp.clip(y + dt * k1, BOUND_LOW, BOUND_HIGH)))
            return jnp.clip(y_next, BOUND_LOW, BOUND_HIGH), None

        return jax.lax.scan(jax.checkpoint(body), x, jnp.arange(n_steps))[0]

    @jax.jit
    def update(rng, state, time, source, target, condition, noise, measured):
        """source, target: (K, n, genes); condition: (K, 1, dim) per covariate; measured: (K, genes)."""
        active = (state.step >= begin) & (jnp.mod(state.step + 1, every) == 0)
        n = source.shape[1]
        flat = lambda a: a.reshape(K * n, a.shape[-1])
        per_cell = {c: jnp.repeat(v, n, axis=0) for c, v in condition.items()} if K > 1 else condition

        def flow(params):
            rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
            xt = solver.probability_path.compute_xt(rng_flow, time, flat(source), flat(target))
            vt, _, _ = state.apply_fn({"params": params}, time, xt, per_cell, encoder_noise=noise,
                                      rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder})
            ut = solver.probability_path.compute_ut(time, xt, flat(source), flat(target))
            fm = jnp.mean((vt - ut) ** 2)
            if kind == "mean_velocity":  # the population-mean component of each condition's velocity
                residual = (vt - ut).reshape(K, n, -1).mean(1)
                term = jnp.mean(residual ** 2)
                return fm + jnp.where(active, weight * every * term, 0.0), (fm, term)
            return fm, (fm, jnp.zeros((), source.dtype))

        def population(params):
            m = min(n_roll, n)
            cells = {c: jnp.repeat(v, m, axis=0) for c, v in condition.items()} if K > 1 else condition
            generated = rollout(state.apply_fn, params, source[:, :m].reshape(K * m, -1), cells)
            generated = jnp.maximum(generated, 0.0).reshape(K, m, -1)
            return jnp.mean((generated.mean(1) - measured) ** 2)

        (total, (fm, term)), grads = jax.value_and_grad(flow, has_aux=True)(state.params)
        if kind == "mean":
            def with_term(_):
                value, g = jax.value_and_grad(population)(state.params)
                ok = jnp.isfinite(value) & jnp.isfinite(optax.global_norm(g))
                return jnp.where(ok, value, 0.0), jax.tree.map(lambda a: jnp.where(ok, a, 0.0), g), ok

            def without(_):
                return jnp.zeros((), source.dtype), jax.tree.map(jnp.zeros_like, state.params), jnp.array(False)

            term, extra, active = jax.lax.cond(active, with_term, without, None)
            grads = jax.tree.map(lambda a, b: a + weight * every * b, grads, extra)
            total = fm + weight * every * term
        return state.apply_gradients(grads=grads), (total, fm, term, active)

    def resample(key, plan, source, target):
        si, ti = solver_utils.sample_joint(key, plan)
        return source[si], target[ti]

    def step(rng, batch):
        source, target = batch["src_cell_data"], batch["tgt_cell_data"]
        if K == 1:  # sample() batches: add the condition axis, keeping the published random stream
            source, target = source[None], target[None]
            indices = jnp.asarray([batch["target_idx"]])
        else:
            indices = jnp.asarray(batch["target_idx"])
        rng_resample, rng_time, rng_step, rng_noise = jax.random.split(rng, 4)
        size = K * source.shape[1]
        time = solver.time_sampler(rng_time, size)
        noise = jax.random.normal(rng_noise, (size, cond_dim))
        if K == 1:
            paired = resample(rng_resample, solver.match_fn(source[0], target[0]), source[0], target[0])
            paired = paired[0][None], paired[1][None]
        else:
            plans = jax.vmap(solver.match_fn)(source, target)
            paired = jax.vmap(resample)(jax.random.split(rng_resample, K), plans, source, target)
        solver.vf_state, metrics = update(rng_step, solver.vf_state, time, paired[0], paired[1], batch["condition"],
                                          noise, means[indices])
        if solver.ema == 1.0:
            solver.vf_state_inference = solver.vf_state
        else:
            solver.vf_state_inference = solver.vf_state_inference.replace(
                params=ema_update(solver.vf_state_inference.params, solver.vf_state.params, solver.ema))
        return metrics

    return step


# ---- generation and scoring ---------------------------------------------------------------------
def conditions(data: Data, split: str) -> list[dict]:
    return [{"condition": c, "drug": r.drug, "dose_nM": float(r.dose_nM), "n_cells": int(r.n_cells),
             "n_rep1": int(r.n_rep1)} for c, r in data.cond[data.cond.split == split].iterrows()]


def prediction_controls(train, seed: int, offset: int, n: int = N_CELLS):
    """Permitted controls, `n` per replicate; offset 0 is the published draw."""
    rng = np.random.default_rng(10000 + seed + 1000 * offset)
    ids = []
    for rep in ("rep1", "rep2"):
        pool = train.obs.index[train.obs["is_control"] & train.obs["replicate"].eq(rep)].to_numpy()
        ids.extend(rng.choice(pool, n, replace=False))
    return train[ids].copy()


def covariate_table(train, conds: list[dict]) -> pd.DataFrame:
    """Prediction targets from labels only, one row per condition x replicate."""
    rows = []
    for c in conds:
        for rep in ("rep1", "rep2"):
            row = train.obs[(train.obs.drug == c["drug"]) & (train.obs.replicate == rep)].iloc[0].copy()
            row["condition"] = c["condition"]
            row["model_condition"] = f"{c['condition']}|{rep}"
            row["dose"] = c["dose_nM"]
            row["log_dose"] = np.float32(np.log1p(c["dose_nM"]) / LOG_DOSE_MAX)
            row["teacher_key"] = teacher_key(c["drug"], c["dose_nM"])
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def generate(cf, train, conds: list[dict], seed: int, offset: int = 0, n: int = N_CELLS):
    """{condition: generated cells} in the target's replicate proportion, and the clipped fraction."""
    import diffrax

    source = prediction_controls(train, seed, offset, n)
    out = cf.predict(source, covariate_table(train, conds), condition_id_key="model_condition", solver=diffrax.Heun(),
                     stepsize_controller=diffrax.ConstantStepSize(), dt0=1 / ODE_STEPS, max_steps=2 * ODE_STEPS)
    latent = train.uns.get("latent")
    cells, negative, total = {}, 0, 0
    for c in conds:
        n1 = int(round(n * c["n_rep1"] / c["n_cells"]))
        parts = []
        for rep, take in (("rep1", n1), ("rep2", n - n1)):
            raw = np.asarray(out[f"{c['condition']}|{rep}"], dtype=np.float32)[:take]
            if not np.isfinite(raw).all():
                raise FloatingPointError(f"non-finite cells for {c['condition']}|{rep}")
            if latent is not None:  # the flow's output rows follow the replicate's source rows
                rows = np.flatnonzero(source.obs["replicate"].eq(rep).to_numpy())[:take]
                raw = decode(raw, source.obsm["X_latent"][rows], np.asarray(source.X[rows]), latent["basis"],
                             latent["centre"], latent["decode"])
            negative += int((raw < 0).sum())
            total += raw.size
            parts.append(np.maximum(raw, 0))
        cells[c["condition"]] = np.vstack(parts)
    return cells, negative / total


def shape_diagnostics(data: Data, X_real: np.ndarray, X_pred: np.ndarray) -> dict:
    """How much of the measured mean change is generated, and how much cell-to-cell variance."""
    m0 = data.spec.control_mean
    d_real = X_real.mean(0).astype(np.float64) - m0
    d_pred = X_pred.mean(0).astype(np.float64) - m0
    deg = np.argsort(-np.abs(d_real), kind="stable")[: data.spec.deg_top_k]
    return {"effect_ratio_deg50": float(d_pred[deg] @ d_real[deg] / (d_real[deg] @ d_real[deg])),
            "effect_ratio_all": float(d_pred @ d_real / (d_real @ d_real)),
            "effect_real_all": float(np.sqrt(np.mean(d_real ** 2))),
            "effect_pred_all": float(np.sqrt(np.mean(d_pred ** 2))),
            "var_ratio": float(X_pred.var(0).sum() / X_real.var(0).sum()),
            "exploded_cells": float((X_pred.max(1) > EXPLODED).mean()), "pred_max": float(X_pred.max()),
            "median_rmse_deg50": float(np.sqrt(np.mean((np.median(X_pred, 0)[deg] - np.median(X_real, 0)[deg]) ** 2))),
            "zero_frac_real": float((X_real == 0).mean()), "zero_frac_pred": float((X_pred == 0).mean())}


def score_rows(data: Data, cells: dict, conds: list[dict], task: str) -> list[dict]:
    rows = []
    for c in conds:
        X_real = data.X[data.idx(c["condition"])]
        rows.append({"task": task, "condition": c["condition"], "drug": c["drug"], "dose": c["dose_nM"],
                     **score(data, X_real, cells[c["condition"]], seed=0),
                     **shape_diagnostics(data, X_real, cells[c["condition"]])})
    return rows


def mean_of(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def save_params(cf, path: Path) -> None:
    from flax import serialization
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(serialization.to_bytes(cf.solver.vf_state_inference.params))
    tmp.rename(path)


def load_params(cf, path: Path) -> None:
    from flax import serialization
    params = serialization.from_bytes(cf.solver.vf_state_inference.params, Path(path).read_bytes())
    cf.solver.vf_state_inference = cf.solver.vf_state_inference.replace(params=params)
    cf.solver.is_trained = True


def cells_to_adata(data: Data, cells: dict):
    import anndata as ad
    labels = np.concatenate([[c] * len(x) for c, x in cells.items()])
    obs = pd.DataFrame({"condition": labels}, index=[f"generated_{i}" for i in range(len(labels))])
    return ad.AnnData(np.vstack(list(cells.values())), obs=obs,
                      var=pd.DataFrame(index=list(data.spec.gene_ids)))


# ---- stages -------------------------------------------------------------------------------------
def train(variant: str, seed: int, root: str, panel: str, config_path: str | None = None) -> None:
    import jax

    cfg, path = read_config(variant, config_path)
    out = run_dir(root, variant, seed)
    digest = sha256(path)
    done = out / "training_complete.json"
    if done.exists():
        if json.loads(done.read_text())["config_sha256"] != digest:
            raise SystemExit(f"{out} was trained with a different config; move it aside to retrain")
        print(f"{variant} seed {seed}: already trained", flush=True)
        return
    if out.exists():
        shutil.rmtree(out)  # a partial run: a seed takes minutes, so it restarts
    out.mkdir(parents=True)
    shutil.copyfile(path, out / "config.json")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit(f"no GPU visible to JAX: {jax.devices()}")
    print("devices", jax.devices(), flush=True)

    data = Data(panel)
    train_ad = model_data(data, cfg, root)
    assert train_ad.n_obs == 19398 and train_ad.n_vars == 2085, train_ad.shape
    assert list(train_ad.var_names) == list(data.spec.gene_ids)
    if cfg.get("latent"):
        U = train_ad.uns["latent"]["basis"]
        assert np.abs(U @ U.T - np.eye(len(U))).max() < 1e-4
    cf, schedule = build(train_ad, cfg, seed)
    sampler = make_sampler(cf, cfg["batch_size"])
    aux, n_conditions = cfg.get("aux"), int(cfg.get("conditions_per_batch", 1))
    custom = aux is not None or n_conditions > 1
    step_fn = make_step(cf.solver, aux, target_means(cf), n_conditions) if custom else None
    val_conds, fit_conds = conditions(data, "val"), conditions(data, "train")
    assert len(val_conds) == 9 and len(fit_conds) == 78
    steps = int(cfg["training_steps"])
    checks = set(cfg["validation_steps"])
    rng_np, rng_jax = np.random.default_rng(seed), jax.random.PRNGKey(seed)
    best, losses, history, start = float("inf"), [], [], time.monotonic()
    rates = np.asarray(schedule(np.arange(steps)))
    with open(out / "training_loss.csv", "w", buffering=1) as log:
        log.write("step,total_loss,flow_matching_loss,aux_loss,aux_active,learning_rate,wall_seconds\n")
        for step in range(1, steps + 1):
            rng_jax, step_rng = jax.random.split(rng_jax)
            batch = sampler.sample(rng_np) if n_conditions == 1 else sampler.sample_many(rng_np, n_conditions)
            if step_fn is None:
                total = float(cf.solver.step_fn(step_rng, batch))
                fm, term, active = total, 0.0, False
            else:
                total, fm, term, active = (float(v) for v in step_fn(step_rng, batch))
            if not np.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: fm {fm} aux {term}")
            losses.append(fm)
            log.write(f"{step},{total},{fm},{term},{int(active)},{rates[step - 1]},{time.monotonic() - start}\n")
            if step == 1 or step % 1000 == 0:
                print(f"{variant} seed {seed} step {step} fm {np.mean(losses[-1000:]):.5f} aux {term:.6f} "
                      f"seconds {time.monotonic() - start:.0f}", flush=True)
            if step not in checks:
                continue
            cf.solver.is_trained = True
            try:
                cells, negative = generate(cf, train_ad, val_conds + fit_conds, seed)
            except FloatingPointError as error:
                print(f"VALIDATION {variant} seed {seed} step {step} skipped: {error}", flush=True)
                continue
            val = score_rows(data, cells, val_conds, "val")
            fit = score_rows(data, cells, fit_conds, "train_fit")
            history += [{"step": step, "seed": seed, "variant": variant, **r} for r in val + fit]
            write_jsonl(out / "history.jsonl", history)
            mmd = mean_of(val, "mmd2")
            selected = mmd < best
            print(f"VALIDATION {variant} seed {seed} step {step} val mmd2 {mmd:.5f} deg50 "
                  f"{mean_of(val, 'rmse_mean_deg50'):.4f} effect {mean_of(val, 'effect_ratio_deg50'):.3f} | train_fit "
                  f"mmd2 {mean_of(fit, 'mmd2'):.5f} deg50 {mean_of(fit, 'rmse_mean_deg50'):.4f} effect "
                  f"{mean_of(fit, 'effect_ratio_deg50'):.3f} var {mean_of(fit, 'var_ratio'):.3f} "
                  f"negative {negative:.3f} exploded {mean_of(val + fit, 'exploded_cells'):.4f} selected {selected}",
                  flush=True)
            if selected:
                best = mmd
                save_params(cf, out / "best_params.msgpack")
                (out / "selected.json").write_text(json.dumps(
                    {"step": step, "selection": "min validation mmd2", "val_mmd2": mmd,
                     "val_rmse_mean_deg50": mean_of(val, "rmse_mean_deg50"), "test_evaluated": False}, indent=2) + "\n")
    done.write_text(json.dumps({"seed": seed, "steps": steps, "seconds": time.monotonic() - start,
                                "config_sha256": digest, "panel_sha256": sha256(panel),
                                "first_1000_fm": float(np.mean(losses[:1000])),
                                "last_1000_fm": float(np.mean(losses[-1000:])),
                                "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename}, indent=2) + "\n")
    print("TRAINING_COMPLETE", variant, seed, flush=True)


def evaluate(variant: str, seed: int, split: str, offset: int, root: str, panel: str) -> None:
    """Generate one split from the selected checkpoint and score it with the frozen evaluator."""
    from sciplex_eval import evaluate_predictions, summarize

    run = run_dir(root, variant, seed)
    params = run / "best_params.msgpack"
    if not (run / "training_complete.json").exists():
        raise SystemExit(f"{run} has not finished training")
    digest = sha256(params)
    dest = run / f"{split}_sample_s{offset}"
    summary_path = dest / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("checkpoint_sha256") == digest:
        print(f"{variant} seed {seed} {split} offset {offset}: up to date", flush=True)
        return
    cfg = json.loads((run / "config.json").read_text())
    data = Data(panel)
    train_ad = model_data(data, cfg, root)
    cf, _ = build(train_ad, cfg, seed)
    load_params(cf, params)
    conds = conditions(data, split)
    cells, negative = generate(cf, train_ad, conds, seed, offset)
    pred = cells_to_adata(data, cells)
    dest.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(dest / "preds.h5ad", compression="gzip")
    records = evaluate_predictions(data.spec, data.adata, pred, split=split, seed=0)
    meta = {"variant": variant, "seed": seed, "split": split, "sampling_offset": offset, "checkpoint_sha256": digest,
            "selected_step": json.loads((run / "selected.json").read_text())["step"], "negative_fraction": negative}
    write_jsonl(dest / "per_condition.jsonl", [{**meta, **r} for r in records])
    summary_path.write_text(json.dumps({**meta, **summarize(records)}, indent=2, default=float) + "\n")
    print(f"{variant} seed {seed} {split} offset {offset}: mmd2 {mean_of(records, 'mmd2'):.5f} deg50 "
          f"{mean_of(records, 'rmse_mean_deg50'):.4f}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "evaluate"])
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", help="config path, instead of configs/interp/cellflow_<variant>.json")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--sampling-seed", type=int, default=0)
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    if a.stage == "train":
        train(a.variant, a.seed, a.root, a.data, a.config)
    else:
        evaluate(a.variant, a.seed, a.split, a.sampling_seed, a.root, a.data)


if __name__ == "__main__":
    main()
