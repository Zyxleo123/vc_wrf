"""Ours (CellFlow + condition response embedding + grouped velocity matching) on the standard benchmark.

The model, losses and generation are those of sciplex_interp.cellflow (configs/interp/
cellflow_z70_lr3e4_mv10_t.json: latent flow on 50 PCs + 20 mean-shift directions with residual
decoding, grouped velocity matching with weight 10, 8 conditions per update, OT coupling, EMA,
64 Heun steps). What changes for chemCPA's three-cell-line data:

  * cell line is a sample covariate of the condition encoder (one-hot), and control cells are matched
    within cell line (split covariate), instead of within replicate;
  * the drug input is its Uni-Mol embedding (standardised; the structure DOLORIS also receives), not a
    one-hot code, so unseen drugs (Table 2) can be encoded;
  * the condition response embedding comes from sciplex_standard.teacher (cell-line input, drug code
    computed from structure);
  * the latent basis is the standard spec PCA (fit on control + train cells) plus 20 directions of the
    training mean shifts (each relative to its cell line's vehicle mean);
  * checkpoint selection: lowest mean MMD² on the `val` conditions (held out of training), at the
    published validation steps; the final weights are also kept ("final" rule).
Generation: 100 cells per condition (DOLORIS's protocol), from 100 vehicle cells of the condition's
cell line drawn once per (seed, sampling seed); negatives set to 0 as in the published generation.

usage (CellFlow environment, GPU): python -m sciplex_standard.ours train --seed 0
                                  python -m sciplex_standard.ours generate --seed 0
Writes <root>/runs/ours/seed_<s>/ and <root>/preds/{ours,ours_final}/seed_<s>/{val,test,ood}.h5ad.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_eval.metrics import mmd2_rbf
from sciplex_interp.cellflow import (LOG_DOSE_MAX, ODE_STEPS, decode, load_params, make_sampler, make_step,
                                     save_params, sha256, target_means)

from .data import DEFAULT_FILE, REPO, StdData, load_spec, project

ROOT = REPO / "results" / "standard"
CONFIG = REPO / "configs" / "standard_ours_v1.json"
N_GEN = 100
SPLITS = ("val", "test", "ood")


def latent_basis(data: StdData, spec: dict, mean_dirs: int) -> tuple[np.ndarray, np.ndarray]:
    U = spec["pca_components"].astype(np.float64)
    if mean_dirs:
        m0 = {c: data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) for c in data.cell_types}
        shifts = np.stack([data.X[data.idx(k)].mean(0, dtype=np.float64) - m0[data.cond.cell_type[k]]
                           for k in data.keys("train", min_cells=1)])
        V = np.linalg.svd(shifts, full_matrices=False)[2][:mean_dirs]
        V = V - (V @ U.T) @ U
        U = np.vstack([U, np.linalg.qr(V.T)[0].T])
    return U.astype(np.float32), spec["pca_mean"].astype(np.float32)


def model_data(data: StdData, spec: dict, cfg: dict, root: Path, mol_path: str):
    import anndata as ad
    rows = data.train_rows()
    o = data.obs.iloc[rows]
    obs = pd.DataFrame({"is_control": o.control.to_numpy(), "cell_type": o.cell_type.to_numpy(),
                        "drug": o.drug.to_numpy(), "dose_nM": o.dose_nM.to_numpy(),
                        "log_dose": (np.log1p(o.dose_nM.to_numpy()) / LOG_DOSE_MAX).astype(np.float32),
                        "model_condition": o.key.to_numpy(),
                        "teacher_key": np.where(o.control, "control", o.key.to_numpy())},
                       index=[f"c{i}" for i in rows])
    a = ad.AnnData(X=data.X[rows], obs=obs)
    z = np.load(mol_path)
    names = [str(d) for d in z["drugs"]]
    E = z["emb"].astype(np.float64)
    E = ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)
    a.uns["drug_embeddings"] = {d: E[names.index(d)] for d in data.drugs}
    a.uns["drug_embeddings"]["control"] = np.zeros(E.shape[1], dtype=np.float32)
    a.uns["cell_type_embeddings"] = {c: np.eye(len(data.cell_types), dtype=np.float32)[i] for i, c in enumerate(data.cell_types)}
    t = np.load(Path(root) / "teacher" / "std_autodec.npz")
    a.uns["teacher_embeddings"] = {str(k): (c / t["scale"]).astype(np.float32) for k, c in zip(t["keys"], t["coef"])}
    a.uns["teacher_embeddings"]["control"] = np.zeros(t["coef"].shape[1], dtype=np.float32)
    U, centre = latent_basis(data, spec, int(cfg["latent"]["mean_dirs"]))
    a.obsm["X_latent"] = ((a.X - centre) @ U.T).astype(np.float32)
    a.uns["latent"] = {"basis": U, "centre": centre, "decode": cfg["latent"]["decode"]}
    return a


def build(train, cfg: dict, seed: int):
    import optax
    from cellflow.model import CellFlow
    from cellflow.utils import match_linear

    cf = CellFlow(train, solver="otfm")
    covariates = {"drug": ("drug",), "dose": ("log_dose",), "teacher": ("teacher_key",)}
    cf.prepare_data(sample_rep="X_latent", control_key="is_control", perturbation_covariates=covariates,
                    perturbation_covariate_reps={"drug": "drug_embeddings", "teacher": "teacher_embeddings"},
                    sample_covariates=["cell_type"], sample_covariate_reps={"cell_type": "cell_type_embeddings"},
                    split_covariates=["cell_type"], max_combination_length=1)
    schedule = optax.warmup_cosine_decay_schedule(init_value=1e-5, peak_value=cfg["learning_rate_peak"],
                                                  warmup_steps=cfg["warmup_steps"], decay_steps=cfg["training_steps"],
                                                  end_value=cfg["learning_rate_end"])
    optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adamw(schedule, weight_decay=cfg["weight_decay"]))
    mlp = lambda dims: {"layer_type": "mlp", "dims": list(dims), "dropout_rate": 0.0}
    cf.prepare_model(condition_mode="deterministic", pooling="mean", regularization=0.0,
                     layers_before_pool={k: mlp(cfg["condition_mlp_dims"]) for k in [*covariates, "cell_type"]},
                     layers_after_pool=mlp([128]), condition_embedding_dim=cfg["condition_embedding_dim"],
                     cond_output_dropout=cfg["condition_output_dropout"], time_freqs=cfg["time_freqs"],
                     time_encoder_dims=tuple(cfg["time_encoder_dims"]), hidden_dims=tuple(cfg["hidden_dims"]),
                     decoder_dims=tuple(cfg["decoder_dims"]), probability_path={"constant_noise": cfg["constant_noise"]},
                     match_fn=functools.partial(match_linear, epsilon=cfg["ot_epsilon"], tau_a=1.0, tau_b=1.0),
                     optimizer=optimizer, solver_kwargs={"ema": cfg["ema"]}, seed=seed)
    return cf, schedule


def sources(train, data: StdData, seed: int, offset: int):
    """N_GEN vehicle cells per cell line, drawn once per (seed, sampling seed)."""
    rng = np.random.default_rng(20000 + seed + 1000 * offset)
    ids = []
    for c in data.cell_types:
        pool = train.obs.index[train.obs.is_control & train.obs.cell_type.eq(c)].to_numpy()
        ids.extend(rng.choice(pool, N_GEN, replace=False))
    return train[ids].copy()


def covariates(train, data: StdData, keys: list[str]) -> pd.DataFrame:
    template = train.obs[~train.obs.is_control].iloc[0]
    rows = []
    for k in keys:
        c = data.cond.loc[k]
        r = template.copy()
        r["cell_type"], r["drug"], r["dose_nM"] = c.cell_type, c.drug, c.dose_nM
        r["log_dose"] = np.float32(np.log1p(c.dose_nM) / LOG_DOSE_MAX)
        r["model_condition"], r["teacher_key"] = k, k
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def generate(cf, train, data: StdData, keys: list[str], seed: int, offset: int = 0) -> dict:
    import diffrax
    src = sources(train, data, seed, offset)
    out = cf.predict(src, covariates(train, data, keys), condition_id_key="model_condition", solver=diffrax.Heun(),
                     stepsize_controller=diffrax.ConstantStepSize(), dt0=1 / ODE_STEPS, max_steps=2 * ODE_STEPS)
    lat = train.uns["latent"]
    cells = {}
    for k in keys:
        rows = np.flatnonzero(src.obs.cell_type.eq(data.cond.cell_type[k]).to_numpy())
        z1 = np.asarray(out[k], dtype=np.float32)
        if not np.isfinite(z1).all():
            raise FloatingPointError(f"non-finite cells for {k}")
        x = decode(z1, src.obsm["X_latent"][rows], np.asarray(src.X[rows]), lat["basis"], lat["centre"], lat["decode"])
        cells[k] = np.maximum(x, 0).astype(np.float32)
    return cells


def val_mmd(data: StdData, spec: dict, cells: dict) -> float:
    vals = []
    for k, X in cells.items():
        real = data.X[data.idx(k)]
        vals.append(mmd2_rbf(project(spec, real), project(spec, X), spec["mmd_sigmas"]))
    return float(np.mean(vals))


def train(seed: int, root: Path, path: str, mol_path: str) -> None:
    import jax

    cfg = json.loads(CONFIG.read_text())
    out = root / "runs" / "ours" / f"seed_{seed}"
    done = out / "training_complete.json"
    if done.exists():
        print(f"seed {seed}: already trained")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(CONFIG, out / "config.json")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit(f"no GPU visible to JAX: {jax.devices()}")
    data, spec = StdData(path), load_spec()
    train_ad = model_data(data, spec, cfg, root, mol_path)
    cf, schedule = build(train_ad, cfg, seed)
    sampler = make_sampler(cf, cfg["batch_size"])
    K = int(cfg["conditions_per_batch"])
    step_fn = make_step(cf.solver, cfg["aux"], target_means(cf), K)
    val_keys = data.keys("val")
    steps, checks = int(cfg["training_steps"]), set(cfg["validation_steps"])
    rng_np, rng_jax = np.random.default_rng(seed), jax.random.PRNGKey(seed)
    best, losses, start = float("inf"), [], time.monotonic()
    with open(out / "history.jsonl", "w", buffering=1) as hist:
        for step in range(1, steps + 1):
            rng_jax, step_rng = jax.random.split(rng_jax)
            total, fm, term, active = (float(v) for v in step_fn(step_rng, sampler.sample_many(rng_np, K)))
            if not np.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: fm {fm} aux {term}")
            losses.append(fm)
            if step == 1 or step % 1000 == 0:
                print(f"seed {seed} step {step} fm {np.mean(losses[-1000:]):.5f} aux {term:.6f} "
                      f"seconds {time.monotonic() - start:.0f}", flush=True)
            if step not in checks:
                continue
            cf.solver.is_trained = True
            mmd = val_mmd(data, spec, generate(cf, train_ad, data, val_keys, seed))
            selected = mmd < best
            hist.write(json.dumps({"step": step, "val_mmd2": mmd, "fm": float(np.mean(losses[-1000:])), "selected": selected}) + "\n")
            print(f"VALIDATION seed {seed} step {step} val mmd2 {mmd:.5f} selected {selected}", flush=True)
            if selected:
                best = mmd
                save_params(cf, out / "best_params.msgpack")
                (out / "selected.json").write_text(json.dumps({"step": step, "val_mmd2": mmd}, indent=1) + "\n")
    save_params(cf, out / "final_params.msgpack")
    done.write_text(json.dumps({"seed": seed, "steps": steps, "seconds": time.monotonic() - start,
                                "config_sha256": sha256(CONFIG), "job": os.environ.get("SLURM_JOB_ID"),
                                "host": os.uname().nodename}, indent=1) + "\n")
    print("TRAINING_COMPLETE", seed, flush=True)


def generate_all(seed: int, root: Path, path: str, mol_path: str, offset: int = 0) -> None:
    import anndata as ad

    run = root / "runs" / "ours" / f"seed_{seed}"
    if not (run / "training_complete.json").exists():
        raise SystemExit(f"{run} has not finished training")
    cfg = json.loads((run / "config.json").read_text())
    data, spec = StdData(path), load_spec()
    train_ad = model_data(data, spec, cfg, root, mol_path)
    cf, _ = build(train_ad, cfg, seed)
    for method, params in (("ours", "best_params.msgpack"), ("ours_final", "final_params.msgpack")):
        todo = [s for s in SPLITS if not (root / "preds" / method / f"seed_{seed}" / f"{s}.h5ad").exists()]
        if not todo:
            continue
        load_params(cf, run / params)
        for split in todo:
            keys = data.keys(split)
            cells = generate(cf, train_ad, data, keys, seed, offset)
            labels = np.concatenate([[k] * len(cells[k]) for k in keys])
            pred = ad.AnnData(np.vstack([cells[k] for k in keys]), obs=pd.DataFrame({"condition": labels},
                              index=[f"g{i}" for i in range(len(labels))]), var=pd.DataFrame(index=data.genes))
            dest = root / "preds" / method / f"seed_{seed}" / f"{split}.h5ad"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.stem + ".tmp.h5ad")
            pred.write_h5ad(tmp, compression="gzip")
            tmp.rename(dest)
            print(f"{method} seed {seed} {split}: {len(keys)} conditions -> {dest}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "generate"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--molecules", default=str(ROOT / "unimol_embed.npz"))
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(ROOT))
    a = p.parse_args(argv)
    if a.stage == "train":
        train(a.seed, Path(a.root), a.file, a.molecules)
    else:
        generate_all(a.seed, Path(a.root), a.file, a.molecules)


if __name__ == "__main__":
    main()
