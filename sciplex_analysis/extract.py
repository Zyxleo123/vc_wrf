"""Compact, model-agnostic extracts of measured and generated cells.

  measured     r(x) of every panel cell, and per-gene mean/SD of each held-out condition.
               Also fits (once) the non-program PCA basis used for MMD outside the
               proliferation program, saved to artifacts/analysis_v1/ so every cluster
               uses the identical basis. It is fitted on control + train cells only.
  predictions  for every generated condition: r(x) of each cell, per-gene mean/SD, and
               MMD² against the measured cells in the non-program basis.

Extracts are a few MB per run, so they can be copied between clusters. CellFlow's
generated cells live on PSC; run the predictions mode there and copy the output back
into results/analysis/extracts/cellflow/:

  python -m sciplex_analysis.extract predictions --model cellflow \\
      --manifest docs/cellflow/artifacts.json --data /path/to/processed/a549_hdaci_v1.h5ad

Existing extracts are skipped, so an interrupted job is simply resubmitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist

from sciplex_eval import EvalSpec, response_score
from sciplex_eval.metrics import mmd2_rbf

from .common import (BASIS_PATH, DEFAULT_PANEL, DEFAULT_ROOT, MODELS, SPEC_DIR, SPLITS, cellflow_manifest,
                     chemcpa_manifest, extract_path)

N_PCS, N_BANDWIDTH_CELLS, SEED = 50, 2000, 0


def dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class Basis:
    """PCA(50) of the non-program eval genes with median-heuristic RBF bandwidths,
    mirroring how the frozen spec built its all-gene MMD space."""

    def __init__(self, path: Path = BASIS_PATH):
        z = np.load(path)
        self.gene_idx, self.mean, self.components = z["gene_idx"], z["mean"], z["components"]
        self.sigmas = [float(s) for s in z["sigmas"]]

    def project(self, X_spec_order: np.ndarray) -> np.ndarray:
        return (X_spec_order[:, self.gene_idx] - self.mean) @ self.components.T


def fit_basis(spec: EvalSpec, panel: ad.AnnData, path: Path = BASIS_PATH) -> None:
    from sklearn.decomposition import PCA
    nonprogram = np.setdiff1d(np.arange(spec.n_genes), spec.program_idx)
    fit = np.flatnonzero(panel.obs.split.isin(["control", "train"]).to_numpy())
    X = spec.align(panel.X, panel.var_names)
    Xfit = dense(X[fit])[:, nonprogram]
    pca = PCA(n_components=N_PCS, svd_solver="full", random_state=SEED).fit(Xfit)
    sub = np.sort(np.random.default_rng(SEED).choice(len(fit), N_BANDWIDTH_CELLS, replace=False))
    Z = (Xfit[sub] - pca.mean_) @ pca.components_.T
    median = float(np.median(pdist(Z)))
    save_npz(path, gene_idx=nonprogram, mean=pca.mean_, components=pca.components_,
             sigmas=np.array([0.5 * median, median, 2 * median]),
             explained_variance_ratio=pca.explained_variance_ratio_,
             fit_cells=np.array(len(fit)), panel_obs_hash=np.array(hashlib.sha256(
                 "".join(panel.obs_names[fit]).encode()).hexdigest()))
    print(f"non-program basis: {len(nonprogram)} genes, {len(fit)} fit cells, "
          f"explained variance {pca.explained_variance_ratio_.sum():.3f}, median distance {median:.3f} -> {path}")


def heldout_conditions(panel: ad.AnnData) -> list[str]:
    obs = panel.obs
    return sorted(obs.loc[obs.split.isin(SPLITS), "condition"].astype(str).unique())


def measured(spec: EvalSpec, panel: ad.AnnData, root: Path, data_path: str) -> None:
    if not BASIS_PATH.exists():
        fit_basis(spec, panel)
    else:
        print(f"using existing non-program basis {BASIS_PATH}")
    X = spec.align(panel.X, panel.var_names)
    obs = panel.obs
    r_all = response_score(X, spec)
    conditions = heldout_conditions(panel)
    cond_arr = obs.condition.astype(str).to_numpy()
    means, sds = [], []
    for c in conditions:
        Xc = dense(X[cond_arr == c])
        means.append(Xc.mean(0))
        sds.append(Xc.std(0, ddof=1))
    out = root / "extracts" / "measured.npz"
    text = lambda column: np.asarray(column, dtype=str)  # unicode, so loading never needs pickle
    save_npz(out, r=r_all, condition=text(cond_arr), drug=text(obs.drug.astype(str)),
             dose=obs.dose.to_numpy(np.float64), replicate=text(obs.replicate.astype(str)),
             split=text(obs.split.astype(str)), heldout=text(conditions),
             mean=np.array(means, np.float32), sd=np.array(sds, np.float32),
             control_mean=spec.control_mean, program_idx=spec.program_idx,
             panel_sha256=np.array(sha256(data_path)))
    print(f"measured: {len(r_all)} cells, {len(conditions)} held-out conditions -> {out}")


def predictions(spec: EvalSpec, panel: ad.AnnData, basis: Basis, entry: dict, out: Path) -> None:
    t0 = time.time()
    pred = ad.read_h5ad(entry["path"])
    if "sha256" in entry and sha256(entry["path"]) != entry["sha256"]:
        raise ValueError(f"checksum mismatch: {entry['path']}")
    Xp = spec.align(pred.X, pred.var_names)
    pc = pred.obs.condition.astype(str).to_numpy()
    obs = panel.obs
    conditions = sorted(obs.loc[obs.split.eq(entry["split"]), "condition"].astype(str).unique())
    cond_arr = obs.condition.astype(str).to_numpy()
    Xr = spec.align(panel.X, panel.var_names)
    rng = np.random.default_rng(SEED)
    r_parts, means, sds, mmd_np, n_pred = [], [], [], [], []
    for c in conditions:
        if not (pc == c).any():
            raise ValueError(f"{entry['path']}: no generated cells for {c}")
        Xc = dense(Xp[pc == c])
        real = dense(Xr[cond_arr == c])
        r_parts.append(response_score(Xc, spec))
        means.append(Xc.mean(0))
        sds.append(Xc.std(0, ddof=1))
        n_pred.append(len(Xc))
        sub_real = real if len(real) <= spec.mmd_n_max else real[rng.choice(len(real), spec.mmd_n_max, replace=False)]
        sub_pred = Xc if len(Xc) <= spec.mmd_n_max else Xc[rng.choice(len(Xc), spec.mmd_n_max, replace=False)]
        mmd_np.append(mmd2_rbf(basis.project(sub_real), basis.project(sub_pred), basis.sigmas))
    save_npz(out, conditions=np.array(conditions), n_pred=np.array(n_pred), r=np.concatenate(r_parts),
             mean=np.array(means, np.float32), sd=np.array(sds, np.float32), mmd2_nonprogram=np.array(mmd_np),
             meta=np.array(json.dumps({k: entry[k] for k in ("method", "seed", "split", "sampling", "path")})))
    print(f"  {entry['method']} seed {entry['seed']} {entry['split']} s{entry['sampling']}: "
          f"{len(conditions)} conditions, {sum(n_pred)} cells, {time.time() - t0:.0f}s -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["measured", "predictions"])
    p.add_argument("--model", choices=sorted(MODELS), help="predictions: which model's runs")
    p.add_argument("--manifest", help="predictions: artifacts.json-style index (default: CellFlow's, or the chemCPA run tree)")
    p.add_argument("--methods", nargs="*", help="predictions: restrict to these methods")
    p.add_argument("--task-index", type=int, help="predictions: process only the i-th method (array jobs)")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--force", action="store_true", help="overwrite existing extracts")
    a = p.parse_args(argv)
    root = Path(a.root)
    spec = EvalSpec.load(SPEC_DIR)
    panel = ad.read_h5ad(a.data)
    if a.mode == "measured":
        measured(spec, panel, root, a.data)
        return
    if a.model is None:
        p.error("predictions needs --model")
    if a.manifest:
        entries = [{"sampling": 0, **e} for e in json.loads(Path(a.manifest).read_text())["predictions"]]
    else:
        entries = chemcpa_manifest() if a.model == "chemcpa" else cellflow_manifest()
    # Registry order, so array task i always maps to the same method.
    methods = a.methods or sorted(MODELS[a.model]["methods"])
    unknown = set(methods) - set(MODELS[a.model]["methods"])
    if unknown:
        raise ValueError(f"methods not registered for {a.model}: {sorted(unknown)}")
    if a.task_index is not None:
        if a.task_index >= len(methods):
            print(f"task {a.task_index}: no method (only {len(methods)}); nothing to do")
            return
        methods = [methods[a.task_index]]
    basis = Basis()
    todo = [e for e in entries if e["method"] in methods]
    print(f"{a.model}: {len(todo)} prediction files for {methods}")
    for entry in todo:
        out = extract_path(root, a.model, entry["method"], entry["seed"], entry["split"], entry["sampling"])
        if out.exists() and not a.force:
            print(f"  exists, skipping: {out}")
            continue
        predictions(spec, panel, basis, entry, out)


if __name__ == "__main__":
    main()
