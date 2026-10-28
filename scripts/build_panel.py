"""Build the A549 HDAC-inhibitor panel, its split, and the frozen evaluation spec (v1).

Panel  A549, 24 h, hash-assigned cells: vehicle controls + every drug annotated by
       sci-Plex as pathway_level_2 == "Histone deacetylation" with "HDAC" in target
       (26 drugs x 4 doses x 2 replicate wells). Human genes only; X = log1p(CP10K)
       with the library size taken over all human genes.
Split  Every drug keeps 10 nM, 10 uM and one intermediate dose in train, and holds out
       its other intermediate dose (100 or 1000 nM) as a whole condition (both wells).
       Drugs are ranked by median r(x) at 10 uM and taken in blocks of 3: one random drug
       per block is a val drug, the rest are test drugs, so both splits span potent and
       inactive drugs. Along the ranking, the held-out dose alternates 100/1000 nM
       within each split. Vehicle cells are the permitted controls for every split.
Spec   Fitted on control + train cells only: 2000 HVGs (+ program genes), mu0/s0/tau_q
       from controls, PCA(50) and median-heuristic RBF bandwidths for MMD.
Pilot  The two program-score variants are compared on 10 nM / 10 uM conditions only
       (always train) and written to <spec>/pilot.json.

usage: python -I scripts/build_panel.py [--data-dir DIR] [--score log_sum|mean_log]
         [--quantiles Q ...] [--primary-q Q] [--version vN] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sciplex_eval.programs import ALIASES, PROLIFERATION  # noqa: E402
from sciplex_eval.spec import EPS, EvalSpec, activity_from_program_genes, response_score  # noqa: E402

try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem

CELL_LINE = "A549"
QUANTILES = [0.9, 0.95, 0.975]  # tau_q = q-quantile of control r(x)
PRIMARY_Q = 0.95
LOW, HIGH, INTERMEDIATE = 10.0, 10000.0, (100.0, 1000.0)
N_HVG, N_PCS, DEG_TOP_K, MMD_N_MAX = 2000, 50, 50, 2000
SEED = 0
SPLIT_RULE = (
    "Each HDACi drug holds out one intermediate dose (100 or 1000 nM; both replicate wells) "
    "and keeps 10 nM, 10 uM and its other intermediate dose in train. Drugs ranked by median "
    "r(x) at 10 uM are taken in blocks of 3; one random drug per block is a val drug, the rest "
    "are test drugs. Along the ranking the held-out dose alternates 100/1000 nM within each split. "
    "All vehicle cells are permitted controls for every split."
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_rows(path, keep, block=100_000):
    """Rows `keep` of the on-disk CSR X, read as one sequential scan (fast with gzip chunks)."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        indptr, n_genes = X["indptr"][:], int(X.attrs["shape"][1])
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        parts = []
        for s in range(0, len(indptr) - 1, block):
            e = min(s + block, len(indptr) - 1)
            if not keep[s:e].any():
                continue
            lo, hi = indptr[s], indptr[e]
            blk = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], indptr[s:e + 1] - lo),
                                shape=(e - s, n_genes))
            parts.append(blk[keep[s:e]])
    return sp.vstack(parts).tocsr(), obs[keep].copy(), var


def program_columns(var):
    ids = []
    for g in PROLIFERATION:
        for name in [g] + ALIASES.get(g, []):
            hits = var.index[var["gene_short_name"] == name]
            if len(hits):
                ids.append(var.loc[hits, "total_counts"].idxmax())
                break
        else:
            log(f"  program gene not found: {g}")
    return ids


def wrf(r, tau):
    return float((r < tau).mean())


def pilot(obs, Xprog, ctrl, score_type, quantiles, primary_q):
    a = activity_from_program_genes(Xprog, score_type)
    mu0, s0 = a[ctrl].mean(), a[ctrl].std(ddof=1)
    r = (mu0 - a) / (s0 + EPS)
    taus = np.quantile(r[ctrl], quantiles)
    tau = taus[quantiles.index(primary_q)]
    drugs = sorted(obs.loc[~ctrl, "drug"].unique())
    auroc = {}
    for d in drugs:
        m = ((obs.drug == d) & (obs.dose == HIGH)).values
        auroc[d] = float(roc_auc_score(np.r_[np.zeros(ctrl.sum()), np.ones(m.sum())], np.r_[r[ctrl], r[m]]))
    rep = []
    for d in drugs:
        for dose in (LOW, HIGH):
            m = ((obs.drug == d) & (obs.dose == dose)).values
            f1 = wrf(r[m & (obs.replicate == "rep1").values], tau)
            f2 = wrf(r[m & (obs.replicate == "rep2").values], tau)
            rep.append((f1, f2))
    rep = np.array(rep)
    per_plate = pd.Series(r[ctrl]).groupby(obs.loc[ctrl, "culture_plate"].astype(str).values).mean()
    return r, dict(mu0=float(mu0), s0=float(s0), taus=[float(t) for t in taus]), {
        "auroc_10uM_vs_control_mean": float(np.mean(list(auroc.values()))),
        "n_drugs_auroc_10uM_gt_0.75": int(sum(v > 0.75 for v in auroc.values())),
        "replicate_wrf_abs_diff_pp_mean": float(100 * np.abs(rep[:, 0] - rep[:, 1]).mean()),
        "replicate_wrf_pearson": float(np.corrcoef(rep[:, 0], rep[:, 1])[0, 1]),
        "control_wrf_rep1": wrf(r[ctrl & (obs.replicate == "rep1").values], tau),
        "control_wrf_rep2": wrf(r[ctrl & (obs.replicate == "rep2").values], tau),
        "control_plate_mean_r_sd": float(per_plate.std(ddof=1)),
        "auroc_10uM_by_drug": auroc,
    }


def make_split(obs, r, rng):
    drugs = sorted(obs.loc[obs.drug != "Vehicle", "drug"].unique())
    top = {d: float(np.median(r[((obs.drug == d) & (obs.dose == HIGH)).values])) for d in drugs}
    ranked = sorted(drugs, key=lambda d: -top[d])
    group = {}
    for i in range(0, len(ranked), 3):
        block = ranked[i:i + 3]
        v = block[rng.integers(len(block))]
        for d in block:
            group[d] = "val" if d == v else "test"
    heldout = {}
    for g in ("val", "test"):
        start = rng.integers(2)
        for j, d in enumerate([d for d in ranked if group[d] == g]):
            heldout[d] = INTERMEDIATE[(start + j) % 2]
    split = np.where(obs.drug == "Vehicle", "control", "train").astype(object)
    for d in drugs:
        split[((obs.drug == d) & (obs.dose == heldout[d])).values] = group[d]
    return split, ranked, top, heldout, group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/user_data/yixiz/sciplex3")
    p.add_argument("--score", default="log_sum", choices=["log_sum", "mean_log"])
    p.add_argument("--quantiles", type=float, nargs="+", default=QUANTILES)
    p.add_argument("--primary-q", type=float, default=PRIMARY_Q)
    p.add_argument("--version", default="v1", help="bump when anything above changes")
    p.add_argument("--force", action="store_true", help="overwrite an existing (frozen) spec")
    a = p.parse_args()
    assert a.primary_q in a.quantiles
    data = Path(a.data_dir)
    out_panel = data / "processed" / f"a549_hdaci_{a.version}.h5ad"
    out_spec = REPO / "artifacts" / f"eval_spec_{a.version}"
    out_split = REPO / "configs" / f"split_{a.version}.json"
    if out_panel.exists() and not a.force:
        sys.exit(f"{out_panel} exists; re-run with --force to rebuild.")
    # With a frozen spec already in the repo, a rebuild (e.g. on another machine) only
    # writes the panel, after checking it reproduces that spec and split.
    frozen = EvalSpec.load(out_spec) if out_spec.exists() and not a.force else None

    base = data / "processed" / "sciplex3_raw_counts.h5ad"
    with h5py.File(base, "r") as f:
        o = read_elem(f["obs"])
    hdac = (o.pathway_level_2.astype(str) == "Histone deacetylation") & o.target.astype(str).str.contains("HDAC")
    vehicle = o.product_name.astype(str) == "Vehicle"
    keep = ((o.cell_type.astype(str) == CELL_LINE) & (o.time_point == 24) & (vehicle | hdac)).values
    log(f"reading {keep.sum()} cells")
    counts, obs, var = read_rows(base, keep)

    human = (var.species == "human").values
    counts, var = counts[:, human].astype(np.int32), var[human].copy()
    obs["n_counts_human"] = np.asarray(counts.sum(1)).ravel()
    var["total_counts"] = np.asarray(counts.sum(0)).ravel()
    Xn = sp.diags(1e4 / obs["n_counts_human"].values) @ counts.astype(np.float64)
    Xn.data = np.log1p(Xn.data)
    Xn = Xn.astype(np.float32).tocsr()

    obs["drug"] = obs.product_name.astype(str)
    obs["dose"] = obs.dose.astype(float)
    obs["replicate"] = obs.replicate.astype(str)
    assert (obs.loc[obs.drug == "Vehicle", "dose"] == 0).all()
    obs["condition"] = [f"{CELL_LINE}|{d}|{x:g}" for d, x in zip(obs.drug, obs.dose)]
    ctrl = (obs.drug == "Vehicle").values
    log(f"controls {ctrl.sum()}, treated {(~ctrl).sum()}, drugs {obs.drug.nunique() - 1}, "
        f"median human UMIs {np.median(obs.n_counts_human):.0f}")

    prog_ids = program_columns(var)
    prog_cols = var.index.get_indexer(prog_ids)
    log(f"program genes found: {len(prog_ids)}/{len(PROLIFERATION)}")

    pilots, scored = {}, {}
    for st in ("log_sum", "mean_log"):
        r, stats, pil = pilot(obs, Xn[:, prog_cols], ctrl, st, a.quantiles, a.primary_q)
        scored[st], pilots[st] = (r, stats), pil
        log(f"pilot {st}: " + ", ".join(f"{k}={v:.3f}" for k, v in pil.items() if not isinstance(v, dict)))
    r, stats = scored[a.score]

    rng = np.random.default_rng(SEED)
    obs["split"], ranked, top, heldout, group = make_split(obs, r, rng)
    log("cells per split: " + str(obs.split.value_counts().to_dict()))

    fit = obs.split.isin(["train", "control"]).values
    ok = np.asarray((Xn[fit] > 0).sum(0)).ravel() >= 20
    tmp = ad.AnnData(Xn[fit][:, ok], var=pd.DataFrame(index=var.index[ok]))
    sc.pp.highly_variable_genes(tmp, n_top_genes=N_HVG, flavor="seurat")
    hvg = set(tmp.var_names[tmp.var.highly_variable])
    in_eval = var.index.isin(hvg) | var.index.isin(prog_ids)
    var["highly_variable"] = var.index.isin(hvg)
    var["in_program"] = var.index.isin(prog_ids)
    eval_ids = list(var.index[in_eval])
    log(f"eval genes: {len(eval_ids)} ({len(hvg)} HVG, {len(prog_ids)} program, "
        f"{len(set(prog_ids) & hvg)} overlap)")

    X_eval = Xn[:, in_eval].tocsr()
    pca = PCA(n_components=N_PCS, svd_solver="full", random_state=SEED).fit(X_eval[fit].toarray())
    sub = rng.choice(np.flatnonzero(fit), min(2000, fit.sum()), replace=False)
    Z = pca.transform(X_eval[sub].toarray())
    d = np.sqrt(np.maximum((Z * Z).sum(1)[:, None] + (Z * Z).sum(1)[None] - 2 * Z @ Z.T, 0))
    med = float(np.median(d[np.triu_indices(len(Z), 1)]))
    log(f"PCA explained variance (50 PCs): {pca.explained_variance_ratio_.sum():.3f}; median dist {med:.3f}")

    spec = EvalSpec(
        version=a.version, cell_line=CELL_LINE, gene_ids=eval_ids,
        gene_symbols=list(var.loc[eval_ids, "gene_short_name"]),
        program_name="tirosh2016_S+G2M", program_gene_ids=prog_ids, score_type=a.score,
        mu0=stats["mu0"], s0=stats["s0"], quantiles=a.quantiles, taus=stats["taus"], primary_q=a.primary_q,
        deg_top_k=DEG_TOP_K, mmd_n_max=MMD_N_MAX, mmd_sigmas=[0.5 * med, med, 2 * med],
        notes={
            "representation": "log1p(UMI / total human-gene UMIs * 1e4); predictions must use this space",
            "controls": f"all {CELL_LINE} 24h vehicle cells (both replicates) = permitted controls",
            "taus": "tau_q = q-quantile of r(x) over controls, so f_tau(controls) = q",
            "fit_cells": "control + train cells only (val/test conditions excluded)",
            "pca": "sklearn PCA(50) on unscaled log1p(CP10K) of eval genes",
            "mmd_sigmas": "0.5/1/2 x median pairwise PCA distance of 2000 fit cells",
            "deg": f"top {DEG_TOP_K} genes by |mean(real condition) - mean(controls)|",
            "panel": out_panel.name,
        },
        pca_mean=pca.mean_.astype(np.float64), pca_components=pca.components_.astype(np.float64),
        control_mean=np.asarray(X_eval[ctrl].mean(0)).ravel().astype(np.float64),
    )
    r_spec = response_score(X_eval, spec)
    assert np.allclose(r_spec, r), "spec r(x) disagrees with pilot r(x)"
    log(f"control WRF at taus: {[round(wrf(r_spec[ctrl], t), 4) for t in spec.taus]}")

    # ---- write ------------------------------------------------------------------------
    keep_cols = ["sample", "n_umi", "n_counts_human", "Size_Factor", "condition", "drug", "dose",
                 "replicate", "split", "culture_plate", "well_oligo", "plate_oligo", "rt_well",
                 "pcr_plate", "time_point", "target", "pathway", "pathway_level_1", "pathway_level_2",
                 "catalog_number", "top_to_second_best_ratio_W"]
    pobs = obs[keep_cols].copy()
    for c in pobs.columns:
        if pobs[c].dtype == object or isinstance(pobs[c].dtype, pd.CategoricalDtype):
            pobs[c] = pobs[c].astype(str).astype("category")
    pvar = var.loc[eval_ids, ["gene_id", "gene_short_name", "highly_variable", "in_program", "total_counts"]]
    panel = ad.AnnData(X_eval, obs=pobs, var=pvar,
                       layers={"counts": counts[:, in_eval].tocsr()},
                       uns={"spec_version": a.version, "normalization": spec.notes["representation"]})

    split_info = {
        "version": a.version, "status": "proposed - confirm with the team before any test-split run",
        "cell_line": CELL_LINE, "seed": SEED, "rule": SPLIT_RULE,
        "drugs_ranked_by_10uM_median_r": [{"drug": dname, "median_r_10uM": round(top[dname], 3),
                                           "group": group[dname], "heldout_dose_nM": heldout[dname]}
                                          for dname in ranked],
        "controls": {"condition": f"{CELL_LINE}|Vehicle|0", "n_cells": int(ctrl.sum())},
        "conditions": [],
    }
    for cond, g in pobs.groupby("condition", observed=True):
        if g.drug.iloc[0] == "Vehicle":
            continue
        split_info["conditions"].append({
            "condition": cond, "drug": g.drug.iloc[0], "dose_nM": float(g.dose.iloc[0]),
            "split": g.split.iloc[0], "n_cells": len(g),
            "n_rep1": int((g.replicate == "rep1").sum()), "n_rep2": int((g.replicate == "rep2").sum())})

    if frozen is not None:
        same = (frozen.gene_ids == spec.gene_ids and frozen.program_gene_ids == spec.program_gene_ids
                and frozen.score_type == spec.score_type and frozen.quantiles == spec.quantiles
                and np.allclose([frozen.mu0, frozen.s0, *frozen.taus], [spec.mu0, spec.s0, *spec.taus])
                and json.loads(out_split.read_text())["conditions"] == split_info["conditions"])
        if not same:
            sys.exit(f"rebuilt spec/split differ from frozen {out_spec}; not writing anything")
        log(f"reproduced frozen {out_spec}; writing the panel only")
    else:
        out_spec.mkdir(parents=True, exist_ok=True)
        spec.save(out_spec)
        (out_spec / "pilot.json").write_text(json.dumps({"chosen": a.score, **pilots}, indent=1))
        out_split.write_text(json.dumps(split_info, indent=1))
    tmp_path = out_panel.with_suffix(".h5ad.tmp")
    panel.write_h5ad(tmp_path, compression="gzip")
    tmp_path.rename(out_panel)
    log(f"wrote {out_panel} {panel.shape}, {out_spec}, {out_split}")


if __name__ == "__main__":
    main()
