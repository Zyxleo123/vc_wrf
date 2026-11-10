"""What the teacher's 20 numbers are (shown to readers as the "condition response embedding"): where conditions sit in them, and what each coordinate means.

The teacher (sciplex_interp.teacher) maps a free 4-number drug code + log10 dose to 20 numbers: how far the condition's mean
moved from vehicle along each of the top 20 directions of the 78 training mean shifts. It never sees a drug's structure or
class, and a coordinate is a dot product with a fixed gene-loading vector, so it can be read directly in genes. CellFlow's
input is these numbers divided by one shared scale. The sign of coordinates 2-20 is arbitrary.

Two questions, answered from files so that figures redraw without the panel:

  Are close conditions close?  Pairwise distances, nearest neighbours against a shuffle of drug labels within each dose, and
    drug-pair distances against drug class and chemical similarity. Conditions lie near one curved path out of vehicle, so
    strength is removed by subtracting that whole curve (`off path`): removing coordinate 1 alone leaves the arch, and drugs whose
    path crosses coordinate 2 = 0 land beside inactive drugs. Same-drug closeness is partly built in for the teacher, which shares
    one code across a drug's doses, so the measured means are always shown beside it. Drugs with no response are the negative
    control: their same-drug closeness cannot be the drug's main effect, and shared culture plates are checked against it.

  What does each coordinate mean, and is it real?  Do a condition's two wells agree on an axis found without that condition's fold
    (replicate_r, 5 folds x 5 repeats)? Also the teacher's fit, seed agreement, dose / drug / interaction shares over the teacher's balanced grid, how
    much of the coordinate is a smooth function of coordinate 1, how much rests on one condition, technical checks (plate,
    sequencing depth, gene abundance, mitochondrial genes), gene loadings and gene sets.

    signal_share (axes.csv) is the first version of "real" and is too generous: it counts any variance beyond cell sampling, so a
    well-level technical effect passes. It is kept for the record; replicate_r is the one used.

No test-condition measurement is read here. Test conditions appear only as teacher predictions, from labels. Validation
means are used, as in sciplex_interp.teacher's own sanity check, and the teacher was chosen on them.

Selectivity classes are not in the panel (23 of 26 drugs have target "HDAC"). SELECTIVITY is hand-written from vendor
datasheets and general pharmacology. The structure class is computed from SMILES.

  collect  writes <root>/figures/teacher/: vectors.csv, measured.csv, drugs.csv, axes.csv, genes.csv, loadings.npz,
           genesets.csv, gene_means.csv, distances.csv, neighbours.csv, inactive_pairs.csv, pairs.csv, similarity.csv, embeddings.csv,
           path.csv, seeds.npz, meta.json
  draw     PNGs from those files only

usage: python -m sciplex_interp.teacher_viz collect
       python -m sciplex_interp.teacher_viz draw
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cellflow_figures import AXIS, GRID, INK, INK_2, MUTED, RAMP, finish, margins, setup
from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data
from .stats import K_BASIS, Context, autodecoder_seeds

GRID_NM = (10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0)
MEASURED_NM = (10.0, 100.0, 1000.0, 10000.0)
GENE_SETS = "/zfsauton/scratch/yixiz/sciplex3/raw/genesets"  # Enrichr's text downloads, 2026-11-06
LIBRARIES = {"MSigDB_Hallmark_2020": "Hallmark", "GO_Biological_Process_2023": "GO BP"}
MIN_SET = 10  # genes of a set that must be in the panel
N_PERM = 5000
N_SHUFFLE = 50
N_BOOT = 2000
K_CODE, WD = 4, 1e-4  # the recipe behind results/interp/teacher/autodec.npz (sciplex_interp.teacher defaults)
REAL_SHARE = 0.5  # an axis counts as real if at least half of one well's variance reappears in the other well (replicate_r)
RESPONDER = 0.25  # share of the strongest drug's coordinate 1 at 10 µM above which a drug counts as a responder
INACTIVE = 0.5  # coordinate 1 at 10 µM below which a drug counts as having no response: the negative control
N_SHUFFLE_R = 20
PATH_DEGREE = 4

BROAD, CLASS1, SELECTIVE, FATTY, ACTIVATOR = "Broad", "Class I", "Selective", "Fatty acid", "Activator"
CLASSES = [BROAD, CLASS1, SELECTIVE, FATTY, ACTIVATOR]
SELECTIVITY = {
    "AR-42": BROAD, "Abexinostat (PCI-24781)": BROAD, "Belinostat (PXD101)": BROAD, "CUDC-101": BROAD, "CUDC-907": BROAD,
    "Dacinostat (LAQ824)": BROAD, "Givinostat (ITF2357)": BROAD, "M344": BROAD, "Panobinostat (LBH589)": BROAD,
    "Pracinostat (SB939)": BROAD, "Quisinostat (JNJ-26481585) 2HCl": BROAD, "Resminostat": BROAD, "Trichostatin A (TSA)": BROAD,
    "Entinostat (MS-275)": CLASS1, "Mocetinostat (MGCD0103)": CLASS1, "Tacedinaline (CI994)": CLASS1, "Tucidinostat (Chidamide)": CLASS1,
    "Droxinostat": SELECTIVE, "MC1568": SELECTIVE, "PCI-34051": SELECTIVE, "TMP195": SELECTIVE, "Tubastatin A HCl": SELECTIVE,
    "Divalproex Sodium": FATTY, "Sodium Phenylbutyrate": FATTY, "Valproic acid sodium salt (Sodium valproate)": FATTY,
    "ITSA-1 (ITSA1)": ACTIVATOR,
}
STRUCTURES = (("Hydroxamate", "[CX3](=O)[NX3;H1][OX2H1]"), ("Benzamide", "[NX3;H2]c1ccccc1[NX3;H1]C(=O)"),
              ("Carboxylate", "[CX3](=O)[O-,OX2H1]"), ("Carboxylate", "[CX3](=O)O[Na]"))  # GEO writes phenylbutyrate's sodium as bonded

# dataviz reference palette: slots 1-3 pass all-pairs in a scatter; the other classes take neutrals, with marker shape as a second cue.
CLASS_COLOR = {BROAD: "#2a78d6", CLASS1: "#eb6834", SELECTIVE: "#1baf7a", FATTY: MUTED, ACTIVATOR: INK_2}
CLASS_MARKER = {BROAD: "o", CLASS1: "s", SELECTIVE: "^", FATTY: "D", ACTIVATOR: "X"}
NEGATIVE, MIDPOINT, POSITIVE = "#2a78d6", "#f0efec", "#e34948"  # diverging: blue <-> red through a neutral gray


def short(drug: str) -> str:
    named = {"Valproic acid sodium salt (Sodium valproate)": "Valproic acid", "Divalproex Sodium": "Divalproex",
             "Sodium Phenylbutyrate": "Phenylbutyrate"}
    return named.get(drug, re.sub(r"\s*(2?HCl)$", "", re.sub(r"\s*\(.*?\)", "", drug)).strip())


def out_dir(root: Path) -> Path:
    return Path(root) / "figures" / "teacher"


def coord_names(k: int = K_BASIS) -> list[str]:
    return [f"c{i + 1}" for i in range(k)]


# ---- statistics, all on plain arrays --------------------------------------------------------------
def crossfit_parts(D_fit: np.ndarray, D_eval: np.ndarray, cells_eval: list[np.ndarray], k: int = K_BASIS):
    """Axes fit on one replicate, measured on the other: coordinates and each coordinate's sampling variance (conditions x axes), and the axes."""
    V = np.linalg.svd(D_fit, full_matrices=False)[2][:k]
    noise = np.array([(X @ V.T).var(0, ddof=1) / len(X) for X in cells_eval])
    return D_eval @ V.T, noise, V


def crossfit_signal(D_fit: np.ndarray, D_eval: np.ndarray, cells_eval: list[np.ndarray], k: int = K_BASIS):
    """1 - sampling noise / variance across conditions, per axis, on a replicate that played no part in finding the axis.

    Returns (share, axes). The other replicate's noise is independent of the axis, so the variance is signal + noise and the share
    is centred near 0 for an axis that only fit noise. Variance across conditions rather than squared length from vehicle: the
    vehicle mean's noise is one draw shared by every condition, and a large draw lifted noise axes over the bar (2% of them, in the
    tests). Centring cancels it exactly. One axis's share is still noisy (SD about sqrt(2 / conditions) under pure noise), so read it
    with signal_lo / signal_hi and the shuffled null in axes.csv.
    """
    coords, noise, V = crossfit_parts(D_fit, D_eval, cells_eval, k)
    return 1 - noise.mean(0) / coords.var(0, ddof=1), V


def shuffled_wells(X: np.ndarray, wells: dict[str, list[np.ndarray]], rng) -> tuple[dict, dict]:
    """Mean shifts and cells with condition labels shuffled within each replicate: sizes and heavy tails stay, nothing real is left."""
    D, cells = {}, {}
    for r in wells:
        order = rng.permutation(np.concatenate(wells[r]))
        cells[r] = [X[i] for i in np.split(order, np.cumsum([len(w) for w in wells[r]])[:-1])]
        D[r] = np.array([c.mean(0) for c in cells[r]]) - X[order].mean(0)
    return D, cells


def shuffled_null(X: np.ndarray, wells: dict[str, list[np.ndarray]], k: int, rng, shuffles: int = N_SHUFFLE) -> np.ndarray:
    """Signal share under shuffled labels (shuffles x axes)."""
    (a, b) = tuple(wells)
    out = []
    for _ in range(shuffles):
        D, cells = shuffled_wells(X, wells, rng)
        out.append((crossfit_signal(D[a], D[b], cells[b], k)[0] + crossfit_signal(D[b], D[a], cells[a], k)[0]) / 2)
    return np.array(out)


def column_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r of matching columns, over the second-to-last axis (so a leading repeats axis is kept)."""
    a, b = a - a.mean(-2, keepdims=True), b - b.mean(-2, keepdims=True)
    return (a * b).sum(-2) / np.sqrt((a ** 2).sum(-2) * (b ** 2).sum(-2))


def replicate_coords(D1: np.ndarray, D2: np.ndarray, V: np.ndarray, rng, folds: int = 5, repeats: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Each condition's two wells on axes found without its fold (repeats x conditions x axes, twice).

    Axes fit with a condition's cells carry its noise into both wells' coordinates, which inflates their agreement (r 0.74-0.94 on
    pure noise, every axis over the bar). So each fold's conditions are scored on axes refit without that fold, axis k taking its
    sign from V (a flip moves both wells together, so it cannot change whether they agree). replicate_r averages r over the repeats.

    Calibrated on pure noise at this study's shape (78 conditions, 20 axes; 2,085 and 300 genes): this gives mean r 0.00, 99th
    percentile 0.16-0.21, and no axis at 0.5. Leaving one condition out instead is badly behaved: r spreads from -0.9 to +0.5 and
    1-3% of noise axes pass 0.5, because the late axes are nearly tied and dropping one condition reorders them, with or without
    rotating the refit onto V. An axis that exists only because of a few conditions cannot be checked this way: read this with
    top_condition_share.
    """
    D = (D1 + D2) / 2
    w1, w2 = np.zeros((repeats, len(D), len(V))), np.zeros((repeats, len(D), len(V)))
    for rep in range(repeats):
        for fold in np.array_split(rng.permutation(len(D)), folds):
            W = np.linalg.svd(np.delete(D, fold, 0), full_matrices=False)[2][:len(V)]
            W = W * np.sign((W * V).sum(1, keepdims=True))
            w1[rep, fold], w2[rep, fold] = D1[fold] @ W.T, D2[fold] @ W.T
    return w1, w2


def replicate_r(w1: np.ndarray, w2: np.ndarray, rows=slice(None)) -> np.ndarray:
    return column_r(w1[:, rows], w2[:, rows]).mean(0)


def replicate_null(X: np.ndarray, wells: dict[str, list[np.ndarray]], k: int, rng, shuffles: int = N_SHUFFLE_R) -> np.ndarray:
    (a, b) = tuple(wells)
    out = []
    for _ in range(shuffles):
        D = shuffled_wells(X, wells, rng)[0]
        V = np.linalg.svd((D[a] + D[b]) / 2, full_matrices=False)[2][:k]
        out.append(replicate_r(*replicate_coords(D[a], D[b], V, rng)))
    return np.array(out)


def path_fit(C: np.ndarray, degree: int = PATH_DEGREE):
    """One curve through the coordinates: each as a polynomial of coordinate 1. Returns the function giving the curve at given points."""
    mean, sd = C[:, 0].mean(), C[:, 0].std()
    design = lambda X: np.vander((X[:, 0] - mean) / sd, degree + 1)
    beta = np.linalg.lstsq(design(C), C, rcond=None)[0]
    return lambda X: design(X) @ beta


def curve_share(C: np.ndarray, fit) -> float:
    """Share of the points' variance a curve holds."""
    return float(1 - ((C - fit(C)) ** 2).sum() / ((C - C.mean(0)) ** 2).sum())


def anova_shares(grid: np.ndarray) -> tuple[float, float, float]:
    """Shares of drug, dose and drug x dose in a balanced drugs x doses table."""
    centred = grid - grid.mean()
    drug = centred.mean(1, keepdims=True)
    dose = centred.mean(0, keepdims=True)
    interaction = centred - drug - dose
    total = (centred ** 2).sum()
    return float((drug ** 2).sum() * grid.shape[1] / total), float((dose ** 2).sum() * grid.shape[0] / total), float((interaction ** 2).sum() / total)


def curve_r2(x: np.ndarray, y: np.ndarray, degree: int = 4) -> float:
    """R² of y as a smooth function (polynomial) of x."""
    scaled = (x - x.mean()) / x.std()
    fit = np.polyval(np.polyfit(scaled, y, degree), scaled)
    return float(1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def pair_distances(C: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric == "cosine":
        unit = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        return 1 - unit @ unit.T
    return np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))


def nearest_share(D: np.ndarray, labels: np.ndarray, strata: np.ndarray, rng, perms: int = N_PERM) -> list[dict]:
    """Share of points whose nearest other point has the same label, against labels shuffled within strata."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    groups = [("all", np.ones(len(labels), bool))] + [(str(s), strata == s) for s in np.unique(strata)]
    observed = labels[nn] == labels
    null = np.zeros((perms, len(labels)), bool)
    code = pd.factorize(labels)[0]
    where = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for p in range(perms):
        shuffled = code.copy()
        for w in where:
            shuffled[w] = rng.permutation(code[w])
        null[p] = shuffled[nn] == shuffled
    rows = []
    for name, mask in groups:
        n = null[:, mask].mean(1)
        rows.append({"group": name, "n": int(mask.sum()), "observed": float(observed[mask].mean()), "null_mean": float(n.mean()),
                     "null_lo": float(np.quantile(n, 0.025)), "null_hi": float(np.quantile(n, 0.975)),
                     "p": float((1 + (n >= observed[mask].mean()).sum()) / (perms + 1))})
    return rows


def upper(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices(len(M), 1)]


def ranks(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank().to_numpy()


def mantel(dist: np.ndarray, sim: np.ndarray, rng, covariate: np.ndarray | None = None, perms: int = N_PERM) -> dict:
    """Spearman of a distance matrix with a similarity matrix, by shuffling drugs in the similarity matrix.

    With `covariate` (a matrix), the distance's ranks are first residualised on the covariate's ranks. rho < 0 means similar
    drugs are close. p is two-sided.
    """
    d = ranks(upper(dist))
    if covariate is not None:
        c = ranks(upper(covariate))
        c = c - c.mean()
        d = d - d.mean()
        d = d - c * (c @ d) / (c @ c)
    d = (d - d.mean()) / d.std()
    n = len(dist)

    def rho(order):
        s = ranks(upper(sim[np.ix_(order, order)]))
        return float((d * (s - s.mean())).mean() / s.std())

    observed = rho(np.arange(n))
    null = np.array([rho(rng.permutation(n)) for _ in range(perms)])
    return {"rho": observed, "p": float((1 + (np.abs(null) >= abs(observed)).sum()) / (perms + 1))}


def gene_set_scores(loadings: np.ndarray, members: np.ndarray) -> dict:
    """Rank test of one gene set along one axis: AUC (0.5 = no tilt) and z, within the panel's genes."""
    r = ranks(loadings)
    n1, n2 = int(members.sum()), int((~members).sum())
    u = r[members].sum() - n1 * (n1 + 1) / 2
    z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return {"auc": float(u / (n1 * n2)), "z": float(z)}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def structure_class(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    for name, smarts in STRUCTURES:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return name
    return "Other"


# ---- collect ---------------------------------------------------------------------------------------
def drug_table(data: Data, chemicals: str) -> tuple[pd.DataFrame, np.ndarray]:
    from rdkit import DataStructs
    from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

    meta = pd.read_csv(chemicals, sep=None, engine="python", usecols=["name", "SMILES"]).dropna().drop_duplicates()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, prints = [], []
    for drug in data.drugs:
        values = meta.loc[meta.name.str.strip() == drug.strip(), "SMILES"].unique()
        assert len(values) == 1, (drug, values)
        rows.append({"drug": drug, "short": short(drug), "selectivity": SELECTIVITY[drug], "structure": structure_class(values[0]),
                     "smiles": values[0]})
        prints.append(generator.GetFingerprint(MolFromSmiles(values[0])))
    tanimoto = np.array([DataStructs.BulkTanimotoSimilarity(p, prints) for p in prints])
    return pd.DataFrame(rows), tanimoto


def measured_table(data: Data, m0: np.ndarray, basis: np.ndarray) -> pd.DataFrame:
    """Measured coordinates and their sampling SE, for training and validation conditions only."""
    ctrl = (data.X[data.ctrl] - m0) @ basis.T
    ctrl_var = ctrl.var(0, ddof=1) / len(ctrl)
    rows = []
    for c, r in data.cond.iterrows():
        if r.split == "test":
            continue
        proj = (data.X[data.idx(c)] - m0) @ basis.T
        se = np.sqrt(proj.var(0, ddof=1) / len(proj) + ctrl_var)
        rows.append({"condition": c, "drug": r.drug, "dose": float(r.dose_nM), "split": r.split, "n_cells": len(proj),
                     **dict(zip(coord_names(), proj.mean(0))), **{f"se{i + 1}": s for i, s in enumerate(se)}})
    return pd.DataFrame(rows)


def axis_table(data: Data, ctx: Context, basis: np.ndarray, teacher: pd.DataFrame, measured: pd.DataFrame, per_seed: np.ndarray,
               keys: list[tuple[str, float]]) -> pd.DataFrame:
    names = coord_names()
    k = len(names)
    train = measured[measured.split == "train"]
    val = measured[measured.split == "val"]
    at = teacher.set_index(["drug", "dose"])
    fit_train = at.loc[list(zip(train.drug, train.dose)), names].to_numpy()
    fit_val = at.loc[list(zip(val.drug, val.dose)), names].to_numpy()
    grid = teacher[teacher.dose > 0].pivot_table(index="drug", columns="dose", values=names)
    out = pd.DataFrame({"axis": np.arange(1, k + 1)})

    C = train[names].to_numpy()
    out["share_measured"] = (C ** 2).sum(0) / (C ** 2).sum()
    all_points = teacher[teacher.dose > 0][names].to_numpy()
    out["share_teacher"] = (all_points ** 2).sum(0) / (all_points ** 2).sum()
    out["sd_teacher"] = all_points.std(0)
    out["median_se"] = np.median(train[[f"se{i + 1}" for i in range(k)]].to_numpy(), 0)

    # Replicates: each well against its own vehicle wells.
    control = data.split_cfg["controls"]["condition"]
    reps = ("rep1", "rep2")
    ctrl = {r: data.X[data.idx(control, r)] for r in reps}
    conds = list(train.condition)
    D = {r: np.array([data.rep_mean(c, r) - ctrl[r].mean(0) for c in conds]) for r in reps}
    cells = {r: [data.X[data.idx(c, r)] for c in conds] for r in reps}
    coord_12, noise_12, V1 = crossfit_parts(D["rep1"], D["rep2"], cells["rep2"], k)
    coord_21, noise_21, V2 = crossfit_parts(D["rep2"], D["rep1"], cells["rep1"], k)
    def share(i=slice(None)):
        return (2 - noise_12[i].mean(0) / coord_12[i].var(0, ddof=1) - noise_21[i].mean(0) / coord_21[i].var(0, ddof=1)) / 2
    out["signal_share"] = share()
    out["signal_share_rep1_axes"] = 1 - noise_12.mean(0) / coord_12.var(0, ddof=1)
    out["signal_share_rep2_axes"] = 1 - noise_21.mean(0) / coord_21.var(0, ddof=1)
    rng = np.random.default_rng(0)
    boot = np.array([share(rng.integers(0, len(conds), len(conds))) for _ in range(N_BOOT)])
    out["signal_lo"], out["signal_hi"] = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
    null = shuffled_null(data.X, {r: [data.idx(c, r) for c in conds] for r in reps}, k, rng)
    print("rep1 vs rep2 axes, cosines of the principal angles between the top-20 spaces:",
          np.round(np.linalg.svd(V1 @ V2.T, compute_uv=False), 2).tolist(), flush=True)
    out["null_mean"], out["null_q99"] = null.mean(0), np.quantile(null, 0.99, axis=0)
    print(f"shuffled labels, signal share over {len(null)} shuffles: mean {null.mean():.3f}, largest {null.max():.3f}; "
          f"share of axes at or above {REAL_SHARE}: {(null >= REAL_SHARE).mean():.4f}", flush=True)
    wells = {r: [data.idx(c, r) for c in conds] for r in reps}
    w1, w2 = replicate_coords(D["rep1"], D["rep2"], basis, rng)
    out["replicate_r"] = replicate_r(w1, w2)
    out["replicate_r_naive"] = column_r(D["rep1"] @ basis.T, D["rep2"] @ basis.T)  # biased upward; kept to show by how much
    boot = []
    for _ in range(N_BOOT):
        boot.append(replicate_r(w1, w2, rng.integers(0, len(conds), len(conds))))
    out["replicate_lo"], out["replicate_hi"] = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
    null_r = replicate_null(data.X, wells, k, rng)
    out["replicate_null_q99"] = np.quantile(null_r, 0.99, axis=0)
    print(f"shuffled labels, replicate r over {len(null_r)} shuffles: mean {null_r.mean():.3f}, largest {null_r.max():.3f}; "
          f"share of axes at or above {REAL_SHARE}: {(null_r >= REAL_SHARE).mean():.4f}", flush=True)
    squared = C ** 2
    out["top_condition"] = [conds[i] for i in squared.argmax(0)]
    out["top_condition_share"] = squared.max(0) / squared.sum(0)
    out["top3_share"] = np.sort(squared, 0)[-3:].sum(0) / squared.sum(0)
    out["cos_rep_axes"] = np.abs((V1 * V2).sum(1))
    out["cos_with_rep1"] = np.abs((V1 * basis).sum(1))
    out["cos_with_rep2"] = np.abs((V2 * basis).sum(1))
    coords = {r: D[r] @ basis.T for r in reps}
    out["split_half_r"] = [np.corrcoef(coords["rep1"][:, i], coords["rep2"][:, i])[0, 1] for i in range(k)]

    # Teacher against measured coordinates.
    M_train, M_val = train[names].to_numpy(), val[names].to_numpy()
    out["fit_train_r"] = [np.corrcoef(fit_train[:, i], M_train[:, i])[0, 1] for i in range(k)]
    out["fit_val_r"] = [np.corrcoef(fit_val[:, i], M_val[:, i])[0, 1] for i in range(k)]
    out["fit_val_rmse"] = np.sqrt(((fit_val - M_val) ** 2).mean(0))
    out["val_se"] = np.sqrt((val[[f"se{i + 1}" for i in range(k)]].to_numpy() ** 2).mean(0))

    # Do seeds agree, overall and at points no training mean pins down?
    seen = {(r.drug, r.dose) for r in train.itertuples()}
    unseen = np.array([q not in seen for q in keys])
    def agreement(P):
        return 1 - P.var(0, ddof=1).mean(0) / P.reshape(-1, k).var(0, ddof=1)
    out["seed_agreement"] = agreement(per_seed)
    out["seed_agreement_unseen"] = agreement(per_seed[:, unseen])

    shares = np.array([anova_shares(grid[n].to_numpy()) for n in names])
    out["drug_share"], out["dose_share"], out["interaction_share"] = shares.T
    first = teacher[teacher.dose > 0]["c1"].to_numpy()
    out["on_c1_teacher"] = [np.nan] + [curve_r2(first, teacher[teacher.dose > 0][n].to_numpy()) for n in names[1:]]
    out["on_c1_measured"] = [np.nan] + [curve_r2(train["c1"].to_numpy(), train[n].to_numpy()) for n in names[1:]]
    mean_abs = grid.abs().mean(0)
    out["peak_dose"] = [float(mean_abs[n].idxmax()) for n in names]
    out["program_share"] = (basis[:, data.spec.program_idx] ** 2).sum(1)
    symbols = np.array([str(g) for g in data.spec.gene_symbols])
    out["mito_share"] = (basis[:, np.char.startswith(symbols, "MT-")] ** 2).sum(1)
    abundance = pd.Series(data.X[data.ctrl].mean(0))
    out["abundance_r"] = [pd.Series(basis[i]).corr(abundance, method="spearman") for i in range(k)]

    # Technical checks. Within-condition differences between the two wells cannot be biology of the drug.
    depth = np.log10(data.adata.obs["n_umi"].to_numpy(dtype=float))
    plates = data.adata.obs["culture_plate"].astype(str).to_numpy()
    well_depth = {r: np.array([depth[data.idx(c, r)].mean() for c in conds]) for r in reps}
    d_coord = coords["rep1"] - coords["rep2"]
    d_depth = well_depth["rep1"] - well_depth["rep2"]
    out["depth_within_r"] = [np.corrcoef(d_coord[:, i], d_depth)[0, 1] for i in range(k)]
    mean_depth = (well_depth["rep1"] + well_depth["rep2"]) / 2
    out["depth_between_r"] = [np.corrcoef(M_train[:, i], mean_depth)[0, 1] for i in range(k)]
    control_cells = (data.X[data.ctrl] - ctx.m0) @ basis.T
    out["depth_control_r"] = [np.corrcoef(control_cells[:, i], depth[data.ctrl])[0, 1] for i in range(k)]
    out["control_var_ratio"] = control_cells.var(0, ddof=1) / data.X[data.ctrl].var(0, ddof=1).mean()
    out["log_cells_r"] = [np.corrcoef(M_train[:, i], np.log10(train.n_cells.to_numpy()))[0, 1] for i in range(k)]
    plate_of = {r: [plates[data.idx(c, r)[0]] for c in conds] for r in reps}
    out = pd.concat([out, pd.DataFrame(plate_effect(np.vstack([coords["rep1"], coords["rep2"]]), plate_of["rep1"] + plate_of["rep2"], len(conds)))], axis=1)
    return out


def plate_effect(well_coords: np.ndarray, plates: list[str], n_cond: int) -> list[dict]:
    """Well coordinate ~ condition + plate: share of the within-condition variance plate explains, with an F test."""
    from scipy import stats

    n = well_coords.shape[0]
    assert n == 2 * n_cond
    cond = np.vstack([np.eye(n_cond), np.eye(n_cond)])
    plate = pd.get_dummies(pd.Series(plates)).to_numpy(float)
    full = np.hstack([cond, plate])
    rank_full = np.linalg.matrix_rank(full)
    rows = []
    for i in range(well_coords.shape[1]):
        y = well_coords[:, i]
        rss = lambda X: ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()
        r0, r1 = rss(cond), rss(full)
        df1, df2 = rank_full - n_cond, n - rank_full
        f = (r0 - r1) / df1 / (r1 / df2)
        rows.append({"plate_partial_r2": float(1 - r1 / r0), "plate_p": float(stats.f.sf(f, df1, df2))})
    return rows


def gene_tables(data: Data, basis: np.ndarray, gene_sets: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy import stats

    symbols = np.array([str(s) for s in data.spec.gene_symbols])
    program = np.zeros(len(symbols), bool)
    program[data.spec.program_idx] = True
    gene_rows, set_rows = [], []
    for a in range(basis.shape[0]):
        order = np.argsort(basis[a])
        for side, part in (("negative", order[:25]), ("positive", order[::-1][:25])):
            gene_rows += [{"axis": a + 1, "side": side, "rank": i + 1, "gene": symbols[g], "loading": float(basis[a, g]),
                           "program": bool(program[g])} for i, g in enumerate(part)]
    for lib, label in LIBRARIES.items():
        path = gene_sets / f"{lib}.gmt"
        terms = []
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            members = np.isin(symbols, [p for p in parts[2:] if p])
            if members.sum() >= MIN_SET:
                terms.append((re.sub(r"\s*\(GO:\d+\)$", "", parts[0]), members))
        for a in range(basis.shape[0]):
            scored = [{"axis": a + 1, "library": label, "term": t, "n_in_panel": int(m.sum()), **gene_set_scores(basis[a], m)} for t, m in terms]
            p = 2 * stats.norm.sf(np.abs([s["z"] for s in scored]))
            q = benjamini_hochberg(p)
            for s, pi, qi, (_, m) in zip(scored, p, q, terms):
                idx = np.flatnonzero(m)
                lead = idx[np.argsort(-basis[a, idx] * np.sign(s["z"]))][:8]
                s.update(p=float(pi), q=float(qi), leading_genes=" ".join(symbols[lead]))
            set_rows += scored
    return pd.DataFrame(gene_rows), pd.DataFrame(set_rows)


def gene_means(data: Data, m0: np.ndarray, genes: pd.DataFrame, measured: pd.DataFrame, axes: int = 5) -> pd.DataFrame:
    """Measured change from vehicle of the genes that load most on the first axes, per training or validation condition."""
    position = {}
    for i, g in enumerate(str(s) for s in data.spec.gene_symbols):
        position.setdefault(g, i)
    keep = sorted(genes[genes.axis <= axes].gene.unique())
    idx = [position[g] for g in keep]
    rows = []
    for r in measured.itertuples():
        delta = data.mean(r.condition)[idx] - m0[idx]
        rows += [{"condition": r.condition, "drug": r.drug, "dose": r.dose, "c1": r.c1, "c2": r.c2, "gene": g, "delta": float(d)} for g, d in zip(keep, delta)]
    return pd.DataFrame(rows)


def inactive_pairs(data: Data, measured: pd.DataFrame, inactive: list[str], names: list[str]) -> pd.DataFrame:
    """Pairs of conditions of drugs with no response: distance by same drug and by culture plates shared."""
    plates = data.adata.obs["culture_plate"].astype(str).to_numpy()
    part = measured[measured.drug.isin(inactive)].reset_index(drop=True)
    plate = {r: [plates[data.idx(c, r)[0]] for c in part.condition] for r in ("rep1", "rep2")}
    D = pair_distances(part[names].to_numpy())
    i, j = np.triu_indices(len(part), 1)
    shared = sum((np.array(plate[r])[i] == np.array(plate[r])[j]).astype(int) for r in plate)
    return pd.DataFrame({"a": part.condition.to_numpy()[i], "b": part.condition.to_numpy()[j], "same_drug": part.drug.to_numpy()[i] == part.drug.to_numpy()[j],
                         "same_dose": part.dose.to_numpy()[i] == part.dose.to_numpy()[j], "shared_plates": shared, "distance": D[i, j]})


def distance_rows(table: pd.DataFrame, names: list[str], source: str) -> list[dict]:
    """Same drug at adjacent measured doses, and other drugs at the same dose."""
    rows = []
    at = {(r.drug, r.dose): np.array([getattr(r, n) for n in names]) for r in table.itertuples()}
    for (drug, dose), v in at.items():
        for lo, hi in zip(MEASURED_NM, MEASURED_NM[1:]):
            if dose == lo and (drug, hi) in at:
                rows.append({"source": source, "kind": "same_drug", "label": f"{lo:g}-{hi:g}", "a": drug, "b": drug,
                             "distance": float(np.linalg.norm(v - at[(drug, hi)]))})
        for (other, dose2), w in at.items():
            if dose2 == dose and other > drug:
                rows.append({"source": source, "kind": "other_drug", "label": f"{dose:g}", "a": drug, "b": other,
                             "distance": float(np.linalg.norm(v - w))})
    return rows


def neighbour_rows(table: pd.DataFrame, source: str, spaces: dict, rng) -> list[dict]:
    rows = []
    drug, dose = table.drug.to_numpy(), table.dose.to_numpy()
    for name, (columns, metric) in spaces.items():
        D = pair_distances(table[columns].to_numpy(), metric)
        rows += [{"source": source, "space": name, "same": "drug", **r} for r in nearest_share(D, drug, dose, rng)]
        rows += [{"source": source, "space": name, "same": "dose", **r} for r in nearest_share(D, dose, drug, rng)]
    return rows


def similarity_rows(drugs: pd.DataFrame, tanimoto: np.ndarray, spaces: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drug pairs at 10 µM: distance against class, structure and chemistry; raw and with the strength gap removed."""
    selectivity, structure = drugs.selectivity.to_numpy(), drugs.structure.to_numpy()
    same_class = (selectivity[:, None] == selectivity[None]).astype(float)
    same_structure = (structure[:, None] == structure[None]).astype(float)
    rows = []
    for (source, space, subset), (C, gap, keep) in spaces.items():
        D = pair_distances(C, "cosine" if "direction" in space else "euclidean")
        for name, sim in (("same_selectivity", same_class), ("same_structure", same_structure), ("tanimoto", tanimoto)):
            sim_part = sim[np.ix_(keep, keep)]
            if np.ptp(upper(sim_part)) == 0:
                continue
            within = upper(D)[upper(sim_part) == 1].mean() if name != "tanimoto" else np.nan
            between = upper(D)[upper(sim_part) == 0].mean() if name != "tanimoto" else np.nan
            rows.append({"source": source, "space": space, "subset": subset, "n_drugs": len(C), "similarity": name, "adjusted": "none",
                         "within": within, "between": between, **mantel(D, sim_part, rng)})
            rows.append({"source": source, "space": space, "subset": subset, "n_drugs": len(C), "similarity": name, "adjusted": "strength_gap",
                         "within": within, "between": between, **mantel(D, sim_part, rng, covariate=gap)})
    pair_table = pd.DataFrame(rows)
    return pair_table, same_class


def embeddings(points: pd.DataFrame, names: list[str], measured: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness

    X = points[names].to_numpy()
    Z = X / X.std(0)
    rows, info = [], {}

    def add(name, table, Y, reference):
        rows.append(pd.DataFrame({"embedding": name, "drug": table.drug.to_numpy(), "dose": table.dose.to_numpy(), "x": Y[:, 0], "y": Y[:, 1]}))
        info[name] = {"trustworthiness_k10": float(trustworthiness(reference, Y, n_neighbors=10))}

    add("coords_1_2", points, X[:, :2], X)
    for name, data in (("raw", X), ("standardised", Z)):
        fit = PCA(n_components=2, random_state=0).fit(data)
        add(f"pca_{name}", points, fit.transform(data), data)
        info[f"pca_{name}"]["variance"] = [float(v) for v in fit.explained_variance_ratio_]
        info[f"pca_{name}"]["cos_with_coords"] = [[float(c) for c in np.abs(component[:3])] for component in fit.components_]
        for seed in range(3):
            add(f"umap_{name}_s{seed}", points, umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=seed).fit_transform(data), data)
    # The control: the same UMAP on measured means, where a drug's doses share no code.
    M = pd.concat([measured[["drug", "dose", *names]], pd.DataFrame([{"drug": "Vehicle", "dose": 0.0, **dict.fromkeys(names, 0.0)}])], ignore_index=True)
    add("umap_measured_s0", M, umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=0).fit_transform(M[names].to_numpy()), M[names].to_numpy())
    return pd.concat(rows, ignore_index=True), info


def collect(panel: str, root: Path, chemicals: str, gene_sets: Path) -> None:
    out = out_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(Path(root) / "teacher" / "autodec.npz")
    keys = [tuple(json.loads(k)) for k in z["keys"]]
    coef, basis = z["coef"].astype(np.float64), z["basis"].astype(np.float64)
    data = Data(panel)
    ctx = Context(data, data.tasks("val")[0])
    assert np.abs(ctx.V[:K_BASIS] - basis).max() < 1e-5, "the saved basis is not this panel's"
    names = coord_names()
    split = {(r.drug, float(r.dose_nM)): r.split for r in data.cond.itertuples()}

    per_seed, seed_drugs, codes = autodecoder_seeds(ctx, keys, K_CODE, WD)
    refit_error = float(np.abs(per_seed.mean(0) - coef).max())
    print(f"refit reproduces the saved teacher to {refit_error:.4f} (largest coordinate difference; coordinates reach {np.abs(coef).max():.1f})", flush=True)
    assert refit_error < 0.05 * np.abs(coef).max(), "refit differs from the saved teacher"
    np.savez_compressed(out / "seeds.npz", per_seed=per_seed.astype(np.float32), drugs=np.array(seed_drugs), codes=codes.astype(np.float32))

    teacher = pd.DataFrame(coef, columns=names)
    teacher.insert(0, "drug", [q[0] for q in keys])
    teacher.insert(1, "dose", [float(q[1]) for q in keys])
    teacher.insert(2, "split", [split.get((q[0], float(q[1])), "grid") for q in keys])
    teacher = pd.concat([teacher, pd.DataFrame([{"drug": "Vehicle", "dose": 0.0, "split": "control", **dict.fromkeys(names, 0.0)}])], ignore_index=True)
    teacher.to_csv(out / "vectors.csv", index=False)

    measured = measured_table(data, ctx.m0, basis)
    measured.to_csv(out / "measured.csv", index=False)
    drugs, tanimoto = drug_table(data, chemicals)
    assert list(drugs.drug) == data.drugs
    at_top = teacher[teacher.dose == 10000].set_index("drug").loc[data.drugs]
    drugs["strength"] = at_top.c1.to_numpy()
    drugs["responder"] = drugs.strength >= RESPONDER * drugs.strength.max()
    drugs.drop(columns="smiles").to_csv(out / "drugs.csv", index=False)
    print(drugs[["short", "selectivity", "structure", "strength", "responder"]].to_string(), flush=True)

    axes = axis_table(data, ctx, basis, teacher, measured, per_seed, keys)
    axes.to_csv(out / "axes.csv", index=False)
    real = axes.replicate_r.to_numpy() >= REAL_SHARE
    k_real = int(np.argmin(real)) if not real.all() else len(real)  # leading run of real axes
    print(axes.drop(columns="top_condition").round(3).T.to_string(), flush=True)
    print(axes[["axis", "top_condition", "top_condition_share", "top3_share"]].round(3).to_string(), flush=True)

    # The curve: fit on the teacher's balanced grid + vehicle, and separately on the measured means + vehicle.
    vehicle = np.zeros((1, K_BASIS))
    teacher_points, measured_points = teacher[names].to_numpy(), np.vstack([measured[names].to_numpy(), vehicle])
    teacher_path, measured_path = path_fit(teacher_points), path_fit(measured_points)
    off_names = [f"off{i + 1}" for i in range(K_BASIS)]
    teacher_off = pd.concat([teacher, pd.DataFrame(teacher_points - teacher_path(teacher_points), columns=off_names)], axis=1)
    measured_off = pd.concat([measured.reset_index(drop=True), pd.DataFrame(measured[names].to_numpy() - measured_path(measured[names].to_numpy()), columns=off_names)], axis=1)
    teacher_off.to_csv(out / "vectors.csv", index=False)
    measured_off.to_csv(out / "measured.csv", index=False)
    span = np.linspace(teacher_points[:, 0].min(), teacher_points[:, 0].max(), 200)
    line = np.zeros((200, K_BASIS))
    line[:, 0] = span
    pd.DataFrame(np.c_[teacher_path(line)[:, :K_BASIS]], columns=names).to_csv(out / "path.csv", index=False)
    off = off_names[1:]

    np.savez_compressed(out / "loadings.npz", basis=basis.astype(np.float32), genes=np.array([str(s) for s in data.spec.gene_symbols]),
                        program_idx=data.spec.program_idx)
    genes, sets = gene_tables(data, basis, Path(gene_sets))
    genes.to_csv(out / "genes.csv", index=False)
    gene_means(data, ctx.m0, genes, measured).to_csv(out / "gene_means.csv", index=False)
    sets.to_csv(out / "genesets.csv", index=False)

    # Neighbourhoods: the same 87 conditions (training + validation) on both sides; all 104 for the teacher as well.
    rng = np.random.default_rng(0)
    shared = teacher_off.merge(measured[["drug", "dose"]], on=["drug", "dose"])
    all_conditions = teacher_off[teacher_off.dose.isin(MEASURED_NM)]
    spaces = {"all 20": (names, "euclidean"), "without c1": (names[1:], "euclidean"), "off path": (off, "euclidean"), "direction": (names, "cosine")}
    inactive = list(drugs.drug[drugs.strength < INACTIVE])
    print(f"negative control, {len(inactive)} drugs with no response:", [short(d) for d in inactive], flush=True)
    neighbours = pd.DataFrame(neighbour_rows(shared, "teacher", spaces, rng) + neighbour_rows(measured_off, "measured", spaces, rng)
                              + neighbour_rows(all_conditions, "teacher, all 104", spaces, rng)
                              + neighbour_rows(shared[shared.drug.isin(inactive)], "teacher, no response", spaces, rng)
                              + neighbour_rows(measured_off[measured_off.drug.isin(inactive)], "measured, no response", spaces, rng))
    neighbours.to_csv(out / "neighbours.csv", index=False)
    print(neighbours[(neighbours.group == "all") & (neighbours.same == "drug")].round(3).to_string(), flush=True)
    pairs_inactive = inactive_pairs(data, measured, inactive, names)
    pairs_inactive.to_csv(out / "inactive_pairs.csv", index=False)
    print(pairs_inactive.groupby(["same_drug", "shared_plates"]).distance.agg(["count", "median"]).round(3).to_string(), flush=True)

    # Replicate floor for measured distances: two half-size wells, rescaled to two full-size means.
    control = data.split_cfg["controls"]["condition"]
    floor_rows = []
    train = data.cond[data.cond.split == "train"]
    for c, r in train.iterrows():
        d = (data.rep_mean(c, "rep1") - data.rep_mean(control, "rep1")) - (data.rep_mean(c, "rep2") - data.rep_mean(control, "rep2"))
        floor_rows.append({"source": "measured", "kind": "floor", "label": f"{float(r.dose_nM):g}", "a": r.drug, "b": r.drug,
                           "distance": float(np.linalg.norm(d @ basis.T) / np.sqrt(2))})
    pd.DataFrame(distance_rows(shared, names, "teacher") + distance_rows(measured, names, "measured") + floor_rows).to_csv(out / "distances.csv", index=False)

    # Drug pairs at 10 µM, a training dose of every drug, so both sides have all 26.
    m_top = measured_off[measured_off.dose == 10000].set_index("drug").loc[data.drugs]
    t_top = teacher_off[teacher_off.dose == 10000].set_index("drug").loc[data.drugs]
    t, m = t_top[names].to_numpy(), m_top[names].to_numpy()
    t_off, m_off = t_top[off].to_numpy(), m_top[off].to_numpy()
    gap_t = np.abs(t[:, 0][:, None] - t[:, 0][None])
    gap_m = np.abs(m[:, 0][:, None] - m[:, 0][None])
    resp = drugs.responder.to_numpy()
    broad = (drugs.selectivity == BROAD).to_numpy()
    spaces = {}
    for source, C, O, gap in (("teacher", t, t_off, gap_t), ("measured", m, m_off, gap_m)):
        everyone = np.arange(len(C))
        spaces[(source, "all 20", "all drugs")] = (C, gap, everyone)
        spaces[(source, "off path", "all drugs")] = (O, gap, everyone)
        for subset, keep in (("responders", resp), ("broad only", broad)):
            where = np.flatnonzero(keep)
            part_gap = gap[np.ix_(where, where)]
            spaces[(source, "all 20", subset)] = (C[where], part_gap, where)
            spaces[(source, "off path", subset)] = (O[where], part_gap, where)
            spaces[(source, "direction", subset)] = (C[where], part_gap, where)
            spaces[(source, "off path direction", subset)] = (O[where], part_gap, where)
    similarity, _ = similarity_rows(drugs, tanimoto, spaces, rng)
    similarity.to_csv(out / "similarity.csv", index=False)
    print(similarity.round(3).to_string(), flush=True)

    i, j = np.triu_indices(len(drugs), 1)
    pairs = pd.DataFrame({"a": drugs.drug.to_numpy()[i], "b": drugs.drug.to_numpy()[j], "tanimoto": tanimoto[i, j], "strength_gap": gap_t[i, j],
                          "same_selectivity": drugs.selectivity.to_numpy()[i] == drugs.selectivity.to_numpy()[j],
                          "same_structure": drugs.structure.to_numpy()[i] == drugs.structure.to_numpy()[j],
                          "both_responders": resp[i] & resp[j],
                          "teacher_all": pair_distances(t)[i, j], "teacher_no_c1": pair_distances(t[:, 1:])[i, j], "teacher_off": pair_distances(t_off)[i, j],
                          "teacher_direction": pair_distances(t, "cosine")[i, j], "teacher_off_direction": pair_distances(t_off, "cosine")[i, j],
                          "measured_all": pair_distances(m)[i, j], "measured_no_c1": pair_distances(m[:, 1:])[i, j], "measured_off": pair_distances(m_off)[i, j],
                          "measured_direction": pair_distances(m, "cosine")[i, j], "measured_off_direction": pair_distances(m_off, "cosine")[i, j]})
    pairs.to_csv(out / "pairs.csv", index=False)

    embedded, info = embeddings(teacher, names, measured)
    embedded.to_csv(out / "embeddings.csv", index=False)
    grid_only = teacher[teacher.dose > 0][names].to_numpy()
    spectrum = np.linalg.svd(grid_only - grid_only.mean(0), compute_uv=False) ** 2
    meta = {"refit_error": refit_error, "real_share": REAL_SHARE, "k_real": k_real, "n_real": int(real.sum()), "responder_share": RESPONDER, "n_responders": int(resp.sum()),
            "inactive": inactive, "path_degree": PATH_DEGREE, "curve_share_teacher": curve_share(teacher_points, teacher_path),
            "curve_share_measured": curve_share(measured_points, measured_path),
            "c1_share_teacher": float(teacher_points[:, 0].var() / teacher_points.var(0).sum()),
            "c1_share_measured": float(measured_points[:, 0].var() / measured_points.var(0).sum()),
            "embeddings": info, "grid_variance_cumulative": np.round(np.cumsum(spectrum) / spectrum.sum(), 4).tolist(),
            "scale": float(z["scale"]), "n_perm": N_PERM,
            "libraries": list(LIBRARIES), "min_set": MIN_SET}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "embeddings"}, indent=1), flush=True)
    print(json.dumps(info, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


# ---- draw ------------------------------------------------------------------------------------------
class Saved:
    def __init__(self, root: Path):
        self.dir = out_dir(root)
        read = lambda name: pd.read_csv(self.dir / f"{name}.csv")
        self.vectors, self.measured, self.drugs, self.axes = read("vectors"), read("measured"), read("drugs"), read("axes")
        self.genes, self.sets, self.distances, self.neighbours = read("genes"), read("genesets"), read("distances"), read("neighbours")
        self.pairs, self.similarity, self.embedded = read("pairs"), read("similarity"), read("embeddings")
        self.path, self.gene_means = read("path"), read("gene_means")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.names = coord_names()
        self.klass = dict(zip(self.drugs.drug, self.drugs.selectivity))
        self.label = dict(zip(self.drugs.drug, self.drugs.short))
        order = self.drugs.assign(k=self.drugs.selectivity.map(CLASSES.index)).sort_values(["k", "strength"], ascending=[True, False])
        self.order = list(order.drug)


def class_handles(plt):
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker=CLASS_MARKER[c], color=CLASS_COLOR[c], lw=0, markersize=6.5, label=c) for c in CLASSES]


DOSE_SIZE = {10.0: 12, 100.0: 26, 1000.0: 48, 10000.0: 80}


def dose_handles(plt):
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="o", color=MUTED, lw=0, markersize=np.sqrt(s), label=f"{d:,.0f} nM") for d, s in DOSE_SIZE.items()]


def drug_paths(ax, ctx, cx: str, cy: str, labels=()) -> None:
    v = ctx.vectors[ctx.vectors.dose > 0]
    for drug in reversed(ctx.order):
        part = v[v.drug == drug].sort_values("dose")
        c = CLASS_COLOR[ctx.klass[drug]]
        ax.plot(part[cx], part[cy], color=c, lw=1.0, alpha=0.7, zorder=2)
        real = part[part.dose.isin(MEASURED_NM)]
        ax.scatter(real[cx], real[cy], s=real.dose.map(DOSE_SIZE), marker=CLASS_MARKER[ctx.klass[drug]], color=c, edgecolor="#fcfcfb", linewidth=0.5, zorder=3)
    ax.scatter([0], [0], marker="*", s=130, color=INK, zorder=4, edgecolor="#fcfcfb", linewidth=0.5)


def label_top(ax, ctx, cx: str, cy: str, drugs, offset=(5, 3)) -> None:
    v = ctx.vectors[ctx.vectors.dose == 10000].set_index("drug")
    for d in drugs:
        ax.annotate(ctx.label[d], (v.loc[d, cx], v.loc[d, cy]), xytext=offset, textcoords="offset points", fontsize=7.5, color=INK_2)


def fig_map(plt, ctx) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.9))
    margins(fig, 0.6, 0.25, top=1.0, bottom=1.25, wspace=0.17)
    top = ctx.vectors[ctx.vectors.dose == 10000].set_index("drug")
    drug_paths(axes[0], ctx, "c1", "c2")
    drug_paths(axes[1], ctx, "c2", "c3")
    axes[0].plot(ctx.path.c1, ctx.path.c2, color=INK, lw=1.6, ls="--", zorder=5)
    axes[1].plot(ctx.path.c2, ctx.path.c3, color=INK, lw=1.6, ls="--", zorder=5)
    named = [d for d in ctx.order if ctx.label[d] in ("CUDC-101", "Resminostat", "Entinostat", "Mocetinostat", "Tacedinaline", "Tucidinostat")]
    label_top(axes[0], ctx, "c1", "c2", named)
    axes[0].annotate("Most broad inhibitors\nend up here at 10 µM", (12.2, 0.2), xytext=(7.6, 1.6), fontsize=8.5, color=INK_2,
                     arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    axes[0].annotate("Vehicle, and the drugs\nwith no response", (0.1, 0.1), xytext=(1.2, 1.1), fontsize=8.5, color=INK_2,
                     arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    label_top(axes[1], ctx, "c2", "c3", list(top.c3.abs().nlargest(6).index))
    axes[0].set(xlabel="Coordinate 1", ylabel="Coordinate 2")
    axes[1].set(xlabel="Coordinate 2", ylabel="Coordinate 3")
    axes[0].set_title(f"Coordinates 1 and 2. Dashed: one smooth curve, holding {ctx.meta['curve_share_teacher']:.0%} of all 20 coordinates' variance")
    axes[1].set_title("Coordinates 2 and 3: the same path, seen with coordinate 1 hidden")
    first = fig.legend(handles=class_handles(plt), loc="lower left", bbox_to_anchor=(0.05, 0.06), ncol=5, title="Selectivity class", alignment="left", fontsize=8.5, title_fontsize=8.5)
    fig.add_artist(first)
    fig.legend(handles=dose_handles(plt), loc="lower left", bbox_to_anchor=(0.55, 0.06), ncol=4, title="Dose (lines run through the embedding's dose grid)", alignment="left", fontsize=8.5, title_fontsize=8.5)
    finish(plt, fig, ctx.dir / "1_map.png", "Drugs do not form clusters: they follow one curved path and differ in how far they get",
           "The condition response embedding of all 26 drugs (its 20 numbers). Star: vehicle (all zeros). Test conditions appear from labels only.")


def fig_each_drug(plt, ctx) -> None:
    fig, axes = plt.subplots(5, 6, figsize=(14, 11.6), sharex=True, sharey=True)
    margins(fig, 0.6, 0.2, top=0.85, bottom=0.7, wspace=0.08, hspace=0.25)
    v = ctx.vectors[ctx.vectors.dose > 0]
    for ax in axes.flat[len(ctx.order):]:
        ax.set_visible(False)
    for ax, drug in zip(axes.flat, ctx.order):
        for other in ctx.order:
            part = v[v.drug == other].sort_values("dose")
            ax.plot(part.c1, part.c2, color=GRID, lw=0.7, zorder=1)
        part = v[v.drug == drug].sort_values("dose")
        c = CLASS_COLOR[ctx.klass[drug]]
        ax.plot(part.c1, part.c2, color=c, lw=1.8, zorder=3)
        real = part[part.dose.isin(MEASURED_NM)]
        ax.scatter(real.c1, real.c2, s=real.dose.map(DOSE_SIZE) * 0.55, color=c, marker=CLASS_MARKER[ctx.klass[drug]], zorder=4, edgecolor="#fcfcfb", linewidth=0.4)
        seen = ctx.measured[ctx.measured.drug == drug]
        ax.scatter(seen.c1, seen.c2, s=seen.dose.map(DOSE_SIZE) * 0.55, facecolor="none", edgecolor=INK, linewidth=0.7, zorder=5)
        ax.set_title(ctx.label[drug], fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Coordinate 1", fontsize=8.5)
        ax.tick_params(labelbottom=True)
    for row in axes:
        row[0].set_ylabel("Coordinate 2", fontsize=8.5)
    finish(plt, fig, ctx.dir / "2_each_drug.png", "One panel per drug: the embedding's path against what was measured",
           "Colored path: embedding. Black rings: measured means of training and validation doses (test means are not read). Gray: the other drugs. Ordered by class, then strength.")


def fig_embeddings(plt, ctx) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    margins(fig, 0.45, 0.25, top=0.95, bottom=1.1, wspace=0.12, hspace=0.22)
    info = ctx.meta["embeddings"]
    panels = [("umap_raw_s0", "UMAP of the embedding, seed 0", "class"), ("umap_raw_s0", "The same UMAP, by dose", "dose"), ("umap_measured_s0", "Control: UMAP of measured means", "class"),
              ("umap_raw_s1", "UMAP of the embedding, seed 1", "class"), ("pca_raw", "PCA of the embedding", "class"), ("pca_standardised", "PCA, every coordinate scaled to SD 1", "class")]
    shade = {d: RAMP[i] for i, d in enumerate((10.0, 31.6, 100.0, 1000.0, 10000.0))}
    for ax, (name, title, colour) in zip(axes.flat, panels):
        part = ctx.embedded[ctx.embedded.embedding == name]
        for drug in reversed(ctx.order):
            p = part[part.drug == drug].sort_values("dose")
            c = CLASS_COLOR[ctx.klass[drug]] if colour == "class" else GRID
            ax.plot(p.x, p.y, color=c, lw=0.8, alpha=0.65, zorder=2)
            if colour == "class":
                ax.scatter(p.x, p.y, s=16, marker=CLASS_MARKER[ctx.klass[drug]], color=c, edgecolor="#fcfcfb", linewidth=0.3, zorder=3)
            else:
                keep = p[p.dose.isin(shade)]
                ax.scatter(keep.x, keep.y, s=18, color=keep.dose.map(shade), edgecolor=AXIS, linewidth=0.3, zorder=3)
        vehicle = part[part.drug == "Vehicle"]
        ax.scatter(vehicle.x, vehicle.y, marker="*", s=120, color=INK, zorder=4, edgecolor="#fcfcfb", linewidth=0.5)
        ax.set_title(f"{title}\nneighbours kept (k = 10): {info[name]['trustworthiness_k10']:.2f}", fontsize=9.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)
    from matplotlib.lines import Line2D
    fig.legend(handles=class_handles(plt), loc="lower left", bbox_to_anchor=(0.03, 0.045), ncol=5, title="Selectivity class", alignment="left", fontsize=8.5, title_fontsize=8.5)
    shades = [Line2D([], [], marker="o", color=RAMP[i], markeredgecolor=AXIS, lw=0, markersize=6, label=l) for i, l in enumerate(("10 nM", "31.6 nM", "100 nM", "1,000 nM", "10,000 nM"))]
    fig.legend(handles=shades, loc="lower left", bbox_to_anchor=(0.52, 0.045), ncol=5, title="Dose (second panel)", alignment="left", fontsize=8.5, title_fontsize=8.5)
    finish(plt, fig, ctx.dir / "3_embeddings.png", "UMAP and PCA of the condition response embedding",
           "Embedding: 182 grid points + vehicle (star). Lines join one drug's doses. UMAP has no units: the weak drugs' islands are about as far from vehicle as two wells of one condition.")


def fig_heatmap(plt, ctx) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    v = ctx.vectors[ctx.vectors.dose.isin(MEASURED_NM)]
    rows = pd.concat([v[v.drug == d].sort_values("dose") for d in ctx.order], ignore_index=True)
    sd = ctx.vectors[ctx.vectors.dose > 0][ctx.names].std()
    M = (rows[ctx.names] / sd).to_numpy()
    cmap = LinearSegmentedColormap.from_list("diverging", [NEGATIVE, MIDPOINT, POSITIVE])
    fig, (top, ax) = plt.subplots(2, 1, figsize=(10.5, 15), gridspec_kw={"height_ratios": [1, 12]}, sharex=True)
    margins(fig, 2.2, 1.15, top=0.95, bottom=0.65, hspace=0.04)
    top.bar(np.arange(20), ctx.axes.replicate_r.clip(lower=0), color=CLASS_COLOR[BROAD], width=0.7)
    top.axhline(ctx.meta["real_share"], color=INK_2, lw=0.8, ls="--")
    top.set(ylabel="Two wells\nagree (r)", ylim=(0, 1))
    top.grid(axis="x", visible=False)
    im = ax.imshow(M, cmap=cmap, vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(np.arange(20), [str(i + 1) for i in range(20)])
    ax.set_xlabel("Coordinate")
    centres, previous = [], None
    for i, drug in enumerate(rows.drug):
        if drug != previous and i:
            ax.axhline(i - 0.5, color="#fcfcfb", lw=1.6)
        previous = drug
    for drug in ctx.order:
        idx = np.flatnonzero(rows.drug == drug)
        centres.append(idx.mean())
    ax.set_yticks(centres, [f"{ctx.label[d]}" for d in ctx.order], fontsize=8.5)
    for drug, y in zip(ctx.order, centres):
        ax.scatter(-0.95, y, marker=CLASS_MARKER[ctx.klass[drug]], color=CLASS_COLOR[ctx.klass[drug]], s=26, clip_on=False, zorder=5)
    ax.tick_params(axis="y", pad=24)
    ax.set_xlim(-0.5, 19.5)
    heldout = np.flatnonzero(rows.split.isin(["val", "test"]))
    ax.scatter(np.full(len(heldout), 19.85), heldout, marker="<", s=12, color=INK_2, clip_on=False, zorder=5)
    cax = fig.add_axes([1 - 0.75 / fig.get_figwidth(), 0.3, 0.13 / fig.get_figwidth(), 0.35])
    fig.colorbar(im, cax=cax, label="Coordinate / its SD")
    cax.tick_params(labelsize=8)
    finish(plt, fig, ctx.dir / "4_heatmap.png", "Every condition's 20 numbers",
           "Rows: 10, 100, 1,000, 10,000 nM within each drug. Arrowheads: held-out doses. Columns scaled by SD, so weak ones look as loud as strong ones.")


def fig_distances(plt, ctx) -> None:
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14, 9.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], left=0.055, right=0.985, top=0.875, bottom=0.075, hspace=0.34, wspace=0.17)
    same, other = CLASS_COLOR[BROAD], CLASS_COLOR[CLASS1]
    categories = [("same_drug", "10-100", "10 to\n100 nM"), ("same_drug", "100-1000", "100 to\n1,000"), ("same_drug", "1000-10000", "1,000 to\n10,000"),
                  ("other_drug", "10", "10 nM"), ("other_drug", "100", "100"), ("other_drug", "1000", "1,000"), ("other_drug", "10000", "10,000")]
    rng = np.random.default_rng(0)
    ax = fig.add_subplot(grid[0, :])
    gap = 8.6
    for s, source in enumerate(("teacher", "measured")):
        part = ctx.distances[ctx.distances.source == source]
        base = s * gap
        for i, (kind, label, _) in enumerate(categories):
            d = part[(part.kind == kind) & (part.label.astype(str) == label)].distance.to_numpy()
            if not len(d):
                continue
            x = base + i + rng.uniform(-0.2, 0.2, len(d))
            ax.scatter(x, d, s=9, color=same if kind == "same_drug" else other, alpha=0.55, linewidth=0, zorder=3)
            ax.plot([base + i - 0.3, base + i + 0.3], [np.median(d)] * 2, color=INK, lw=1.8, zorder=4)
    floor = ctx.distances[ctx.distances.kind == "floor"].distance.median()
    ax.plot([gap - 0.5, gap + 6.5], [floor, floor], color=INK_2, lw=1, ls="--", zorder=2)
    ax.annotate("Two wells of one condition", (gap + 6.5, floor), xytext=(0, -3), textcoords="offset points", ha="right", va="top", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_xticks([b + i for b in (0, gap) for i in range(7)], [c[2] for _ in (0, 1) for c in categories], fontsize=8.5)
    ax.set_xlim(-0.7, 2 * gap - 0.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Distance over the 20 coordinates")
    ax.set_title("Pair distances: embedding (left), measured means (right)")
    ax.legend(handles=[Line2D([], [], marker="o", color=same, lw=0, label="Same drug, adjacent doses"), Line2D([], [], marker="o", color=other, lw=0, label="Different drugs, same dose"),
                       Line2D([], [], color=INK, lw=1.8, label="Median")], loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8.5)
    panels = [("all 20", ("teacher", "measured"), "All 20 coordinates"), ("off path", ("teacher", "measured"), "Off the path (strength removed)"),
              ("all 20", ("teacher, no response", "measured, no response"), "Drugs with no response")]
    groups = ["all", "10.0", "100.0", "1000.0", "10000.0"]
    for i, (space, sources, title) in enumerate(panels):
        a = fig.add_subplot(grid[1, i])
        part = ctx.neighbours[(ctx.neighbours.space == space) & (ctx.neighbours.same == "drug")]
        for s, (source, c, m, name) in enumerate(((sources[0], same, "o", "Embedding"), (sources[1], INK, "D", "Measured"))):
            p = part[part.source == source].set_index("group").loc[groups]
            x = np.arange(5) + (s - 0.5) * 0.28
            a.vlines(x, p.null_lo, p.null_hi, color=AXIS, lw=5, zorder=2)
            a.scatter(x, p.observed, color=c, marker=m, s=36, zorder=4, label=name, edgecolor="#fcfcfb", linewidth=0.5)
        a.set_xticks(np.arange(5), ["All", "10 nM", "100", "1,000", "10,000"], fontsize=8.5)
        a.set_ylim(-0.02, 1.02)
        a.grid(axis="x", visible=False)
        a.set_title(f"Same drug is the nearest neighbour\n{title}", fontsize=9.2)
        if i == 0:
            a.set_ylabel("Share of conditions")
            a.legend(loc="upper left", fontsize=8.5)
    finish(plt, fig, ctx.dir / "5_are_close_conditions_close.png", "Are close conditions close?",
           "87 training and validation conditions on both sides. Gray bars: 95% range with drug labels shuffled within each dose. Same-drug closeness is partly built in for the embedding.")


def fig_similarity(plt, ctx) -> None:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    names = list(ctx.drugs.drug)
    index = {d: i for i, d in enumerate(names)}

    def matrix(column):
        M = np.zeros((len(names), len(names)))
        for r in ctx.pairs.itertuples():
            M[index[r.a], index[r.b]] = M[index[r.b], index[r.a]] = getattr(r, column)
        return M

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.0), gridspec_kw={"width_ratios": [1, 1, 0.82]})
    margins(fig, 1.1, 0.35, top=0.95, bottom=1.0, wspace=0.42)
    for ax, column, title in ((axes[0], "teacher_all", "All 20 coordinates"), (axes[1], "teacher_off", "Off the path")):
        M = matrix(column)
        order = leaves_list(linkage(squareform(M, checks=False), "average"))
        im = ax.imshow(M[np.ix_(order, order)], cmap=_ramp(), aspect="equal", interpolation="nearest", vmin=0)
        ax.grid(False)
        labels = [ctx.label[names[i]] for i in order]
        ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7.2)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.2)
        ax.tick_params(axis="both", pad=11)
        for k, i in enumerate(order):
            kl = ctx.klass[names[i]]
            ax.scatter(-1.05, k, marker=CLASS_MARKER[kl], color=CLASS_COLOR[kl], s=14, clip_on=False, zorder=5)
            ax.scatter(k, len(order) - 1 + 1.05, marker=CLASS_MARKER[kl], color=CLASS_COLOR[kl], s=14, clip_on=False, zorder=5)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_title(f"{title}: distance between drugs at 10 µM")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=8)
    ax = axes[2]
    part = ctx.pairs[ctx.pairs.both_responders]
    same = part.same_selectivity
    ax.scatter(part.tanimoto[~same], part.teacher_off_direction[~same], s=26, color=CLASS_COLOR[CLASS1], marker="s", edgecolor="#fcfcfb", linewidth=0.4, label="Broad vs class I")
    ax.scatter(part.tanimoto[same], part.teacher_off_direction[same], s=24, color=CLASS_COLOR[BROAD], edgecolor="#fcfcfb", linewidth=0.4, label="Same class")
    row = ctx.similarity[(ctx.similarity.source == "teacher") & (ctx.similarity.space == "off path direction") & (ctx.similarity.subset == "responders")
                         & (ctx.similarity.similarity == "tanimoto") & (ctx.similarity.adjusted == "none")].iloc[0]
    ax.set(xlabel="Chemical similarity (Tanimoto, Morgan fingerprints)", ylabel="Angle between off-path vectors (1 - cosine)")
    ax.set_title(f"The {ctx.meta['n_responders']} responders\nSpearman {row.rho:.2f} (p = {row.p:.2g})", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.legend(handles=class_handles(plt), loc="lower left", bbox_to_anchor=(0.05, 0.015), ncol=5, title="Selectivity class", alignment="left", fontsize=8.5, title_fontsize=8.5)
    finish(plt, fig, ctx.dir / "6_drug_similarity.png", "Do similar drugs get similar vectors?",
           "Embedding at 10 µM, a training dose of every drug, so it nearly equals the measured means. Matrices clustered separately. The embedding never sees structure or class.")


def _ramp():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sequential", ["#fcfcfb"] + RAMP)


def fig_quality(plt, ctx) -> None:
    a = ctx.axes
    y = np.arange(20)
    fig, axes = plt.subplots(1, 8, figsize=(18.5, 6.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.1, 1, 1, 1.2, 1, 1, 1]})
    margins(fig, 0.75, 0.2, top=1.05, bottom=0.8, wspace=0.13)
    blue = CLASS_COLOR[BROAD]
    unit = [0, 0.5, 1]

    def bars(ax, values, title, xlim=None, color=blue, xlabel="", ticks=unit):
        ax.barh(y, values, color=color, height=0.68)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_xticks(ticks)

    bars(axes[0], a.share_measured, "Share of squared\nlength (measured)", xlabel="Log scale")
    axes[0].set_xscale("log")
    bars(axes[1], a.replicate_r.clip(lower=0), "Two wells agree (r),\naxis refit without them", (0, 1), xlabel="Dashed: counted as real")
    axes[1].axvline(ctx.meta["real_share"], color=INK_2, lw=0.9, ls="--")
    axes[1].hlines(y, a.replicate_lo.clip(0, 1), a.replicate_hi.clip(0, 1), color=INK, lw=1.1, zorder=3)
    axes[1].scatter(a.replicate_null_q99.clip(0, 1), y, marker="|", s=70, color=CLASS_COLOR[CLASS1], linewidth=1.8, zorder=4, label="Shuffled, 99th pct.")
    axes[1].legend(fontsize=7.5, loc="lower right")
    for column, label, c, m in (("fit_train_r", "Training (fit)", blue, "o"), ("fit_val_r", "Validation (9)", INK, "D")):
        axes[2].scatter(a[column], y, s=26, color=c, label=label, marker=m, edgecolor="#fcfcfb", linewidth=0.4, zorder=3)
    axes[2].set_title("Embedding vs measured\ncoordinate (r)", fontsize=9.5)
    axes[2].set_xlim(-1, 1.05)
    axes[2].set_xticks([-1, 0, 1])
    axes[2].axvline(0, color=AXIS, lw=0.8)
    axes[2].legend(fontsize=7.5, loc="lower left")
    axes[2].grid(axis="y", visible=False)
    bars(axes[3], a.seed_agreement_unseen.clip(lower=0), "Embedding seeds agree\n(unseen doses)", (0, 1))
    left = np.zeros(20)
    for column, label, c in (("dose_share", "Dose", INK_2), ("drug_share", "Drug", blue), ("interaction_share", "Drug x dose", CLASS_COLOR[CLASS1])):
        axes[4].barh(y, a[column], left=left, color=c, height=0.68, label=label, edgecolor="#fcfcfb", linewidth=1.0)
        left += a[column].to_numpy()
    axes[4].set_xlim(0, 1)
    axes[4].set_xticks(unit)
    axes[4].set_title("What moves it\n(embedding's grid)", fontsize=9.5)
    axes[4].legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.05), columnspacing=0.8, handlelength=1)
    axes[4].grid(axis="y", visible=False)
    axes[5].scatter(a.on_c1_teacher, y, s=26, color=blue, label="Embedding", edgecolor="#fcfcfb", linewidth=0.4, zorder=3)
    axes[5].scatter(a.on_c1_measured, y, s=24, color=INK, marker="D", label="Measured", edgecolor="#fcfcfb", linewidth=0.4, zorder=3)
    axes[5].set_xlim(-0.02, 1.02)
    axes[5].set_xticks(unit)
    axes[5].set_title("A curve of coordinate 1\n(R² of smooth fit)", fontsize=9.5)
    axes[5].legend(fontsize=7.5, loc="lower right")
    axes[5].grid(axis="y", visible=False)
    bars(axes[6], a.top_condition_share, "Largest single\ncondition's share", (0, 1))
    axes[7].scatter(a.depth_within_r.abs(), y, s=26, color=INK, marker="D", edgecolor="#fcfcfb", linewidth=0.4, zorder=3, label="Between a\ncondition's wells")
    axes[7].scatter(a.depth_control_r.abs(), y, s=26, color=blue, edgecolor="#fcfcfb", linewidth=0.4, zorder=3, label="Vehicle cells")
    axes[7].set_xlim(-0.02, 1.02)
    axes[7].set_xticks(unit)
    axes[7].set_title("Moves with sequencing\ndepth (|r|)", fontsize=9.5)
    axes[7].legend(fontsize=7.5, loc="lower right")
    axes[7].grid(axis="y", visible=False)
    axes[0].set_yticks(y, [str(i + 1) for i in y])
    axes[0].set_ylabel("Coordinate")
    axes[0].invert_yaxis()
    finish(plt, fig, ctx.dir / "7_axis_quality.png", "Which coordinates carry something real?",
           "Two wells agree: a condition's two wells, on the axis refit without its fold (5 folds x 5 repeats). About 0 means it does not replicate. Whiskers: 95% CI over conditions.")


def fig_axis_curves(plt, ctx) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(15, 10.6), sharex=True)
    margins(fig, 0.55, 0.2, top=0.92, bottom=1.05, wspace=0.24, hspace=0.3)
    v = ctx.vectors[ctx.vectors.dose > 0]
    ticks = [10, 100, 1000, 10000]
    for i, ax in enumerate(axes.flat):
        name = f"c{i + 1}"
        for drug in reversed(ctx.order):
            part = v[v.drug == drug].sort_values("dose")
            ax.plot(part.dose, part[name], color=CLASS_COLOR[ctx.klass[drug]], lw=0.9, alpha=0.75)
        extreme = v.loc[v[name].abs().idxmax()]
        low, high = v[name].min(), v[name].max()
        ax.set_ylim(low - 0.13 * (high - low), high + 0.13 * (high - low))
        ax.annotate(ctx.label[extreme.drug], (extreme.dose, extreme[name]), xytext=(4 if extreme.dose < 300 else -4, 3 if extreme[name] > 0 else -3), textcoords="offset points",
                    ha="left" if extreme.dose < 300 else "right", va="bottom" if extreme[name] > 0 else "top", fontsize=7, color=INK_2)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ticks, ["10", "100", "1k", "10k"])
        ax.minorticks_off()
        ax.set_title(f"Coordinate {i + 1}  (wells agree r = {ctx.axes.replicate_r[i]:.2f})", fontsize=9.5)
        if i >= 15:
            ax.set_xlabel("Dose (nM)", fontsize=8.5)
    fig.legend(handles=[plt.Line2D([], [], color=CLASS_COLOR[c], lw=2, label=c) for c in CLASSES], loc="lower left", bbox_to_anchor=(0.04, 0.012), ncol=5, title="Selectivity class", alignment="left", fontsize=8.5, title_fontsize=8.5)
    finish(plt, fig, ctx.dir / "8_axis_dose_curves.png", "How each coordinate responds to dose, drug by drug",
           "The embedding on its dose grid. Each panel has its own y scale: coordinate 1 runs to tens and the late ones stay near zero. Label: the drug furthest from zero.")


def wrap_term(term: str, width: int = 44) -> str:
    return term if len(term) <= width else term[:width - 1].rstrip() + "…"


def fig_axis_genes(plt, ctx) -> None:
    n = 6
    fig, axes = plt.subplots(2, n // 2, figsize=(15.5, 12.4))
    margins(fig, 0.95, 0.25, top=0.85, bottom=0.6, wspace=0.42, hspace=0.46)
    for a, ax in zip(range(1, n + 1), axes.flat):
        part = ctx.genes[(ctx.genes.axis == a) & (ctx.genes["rank"] <= 9)]
        neg = part[part.side == "negative"].sort_values("loading")
        pos = part[part.side == "positive"].sort_values("loading")
        rows = pd.concat([neg, pos])
        y = np.arange(len(rows))
        ax.barh(y, rows.loading, color=np.where(rows.loading > 0, POSITIVE, NEGATIVE), height=0.7)
        ax.set_yticks(y, rows.gene, fontsize=7.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Coordinate {a}  (wells agree r = {ctx.axes.replicate_r[a - 1]:.2f}, program genes {ctx.axes.program_share[a - 1]:.0%} of weight)", fontsize=9.5)
        sets = ctx.sets[(ctx.sets.axis == a) & (ctx.sets.q < 0.05) & (ctx.sets.library == "Hallmark")]
        up = sets.nlargest(2, "z")
        down = sets.nsmallest(2, "z")
        up = up[up.z > 0]
        down = down[down.z < 0]
        lines = [f"+  {wrap_term(t)}" for t in up.term] + [f"−  {wrap_term(t)}" for t in down.term]
        ax.set_xlabel("Hallmark: " + ("; ".join(lines) if lines else "none at q < 0.05"), fontsize=7.6, color=INK_2, wrap=True)
    finish(plt, fig, ctx.dir / "9_axis_genes.png", "The genes behind the first six coordinates",
           "Nine strongest loadings on each side. The sign of coordinates 2-20 is arbitrary, so + and − only say which genes move together.")


def fig_path_genes(plt, ctx) -> None:
    """Coordinate 2 in measured genes: no model, each point a measured condition mean."""
    top = ctx.genes[ctx.genes.axis == 2]
    negative = list(top[top.side == "negative"].sort_values("rank").gene[:4])
    positive = list(top[top.side == "positive"].sort_values("rank").gene[:4])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4), sharex=True)
    margins(fig, 0.6, 0.2, top=0.92, bottom=0.65, wspace=0.2, hspace=0.3)
    for ax, gene in zip(axes.flat, negative + positive):
        part = ctx.gene_means[ctx.gene_means.gene == gene]
        for k in reversed(CLASSES):
            p = part[part.drug.map(ctx.klass) == k]
            ax.scatter(p.c1, p.delta, s=15, marker=CLASS_MARKER[k], color=CLASS_COLOR[k], edgecolor="#fcfcfb", linewidth=0.3, zorder=3)
        x = (part.c1 - part.c1.mean()) / part.c1.std()
        span = np.linspace(x.min(), x.max(), 100)
        ax.plot(span * part.c1.std() + part.c1.mean(), np.polyval(np.polyfit(x, part.delta, PATH_DEGREE), span), color=INK, lw=1.4, zorder=4)
        ax.axhline(0, color=AXIS, lw=0.8)
        loading = top.set_index("gene").loading[gene]
        ax.set_title(f"{gene}  (coordinate 2 loading {loading:+.2f})", fontsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("How far along the path (measured coordinate 1)", fontsize=8.5)
    for row in axes:
        row[0].set_ylabel("Change from vehicle (log expression)", fontsize=8.5)
    fig.legend(handles=class_handles(plt), loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=5, fontsize=8.5)
    finish(plt, fig, ctx.dir / "10_path_genes.png", "What coordinate 2 means, in measured genes",
           "The four strongest genes on each side of coordinate 2. Each point is one condition's measured mean (training and validation). Line: smooth fit.")


def fig_off_path(plt, ctx) -> None:
    """Which drugs leave the shared path at 10 µM, and along which coordinate."""
    off = [f"off{i + 1}" for i in range(1, 20)]
    teacher = ctx.vectors[ctx.vectors.dose == 10000].set_index("drug")[off]
    measured = ctx.measured[ctx.measured.dose == 10000].set_index("drug")[off]
    size = np.sqrt((teacher ** 2).sum(1)).sort_values()
    floor = ctx.distances[(ctx.distances.kind == "floor") & (ctx.distances.label.astype(str) == "10000")].distance.median()
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    margins(fig, 1.35, 1.75, top=0.85, bottom=1.05)
    y = np.arange(len(size))
    ax.barh(y, size, color=[CLASS_COLOR[ctx.klass[d]] for d in size.index], height=0.68)
    ax.scatter(np.sqrt((measured.loc[size.index] ** 2).sum(1)), y, s=26, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=4, label="Measured")
    ax.axvline(floor, color=INK_2, lw=1, ls="--", label="Two wells of one condition")
    ax.set_yticks(y, [ctx.label[d] for d in size.index], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Distance off the shared path at 10 µM")
    right = max(size.max(), np.sqrt((measured ** 2).sum(1)).max())
    ax.set_xlim(0, right * 1.03)
    for i, d in enumerate(size.index):
        strongest = teacher.loc[d].abs().sort_values(ascending=False).index[:2]
        text = ", ".join(f"c{name[3:]} {teacher.loc[d, name]:+.1f}" for name in strongest)
        ax.annotate(text, (1, i), xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.8, color=INK_2, annotation_clip=False)
    ax.annotate("Largest coordinates", (1, len(size) - 0.1), xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points", va="bottom", fontsize=8, color=INK, annotation_clip=False)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.legend(handles=class_handles(plt), loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=5, fontsize=8.5)
    finish(plt, fig, ctx.dir / "11_off_path.png", "Which drugs leave the shared path, and how",
           "Bars: the embedding at 10 µM minus the curve shared by all drugs. Rings: the same for measured means.")


def draw(root: Path) -> None:
    plt = setup()
    ctx = Saved(root)
    for fig in (fig_map, fig_each_drug, fig_embeddings, fig_heatmap, fig_distances, fig_similarity, fig_quality, fig_axis_curves, fig_axis_genes, fig_path_genes, fig_off_path):
        fig(plt, ctx)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["collect", "draw"])
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--chemicals", default="/zfsauton/scratch/yixiz/sciplex3/raw/GSE139944/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz")
    p.add_argument("--gene-sets", default=GENE_SETS)
    a = p.parse_args(argv)
    if a.step == "collect":
        collect(a.data, Path(a.root), a.chemicals, Path(a.gene_sets))
    else:
        draw(Path(a.root))


if __name__ == "__main__":
    main()
