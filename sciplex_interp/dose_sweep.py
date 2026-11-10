"""Does a generator's population change with dose? Dose sweeps on the validation drugs.

Each model generates the same control cells at every dose of a 7-dose grid, so dose is the only input
that changes (CellFlow's flow is deterministic given its source cells; chemCPA reuses one sampling seed
per run). Only the 9 validation drugs are swept: their four doses are training or validation conditions,
so no test cell is read, generated for or scored here. Each model is at its selected checkpoint.

  axes      which low-dimensional axes of the genes carry dose: <root>/figures/axes.csv, axes.json
  generate  one model and seed: <root>/figures/dose/<variant>/seed_<s>.npz (needs the model's environment)
  collect   curves.csv, gain.csv, gain_paired.csv and hist.npz in <root>/figures/dose
  draw      PNGs from those files only

Measures (fixed before any sweep was read). x is measured, y generated, per drug over its doses:
  effect     mean change from vehicle on the drug's top-50 DEGs, projected on the direction of its measured
             10 µM change (a training dose).
  gain       Σ(x−x̄)(y−ȳ) / Σ(x−x̄)², centred within each drug and pooled over drugs and the 4 measured doses.
             1: the full measured dose response. 0: dose-independent.
  unseen     the same through the origin on the two steps into and out of the validation dose
             (d* − lower neighbour, upper neighbour − d*). Three of the four doses are training doses, so
             `gain` can come from memorised conditions; this one cannot.
  separation Σ MMD²(10 nM, 10 µM) generated / Σ MMD²(10 nM, 10 µM) measured, in the spec PCA. Generated cells
             share their sources across doses, so alternate cells are used to keep the two samples disjoint.
  pc1        added after the first gains were read, at the user's request: `gain` and `unseen` on the first
             principal component of the frozen spec PCA of all genes (fit on real control and training cells).
             The response score r is kept in gain.csv but not drawn.
95% CIs bootstrap the 9 drugs. Oracle: replicate well 2 against replicate well 1. Dummy: vehicle cells, zero by construction.

usage: python -m sciplex_interp.dose_sweep generate --family cellflow --variant base --seed 0
       python -m sciplex_interp.dose_sweep collect --arm z70_lr3e4_mv10_t
       python -m sciplex_interp.dose_sweep draw --arm z70_lr3e4_mv10_t
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_eval import mmd2_rbf, response_score

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, dense

GRID_NM = (10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0)  # the teacher's grid (sciplex_interp.teacher)
MEASURED_NM = (10.0, 100.0, 1000.0, 10000.0)
N_BOOT = 10000
BACKBONE, BASE, CHEMCPA, CHEMCPA_ORIGINAL = "z70_k8_res_lr3e4", "base", "gaussmean_l100", "gauss_v1"
MEASURES = {"gain_pc1": "PC1, all 4 doses", "unseen_pc1": "PC1, unseen dose", "gain_effect": "Drug direction, all 4 doses",
            "unseen_effect": "Drug direction, unseen dose", "separation": "10 nM vs 10 µM population gap"}
ALSO_SAVED = ("gain_r", "unseen_r", "gain_pc2", "unseen_pc2")


def out_dir(root: Path) -> Path:
    return Path(root) / "figures" / "dose"


def axis_file(root: Path) -> Path:
    return Path(root) / "figures" / "axes.json"


def pc1(z: np.ndarray, ax: dict) -> np.ndarray:
    """First spec PC with vehicle at 0 and a stronger drug response positive."""
    return ax["sign"] * (np.asarray(z, dtype=np.float64)[:, 0] - ax["centre"])


def key(drug: str, dose: float, what: str) -> str:
    return f"{drug}::{dose:g}::{what}"


def directions(data: Data) -> dict:
    """Each validation drug's top-50 DEGs and unit direction of its measured 10 µM change (training cells)."""
    out = {}
    for t in data.tasks("val"):
        d = data.mean(data.condition_of(t.drug, 10000.0)) - data.spec.control_mean
        deg = np.argsort(-np.abs(d), kind="stable")[: data.spec.deg_top_k]
        out[t.drug] = (deg, d[deg] / np.linalg.norm(d[deg]))
    return out


def summarise(data: Data, dirs: dict, drug: str, X: np.ndarray) -> dict:
    deg, u = dirs[drug]
    return {"r": response_score(X, data.spec).astype(np.float32), "z": data.spec.project(X).astype(np.float32),
            "effect": ((X[:, deg] - data.spec.control_mean[deg]) @ u).astype(np.float32)}


def cellflow_cells(variant: str, seed: int, root: Path, data: Data, drugs: list[str]) -> dict:
    from .cellflow import build, generate as flow_cells, load_params, model_data, run_dir

    run = run_dir(root, variant, seed)
    cfg = json.loads((run / "config.json").read_text())
    train = model_data(data, cfg, root)
    cf, _ = build(train, cfg, seed)
    load_params(cf, run / "best_params.msgpack")
    conds = [{"condition": f"{d}@{x:g}", "drug": d, "dose_nM": x, "n_cells": 2, "n_rep1": 1} for d in drugs for x in GRID_NM]
    cells, _ = flow_cells(cf, train, conds, seed, 0)
    return {(c["drug"], c["dose_nM"]): cells[c["condition"]] for c in conds}


def chemcpa_cells(variant: str, seed: int, data: Data, drugs: list[str]) -> dict:
    import torch

    from sciplex_baselines import chemcpa as C

    from .chemcpa_compare import run_root

    spec = data.spec
    checkpoint = torch.load(run_root(variant) / f"seed_{seed}" / "model.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    likelihood, outputs = config["likelihood"], C.output_genes(config)
    model = C.make_model(C.n_features(spec, likelihood), checkpoint["fingerprints"], config, checkpoint["seed"], "cpu",
                         C.n_outputs(spec, config))
    model.load_state_dict(checkpoint["model"])
    controls, _ = C.prediction_inputs(data.adata, spec, "val", likelihood, outputs)
    gene_ids = C.output_gene_ids(spec, config)
    cells = {}
    for d in drugs:
        for x in GRID_NM:  # one seed for every dose: the same control cells and the same noise
            pred = C.generate(model, controls, [{"condition": "sweep", "drug": d, "dose_nM": x, "n_cells": 0}], checkpoint["drugs"],
                              gene_ids, n_cells=config["n_prediction_cells"], seed=seed + 30000, mode=config["generation"])
            cells[d, x] = dense(spec.align(pred.X, pred.var_names))
    return cells


def generate(family: str, variant: str, seed: int, root: Path, panel: str) -> None:
    path = out_dir(root) / variant / f"seed_{seed}.npz"
    if path.exists():
        print(f"{path} exists, skipping", flush=True)
        return
    data = Data(panel)
    dirs = directions(data)
    drugs = list(dirs)
    cells = cellflow_cells(variant, seed, root, data, drugs) if family == "cellflow" else chemcpa_cells(variant, seed, data, drugs)
    arrays = {}
    for (drug, dose), X in cells.items():
        if not np.isfinite(X).all():
            raise SystemExit(f"non-finite cells: {variant} seed {seed} {drug} {dose}")
        arrays.update({key(drug, dose, k): v for k, v in summarise(data, dirs, drug, X).items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f"wrote {path}: {len(cells)} (drug, dose) populations", flush=True)


# ---- axes ---------------------------------------------------------------------------------------
def axes(root: Path, panel: str) -> None:
    """Which low-dimensional axes carry dose? Control and training cells only: nothing held out is read.

    Per frozen spec PC: share of all cell variance, share of the variance between training-condition means,
    and how well log dose explains the means within a drug. The same for the top axes of the training mean shifts
    (the teacher's basis), for comparison: those are not neutral, the teacher arms are handed them.
    """
    data = Data(panel)
    train = data.cond[data.cond.split == "train"]
    cells = np.concatenate([data.ctrl, *[data.idx(c) for c in train.index]])
    total = data.X[cells].astype(np.float64).var(0).sum()
    Z = np.asarray(data.Z, dtype=np.float64)
    m0 = Z[data.ctrl].mean(0)
    M = np.array([Z[data.idx(c)].mean(0) for c in train.index]) - m0
    D = np.array([data.mean(c) - data.ctrl_mean for c in train.index])
    dose = np.log10(train.dose_nM.to_numpy(float))
    drug = train.drug.to_numpy()
    centre = lambda v: v - pd.Series(v).groupby(drug).transform("mean").to_numpy()

    n_ctrl = len(data.ctrl)  # `cells` starts with the controls

    def row(name: str, cell_coord: np.ndarray, mean_coord: np.ndarray) -> dict:
        yc, xc = centre(mean_coord), centre(dose)
        return {"axis": name, "cell_var_share": cell_coord.var() / total, "between_share": (mean_coord ** 2).sum() / (D ** 2).sum(),
                "dose_r2_within_drug": float(np.corrcoef(xc, yc)[0, 1] ** 2), "between_to_control": mean_coord.var() / cell_coord[:n_ctrl].var()}

    rows = [row(f"PC{k + 1}", Z[cells, k], M[:, k]) for k in range(10)]
    V = np.linalg.svd(D, full_matrices=False)[2]
    Xc = data.X[cells].astype(np.float64) - data.ctrl_mean
    for k in range(3):
        rows.append(row(f"shift{k + 1}", Xc @ V[k], D @ V[k]))
    table = pd.DataFrame(rows)
    figures = axis_file(root).parent
    figures.mkdir(parents=True, exist_ok=True)
    table.to_csv(figures / "axes.csv", index=False)

    spec = data.spec
    U = np.asarray(spec.pca_components, dtype=np.float64)
    top_dose = (train.dose_nM == 10000).to_numpy()
    sign = 1.0 if M[top_dose, 0].mean() >= 0 else -1.0
    loadings = U[0] ** 2
    order = np.argsort(-loadings)[:12]
    r = response_score(data.X[cells], spec)
    info = {"sign": sign, "centre": float(m0[0]), "between_share": float(table.between_share[0]), "cell_var_share": float(table.cell_var_share[0]),
            "cosine_with_teacher_axis": float(abs(V[0] @ U[0])), "program_weight": float(loadings[spec.program_idx].sum()),
            "program_weight_if_even": len(spec.program_idx) / U.shape[1], "corr_with_r_over_cells": float(np.corrcoef(sign * Z[cells, 0], r)[0, 1]),
            "top_genes": [spec.gene_symbols[i] for i in order], "top_genes_weight": float(loadings[order].sum())}
    axis_file(root).write_text(json.dumps(info, indent=2) + "\n")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(info, indent=1))


# ---- collect ------------------------------------------------------------------------------------
def centred(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    return (xc * yc).sum(1), (xc * xc).sum(1)


def steps(x: np.ndarray, y: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per drug, the two steps into and out of the validation dose (position in MEASURED_NM)."""
    rows = np.arange(len(x))
    into = (x[rows, position] - x[rows, position - 1], y[rows, position] - y[rows, position - 1])
    out = (x[rows, position + 1] - x[rows, position], y[rows, position + 1] - y[rows, position])
    return into[0] * into[1] + out[0] * out[1], into[0] ** 2 + out[0] ** 2


def separation(a: dict, drugs: list[str], sigmas, alternate: bool) -> np.ndarray:
    low, high = (slice(0, None, 2), slice(1, None, 2)) if alternate else (slice(None), slice(None))
    return np.array([mmd2_rbf(a[d, 10.0]["z"][low], a[d, 10000.0]["z"][high], sigmas) for d in drugs])


def collect(arm: str, root: Path, panel: str) -> None:
    data = Data(panel)
    dirs, spec, out = directions(data), data.spec, out_dir(root)
    ax = json.loads(axis_file(root).read_text())
    centre2 = float(np.asarray(data.Z)[data.ctrl, 1].mean())
    drugs = list(dirs)
    position = np.array([MEASURED_NM.index(t.dose) for t in data.tasks("val")])
    real = {name: {(d, x): summarise(data, dirs, d, data.X[data.idx(data.condition_of(d, x), rep)]) for d in drugs for x in MEASURED_NM}
            for name, rep in (("measured", None), ("rep1", "rep1"), ("rep2", "rep2"))}
    sweeps = {p.name: [dict(np.load(f)) for f in sorted(p.glob("seed_*.npz"))] for p in sorted(out.iterdir()) if p.is_dir()}
    if arm not in sweeps:
        raise SystemExit(f"no sweep for {arm} under {out}")

    curves = [{"model": name, "seed": 0, "drug": d, "dose": x, "mean_r": float(v["r"].mean()), "effect": float(v["effect"].mean()),
               "pc1": float(pc1(v["z"], ax).mean()), "pc2": float(v["z"][:, 1].mean() - centre2), "unseen": x == t.dose} for name in real for t in data.tasks("val") for d, x, v in
              ((t.drug, x, real[name][t.drug, x]) for x in MEASURED_NM)]
    for name, seeds in sweeps.items():
        curves += [{"model": name, "seed": s, "drug": d, "dose": x, "mean_r": float(z[key(d, x, "r")].mean()),
                    "effect": float(z[key(d, x, "effect")].mean()), "pc1": float(pc1(z[key(d, x, "z")], ax).mean()),
                    "pc2": float(z[key(d, x, "z")][:, 1].mean() - centre2), "unseen": False}
                   for s, z in enumerate(seeds) for d in drugs for x in GRID_NM]
    curves = pd.DataFrame(curves)
    curves.to_csv(out / "curves.csv", index=False)

    def table(model: str, column: str) -> np.ndarray:  # (drug, measured dose), seeds averaged
        part = curves[(curves.model == model) & curves.dose.isin(MEASURED_NM)]
        return part.pivot_table(index="drug", columns="dose", values=column, aggfunc="mean").loc[drugs].to_numpy()

    sep_real = separation(real["measured"], drugs, spec.mmd_sigmas, False)
    parts = {}  # model -> measure -> (numerator, denominator) per drug
    for name in ["oracle", *sweeps]:
        x_model, y_model = ("rep1", "rep2") if name == "oracle" else ("measured", name)
        parts[name] = {}
        for column, tag in (("effect", "effect"), ("mean_r", "r"), ("pc1", "pc1"), ("pc2", "pc2")):
            x, y = table(x_model, column), table(y_model, column)
            parts[name][f"gain_{tag}"] = centred(x, y)
            parts[name][f"unseen_{tag}"] = steps(x, y, position)
        if name == "oracle":
            parts[name]["separation"] = (separation(real["rep2"], drugs, spec.mmd_sigmas, False),
                                         separation(real["rep1"], drugs, spec.mmd_sigmas, False))
        else:
            per_seed = [separation({(d, x): {"z": z[key(d, x, "z")]} for d in drugs for x in (10.0, 10000.0)}, drugs, spec.mmd_sigmas, True)
                        for z in sweeps[name]]
            parts[name]["separation"] = (np.mean(per_seed, axis=0), sep_real)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(drugs), (N_BOOT, len(drugs)))
    boots = {n: {m: num[idx].sum(1) / den[idx].sum(1) for m, (num, den) in p.items()} for n, p in parts.items()}
    point = {n: {m: float(num.sum() / den.sum()) for m, (num, den) in p.items()} for n, p in parts.items()}
    gain = [{"model": n, "measure": m, "value": point[n][m], "ci95_low": np.percentile(boots[n][m], 2.5),
             "ci95_high": np.percentile(boots[n][m], 97.5), "n_seeds": len(sweeps.get(n, [0]))} for n in parts for m in (*MEASURES, *ALSO_SAVED)]
    paired = [{"ref": n, "measure": m, "arm": point[arm][m], "value_ref": point[n][m], "delta": point[arm][m] - point[n][m],
               "ci95_low": np.percentile(boots[arm][m] - boots[n][m], 2.5), "ci95_high": np.percentile(boots[arm][m] - boots[n][m], 97.5),
               "n_drugs_arm_higher": int(np.sum(np.divide(*parts[arm][m]) > np.divide(*parts[n][m])))}
              for n in parts if n != arm for m in (*MEASURES, *ALSO_SAVED)]
    pd.DataFrame(gain).to_csv(out / "gain.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "gain_paired.csv", index=False)

    x = table("measured", "effect")
    cases = [drugs[i] for i in np.argsort(-(x.max(1) - x.min(1)))[:2]]  # by rule: the two largest measured dose ranges
    hist = {}
    for d in cases:
        for dose in MEASURED_NM:
            hist[f"measured::{key(d, dose, 'effect')}"] = real["measured"][d, dose]["effect"]
            hist[f"measured::{key(d, dose, 'pc1')}"] = pc1(real["measured"][d, dose]["z"], ax)
            for name, seeds in sweeps.items():
                hist[f"{name}::{key(d, dose, 'effect')}"] = np.concatenate([z[key(d, dose, "effect")] for z in seeds])
                hist[f"{name}::{key(d, dose, 'pc1')}"] = np.concatenate([pc1(z[key(d, dose, "z")], ax) for z in seeds])
    np.savez_compressed(out / "hist.npz", **hist)
    (out / "meta.json").write_text(json.dumps({"arm": arm, "drugs": drugs, "cases": cases, "models": list(sweeps)}, indent=2) + "\n")
    print(pd.DataFrame(gain).pivot(index="model", columns="measure", values="value")[[*MEASURES, *ALSO_SAVED]].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"case drugs: {cases}\nwrote {out}")


# ---- draw ---------------------------------------------------------------------------------------
def draw(arm: str, root: Path) -> None:
    from matplotlib.lines import Line2D

    from .cellflow_figures import (ARM_LABELS, BACKBONE_LABEL, COLORS, INK, INK_2, MUTED, RAMP, ROW_LABELS, SURFACE, dose_text, finish, margins, pc1_text, setup, stack,
                                   uses_teacher)

    plt = setup()
    out = out_dir(root)
    meta = json.loads((out / "meta.json").read_text())
    curves, gain, paired = (pd.read_csv(out / f"{n}.csv") for n in ("curves", "gain", "gain_paired"))
    hist = np.load(out / "hist.npz")
    ax_info = json.loads(axis_file(root).read_text())
    labels = {arm: ARM_LABELS.get(arm, arm), BACKBONE: BACKBONE_LABEL, BASE: ROW_LABELS["cellflow"],
              CHEMCPA: ROW_LABELS["chemcpa"], CHEMCPA_ORIGINAL: "chemCPA-Gauss (original)"}
    style = {arm: (COLORS["arm"], "-"), BACKBONE: (MUTED, "-"), BASE: (COLORS["cellflow"], "-"), CHEMCPA: (COLORS["chemcpa"], "-"),
             CHEMCPA_ORIGINAL: ("#e87ba4", (0, (5, 2.5)))}
    models = [m for m in (arm, BACKBONE, CHEMCPA, CHEMCPA_ORIGINAL, BASE) if m in meta["models"]]
    note = ("Validation drugs only; no test cell is used. The arm and its embedding were chosen on these drugs' held-out doses, so its "
            "unseen-dose numbers are optimistic. Each model is at its lowest validation MMD² checkpoint.")
    pc_note = pc1_text(ax_info, uses_teacher(arm))
    ticks = list(MEASURED_NM)

    # Drugs ordered by measured dose range, the same in both curve figures.
    real = curves[curves.model == "measured"]
    span = real.groupby("drug").effect.agg(lambda v: v.max() - v.min()).sort_values(ascending=False)

    def curve_figure(column: str, ylabel: str, name: str, title: str, prefix: str = "") -> None:
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.4), sharex=True)
        margins(fig, 0.72, 0.2, top=1.62, bottom=1.3, hspace=0.3, wspace=0.2)
        for ax, drug in zip(axes.flat, span.index):
            ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
            for m in models:
                color, ls = style[m]
                part = curves[(curves.model == m) & (curves.drug == drug)]
                by = part.groupby("dose")[column]
                ax.fill_between(by.min().index, by.min(), by.max(), color=color, alpha=0.14, linewidths=0)
                ax.plot(by.median().index, by.median(), color=color, ls=ls, lw=2.2 if m == arm else 1.5, zorder=4 if m == arm else 3)
            r = real[real.drug == drug].sort_values("dose")
            ax.plot(r.dose, r[column], color=INK, lw=1.2, zorder=5)
            seen = r[~r.unseen]
            ax.scatter(seen.dose, seen[column], s=34, color=INK, zorder=6)
            ax.scatter(r[r.unseen].dose, r[r.unseen][column], s=38, color=SURFACE, edgecolors=INK, linewidths=1.6, zorder=6)
            ax.set_xscale("log")
            ax.set_xticks(ticks, [dose_text(t) for t in ticks])
            ax.minorticks_off()
            ax.set_title(drug.split(" (")[0])
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        w, h = fig.get_size_inches()
        handles = [Line2D([], [], color=INK, marker="o", lw=1.2, ms=5.5),
                   Line2D([], [], color=INK, marker="o", mfc=SURFACE, mew=1.5, lw=0, ms=6)]
        handles += [Line2D([], [], color=style[m][0], ls=style[m][1], lw=2.2 if m == arm else 1.5) for m in models]
        handles.append(Line2D([], [], color=MUTED, lw=1.0))
        names = ["Measured, training dose", "Measured, unseen dose", *[labels[m] for m in models], "Dummy: vehicle (no change)"]
        fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.66 / h), ncol=3, fontsize=8.5, handlelength=2.0, columnspacing=1.4)
        finish(plt, fig, out / name, title, "Same control cells at every dose, so dose is the only change. Line: median of 3 seeds; band: seed range. "
               "Biggest measured dose effect first.", prefix + note)

    curve_figure("pc1", "Mean PC1", "1_dose_curves_pc1.png", "Dose-response curves on PC1 of all genes", pc_note)
    curve_figure("effect", "Effect along the drug's direction", "2_dose_curves_drug_direction.png", "Dose-response curves on each drug's top-50 DEGs")

    # Dose paths in the PC1 x PC2 plane: a model that ignores dose stays on one spot.
    pc2_share = float(pd.read_csv(root / "figures" / "axes.csv").between_share.iloc[1])
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.8))
    margins(fig, 0.72, 0.2, top=1.62, bottom=1.5, hspace=0.34, wspace=0.2)
    sizes = {10.0: 12, 100.0: 26, 1000.0: 46, 10000.0: 74}
    for ax, drug in zip(axes.flat, span.index):
        ax.scatter([0], [0], marker="P", s=70, color=MUTED, edgecolors=SURFACE, linewidths=0.6, zorder=2)
        for m in models:
            color, ls = style[m]
            path = curves[(curves.model == m) & (curves.drug == drug)].groupby("dose")[["pc1", "pc2"]].mean()
            ax.plot(path.pc1, path.pc2, color=color, ls=ls, lw=2.2 if m == arm else 1.4, zorder=4 if m == arm else 3)
            at = path.loc[list(MEASURED_NM)]
            ax.scatter(at.pc1, at.pc2, s=[sizes[x] for x in at.index], color=color, edgecolors=SURFACE, linewidths=0.6, zorder=5 if m == arm else 3)
        r = real[real.drug == drug].sort_values("dose")
        ax.plot(r.pc1, r.pc2, color=INK, lw=1.3, zorder=6)
        ax.scatter(r.pc1, r.pc2, s=[sizes[x] for x in r.dose], color=[SURFACE if u else INK for u in r.unseen], edgecolors=INK, linewidths=1.3, zorder=7)
        ax.set_title(drug.split(" (")[0])
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    for ax in axes[-1, :]:
        ax.set_xlabel("PC1 (vehicle = 0)")
    w, h = fig.get_size_inches()
    handles = [Line2D([], [], color=INK, marker="o", lw=1.3, ms=6), Line2D([], [], color=INK, marker="o", mfc=SURFACE, mew=1.3, lw=0, ms=6)]
    handles += [Line2D([], [], color=style[m][0], ls=style[m][1], lw=2.2 if m == arm else 1.4, marker="o", ms=5) for m in models]
    handles.append(Line2D([], [], color=MUTED, marker="P", lw=0, ms=8))
    fig.legend(handles, ["Measured, training dose", "Measured, unseen dose", *[labels[m] for m in models], "Dummy: vehicle stays here"], loc="upper left",
               bbox_to_anchor=(0.008, 1 - 0.66 / h), ncol=3, fontsize=8.5, handlelength=2.0, columnspacing=1.4)
    finish(plt, fig, out / "5_dose_paths_pc1_pc2.png", "Each model's dose path in the PC1 x PC2 plane",
           "Marker size grows with dose (10 nM to 10 µM). A model that ignores dose stays on one spot. Biggest measured dose effect first; each panel has its own scale.",
           f"PC2 holds only {100 * pc2_share:.1f}% of the variance between training conditions, yet the strongest drugs dip on it at mid doses and rise at "
           "10 µM. " + pc_note + note)
    dip = real.pivot_table(index="drug", columns="dose", values="pc2").loc[span.index]
    print("measured PC2 by dose, biggest dose effect first\n" + dip.to_string(float_format=lambda v: f"{v:+.2f}"))

    rows = ["oracle", *models, "dummy"]
    names = {**labels, "oracle": "Oracle: replicate well 2 vs 1", "dummy": ROW_LABELS["dummy"]}
    colors = {**{m: style[m][0] for m in models}, "oracle": COLORS["oracle"], "dummy": COLORS["dummy"]}
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(14.5, 0.46 * len(rows) + 2.55), sharey=True)
    margins(fig, 3.45, 0.55, bottom=1.3, wspace=0.34)
    y = {k: len(rows) - 1 - i for i, k in enumerate(rows)}
    for ax, (m, title) in zip(axes, MEASURES.items()):
        ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(1, color=INK_2, lw=1.0, zorder=1)
        for k in rows:
            g = gain[(gain.model == k) & (gain.measure == m)]
            value, low, high = (0.0, 0.0, 0.0) if k == "dummy" else (g.value.iloc[0], g.ci95_low.iloc[0], g.ci95_high.iloc[0])
            ax.plot([low, high], [y[k], y[k]], color=colors[k], lw=2.4, solid_capstyle="round", zorder=3)
            ax.scatter([value], [y[k]], s=50, color=colors[k], edgecolors=SURFACE, linewidths=1.2, zorder=4)
            ax.text(1.03, y[k], f"{value:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK_2,
                    fontweight="bold" if k == arm else "normal")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y.values()), [stack(names[k]) for k in y], color=INK)
    axes[0].get_yticklabels()[rows.index(arm)].set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    finish(plt, fig, out / "3_dose_gain.png", "How much of the real dose response does each model reproduce?",
           "0: the model ignores dose. 1: the full measured response. Dot: pooled over the 9 validation drugs. Bar: 95% CI (bootstrap over drugs). "
           "Number: the value.", "Unseen dose: only the steps into and out of the held-out dose, which no model trained on. " + pc_note + note)

    ramp = [RAMP[1], "#5598e7", RAMP[3], RAMP[4]]  # ordinal: light = low dose
    rows = ["measured", *models]
    fig, axes = plt.subplots(len(rows), 2 * len(meta["cases"]), figsize=(14.5, 1.55 * len(rows) + 2.3), sharex="col")
    margins(fig, 2.45, 0.2, top=1.2, bottom=1.3, hspace=0.16, wspace=0.14)
    for c, drug in enumerate(meta["cases"]):
        for w, what in enumerate(("pc1", "effect")):
            lo, hi = np.percentile(np.concatenate([hist[f"measured::{key(drug, x, what)}"] for x in MEASURED_NM]), [0.5, 99.5])
            edges = np.linspace(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo), 28)  # tails fall outside: clipping would draw false spikes
            col = 2 * c + w
            for i, m in enumerate(rows):
                ax = axes[i, col]
                for x, color in zip(MEASURED_NM, ramp):
                    ax.hist(hist[f"{m}::{key(drug, x, what)}"], edges, density=True, histtype="step", color=color, lw=1.5)
                ax.set_yticks([])
                ax.grid(False)
                if col == 0:
                    ax.set_ylabel("Measured" if m == "measured" else labels[m].replace(" + ", " +\n").replace(", no", ",\nno"), rotation=0, ha="right",
                                  va="center", fontsize=8.5, color=INK, labelpad=8)
            axes[0, col].set_title(f"{drug.split(' (')[0]}: " + ("PC1 of all genes" if what == "pc1" else "its top-50 DEGs"), fontsize=9.5)
            axes[-1, col].set_xlabel("PC1" if what == "pc1" else "Effect along the drug's direction")
    w, h = fig.get_size_inches()
    fig.legend([Line2D([], [], color=c, lw=1.8) for c in ramp], [dose_text(x) for x in MEASURED_NM], loc="upper left",
               bbox_to_anchor=(0.008, 1 - 0.66 / h), ncol=4, fontsize=8.5, handlelength=1.8)
    finish(plt, fig, out / "4_dose_histograms.png", "Each model's populations at the four doses, overlaid",
           "If a model ignores dose its four lines sit on top of each other. Compare each row with Measured. Drugs: the two largest measured dose effects.", pc_note + note)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["axes", "generate", "collect", "draw"])
    p.add_argument("--family", choices=["cellflow", "chemcpa"])
    p.add_argument("--variant")
    p.add_argument("--seed", type=int)
    p.add_argument("--arm")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    root = Path(a.root)
    if a.step == "axes":
        axes(root, a.data)
    elif a.step == "generate":
        generate(a.family, a.variant, a.seed, root, a.data)
    elif a.step == "collect":
        collect(a.arm, root, a.data)
    else:
        draw(a.arm, root)


if __name__ == "__main__":
    main()
