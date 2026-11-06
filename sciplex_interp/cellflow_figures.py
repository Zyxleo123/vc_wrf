"""Analysis figures: one CellFlow arm against the oracle, CellFlow, chemCPA and dummy baselines.

Three steps, each a CPU job (slurm/interp/submit.sh stages cellflow_fig_arms, cellflow_fig_collect and
cellflow_fig_draw). No step loads a model or generates cells: an arm's split must already have been
scored by cellflow_test, so these figures can never be what first scores an arm on the test set.

  arms     every finished arm at its lowest validation MMD² checkpoints, from history.jsonl only.
           Writes <root>/figures/arms.csv and curves.csv and prints the ranking. The pick is the lowest
           validation MMD² among arms where no seed ended ignoring the condition (docs/interp/log.md).
  collect  metric rows and per-cell profiles of the arm and the baselines on one split. Writes
           <root>/figures/<arm>_<split>/{rows,conditions,summary,paired,floors}.csv, profiles.npz, meta.json.
  draw     PNGs from those files only, so a figure redraws without the panel.

Baselines (every model at its lowest validation MMD² checkpoint; published WRF-chosen rows are left out):
  oracle       replicate well: the condition's real cells of one well scored against the other's. Its mean
               RMSEs compare two half-size wells, with DEGs picked on one of them, so they are not a limit:
               the RMSE panels carry the sampling floor of the measured mean instead (sciplex_interp.diagnose).
  oracle_mean  the adjacent-dose mixture shifted onto the measured mean (diagnostics; val and inner only).
               Its mean RMSEs are zero by construction, so only MMD² and W1(r) are shown.
  cellflow     the published recipe retrained (`base`).
  chemcpa      chemCPA-Gauss + mean consistency (`gaussmean_l100`).
  adjacent     adjacent-dose mixture: resamples real neighbouring-dose cells, a reference and not a generator.
  dummy        vehicle cells: predicts that the drug does nothing.

usage: python -m sciplex_interp.cellflow_figures arms
       python -m sciplex_interp.cellflow_figures collect --arm z70_lr3e4_mv10_t --split val
       python -m sciplex_interp.cellflow_figures draw --arm z70_lr3e4_mv10_t --split val
       python -m sciplex_interp.cellflow_figures umap --arm z70_lr3e4_mv10_t --split val
"""
from __future__ import annotations

import argparse
import json
import textwrap
import zlib
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from sciplex_analysis.analyze import paired_stats
from sciplex_eval import response_score

from .cellflow_compare import cellflow_rows
from .cellflow_report import COLS, condition_signal, load, selected_rows
from .chemcpa_compare import LABELS, METRICS, MODEL_TABLE, matrix, model_rows, run_root, variant_rows
from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, dense, read_jsonl
from .diagnose import floors as sampling_floors

REPO = Path(__file__).resolve().parents[1]
CHEMCPA = "gaussmean_l100"
BASE = "base"
BACKBONE = "z70_k8_res_lr3e4"  # the latent, 8-conditions arm without the average-arrow term or teacher input
IGNORED_SPREAD = 0.05  # SD of generated / measured effect size across training conditions (the log's rule)
N_BOOT = 10000
ARM_LABELS = {"z70_lr3e4_mv10_t": "CellFlow latent + average-arrow + teacher",
              "z70_k8_res_lr3e4": "CellFlow latent, 8 conditions per update",
              "meanv_l10": "CellFlow + average-arrow term"}
ROW_LABELS = {"oracle": "Oracle: replicate well", "oracle_mean": "Oracle mean: mixture on measured mean",
              "chemcpa": "chemCPA-Gauss + mean consistency", "cellflow": "CellFlow retrain",
              "adjacent": "Adjacent-dose mixture", "dummy": "Dummy: vehicle cells"}
SHORT = {"oracle": "Oracle", "oracle_mean": "Oracle mean", "arm": "Best arm", "chemcpa": "chemCPA", "cellflow": "CellFlow",
         "adjacent": "Adjacent", "dummy": "Dummy"}
ORDER = ["oracle", "oracle_mean", "arm", "chemcpa", "cellflow", "adjacent", "dummy"]
GENERATIVE = ("arm", "chemcpa", "cellflow")
WITH_CELLS = ["oracle", "arm", "chemcpa", "cellflow", "adjacent", "dummy"]  # oracle_mean's cells are not saved
RMSE_METRICS = ["rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program"]
N_POINTS = 350  # cells drawn per method in the overlay
MARKERS = {"oracle": "v", "arm": "o", "chemcpa": "s", "cellflow": "^", "adjacent": "D", "dummy": "X"}  # a second cue: five colours alone are not safe in a scatter

# dataviz reference palette (light): categorical slots 1-4 in order, neutrals for the two bounds.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"arm": "#2a78d6", "cellflow": "#eb6834", "chemcpa": "#1baf7a", "adjacent": "#eda100",
          "oracle": INK_2, "oracle_mean": INK_2, "dummy": MUTED}
RAMP = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]  # sequential blue, light = small error


def fig_dir(root: Path, arm: str, split: str) -> Path:
    return root / "figures" / f"{arm}_{split}"


def uses_teacher(variant: str) -> bool:
    path = REPO / "configs" / "interp" / f"cellflow_{variant}.json"
    return path.exists() and "teacher" in json.loads(path.read_text()).get("covariates", ())


# ---- arms ---------------------------------------------------------------------------------------
def arms(root: Path) -> None:
    """Per-seed table of every finished arm, training curves, and the ranking the pick rests on."""
    df = load(root, None)
    seeds = {v: sum((root / "cellflow" / v / f"seed_{s}" / "training_complete.json").exists() for s in range(3))
             for v in df.variant.unique()}
    skipped = sorted(v for v, n in seeds.items() if n < 3)
    df = df[df.variant.map(seeds) == 3]
    keys = ["variant", "seed", "task", "step"]
    curves = df.groupby(keys)[COLS].mean()
    curves["var_ratio_median"] = df.groupby(keys).var_ratio.median()
    curves = curves.join(df.groupby(keys)[["effect_real_all", "effect_pred_all"]].apply(condition_signal)).reset_index()

    sel = selected_rows(df)
    table = sel[sel.task == "val"].groupby(["variant", "seed"]).agg(
        step=("step", "first"), **{m: (m, "mean") for m in METRICS}).reset_index()
    fit = curves[curves.task == "train_fit"].sort_values("step").groupby(["variant", "seed"]).tail(1)
    end = curves[curves.task == "val"].sort_values("step").groupby(["variant", "seed"]).tail(1)
    table = table.merge(fit[["variant", "seed", "step", "spread", "var_ratio_median"]].rename(
        columns={"step": "end_step", "spread": "end_spread", "var_ratio_median": "end_var"}), on=["variant", "seed"])
    table = table.merge(end[["variant", "seed", "mmd2"]].rename(columns={"mmd2": "end_mmd2"}), on=["variant", "seed"])
    table["ignored"] = table.end_spread < IGNORED_SPREAD
    table["teacher"] = table.variant.map(uses_teacher)

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "arms.csv", index=False)
    curves.to_csv(out / "curves.csv", index=False)

    ranked = table.groupby("variant").agg(mmd2=("mmd2", "mean"), mmd2_sd=("mmd2", "std"), deg50=("rmse_mean_deg50", "mean"),
                                          program=("rmse_mean_program", "mean"), w1_r=("w1_r", "mean"),
                                          end_mmd2=("end_mmd2", "mean"), end_var=("end_var", "mean"),
                                          ignored=("ignored", "sum"), teacher=("teacher", "first")).sort_values("mmd2")
    print(ranked.to_string(float_format=lambda v: f"{v:.4f}"))
    if skipped:
        print(f"left out, fewer than 3 finished seeds: {skipped}")
    print(f"pick: {ranked[ranked.ignored == 0].index[0]} (lowest validation MMD², no seed ended ignoring the condition)")


# ---- collect ------------------------------------------------------------------------------------
def metric_frames(root: Path, arm: str, split: str) -> dict:
    table = pd.read_csv(MODEL_TABLE)
    frames = {"arm": cellflow_rows(root, arm, split, 0), "cellflow": cellflow_rows(root, BASE, split, 0),
              "chemcpa": variant_rows(CHEMCPA, split, 0),
              "adjacent": model_rows(table, "reference/adjacent_dose_mixture", split),
              "dummy": model_rows(table, "reference/vehicle", split), "oracle": model_rows(table, "reference/replicate", split)}
    shapes = pd.DataFrame(read_jsonl(root / "diagnose" / "shapes.jsonl"))
    part = shapes[(shapes.task_set == split) & (shapes.method == "mix_logdose+oracle_mean")]
    if len(part):
        frames["oracle_mean"] = part[["seed", "condition", *METRICS]].assign(**{m: np.nan for m in RMSE_METRICS})
    return frames


def read_cells(path: Path, spec) -> dict:
    a = ad.read_h5ad(path)
    X = dense(spec.align(a.X, a.var_names))
    labels = a.obs["condition"].astype(str).to_numpy()
    return {c: X[labels == c] for c in pd.unique(labels)}


def generated_paths(root: Path, key: str, arm: str, split: str) -> list[tuple[int, Path]]:
    base, name = ((run_root(CHEMCPA), "predictions.h5ad") if key == "chemcpa" else
                  (root / "cellflow" / (arm if key == "arm" else BASE), "preds.h5ad"))
    paths = sorted(base.glob(f"seed_*/{split}_sample_s0/{name}"))
    if not paths:
        raise SystemExit(f"no {split} cells for {key} under {base}")
    return [(int(p.parent.parent.name.split("_")[1]), p) for p in paths]


def collect(arm: str, split: str, root: Path, panel: str) -> None:
    frames = metric_frames(root, arm, split)  # first: fails at once if the arm has no rows on this split
    data = Data(panel)
    spec, tasks = data.spec, data.tasks(split)
    out = fig_dir(root, arm, split)
    out.mkdir(parents=True, exist_ok=True)
    axis = json.loads((root / "figures" / "axes.json").read_text())  # from sciplex_interp.dose_sweep axes
    centre2 = float(np.asarray(data.Z)[data.ctrl, 1].mean())
    arrays = {"control_mean": spec.control_mean.astype(np.float32), "deg_top_k": np.array(spec.deg_top_k),
              "program_idx": np.asarray(spec.program_idx)}

    directions = {}  # each condition's top-50 DEGs and unit direction of its measured change, for the case study
    for t in tasks:
        change = data.mean(t.target) - spec.control_mean
        deg = np.argsort(-np.abs(change), kind="stable")[: spec.deg_top_k]
        directions[t.target] = (deg, change[deg] / np.linalg.norm(change[deg]))

    def add(method: str, seed: int, condition: str, X: np.ndarray) -> np.ndarray:
        r = response_score(X, spec).astype(np.float32)
        arrays[f"{method}::{seed}::{condition}::r"] = r
        z = spec.project(X)
        arrays[f"{method}::{seed}::{condition}::pc1"] = (axis["sign"] * (z[:, 0] - axis["centre"])).astype(np.float32)
        arrays[f"{method}::{seed}::{condition}::pc2"] = (z[:, 1] - centre2).astype(np.float32)
        deg, u = directions[condition]
        arrays[f"{method}::{seed}::{condition}::effect"] = ((X[:, deg] - spec.control_mean[deg]) @ u).astype(np.float32)
        arrays[f"{method}::{seed}::{condition}::mean"] = X.mean(0, dtype=np.float64).astype(np.float32)
        arrays[f"{method}::{seed}::{condition}::sd"] = X.std(0, dtype=np.float64).astype(np.float32)
        return r

    vehicle = data.X[data.ctrl]
    conditions = []
    for t in tasks:
        r = add("measured", 0, t.target, data.X[data.idx(t.target)])
        add("oracle_ref", 0, t.target, data.X[data.idx(t.target, "rep1")])
        add("oracle", 0, t.target, data.X[data.idx(t.target, "rep2")])
        add("dummy", 0, t.target, vehicle)
        for seed in range(3):  # the population stage's recipe and random stream for `adjacent`
            rng = np.random.default_rng([seed, zlib.crc32(t.target.encode()), zlib.crc32(split.encode())])
            add("adjacent", seed, t.target, data.X[data.mixture(t, {t.low: 1 - t.w_logdose, t.high: t.w_logdose}, rng)])
        conditions.append({"condition": t.target, "drug": t.drug, "dose": t.dose, "n_real": len(r), "mean_r_real": float(r.mean())})

    for key in GENERATIVE:
        for seed, path in generated_paths(root, key, arm, split):
            cells = read_cells(path, spec)
            saved = {row["condition"]: row["mean_r_pred"] for row in read_jsonl(path.parent / "per_condition.jsonl")}
            for t in tasks:
                r = add(key, seed, t.target, cells[t.target])
                if abs(float(r.mean()) - saved[t.target]) > 1e-3:  # these must be the very cells the rows scored
                    raise SystemExit(f"{path}: {t.target} mean r {r.mean():.4f}, saved row {saved[t.target]:.4f}")

    rng = np.random.default_rng(0)
    summary, paired = [], []
    for key, df in frames.items():
        for m in METRICS:
            if df[m].isna().all():
                continue
            d = matrix(df, m).mean(axis=1).to_numpy()
            boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
            summary.append({"method": key, "metric": m, "mean": d.mean(), "ci95_low": np.percentile(boots, 2.5),
                            "ci95_high": np.percentile(boots, 97.5), "seed_sd": df.groupby("seed")[m].mean().std(ddof=1),
                            "n_seeds": df.seed.nunique(), "n_conditions": len(d)})
            if key != "arm":
                paired.append({"ref": key, "metric": m, **paired_stats(matrix(df, m), matrix(frames["arm"], m), rng, N_BOOT)})

    pd.concat([f.assign(method=k) for k, f in frames.items()]).to_csv(out / "rows.csv", index=False)
    pd.DataFrame(conditions).to_csv(out / "conditions.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(sampling_floors(data, (split,))).to_csv(out / "floors.csv", index=False)
    np.savez_compressed(out / "profiles.npz", **arrays)
    steps = [json.loads((root / "cellflow" / arm / f"seed_{s}" / "selected.json").read_text())["step"] for s in range(3)]
    (out / "meta.json").write_text(json.dumps({"arm": arm, "split": split, "n_conditions": len(tasks), "selected_steps": steps,
                                               "teacher": uses_teacher(arm)}, indent=2) + "\n")
    print(pd.DataFrame(summary).pivot(index="method", columns="metric", values="mean")[METRICS].reindex(
        [k for k in ORDER if k in frames]).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}")


# ---- draw ---------------------------------------------------------------------------------------
def dose_text(nM: float) -> str:
    return f"{nM / 1000:g} µM" if nM >= 1000 else f"{nM:g} nM"


def condition_text(row) -> str:
    return f"{row.drug.split(' (')[0]}, {dose_text(row.dose)}"


def load_context(root: Path, arm: str, split: str) -> SimpleNamespace:
    d = fig_dir(root, arm, split)
    meta = json.loads((d / "meta.json").read_text())
    conds = pd.read_csv(d / "conditions.csv").sort_values("mean_r_real", ascending=False).reset_index(drop=True)
    rows = pd.read_csv(d / "rows.csv")
    z = np.load(d / "profiles.npz")
    floors = pd.read_csv(d / "floors.csv")
    return SimpleNamespace(dir=d, meta=meta, arm=arm, split=split, conds=conds, rows=rows, profiles={k: z[k] for k in z.files},
                           axis=json.loads((root / "figures" / "axes.json").read_text()),
                           pc2_share=float(pd.read_csv(root / "figures" / "axes.csv").between_share.iloc[1]),
                           floor={m: float(floors[f"floor_{m}"].mean()) for m in RMSE_METRICS},
                           summary=pd.read_csv(d / "summary.csv"), paired=pd.read_csv(d / "paired.csv"),
                           order=[k for k in ORDER if k in set(rows.method)],
                           labels={**ROW_LABELS, "arm": ARM_LABELS.get(arm, arm)})


def bound(ctx, metric: str) -> tuple[str, float]:
    """The reference line of a metric: the sampling floor for mean RMSEs, the replicate well otherwise."""
    if metric in RMSE_METRICS:
        return "Floor", ctx.floor[metric]
    return "Oracle", float(ctx.summary[(ctx.summary.method == "oracle") & (ctx.summary.metric == metric)]["mean"].iloc[0])


def seeds_of(ctx, method: str, condition: str) -> list[int]:
    return sorted({int(k.split("::")[1]) for k in ctx.profiles if k.startswith(f"{method}::") and f"::{condition}::" in k})


def profile(ctx, method: str, condition: str, what: str) -> list[np.ndarray]:
    return [ctx.profiles[f"{method}::{s}::{condition}::{what}"] for s in seeds_of(ctx, method, condition)]


def footnote(ctx) -> str:
    n = ctx.meta["n_conditions"]
    if ctx.split == "val":
        text = f"Validation only: this arm was chosen on these {n} conditions"
        text += ", and so was its teacher, so its scores are optimistic." if ctx.meta["teacher"] else ", so its scores are optimistic."
    else:
        text = f"Test, {n} held-out conditions."
        if not ctx.meta["teacher"]:
            text += " Best arm with a test score; the teacher-input arms have none."
    return (text + " Oracles read the held-out condition itself. The adjacent mixture resamples real cells of neighbouring "
            "doses, so it is a reference, not a generator.")


def pc1_text(axis: dict, teacher: bool) -> str:
    return (f"PC1: first principal component of all 2,085 genes, from the frozen PCA of real control and training cells; it holds "
            f"{100 * axis['between_share']:.0f}% of the variance between training conditions. Vehicle = 0. The arm's latent contains PC1"
            + (" and its teacher predicts nearly this axis" if teacher else "") + ", so the arm has a home advantage here. ")


def pc_note(ctx) -> str:
    return pc1_text(ctx.axis, ctx.meta["teacher"])


REPLICATE_RMSE = ("The replicate well compares two half-size wells, so its RMSEs sit far above the sampling floor and are not a "
                  "limit: scoring below them does not make a model better than a measurement.")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9.5,
                         "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlesize": 10,
                         "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.grid": True, "axes.axisbelow": True,
                         "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
                         "text.color": INK, "legend.frameon": False})
    return plt


def finish(plt, fig, path: Path, title: str, subtitle: str, note: str = "") -> None:
    w, h = fig.get_size_inches()
    fig.text(0.012, 1 - 0.14 / h, title, fontsize=13, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 1 - 0.42 / h, subtitle, fontsize=9.5, va="top", color=INK_2)
    if note:
        fig.text(0.012, 0.08 / h, "\n".join(textwrap.wrap(note, int(w * 12.5))), fontsize=8.5, va="bottom", color=INK_2)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def margins(fig, left: float, right: float, top: float = 0.78, bottom: float = 0.95, **kw) -> None:
    """Margins in inches, so titles and notes keep their room at any figure size."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h, **kw)


def fmt(metric: str, v: float) -> str:
    return f"{v:.4f}" if metric == "mmd2" else f"{v:.3f}"


def fig_scoreboard(plt, ctx) -> None:
    """Every metric, every method: condition means (faint), their average and a condition-bootstrap 95% CI."""
    order = ctx.order
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14.5, 0.46 * len(order) + 2.55), sharey=True)
    margins(fig, 3.45, 0.62, bottom=1.25, wspace=0.42)
    rng = np.random.default_rng(0)
    y = {k: len(order) - 1 - i for i, k in enumerate(order)}
    per_condition = ctx.rows.groupby(["method", "condition"])[METRICS].mean()
    for ax, m in zip(axes, METRICS):
        for k in order:
            s = ctx.summary[(ctx.summary.method == k) & (ctx.summary.metric == m)]
            if s.empty:
                ax.text(1.02, y[k], "–", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=8.5)
                continue
            s = s.iloc[0]
            v = per_condition.loc[k][m].to_numpy()
            ax.scatter(v, y[k] + rng.uniform(-0.17, 0.17, len(v)), s=11, color=COLORS[k], alpha=0.32, linewidths=0)
            ax.plot([s.ci95_low, s.ci95_high], [y[k], y[k]], color=COLORS[k], lw=2.4, solid_capstyle="round", zorder=3)
            ax.scatter([s["mean"]], [y[k]], s=50, color=COLORS[k], edgecolors=SURFACE, linewidths=1.2, zorder=4)
            ax.text(1.02, y[k], fmt(m, s["mean"]), transform=ax.get_yaxis_transform(), va="center", color=INK_2, fontsize=8.5,
                    fontweight="bold" if k == "arm" else "normal")
        ax.set_title(LABELS[m] + (" (log scale)" if m == "mmd2" else ""))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=min(0, ax.get_xlim()[0]))
        if m == "mmd2":  # a few strong conditions would otherwise squash every mean against zero
            ax.set_xscale("symlog", linthresh=1e-3, linscale=0.45)
            ax.set_xticks([0, 1e-2, 1e-1], ["0", "0.01", "0.1"])
            ax.minorticks_off()
        if m in RMSE_METRICS:
            ax.axvline(ctx.floor[m], color=INK_2, lw=1.0, zorder=1)
            ax.text(ctx.floor[m], 0.012, " floor", transform=ax.get_xaxis_transform(), va="bottom", fontsize=8, color=INK_2)
    axes[0].set_yticks(list(y.values()), [ctx.labels[k] for k in y])
    axes[0].get_yticklabels()[order.index("arm")].set_fontweight("bold")
    for t in axes[0].get_yticklabels():
        t.set_color(INK)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    finish(plt, fig, ctx.dir / "1_scoreboard.png", f"How the best arm scores against the baselines ({ctx.split})",
           f"Lower is better. Big dot: mean over {ctx.meta['n_conditions']} held-out conditions, bar: 95% CI (bootstrap over "
           "conditions), faint dots: each condition. Number: the mean. Floor: sampling noise of the measured mean, which no method can beat.",
           REPLICATE_RMSE + " " + footnote(ctx))


def fig_paired(plt, ctx) -> None:
    """Paired difference of each baseline from the arm: the same conditions, compared one by one."""
    refs = [k for k in ctx.order if k != "arm"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14.5, 0.46 * len(refs) + 2.55), sharey=True)
    margins(fig, 3.05, 0.62, bottom=1.25, wspace=0.42)
    y = {k: len(refs) - 1 - i for i, k in enumerate(refs)}
    for ax, m in zip(axes, METRICS):
        ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
        for k in refs:
            p = ctx.paired[(ctx.paired.ref == k) & (ctx.paired.metric == m)]
            if p.empty or (k == "oracle" and m in RMSE_METRICS):  # not a limit, so not a comparison (paired.csv keeps it)
                ax.text(1.02, y[k], "–", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=8.5)
                continue
            p = p.iloc[0]
            clear = p.ci95_low > 0 or p.ci95_high < 0
            ax.plot([p.ci95_low, p.ci95_high], [y[k], y[k]], color=COLORS[k], lw=2.4, solid_capstyle="round", zorder=2)
            ax.scatter([p.delta_mean], [y[k]], s=50, color=COLORS[k] if clear else SURFACE, edgecolors=COLORS[k], linewidths=1.6, zorder=3)
            ax.text(1.02, y[k], f"{int(p.n_win)}/{int(p.n_conditions)}", transform=ax.get_yaxis_transform(), va="center",
                    color=INK_2, fontsize=8.5)
        ax.set_title(LABELS[m])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y.values()), [ctx.labels[k] for k in y])
    for t in axes[0].get_yticklabels():
        t.set_color(INK)
    axes[0].set_ylim(-0.6, len(refs) - 0.4)
    finish(plt, fig, ctx.dir / "2_paired_differences.png", f"Is the arm really better, condition by condition? ({ctx.split})",
           "Baseline error minus arm error on the same conditions: right of zero, the arm is better. Bar: 95% CI. "
           "Filled dot: CI excludes zero. Number: conditions the arm wins.",
           "Replicate-well RMSEs are left out because they are not a limit: they compare two half-size wells. " + footnote(ctx))


def fig_per_condition(plt, ctx) -> None:
    """Where each method wins and fails, strongest measured response at the top."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    cmap = LinearSegmentedColormap.from_list("error", RAMP)
    order, n = ctx.order, len(ctx.conds)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 0.40 * n + 2.45), sharey=True)
    margins(fig, 1.95, 0.25, top=1.12, bottom=0.82, wspace=0.05)
    cmap.set_bad(SURFACE)
    means = ctx.rows.groupby(["method", "condition"])[METRICS].mean()
    for ax, m, title in zip(axes, ("mmd2", "rmse_mean_deg50"), ("MMD² (colour on a log scale)", "RMSE of top-50 DEG means")):
        grid = np.array([[means[m].get((k, c), np.nan) for k in order] for c in ctx.conds.condition])
        positive = grid[np.isfinite(grid) & (grid > 0)]
        norm = LogNorm(positive.min(), positive.max()) if m == "mmd2" else Normalize(0, np.nanmax(grid))
        ax.imshow(np.ma.masked_invalid(np.where(grid <= 0, positive.min(), grid)), cmap=cmap, norm=norm, aspect="auto")
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(order)), [SHORT[k] for k in order], color=INK, fontsize=9)
        ax.set_title(title, pad=22)
        generative = [order.index(k) for k in GENERATIVE]
        for i in range(n):
            best = generative[int(np.nanargmin(grid[i, generative]))]
            for j in range(len(order)):
                v = grid[i, j]
                dark = np.isfinite(v) and norm(max(v, positive.min())) > 0.55
                ax.text(j, i, "–" if np.isnan(v) else fmt(m, v), ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if dark else INK, fontweight="bold" if j == best else "normal")
            ax.add_patch(plt.Rectangle((best - 0.47, i - 0.46), 0.94, 0.92, fill=False, edgecolor=INK, lw=1.2))
    axes[0].set_yticks(range(n), [condition_text(r) for r in ctx.conds.itertuples()], color=INK)
    finish(plt, fig, ctx.dir / "3_per_condition.png", f"Error on each held-out condition ({ctx.split})",
           "Strongest measured response at the top. Darker is worse. Box: the best of the three generative models on that condition.",
           REPLICATE_RMSE + " " + footnote(ctx))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan")


def fig_effect(plt, ctx) -> None:
    """Does the generated population's average position on PC1 follow the condition?"""
    keys = [k for k in WITH_CELLS if k in ctx.order]
    fig, axes = plt.subplots(1, len(keys), figsize=(14.5, 3.9), sharex=True, sharey=True)
    margins(fig, 0.6, 0.2, top=1.05, bottom=1.42, wspace=0.09)
    conds = ctx.conds.condition
    span = [float("inf"), -float("inf")]
    for ax, k in zip(axes, keys):
        ref = "oracle_ref" if k == "oracle" else "measured"
        x = np.array([profile(ctx, ref, c, "pc1")[0].mean() for c in conds])
        per_seed = np.array([[v.mean() for v in profile(ctx, k, c, "pc1")] for c in conds])
        y = per_seed.mean(1)
        for s in range(per_seed.shape[1]):
            ax.scatter(x, per_seed[:, s], s=12, color=COLORS[k], alpha=0.3, linewidths=0)
        ax.scatter(x, y, s=40, color=COLORS[k], edgecolors=SURFACE, linewidths=1.0, zorder=3)
        span = [min(span[0], x.min(), y.min()), max(span[1], x.max(), y.max())]
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        r = pearson(x, y)
        ax.set_title(SHORT[k] + (" (rep 2 vs rep 1)" if k == "oracle" else ""))
        fit = round(float(np.polyfit(x, y, 1)[0]), 2) + 0.0
        corner = (0.04, 0.96, "left", "top") if k == "dummy" else (0.97, 0.04, "right", "bottom")  # the empty side of each panel
        ax.text(corner[0], corner[1], f"slope = {fit:.2f}\n" + (f"corr = {r:.2f}\n" if np.isfinite(r) else "corr undefined\n") + f"RMSE = {rmse:.2f}",
                transform=ax.transAxes, ha=corner[2], va=corner[3], fontsize=8.5, color=INK_2)
        ax.set_xlabel("Measured mean PC1")
    pad = 0.06 * (span[1] - span[0])
    for ax in axes:
        ax.plot([span[0] - pad, span[1] + pad], [span[0] - pad, span[1] + pad], color=AXIS, lw=1.0, zorder=1)
        ax.set_xlim(span[0] - pad, span[1] + pad)
        ax.set_ylim(span[0] - pad, span[1] + pad)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Generated mean PC1")
    finish(plt, fig, ctx.dir / "4_effect_tracking.png", f"Does the generated population follow the condition on PC1? ({ctx.split})",
           "Each dot is one held-out condition. On the grey line, generated = measured. Faint dots: single seeds. Slope: least-squares fit.",
           pc_note(ctx) + footnote(ctx))


def fig_ecdf(plt, ctx) -> None:
    """Whole PC1 distributions, condition by condition, with each method's distance from the measured cells."""
    from scipy.stats import wasserstein_distance
    conds = ctx.conds
    ncols = 3 if len(conds) <= 9 else 5
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5 if ncols == 5 else 11.5, 2.15 * nrows + 2.5), sharex=True, sharey=True)
    margins(fig, 0.6, 0.2, top=1.36, bottom=1.4, hspace=0.32, wspace=0.07)
    measured = {c: profile(ctx, "measured", c, "pc1")[0] for c in conds.condition}
    lo, hi = np.percentile(np.concatenate(list(measured.values())), [0.3, 99.7])
    keys = ["dummy", "adjacent", "cellflow", "chemcpa", "arm"]
    w1 = {k: {c: wasserstein_distance(np.concatenate(profile(ctx, k, c, "pc1")), measured[c]) for c in measured} for k in keys}
    w1["oracle"] = {c: wasserstein_distance(profile(ctx, "oracle", c, "pc1")[0], profile(ctx, "oracle_ref", c, "pc1")[0]) for c in measured}
    pd.DataFrame(w1).rename_axis("condition").to_csv(ctx.dir / "w1_pc1.csv")
    for ax, row in zip(axes.flat, conds.itertuples()):
        for k in keys:
            v = np.sort(np.concatenate(profile(ctx, k, row.condition, "pc1")))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[k], lw=2.0 if k == "arm" else 1.4,
                    ls=(0, (5, 2.5)) if k == "dummy" else "-", label=f"{ctx.labels[k]} (W1 {np.mean(list(w1[k].values())):.2f})",
                    zorder=4 if k == "arm" else 2)
        v = np.sort(measured[row.condition])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=INK, lw=2.2, label="Measured cells", zorder=5)
        ax.set_title(condition_text(row), fontsize=9.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
    for ax in axes.flat[len(conds):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of cells")
    for i, ax in enumerate(axes.flat):
        if i >= len(conds) - ncols and i < len(conds):
            ax.set_xlabel("PC1")
            ax.tick_params(labelbottom=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = [5, 4, 3, 2, 1, 0]
    w, h = fig.get_size_inches()
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", bbox_to_anchor=(0.008, 1 - 0.68 / h),
               ncol=3, fontsize=8.5, handlelength=1.8, columnspacing=1.3)
    finish(plt, fig, ctx.dir / "5_pc1_distributions.png", f"Whole distributions on PC1 ({ctx.split})",
           f"Cumulative share of cells along PC1, seeds pooled. Closer to the black line is better. W1: mean distance from the measured cells; "
           f"oracle (replicate wells) {np.mean(list(w1['oracle'].values())):.2f}.", pc_note(ctx) + footnote(ctx))


def gene_points(ctx, k: str) -> dict:
    """Top-50 DEG mean change and SD of method k against measured, over all conditions (seeds averaged)."""
    m0, top = ctx.profiles["control_mean"].astype(np.float64), int(ctx.profiles["deg_top_k"])
    ref = "oracle_ref" if k == "oracle" else "measured"
    program = ctx.profiles["program_idx"]
    xs, ys, px, py, sx, sy, kept = [], [], [], [], [], [], []
    for c in ctx.conds.condition:
        full = profile(ctx, "measured", c, "mean")[0].astype(np.float64) - m0
        deg = np.argsort(-np.abs(full), kind="stable")[:top]
        real_mean, real_sd = profile(ctx, ref, c, "mean")[0], profile(ctx, ref, c, "sd")[0]
        mean, sd = np.mean(profile(ctx, k, c, "mean"), axis=0), np.mean(profile(ctx, k, c, "sd"), axis=0)
        xs.append(real_mean[deg] - m0[deg])
        ys.append(mean[deg] - m0[deg])
        px.append(real_mean[program] - m0[program])
        py.append(mean[program] - m0[program])
        sx.append(real_sd[deg])
        sy.append(sd[deg])
        kept.append(np.mean([np.sum(s.astype(np.float64) ** 2) / np.sum(real_sd.astype(np.float64) ** 2) for s in profile(ctx, k, c, "sd")]))
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "px": np.concatenate(px), "py": np.concatenate(py),
            "sx": np.concatenate(sx), "sy": np.concatenate(sy), "kept": float(np.median(kept))}


def fig_genes(plt, ctx) -> None:
    """Gene-level fidelity: is the size of the change right, and is the cell-to-cell spread right?"""
    keys = [k for k in WITH_CELLS if k in ctx.order]
    fig, axes = plt.subplots(3, len(keys), figsize=(14.5, 9.4), sharex="row", sharey="row")
    margins(fig, 0.9, 0.2, top=1.0, bottom=1.05, hspace=0.42, wspace=0.09)
    slope = lambda x, y: round(float(x @ y / (x @ x)), 2) + 0.0
    points = {k: gene_points(ctx, k) for k in keys}
    extent = lambda *names: max(np.abs(np.concatenate([p[a] for p in points.values() for a in names])).max(), 1e-6) * 1.05
    lim, lim_program, top = extent("x", "y"), extent("px", "py"), extent("sx", "sy")

    def change(ax, x, y, k, limit, label):
        ax.scatter(x, y, s=6, color=COLORS[k], alpha=0.4, linewidths=0, rasterized=True)
        ax.plot([-limit, limit], [-limit, limit], color=AXIS, lw=1.0, zorder=1)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.text(0.04, 0.96, f"slope = {slope(x, y):.2f}" + (f"\nr = {pearson(x, y):.2f}" if np.std(y) > 1e-6 else ""),
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
        ax.set_xlabel(label)

    for j, k in enumerate(keys):
        p, spread = points[k], axes[2, j]
        axes[0, j].set_title(SHORT[k] + (" (rep 2 vs rep 1)" if k == "oracle" else ""))
        change(axes[0, j], p["x"], p["y"], k, lim, "Measured change")
        change(axes[1, j], p["px"], p["py"], k, lim_program, "Measured change")
        spread.scatter(p["sx"], p["sy"], s=6, color=COLORS[k], alpha=0.4, linewidths=0, rasterized=True)
        spread.plot([0, top], [0, top], color=AXIS, lw=1.0, zorder=1)
        spread.set_xlim(0, top)
        spread.set_ylim(0, top)
        spread.text(0.97, 0.04, f"slope = {slope(p['sx'], p['sy']):.2f}\nall-gene variance kept = {p['kept']:.2f}", transform=spread.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color=INK_2)
        spread.set_xlabel("Measured SD")
    axes[0, 0].set_ylabel("Generated change,\ntop-50 DEGs")
    axes[1, 0].set_ylabel("Generated change,\nprogram genes")
    axes[2, 0].set_ylabel("Generated SD,\ntop-50 DEGs")
    finish(plt, fig, ctx.dir / "6_gene_fidelity.png", f"Gene-level fidelity ({ctx.split})",
           "Top: mean change from vehicle on each condition's top-50 DEGs. Middle: the same on the 97 response-program genes. Bottom: cell-to-cell SD. "
           "Grey line: generated = measured. Slopes go through the origin.", footnote(ctx))


def case_picks(ctx) -> list:
    """Picked by rule, not by result: the strongest and the median measured response."""
    return [(ctx.conds.iloc[0], "strongest response"), (ctx.conds.iloc[len(ctx.conds) // 2], "median response")]


def cells_2d(ctx, method: str, condition: str) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(profile(ctx, method, condition, "pc1")), np.concatenate(profile(ctx, method, condition, "pc2"))


def target_cells(path: Path, spec, condition: str) -> np.ndarray:
    """One condition from a prediction file, aligned to the frozen evaluation genes."""
    a = ad.read_h5ad(path)
    take = a.obs["condition"].astype(str).to_numpy() == condition
    if not take.any():
        raise ValueError(f"{path}: no cells for {condition}")
    return dense(spec.align(a.X[take], a.var_names))


def fig_umap_cloud(plt, arm: str, split: str, root: Path, panel: str) -> None:
    """A separate joint UMAP point cloud for the strongest-response case-study condition."""
    import umap

    ctx = load_context(root, arm, split)
    row = case_picks(ctx)[0][0]
    condition = row.condition
    data = Data(panel)
    task = next(t for t in data.tasks(split) if t.target == condition)
    rng = np.random.default_rng(0)

    adjacent = []
    for seed in range(3):
        mix_rng = np.random.default_rng([seed, zlib.crc32(condition.encode()), zlib.crc32(split.encode())])
        adjacent.append(data.X[data.mixture(task, {task.low: 1 - task.w_logdose, task.high: task.w_logdose}, mix_rng)])
    cells = {
        "measured": data.X[data.idx(condition)],
        "oracle": data.X[data.idx(condition, "rep2")],
        "dummy": data.X[data.ctrl],
        "adjacent": np.concatenate(adjacent),
    }
    for k in GENERATIVE:
        cells[k] = np.concatenate([target_cells(path, data.spec, condition)
                                   for _, path in generated_paths(root, k, arm, split)])

    keys = ["measured", "arm", "chemcpa", "cellflow", "adjacent", "dummy", "oracle"]
    sampled = {}
    for k in keys:
        take = rng.permutation(len(cells[k]))[:N_POINTS]
        sampled[k] = data.spec.project(cells[k][take])
    sizes = {k: len(sampled[k]) for k in keys}
    Z = np.concatenate([sampled[k] for k in keys])
    embedding = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="euclidean",
                          random_state=0, n_jobs=1).fit_transform(Z)
    points, start = {}, 0
    for k in keys:
        points[k] = embedding[start:start + sizes[k]]
        start += sizes[k]

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    margins(fig, 0.72, 0.25, top=1.45, bottom=1.05)
    # Interleave small batches so no method owns the top drawing layer throughout the cloud.
    for batch in range(10):
        for k in keys:
            p = points[k][batch::10]
            ax.scatter(p[:, 0], p[:, 1], s=18 if k == "measured" else 13,
                       marker="*" if k == "measured" else MARKERS[k],
                       color=INK if k == "measured" else COLORS[k], alpha=0.58,
                       edgecolors="none", zorder=2, rasterized=True)
    for k in keys:
        centre = points[k].mean(0)
        ax.scatter(*centre, s=155 if k == "measured" else 82,
                   marker="*" if k == "measured" else MARKERS[k],
                   color=INK if k == "measured" else COLORS[k], edgecolors=SURFACE,
                   linewidths=1.1, zorder=4)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(condition_text(row))
    handles = [plt.Line2D([], [], ls="", marker="*" if k == "measured" else MARKERS[k], ms=8,
                          color=INK if k == "measured" else COLORS[k], mec=SURFACE) for k in keys]
    names = ["Measured cells", *[ctx.labels[k] + (" (well 2)" if k == "oracle" else "") for k in keys[1:]]]
    w, h = fig.get_size_inches()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.67 / h),
               ncol=3, fontsize=8.5, handletextpad=0.35, columnspacing=1.25)
    finish(plt, fig, ctx.dir / "12_umap_point_cloud.png", f"Joint UMAP of cell populations ({ctx.split})",
           f"Up to {N_POINTS} cells per population; generated seeds pooled. Large markers: population centres in the embedding.",
           "UMAP was fit jointly on the frozen 50-PC cell representations. It is a descriptive view: local overlap is meaningful, "
           "but global distances and cluster sizes are not quantitative errors.")


def contour(ax, x: np.ndarray, y: np.ndarray, grid: tuple, rng, **style) -> None:
    """The outline holding 80% of a population, from a kernel density estimate."""
    from scipy.stats import gaussian_kde
    keep = rng.permutation(len(x))[:1500]
    gx, gy = grid
    density = gaussian_kde(np.vstack([x[keep], y[keep]]))(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ordered = np.sort(density.ravel())[::-1]
    ax.contour(gx, gy, density, levels=[ordered[np.searchsorted(np.cumsum(ordered) / ordered.sum(), 0.8)]], **style)


def fig_pc_overlay(plt, ctx) -> None:
    """Every method in the PC1 x PC2 plane: contour-led overlays, then every condition's centre."""
    rng = np.random.default_rng(0)
    keys = ["oracle", "dummy", "adjacent", "cellflow", "chemcpa", "arm"]
    style = {k: {"lw": 2.4 if k == "arm" else 1.5, "ls": (0, (1, 1.6)) if k == "oracle" else ((0, (5, 2.5)) if k == "dummy" else "-")} for k in keys}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    margins(fig, 0.6, 0.2, top=1.42, bottom=1.25, wspace=0.14)
    for ax, (row, why) in zip(axes, case_picks(ctx)):
        real = cells_2d(ctx, "measured", row.condition)
        pools = {k: cells_2d(ctx, k, row.condition) for k in keys}
        everything = [real, *pools.values()]
        x_lo, x_hi = (f(np.percentile(p[0], q) for p in everything) for f, q in ((min, 0.5), (max, 99.5)))
        y_lo, y_hi = (f(np.percentile(p[1], q) for p in everything) for f, q in ((min, 0.5), (max, 99.5)))
        grid = np.meshgrid(np.linspace(x_lo, x_hi, 130), np.linspace(y_lo, y_hi, 130))
        # A many-colour cell overlay becomes muddy exactly where the populations overlap.  Keep the
        # measured cells as neutral context and use contours for the methods; Fig. 11 shows each
        # method's individual cells without overplotting them with the other methods.
        shown = rng.permutation(len(real[0]))[:N_POINTS]
        ax.scatter(real[0][shown], real[1][shown], s=10, marker="o", color=MUTED,
                   alpha=0.28, linewidths=0, zorder=1)
        contour(ax, *real, grid, rng, colors=[INK], linewidths=[1.9], zorder=5)
        ax.scatter([real[0].mean()], [real[1].mean()], marker="*", s=170, color=INK, edgecolors=SURFACE, linewidths=0.8, zorder=8)
        for k in keys:
            contour(ax, *pools[k], grid, rng, colors=[COLORS[k]], linewidths=[style[k]["lw"]],
                    linestyles=[style[k]["ls"]], zorder=4 if k == "arm" else 3)
            ax.scatter([pools[k][0].mean()], [pools[k][1].mean()], marker=MARKERS[k], s=76,
                       color=COLORS[k], edgecolors=SURFACE, linewidths=1.1, zorder=7)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{condition_text(row)} ({why})")
        ax.set_xlabel("PC1 (vehicle = 0)")
    axes[0].set_ylabel("PC2")
    ax = axes[2]
    errors = []
    for c in ctx.conds.condition:
        real = [v.mean() for v in cells_2d(ctx, "measured", c)]
        for k in keys:
            x, y = (v.mean() for v in cells_2d(ctx, k, c))
            errors.append({"method": k, "condition": c, "pc1_error": x - real[0], "pc2_error": y - real[1]})
            ax.plot([real[0], x], [real[1], y], color=COLORS[k], lw=0.8, alpha=0.55, zorder=2)
            ax.scatter([x], [y], marker=MARKERS[k], s=30, color=COLORS[k], edgecolors=SURFACE, linewidths=0.6, zorder=3)
        ax.scatter([real[0]], [real[1]], marker="*", s=95, color=INK, edgecolors=SURFACE, linewidths=0.6, zorder=4)
    errors = pd.DataFrame(errors)
    errors.to_csv(ctx.dir / "pc_centre_errors.csv", index=False)
    rms = errors.groupby("method")[["pc1_error", "pc2_error"]].agg(lambda v: float(np.sqrt(np.mean(v ** 2))))
    print("RMS error of population centres over conditions\n" + rms.join(errors.groupby("method").pc2_error.mean().rename("pc2_bias")).to_string(float_format=lambda v: f"{v:.3f}"))
    ax.set_title(f"All {len(ctx.conds)} conditions: population centres")
    ax.set_xlabel("PC1 (vehicle = 0)")
    handles = [plt.Line2D([], [], color=INK, lw=2.4, marker="*", ms=11, mec=SURFACE)]
    handles += [plt.Line2D([], [], color=COLORS[k], lw=style[k]["lw"], ls=style[k]["ls"], marker=MARKERS[k], ms=7, mec=SURFACE) for k in reversed(keys)]
    names = ["Measured cells", *[ctx.labels[k] + (" (well 2)" if k == "oracle" else "") for k in reversed(keys)]]
    w, h = fig.get_size_inches()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.68 / h), ncol=4, fontsize=8.5, handlelength=2.4, columnspacing=1.4)
    finish(plt, fig, ctx.dir / "10_pc1_pc2_overlay.png", f"Every method in the PC1 x PC2 plane ({ctx.split})",
           "Grey points: measured cells. Coloured outlines: 80% population regions. Large markers: population centres. "
           "Right: each method's centre joins the measured star. See Fig. 11 for separate cell clouds.",
           f"PC2 holds only {100 * ctx.pc2_share:.1f}% of the variance between training conditions and mostly shows cell-to-cell spread, but centres still separate on it. "
           "The two axes have different scales. "
           + pc_note(ctx) + footnote(ctx))


def fig_pc_by_method(plt, ctx) -> None:
    """The same two conditions, one panel per method: its cells over the measured cells."""
    rng = np.random.default_rng(0)
    keys = [k for k in WITH_CELLS if k in ctx.order]
    fig, axes = plt.subplots(2, len(keys), figsize=(14.5, 6.7), sharex="row", sharey="row")
    margins(fig, 0.72, 0.2, top=1.0, bottom=1.42, hspace=0.36, wspace=0.07)
    for i, (row, why) in enumerate(case_picks(ctx)):
        limits = [cells_2d(ctx, k, row.condition) for k in keys]
        x_lo, x_hi = min(np.percentile(p[0], 0.5) for p in limits), max(np.percentile(p[0], 99.5) for p in limits)
        y_lo, y_hi = min(np.percentile(p[1], 0.5) for p in limits), max(np.percentile(p[1], 99.5) for p in limits)
        for j, k in enumerate(keys):
            ax = axes[i, j]
            real = cells_2d(ctx, "oracle_ref" if k == "oracle" else "measured", row.condition)
            x, y = cells_2d(ctx, k, row.condition)
            keep = rng.permutation(len(x))[:700]  # the same ink for every method, whatever its cell count
            ax.scatter(*real, s=7, color="#c9c8c1", linewidths=0, zorder=1)
            ax.scatter(x[keep], y[keep], s=7, color=COLORS[k], alpha=0.5, linewidths=0, zorder=2)
            ax.scatter([real[0].mean()], [real[1].mean()], marker="*", s=120, color=INK, edgecolors=SURFACE, linewidths=0.7, zorder=4)
            ax.scatter([x.mean()], [y.mean()], marker=MARKERS[k], s=56, color=COLORS[k], edgecolors=SURFACE, linewidths=0.9, zorder=5)
            gap = float(np.hypot(x.mean() - real[0].mean(), y.mean() - real[1].mean()))
            ax.text(0.96, 0.95, f"gap {gap:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("PC1")
            if i == 0:
                ax.set_title(SHORT[k] + (" (well 2 vs 1)" if k == "oracle" else ""))
        axes[i, 0].set_ylabel(f"{condition_text(row)}\nPC2")
    finish(plt, fig, ctx.dir / "11_pc1_pc2_by_method.png", f"Each method's cells over the measured cells ({ctx.split})",
           "Grey: measured cells. Colour: the method's cells, 700 drawn from pooled seeds. Star and coloured marker: their centres. Gap: distance between the two centres.",
           f"Top: strongest measured response. Bottom: median. PC2 holds only {100 * ctx.pc2_share:.1f}% of the variance between training conditions. " + pc_note(ctx) + footnote(ctx))


def fig_case_study(plt, ctx) -> None:
    """Two conditions picked by rule, the strongest and the median measured response, with every method overlaid."""
    picks = case_picks(ctx)
    keys = ["oracle", "dummy", "adjacent", "cellflow", "chemcpa", "arm"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    margins(fig, 0.3, 0.2, top=1.38, bottom=1.4, hspace=0.32, wspace=0.08)
    for i, (row, why) in enumerate(picks):
        for j, (what, xlabel) in enumerate((("pc1", "PC1 of all genes (vehicle = 0)"), ("effect", "Effect along the condition's measured direction"))):
            ax = axes[i, j]
            real = profile(ctx, "measured", row.condition, what)[0]
            pools = {k: np.concatenate(profile(ctx, k, row.condition, what)) for k in keys}
            # Bins span every method: clipping tails into the end bins would draw false spikes.
            edges = np.linspace(min(np.percentile(v, 0.5) for v in [real, *pools.values()]),
                                max(np.percentile(v, 99.5) for v in [real, *pools.values()]), 44)
            top = ax.hist(real, edges, density=True, color="#dcdbd4", edgecolor="none", label="Measured cells", zorder=1)[0].max()
            for k in keys:
                ax.hist(pools[k], edges, density=True, histtype="step", color=COLORS[k], lw=2.1 if k == "arm" else 1.3, zorder=4 if k == "arm" else 3,
                        ls=(0, (1, 1.6)) if k == "oracle" else ((0, (5, 2.5)) if k == "dummy" else "-"),
                        label=ctx.labels[k] + (" (well 2)" if k == "oracle" else ""))
            ax.set_ylim(0, 1.7 * top)  # vehicle's tall peak would otherwise flatten everything else
            ax.set_title(f"{condition_text(row)} ({why})")
            ax.set_xlabel(xlabel)
            ax.set_yticks([])
            ax.grid(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index("Measured cells"), *[labels.index(ctx.labels[k] + (" (well 2)" if k == "oracle" else "")) for k in reversed(keys)]]
    w, h = fig.get_size_inches()
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", bbox_to_anchor=(0.008, 1 - 0.66 / h), ncol=4, fontsize=8.5,
               handlelength=2.0, columnspacing=1.4)
    finish(plt, fig, ctx.dir / "9_case_study.png", f"Case study: the populations of two held-out conditions ({ctx.split})",
           "Strongest and median measured response. Left: PC1 of all genes. Right: the condition's top-50 DEGs, projected on its measured change. "
           "Seeds pooled; vehicle's peak can run off the top.", pc_note(ctx) + footnote(ctx))


def fig_arms(plt, ctx, root: Path) -> None:
    """Every arm on validation, to show the pick and how little separates the top arms."""
    table = pd.read_csv(root / "figures" / "arms.csv")
    order = table.groupby("variant").mmd2.mean().sort_values().index.tolist()
    metrics = ["mmd2", "rmse_mean_deg50", "rmse_mean_program", "w1_r"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 0.235 * len(order) + 2.5), sharey=True)
    margins(fig, 1.7, 0.3, top=1.05, bottom=1.0, wspace=0.09)
    y = {v: len(order) - 1 - i for i, v in enumerate(order)}
    tone = lambda v, teacher: COLORS["arm"] if v == ctx.arm else (RAMP[1] if teacher else MUTED)
    for ax, m in zip(axes, metrics):
        for _, r in table.iterrows():
            c = tone(r.variant, r.teacher)
            ax.scatter([r[m]], [y[r.variant]], s=24, color=SURFACE if r.ignored else c, edgecolors=c, linewidths=1.1, zorder=3)
        for v, part in table.groupby("variant"):
            c = tone(v, part.teacher.iloc[0])
            ax.plot([part[m].mean()] * 2, [y[v] - 0.34, y[v] + 0.34], color=INK if v == ctx.arm else INK_2, lw=1.4, zorder=4)
        dummy = float(ctx.summary[(ctx.summary.method == "dummy") & (ctx.summary.metric == m)]["mean"].iloc[0])
        for name, value in (bound(ctx, m), ("Dummy", dummy)):
            ax.axvline(value, color=INK_2 if name != "Dummy" else MUTED, lw=1.0, zorder=1)
            ax.text(value, len(order) - 0.15, name, ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_title(LABELS[m], pad=16)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y.values()), list(y), fontsize=8.5)
    axes[0].get_yticklabels()[order.index(ctx.arm)].set_fontweight("bold")
    axes[0].get_yticklabels()[order.index(ctx.arm)].set_color(INK)
    axes[0].set_ylim(-0.7, len(order) - 0.2)
    finish(plt, fig, ctx.dir / "7_arm_selection.png", "Every CellFlow arm on validation, sorted by MMD²",
           "Dots: the three seeds at their lowest validation MMD² checkpoint. Tick: mean. Dark blue: the pick. Light blue: other arms "
           "with teacher input. Hollow: that seed ended ignoring the condition. Floor: sampling noise of the measured mean.", footnote(ctx))


def fig_curves(plt, ctx, root: Path) -> None:
    """Training curves of the arm, its no-teacher backbone and the retrain (median of seeds; thin lines: seeds)."""
    curves = pd.read_csv(root / "figures" / "curves.csv")
    arms_table = pd.read_csv(root / "figures" / "arms.csv")
    lines = [(ctx.arm, COLORS["arm"], ctx.labels["arm"]), (BACKBONE, MUTED, "Same backbone, no teacher or average-arrow"),
             (BASE, COLORS["cellflow"], ROW_LABELS["cellflow"])]
    panels = [("val", "mmd2", "Validation MMD²"), ("val", "rmse_mean_deg50", "Validation RMSE DEG50"),
              ("train_fit", "var_ratio_median", "In-sample variance kept (generated / real)")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    margins(fig, 0.6, 0.2, top=1.15, bottom=1.15, wspace=0.17)
    ticks = [500, 1000, 2000, 5000, 10000, 30000]
    for ax, (task, m, title) in zip(axes, panels):
        tops = []
        for v, color, label in lines:
            part = curves[(curves.variant == v) & (curves.task == task)]
            for s, one in part.groupby("seed"):
                ax.plot(one.step, one[m], color=color, lw=0.8, alpha=0.35)
                step = arms_table[(arms_table.variant == v) & (arms_table.seed == s)].step.iloc[0]
                hit = one[one.step == step]
                ax.scatter(hit.step, hit[m], s=30, color=color, edgecolors=SURFACE, linewidths=0.9, zorder=4)
            med = part.groupby("step")[m].median()
            ax.plot(med.index, med.values, color=color, lw=2.0, label=label, zorder=3)
            tops.append(np.nanmax(med.values))
        if task == "val":  # seeds can explode; the axis follows the medians
            ax.axhline(bound(ctx, m)[1], color=INK_2, lw=1.0, zorder=1)
            ax.set_ylim(0, 1.08 * max(tops))
        else:
            ax.axhline(1.0, color=INK_2, lw=1.0, zorder=1)
            ax.set_ylim(0, 1.15)
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("Training step (log scale)")
    w, h = fig.get_size_inches()
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, lw=1.0))
    labels.append("Reference: oracle (MMD²), sampling floor (RMSE), all variance kept")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.66 / h), ncol=4, fontsize=8.5, handlelength=1.8)
    finish(plt, fig, ctx.dir / "8_training_curves.png", "When the arm learns, and when it gets worse",
           "Thick line: median of 3 seeds. Thin lines: seeds. Dots: each seed's selected checkpoint (lowest validation MMD²).",
           "Only the median sets the axis, so a seed that diverged can run off the top. " + footnote(ctx))


def draw(arm: str, split: str, root: Path) -> None:
    plt = setup()
    ctx = load_context(root, arm, split)
    fig_scoreboard(plt, ctx)
    fig_paired(plt, ctx)
    fig_per_condition(plt, ctx)
    fig_effect(plt, ctx)
    fig_ecdf(plt, ctx)
    fig_genes(plt, ctx)
    fig_case_study(plt, ctx)
    fig_pc_overlay(plt, ctx)
    fig_pc_by_method(plt, ctx)
    if split == "val":  # development views: selection and training are judged on validation only
        fig_arms(plt, ctx, root)
        fig_curves(plt, ctx, root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["arms", "collect", "draw", "umap"])
    p.add_argument("--arm")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    root = Path(a.root)
    if a.step == "arms":
        arms(root)
    elif not a.arm:
        p.error(f"{a.step} needs --arm")
    elif a.step == "collect":
        collect(a.arm, a.split, root, a.data)
    elif a.step == "umap":
        fig_umap_cloud(setup(), a.arm, a.split, root, a.data)
    else:
        draw(a.arm, a.split, root)


if __name__ == "__main__":
    main()
