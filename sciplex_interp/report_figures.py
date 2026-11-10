"""Report figures: one axis per plot, the restricted baseline set, for a report page.

Each figure is a single matplotlib axis sized for one column of a report, so it stays legible without
squeezing. The comparison is limited to four methods, dummy (vehicle) dropped:

  ours       the CellFlow arm with the condition response embedding and grouped velocity matching
             (ARM, z70_lr3e4_mv10_t). Called "Ours" in every plot.
  cellflow   the matched retrained CellFlow baseline (`base`).
  chemcpa    original chemCPA-Gauss (`gauss_v1`), the baseline without the mean-consistency fix.
  adjacent   adjacent-dose mixture of real neighbouring-dose cells: the trivial oracle, a reference.

It draws only from the files sciplex_interp.cellflow_figures collect wrote, so it never loads a model or
the panel. Collect the split with --chemcpa gauss_v1 first, so its `chemcpa` method is the original:

  python -m sciplex_interp.cellflow_figures collect --arm z70_lr3e4_mv10_t --split test --chemcpa gauss_v1
  python -m sciplex_interp.report_figures draw --arm z70_lr3e4_mv10_t --split test

Figures (written to <root>/figures/<arm>_<split>/report/):
  cmp_<metric>.png          one per metric: each method's mean over conditions and its 95% CI
  paired_<metric>.png       one per metric: ours minus each baseline, condition by condition
  effect_pc1.png            measured vs generated mean position on PC1, every method on one axis
  ecdf_case.png             one condition's whole PC1 distribution, every method against the measured cells
  gene_change.png           ours: generated vs measured mean change on each condition's top-50 DEGs
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cellflow_figures import (AXIS, COLORS, INK, INK_2, MARKERS, MUTED, SURFACE, cells_2d, condition_text,
                               fmt, load_context, pearson, profile, setup)
from .chemcpa_compare import LABELS
from .core import DEFAULT_PANEL, DEFAULT_ROOT

# The four methods, in the order the report lists them (top to bottom on the dot plots); "arm" is ours.
ORDER = ["adjacent", "arm", "chemcpa", "cellflow"]
BASELINES = ["adjacent", "chemcpa", "cellflow"]  # everything ours is compared against, oracle at the top
LABELS_METHOD = {"arm": "Ours", "cellflow": "CellFlow (baseline)", "chemcpa": "chemCPA (baseline)",
                 "adjacent": "Adjacent-dose mixture (oracle)"}
SHORT = {"arm": "Ours", "cellflow": "CellFlow", "chemcpa": "chemCPA", "adjacent": "Adjacent"}
METRICS = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program"]  # response-score/W1 excluded
RMSE_METRICS = ["rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program"]
N_BARS = 4  # conditions shown in the per-condition bar chart, picked by measured-response quantile (a rule)


def slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


def report_dir(ctx) -> Path:
    d = ctx.dir / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def present(ctx, methods: list[str]) -> list[str]:
    return [k for k in methods if k in set(ctx.rows.method)]


def finish(plt, fig, path: Path, title: str, caption: str) -> None:
    """A compact title and a one-line caption, sized for a report figure (not the big analysis block)."""
    import textwrap
    w, h = fig.get_size_inches()
    fig.text(0.015, 1 - 0.20 / h, title, fontsize=11, fontweight="bold", va="top", color=INK)
    if caption:
        fig.text(0.015, 0.07 / h, "\n".join(textwrap.wrap(caption, int(w * 15))), fontsize=7.5, va="bottom", color=INK_2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_cmp(plt, ctx, metric: str) -> None:
    """One metric: each method's mean over conditions (big dot), its 95% CI (bar), each condition (faint)."""
    order = present(ctx, ORDER)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.2, 0.52 * len(order) + 1.5))
    fig.subplots_adjust(left=0.30, right=0.90, top=1 - 0.62 / (0.52 * len(order) + 1.5), bottom=0.9 / (0.52 * len(order) + 1.5))
    y = {k: len(order) - 1 - i for i, k in enumerate(order)}
    per_condition = ctx.rows.groupby(["method", "condition"])[metric].mean()
    for k in order:
        s = ctx.summary[(ctx.summary.method == k) & (ctx.summary.metric == metric)]
        if s.empty:
            continue
        s = s.iloc[0]
        v = per_condition.loc[k].to_numpy()
        ax.scatter(v, y[k] + rng.uniform(-0.17, 0.17, len(v)), s=12, color=COLORS[k], alpha=0.32, linewidths=0)
        ax.plot([s.ci95_low, s.ci95_high], [y[k], y[k]], color=COLORS[k], lw=2.6, solid_capstyle="round", zorder=3)
        ax.scatter([s["mean"]], [y[k]], s=58, color=COLORS[k], edgecolors=SURFACE, linewidths=1.3, zorder=4)
        ax.text(1.02, y[k], fmt(metric, s["mean"]), transform=ax.get_yaxis_transform(), va="center",
                color=INK, fontsize=8.5, fontweight="bold" if k == "arm" else "normal")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    if metric == "mmd2":
        ax.set_xscale("symlog", linthresh=1e-3, linscale=0.45)
        ax.set_xticks([0, 1e-2, 1e-1], ["0", "0.01", "0.1"])
        ax.minorticks_off()
    if metric in RMSE_METRICS:
        ax.axvline(ctx.floor[metric], color=INK_2, lw=1.0, zorder=1)
        ax.text(ctx.floor[metric], 0.015, " floor", transform=ax.get_xaxis_transform(), va="bottom", fontsize=7.5, color=INK_2)
    ax.set_yticks(list(y.values()), [LABELS_METHOD[k] for k in y])
    ax.get_yticklabels()[order.index("arm")].set_fontweight("bold")
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(f"{LABELS[metric]}  (lower is better)")
    floor = "  Floor: sampling noise of the measured mean, which no method can beat." if metric in RMSE_METRICS else ""
    finish(plt, fig, report_dir(ctx) / f"cmp_{metric}.png", f"{LABELS[metric]} on {ctx.meta['n_conditions']} held-out conditions ({ctx.split})",
           f"Dot: mean over conditions. Bar: 95% CI (bootstrap over conditions). Faint dots: each condition.{floor} "
           "Adjacent mixture resamples real neighbouring-dose cells, so it is a reference, not a generator.")


def fig_condition_bars(plt, ctx, metric: str = "mmd2") -> None:
    """A few conditions, four bars each (one per method): the metric on that single condition. Replaces paired."""
    keys = present(ctx, ["arm", "cellflow", "chemcpa", "adjacent"])  # ours first in every group
    n = min(N_BARS, len(ctx.conds))
    picks = ctx.conds.iloc[:n]  # the strongest measured responses (ctx.conds is sorted by response), where methods separate
    per = ctx.rows.groupby(["method", "condition"])[metric].mean()
    fig, ax = plt.subplots(figsize=(1.45 * n + 1.8, 4.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.34)
    x = np.arange(n)
    width = 0.80 / len(keys)
    top = 0.0
    for i, k in enumerate(keys):
        vals = [per.get((k, c), np.nan) for c in picks.condition]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width, color=COLORS[k], label=SHORT[k], zorder=3)
        top = max(top, np.nanmax(vals))
    ax.set_ylim(0, 1.26 * top)  # headroom so the legend clears the tallest bar
    ax.set_xticks(x, [condition_text(r) for r in picks.itertuples()], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel(f"{LABELS[metric]}  (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=7.5, ncol=len(keys), handlelength=1.1, columnspacing=1.0, borderpad=0.3)
    finish(plt, fig, report_dir(ctx) / f"bars_{metric}.png", f"{LABELS[metric]} on individual conditions ({ctx.split})",
           f"Each group is one held-out condition, four bars one per method. The {n} strongest measured responses "
           "(a rule, independent of result), strongest at left; this is where methods separate. Lower is better.")


def fig_condition_hists(plt, ctx) -> None:
    """One PC1 histogram per held-out condition: every method's cells against the measured cells, one axis each."""
    out = report_dir(ctx) / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    keys = present(ctx, ORDER)
    for row in ctx.conds.itertuples():
        c = row.condition
        measured = profile(ctx, "measured", c, "pc1")[0]
        pools = {k: np.concatenate(profile(ctx, k, c, "pc1")) for k in keys}
        lo = min(np.percentile(v, 0.5) for v in [measured, *pools.values()])
        hi = max(np.percentile(v, 99.5) for v in [measured, *pools.values()])
        edges = np.linspace(lo, hi, 40)
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.26)
        top = ax.hist(measured, edges, density=True, color="#dcdbd4", edgecolor="none", label="Measured cells", zorder=1)[0].max()
        for k in keys:
            ax.hist(pools[k], edges, density=True, histtype="step", color=COLORS[k], lw=2.1 if k == "arm" else 1.3,
                    ls=(0, (5, 2.5)) if k == "adjacent" else "-", label=SHORT[k], zorder=4 if k == "arm" else 3)
        ax.set_ylim(0, 1.5 * max(top, 1e-9))
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("PC1 of all genes (vehicle = 0)")
        ax.legend(loc="upper right", fontsize=7.5, handlelength=1.6)
        finish(plt, fig, out / f"hist_{slug(c)}.png", f"PC1 distribution: {condition_text(row)} ({ctx.split})",
               "Measured cells (filled) against each method (outlines), seeds pooled. Closer to the grey shape is better. "
               "Adjacent mixture resamples real neighbouring-dose cells, so it is a reference, not a generator.")


def fig_effect(plt, ctx) -> None:
    """Measured vs generated mean position on PC1, every method on one axis: does the population follow the condition?"""
    keys = present(ctx, ORDER)
    conds = ctx.conds.condition
    fig, ax = plt.subplots(figsize=(4.9, 4.9))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.21)
    x = np.array([profile(ctx, "measured", c, "pc1")[0].mean() for c in conds])
    span = [x.min(), x.max()]
    handles = []
    for k in keys:
        y = np.array([np.mean([v.mean() for v in profile(ctx, k, c, "pc1")]) for c in conds])
        span = [min(span[0], y.min()), max(span[1], y.max())]
        ax.scatter(x, y, s=34, marker=MARKERS.get(k, "o"), color=COLORS[k], edgecolors=SURFACE, linewidths=0.8, zorder=3,
                   label=f"{SHORT[k]} (slope {np.polyfit(x, y, 1)[0]:.2f}, r {pearson(x, y):.2f})")
    pad = 0.06 * (span[1] - span[0])
    ax.plot([span[0] - pad, span[1] + pad], [span[0] - pad, span[1] + pad], color=AXIS, lw=1.0, zorder=1)
    ax.set_xlim(span[0] - pad, span[1] + pad)
    ax.set_ylim(span[0] - pad, span[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured mean PC1")
    ax.set_ylabel("Generated mean PC1")
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.3, borderpad=0.3)
    finish(plt, fig, report_dir(ctx) / "effect_pc1.png", f"Does the population follow the condition on PC1? ({ctx.split})",
           "Each marker is one held-out condition. On the grey line, generated = measured. PC1 is the first principal "
           "component of all genes; vehicle = 0.")


def fig_ecdf(plt, ctx) -> None:
    """One condition's whole PC1 distribution, every method against the measured cells."""
    from scipy.stats import wasserstein_distance
    row = ctx.conds.iloc[0]  # strongest measured response, picked by rule
    condition = row.condition
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.30)
    measured = profile(ctx, "measured", condition, "pc1")[0]
    for k in present(ctx, ORDER):
        v = np.sort(np.concatenate(profile(ctx, k, condition, "pc1")))
        w1 = wasserstein_distance(v, measured)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[k], lw=2.2 if k == "arm" else 1.5,
                ls=(0, (5, 2.5)) if k == "adjacent" else "-", zorder=4 if k == "arm" else 2,
                label=f"{SHORT[k]} (W1 {w1:.2f})")
    v = np.sort(measured)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=INK, lw=2.4, label="Measured cells", zorder=5)
    ax.set_xlabel("PC1 of all genes (vehicle = 0)")
    ax.set_ylabel("Share of cells")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=7.5, handlelength=1.8)
    finish(plt, fig, report_dir(ctx) / "ecdf_case.png", f"Whole distribution on PC1: {condition_text(row)} ({ctx.split})",
           "Cumulative share of cells along PC1, seeds pooled. Closer to the black line is better. "
           "W1: distance from the measured cells. Strongest-response condition, picked by rule.")


def gene_change(ctx, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Measured and method mean change on each condition's top-50 DEGs, concatenated over conditions."""
    m0, top = ctx.profiles["control_mean"].astype(np.float64), int(ctx.profiles["deg_top_k"])
    xs, ys = [], []
    for c in ctx.conds.condition:
        deg = np.argsort(-np.abs(profile(ctx, "measured", c, "mean")[0].astype(np.float64) - m0), kind="stable")[:top]
        xs.append(profile(ctx, "measured", c, "mean")[0][deg] - m0[deg])
        ys.append(np.mean(profile(ctx, method, c, "mean"), axis=0)[deg] - m0[deg])
    return np.concatenate(xs), np.concatenate(ys)


def fig_gene_change(plt, ctx) -> None:
    """Ours: generated vs measured mean change on each condition's top-50 DEGs, one point per gene per condition."""
    slope = lambda x, y: float(x @ y / (x @ x))
    x, y = gene_change(ctx, "arm")
    slopes = {k: slope(*gene_change(ctx, k)) for k in present(ctx, ORDER)}
    lim = max(np.abs(np.concatenate([x, y])).max(), 1e-6) * 1.05
    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.85, bottom=0.30)
    ax.scatter(x, y, s=7, color=COLORS["arm"], alpha=0.4, linewidths=0, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1.0, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Measured change")
    ax.set_ylabel("Generated change (Ours)")
    ax.text(0.04, 0.96, f"Ours slope = {slopes['arm']:.2f}\nr = {pearson(x, y):.2f}", transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
    others = ", ".join(f"{SHORT[k]} {slopes[k]:.2f}" for k in present(ctx, BASELINES))
    finish(plt, fig, report_dir(ctx) / "gene_change.png", f"Is the size of the change right? ({ctx.split})",
           f"Each point: one of a condition's top-50 differentially expressed genes, over all {ctx.meta['n_conditions']} conditions. "
           f"On the grey line, generated = measured. Slope through the origin. Other methods' slopes: {others}.")


# ---- dose story (validation drugs) --------------------------------------------------------------
# One axis for "does the model follow dose?", and one per example drug. chemCPA is the original here too.
DOSE_LABELS = {"arm": "Ours", "gauss_v1": "chemCPA (baseline)", "base": "CellFlow (baseline)", "oracle": "Oracle: replicate well"}
DOSE_SHORT = {"arm": "Ours", "gauss_v1": "chemCPA", "base": "CellFlow", "oracle": "Oracle"}


def dose_dir(root: Path) -> Path:
    return root / "figures" / "dose"


def dose_report_dir(root: Path) -> Path:
    d = dose_dir(root) / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dose_models(arm: str) -> tuple[dict, dict, list]:
    """Map the report's method keys to the dose sweep's model names, and give colours and the plotting order."""
    from .dose_sweep import BASE, CHEMCPA_ORIGINAL
    names = {"arm": arm, "gauss_v1": CHEMCPA_ORIGINAL, "base": BASE}  # the model name in the dose files
    colors = {"arm": COLORS["arm"], "gauss_v1": COLORS["chemcpa"], "base": COLORS["cellflow"], "oracle": COLORS["oracle"]}
    return names, colors, ["oracle", "arm", "gauss_v1", "base"]


def fig_dose_gain(plt, root: Path, arm: str) -> None:
    """One number per drug for 'does the model follow dose?': the dose gain, with the median over drugs."""
    import json
    names, colors, order = dose_models(arm)
    table = pd.read_csv(dose_dir(root) / "gain_by_drug.csv")
    summary = pd.read_csv(dose_dir(root) / "gain_median.csv").set_index(["axis", "model"])
    whole = table[(table.axis == "effect") & (table.seed == -1)]
    gain = whole.pivot(index="drug", columns="model", values="gain")
    cases = json.loads((dose_dir(root) / "gain_cases.json").read_text())
    case_drugs = [cases["median"], cases["q75"]]
    model_of = lambda k: "oracle" if k == "oracle" else names[k]
    h = 0.52 * len(order) + 1.6
    fig, ax = plt.subplots(figsize=(6.4, h))
    fig.subplots_adjust(left=0.30, right=0.89, top=1 - 0.62 / h, bottom=1.0 / h)
    y = {k: len(order) - 1 - i for i, k in enumerate(order)}
    offset = dict(zip(gain.index, np.linspace(-0.22, 0.22, len(gain))))
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    ax.axvline(1, color=INK_2, lw=1.0, zorder=1)
    for k in order:
        m = model_of(k)
        ax.scatter(gain[m], [y[k] + offset[d] for d in gain.index], s=30, color=colors[k], alpha=0.55, linewidths=0, zorder=3)
        for d in case_drugs:  # ring the two example drugs shown at right
            ax.scatter([gain.loc[d, m]], [y[k] + offset[d]], s=66, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=4)
        value = summary.loc[("effect", m), "gain"]
        ax.plot([value, value], [y[k] - 0.34, y[k] + 0.34], color=colors[k], lw=3.4, solid_capstyle="round", zorder=5)
        ax.text(1.02, y[k], f"{value:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK,
                fontweight="bold" if k == "arm" else "normal")
    ax.set_yticks(list(y.values()), [DOSE_LABELS[k] for k in y])
    ax.get_yticklabels()[order.index("arm")].set_fontweight("bold")
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.12, 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dose gain  (0: ignores dose, 1: full response)")
    finish(plt, fig, dose_report_dir(root) / "dose_gain.png", "Does the model follow dose? (validation drugs)",
           "Dose gain per drug: slope of generated on measured mean effect over a drug's 4 doses, along its top-50 DEGs. "
           "Each dot a drug; bar and number: median over 9 drugs. Ringed: the two example drugs. chemCPA is the original (no mean consistency).")


def fig_dose_case(plt, root: Path, arm: str, which: str) -> None:
    """One example drug's dose-response curve: mean effect over its 4 doses, every method on one axis."""
    import json
    from matplotlib.lines import Line2D

    from .cellflow_figures import dose_text
    from .dose_sweep import MEASURED_NM
    names, colors, order = dose_models(arm)
    curves = pd.read_csv(dose_dir(root) / "curves.csv")
    table = pd.read_csv(dose_dir(root) / "gain_by_drug.csv")
    gain = table[(table.axis == "effect") & (table.seed == -1)].pivot(index="drug", columns="model", values="gain")
    cases = json.loads((dose_dir(root) / "gain_cases.json").read_text())
    drug = cases[which]
    drawn = ["arm", "gauss_v1", "base"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.85, bottom=0.30)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    for k in reversed(drawn):
        part = curves[(curves.model == names[k]) & (curves.drug == drug)]
        by = part.groupby("dose").effect
        ax.fill_between(by.min().index, by.min(), by.max(), color=colors[k], alpha=0.14, linewidths=0)
        ax.plot(by.mean().index, by.mean(), color=colors[k], lw=2.4 if k == "arm" else 1.7, zorder=4 if k == "arm" else 3,
                label=f"{DOSE_SHORT[k]} (gain {gain.loc[drug, names[k]]:.2f})")
    r = curves[(curves.model == "measured") & (curves.drug == drug)].sort_values("dose")
    ax.plot(r.dose, r.effect, color=INK, lw=1.4, zorder=5, label="Measured")
    seen = r[~r.unseen]
    ax.scatter(seen.dose, seen.effect, s=34, color=INK, zorder=6)
    ax.scatter(r[r.unseen].dose, r[r.unseen].effect, s=44, color=SURFACE, edgecolors=INK, linewidths=1.7, zorder=6)
    ax.set_xscale("log")
    ax.set_xticks(list(MEASURED_NM), [dose_text(t) for t in MEASURED_NM])
    ax.minorticks_off()
    ax.set_xlabel("Dose")
    ax.set_ylabel("Mean effect along the drug's direction")
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.8)
    name = "median" if which == "median" else "75th-percentile"
    finish(plt, fig, dose_report_dir(root) / f"dose_case_{which}.png", f"{drug.split(' (')[0]} across doses ({name} case)",
           "Mean effect of each population across the drug's 4 doses; line: mean of 3 seeds, band: seed range. Open circle: the held-out "
           "dose, which no model trained on. The example drug is picked by the baselines' gain, so Ours plays no part in the pick.")


def draw(arm: str, split: str, root: Path) -> None:
    plt = setup()
    ctx = load_context(root, arm, split)
    for metric in METRICS:
        fig_cmp(plt, ctx, metric)
    fig_condition_bars(plt, ctx, "mmd2")  # per-condition bars, in place of the paired figures
    fig_condition_hists(plt, ctx)
    fig_effect(plt, ctx)
    fig_ecdf(plt, ctx)
    fig_gene_change(plt, ctx)


def draw_dose(arm: str, root: Path) -> None:
    plt = setup()
    fig_dose_gain(plt, root, arm)
    for which in ("median", "q75"):
        fig_dose_case(plt, root, arm, which)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["draw", "dose"])
    p.add_argument("--arm", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    if a.step == "dose":
        draw_dose(a.arm, Path(a.root))
    else:
        draw(a.arm, a.split, Path(a.root))


if __name__ == "__main__":
    main()
