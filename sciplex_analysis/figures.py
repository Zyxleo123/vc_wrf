"""Figures for the six WRF analyses, drawn only from analysis tables and extracts.

One figure family per analysis, for each model (chemCPA, CellFlow) and split:
  s1_ecdf / s1_density / s1_moments   response-distribution shape
  s2_per_condition                     who benefits, who gets worse
  s3_thresholds                        threshold sensitivity and W1
  s4_expression                        expression fidelity
  s5_forest, s5_lambda_chemcpa         ablations
  s6_baselines_uncertainty             interpolation baseline, bootstrap, seeds, measured noise
  overview                             headline numbers for both models
CellFlow panels that need generated cells fall back to the 13-threshold CSV, or say
that the PSC extract is pending. Writes <root>/figures/*.png and <root>/report.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .common import (AXIS, DEFAULT_ROOT, GRID, GRID13_QS, INK, INK_2, MODELS, MUTED, PRIMARY_Q, REFERENCES, ROLES,
                     SPEC_QS, extract_path, method_label, role_method, short_condition)

SURFACE = "#fcfcfb"
MARKERS = {"base": "o", "wrf3": "s", "wrf13": "^", "mmd": "D"}
ROLE_ORDER = ["base", "wrf3", "wrf13", "mmd"]
SCATTER_ROLES = ["base", "wrf3", "wrf13"]  # validated all-pairs cap: three categorical slots
DIVERGING = LinearSegmentedColormap.from_list("signed", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a8302f"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
                     "legend.fontsize": 8, "axes.titlecolor": INK, "text.color": INK})


def style(ax, grid_axis="both"):
    ax.grid(color=GRID, linewidth=0.7, axis=grid_axis)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def pending(ax, text="needs CellFlow generated cells\n(PSC extract pending; see report)"):
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=9, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path.with_suffix(".png")


class Data:
    def __init__(self, root: Path):
        self.root = root
        t = root / "tables"
        read = lambda name: pd.read_csv(t / name)
        self.pc, self.thr, self.paired = read("per_condition.csv"), read("thresholds.csv"), read("paired.csv")
        self.paired_thr, self.methods = read("paired_thresholds.csv"), read("methods.csv")
        self.conditions, self.variability = read("conditions.csv"), read("measured_variability.csv")
        self.noise, self.dist, self.avail = read("generation_noise.csv"), read("distribution_summary.csv"), read("availability.csv")
        self.summary = json.loads((t / "summary.json").read_text())
        z = np.load(root / "extracts" / "measured.npz")
        self._mr = pd.Series(np.arange(len(z["r"]))).groupby(z["condition"].astype(str)).indices
        self._r = z["r"]
        self.control_r = z["r"][z["split"].astype(str) == "control"]
        self.primary_tau = self.summary["taus"]["primary"]
        self.spec_taus = np.quantile(self.control_r, SPEC_QS)

    def measured_r(self, condition):
        return self._r[self._mr[condition]]

    def has_extract(self, model, split):
        a = self.avail[(self.avail.model == model) & (self.avail.split == split)
                       & self.avail.role.isin(["base", "wrf3", "wrf13"])]
        return len(a) > 0 and bool(a.extract.all())

    def generated_r(self, model, method, split, condition):
        parts = []
        for seed in range(3):
            path = extract_path(self.root, model, method, seed, split, 0)
            if not path.exists():
                continue
            z = np.load(path)
            conds = [str(c) for c in z["conditions"]]
            off = np.concatenate([[0], np.cumsum(z["n_pred"])])
            i = conds.index(condition)
            parts.append(z["r"][off[i]:off[i + 1]])
        return np.concatenate(parts) if parts else None

    def roles(self, model, which=ROLE_ORDER):
        return [(r, role_method(model, r)) for r in which if role_method(model, r)]

    def split_conditions(self, split):
        c = self.conditions[self.conditions.split == split]
        return c.sort_values("wrf_real_q0.95").condition.tolist()

    def seed_mean(self, model, method, split):
        part = self.pc[(self.pc.model == model) & (self.pc.method == method) & (self.pc.split == split)]
        return part.groupby("condition").mean(numeric_only=True)

    def pair(self, model, contrast, split, metric):
        p = self.paired[(self.paired.model == model) & (self.paired.contrast == contrast)
                        & (self.paired.split == split) & (self.paired.metric == metric)]
        return None if p.empty else p.iloc[0]


def ecdf(ax, r, **kw):
    r = np.sort(r)
    ax.step(np.r_[r[0], r], np.r_[0, np.arange(1, len(r) + 1) / len(r)], where="post", **kw)


def mark_taus(ax, D, primary_only=False, vertical=True):
    for q, t in zip(SPEC_QS, D.spec_taus):
        if primary_only and q != PRIMARY_Q:
            continue
        (ax.axvline if vertical else ax.axhline)(t, color=INK_2, linewidth=1.1 if q == PRIMARY_Q else 0.6,
                                                 linestyle=(0, (4, 2)), alpha=0.9 if q == PRIMARY_Q else 0.5, zorder=1)


# ---------------------------------------------------------------------------------------------
# 1. Response distributions
def fig_ecdf(D: Data, model, split, out):
    conds = D.split_conditions(split)
    extract = D.has_extract(model, split)
    roles = D.roles(model, SCATTER_ROLES)
    ncols = 6
    nrows = int(np.ceil((len(conds) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows + 0.8), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = axes.ravel()
    thr = D.thr[(D.thr.model == model) & (D.thr.split == split) & (D.thr.grid == "grid13")]
    for ax, c in zip(axes, conds):
        ecdf(ax, D.measured_r(c), color=INK, linewidth=2.2, label="Measured", zorder=5)
        for role, method in roles:
            if extract:
                r = D.generated_r(model, method, split, c)
                ecdf(ax, r, color=ROLES[role]["color"], linewidth=1.5, label=method_label(model, method), zorder=4)
            else:
                part = thr[(thr.method == method) & (thr.condition == c)].groupby("tau").f_pred.mean()
                ax.plot(part.index, part.values, color=ROLES[role]["color"], marker=MARKERS[role], markersize=4,
                        linewidth=1.3, label=method_label(model, method), zorder=4)
        mark_taus(ax, D, primary_only=True)
        meas = D.conditions.set_index("condition").loc[c, "wrf_real_q0.95"]
        ax.set_title(short_condition(c), fontsize=8.5)
        ax.text(0.03, 0.95, f"measured WRF {100 * meas:.0f}%", transform=ax.transAxes, fontsize=7, color=INK_2, va="top")
        style(ax)
        ax.set_xlim(-2.6, 3.9)
    for ax in axes[len(conds):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, linestyle=(0, (4, 2)), linewidth=1.1))
    labels.append("Primary τ (q = 0.95)")
    axes[len(conds)].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    for ax in axes[::ncols]:
        ax.set_ylabel("Fraction of cells with r < t")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Response score t")
    how = ("pooled over 3 training seeds" if extract else
           "CellFlow: 13-threshold CDF points from docs/cellflow/thresholds.csv, seed mean (full curves need the PSC extract)")
    fig.suptitle(f"{MODELS[model]['label']}, {split}: response-score CDFs per held-out condition "
                 f"(sorted by measured WRF; {how})", fontsize=11, color=INK, fontweight="bold")
    return save(fig, out / f"s1_ecdf_{model}_{split}")


def selected_conditions(D: Data, model, split):
    """Most improved, most worsened (3-τ loss vs λ=0), most and least potent."""
    base, new = role_method(model, "base"), role_method(model, "wrf3")
    eb = D.seed_mean(model, base, split)["wrf_err_pp_q0.95"]
    en = D.seed_mean(model, new, split)["wrf_err_pp_q0.95"]
    delta = (eb - en).sort_values()
    meas = D.conditions[D.conditions.split == split].set_index("condition")["wrf_real_q0.95"].sort_values()
    picks = [("largest gain", delta.index[-1]), ("largest loss", delta.index[0]),
             ("lowest measured WRF", meas.index[0]), ("highest measured WRF", meas.index[-1])]
    seen, out = set(), []
    for why, c in picks:
        if c not in seen:
            seen.add(c)
            out.append((why, c, delta[c]))
    return out


def fig_density(D: Data, model, split, out):
    if not D.has_extract(model, split):
        return None
    picks = selected_conditions(D, model, split)
    roles = D.roles(model, SCATTER_ROLES)
    fig, axes = plt.subplots(len(roles), len(picks), figsize=(4.2 * len(picks), 2.3 * len(roles) + 0.9),
                             sharex=True, constrained_layout=True)
    bins = np.linspace(-2.6, 3.9, 66)
    for j, (why, c, delta) in enumerate(picks):
        r_m = D.measured_r(c)
        for i, (role, method) in enumerate(roles):
            ax = axes[i, j]
            ax.hist(r_m, bins=bins, density=True, color="#d6d5cf", label="Measured")
            r = D.generated_r(model, method, split, c)
            ax.hist(r, bins=bins, density=True, histtype="step", linewidth=1.8, color=ROLES[role]["color"],
                    label=method_label(model, method))
            mark_taus(ax, D)
            fm, fp = np.mean(r_m < D.primary_tau), np.mean(r < D.primary_tau)
            ax.text(0.02, 0.96, f"μ {r_m.mean():.2f} → {r.mean():.2f}\nv {r_m.var():.2f} → {r.var():.2f}\n"
                    f"WRF {100 * fm:.0f}% → {100 * fp:.0f}%", transform=ax.transAxes, va="top", fontsize=7.2,
                    color=INK_2, family="monospace")
            style(ax)
            if i == 0:
                ax.set_title(f"{short_condition(c)}\n{why}; Δ = {delta:+.1f} pp", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ROLES[role]['label']}\ndensity", fontsize=8)
            if i == len(roles) - 1:
                ax.set_xlabel("Response score r (dashed: τ at q = 0.9 / 0.95 / 0.975)")
    fig.suptitle(f"{MODELS[model]['label']}, {split}: measured (grey) vs generated (line) response scores; "
                 "labels show measured → generated mean μ, variance v and WRF at the primary τ",
                 fontsize=10.5, fontweight="bold")
    return save(fig, out / f"s1_density_{model}_{split}")


def fig_moments(D: Data, model, split, out):
    extract = D.has_extract(model, split)
    roles = D.roles(model, SCATTER_ROLES)
    fig = plt.figure(figsize=(17, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    specs = [("mean_r_real", "mean_r_pred", "Mean response μ_c", True),
             ("var_r_real", "var_r_pred", "Variance v_c (cell-to-cell)", extract),
             ("frac_zero_program_r_real", "frac_zero_program_r_pred", "Fraction of cells with 0 program UMIs", extract)]
    for ax, (xk, yk, title, ok) in zip(axes, specs):
        if not ok:
            pending(ax)
            ax.set_title(title)
            continue
        lo, hi = np.inf, -np.inf
        for role, method in roles:
            sm = D.seed_mean(model, method, split)
            ax.scatter(sm[xk], sm[yk], s=42, color=ROLES[role]["color"], marker=MARKERS[role], edgecolor=SURFACE,
                       linewidth=1, label=ROLES[role]["label"], zorder=3)
            lo, hi = min(lo, sm[xk].min(), sm[yk].min()), max(hi, sm[xk].max(), sm[yk].max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1, zorder=1)
        ax.set(xlabel=f"Measured", ylabel="Generated (seed mean)", title=title)
        style(ax)
    axes[0].legend(frameon=False, loc="upper left")
    for k, (role, method) in enumerate(roles):
        ax = fig.add_subplot(gs[1, k])
        if not extract:
            pending(ax)
            ax.set_title(f"Signed WRF error decomposition: {ROLES[role]['label']}")
            continue
        sm = D.seed_mean(model, method, split).loc[D.split_conditions(split)]
        y = np.arange(len(sm))
        left_pos, left_neg = np.zeros(len(sm)), np.zeros(len(sm))
        for comp, color, name in (("loc", "#6250d6", "location (mean shift)"), ("scale", "#e87ba4", "scale (spread)"),
                                  ("shape", "#008300", "shape (subpopulations, point masses)")):
            v = sm[f"dec_{comp}_q0.95"].to_numpy()
            start = np.where(v >= 0, left_pos, left_neg)
            ax.barh(y, v, left=start, height=0.62, color=color, label=name, edgecolor=SURFACE, linewidth=1)
            left_pos += np.where(v >= 0, v, 0)
            left_neg += np.where(v < 0, v, 0)
        ax.scatter(sm["dec_total_q0.95"], y, color=INK, s=16, zorder=4, label="total signed error")
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, [short_condition(c) for c in sm.index], fontsize=7)
        ax.set(xlabel="Contribution to predicted − measured WRF at τ0.95 (pp)",
               title=f"Decomposition: {ROLES[role]['label']}")
        style(ax, "x")
        if k == 0:
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(f"{MODELS[model]['label']}, {split}: does the model get the location, spread and weak-responder "
                 "subpopulation right? (conditions sorted by measured WRF, top = highest)", fontsize=11,
                 fontweight="bold")
    return save(fig, out / f"s1_moments_{model}_{split}")


# ---------------------------------------------------------------------------------------------
# 2. Per-condition analysis
def waterfall(ax, D, model, contrast, split, color, title):
    ref, new = contrast
    eb = D.seed_mean(model, ref, split)["wrf_err_pp_q0.95"]
    en = D.seed_mean(model, new, split)["wrf_err_pp_q0.95"]
    d = (eb - en).sort_values(ascending=False)
    x = np.arange(len(d))
    ax.bar(x, d.values, color=[color if v > 0 else "#b9b8b1" for v in d.values], width=0.8, edgecolor=SURFACE)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axhline(d.mean(), color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
    ax.set_xticks(x, [short_condition(c) for c in d.index], rotation=70, ha="right", fontsize=6.8)
    key = {("base", "wrf3"): "base_vs_wrf3", ("base", "wrf13"): "base_vs_wrf13"}
    st = D.pair(model, key[tuple(MODELS[model]["methods"][m]["role"] for m in contrast)], split, "wrf_err_pp_q0.95")
    ax.text(0.98, 0.96, f"mean Δ = {st.delta_mean:+.2f} pp  [{st.ci95_low:+.2f}, {st.ci95_high:+.2f}]\n"
            f"improved {st.n_win}/{st.n_conditions} ({100 * st.win_fraction:.0f}%)\n"
            f"mean without largest gain {st.delta_mean_without_largest_gain:+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2)
    ax.set(ylabel="Δ_c = e_base − e_new (pp; > 0 better)", title=title)
    style(ax, "y")


def fig_per_condition(D: Data, model, split, out):
    base, wrf3, wrf13 = (role_method(model, r) for r in ("base", "wrf3", "wrf13"))
    fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    waterfall(fig.add_subplot(gs[0, 0]), D, model, (base, wrf3), split, ROLES["wrf3"]["color"],
              f"{ROLES['wrf3']['label']} vs λ = 0, per condition (grey: worse)")
    waterfall(fig.add_subplot(gs[0, 1]), D, model, (base, wrf13), split, ROLES["wrf13"]["color"],
              f"{ROLES['wrf13']['label']} vs λ = 0, per condition (grey: worse)")
    # signed error, conditions sorted by measured WRF
    ax = fig.add_subplot(gs[0, 2])
    conds = D.split_conditions(split)
    y = np.arange(len(conds))
    offsets = {"base": -0.22, "wrf3": 0, "wrf13": 0.22}
    for role, method in D.roles(model, SCATTER_ROLES):
        b = D.seed_mean(model, method, split).loc[conds, "bias_pp_q0.95"]
        ax.scatter(b, y + offsets[role], color=ROLES[role]["color"], marker=MARKERS[role], s=30, edgecolor=SURFACE,
                   label=ROLES[role]["label"], zorder=3)
    ax.axvline(0, color=AXIS, linewidth=1)
    meas = D.conditions.set_index("condition").loc[conds, "wrf_real_q0.95"]
    ax.set_yticks(y, [f"{short_condition(c)} ({100 * m:.0f}%)" for c, m in zip(conds, meas)], fontsize=7)
    ax.set(xlabel="Signed error b = predicted − measured WRF (pp; > 0: too many weak responders)",
           title="Signed error per condition (measured WRF in brackets)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style(ax, "x")
    # signed error vs measured WRF, by dose, vs potency
    attrs = D.conditions.set_index("condition")
    panels = [("wrf_real_q0.95", "Measured WRF at τ0.95", 100),
              ("potency_mean_r_10uM", "Drug potency: mean r at 10 µM (train dose)", 1),
              ("dose", "Held-out dose (nM)", 1)]
    for k, (key, xlabel, scale) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        for role, method in D.roles(model, SCATTER_ROLES):
            sm = D.seed_mean(model, method, split)
            x = attrs.loc[sm.index, key].to_numpy() * scale
            if key == "dose":
                x = np.log10(x) + {"base": -0.08, "wrf3": 0, "wrf13": 0.08}[role]
            ax.scatter(x, sm["bias_pp_q0.95"], color=ROLES[role]["color"], marker=MARKERS[role], s=30,
                       edgecolor=SURFACE, label=ROLES[role]["label"], zorder=3)
            if key != "dose" and len(sm) > 2:
                xx = attrs.loc[sm.index, key].to_numpy() * scale
                slope, icpt = np.polyfit(xx, sm["bias_pp_q0.95"], 1)
                grid = np.linspace(xx.min(), xx.max(), 20)
                ax.plot(grid, slope * grid + icpt, color=ROLES[role]["color"], linewidth=1.3, alpha=0.8)
        ax.axhline(0, color=AXIS, linewidth=1)
        if key == "dose":
            ax.set_xticks([2, 3], ["100 nM", "1000 nM"])
        ax.set(xlabel=xlabel, ylabel="Signed error b at τ0.95 (pp)")
        style(ax)
    fig.axes[-3].set_title("Systematic failure vs measured WRF (lines: least squares)")
    fig.axes[-2].set_title("…vs drug potency seen in training")
    fig.axes[-1].set_title("…vs held-out dose")
    fig.suptitle(f"{MODELS[model]['label']}, {split}: who benefits from the WRF loss, and who gets worse? "
                 "(primary threshold q = 0.95, errors averaged over 3 seeds)", fontsize=11, fontweight="bold")
    return save(fig, out / f"s2_per_condition_{model}_{split}")


# ---------------------------------------------------------------------------------------------
# 3. Threshold sensitivity
def threshold_curves(D, model, split, grid):
    thr = D.thr[(D.thr.split == split) & (D.thr.grid == grid)]
    curves = []
    for role, method in D.roles(model):
        part = thr[(thr.model == model) & (thr.method == method)]
        if part.empty:
            continue
        per_cond = part.groupby(["q", "condition"]).agg(err=("err_pp", "mean"), bias=("bias_pp", "mean"),
                                                       trained=("trained", "first")).reset_index()
        agg = per_cond.groupby("q").agg(err=("err", "mean"), bias=("bias", "mean"), trained=("trained", "first"))
        curves.append((role, method, agg))
    refs = []
    for name in ("vehicle", "interpolation", "nearest_dose", "adjacent_dose_mixture", "replicate"):
        part = thr[(thr.model == "reference") & (thr.method == name)]
        if part.empty:
            continue
        refs.append((name, part.groupby("q").agg(err=("err_pp", "mean"), bias=("bias_pp", "mean"))))
    return curves, refs


def fig_thresholds(D: Data, model, split, out):
    extract = D.has_extract(model, split)
    grid = "fine" if extract else "grid13"
    curves, refs = threshold_curves(D, model, split, grid)
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1, 0.95])
    ax = fig.add_subplot(gs[0, 0:2])
    for name, agg in refs:
        ax.plot(agg.index, agg.err, color=INK_2 if name != "replicate" else MUTED, linestyle=REFERENCES[name]["dash"],
                linewidth=1.4, label=REFERENCES[name]["label"])
    for role, method, agg in curves:
        color = ROLES[role]["color"]
        ax.plot(agg.index, agg.err, color=color, linewidth=2, label=method_label(model, method), zorder=3)
        t = agg[agg.trained]
        ax.scatter(t.index, t.err, color=color, marker=MARKERS[role], s=46, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    for q in SPEC_QS:
        ax.axvline(q, color=MUTED, linewidth=0.7, alpha=0.6)
    ax.set(xlabel="Threshold t, as the q-quantile of control response scores", ylabel="E_m(t): mean |F̂ − F| (pp)",
           title=f"Threshold sensitivity ({'fine grid, every 0.025' if extract else '13-threshold grid'}); "
                 "filled markers = thresholds in that method's training loss", xlim=(0, 1.0))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    style(ax)
    ax = fig.add_subplot(gs[0, 2])
    for role, method, agg in curves:
        ax.plot(agg.index, agg.bias, color=ROLES[role]["color"], linewidth=2, label=ROLES[role]["label"])
    for name, agg in refs:
        if name in ("vehicle", "interpolation"):
            ax.plot(agg.index, agg.bias, color=INK_2, linestyle=REFERENCES[name]["dash"], linewidth=1.3,
                    label=REFERENCES[name]["label"])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set(xlabel="Threshold quantile q", ylabel="Mean signed error (pp; > 0: too many below t)",
           title="Signed bias across thresholds", xlim=(0, 1.0))
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    # improvement vs threshold with bootstrap band
    ax = fig.add_subplot(gs[1, 0])
    for contrast, role in (("base_vs_wrf3", "wrf3"), ("base_vs_wrf13", "wrf13"), ("base_vs_mmd", "mmd")):
        p = D.paired_thr[(D.paired_thr.model == model) & (D.paired_thr.contrast == contrast)
                         & (D.paired_thr.split == split) & (D.paired_thr.grid == grid)].sort_values("q")
        if p.empty:
            continue
        color = ROLES[role]["color"]
        ax.fill_between(p.q, p.ci95_low, p.ci95_high, color=color, alpha=0.18, linewidth=0)
        ax.plot(p.q, p.delta_mean, color=color, linewidth=2, label=f"{ROLES[role]['label']} vs λ = 0")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set(xlabel="Threshold quantile q", ylabel="ΔE(t) = E_base − E_new (pp; > 0 better)",
           title="Improvement at every threshold (band: condition bootstrap 95% CI)", xlim=(0, 1.0))
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    # win fraction vs threshold
    ax = fig.add_subplot(gs[1, 1])
    for contrast, role in (("base_vs_wrf3", "wrf3"), ("base_vs_wrf13", "wrf13"), ("base_vs_mmd", "mmd")):
        p = D.paired_thr[(D.paired_thr.model == model) & (D.paired_thr.contrast == contrast)
                         & (D.paired_thr.split == split) & (D.paired_thr.grid == grid)].sort_values("q")
        if not p.empty:
            ax.plot(p.q, 100 * p.win_fraction, color=ROLES[role]["color"], linewidth=2, label=ROLES[role]["label"])
    ax.axhline(50, color=AXIS, linewidth=1)
    ax.set(xlabel="Threshold quantile q", ylabel="Conditions improved (%)", title="Win fraction across thresholds",
           ylim=(0, 100), xlim=(0, 1.0))
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    # W1 per condition (paired)
    ax = fig.add_subplot(gs[1, 2])
    roles = D.roles(model)
    for k, (role, method) in enumerate(roles):
        sm = D.seed_mean(model, method, split)["w1_r"]
        ax.scatter(np.full(len(sm), k) + np.linspace(-0.12, 0.12, len(sm)), sm, color=ROLES[role]["color"],
                   marker=MARKERS[role], s=26, edgecolor=SURFACE, zorder=3)
        ax.plot([k - 0.3, k + 0.3], [sm.mean()] * 2, color=INK, linewidth=2)
    base = D.seed_mean(model, roles[0][1], split)["w1_r"]
    for c in base.index:
        ax.plot(range(len(roles)), [D.seed_mean(model, m, split).loc[c, "w1_r"] for _, m in roles], color=GRID,
                linewidth=0.8, zorder=1)
    for name in ("vehicle", "replicate"):
        v = D.pc[(D.pc.method == name) & (D.pc.split == split)].w1_r.mean()
        ax.axhline(v, color=INK_2 if name == "vehicle" else MUTED, linestyle=REFERENCES[name]["dash"], linewidth=1.2,
                   label=f"{REFERENCES[name]['label']} {v:.2f}")
    ax.set_xticks(range(len(roles)), [ROLES[r]["label"].replace(" (no WRF loss)", "") for r, _ in roles], fontsize=7.5)
    ax.set(ylabel="W₁(F_c, F̂_c) (control-SD units)", title="Whole-distribution distance W₁ per condition (bar: mean)")
    ax.legend(frameon=False, fontsize=7)
    style(ax, "y")
    # signed-error heatmaps
    thr = D.thr[(D.thr.split == split) & (D.thr.grid == grid) & (D.thr.model == model)]
    conds = D.split_conditions(split)
    for k, (role, method) in enumerate(D.roles(model, SCATTER_ROLES)):
        ax = fig.add_subplot(gs[2, k])
        mat = thr[thr.method == method].pivot_table(index="condition", columns="q", values="bias_pp", aggfunc="mean").loc[conds]
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=DIVERGING, vmin=-40, vmax=40, interpolation="nearest")
        qs = mat.columns.to_numpy()
        ticks = [i for i, q in enumerate(qs) if any(abs(q - t) < 1e-9 for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.975))]
        ax.set_xticks(ticks, [f"{qs[i]:g}" for i in ticks], fontsize=7)
        ax.set_yticks(range(len(conds)), [short_condition(c) for c in conds], fontsize=6.5)
        trained = MODELS[model]["methods"][method]["train_qs"]
        for i, q in enumerate(qs):
            if any(abs(q - t) < 1e-9 for t in trained):
                ax.plot(i, -0.9, marker="v", color=INK, markersize=4, clip_on=False)
        ax.set(xlabel="Threshold quantile q (▼ trained)", title=f"Signed error (pp): {ROLES[role]['label']}")
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    fig.colorbar(im, ax=fig.axes[-3:], shrink=0.8, label="predicted − measured F(t) (pp); red: too many below t")
    fig.suptitle(f"{MODELS[model]['label']}, {split}: did the loss learn the distribution or match the supervised "
                 "fractions? (rows of heatmaps sorted by measured WRF)", fontsize=11, fontweight="bold")
    return save(fig, out / f"s3_thresholds_{model}_{split}")


# ---------------------------------------------------------------------------------------------
# 4. Expression fidelity
EXPR_METRICS = [("rmse_mean_all", "RMSE of means: all 2085 genes"), ("rmse_mean_deg50", "RMSE of means: top-50 DEGs"),
                ("rmse_mean_nonprogram", "RMSE of means: 1988 non-program genes"),
                ("rmse_mean_program", "RMSE of means: 97 program genes"),
                ("rmse_sd_nonprogram", "RMSE of per-gene SD: non-program"), ("mmd2", "MMD²: spec PCA (all genes)"),
                ("mmd2_nonprogram", "MMD²: non-program PCA"), ("effect_corr_deg50", "Effect correlation: top-50 DEGs ↑")]


def fig_expression(D: Data, model, split, out):
    extract = D.has_extract(model, split)
    fig = plt.figure(figsize=(18, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25])
    roles = D.roles(model)
    for k, (metric, title) in enumerate(EXPR_METRICS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        rows = D.methods[(D.methods.model == model) & (D.methods.split == split)]
        if f"{metric}_mean" not in rows or rows[f"{metric}_mean"].isna().all():
            pending(ax)
            ax.set_title(title, fontsize=9)
            continue
        labels = []
        for i, (role, method) in enumerate(roles):
            row = rows[rows.method == method].iloc[0]
            per_seed = D.pc[(D.pc.model == model) & (D.pc.method == method) & (D.pc.split == split)].groupby("seed")[metric].mean()
            ax.scatter(per_seed, [i] * len(per_seed), color=ROLES[role]["color"], s=14, alpha=0.6, zorder=2)
            ax.scatter([row[f"{metric}_mean"]], [i], color=ROLES[role]["color"], marker=MARKERS[role], s=70,
                       edgecolor=SURFACE, linewidth=1.2, zorder=3)
            labels.append(ROLES[role]["label"].replace(" (no WRF loss)", ""))
        for j, name in enumerate(("vehicle", "replicate", "adjacent_dose_mixture")):
            v = D.pc[(D.pc.method == name) & (D.pc.split == split)][metric] if metric in D.pc else pd.Series(dtype=float)
            if v.notna().any():
                ax.axvline(v.mean(), color=INK_2 if name != "replicate" else MUTED, linestyle=REFERENCES[name]["dash"],
                           linewidth=1.1, label=REFERENCES[name]["label"])
        ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        style(ax, "x")
        if k == 0:
            ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    base, wrf3, wrf13 = (role_method(model, r) for r in ("base", "wrf3", "wrf13"))
    sb = D.seed_mean(model, base, split)
    for k, metric in enumerate(["rmse_mean_nonprogram", "rmse_mean_deg50"]):
        ax = fig.add_subplot(gs[2, k])
        for role, method in (("wrf3", wrf3), ("wrf13", wrf13)):
            sn = D.seed_mean(model, method, split).loc[sb.index]
            ax.scatter(sb[metric], sn[metric], color=ROLES[role]["color"], marker=MARKERS[role], s=34, edgecolor=SURFACE,
                       label=ROLES[role]["label"], zorder=3)
        lo, hi = min(sb[metric].min(), 0), max(sb[metric].max(), D.seed_mean(model, wrf3, split)[metric].max(),
                                                D.seed_mean(model, wrf13, split)[metric].max())
        ax.plot([lo, hi * 1.05], [lo, hi * 1.05], color=MUTED, linewidth=1)
        ax.set(xlabel=f"λ = 0 (baseline)", ylabel="With WRF loss",
               title=f"Per condition: {dict(EXPR_METRICS)[metric]}\n(above the line = loss made it worse)")
        ax.legend(frameon=False, fontsize=7)
        style(ax)
    ax = fig.add_subplot(gs[2, 2:])
    for role, method in (("wrf3", wrf3), ("wrf13", wrf13)):
        sn = D.seed_mean(model, method, split).loc[sb.index]
        dx = sb["wrf_err_pp_q0.95"] - sn["wrf_err_pp_q0.95"]
        dy = sb["rmse_mean_nonprogram"] - sn["rmse_mean_nonprogram"]
        ax.scatter(dx, dy, color=ROLES[role]["color"], marker=MARKERS[role], s=40, edgecolor=SURFACE,
                   label=ROLES[role]["label"], zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=1)
    for (x, y, text) in ((0.98, 0.98, "better WRF,\nbetter expression"), (0.98, 0.02, "better WRF,\nworse expression"),
                         (0.02, 0.02, "worse on both"), (0.02, 0.98, "worse WRF,\nbetter expression")):
        ax.text(x, y, text, transform=ax.transAxes, ha="right" if x > 0.5 else "left", va="top" if y > 0.5 else "bottom",
                fontsize=7.5, color=MUTED)
    ax.set(xlabel="Δ WRF error at τ0.95 (pp; > 0: loss improved WRF)",
           ylabel="Δ non-program RMSE (> 0: loss improved expression)",
           title="Trade-off per condition: what did the WRF gain cost outside the program?")
    ax.legend(frameon=False, fontsize=7, loc="center left")
    style(ax)
    note = "" if extract else " — CellFlow non-program RMSE derived exactly from all-gene and program RMSE"
    fig.suptitle(f"{MODELS[model]['label']}, {split}: expression fidelity (large marker = mean of 3 seeds, "
                 f"small = seeds; lower is better unless marked ↑){note}", fontsize=11, fontweight="bold")
    return save(fig, out / f"s4_expression_{model}_{split}")


# ---------------------------------------------------------------------------------------------
# 5. Ablations
FOREST_METRICS = [("wrf_err_pp_q0.95", "WRF error q0.95 (pp)"), ("wrf_err_pp_grid13_mean", "Mean WRF error, 13 τ (pp)"),
                  ("w1_r", "W₁(r)"), ("mmd2", "MMD² (spec)"), ("rmse_mean_deg50", "RMSE DEG50"),
                  ("rmse_mean_nonprogram", "RMSE non-program")]
FOREST_ROWS = [("chemcpa", "base_vs_wrf3", "test"), ("chemcpa", "base_vs_wrf13", "test"), ("chemcpa", "wrf3_vs_wrf13", "test"),
               ("chemcpa", "matched_l10_h0.5_3vs13", "val"), ("chemcpa", "matched_l100_h0.5_3vs13", "val"),
               ("cellflow", "base_vs_wrf3", "test"), ("cellflow", "base_vs_wrf13", "test"),
               ("cellflow", "wrf3_vs_wrf13", "test"), ("cellflow", "base_vs_mmd", "test"),
               ("cellflow", "mmd_vs_wrf3", "test"), ("cellflow", "mmd_vs_wrf13", "test")]


def fig_forest(D: Data, out):
    fig, axes = plt.subplots(1, len(FOREST_METRICS), figsize=(20, 7.2), sharey=True, constrained_layout=True)
    rows = [r for r in FOREST_ROWS if D.pair(*r[:2], r[2], "wrf_err_pp_q0.95") is not None]
    y = np.arange(len(rows))[::-1]
    labels = []
    for (model, contrast, split), yy in zip(rows, y):
        p = D.pair(model, contrast, split, "wrf_err_pp_q0.95")
        labels.append(f"{MODELS[model]['label']}: {p.label}  [{split}]")
    for ax, (metric, title) in zip(axes, FOREST_METRICS):
        for (model, contrast, split), yy in zip(rows, y):
            p = D.pair(model, contrast, split, metric)
            if p is None:
                continue
            role = MODELS[model]["methods"][p.new]["role"] if p.new in MODELS[model]["methods"] else "wrf13"
            color = ROLES.get(role, ROLES["wrf13"])["color"] if role in ROLES else ROLES["wrf13"]["color"]
            ax.plot([p.ci95_low, p.ci95_high], [yy, yy], color=color, linewidth=2.2, solid_capstyle="round")
            ax.scatter([p.delta_mean], [yy], color=color, marker=MARKERS.get(role, "^"), s=46, edgecolor=SURFACE, zorder=3)
            ax.text(1.0, yy + 0.25, f"{int(p.n_win)}/{int(p.n_conditions)}", transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=6.5, color=MUTED)
        ax.axvline(0, color=INK_2, linewidth=1)
        sep = len([r for r in rows if r[0] == "chemcpa"])
        ax.axhline(len(rows) - sep - 0.5, color=AXIS, linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Δ (ref − new; > 0: new better)", fontsize=7.5)
        style(ax, "x")
    axes[0].set_yticks(y, labels, fontsize=8)
    fig.suptitle("Ablations: paired mean difference with 95% condition-bootstrap CI (numbers: conditions improved). "
                 "Same training setup within each model except the loss; chemCPA 3 vs 13 τ at test also changes h "
                 "(0.5 → 0.25), so the matched-h rows are validation", fontsize=10.5, fontweight="bold")
    return save(fig, out / "s5_forest")


def fig_lambda(D: Data, out):
    m = D.methods[(D.methods.model == "chemcpa") & (D.methods.split == "val")].copy()
    sweep = m[m.lam > 0]
    base = m[m.role == "base"].iloc[0]
    veh = D.methods[(D.methods.method == "vehicle") & (D.methods.split == "val")].iloc[0]
    metrics = [("wrf_err_pp_q0.95", "WRF error q0.95 (pp)"), ("wrf_err_pp_grid13_mean", "Mean WRF error, 13 τ (pp)"),
               ("w1_r", "W₁(r)"), ("mmd2", "MMD² (spec)"), ("rmse_mean_deg50", "RMSE DEG50"),
               ("rmse_mean_nonprogram", "RMSE non-program genes"), ("rmse_mean_program", "RMSE program genes"),
               ("mmd2_nonprogram", "MMD² non-program PCA")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for (n_taus, h), part in sweep.groupby(["n_taus", "h"]):
            part = part.sort_values("lam")
            color = ROLES["wrf3" if n_taus == 3 else "wrf13"]["color"]
            dash = "solid" if h in (0.5,) else (0, (4, 2))
            ax.errorbar(part.lam, part[f"{metric}_mean"], yerr=part[f"{metric}_seed_sd"], color=color, linestyle=dash,
                        marker=MARKERS["wrf3" if n_taus == 3 else "wrf13"], markersize=6, capsize=2.5, linewidth=1.8,
                        markeredgecolor=SURFACE, label=f"{int(n_taus)} τ, h = {h:g}")
            sel = part[part.role.isin(["wrf3", "wrf13"])]
            if len(sel):
                ax.scatter(sel.lam, sel[f"{metric}_mean"], s=170, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
        ax.axhline(base[f"{metric}_mean"], color=ROLES["base"]["color"], linewidth=1.8, label="λ = 0 (no WRF loss)")
        if f"{metric}_mean" in veh and pd.notna(veh[f"{metric}_mean"]):
            ax.axhline(veh[f"{metric}_mean"], color=INK_2, linestyle=REFERENCES["vehicle"]["dash"], linewidth=1.2,
                       label=REFERENCES["vehicle"]["label"])
        ax.set_xscale("log")
        ax.set(xlabel="λ (fraction-loss weight)", title=title)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("chemCPA-NB, validation (9 conditions): how strong is the WRF–expression trade-off as λ grows? "
                 "(mean ± seed SD; circled = configuration selected for test)", fontsize=11, fontweight="bold")
    return save(fig, out / "s5_lambda_chemcpa")


def fig_pareto(D: Data, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, (x, xl), (yk, yl) in ((axes[0], ("wrf_err_pp_q0.95", "WRF error q0.95 (pp)"), ("rmse_mean_deg50", "RMSE DEG50")),
                                  (axes[1], ("wrf_err_pp_grid13_mean", "Mean WRF error over 13 τ (pp)"), ("mmd2", "MMD² (spec)"))):
        for model, marker_fill in (("chemcpa", True), ("cellflow", False)):
            m = D.methods[(D.methods.model == model) & (D.methods.split == "val")]
            for _, row in m.iterrows():
                role = row.role if row.role in ROLES else ("wrf3" if row.n_taus == 3 else "wrf13" if row.n_taus == 13 else None)
                if role is None:
                    continue
                color = ROLES[role]["color"]
                ax.scatter(row[f"{x}_mean"], row[f"{yk}_mean"], s=70 if row.role in ROLES else 34,
                           marker=MARKERS[role], facecolor=color if marker_fill else SURFACE, edgecolor=color,
                           linewidth=1.6, alpha=1 if row.role in ROLES else 0.55, zorder=3)
                if row.role in ROLES:
                    ax.annotate(f"{MODELS[model]['label']} {ROLES[role]['label'].replace(' (no WRF loss)', '')}",
                                (row[f"{x}_mean"], row[f"{yk}_mean"]), xytext=(5, 4), textcoords="offset points",
                                fontsize=7, color=INK_2)
        for name in ("vehicle", "replicate", "adjacent_dose_mixture"):
            r = D.methods[(D.methods.method == name) & (D.methods.split == "val")]
            if len(r) and pd.notna(r.iloc[0].get(f"{yk}_mean")) and pd.notna(r.iloc[0].get(f"{x}_mean")):
                ax.scatter(r.iloc[0][f"{x}_mean"], r.iloc[0][f"{yk}_mean"], marker="X", s=60, color=INK_2, zorder=3)
                ax.annotate(REFERENCES[name]["label"], (r.iloc[0][f"{x}_mean"], r.iloc[0][f"{yk}_mean"]),
                            xytext=(5, -9), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set(xlabel=xl, ylabel=yl)
        style(ax)
    axes[0].set_title("Validation trade-off: primary WRF vs DEG expression (filled = chemCPA, hollow = CellFlow; "
                      "small = other λ/h settings)", fontsize=9)
    axes[1].set_title("Validation trade-off: whole response distribution vs multigene distribution", fontsize=9)
    return save(fig, out / "s5_pareto_val")


# ---------------------------------------------------------------------------------------------
# 6. Simple baselines and uncertainty
def fig_baselines(D: Data, split, out):
    fig = plt.figure(figsize=(19, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    # (a) q0.95 error for every method incl. simple baselines, both models
    ax = fig.add_subplot(gs[0, 0])
    entries = []
    for model in MODELS:
        for role, method in D.roles(model):
            per_seed = D.pc[(D.pc.model == model) & (D.pc.method == method) & (D.pc.split == split)].groupby("seed")["wrf_err_pp_q0.95"].mean()
            entries.append((f"{MODELS[model]['label']} {ROLES[role]['label'].replace(' (no WRF loss)', '')}",
                            per_seed, ROLES[role]["color"], MARKERS[role]))
    for name in ("interpolation", "adjacent_dose_mixture", "nearest_dose", "vehicle", "replicate"):
        v = D.pc[(D.pc.method == name) & (D.pc.split == split)]["wrf_err_pp_q0.95"]
        entries.append((REFERENCES[name]["label"], pd.Series([v.mean()]), INK_2 if name != "replicate" else MUTED, "X"))
    entries.sort(key=lambda e: e[1].mean())
    for i, (label, vals, color, marker) in enumerate(entries):
        ax.scatter(vals, [i] * len(vals), color=color, s=14, alpha=0.55)
        ax.scatter([vals.mean()], [i], color=color, marker=marker, s=60, edgecolor=SURFACE, zorder=3)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries], fontsize=7.5)
    ax.invert_yaxis()
    ax.set(xlabel="WRF error at τ0.95 (pp; mean over conditions; small dots = seeds)",
           title="Does the learned model beat simple WRF estimates?")
    style(ax, "x")
    # (b) interpolation vs model per condition
    ax = fig.add_subplot(gs[0, 1])
    interp = D.pc[(D.pc.method == "interpolation") & (D.pc.split == split)].set_index("condition")["wrf_err_pp_q0.95"]
    for model, fill in (("chemcpa", True), ("cellflow", False)):
        for role in ("wrf3",):
            sm = D.seed_mean(model, role_method(model, role), split)["wrf_err_pp_q0.95"]
            color = ROLES[role]["color"] if model == "chemcpa" else ROLES["base"]["color"]
            ax.scatter(interp.loc[sm.index], sm, s=40, marker="o" if model == "chemcpa" else "s",
                       facecolor=color if fill else SURFACE, edgecolor=color, linewidth=1.5,
                       label=f"{MODELS[model]['label']} + WRF loss, 3 τ")
    hi = max(interp.max(), 1) * 1.08
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1)
    ax.set(xlabel="Log-dose interpolation error (pp)", ylabel="Model error (pp)",
           title="Per condition at τ0.95: above the line, interpolation wins")
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    # (c) E(t) on the 13-threshold grid: interpolation vs models
    ax = fig.add_subplot(gs[0, 2])
    thr = D.thr[(D.thr.split == split) & (D.thr.grid == "grid13")]
    for model, dash in (("chemcpa", "solid"), ("cellflow", (0, (5, 2)))):
        for role in ("base", "wrf3", "wrf13"):
            part = thr[(thr.model == model) & (thr.method == role_method(model, role))]
            agg = part.groupby(["q", "condition"]).err_pp.mean().groupby("q").mean()
            ax.plot(agg.index, agg.values, color=ROLES[role]["color"], linestyle=dash, linewidth=1.8,
                    label=f"{MODELS[model]['label']} {ROLES[role]['label'].replace(' (no WRF loss)', '')}")
    for name in ("interpolation", "adjacent_dose_mixture", "vehicle", "replicate"):
        agg = thr[(thr.model == "reference") & (thr.method == name)].groupby("q").err_pp.mean()
        ax.plot(agg.index, agg.values, color=INK if name == "interpolation" else MUTED,
                linestyle=REFERENCES[name]["dash"], linewidth=2 if name == "interpolation" else 1.2,
                label=REFERENCES[name]["label"])
    ax.set(xlabel="Threshold quantile q", ylabel="Mean WRF error (pp)", title="Across the 13 thresholds (solid: chemCPA, dashed: CellFlow)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.3, ncol=2)
    style(ax)
    # (d) bootstrap distributions of mean Δ (recomputed for display from per-condition errors)
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    yoff = 0
    ticks = []
    for model in MODELS:
        for contrast, role in (("base_vs_wrf3", "wrf3"), ("base_vs_wrf13", "wrf13")):
            ref, new = role_method(model, "base"), role_method(model, role)
            eb = D.seed_mean(model, ref, split)["wrf_err_pp_q0.95"]
            en = D.seed_mean(model, new, split)["wrf_err_pp_q0.95"].loc[eb.index]
            d = (eb - en).to_numpy()
            boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            parts = ax.violinplot([boots], positions=[yoff], vert=False, widths=0.8, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(ROLES[role]["color"])
                body.set_alpha(0.35)
            st = D.pair(model, contrast, split, "wrf_err_pp_q0.95")
            ax.plot([st.ci95_low, st.ci95_high], [yoff, yoff], color=ROLES[role]["color"], linewidth=2.5)
            ax.plot([st.ci95_two_level_low, st.ci95_two_level_high], [yoff - 0.28] * 2, color=ROLES[role]["color"],
                    linewidth=1.2, linestyle=(0, (2, 1)))
            pc = D.pc[(D.pc.model == model) & (D.pc.split == split)]
            sb = pc[pc.method == ref].groupby("seed")["wrf_err_pp_q0.95"].mean()
            sn = pc[pc.method == new].groupby("seed")["wrf_err_pp_q0.95"].mean()
            pairs = np.array([a - b for a in sb for b in sn])
            ax.scatter(pairs, [yoff + 0.3] * len(pairs), color=INK, s=10, zorder=4)
            ticks.append((yoff, f"{MODELS[model]['label']}: {ROLES[role]['label']}"))
            yoff += 1
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks], fontsize=8)
    ax.set(xlabel="Mean Δ at τ0.95 vs λ = 0 (pp; > 0 better)",
           title="Uncertainty: condition bootstrap (violin, solid 95% CI),\n+ seed resampling (dashed CI); dots = all 9 seed pairs")
    style(ax, "x")
    # (e) measured variability per condition at q0.95
    ax = fig.add_subplot(gs[1, 1])
    v = D.variability[(D.variability.split == split) & np.isclose(D.variability.q, PRIMARY_Q)].set_index("condition")
    conds = [c for c in D.split_conditions(split)]
    y = np.arange(len(conds))
    ax.scatter(v.loc[conds, "replicate_err_pp"], y, marker="X", color=INK_2, s=34, label="Replicate well vs well", zorder=3)
    ax.scatter(v.loc[conds, "random_split_err_pp"], y, marker="|", color=INK, s=120, linewidth=2,
               label="Random cell split (same sizes)", zorder=3)
    ax.scatter(v.loc[conds, "perfect_model_err_pp_n1024"], y, marker="o", facecolor=SURFACE, edgecolor=MUTED, s=26,
               label="Perfect generator, binomial noise", zorder=3)
    for model, marker in (("chemcpa", "s"), ("cellflow", "D")):
        sm = D.seed_mean(model, role_method(model, "wrf3"), split).loc[conds, "wrf_err_pp_q0.95"]
        ax.scatter(sm, y, marker=marker, s=26, facecolor=ROLES["wrf3"]["color"] if model == "chemcpa" else SURFACE,
                   edgecolor=ROLES["wrf3"]["color"], linewidth=1.4, label=f"{MODELS[model]['label']} + WRF 3 τ", zorder=4)
    ax.set_yticks(y, [short_condition(c) for c in conds], fontsize=7)
    ax.set(xlabel="|difference| in WRF at τ0.95 (pp)", title="How noisy is the measured WRF itself? (per condition)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style(ax, "x")
    # (f) noise floors across thresholds
    ax = fig.add_subplot(gs[1, 2])
    vv = D.variability[D.variability.split == split].groupby("q").mean(numeric_only=True)
    ax.plot(vv.index, vv.replicate_err_pp, color=INK_2, marker="X", label="Replicate well vs well")
    ax.plot(vv.index, vv.random_split_err_pp, color=INK, linewidth=2, label="Random cell split (same sizes)")
    ax.plot(vv.index, vv.perfect_model_err_pp_n1024, color=MUTED, linestyle=(0, (1, 2)), linewidth=1.6,
            label="Perfect generator (1024 cells)")
    if split == "test" and len(D.noise):
        g = D.noise[D.noise.role.isin(["base", "wrf3", "wrf13"])].groupby("q").resample_err_pp.mean()
        ax.plot(g.index, g.values, color=MUTED, linewidth=1.2, label="chemCPA: two sampling draws, same model")
    for model, dash in (("chemcpa", "solid"), ("cellflow", (0, (5, 2)))):
        part = thr[(thr.model == model) & (thr.method == role_method(model, "wrf3"))]
        agg = part.groupby(["q", "condition"]).err_pp.mean().groupby("q").mean()
        ax.plot(agg.index, agg.values, color=ROLES["wrf3"]["color"], linestyle=dash, linewidth=1.8,
                label=f"{MODELS[model]['label']} + WRF 3 τ")
    ax.set(xlabel="Threshold quantile q", ylabel="Mean |difference| (pp)", title="Noise floors vs model error across thresholds")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    fig.suptitle(f"{split}: is the gain useful (vs cheap WRF estimates) and reliable (conditions, seeds, measurement noise)?",
                 fontsize=11, fontweight="bold")
    return save(fig, out / f"s6_baselines_uncertainty_{split}")


def fig_overview(D: Data, out):
    metrics = [("wrf_err_pp_q0.95", "WRF error q0.95 (pp)"), ("wrf_err_pp_grid13_mean", "Mean WRF error, 13 τ (pp)"),
               ("w1_r", "W₁(r)"), ("mmd2", "MMD² (spec)"), ("rmse_mean_deg50", "RMSE DEG50"),
               ("rmse_mean_nonprogram", "RMSE non-program")]
    rows = []
    for model in MODELS:
        for role, method in D.roles(model):
            rows.append((f"{MODELS[model]['label']}  {ROLES[role]['label']}", model, method, ROLES[role]["color"], MARKERS[role]))
    refs = [(REFERENCES[n]["label"], "reference", n, INK_2 if n != "replicate" else MUTED, "X")
            for n in ("interpolation", "adjacent_dose_mixture", "nearest_dose", "vehicle", "replicate")]
    allrows = rows + refs
    fig, axes = plt.subplots(1, len(metrics), figsize=(21, 6.8), sharey=True, constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        for i, (label, model, method, color, marker) in enumerate(allrows):
            part = D.pc[(D.pc.model == model) & (D.pc.method == method) & (D.pc.split == "test")]
            if part.empty or metric not in part or part[metric].isna().all():
                continue
            per_seed = part.groupby("seed")[metric].mean()
            ax.scatter(per_seed, [i] * len(per_seed), color=color, s=12, alpha=0.55)
            ax.scatter([per_seed.mean()], [i], color=color, marker=marker, s=60, edgecolor=SURFACE, zorder=3)
        ax.axhline(len(rows) - 0.5, color=AXIS, linewidth=1)
        ax.axhline(len([r for r in rows if r[1] == "chemcpa"]) - 0.5, color=GRID, linewidth=1)
        ax.set_title(title, fontsize=9)
        style(ax, "x")
    axes[0].set_yticks(range(len(allrows)), [r[0] for r in allrows], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Test split (17 held-out doses): both models and the simple references; lower is better "
                 "(large marker = mean of seeds, small = individual seeds)", fontsize=11, fontweight="bold")
    return save(fig, out / "overview")


# ---------------------------------------------------------------------------------------------
def write_report(D: Data, figures: dict, root: Path):
    def fmt_pair(model, contrast, split, metric="wrf_err_pp_q0.95"):
        p = D.pair(model, contrast, split, metric)
        if p is None:
            return "n/a"
        return (f"{p.delta_mean:+.2f} [{p.ci95_low:+.2f}, {p.ci95_high:+.2f}], {int(p.n_win)}/{int(p.n_conditions)} improved; "
                f"seed pairs {p.seed_pair_delta_min:+.2f}…{p.seed_pair_delta_max:+.2f}")
    lines = ["# WRF loss analysis: chemCPA and CellFlow", "",
             "Generated by `slurm/analysis/submit.sh` (figures redraw from `tables/` with `STAGES=figures`).", "",
             f"CellFlow extracts from PSC available: **{D.summary['cellflow_extracts_available']}**. "
             "Without them, CellFlow distribution panels use the 13-threshold CSV and some panels say so.", "",
             "## Consistency checks", "", "| check | value |", "|---|---:|"]
    lines += [f"| {k} | {v:.3g} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in D.summary["checks"].items()]
    lines += ["", "## Paired comparisons at the primary threshold (test)", "",
              "Δ = e_ref − e_new in pp (> 0: new better), condition-bootstrap 95% CI.", "",
              "| model | contrast | Δ q0.95 | Δ 13-τ mean | Δ RMSE non-program |", "|---|---|---|---|---|"]
    for model in MODELS:
        for contrast in ("base_vs_wrf3", "base_vs_wrf13", "wrf3_vs_wrf13", "base_vs_mmd", "vehicle_vs_wrf3",
                         "interpolation_vs_wrf3", "interpolation_vs_base"):
            if D.pair(model, contrast, "test", "wrf_err_pp_q0.95") is None:
                continue
            lines.append(f"| {MODELS[model]['label']} | {contrast} | {fmt_pair(model, contrast, 'test')} | "
                         f"{fmt_pair(model, contrast, 'test', 'wrf_err_pp_grid13_mean')} | "
                         f"{fmt_pair(model, contrast, 'test', 'rmse_mean_nonprogram')} |")
    lines += ["", "## Figures", ""]
    for name, path in figures.items():
        if path is not None:
            lines += [f"### {name}", "", f"![{name}](figures/{path.name})", ""]
    (root / "report.md").write_text("\n".join(lines))
    return root / "report.md"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    root = Path(a.root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    D = Data(root)
    figures = {"Overview (test)": fig_overview(D, out)}
    for split in ("test", "val"):
        for model in MODELS:
            label = f"{MODELS[model]['label']}, {split}"
            figures[f"1. Response-score CDFs — {label}"] = fig_ecdf(D, model, split, out)
            figures[f"1. Response-score histograms, selected conditions — {label}"] = fig_density(D, model, split, out)
            figures[f"1. Moments and error decomposition — {label}"] = fig_moments(D, model, split, out)
            figures[f"2. Per-condition analysis — {label}"] = fig_per_condition(D, model, split, out)
            figures[f"3. Threshold sensitivity — {label}"] = fig_thresholds(D, model, split, out)
            figures[f"4. Expression fidelity — {label}"] = fig_expression(D, model, split, out)
        figures[f"6. Simple baselines and uncertainty — {split}"] = fig_baselines(D, split, out)
    figures["5. Ablations: paired contrasts"] = fig_forest(D, out)
    figures["5. Ablations: chemCPA λ sweep (validation)"] = fig_lambda(D, out)
    figures["5. Ablations: validation trade-off"] = fig_pareto(D, out)
    report = write_report(D, figures, root)
    print("\n".join(str(v) for v in figures.values() if v is not None))
    print(report)


if __name__ == "__main__":
    main()
