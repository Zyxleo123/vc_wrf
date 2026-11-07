"""One number for "does the model respond to dose?", and the drugs where it sits at its median and 75th percentile.

The number is the dose gain of sciplex_interp.dose_sweep, fixed there before any sweep was read, taken per drug
instead of pooled. For each validation drug, x is the measured and y the generated mean effect at its four doses, on
the drug's own direction (its top-50 DEGs and its measured 10 µM change, a training dose):

  gain = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²    1: the full measured dose response. 0: ignores dose.

`unseen` is the same through the origin on the two steps into and out of the held-out dose, the part no model trained
on. PC1 of all genes is kept beside the drug direction as a check; three drugs have almost no dose range on it, so their
per-drug ratios there are noise. The response score plays no part. Oracle: replicate well 2 against replicate well 1.

Cases are ranked on the baselines alone, the mean gain of CellFlow retrain and original chemCPA on each drug, so the
arm's own result plays no part in picking them. The 75th percentile is where the baselines do better than usual.

  collect  gain_by_drug.csv, gain_median.csv, gain_paired_median.csv, gain_by_dose.csv, gain_cases.json and gain_cells.npz
           in <root>/figures/dose. gain_by_dose.csv asks whether the failure is dose or condition: per measured dose,
           generated over measured change from vehicle, and how well a model separates drugs at that one dose.
           (needs dose_collect's curves.csv and the sweeps; reads only the validation drugs' cells)
  draw     6_dose_gain_cases.png and 7_dose_gain_cells.png from those files only

usage: python -m sciplex_interp.dose_gain collect --arm z70_lr3e4_mv10_t
       python -m sciplex_interp.dose_gain draw --arm z70_lr3e4_mv10_t
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data
from .dose_sweep import (BACKBONE, BASE, CHEMCPA, CHEMCPA_ORIGINAL, MEASURED_NM, N_BOOT, centred, directions, key, out_dir, steps)

AXES = ("effect", "pc1")
BASELINES = (BASE, CHEMCPA_ORIGINAL)
ALL_SEEDS = -1  # the `seed` of a row computed on the mean over seeds


def means(curves: pd.DataFrame, model: str, column: str, seed: int | None = None) -> pd.DataFrame:
    part = curves[(curves.model == model) & curves.dose.isin(MEASURED_NM)]
    if seed is not None:
        part = part[part.seed == seed]
    return part.pivot_table(index="drug", columns="dose", values=column, aggfunc="mean")


def by_drug(curves: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    measured = curves[curves.model == "measured"]
    held_out = measured[measured.unseen].set_index("drug").dose
    rows = []
    for column in AXES:
        for model in ["oracle", *models]:
            name, reference = ("rep2", "rep1") if model == "oracle" else (model, "measured")
            x = means(curves, reference, column)
            position = np.array([MEASURED_NM.index(held_out[d]) for d in x.index])
            seeds = [] if model == "oracle" else sorted(curves[curves.model == name].seed.unique())
            for seed in [None, *seeds]:
                xs, ys = x.to_numpy(), means(curves, name, column, seed).loc[x.index].to_numpy()
                num, den = centred(xs, ys)
                step_num, step_den = steps(xs, ys, position)
                rows += [{"axis": column, "model": model, "seed": ALL_SEEDS if seed is None else int(seed), "drug": d, "gain": n / m,
                          "unseen": sn / sd, "num": n, "den": m} for d, n, m, sn, sd in zip(x.index, num, den, step_num, step_den)]
    return pd.DataFrame(rows)


def medians(table: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median over drugs with a drug bootstrap, the range of the per-seed medians, and arm-minus-model paired."""
    rows, paired = [], []
    for column in AXES:
        part = table[table.axis == column]
        whole = part[part.seed == ALL_SEEDS]
        wide = {w: whole.pivot(index="drug", columns="model", values=w) for w in ("gain", "unseen")}
        n = len(wide["gain"])
        idx = np.random.default_rng(0).integers(0, n, (N_BOOT, n))
        boot = lambda v: np.median(v[idx], axis=1)
        for model in wide["gain"].columns:
            one = whole[whole.model == model]
            per_seed = part[(part.model == model) & (part.seed != ALL_SEEDS)].groupby("seed").gain.median()
            row = {"axis": column, "model": model, "pooled": one.num.sum() / one.den.sum(),
                   "seed_low": per_seed.min() if len(per_seed) else np.nan, "seed_high": per_seed.max() if len(per_seed) else np.nan}
            for what, w in wide.items():
                v = w[model].to_numpy()
                row |= {what: np.median(v), f"{what}_low": np.percentile(boot(v), 2.5), f"{what}_high": np.percentile(boot(v), 97.5)}
            rows.append(row)
            if model == arm:
                continue
            for what, w in wide.items():
                a, v = w[arm].to_numpy(), w[model].to_numpy()
                diff = boot(a) - boot(v)
                paired.append({"axis": column, "measure": what, "ref": model, "arm": np.median(a), "value_ref": np.median(v),
                               "delta": np.median(a) - np.median(v), "ci95_low": np.percentile(diff, 2.5), "ci95_high": np.percentile(diff, 97.5),
                               "n_drugs_arm_higher": int((a > v).sum())})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def by_dose(curves: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for column in AXES:
        x = means(curves, "measured", column)
        for model in models:
            y = means(curves, model, column).loc[x.index]
            for dose in MEASURED_NM:
                a, b = x[dose].to_numpy(), y[dose].to_numpy()
                ac, bc = a - a.mean(), b - b.mean()
                rows.append({"axis": column, "model": model, "dose": dose, "generated_over_measured": b.sum() / a.sum(),
                             "across_drug_slope": (ac * bc).sum() / (ac * ac).sum(), "across_drug_r": np.corrcoef(a, b)[0, 1]})
    return pd.DataFrame(rows)


def pick_cases(table: pd.DataFrame) -> dict:
    whole = table[(table.axis == "effect") & (table.seed == ALL_SEEDS) & table.model.isin(BASELINES)]
    rank = whole.groupby("drug").gain.mean().sort_values()
    at = lambda q: rank.index[int(round(q * (len(rank) - 1)))]
    return {"median": at(0.5), "q75": at(0.75), "baseline_gain": {d: float(v) for d, v in rank.items()}}


def case_cells(data: Data, dirs: dict, drugs: list[str], models: list[str], out: Path) -> dict:
    """Per-cell effect at the four doses: every saved sweep's cells (all seeds) and the measured cells."""
    cells = {}
    for drug in drugs:
        deg, u = dirs[drug]
        for dose in MEASURED_NM:
            X = data.X[data.idx(data.condition_of(drug, dose))]
            cells[f"measured::{key(drug, dose, 'effect')}"] = ((X[:, deg] - data.spec.control_mean[deg]) @ u).astype(np.float32)
    for model in models:
        sweeps = [np.load(p) for p in sorted((out / model).glob("seed_*.npz"))]
        for drug in drugs:
            for dose in MEASURED_NM:
                cells[f"{model}::{key(drug, dose, 'effect')}"] = np.concatenate([z[key(drug, dose, "effect")] for z in sweeps])
    return cells


def collect(arm: str, root: Path, panel: str) -> None:
    out = out_dir(root)
    curves = pd.read_csv(out / "curves.csv")
    models = [m for m in (arm, CHEMCPA, BACKBONE, BASE, CHEMCPA_ORIGINAL) if m in set(curves.model)]
    table = by_drug(curves, models)
    summary, paired = medians(table, arm)

    # The pooled values must reproduce dose_collect's gain.csv, which uses the same parts.
    earlier = pd.read_csv(out / "gain.csv").set_index(["model", "measure"]).value
    for row in summary.itertuples():
        if not np.isclose(row.pooled, earlier[row.model, f"gain_{row.axis}"]):
            raise SystemExit(f"{row.model} {row.axis}: pooled {row.pooled} differs from gain.csv {earlier[row.model, f'gain_{row.axis}']}")

    cases = pick_cases(table)
    doses = by_dose(curves, models)
    data = Data(panel)
    dirs = directions(data)
    program = set(np.asarray(data.spec.program_idx).tolist())
    cases["program_genes_in_top50"] = {d: len(program & set(deg.tolist())) for d, (deg, _) in dirs.items()}  # the score's genes on this axis
    cells = case_cells(data, dirs, [cases["median"], cases["q75"]], models, out)

    table.to_csv(out / "gain_by_drug.csv", index=False)
    summary.to_csv(out / "gain_median.csv", index=False)
    paired.to_csv(out / "gain_paired_median.csv", index=False)
    doses.to_csv(out / "gain_by_dose.csv", index=False)
    (out / "gain_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    np.savez_compressed(out / "gain_cells.npz", **cells)

    whole = table[table.seed == ALL_SEEDS]
    for column in AXES:
        print(f"\nper-drug gain, {column}")
        print(whole[whole.axis == column].pivot(index="drug", columns="model", values="gain").to_string(float_format=lambda v: f"{v:.2f}"))
        print(f"\nper-drug unseen, {column}")
        print(whole[whole.axis == column].pivot(index="drug", columns="model", values="unseen").to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n" + summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + paired.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + doses.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n" + json.dumps(cases, indent=1))


def draw(arm: str, root: Path) -> None:
    from matplotlib.lines import Line2D

    from .cellflow_figures import ARM_LABELS, COLORS, INK, INK_2, MUTED, RAMP, ROW_LABELS, SURFACE, dose_text, finish, margins, setup

    plt = setup()
    out = out_dir(root)
    curves = pd.read_csv(out / "curves.csv")
    table = pd.read_csv(out / "gain_by_drug.csv")
    summary = pd.read_csv(out / "gain_median.csv").set_index(["axis", "model"])
    cases = json.loads((out / "gain_cases.json").read_text())
    cells = np.load(out / "gain_cells.npz")

    whole = table[(table.axis == "effect") & (table.seed == ALL_SEEDS)]
    gain = whole.pivot(index="drug", columns="model", values="gain")
    drawn = [arm, CHEMCPA, BASE, CHEMCPA_ORIGINAL]
    labels = {arm: ARM_LABELS.get(arm, arm), CHEMCPA: ROW_LABELS["chemcpa"], BASE: ROW_LABELS["cellflow"], CHEMCPA_ORIGINAL: "chemCPA-Gauss (original)",
              "oracle": "Oracle: replicate well 2 vs 1"}
    style = {arm: (COLORS["arm"], "-"), CHEMCPA: (COLORS["chemcpa"], "-"), BASE: (COLORS["cellflow"], "-"), CHEMCPA_ORIGINAL: ("#e87ba4", (0, (5, 2.5))),
             "oracle": (COLORS["oracle"], "-")}
    case_drugs = [cases["median"], cases["q75"]]
    short = lambda d: d.split(" (")[0]
    head = lambda m: summary.loc[("effect", m), "gain"]
    note = ("Dose gain, per drug: slope of generated on measured mean effect over the drug's 4 doses, along its top-50 DEGs and measured 10 µM direction. "
            "Same control cells at every dose. Validation drugs only; no test cell is used. The arm and its embedding were chosen on these drugs' held-out doses, "
            "so its numbers are optimistic. Three of the 4 doses are training doses. Each model is at its lowest validation MMD² checkpoint. Drugs are ranked by "
            "the two baselines' mean gain, so the arm plays no part in picking them; the 75th percentile is where the baselines do better than usual.")
    title = (f"CellFlow retrain and original chemCPA reproduce {100 * head(BASE):.0f}% and {100 * head(CHEMCPA_ORIGINAL):.0f}% of the measured dose "
             f"response; the arm {100 * head(arm):.0f}%")

    # The number and the two cases.
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    margins(fig, 2.5, 0.25, top=1.28, bottom=1.5, wspace=0.6)
    ax = axes[0]
    rows = ["oracle", *drawn]
    y = {m: len(rows) - 1 - i for i, m in enumerate(rows)}
    offset = dict(zip(gain.index, np.linspace(-0.2, 0.2, len(gain))))
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    ax.axvline(1, color=INK_2, lw=1.0, zorder=1)
    for m in rows:
        color = style[m][0]
        ax.scatter(gain[m], [y[m] + offset[d] for d in gain.index], s=32, color=color, alpha=0.55, linewidths=0, zorder=3)
        for d in case_drugs:
            ax.scatter([gain.loc[d, m]], [y[m] + offset[d]], s=68, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=4)
        value = head(m)
        ax.plot([value, value], [y[m] - 0.34, y[m] + 0.34], color=color, lw=3.2, solid_capstyle="round", zorder=5)
        text = f"{value:.2f}"
        if m != "oracle":
            text += f"\nseeds {summary.loc[('effect', m), 'seed_low']:.2f}–{summary.loc[('effect', m), 'seed_high']:.2f}"
        ax.text(1.02, y[m], text, transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK_2, fontweight="bold" if m == arm else "normal")
    ax.set_yticks(list(y.values()), [labels[m].replace(" + ", " +\n").replace(": ", ":\n") for m in y], color=INK)
    ax.get_yticklabels()[rows.index(arm)].set_fontweight("bold")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.12, 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dose gain (0: ignores dose, 1: full response)")
    ax.set_title("Each dot a drug; bar and number: median of 9")

    for ax, drug, name in zip(axes[1:], case_drugs, ("Median case", "75th percentile case")):
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        handles, values = [], []
        for m in reversed(drawn):
            color, ls = style[m]
            part = curves[(curves.model == m) & (curves.drug == drug)]
            by = part.groupby("dose").effect
            ax.fill_between(by.min().index, by.min(), by.max(), color=color, alpha=0.14, linewidths=0)
            ax.plot(by.mean().index, by.mean(), color=color, ls=ls, lw=2.4 if m == arm else 1.6, zorder=4 if m == arm else 3)
        for m in drawn:
            handles.append(Line2D([], [], color=style[m][0], ls=style[m][1], lw=2.4 if m == arm else 1.6))
            values.append(f"{gain.loc[drug, m]:.2f}")
        r = curves[(curves.model == "measured") & (curves.drug == drug)].sort_values("dose")
        ax.plot(r.dose, r.effect, color=INK, lw=1.2, zorder=5)
        seen = r[~r.unseen]
        ax.scatter(seen.dose, seen.effect, s=34, color=INK, zorder=6)
        ax.scatter(r[r.unseen].dose, r[r.unseen].effect, s=38, color=SURFACE, edgecolors=INK, linewidths=1.6, zorder=6)
        ax.set_xscale("log")
        ax.set_xticks(list(MEASURED_NM), [dose_text(t) for t in MEASURED_NM])
        ax.minorticks_off()
        ax.set_xlabel("Dose")
        ax.set_ylabel("Mean effect along the drug's direction")
        ax.set_title(f"{name}: {short(drug)}")
        ax.legend(handles, values, title=f"Gain on this drug\n(baselines' mean {cases['baseline_gain'][drug]:.2f})", loc="upper left", fontsize=8.5,
                  title_fontsize=8.5, handlelength=2.0, alignment="left")
    w, h = fig.get_size_inches()
    handles = [Line2D([], [], color=INK, marker="o", lw=1.2, ms=5.5), Line2D([], [], color=INK, marker="o", mfc=SURFACE, mew=1.5, lw=0, ms=6)]
    handles += [Line2D([], [], color=style[m][0], ls=style[m][1], lw=2.4 if m == arm else 1.6) for m in drawn]
    handles.append(Line2D([], [], color=INK, marker="o", mfc="none", mew=1.1, lw=0, ms=8))
    fig.legend(handles, ["Measured, training dose", "Measured, unseen dose", *[labels[m] for m in drawn], "Ringed: the two drugs at right"], loc="upper left",
               bbox_to_anchor=(0.008, 1 - 0.62 / h), ncol=4, fontsize=8.5, handlelength=2.0, columnspacing=1.4)
    finish(plt, fig, out / "6_dose_gain_cases.png", title,
           "How far a model's population moves across a drug's doses, relative to how far the measured cells move. Line: mean of 3 seeds; band: seed range.", note)

    # The same two cases as populations of cells.
    ramp = [RAMP[1], "#5598e7", RAMP[3], RAMP[4]]  # ordinal: light = low dose
    rows = ["measured", *drawn]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11.5, 1.45 * len(rows) + 2.35), sharex="col")
    margins(fig, 2.25, 0.25, top=1.15, bottom=1.45, hspace=0.16, wspace=0.1)
    for col, (drug, name) in enumerate(zip(case_drugs, ("Median case", "75th percentile case"))):
        lo, hi = np.percentile(np.concatenate([cells[f"measured::{key(drug, x, 'effect')}"] for x in MEASURED_NM]), [0.5, 99.5])
        edges = np.linspace(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo), 30)  # tails fall outside: clipping would draw false spikes
        for i, m in enumerate(rows):
            ax = axes[i, col]
            for x, color in zip(MEASURED_NM, ramp):
                ax.hist(cells[f"{m}::{key(drug, x, 'effect')}"], edges, density=True, histtype="step", color=color, lw=1.5)
            ax.set_yticks([])
            ax.grid(False)
            if m != "measured":
                ax.text(0.985, 0.82, f"gain {gain.loc[drug, m]:.2f}", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_2,
                        fontweight="bold" if m == arm else "normal")
            if col == 0:
                ax.set_ylabel("Measured" if m == "measured" else labels[m].replace(" + ", " +\n"), rotation=0,
                              ha="right", va="center", fontsize=8.5, color=INK, labelpad=8)
        axes[0, col].set_title(f"{name}: {short(drug)}")
        axes[-1, col].set_xlabel("Effect along the drug's direction, per cell")
    w, h = fig.get_size_inches()
    fig.legend([Line2D([], [], color=c, lw=1.8) for c in ramp], [dose_text(x) for x in MEASURED_NM], loc="upper left", bbox_to_anchor=(0.008, 1 - 0.62 / h),
               ncol=4, fontsize=8.5, handlelength=1.8)
    finish(plt, fig, out / "7_dose_gain_cells.png", "The same two drugs as populations of cells",
           "Four doses overlaid per row. Compare how far apart each row's lines sit with the Measured row. Generated cells pool 3 seeds.", note)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["collect", "draw"])
    p.add_argument("--arm", required=True)
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    if a.step == "collect":
        collect(a.arm, Path(a.root), a.data)
    else:
        draw(a.arm, Path(a.root))


if __name__ == "__main__":
    main()
