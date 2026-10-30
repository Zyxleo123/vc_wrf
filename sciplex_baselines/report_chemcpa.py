"""Aggregate frozen chemCPA runs without treating synthetic cells as replicates.

Reads only saved artefacts (per-condition scores, generated cells, the panel), so the
tables and figures can be redrawn without the model. Besides the frozen v1 metrics it
writes a supplementary threshold-sensitivity table: WRF error at control quantiles
beyond the frozen ones, using the same r(x) and the same tau rule (q-quantile of
control r). It is not a metric choice; the frozen thresholds stay primary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sciplex_eval import EvalSpec, response_score, summarize
from sciplex_eval.evaluate import DEFAULT_PANEL, DEFAULT_SPEC, _real, load_panel, replicate_reference, split_conditions
from sciplex_eval.metrics import evaluate_condition, weak_responder_fraction
from .chemcpa import GENERATIONS, write_json

SENSITIVITY_QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]
# Fixed per entity (dataviz reference palette, categorical slots 1-3); references are neutral.
COLORS = {"nb": "#2a78d6", "gauss": "#eb6834", "nbwrf": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REFERENCE_STYLE = {"vehicle": (INK_2, (0, (6, 3))), "replicate": (INK_2, (0, (1, 2.5)))}
LABELS = {"nb": "chemCPA-NB", "gauss": "chemCPA-Gauss", "nbwrf": "chemCPA-NB + WRF loss",
          "vehicle": "Vehicle as prediction", "replicate": "Replicate well (noise floor)"}


def family_and_label(config):
    """Plot identity of a run: its likelihood, or nbwrf when trained with the fraction loss."""
    if "fraction_loss" in config:
        settings = config["fraction_loss"]
        n_taus = len(settings["quantiles"]) if "quantiles" in settings else "spec"
        return "nbwrf", f"{LABELS['nbwrf']} ({n_taus} τs, λ={settings['weight']:g}, h={settings['h']:g})"
    return config["likelihood"], LABELS[config["likelihood"]]


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def threshold_sensitivity(spec, panel, split, predictions, qs=SENSITIVITY_QS):
    """WRF error per (method, condition, q). `predictions` maps seed -> AnnData."""
    controls = spec.align(panel.X[panel.obs.split.eq("control").to_numpy()], panel.var_names)
    r_ctrl = response_score(controls, spec)
    taus = np.quantile(r_ctrl, qs)
    pred_r = {seed: (response_score(spec.align(p.X, p.var_names), spec), p.obs.condition.astype(str).to_numpy())
              for seed, p in predictions.items()}
    rows = []
    for cond in split_conditions(panel, split):
        f_real = weak_responder_fraction(response_score(_real(panel, spec, cond), spec), taus)
        by_method = {"model": np.mean([weak_responder_fraction(r[c == cond], taus) for r, c in pred_r.values()], axis=0),
                     "vehicle": weak_responder_fraction(r_ctrl, taus)}
        reps = [weak_responder_fraction(response_score(_real(panel, spec, cond, rep), spec), taus)
                for rep in ("rep1", "rep2")]
        for q, tau, fr, f1, f2 in zip(qs, taus, f_real, *reps):
            rows.append({"method": "replicate", "condition": cond, "q": q, "tau": tau,
                         "wrf_real": fr, "wrf_pred": np.nan, "err_pp": 100 * abs(f1 - f2), "bias_pp": np.nan})
        for method, f_pred in by_method.items():
            for q, tau, fr, fp in zip(qs, taus, f_real, f_pred):
                # Seed-averaged prediction; per-seed errors are in per_condition.csv at frozen q.
                rows.append({"method": method, "condition": cond, "q": q, "tau": tau, "wrf_real": fr,
                             "wrf_pred": fp, "err_pp": 100 * abs(fp - fr), "bias_pp": 100 * (fp - fr)})
    table = pd.DataFrame(rows)
    return table.groupby(["method", "q"], as_index=False).agg(
        tau=("tau", "first"), wrf_real=("wrf_real", "mean"), wrf_pred=("wrf_pred", "mean"),
        err_pp=("err_pp", "mean"), bias_pp=("bias_pp", "mean"), n_conditions=("condition", "nunique"))


def plot_sensitivity(ax, curves, primary_q, frozen_qs, colors=None):
    """curves: list of (key, label, DataFrame with q, err_pp); colors: optional
    per-curve overrides for model curves (None keeps the key's fixed color)."""
    colors = colors or [None] * len(curves)
    for (key, label, frame), override in zip(curves, colors):
        color, dash = REFERENCE_STYLE.get(key, (override or COLORS.get(key, INK), "solid"))
        marker = None if key in REFERENCE_STYLE else "o"
        ax.plot(frame.q, frame.err_pp, color=color, linestyle=dash, linewidth=2, marker=marker,
                markersize=5, markeredgecolor="white", markeredgewidth=1, label=label)
    for q in frozen_qs:
        ax.axvline(q, color=INK_2, linewidth=0.8, alpha=0.6 if q == primary_q else 0.3)
    ax.set_ylim(bottom=0)
    # Direct labels at the right end, nudged apart so they never overlap.
    low, high = ax.get_ylim()
    gap, previous = 0.05 * (high - low), -np.inf
    for y, label, x in sorted((f.err_pp.iloc[-1], label, f.q.iloc[-1]) for _, label, f in curves):
        previous = max(y, previous + gap)
        ax.annotate(label, (x, y), xytext=(1.0, previous), textcoords="data", va="center",
                    fontsize=8, color=INK_2)
    ax.set(xlim=(0, 1.32), xticks=np.linspace(0, 1, 6), xlabel="Threshold: q-quantile of control response score",
           ylabel="WRF absolute error (pp, mean over conditions)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)
    style(ax)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--generation", choices=GENERATIONS, default="sample")
    p.add_argument("--sampling-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec, panel = EvalSpec.load(a.spec), load_panel(a.data)
    paths = [Path(a.runs) / f"seed_{seed}" / f"{a.split}_{a.generation}_s{a.sampling_seed}" for seed in a.seeds]
    runs = [json.loads((path.parent / "run.json").read_text()) for path in paths]
    if any(run["config"] != runs[0]["config"] for run in runs):
        raise ValueError("seeds were trained with different configs")
    family, label = family_and_label(runs[0]["config"])
    records = pd.concat([pd.read_json(path / "per_condition.jsonl", lines=True) for path in paths], ignore_index=True)
    wells = pd.concat([pd.read_json(path / "per_well.jsonl", lines=True) for path in paths], ignore_index=True)
    metrics = [c for c in records if c.startswith(("wrf_err", "w1_", "mmd", "rmse"))]
    records.to_csv(out / "per_condition.csv", index=False)
    wells.to_csv(out / "per_well.csv", index=False)
    per_seed = records.groupby("seed")[metrics].mean()
    per_seed.to_csv(out / "per_seed.csv")
    condition_means = records.groupby("condition")[metrics].mean().reset_index()
    result = {
        "method": f"{runs[0]['method']}_{a.generation}", "family": family, "label": label,
        "variant": Path(a.runs).name, "likelihood": runs[0]["config"]["likelihood"],
        "generation": a.generation, "split": a.split, "seeds": a.seeds,
        "primary_q": spec.primary_q, "sampling_offset": a.sampling_seed,
        "uncertainty": "CI resamples conditions after averaging seed-specific errors. Seed SD is training variability. Neither is a biological-replicate CI; wells were not resampled for training.",
        "condition_bootstrap": summarize(condition_means.to_dict("records")),
        "across_seeds": {k: {"mean": float(per_seed[k].mean()), "sd": float(per_seed[k].std()),
                              "values": [float(x) for x in per_seed[k]]} for k in metrics},
        "runs": runs,
        "selected_checkpoints": [json.loads((path.parent / "selected.json").read_text()) for path in paths],
        "control_diagnostics": [json.loads((path / "control_diagnostic.json").read_text()) for path in paths],
    }
    refs = {"replicate": replicate_reference(spec, panel, a.split)}
    controls = spec.align(panel.X[panel.obs.split.eq("control").to_numpy()], panel.var_names)
    refs["vehicle"] = [{"condition": c, **evaluate_condition(spec, _real(panel, spec, c), X_pred=controls)}
                       for c in split_conditions(panel, a.split)]
    result["references"] = {name: summarize(rows) for name, rows in refs.items()}
    pd.concat([pd.DataFrame(rows).assign(method=name) for name, rows in refs.items()]).to_csv(out / "references.csv", index=False)

    predictions = {seed: ad.read_h5ad(path / "predictions.h5ad") for seed, path in zip(a.seeds, paths)}
    sensitivity = threshold_sensitivity(spec, panel, a.split, predictions)
    sensitivity["method"] = sensitivity["method"].replace({"model": family})
    sensitivity.to_csv(out / "threshold_sensitivity.csv", index=False)
    result["threshold_sensitivity"] = sensitivity.to_dict("records")
    write_json(out / "summary.json", result)

    q = f"q{spec.primary_q:g}"
    grouped = records.groupby("condition")
    true = grouped[f"wrf_real_{q}"].mean() * 100
    pred = grouped[f"wrf_pred_{q}"].mean() * 100
    pred_sd = grouped[f"wrf_pred_{q}"].std().fillna(0) * 100
    err = grouped[f"wrf_err_pp_{q}"].mean().sort_values()
    color = COLORS[family]
    fig, axes = plt.subplots(1, 4, figsize=(23, 6), constrained_layout=True)
    axes[0].errorbar(true, pred, yerr=pred_sd, fmt="o", markersize=6, capsize=3, color=color,
                     markeredgecolor="white", elinewidth=1.2)
    limits = [max(0, min(true.min(), (pred - pred_sd).min()) - 4), 101]
    axes[0].plot(limits, limits, "--", color=INK_2, linewidth=1)
    axes[0].set(xlim=limits, ylim=limits, xlabel="Measured weak responders (%)", ylabel="Predicted weak responders (%)",
                title=f"{label}, {a.split}: mean ± SD over {len(a.seeds)} seeds ({q})")
    labels = [c.split("|")[1] + " / " + c.split("|")[2] + " nM" for c in err.index]
    axes[1].barh(labels, err.values, color=color, height=0.6)
    axes[1].set(xlabel="WRF absolute error (percentage points)", title=f"Per condition, frozen primary threshold ({q})")
    axes[1].tick_params(axis="y", labelsize=8)
    worst = grouped.w1_r.mean().idxmax()
    reference_r = response_score(_real(panel, spec, worst), spec)
    axes[2].step(np.sort(reference_r), np.arange(1, len(reference_r) + 1) / len(reference_r), where="post",
                 color=INK, linewidth=2, label="Measured")
    for i, (seed, prediction) in enumerate(predictions.items()):
        r = response_score(spec.align(prediction.X[prediction.obs.condition.eq(worst).to_numpy()], prediction.var_names), spec)
        axes[2].step(np.sort(r), np.arange(1, len(r) + 1) / len(r), where="post", color=color,
                     alpha=1 - 0.25 * i, linewidth=1.5, label=f"{label} seed {seed}")
    axes[2].axvline(spec.tau_by_q[spec.primary_q], color=INK_2, linestyle="--", linewidth=1, label="Primary threshold")
    axes[2].set(xlabel="Response score r (control SD units)", ylabel="Cumulative fraction",
                title="Largest response W1: " + worst.split("|")[1] + " / " + worst.split("|")[2] + " nM")
    axes[2].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    curves = [(family, label, sensitivity[sensitivity.method == family])]
    curves += [(key, LABELS[key], sensitivity[sensitivity.method == key]) for key in ("vehicle", "replicate")]
    plot_sensitivity(axes[3], curves, spec.primary_q, spec.quantiles)
    axes[3].set_title("Supplementary: threshold sensitivity")
    for ax in axes[:3]:
        style(ax)
    fig.savefig(out / "diagnostics.png", dpi=160)
    fig.savefig(out / "diagnostics.pdf")
    plt.close(fig)

    if family == "gauss":
        support = ("Gaussian decoder samples retain negative log-expression values; the shared response scorer "
                   "clips negative program-gene values as specified in v1. The Gaussian decoder has no count "
                   "sparsity, so it cannot place cells at the zero/low-UMI program values the v1 thresholds fall on.")
    else:
        support = ("NB samples are integer UMI draws over the eval genes plus one rest-of-transcriptome feature, "
                   "normalized by their generated total like real cells. Each generated cell keeps the library "
                   "size of its source control cell; treatment-induced library-size shifts are not modelled.")
    if family == "nbwrf":
        settings = runs[0]["config"]["fraction_loss"]
        support += (f" Trained with the response-fraction loss (weight {settings['weight']:g}, h {settings['h']:g}, "
                    "spec taus) on training doses and vehicle only; held-out doses supply no fractions.")
    lines = [f"# {label}: {a.split} evaluation", "",
             f"{len(true)} withheld-dose conditions; seeds {a.seeds}; frozen v1 specification; "
             f"{a.generation} generation (sampling offset {a.sampling_seed}).", "",
             f"| Metric | {label} mean ± seed SD | Condition-bootstrap 95% CI | Vehicle | Replicate disagreement |",
             "|---|---:|---:|---:|---:|"]
    for key in metrics:
        v = result["across_seeds"][key]
        lo, hi = result["condition_bootstrap"][key]["ci95"]
        vehicle = result["references"]["vehicle"][key]["mean"]
        replicate = result["references"]["replicate"][key]["mean"]
        lines.append(f"| {key} | {v['mean']:.4f} ± {v['sd']:.4f} | [{lo:.4f}, {hi:.4f}] | {vehicle:.4f} | {replicate:.4f} |")
    lines += ["", "## Supplementary: WRF error (pp) across thresholds", "",
              "Seed-averaged predicted fractions; tau_q is the q-quantile of control r(x), as in the frozen spec. "
              "Not used for model selection.", "",
              "| q | tau | measured WRF | " + " | ".join(lab for _, lab, _ in curves) + " |",
              "|---:|---:|---:|---:|---:|---:|"]
    wide = sensitivity.pivot(index="q", columns="method", values="err_pp")
    first = sensitivity[sensitivity.method == family].set_index("q")
    for qv in SENSITIVITY_QS:
        lines.append(f"| {qv:g} | {first.tau[qv]:.3f} | {100 * first.wrf_real[qv]:.1f}% | "
                     + " | ".join(f"{wide.loc[qv, k]:.2f}" for k, _, _ in curves) + " |")
    lines += ["", "The bootstrap interval resamples conditions after averaging seed-specific errors. Seed SD reports training variability. These intervals do not establish uncertainty over biological experiments; models were not retrained on resampled wells.", "",
              "The existing v1 threshold is weakly sensitive to drug response, and q=0.975 is degenerate (see the main README). The split remains labelled proposed in the repository. These are exploratory benchmark results under that fixed split.", "",
              "The model uses the pinned 2022 chemCPA code, fixed Morgan fingerprints, a smaller network, and no pretraining. Checkpoints were chosen by validation MMD²; test expression was used only after model selection. "
              + ("No WRF auxiliary loss was applied. " if family != "nbwrf" else "") + "No posthoc response calibration was applied.", "",
              support + " Decoder-mean predictions, when reported, are a separate diagnostic. Control reconstruction metrics in summary.json help assess score distortion.", "",
              "Per-condition and per-well results, seed summaries, configuration, software versions, fingerprints/SMILES provenance, and data/checkpoint hashes accompany this report.", "",
              "![Diagnostics](diagnostics.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(out / "report.md")


if __name__ == "__main__":
    main()
