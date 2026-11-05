"""Side-by-side table and threshold-sensitivity figure for chemCPA report directories.

Reads only report_chemcpa outputs (summary.json, threshold_sensitivity.csv), so it
redraws in seconds. All reports must cover the same split and sampling offset.
Reports may score different metric sets (program-gene-output runs have no MMD² or
all-gene/DEG RMSE); a metric a report lacks is left blank, never imputed.

Paired differences (paired_differences.csv): for every pair of reports when there are
at most MAX_PLOTTED_MODELS, otherwise each report against the first, the per-condition
difference of seed-averaged errors with a condition bootstrap (sciplex_analysis's
paired_stats). Negative = the second model has lower error.

usage: python -m sciplex_baselines.compare_chemcpa --reports R1 R2 --out DIR
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sciplex_analysis.analyze import paired_stats

from .report_chemcpa import COLORS, LABELS, PROGRAM_OUTPUT_FAMILIES, SLOTS, plot_sensitivity

# Line chart, so the adjacent-pair palette check applies: up to four categorical
# slots (the full 2x2 of gene set x loss). The table lists every report.
MAX_PLOTTED_MODELS = 4
N_BOOT = 2000
PAIRED_SHOWN = ("wrf_err_pp_q0.95", "w1_r")


def paired_differences(summaries, reports, metrics, n_max=MAX_PLOTTED_MODELS, seed=0):
    """new − ref per condition (seed-averaged), condition-bootstrap 95% CI, per metric both score."""
    frames = [pd.read_csv(Path(r) / "per_condition.csv") for r in reports]
    n = len(frames)
    pairs = itertools.combinations(range(n), 2) if n <= n_max else ((0, j) for j in range(1, n))
    rng, rows = np.random.default_rng(seed), []
    for i, j in pairs:
        for m in metrics:
            if m not in frames[i] or m not in frames[j]:
                continue
            ref, new = (f.pivot_table(index="condition", columns="seed", values=m, aggfunc="mean") for f in (frames[i], frames[j]))
            st = paired_stats(ref, new, rng, N_BOOT)
            rows.append({"ref": summaries[i]["label"], "ref_variant": summaries[i]["variant"],
                         "new": summaries[j]["label"], "new_variant": summaries[j]["variant"], "metric": m,
                         "n_conditions": st["n_conditions"], "mean_ref": st["mean_ref"], "mean_new": st["mean_new"],
                         "delta_new_minus_ref": -st["delta_mean"],
                         "ci95_low": -st["ci95_high"], "ci95_high": -st["ci95_low"],
                         "ci95_two_level_low": -st["ci95_two_level_high"], "ci95_two_level_high": -st["ci95_two_level_low"],
                         "n_new_better": st["n_win"], "n_new_worse": st["n_loss"],
                         "seed_pairs_new_better": st["seed_pairs_positive"]})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((Path(r) / "summary.json").read_text()) for r in a.reports]
    keys = {(s["split"], s["sampling_offset"], s["primary_q"]) for s in summaries}
    if len(keys) != 1:
        raise ValueError(f"reports differ in split/sampling offset/primary q: {keys}")
    split, offset, primary_q = keys.pop()
    scored = list(max((s["across_seeds"] for s in summaries), key=len))
    scored += [k for s in summaries for k in s["across_seeds"] if k not in scored]
    frozen_qs = sorted(float(k.split("_q")[1]) for k in scored if k.startswith("wrf_err_pp_q"))
    metrics = [f"wrf_err_pp_q{q:g}" for q in frozen_qs] + [k for k in scored if not k.startswith("wrf_err")]
    rows = []
    for s in summaries:
        rows.append({"method": s["label"], "variant": s["variant"], "generation": s["generation"],
                     "selected_epochs": [c["epoch"] for c in s["selected_checkpoints"]],
                     **{f"{m}_mean": s["across_seeds"].get(m, {}).get("mean", float("nan")) for m in metrics},
                     **{f"{m}_seed_sd": s["across_seeds"].get(m, {}).get("sd", float("nan")) for m in metrics}})
    for name in ("vehicle", "replicate"):
        rows.append({"method": LABELS[name], **{f"{m}_mean": summaries[0]["references"][name][m]["mean"] for m in metrics}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)

    lines = [f"# chemCPA comparison: {split}", "",
             f"Frozen v1 metrics, equal-weight mean over conditions; model rows are mean ± SD over seeds "
             f"(sampling offset {offset}). Vehicle and replicate rows are the shared references. "
             f"Primary metric: wrf_err_pp_q{primary_q:g}.", "",
             "| Method | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for row in rows:
        cells = []
        for m in metrics:
            mean, sd = row[f"{m}_mean"], row.get(f"{m}_seed_sd")
            if mean != mean:  # NaN: this report does not score the metric
                cells.append("–")
                continue
            cells.append(f"{mean:.4f}" + (f" ± {sd:.4f}" if sd is not None else ""))
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
    curves, frames = [], []
    for s, report in zip(summaries, a.reports):
        frame = pd.read_csv(Path(report) / "threshold_sensitivity.csv")
        curves.append((s["family"], s["label"], frame[frame.method == s["family"]]))
        frames.append(frame)
    n_models = len(curves)
    for name in ("vehicle", "replicate"):
        curves.append((name, LABELS[name], frames[0][frames[0].method == name]))
    lines += ["", "## Supplementary: WRF error (pp) across control-quantile thresholds", "",
              "| q | measured WRF | " + " | ".join(label for _, label, _ in curves) + " |",
              "|---:|---:|" + "---:|" * len(curves)]
    measured = curves[0][2].set_index("q").wrf_real
    for q in measured.index:
        lines.append(f"| {q:g} | {100 * measured[q]:.1f}% | "
                     + " | ".join(f"{frame.set_index('q').err_pp[q]:.2f}" for _, _, frame in curves) + " |")
    if any(s["family"] in PROGRAM_OUTPUT_FAMILIES for s in summaries):
        lines += ["", "–: not scored. Program-gene-output runs generate only the program genes, so MMD² and the "
                  "all-gene and DEG RMSEs are undefined for them."]
    paired = paired_differences(summaries, a.reports, metrics)
    paired.to_csv(out / "paired_differences.csv", index=False)
    shown = paired[paired.metric.isin(PAIRED_SHOWN)] if len(paired) else paired
    if len(shown):
        lines += ["", "## Paired differences (new − ref, per condition, seed-averaged)", "",
                  f"Condition bootstrap 95% CI ({N_BOOT} resamples); the two-level CI also resamples seeds. "
                  "Negative = the new model has lower error. All metrics: paired_differences.csv.", "",
                  "| ref | new | metric | Δ | 95% CI | two-level CI | new better | seed pairs new better |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in shown.itertuples():
            lines.append(f"| {r.ref_variant} | {r.new_variant} | {r.metric} | {r.delta_new_minus_ref:+.3f} | "
                         f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | [{r.ci95_two_level_low:+.3f}, {r.ci95_two_level_high:+.3f}] | "
                         f"{r.n_new_better}/{r.n_conditions} | {100 * r.seed_pairs_new_better:.0f}% |")
    lines += ["", "![Threshold sensitivity](threshold_sensitivity.png)", ""]
    (out / "comparison.md").write_text("\n".join(lines))

    shown = curves[:min(n_models, MAX_PLOTTED_MODELS)] + curves[n_models:]
    # Direct labels drop the loss settings (the legend keeps them) when that stays unambiguous.
    short = [label.split(" (")[0] for _, label, _ in shown]
    direct = short if len(set(short)) == len(short) else None
    longest = max(len(label) for _, label, _ in shown)
    fig, ax = plt.subplots(figsize=(max(8, 2.5 + 0.13 * longest), 6), constrained_layout=True)
    # A family keeps its fixed color; a repeated family takes the first slot not yet
    # used with its marker style (program-output families draw hollow squares).
    used, colors = set(), []
    for key, _, _ in shown[:-2]:
        hollow = key in PROGRAM_OUTPUT_FAMILIES
        color = COLORS[key] if (COLORS[key], hollow) not in used else next(c for c in SLOTS if (c, hollow) not in used)
        used.add((color, hollow))
        colors.append(color)
    plot_sensitivity(ax, shown, primary_q, frozen_qs, colors + [None, None], direct)
    ax.set_title(f"WRF error vs threshold, {split} (vertical lines: frozen v1 thresholds)"
                 + (f"\nfirst {MAX_PLOTTED_MODELS} of {n_models} models shown; all are in comparison.md"
                    if n_models > MAX_PLOTTED_MODELS else ""))
    fig.savefig(out / "threshold_sensitivity.png", dpi=160)
    fig.savefig(out / "threshold_sensitivity.pdf")
    plt.close(fig)
    print(out / "comparison.md")


if __name__ == "__main__":
    main()
