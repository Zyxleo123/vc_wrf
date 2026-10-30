"""Side-by-side table and threshold-sensitivity figure for chemCPA report directories.

Reads only report_chemcpa outputs (summary.json, threshold_sensitivity.csv), so it
redraws in seconds. All reports must cover the same split and sampling offset.

usage: python -m sciplex_baselines.compare_chemcpa --reports R1 R2 --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .report_chemcpa import COLORS, LABELS, plot_sensitivity

MAX_PLOTTED_MODELS = 3  # categorical palette cap; the table lists every report


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
    scored = list(summaries[0]["across_seeds"])
    frozen_qs = sorted(float(k.split("_q")[1]) for k in scored if k.startswith("wrf_err_pp_q"))
    metrics = [f"wrf_err_pp_q{q:g}" for q in frozen_qs] + [k for k in scored if not k.startswith("wrf_err")]
    rows = []
    for s in summaries:
        rows.append({"method": s["label"], "variant": s["variant"], "generation": s["generation"],
                     "selected_epochs": [c["epoch"] for c in s["selected_checkpoints"]],
                     **{f"{m}_mean": s["across_seeds"][m]["mean"] for m in metrics},
                     **{f"{m}_seed_sd": s["across_seeds"][m]["sd"] for m in metrics}})
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
            sd = row.get(f"{m}_seed_sd")
            cells.append(f"{row[f'{m}_mean']:.4f}" + (f" ± {sd:.4f}" if sd is not None else ""))
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
    lines += ["", "![Threshold sensitivity](threshold_sensitivity.png)", ""]
    (out / "comparison.md").write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    shown = curves[:min(n_models, MAX_PLOTTED_MODELS)] + curves[n_models:]
    # A family keeps its fixed color; a repeated family takes the first unused slot.
    slots, used, colors = list(COLORS.values()), set(), []
    for key, _, _ in shown[:-2]:
        color = COLORS[key] if COLORS[key] not in used else next(c for c in slots if c not in used)
        used.add(color)
        colors.append(color)
    plot_sensitivity(ax, shown, primary_q, frozen_qs, colors + [None, None])
    ax.set_title(f"WRF error vs threshold, {split} (vertical lines: frozen v1 thresholds)"
                 + (f"\nfirst {MAX_PLOTTED_MODELS} of {n_models} models shown; all are in comparison.md"
                    if n_models > MAX_PLOTTED_MODELS else ""))
    fig.savefig(out / "threshold_sensitivity.png", dpi=160)
    fig.savefig(out / "threshold_sensitivity.pdf")
    plt.close(fig)
    print(out / "comparison.md")


if __name__ == "__main__":
    main()
