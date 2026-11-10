"""Quick, no-refit held-out evaluation of the saved response embedding in gene space.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sciplex_interp.teacher_mean_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DEFAULT_PANEL, DEFAULT_ROOT, Data, mean_errors


def plot_histograms(records, out):
    """Histogram entries are genes, after averaging over all cells of each condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = {"truth": "Ground truth", "predicted": "Embedding prediction", "adjacent": "Adjacent-dose interpolation"}
    colors = {"truth": "#252525", "predicted": "#0072B2", "adjacent": "#D55E00"}
    styles = {"truth": "-", "predicted": "-", "adjacent": "--"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def draw(ax, values):
        bins = np.linspace(min(v.min() for v in values.values()), max(v.max() for v in values.values()), 65)
        ax.hist(values["truth"], bins=bins, density=True, color=colors["truth"], alpha=.10)
        for name, vec in values.items():
            ax.hist(vec, bins=bins, density=True, histtype="step", linewidth=1.6,
                    color=colors[name], linestyle=styles[name])
        ax.axvline(0, color="#888888", lw=.7, alpha=.5)
        ax.grid(axis="y", alpha=.15)

    legend = [Line2D([], [], color=colors[k], ls=styles[k], lw=2, label=labels[k]) for k in labels]
    ranked = sorted((r for r in records if r["split"] == "test"), key=lambda r: np.linalg.norm(r["truth"]))
    selected = [ranked[round(q * (len(ranked) - 1))] for q in (.5, .75)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, rec, quantile in zip(axes, selected, ("Median response", "Upper-quartile response")):
        draw(ax, {k: rec[k] for k in labels})
        ax.set_yscale("log")
        ax.set_title(f"{rec['drug']} · {rec['dose_nM']:g} nM\n{quantile} · {rec['n_cells']:,} treated cells", fontsize=11)
        ax.set_xlabel("Gene-wise mean shift from vehicle")
        print(f"Histogram example: {rec['drug']}, {rec['dose_nM']:g} nM ({quantile.lower()})")
    axes[0].set_ylabel("Density across genes (log scale)")
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
    fig.suptitle("Mean response in two individual held-out test conditions", fontsize=15, y=.99)
    fig.text(.5, .02, "2,085 genes per condition · mean over all treated cells minus vehicle mean · log-expression units",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, .81))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"mean_response_histogram_examples_test.{ext}", dpi=200)
    plt.close(fig)
    for split in ("val", "test"):
        part = [r for r in records if r["split"] == split]
        ncols = 3
        nrows = (len(part) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.65 * nrows + 1), squeeze=False)
        for ax, rec in zip(axes.flat, part):
            draw(ax, {k: rec[k] for k in labels})
            ax.set_title(f"{rec['drug']} · {rec['dose_nM']:g} nM\n{rec['n_cells']:,} treated cells", fontsize=10)
        for ax in list(axes.flat)[len(part):]:
            ax.set_visible(False)
        fig.suptitle(f"Mean-response distributions · held-out {split} conditions", y=.992, fontsize=16)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .967), ncol=3, frameon=False)
        fig.supxlabel("Gene-wise mean shift from vehicle (log-expression units)", y=.025)
        fig.supylabel("Density across genes")
        fig.text(.5, .006, "Each histogram contains 2,085 gene means. All treated cells pooled across replicates; axis ranges vary by condition.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(.025, .045, 1, .935))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"mean_response_histograms_{split}.{ext}", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        draw(ax, {k: np.concatenate([r[k] for r in part]) for k in labels})
        ax.set(title=f"Mean response · {len(part)} held-out {split} conditions pooled",
               xlabel="Gene-wise mean shift from vehicle (log-expression units)", ylabel="Density across condition–gene pairs")
        ax.legend(handles=legend, frameon=False)
        fig.text(.5, .015, "Each entry is one gene's mean over treated cells, minus its vehicle mean. Conditions have equal weight.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .045, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"mean_response_histogram_pooled_{split}.{ext}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_PANEL)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plot", action="store_true", help="Save overlaid gene-wise mean-shift histograms")
    args = parser.parse_args()
    saved = np.load(args.root / "teacher" / "autodec.npz")
    basis = saved["basis"].astype(float)
    control = saved["control_mean"].astype(float)
    lookup = {tuple(json.loads(k)): c.astype(float) for k, c in zip(saved["keys"], saved["coef"])}
    assert basis.shape[0] == 20
    np.testing.assert_allclose(basis @ basis.T, np.eye(20), atol=1e-6)
    data = Data(args.data)
    np.testing.assert_allclose(control, data.ctrl_mean, atol=1e-6)
    rows, histograms = [], []
    for split in ("val", "test"):
        for task in data.tasks(split):
            assert data.cond.loc[task.target, "split"] == split
            assert all(data.cond.loc[data.condition_of(task.drug, d), "split"] == "train" for d in (task.low, task.high))
            truth = data.mean(task.target) - control
            predicted = lookup[(task.drug, task.dose)] @ basis
            adjacent = ((1 - task.w_logdose) * data.mean(data.condition_of(task.drug, task.low))
                        + task.w_logdose * data.mean(data.condition_of(task.drug, task.high)) - control)
            histograms.append(dict(split=split, drug=task.drug, dose_nM=task.dose,
                                   n_cells=len(data.idx(task.target)), truth=truth, predicted=predicted, adjacent=adjacent))
            # Oracle is diagnostic only: true held-out coefficients on the fixed training basis.
            oracle = (truth @ basis.T) @ basis
            for method, delta in (("embedding20", predicted), ("adjacent", adjacent),
                                  ("zero", np.zeros_like(truth)), ("oracle20", oracle)):
                sse = float(np.sum((delta - truth) ** 2))
                energy = float(truth @ truth)
                denom = np.linalg.norm(delta) * np.linalg.norm(truth)
                rows.append(dict(split=split, condition=task.target, drug=task.drug, dose_nM=task.dose,
                                 method=method, **mean_errors(data, task.target, control + delta),
                                 cosine=float(delta @ truth / denom) if denom > 0 else np.nan,
                                 shift_skill=1 - sse / energy, sse=sse, true_shift_energy=energy,
                                 norm_ratio=float(np.linalg.norm(delta) / np.linalg.norm(truth))))
    frame = pd.DataFrame(rows)
    metrics = ["rmse_mean_all", "rmse_mean_deg50", "rmse_mean_program", "pearson_delta_all", "cosine", "shift_skill", "norm_ratio"]
    summary = frame.groupby(["split", "method"], sort=False)[metrics].mean()
    pooled = frame.groupby(["split", "method"], sort=False)[["sse", "true_shift_energy"]].sum()
    summary["pooled_shift_skill"] = 1 - pooled.sse / pooled.true_shift_energy
    paired = []
    rng = np.random.default_rng(0)
    for split in ("val", "test"):
        for metric in ("rmse_mean_all", "rmse_mean_deg50"):
            wide = frame[frame.split == split].pivot(index="condition", columns="method", values=metric)
            diff = (wide.embedding20 - wide.adjacent).to_numpy()
            boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(1)
            paired.append(dict(split=split, metric=metric, difference=float(diff.mean()),
                               ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                               wins=int((diff < 0).sum()), n=len(diff)))
    out = args.root / "teacher" / "mean_check"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "per_condition.csv", index=False)
    summary.to_csv(out / "summary.csv")
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    if args.plot:
        plot_histograms(histograms, out)
    print(summary.round(5).to_string())
    print(pd.DataFrame(paired).round(5).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
