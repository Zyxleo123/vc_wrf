"""Report validation ablations for a selected CellFlow arm.

This report intentionally excludes response-score and weak-responder metrics.  It reads the
uniform frozen-evaluator rows written by ``sciplex_interp.cellflow evaluate`` and reports only
MMD² and expression-space mean errors.

Example::

    python -m sciplex_interp.cellflow_ablation \
      --reference z70_lr3e4_mv10_t \
      --variants z70_lr3e4_t z70_lr3e4_tonly meanv_l10_t \
      --labels z70_lr3e4_mv10_t="full arm" z70_lr3e4_t="without grouped velocity matching"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_analysis.analyze import paired_stats

from .core import DEFAULT_ROOT, read_jsonl


REPO = Path(__file__).resolve().parents[1]
METRICS = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program"]
LABELS = {
    "mmd2": "MMD²",
    "rmse_mean_deg50": "RMSE DEG50",
    "rmse_mean_all": "RMSE all genes",
    "rmse_mean_program": "RMSE program",
}


def rows(root: Path, variant: str, split: str, offset: int) -> pd.DataFrame:
    paths = sorted((root / "cellflow" / variant).glob(
        f"seed_*/{split}_sample_s{offset}/per_condition.jsonl"
    ))
    if not paths:
        raise SystemExit(f"no {split} offset-{offset} evaluation for {variant}")
    records = [
        {**record, "seed": int(path.parent.parent.name.split("_")[1])}
        for path in paths
        for record in read_jsonl(path)
    ]
    return pd.DataFrame(records)[["seed", "condition", "selected_step", *METRICS]]


def matrix(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.pivot_table(index="condition", columns="seed", values=metric, aggfunc="mean")


def summary_cell(frame: pd.DataFrame, metric: str) -> str:
    per_seed = frame.groupby("seed")[metric].mean()
    mean = per_seed.mean()
    fmt = "{:.4f}" if metric == "mmd2" else "{:.3f}"
    return f"{fmt.format(mean)} ± {fmt.format(per_seed.std(ddof=1))}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="the complete/strongest arm")
    parser.add_argument("--variants", nargs="+", required=True, help="ablation variants")
    parser.add_argument("--labels", nargs="*", default=[], help="variant=display label pairs")
    parser.add_argument("--extra-pairs", nargs="*", default=[],
                        help="additional candidate=reference paired contrasts")
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=str(REPO / "docs" / "interp"))
    parser.add_argument("--name", default="cellflow_ablation", help="output filename stem")
    parser.add_argument("--title", default="CellFlow strongest-arm ablations")
    args = parser.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [args.reference, *[v for v in args.variants if v != args.reference]]
    names = dict(item.split("=", 1) for item in args.labels)
    label = lambda variant: names.get(variant, variant)
    frames = {variant: rows(root, variant, args.split, args.offset) for variant in variants}
    n_conditions = frames[args.reference].condition.nunique()

    md = [
        f"# {args.title}",
        "",
        f"Validation split ({n_conditions} held-out-dose conditions), 512 generated cells per condition, "
        "3 training seeds. Each seed uses its minimum-validation-MMD² checkpoint. Values are mean ± "
        "sample SD over seeds. Response-score and weak-responder quantities are excluded.",
        "",
        "| Arm | Selected steps | " + " | ".join(LABELS[m] for m in METRICS) + " |",
        "|---" * (len(METRICS) + 2) + "|",
    ]
    summary = []
    for variant, frame in frames.items():
        steps = "/".join(str(int(step)) for step in frame.groupby("seed").selected_step.first())
        md.append(
            f"| {label(variant)} | {steps} | "
            + " | ".join(summary_cell(frame, m) for m in METRICS)
            + " |"
        )
        per_seed = frame.groupby("seed")[METRICS].mean()
        summary.append({
            "variant": variant,
            "label": label(variant),
            "split": args.split,
            "n_conditions": frame.condition.nunique(),
            "n_seeds": frame.seed.nunique(),
            "selected_steps": steps,
            **{f"{m}_mean": per_seed[m].mean() for m in METRICS},
            **{f"{m}_seed_sd": per_seed[m].std(ddof=1) for m in METRICS},
        })

    md += [
        "",
        f"Paired differences versus **{label(args.reference)}**. Δ = arm − reference, so positive "
        "values are worse; intervals are condition-bootstrap 95% CIs.",
        "",
        "| Arm | " + " | ".join(LABELS[m] for m in METRICS) + " |",
        "|---" * (len(METRICS) + 1) + "|",
    ]
    paired = []
    rng = np.random.default_rng(0)
    ref = frames[args.reference]
    for variant in variants[1:]:
        cells = []
        for metric in METRICS:
            # paired_stats reports first - second. Passing the candidate first makes positive = worse.
            stats = paired_stats(matrix(frames[variant], metric), matrix(ref, metric), rng, args.n_boot)
            paired.append({
                "reference": args.reference,
                "candidate": variant,
                "metric": metric,
                **stats,
            })
            fmt = (lambda value: f"{value:+.4f}") if metric == "mmd2" else (lambda value: f"{value:+.3f}")
            cells.append(
                f"{fmt(stats['delta_mean'])} [{fmt(stats['ci95_low'])}, "
                f"{fmt(stats['ci95_high'])}]"
            )
        md.append(f"| {label(variant)} | " + " | ".join(cells) + " |")

    if args.extra_pairs:
        md += [
            "",
            "Additional paired contrasts. Δ = first arm − second arm; positive values are worse.",
            "",
            "| Contrast | " + " | ".join(LABELS[m] for m in METRICS) + " |",
            "|---" * (len(METRICS) + 1) + "|",
        ]
        for item in args.extra_pairs:
            candidate, reference = item.split("=", 1)
            if candidate not in frames or reference not in frames:
                raise SystemExit(f"extra pair {item} is not among the loaded variants")
            cells = []
            for metric in METRICS:
                stats = paired_stats(matrix(frames[candidate], metric), matrix(frames[reference], metric),
                                     rng, args.n_boot)
                paired.append({"reference": reference, "candidate": candidate, "metric": metric, **stats})
                fmt = ((lambda value: f"{value:+.4f}") if metric == "mmd2"
                       else (lambda value: f"{value:+.3f}"))
                cells.append(f"{fmt(stats['delta_mean'])} [{fmt(stats['ci95_low'])}, "
                             f"{fmt(stats['ci95_high'])}]")
            md.append(f"| {label(candidate)} − {label(reference)} | " + " | ".join(cells) + " |")

    pd.DataFrame(summary).to_csv(out / f"{args.name}_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / f"{args.name}_paired.csv", index=False)
    report = "\n".join(md) + "\n"
    (out / f"{args.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
