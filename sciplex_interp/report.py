"""Tables, paired comparisons and figures for the dose-interpolation study, from saved rows only.

Reads <root>/stats/rows.jsonl (+ rows.test*.jsonl), <root>/populations/*.jsonl,
<root>/chemcpa_eval/**, <root>/diagnose/{floors,structure}.json[l] and the frozen model rows in
results/analysis/tables/per_condition.csv, and writes <out>/ (default docs/interp/):

  mean_predictors.{md,csv}    predicted-mean errors per task set, with noise floors
  populations.{md,csv}        population metrics of every generator (mean over conditions, ± SD over
                              sampling seeds), with the models' frozen rows for val and test
  chemcpa.{md,csv}            chemCPA variants: in-sample fit and held-out doses
  paired.csv, paired.md       paired condition bootstraps of chosen methods against references
  *.png                       figures (redrawn from the tables above)

usage: python -m sciplex_interp.report --root results/interp --out docs/interp --chosen <generator> ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DEFAULT_ROOT, HIGHER_BETTER, METRICS, read_jsonl

REPO = Path(__file__).resolve().parents[1]
MODEL_TABLE = REPO / "results" / "analysis" / "tables" / "per_condition.csv"
FROZEN_ONLY = ["mmd2", "rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "w1_r"]
MEAN_METRICS = ["rmse_mean_deg50", "rmse_mean_all", "rmse_mean_program", "pearson_delta_deg50", "pearson_delta_all"]
LABELS = {"mmd2": "MMD²", "rmse_mean_deg50": "RMSE DEG50", "rmse_mean_all": "RMSE all", "rmse_mean_program":
          "RMSE program", "w1_r": "W1(r)", "edist": "E-dist", "pearson_delta_deg50": "Pearson Δ DEG50",
          "pearson_delta_all": "Pearson Δ all"}
MODELS = {("cellflow", "cellflow"): "CellFlow", ("cellflow", "cellflow_mmd"): "CellFlow + MMD loss",
          ("chemcpa", "gauss_v1"): "chemCPA-Gauss", ("chemcpa", "nb_v1"): "chemCPA-NB",
          ("reference", "replicate"): "Replicate well (noise floor)", ("reference", "vehicle"): "Vehicle cells",
          ("reference", "adjacent_dose_mixture"): "Adjacent-dose mixture (CellFlow's published draw)"}


def fmt(metric: str, v: float) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{v:.4f}" if metric in ("mmd2",) else f"{v:.3f}"


def table(df: pd.DataFrame, names: list[str], metrics: list[str], label=lambda n: n) -> str:
    """Markdown: rows = names, cells = mean over conditions (± SD over seeds when > 1 seed)."""
    lines = ["| | " + " | ".join(LABELS.get(m, m) for m in metrics) + " |",
             "|---" * (len(metrics) + 1) + "|"]
    for n in names:
        part = df[df.name == n]
        if part.empty:
            continue
        cells = []
        for m in metrics:
            if m not in part or part[m].isna().all():
                cells.append("–")
                continue
            per_seed = part.groupby("seed")[m].mean()
            s = fmt(m, part.groupby("condition")[m].mean().mean())
            if len(per_seed) > 1:
                s += f" ± {fmt(m, per_seed.std(ddof=1))}"
            cells.append(s)
        lines.append(f"| {label(n)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired(ref: pd.DataFrame, new: pd.DataFrame, metric: str, rng, n_boot: int) -> dict:
    """Δ = ref − new on the condition means (> 0: new better; Pearson is sign-flipped), with a
    condition bootstrap 95% CI, and the share of conditions where new is better."""
    a = ref.groupby("condition")[metric].mean()
    b = new.groupby("condition")[metric].mean()
    conds = a.index.intersection(b.index)
    d = (a.loc[conds] - b.loc[conds]).to_numpy()
    if metric in HIGHER_BETTER:
        d = -d
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"metric": metric, "n_conditions": len(d), "mean_ref": float(a.loc[conds].mean()),
            "mean_new": float(b.loc[conds].mean()), "delta": float(d.mean()),
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "n_better": int((d > 0).sum())}


def load(root: Path) -> dict:
    out = {}
    stats = [r for p in sorted((root / "stats").glob("rows*.jsonl")) if ".shard" not in p.name for r in read_jsonl(p)]
    out["stats"] = pd.DataFrame(stats).drop_duplicates(["task_set", "condition", "predictor"], keep="last")
    pops = [r for p in sorted((root / "populations").glob("*.jsonl")) for r in read_jsonl(p)]
    pop = pd.DataFrame(pops)
    pop["name"] = pop.generator
    out["pop"] = pop
    chem = [r for p in sorted((root / "chemcpa_eval").glob("*/*.jsonl")) for r in read_jsonl(p)]
    out["chem"] = pd.DataFrame(chem) if chem else pd.DataFrame()
    out["floors"] = pd.DataFrame(read_jsonl(root / "diagnose" / "floors.jsonl"))
    models = pd.read_csv(MODEL_TABLE)
    keep = [models[(models.model == m) & (models.method == k)].assign(name=f"model:{m}/{k}")
            for (m, k) in MODELS]
    mt = pd.concat(keep, ignore_index=True)
    mt["task_set"] = mt.split
    mt["seed"] = mt.seed.fillna(0)
    out["models"] = mt
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=str(REPO / "docs" / "interp"))
    p.add_argument("--predictors", nargs="+", required=True, help="mean predictors to tabulate, in order")
    p.add_argument("--generators", nargs="+", required=True, help="population generators to tabulate, in order")
    p.add_argument("--chosen", nargs="+", required=True, help="generators compared against every reference")
    p.add_argument("--chemcpa", nargs="*", default=[], help="chemCPA variants to tabulate")
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(root)
    rng = np.random.default_rng(0)
    md = []

    # ---- mean predictors -------------------------------------------------------------------
    st = D["stats"].rename(columns={"predictor": "name"})
    st["seed"] = 0
    fl = D["floors"]
    md.append("## Predicted means (one condition's mean from the other conditions)\n")
    for ts in ("inner", "val", "test"):
        part = st[st.task_set == ts]
        if part.empty:
            continue
        md.append(f"### {ts} ({part.condition.nunique()} conditions)\n")
        md.append(table(part, a.predictors, MEAN_METRICS))
        f = fl[fl.task_set == ts]
        if not f.empty:
            md.append(f"| noise floor of the measured mean | {f.floor_rmse_mean_deg50.mean():.3f} "
                      f"({f.floor512_rmse_mean_deg50.mean():.3f} with a 512-cell prediction) | "
                      f"{f.floor_rmse_mean_all.mean():.3f} | {f.floor_rmse_mean_program.mean():.3f} | | |")
        md.append("")
    st[st.name.isin(a.predictors)].to_csv(out / "mean_predictors.csv", index=False)

    # ---- populations -----------------------------------------------------------------------
    pop, models = D["pop"], D["models"]
    md.append("## Populations\n")
    rows_csv = []
    for ts in ("inner", "val", "test"):
        part = pop[pop.task_set == ts]
        if part.empty:
            continue
        mparts = models[models.task_set == ts]
        both = pd.concat([part, mparts], ignore_index=True)
        names = [g for g in a.generators if g in set(part.name)] + [n for n in
                 (f"model:{m}/{k}" for (m, k) in MODELS) if n in set(mparts.name)]
        label = lambda n: MODELS.get(tuple(n.removeprefix("model:").split("/")), n) if n.startswith("model:") else n
        md.append(f"### {ts} ({part.condition.nunique()} conditions; generators: {part.seed.nunique()} sampling "
                  f"seeds, models: 3 training seeds)\n")
        md.append(table(both, names, METRICS, label))
        md.append("")
        for n in names:
            sub = both[both.name == n]
            rows_csv.append({"task_set": ts, "name": n, **{m: sub.groupby("condition")[m].mean().mean()
                                                           for m in METRICS if m in sub}})
    pd.DataFrame(rows_csv).to_csv(out / "populations.csv", index=False)

    # ---- paired comparisons -----------------------------------------------------------------
    pairs = []
    for ts in ("val", "test"):
        part = pop[pop.task_set == ts]
        if part.empty:
            continue
        refs = ["adjacent", "adjacent_w0.3"] + [n for n in (f"model:{m}/{k}" for (m, k) in MODELS)]
        for new in a.chosen:
            if new not in set(part.name):
                continue
            for ref in refs:
                R = part[part.name == ref] if not ref.startswith("model:") else models[(models.name == ref)
                                                                                       & (models.task_set == ts)]
                if R.empty:
                    continue
                for m in METRICS:
                    if m not in R or R[m].isna().all():
                        continue
                    pairs.append({"task_set": ts, "ref": ref, "new": new,
                                  **paired(R, part[part.name == new], m, rng, a.n_boot)})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "paired.csv", index=False)
    if not pairs.empty:
        md.append("## Paired condition bootstrap (Δ = ref − new on condition means; > 0: new better; "
                  "Pearson sign-flipped)\n")
        for (ts, new), g in pairs.groupby(["task_set", "new"], sort=False):
            md.append(f"### {ts}: {new}\n")
            md.append("| ref | " + " | ".join(LABELS[m] for m in METRICS) + " |")
            md.append("|---" * (len(METRICS) + 1) + "|")
            for ref, gg in g.groupby("ref", sort=False):
                cells = []
                for m in METRICS:
                    r = gg[gg.metric == m]
                    if r.empty:
                        cells.append("–")
                        continue
                    r = r.iloc[0]
                    f = (lambda v: f"{v:+.4f}") if m == "mmd2" else (lambda v: f"{v:+.3f}")
                    cells.append(f"{f(r.delta)} [{f(r.ci_low)}, {f(r.ci_high)}] {r.n_better}/{r.n_conditions}")
                lab = MODELS.get(tuple(ref.removeprefix("model:").split("/")), ref) if ref.startswith("model:") else ref
                md.append(f"| {lab} | " + " | ".join(cells) + " |")
            md.append("")

    # ---- chemCPA ----------------------------------------------------------------------------
    chem = D["chem"]
    if not chem.empty and a.chemcpa:
        md.append("## chemCPA variants (population metrics; 3 training seeds)\n")
        chem = chem.assign(name=chem.variant + " · " + chem.kind)
        for ts in ("val", "test"):
            part = chem[(chem.task_set == ts)]
            if part.empty:
                continue
            for kind in ("train_fit", "direct", "oracle_shift"):
                names = [f"{v} · {kind}" for v in a.chemcpa if f"{v} · {kind}" in set(part.name)]
                if names:
                    md.append(f"### {ts}: {kind}\n")
                    md.append(table(part, names, METRICS))
                    md.append("")
        chem.to_csv(out / "chemcpa.csv", index=False)

    (out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
