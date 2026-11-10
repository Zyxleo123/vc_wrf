"""Score generated cells on the standard benchmark with DOLORIS's protocol, and build its tables.

Per held-out condition (test: DOLORIS Table 1, conditions with >= 10 cells as in test.py; ood:
Table 2, every condition), generated cells against all measured cells of the condition:
DOLORIS's nine metrics (RMSE of the mean, E-distance, per-gene EMD on all genes, DE20 and DE40,
with the file's `lincs_DEGs` exactly as DOLORIS reads them) and MMD² in the spec PCA space
(fit on control + train cells). References built from measured cells: vehicle cells of the same
cell line (100, as many as a method generates) and one replicate well against the other.

Predictions: <root>/preds/<method>/seed_<s>/<split>.h5ad, obs["condition"] = cov_drug_dose_name,
var_names = genes (any order).

usage: python -m sciplex_standard.evaluate score --split test
       python -m sciplex_standard.evaluate report
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_doloris.metrics import COLUMNS, GENE_SETS, KINDS, doloris_metrics
from sciplex_eval.metrics import mmd2_rbf

from .data import DEFAULT_FILE, REPO, StdData, load_spec, project

ROOT = REPO / "results" / "standard"
N_REF = 100
PAPER = REPO / "configs" / "doloris_paper_tables.json"
LABELS = {"replicate": "Replicate well (noise floor)", "vehicle": "Vehicle cells (no drug effect)"}


def score_condition(data: StdData, spec: dict, key: str, pred: np.ndarray, real: np.ndarray | None = None) -> dict:
    real = data.X[data.idx(key)] if real is None else real
    rec = {"n_real": int(len(real)), "n_pred": int(len(pred)), "negative_fraction": float((pred < 0).mean())}
    rec.update(doloris_metrics(real, pred, data.de[key]))
    rng = np.random.default_rng(0)
    n_max = int(spec["mmd_n_max"])
    sub = lambda X: X if len(X) <= n_max else X[np.sort(rng.choice(len(X), n_max, replace=False))]
    rec["mmd2"] = mmd2_rbf(project(spec, sub(real)), project(spec, sub(pred)), spec["mmd_sigmas"])
    return rec


def read_cells(path: Path, genes: list[str]) -> dict[str, np.ndarray]:
    import anndata as ad
    a = ad.read_h5ad(path)
    pos = {g: i for i, g in enumerate(map(str, a.var_names))}
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.asarray(X[:, [pos[g] for g in genes]], dtype=np.float32)
    labels = a.obs["condition"].astype(str).to_numpy()
    return {c: X[labels == c] for c in pd.unique(labels)}


def score(split: str, root: Path, path: str) -> None:
    data, spec = StdData(path), load_spec()
    keys = data.keys(split)
    rows = []
    for pred_path in sorted((root / "preds").glob(f"*/seed_*/{split}.h5ad")):
        method, seed = pred_path.parent.parent.name, int(pred_path.parent.name.split("_")[1])
        cells = read_cells(pred_path, data.genes)
        missing = [k for k in keys if k not in cells]
        if missing:
            raise SystemExit(f"{pred_path}: no cells for {len(missing)} {split} conditions, e.g. {missing[:3]}")
        rows += [{"method": method, "seed": seed, "condition": k, **score_condition(data, spec, k, cells[k])} for k in keys]
        print(f"{method} seed {seed}: {len(keys)} conditions", flush=True)
    for k in keys:
        c = data.cond.loc[k]
        rng = np.random.default_rng([zlib.crc32(k.encode()), 1])
        ctrl = data.ctrl_idx(c.cell_type)
        rows.append({"method": "vehicle", "seed": 0, "condition": k,
                     **score_condition(data, spec, k, data.X[np.sort(rng.choice(ctrl, N_REF, replace=False))])})
        idx = data.idx(k)
        reps = [idx[data.obs.replicate.to_numpy()[idx] == r] for r in ("rep1", "rep2")]
        if min(len(r) for r in reps) >= 2:  # E-distance needs two cells per side
            one = score_condition(data, spec, k, data.X[reps[1]], real=data.X[reps[0]])
            two = score_condition(data, spec, k, data.X[reps[0]], real=data.X[reps[1]])
            rows.append({"method": "replicate", "seed": 0, "condition": k, **{m: (one[m] + two[m]) / 2 for m in one}})
    out = root / "scores" / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    tmp.rename(out)
    df = pd.DataFrame(rows)
    print(df.groupby(["method", "condition"])[["mmd2", *COLUMNS]].mean().groupby("method").mean()
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}: {len(rows)} rows, {len(keys)} conditions")


def table(df: pd.DataFrame, methods: list[str], labels: dict, paper: dict | None) -> list[str]:
    heads = [f"{g.upper() if g != 'all' else 'All'} {dict(rmse='RMSE', edist='E-dist', emd='EMD')[k]}" for g in GENE_SETS for k in KINDS]
    out = ["| | " + " | ".join(heads) + " | MMD² |", "|---" * (len(heads) + 2) + "|"]
    for m in methods:
        per = df[df.method == m].groupby("condition")[COLUMNS + ["mmd2"]].mean()  # seeds averaged per condition
        seeds = df[df.method == m].seed.nunique()
        cells = [f"{per[c].mean():.4f} ± {per[c].std(ddof=1):.4f}" for c in COLUMNS]
        out.append(f"| {labels.get(m, m)}{f' ({seeds} seeds)' if seeds > 1 else ''} | " + " | ".join(cells)
                   + f" | {per['mmd2'].mean():.4f} |")
    for name, vals in (paper or {}).items():
        out.append(f"| *paper: {name}* | " + " | ".join(f"*{v}*" for v in vals) + " | |")
    return out


def report(root: Path) -> None:
    paper = json.loads(PAPER.read_text())
    md = ["# Standard sci-Plex3 benchmark (chemCPA data, DOLORIS protocol)", "",
          "Mean ± SD over conditions, each condition averaged over seeds; lower is better. 100 generated cells per "
          "condition. DE20/DE40 from the file's `lincs_DEGs`. Paper rows are copied from DOLORIS (ICLR 2026) Tables 1 and 2 "
          "for reference: their random 30% draw is not published, so they are not on the same conditions.", ""]
    for split, title, key in (("test", "Table 1: unseen drug - cell line - dosage combinations", "table1"),
                              ("ood", "Table 2: OOD drugs (9 unseen drugs, all doses and cell lines)", "table2")):
        path = root / "scores" / f"{split}.jsonl"
        if not path.exists():
            continue
        df = pd.DataFrame([json.loads(l) for l in path.open()])
        gen = sorted(m for m in df.method.unique() if m not in LABELS)
        methods = gen + [m for m in LABELS if m in set(df.method)]
        md += [f"## {title} ({df.condition.nunique()} conditions)", "",
               *table(df, methods, LABELS, paper[key]["rows"]), "", paper[key]["note"], ""]
        if "ours" in gen:
            from sciplex_analysis.analyze import paired_stats
            rng = np.random.default_rng(0)
            m_ours = lambda c: df[df.method == "ours"].pivot_table(index="condition", columns="seed", values=c)
            md += ["Paired against Ours: Δ = other − Ours (> 0 favours Ours), [95% bootstrap CI over conditions], "
                   "conditions where Ours is lower.", "", "| metric | " + " | ".join(o for o in gen if o != "ours") + " |",
                   "|---" * len(gen) + "|"]
            for c in COLUMNS + ["mmd2"]:
                cells = []
                for o in gen:
                    if o == "ours":
                        continue
                    s = paired_stats(df[df.method == o].pivot_table(index="condition", columns="seed", values=c),
                                     m_ours(c), rng, 5000)
                    cells.append(f"{s['delta_mean']:+.4f} [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] {s['n_win']}/{s['n_conditions']}")
                md.append(f"| {c} | " + " | ".join(cells) + " |")
            md.append("")
    out = root / "report.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["score", "report"])
    p.add_argument("--split", choices=["val", "test", "ood"], default="test")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--file", default=DEFAULT_FILE)
    a = p.parse_args(argv)
    if a.stage == "score":
        score(a.split, Path(a.root), a.file)
    else:
        report(Path(a.root))


if __name__ == "__main__":
    main()
