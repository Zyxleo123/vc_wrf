"""Failure-case diagnosis on the standard benchmark, with the unseen drugs (Table 2) in view.

Three questions, all answered from artefacts already on disk (no retraining):

  1. shift   Does a method move the cells at all, and in the right direction? Per condition the
             mean shift of the generated cells from their cell line's vehicle mean is compared with
             the measured one: gain = ||pred shift|| / ||true shift||, cos = their cosine. A method
             that copies the control has gain 0; a method with the right programme but too timid has
             cos near 1 and gain < 1.
  2. rescale What would be gained by fixing the magnitude alone: every generated cell is moved by
             (alpha - 1) * pred_shift (so the mean shift becomes alpha * pred_shift, the spread is
             untouched) and rescored with DOLORIS's metrics, for a grid of global alphas and for the
             per-condition oracle alpha. The remaining error is the direction error.
  3. chem    Is the molecular representation able to find the right neighbour? Per drug response
             signatures (top-dose mean shift, averaged over cell lines) are compared with Uni-Mol
             distances: rank correlation over training pairs, and, for each unseen drug, the
             response cosine of its chemical nearest neighbour against the best training drug.

Also reports the condition response embedding ("teacher") on unseen drugs: its own gain and cosine,
and the share of each true shift that lies in its 20-direction basis (the best its parameterisation
could do).

usage: python -m sciplex_standard.diagnose --root results/standard --splits ood test
Writes <root>/diagnose/{shifts.jsonl,rescale.jsonl,teacher.jsonl,chem.json,summary.md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sciplex_doloris.metrics import doloris_metrics

from .md import md_table
from .data import DEFAULT_FILE, REPO, StdData
from .evaluate import read_cells

ROOT = REPO / "results" / "standard"
METHODS = ("ours", "doloris_bestval", "vehicle")
ALPHAS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
TOP_DOSE = 10000.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def seeds_of(root: Path, method: str) -> list[int]:
    d = root / "preds" / method
    return sorted(int(p.name.split("_")[1]) for p in d.glob("seed_*")) if d.exists() else [0]


def cells_of(data: StdData, root: Path, method: str, split: str, seed: int, keys: list[str]) -> dict:
    """Generated cells per condition; 'vehicle' answers with that cell line's vehicle cells."""
    if method == "vehicle":
        rng = np.random.default_rng(seed)
        out = {}
        for k in keys:
            pool = data.ctrl_idx(data.cond.cell_type[k])
            out[k] = data.X[np.sort(rng.choice(pool, 100, replace=False))]
        return out
    return read_cells(root / "preds" / method / f"seed_{seed}" / f"{split}.h5ad", data.genes)


def shift_rows(data: StdData, root: Path, splits: list[str], methods: list[str]) -> pd.DataFrame:
    m0 = {c: data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) for c in data.cell_types}
    rows = []
    for split in splits:
        keys = data.keys(split)
        true = {k: data.X[data.idx(k)].mean(0, dtype=np.float64) - m0[data.cond.cell_type[k]] for k in keys}
        spread = {k: float(np.sqrt(((data.X[data.idx(k)] - data.X[data.idx(k)].mean(0)) ** 2).sum(1).mean()))
                  for k in keys}
        for method in methods:
            for seed in seeds_of(root, method):
                cells = cells_of(data, root, method, split, seed, keys)
                for k in keys:
                    t, p = true[k], cells[k].mean(0, dtype=np.float64) - m0[data.cond.cell_type[k]]
                    X = np.asarray(cells[k], dtype=np.float64)
                    sp = float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))
                    de = data.de[k][:20]
                    rows.append({"split": split, "method": method, "seed": seed, "condition": k,
                                 "drug": data.cond.drug[k], "cell_type": data.cond.cell_type[k],
                                 "dose_nM": float(data.cond.dose_nM[k]),
                                 "true_norm": float(np.linalg.norm(t)), "pred_norm": float(np.linalg.norm(p)),
                                 "gain": float(np.linalg.norm(p) / max(np.linalg.norm(t), 1e-12)), "cos": cosine(t, p),
                                 "true_norm_de20": float(np.linalg.norm(t[de])),
                                 "gain_de20": float(np.linalg.norm(p[de]) / max(np.linalg.norm(t[de]), 1e-12)),
                                 "cos_de20": cosine(t[de], p[de]),
                                 "spread": sp, "true_spread": spread[k], "spread_ratio": sp / max(spread[k], 1e-12)})
                print(f"shift {split} {method} seed {seed}: {len(keys)} conditions", flush=True)
    return pd.DataFrame(rows)


def rescale_rows(data: StdData, root: Path, splits: list[str], method: str) -> pd.DataFrame:
    """Rescore with the mean shift multiplied by alpha (cells moved rigidly), plus the oracle alpha."""
    m0 = {c: data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) for c in data.cell_types}
    rows = []
    for split in splits:
        keys = data.keys(split)
        for seed in seeds_of(root, method):
            cells = cells_of(data, root, method, split, seed, keys)
            for k in keys:
                real = data.X[data.idx(k)]
                base = m0[data.cond.cell_type[k]]
                t = real.mean(0, dtype=np.float64) - base
                X = np.asarray(cells[k], dtype=np.float64)
                p = X.mean(0) - base
                n2 = float(p @ p)
                a_star = float(t @ p / n2) if n2 > 0 else 0.0
                for alpha in (*ALPHAS, a_star):
                    Y = np.maximum(X + (alpha - 1.0) * p, 0.0)
                    rec = {"split": split, "method": method, "seed": seed, "condition": k,
                           "drug": data.cond.drug[k], "dose_nM": float(data.cond.dose_nM[k]),
                           "alpha": float(alpha), "oracle": alpha == a_star, "alpha_star": a_star}
                    rec.update(doloris_metrics(real, Y.astype(np.float32), data.de[k]))
                    rows.append(rec)
            print(f"rescale {split} {method} seed {seed}: {len(keys)} conditions", flush=True)
    return pd.DataFrame(rows)


def teacher_rows(data: StdData, root: Path) -> pd.DataFrame:
    t = np.load(root / "teacher" / "std_autodec.npz")
    coef = dict(zip([str(x) for x in t["keys"]], t["coef"].astype(np.float64)))
    V = t["basis"].astype(np.float64)
    ct = [str(c) for c in t["cell_types"]]
    m0 = {c: data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) for c in data.cell_types}
    rows = []
    for split in ("train", "val", "test", "ood"):
        for k in data.keys(split, min_cells=1):
            true = data.X[data.idx(k)].mean(0, dtype=np.float64) - m0[data.cond.cell_type[k]]
            pred = coef[k] @ V
            proj = (true @ V.T) @ V  # best any 20-coefficient teacher could do
            rows.append({"split": split, "condition": k, "drug": data.cond.drug[k],
                         "cell_type": data.cond.cell_type[k], "dose_nM": float(data.cond.dose_nM[k]),
                         "true_norm": float(np.linalg.norm(true)), "gain": float(np.linalg.norm(pred) / max(np.linalg.norm(true), 1e-12)),
                         "cos": cosine(true, pred), "basis_share": float(np.linalg.norm(proj) / max(np.linalg.norm(true), 1e-12)),
                         "rmse": float(np.sqrt(np.mean((pred - true) ** 2))), "rmse_zero": float(np.sqrt(np.mean(true ** 2)))})
        print(f"teacher {split}: done", flush=True)
    assert ct == list(data.cell_types)
    return pd.DataFrame(rows)


def chem_analysis(data: StdData, mol_path: str) -> dict:
    """Per-drug response signatures against Uni-Mol distances: can chemistry retrieve the neighbour?"""
    from scipy.stats import spearmanr

    m0 = {c: data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) for c in data.cell_types}
    top = data.cond[np.isclose(data.cond.dose_nM, TOP_DOSE)]
    sig, split_of = {}, {}
    for drug, g in top.groupby("drug"):
        parts = [data.X[data.idx(k)].mean(0, dtype=np.float64) - m0[g.cell_type[k]] for k in g.index]
        sig[drug] = np.mean(parts, 0)
        split_of[drug] = "ood" if (g.split == "ood").all() else "train"
    drugs = sorted(sig)
    S = np.stack([sig[d] for d in drugs])
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    R = S @ S.T  # response cosine

    z = np.load(mol_path)
    names = [str(x) for x in z["drugs"]]
    E = z["emb"][[names.index(d) for d in drugs]].astype(np.float64)
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    u, s, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    F = u[:, :32] * s[:32]
    F = F / F[:, 0].std()
    Dc = np.sqrt(np.maximum(((F[:, None] - F[None]) ** 2).sum(-1), 0))  # the teacher's own input space

    tr = np.array([split_of[d] == "train" for d in drugs])
    iu = np.triu_indices(tr.sum(), 1)
    rho_tr = spearmanr(Dc[np.ix_(tr, tr)][iu], R[np.ix_(tr, tr)][iu])
    out = {"n_drugs": len(drugs), "n_train": int(tr.sum()), "n_ood": int((~tr).sum()),
           "spearman_chem_distance_vs_response_cosine_train_pairs": {"rho": float(rho_tr.statistic), "p": float(rho_tr.pvalue)},
           "response_cosine_train_pairs": {"mean": float(R[np.ix_(tr, tr)][iu].mean()),
                                           "q90": float(np.quantile(R[np.ix_(tr, tr)][iu], 0.9))},
           "ood": {}}
    idx_tr = np.flatnonzero(tr)
    for i, d in enumerate(drugs):
        if split_of[d] == "train":
            continue
        order = idx_tr[np.argsort(Dc[i, idx_tr])]
        best = idx_tr[np.argmax(R[i, idx_tr])]
        out["ood"][d] = {"chem_nn": drugs[order[0]], "chem_nn_response_cos": float(R[i, order[0]]),
                         "chem_nn5_best_response_cos": float(max(R[i, j] for j in order[:5])),
                         "best_response_drug": drugs[best], "best_response_cos": float(R[i, best]),
                         "median_response_cos": float(np.median(R[i, idx_tr])),
                         "effect_norm": float(np.linalg.norm(sig[d]))}
    vals = out["ood"].values()
    out["ood_mean"] = {k: float(np.mean([v[k] for v in vals])) for k in
                       ("chem_nn_response_cos", "chem_nn5_best_response_cos", "best_response_cos", "median_response_cos")}
    return out


def summarise(sh: pd.DataFrame, rs: pd.DataFrame, te: pd.DataFrame, chem: dict) -> str:
    L = ["# Failure-case diagnosis (standard benchmark)", "",
         "## 1. Does the model move the cells? (mean shift from the cell line's vehicle mean)", "",
         "gain = ||pred shift|| / ||true shift||, cos = cosine of the two; mean over conditions, then seeds.", ""]
    g = sh.groupby(["split", "method"]).agg(gain=("gain", "mean"), cos=("cos", "mean"),
                                            gain_de20=("gain_de20", "mean"), cos_de20=("cos_de20", "mean"),
                                            spread_ratio=("spread_ratio", "mean"),
                                            true_norm=("true_norm", "mean")).round(3)
    L += [md_table(g), ""]
    L += ["### unseen drugs, by drug (Ours)", ""]
    o = sh[(sh.split == "ood") & (sh.method == "ours")].groupby("drug").agg(
        true_norm=("true_norm", "mean"), gain=("gain", "mean"), cos=("cos", "mean"), cos_de20=("cos_de20", "mean")).round(3)
    L += [md_table(o.sort_values("true_norm", ascending=False)), ""]
    L += ["### unseen drugs, by dose (Ours)", ""]
    d = sh[(sh.split == "ood") & (sh.method == "ours")].groupby("dose_nM").agg(
        true_norm=("true_norm", "mean"), gain=("gain", "mean"), cos=("cos", "mean")).round(3)
    L += [md_table(d), ""]
    L += ["## 2. What magnitude alone would buy (rescaled mean shift, DOLORIS metrics)", ""]
    for split in sorted(rs.split.unique()):
        sub = rs[rs.split == split]
        tab = sub[~sub.oracle].groupby("alpha")[["rmse_all", "edist_all", "emd_all", "rmse_de20", "emd_de20"]].mean()
        orc = sub[sub.oracle][["rmse_all", "edist_all", "emd_all", "rmse_de20", "emd_de20"]].mean().rename("oracle alpha")
        tab = pd.concat([tab, orc.to_frame().T]).round(4)
        L += [f"### {split} (alpha_star mean {sub[sub.oracle].alpha_star.mean():.2f})", "", md_table(tab), ""]
    L += ["## 3. Condition response embedding (teacher) by split", "",
          md_table(te.groupby("split").agg(gain=("gain", "mean"), cos=("cos", "mean"), basis_share=("basis_share", "mean"),
                                           rmse=("rmse", "mean"), rmse_zero=("rmse_zero", "mean")).round(3)), "",
          "### unseen drugs, teacher by drug", "",
          md_table(te[te.split == "ood"].groupby("drug").agg(true_norm=("true_norm", "mean"), gain=("gain", "mean"),
                                                             cos=("cos", "mean"), basis_share=("basis_share", "mean")
                                                             ).round(3).sort_values("true_norm", ascending=False)), "",
          "## 4. Can the molecular representation retrieve the right neighbour?", "",
          "```", json.dumps({k: v for k, v in chem.items() if k != "ood"}, indent=1), "```", "",
          md_table(pd.DataFrame(chem["ood"]).T.round(3)), ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--molecules", default=str(ROOT / "unimol_embed.npz"))
    p.add_argument("--splits", nargs="+", default=["ood", "test"])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--render", action="store_true", help="rebuild summary.md from the saved rows, computing nothing")
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.root) / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    if a.render:
        sh, rs, te = (pd.read_json(out / f"{n}.jsonl", lines=True) for n in ("shifts", "rescale", "teacher"))
        chem = json.loads((out / "chem.json").read_text())
        (out / "summary.md").write_text(summarise(sh, rs, te, chem) + "\n")
        print(f"wrote {out}/summary.md")
        return
    data = StdData(a.file)
    print(f"obs columns: {list(data.obs.columns)}", flush=True)

    sh = shift_rows(data, root, a.splits, a.methods)
    sh.to_json(out / "shifts.jsonl", orient="records", lines=True)
    rs = rescale_rows(data, root, a.splits, "ours")
    rs.to_json(out / "rescale.jsonl", orient="records", lines=True)
    te = teacher_rows(data, root)
    te.to_json(out / "teacher.jsonl", orient="records", lines=True)
    chem = chem_analysis(data, a.molecules)
    (out / "chem.json").write_text(json.dumps(chem, indent=1) + "\n")
    (out / "summary.md").write_text(summarise(sh, rs, te, chem) + "\n")
    print(f"wrote {out}/summary.md", flush=True)


if __name__ == "__main__":
    main()
