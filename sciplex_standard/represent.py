"""Leave-one-drug-out study of the structure -> response map, off-line from the flow model.

The condition response embedding of Ours ("teacher") maps (drug structure, cell line, dose) to the
coordinates of the condition's mean shift. On unseen drugs that map is what has to generalise, so it
can be studied on its own, without training a generative model: hold out one drug at a time, fit the
map on the remaining drugs' `train`/`val` conditions, and score the predicted mean shift of the
held-out drug's conditions against the measured one (cosine, gain, RMSE against predicting nothing).

Compared here, every pair of
  representation  unimol_pc32   the teacher's own input (Uni-Mol, standardised, 32 PCs)
                  unimol_raw    Uni-Mol's 512 numbers, standardised
                  rdkit         RDKit's descriptor block (nan/constant columns dropped, rank-reduced)
                  morgan        Morgan fingerprint, radius 2, 2048 bits (Jaccard kernel)
                  rdkit_morgan  the two concatenated after whitening
                  target        one-hot of the annotated target (not structure: an upper bound)
                  pathway2      one-hot of pathway_level_2 (coarser annotation)
  predictor       ridge         ridge regression per (cell line, dose), inner-CV penalty
                  kernel        similarity-weighted average of the other drugs' measured shifts
                  teacher_mlp   the current recipe: 4-number drug code from a linear map, tanh MLP
                  zero          predict the vehicle mean (no drug effect)
                  class_mean    average shift of the other drugs with the same annotated target

A fold's response basis is refit without the held-out drug, so nothing of it enters the fit.

usage: python -m sciplex_standard.represent --root results/standard
Writes <root>/diagnose/represent.{jsonl,md} and <root>/diagnose/shifts.npz (mean shifts, reusable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .md import md_table
from .data import DEFAULT_FILE, REPO, StdData

ROOT = REPO / "results" / "standard"
K_BASIS = 20
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
KERNEL_TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)


def annotations(path: str, cache: Path | None = None) -> pd.DataFrame:
    """Target and pathway labels per drug. Reading them out of the h5ad takes minutes, so cache."""
    cache = Path(cache) if cache else REPO / "results" / "standard" / "diagnose" / "drug_annotations.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index("drug")
    import anndata as ad
    o = ad.read_h5ad(path, backed="r").obs
    o = o[o["control"].astype(int) == 0]
    t = pd.DataFrame({"drug": o["condition"].astype(str).to_numpy(), "target": o["target"].astype(str).to_numpy(),
                      "pathway1": o["pathway_level_1"].astype(str).to_numpy(),
                      "pathway2": o["pathway_level_2"].astype(str).to_numpy()})
    t = t.groupby("drug").first()
    cache.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(cache)
    return t


def shift_table(data: StdData) -> dict:
    """Each condition's mean shift from its cell line's vehicle mean, and the error on that mean.

    `sem2` is the per-gene variance of the condition mean (within-condition variance over the cell
    count, plus the vehicle mean's own, much smaller, term). A condition measured on 40 cells and one
    measured on 400 carry very different amounts of noise, and the size of a mean shift is inflated
    by it, so anything that averages measured shifts needs this to tell signal from cell count.
    """
    m0, v0 = {}, {}
    for c in data.cell_types:
        Xc = data.X[data.ctrl_idx(c)]
        m0[c] = Xc.mean(0, dtype=np.float64)
        v0[c] = Xc.var(0, dtype=np.float64) / len(Xc)
    keys = [k for k in data.cond.index if data.cond.n_cells[k] >= 1]
    S, E = [], []
    for k in keys:
        Xk = data.X[data.idx(k)]
        ct = data.cond.cell_type[k]
        S.append(Xk.mean(0, dtype=np.float64) - m0[ct])
        E.append(Xk.var(0, dtype=np.float64) / max(len(Xk), 1) + v0[ct])
    S = np.stack(S)
    c = data.cond.loc[keys]
    return {"keys": np.array(keys), "shift": S.astype(np.float32), "sem2": np.stack(E).astype(np.float32),
            "drug": c.drug.to_numpy().astype(str),
            "cell_type": c.cell_type.to_numpy().astype(str), "dose_nM": c.dose_nM.to_numpy().astype(float),
            "split": c.split.to_numpy().astype(str),
            "control_means": np.stack([m0[c] for c in data.cell_types]).astype(np.float32),
            "cell_types": np.array(data.cell_types)}


def whiten(F: np.ndarray, k: int = 64) -> np.ndarray:
    F = np.asarray(F, np.float64)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    u, s, _ = np.linalg.svd(F - F.mean(0), full_matrices=False)
    k = min(k, (s > 1e-8 * s[0]).sum())
    out = u[:, :k] * s[:k]
    return out / out[:, 0].std()


def representations(drugs: list[str], mol_path: str, smiles_path: str, ann: pd.DataFrame) -> dict:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    z = np.load(mol_path)
    names = [str(d) for d in z["drugs"]]
    E = z["emb"][[names.index(d) for d in drugs]].astype(np.float64)
    reps = {"unimol_raw": (E - E.mean(0)) / (E.std(0) + 1e-8), "unimol_pc32": whiten(E, 32)}

    sm = pd.read_csv(smiles_path).set_index("drug").smiles.to_dict()
    mols = [Chem.MolFromSmiles(sm[d].split("|")[0]) for d in drugs]
    if any(m is None for m in mols):
        raise SystemExit("RDKit could not parse every SMILES")
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    ok = np.isfinite(desc).all(0) & (desc.std(0) > 0)
    reps["rdkit"] = whiten(desc[:, ok], 64)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = np.array([list(gen.GetFingerprint(m)) for m in mols], dtype=np.float64)
    reps["morgan"] = fp[:, fp.sum(0) > 0]
    reps["rdkit_morgan"] = np.hstack([reps["rdkit"], whiten(reps["morgan"], 64)])
    # the three structure views answer for different drugs (Uni-Mol finds Quisinostat's analogue,
    # RDKit finds Givinostat's), so a view that carries all three is worth its own arm
    reps["structure3"] = np.hstack([whiten(E, 32), reps["rdkit"], whiten(reps["morgan"], 64)])
    for name, col in (("target", "target"), ("pathway2", "pathway2")):
        lab = ann.loc[drugs, col].to_numpy()
        cats = sorted(set(lab))
        reps[name] = np.eye(len(cats))[[cats.index(x) for x in lab]]
    # structure plus the annotated target: the HSP90 pair has no structural analogue in training
    reps["structure3_target"] = np.hstack([reps["structure3"], 2.0 * reps["target"]])
    return reps


def similarity(F: np.ndarray, name: str) -> np.ndarray:
    """Similarity in [0, 1]-ish between drugs: Jaccard for the fingerprint, else a Gaussian kernel."""
    if name == "morgan":
        inter = F @ F.T
        n = F.sum(1)
        return inter / np.maximum(n[:, None] + n[None] - inter, 1e-9)
    D = np.sqrt(np.maximum(((F[:, None] - F[None]) ** 2).sum(-1), 0))
    return np.exp(-(D / np.median(D[np.triu_indices(len(F), 1)])) ** 2)


def predict_fold(rep: np.ndarray, sim: np.ndarray, Y: np.ndarray, groups: np.ndarray, tr: np.ndarray,
                 te: np.ndarray, kind: str, cls: np.ndarray | None) -> np.ndarray:
    """Coordinates for the held-out conditions `te`, fit on the rows `tr`; grouped by (cell line, dose)."""
    out = np.zeros((len(te), Y.shape[1]))
    for g in np.unique(groups[te]):
        gi_te = te[groups[te] == g]
        gi_tr = tr[groups[tr] == g]
        if len(gi_tr) < 5:
            continue
        if kind == "zero":
            continue
        if kind == "class_mean":
            for j, i in enumerate(gi_te):
                same = gi_tr[cls[gi_tr] == cls[i]]
                src = same if len(same) else gi_tr
                out[np.flatnonzero(te == i)[0]] = Y[src].mean(0)
            continue
        if kind == "kernel":
            best, best_err = None, np.inf          # temperature by inner leave-one-out on the fold
            for tau in KERNEL_TAUS:
                W = np.exp(sim[np.ix_(gi_tr, gi_tr)] / tau)
                np.fill_diagonal(W, 0.0)
                P = (W / W.sum(1, keepdims=True)) @ Y[gi_tr]
                err = ((P - Y[gi_tr]) ** 2).sum()
                if err < best_err:
                    best, best_err = tau, err
            W = np.exp(sim[np.ix_(gi_te, gi_tr)] / best)
            out[np.isin(te, gi_te)] = (W / W.sum(1, keepdims=True)) @ Y[gi_tr]
            continue
        X, Yg = rep[gi_tr], Y[gi_tr]                # ridge, penalty by inner leave-one-out
        xm, ym = X.mean(0), Yg.mean(0)
        Xc, Yc = X - xm, Yg - ym
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        UtY = U.T @ Yc
        best, best_err = RIDGE_GRID[0], np.inf
        for lam in RIDGE_GRID:
            d = s / (s ** 2 + lam)
            H = (U * (s * d)).sum(1)                 # leverage of each row
            fit = U @ (d[:, None] * UtY)
            r = (Yc - fit) / np.maximum(1 - H[:, None], 1e-6)
            if (err := (r ** 2).sum()) < best_err:
                best, best_err = lam, err
        W = Vt.T @ ((s / (s ** 2 + best))[:, None] * UtY)
        out[np.isin(te, gi_te)] = (rep[gi_te] - xm) @ W + ym
    return out


def teacher_mlp(feats: np.ndarray, Y: np.ndarray, cell_idx: np.ndarray, logdose: np.ndarray,
                tr: np.ndarray, te: np.ndarray, seeds: int = 3, steps: int = 3000) -> np.ndarray:
    """The current teacher recipe (sciplex_standard.teacher), refit on the fold."""
    import torch
    scale = float(Y[tr].std())
    x = lambda i: (torch.tensor(feats[i], dtype=torch.float32), torch.tensor(cell_idx[i]),
                   torch.tensor(logdose[i][:, None], dtype=torch.float32))
    x_tr, x_te = x(tr), x(te)
    Yt = torch.tensor(Y[tr] / scale, dtype=torch.float32)
    preds = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        drug_map = torch.nn.Linear(feats.shape[1], 4)
        cell_code = torch.nn.Parameter(0.1 * torch.randn(int(cell_idx.max()) + 1, 2))
        net = torch.nn.Sequential(torch.nn.Linear(7, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                  torch.nn.Tanh(), torch.nn.Linear(64, Y.shape[1]))
        opt = torch.optim.AdamW([*drug_map.parameters(), cell_code, *net.parameters()], lr=3e-3, weight_decay=1e-4)
        f = lambda a: net(torch.cat([drug_map(a[0]), cell_code[a[1]], a[2]], 1))
        for _ in range(steps):
            opt.zero_grad()
            ((f(x_tr) - Yt) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():
            preds.append(f(x_te).numpy() * scale)
    return np.mean(preds, 0)


def run(data: StdData, sh: dict, reps: dict, ann: pd.DataFrame, out_dir: Path, mlp_rep: str = "unimol_pc32",
        folds: list[str] | None = None, with_mlp: bool = True) -> pd.DataFrame:
    drugs = sorted(set(sh["drug"]))
    d_of = {d: i for i, d in enumerate(drugs)}
    S = sh["shift"].astype(np.float64)
    usable = np.isin(sh["split"], ["train", "val"])          # conditions a model may fit on
    groups = np.array([f"{c}|{d:g}" for c, d in zip(sh["cell_type"], sh["dose_nM"])])
    gidx = pd.Categorical(groups).codes
    cell_idx = pd.Categorical(sh["cell_type"], categories=list(data.cell_types)).codes.astype(np.int64)
    logdose = (np.log10(sh["dose_nM"]) / 4).astype(np.float64)
    drug_idx = np.array([d_of[d] for d in sh["drug"]])
    cls = pd.Categorical(ann.loc[sh["drug"], "target"].to_numpy()).codes
    sims = {n: similarity(np.asarray(F, np.float64), n)[np.ix_(drug_idx, drug_idx)] for n, F in reps.items()}
    rows_of = {n: np.asarray(F, np.float64)[drug_idx] for n, F in reps.items()}
    rows = []
    for d in (folds or drugs):
        te = np.flatnonzero(sh["drug"] == d)
        tr = np.flatnonzero(usable & (sh["drug"] != d))
        V = np.linalg.svd(S[tr], full_matrices=False)[2][:K_BASIS]      # basis without the held-out drug
        Y = S[tr] @ V.T
        Yfull = np.zeros((len(S), K_BASIS))
        Yfull[tr] = Y
        for rname in reps:
            Frow, Srow = rows_of[rname], sims[rname]
            for kind in ("ridge", "kernel", "class_mean", "zero"):
                if kind == "class_mean" and rname != "unimol_pc32":
                    continue                                   # annotation predictor, representation-free
                if kind == "zero" and rname != "unimol_pc32":
                    continue
                C = predict_fold(Frow, Srow, Yfull, gidx, tr, te, kind, cls)
                P = C @ V
                for j, i in enumerate(te):
                    t, p = S[i], P[j]
                    nt = np.linalg.norm(t)
                    rows.append({"drug": d, "condition": sh["keys"][i], "cell_type": sh["cell_type"][i],
                                 "dose_nM": float(sh["dose_nM"][i]), "split": sh["split"][i],
                                 "rep": rname if kind not in ("class_mean", "zero") else "-", "predictor": kind,
                                 "true_norm": float(nt), "gain": float(np.linalg.norm(p) / max(nt, 1e-12)),
                                 "cos": float(t @ p / max(nt * np.linalg.norm(p), 1e-12)),
                                 "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
                                 "rmse_zero": float(np.sqrt(np.mean(t ** 2)))})
        if not with_mlp:
            print(f"drug {d}: done ({len(te)} conditions)", flush=True)
            continue
        C = teacher_mlp(np.asarray(reps[mlp_rep], np.float64)[drug_idx], Yfull, cell_idx, logdose, tr, te)
        P = C @ V
        for j, i in enumerate(te):
            t, p = S[i], P[j]
            nt = np.linalg.norm(t)
            rows.append({"drug": d, "condition": sh["keys"][i], "cell_type": sh["cell_type"][i],
                         "dose_nM": float(sh["dose_nM"][i]), "split": sh["split"][i], "rep": mlp_rep,
                         "predictor": "teacher_mlp", "true_norm": float(nt),
                         "gain": float(np.linalg.norm(p) / max(nt, 1e-12)),
                         "cos": float(t @ p / max(nt * np.linalg.norm(p), 1e-12)),
                         "rmse": float(np.sqrt(np.mean((p - t) ** 2))), "rmse_zero": float(np.sqrt(np.mean(t ** 2)))})
        print(f"drug {d}: done ({len(te)} conditions)", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, ood: list[str]) -> str:
    L = ["# Leave-one-drug-out: can the structure -> response map find a held-out drug's response?", "",
         "cos = cosine of the predicted and measured mean shift, gain = its relative size,",
         "rmse_ratio = RMSE of the predicted mean / RMSE of predicting no effect (< 1 beats the control).", ""]
    folds = [(f"every drug held out in turn ({df.drug.nunique()} folds)", df)]
    if not set(df.drug) <= set(ood):
        folds.append((f"the {len(ood)} benchmark OOD drugs", df[df.drug.isin(ood)]))
    for name, sub in folds:
        g = sub.assign(rmse_ratio=sub.rmse / sub.rmse_zero).groupby(["predictor", "rep"]).agg(
            cos=("cos", "mean"), gain=("gain", "mean"), rmse_ratio=("rmse_ratio", "mean"),
            n=("cos", "size")).round(3).sort_values("cos", ascending=False)
        L += [f"## {name}", "", md_table(g), ""]
    best = df.assign(r=df.rmse / df.rmse_zero).groupby(["predictor", "rep"]).r.mean().idxmin()
    sub = df[(df.predictor == best[0]) & (df.rep == best[1]) & df.drug.isin(ood)]
    L += [f"## {best[0]} / {best[1]} on the OOD drugs, by drug", "",
          md_table(sub.groupby("drug").agg(true_norm=("true_norm", "mean"), cos=("cos", "mean"), gain=("gain", "mean")
                                           ).round(3).sort_values("true_norm", ascending=False)), ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--molecules", default=str(ROOT / "unimol_embed.npz"))
    p.add_argument("--smiles", default=str(ROOT / "smiles.csv"))
    p.add_argument("--tag", default="represent", help="output file stem under <root>/diagnose")
    p.add_argument("--only-ood", action="store_true", help="hold out only the 9 benchmark OOD drugs")
    p.add_argument("--no-mlp", action="store_true", help="skip the current teacher recipe (the slow arm)")
    p.add_argument("--render", action="store_true", help="rebuild the table from the saved rows, computing nothing")
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.root) / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    from .data import OOD_DRUGS
    if a.render:
        df = pd.read_json(out / f"{a.tag}.jsonl", lines=True)
        (out / f"{a.tag}.md").write_text(summarise(df, OOD_DRUGS) + "\n")
        print(f"wrote {out}/{a.tag}.md")
        return
    cache = out / "shifts.npz"
    data = StdData(a.file, load_X=not cache.exists())   # the cached mean shifts make the matrix unnecessary
    if cache.exists():
        sh = {k: v for k, v in np.load(cache, allow_pickle=True).items()}
    else:
        sh = shift_table(data)
        np.savez_compressed(cache, **sh)
    ann = annotations(a.file)
    drugs = sorted(set(sh["drug"]))
    reps = representations(drugs, a.molecules, a.smiles, ann)
    print({k: np.shape(v) for k, v in reps.items()}, flush=True)
    df = run(data, sh, reps, ann, out, folds=OOD_DRUGS if a.only_ood else None, with_mlp=not a.no_mlp)
    df.to_json(out / f"{a.tag}.jsonl", orient="records", lines=True)
    (out / f"{a.tag}.md").write_text(summarise(df, OOD_DRUGS) + "\n")
    print(f"wrote {out}/{a.tag}.md", flush=True)


if __name__ == "__main__":
    main()
