"""Condition response embedding (the "teacher" of Ours) for the standard benchmark.

The recipe of sciplex_interp.teacher / stats.autodecoder_seeds (an MLP maps a drug code and log10
dose to the coordinates of the condition's mean shift in the top-K principal directions of the
training mean shifts; 5 seeds averaged), extended to the standard setting:

  * three cell lines: a shift is the condition's mean minus the vehicle mean of its own cell line,
    and the MLP also receives a learned 2-number code per cell line;
  * unseen drugs: the drug's 4-number code is computed from its structure, a learned linear map of
    its Uni-Mol embedding (standardised over the 187 drugs and reduced to 32 principal components;
    structures only, no expression), instead of a free code per drug. So a drug never seen in
    training (Table 2) gets a code, and seen drugs share the same map.

Fit on the means of `train` conditions only (val, test and ood conditions are never read). Writes
<root>/teacher/std_autodec.npz: keys (cov_drug_dose_name of every condition), coef (n x K, unscaled
coordinates), basis (K x genes), scale (SD of the training coordinates), control_means (cell lines x
genes), cell_types. A validation check prints the teacher-mean RMSE on val conditions.

usage: python -m sciplex_standard.teacher --molecules results/standard/unimol_embed.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import DEFAULT_FILE, REPO, StdData

K_BASIS, K_DRUG, K_CELL, K_STRUCT = 20, 4, 2, 32


def structure_features(drugs: list[str], mol_path: str) -> np.ndarray:
    z = np.load(mol_path)
    names = [str(d) for d in z["drugs"]]
    missing = sorted(set(drugs) - set(names))
    if missing:
        raise SystemExit(f"no Uni-Mol embedding for {missing}")
    E = z["emb"][[names.index(d) for d in drugs]].astype(np.float64)
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    u, s, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    F = u[:, :K_STRUCT] * s[:K_STRUCT]
    return (F / F[:, 0].std()).astype(np.float32)


def fit(data: StdData, mol_path: str, seeds: int = 5, steps: int = 3000, width: int = 64, wd: float = 1e-4) -> dict:
    import torch

    m0 = {c: data.X[data.ctrl_idx(c)].mean(0, dtype=np.float64) for c in data.cell_types}
    train = data.keys("train", min_cells=1)
    shift = np.stack([data.X[data.idx(k)].mean(0, dtype=np.float64) - m0[data.cond.cell_type[k]] for k in train])
    V = np.linalg.svd(shift, full_matrices=False)[2][:K_BASIS]
    Y = shift @ V.T
    scale = float(Y.std())
    feats = structure_features(data.drugs, mol_path)
    d_of = {d: i for i, d in enumerate(data.drugs)}
    c_of = {c: i for i, c in enumerate(data.cell_types)}

    def inputs(keys):
        c = data.cond.loc[keys]
        return (torch.tensor(feats[[d_of[d] for d in c.drug]]), torch.tensor([c_of[x] for x in c.cell_type]),
                torch.tensor(np.log10(c.dose_nM.to_numpy())[:, None] / 4, dtype=torch.float32))

    x_tr, q_all = inputs(train), list(data.cond.index)
    x_q = inputs(q_all)
    Yt = torch.tensor(Y / scale, dtype=torch.float32)
    preds = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        drug_map = torch.nn.Linear(K_STRUCT, K_DRUG)
        cell_code = torch.nn.Parameter(0.1 * torch.randn(len(data.cell_types), K_CELL))
        net = torch.nn.Sequential(torch.nn.Linear(K_DRUG + K_CELL + 1, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, K_BASIS))
        params = [*drug_map.parameters(), cell_code, *net.parameters()]
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=wd)
        f = lambda x: net(torch.cat([drug_map(x[0]), cell_code[x[1]], x[2]], 1))
        for _ in range(steps):
            opt.zero_grad()
            loss = ((f(x_tr) - Yt) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds.append(f(x_q).numpy() * scale)
        print(f"seed {seed}: training MSE (scaled) {loss.item():.4f}", flush=True)
    coef = np.mean(preds, 0)
    return {"keys": np.array(q_all), "coef": coef.astype(np.float32), "basis": V.astype(np.float32),
            "scale": np.float32(scale), "control_means": np.stack([m0[c] for c in data.cell_types]).astype(np.float32),
            "cell_types": np.array(data.cell_types), "n_train_conditions": np.int64(len(train))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecules", required=True)
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(REPO / "results" / "standard"))
    a = p.parse_args(argv)
    out = Path(a.root) / "teacher" / "std_autodec.npz"
    if out.exists():
        print(f"{out} exists")
        return
    data = StdData(a.file)
    t = fit(data, a.molecules)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **t)
    lookup = dict(zip(t["keys"], t["coef"]))
    ct = list(t["cell_types"])
    err, base = [], []
    for k in data.keys("val"):  # validation only; test and ood means are not read here
        real = data.X[data.idx(k)].mean(0, dtype=np.float64)
        m0 = t["control_means"][ct.index(data.cond.cell_type[k])].astype(np.float64)
        err.append(np.sqrt(np.mean((m0 + lookup[k] @ t["basis"] - real) ** 2)))
        base.append(np.sqrt(np.mean((m0 - real) ** 2)))
    print(f"wrote {out}: {len(t['keys'])} conditions, {int(t['n_train_conditions'])} training means; validation RMSE "
          f"(All) of the teacher mean {np.mean(err):.4f} vs vehicle mean {np.mean(base):.4f}")


if __name__ == "__main__":
    main()
