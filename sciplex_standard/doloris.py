"""DOLORIS (the parity-checked port in sciplex_doloris) on the standard benchmark.

Same networks, diffusion and training settings as the A549 run (configs/doloris_v1.json), with what
the released code does on chemCPA's three cell lines:
  * cell type input: a learned embedding (256) feeding the shared trunk and the condition code, and
    (512) the mask model, as the released classes do when cell_type_num > 1;
  * control prior per cell type: mean / variance of that cell line's (scaled) vehicle cells;
  * inference (test_util.predict): 100 vehicle cells per cell line, drawn once, DDIM-inverted under the
    control condition; each condition is DDIM-sampled from its cell line's latents; the mask's pool is
    every training (drug, cell line, dose) condition and the vehicles; mask thresholds 0.99 / 0.01 for
    unseen combinations (test, val) and 0.95 / 0.05 for unseen drugs (ood), as the released branches use.
Checkpoints every 5k of 60k updates; rules "final" (released) and "bestval" (lowest val RMSE (All)).

usage (GPU): python -m sciplex_standard.doloris train --seed 0
             python -m sciplex_standard.doloris generate --seed 0
Writes <root>/runs/doloris/seed_<s>/ and <root>/preds/{doloris_final,doloris_bestval}/seed_<s>/{val,test,ood}.h5ad.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sciplex_doloris.data import coexpression_adjacency, load_molecules
from sciplex_doloris.diffusion import SpacedDiffusion
from sciplex_doloris.generate import sample_mask
from sciplex_doloris.model import MaskModel, TargetModel, graph_edges

from .data import DEFAULT_FILE, REPO, StdData

ROOT = REPO / "results" / "standard"
CONFIG = REPO / "configs" / "standard_doloris_v1.json"
SPLITS = ("val", "test", "ood")


class Inputs:
    """Training rows (train conditions + vehicles), scaled by x_max, with per-row covariates."""

    def __init__(self, data: StdData, mols):
        rows = data.train_rows()
        o = data.obs.iloc[rows]
        X = data.X[rows]
        self.x_max = float(X.max())
        self.X = (X / self.x_max).astype(np.float32)
        self.is_ctrl = o.control.to_numpy()
        self.cell = np.array([data.cell_types.index(c) for c in o.cell_type], dtype=np.int64)
        self.drug_idx = np.array([-1 if c else mols.drugs.index(d) for d, c in zip(o.drug, o.control)], dtype=np.int64)
        self.dosage = np.where(self.is_ctrl, 0.0, o.dose_val.to_numpy()).astype(np.float32)[:, None]
        self.key = np.where(self.is_ctrl, "control|" + o.cell_type.to_numpy(), o.key.to_numpy())
        ct = len(data.cell_types)
        self.ctrl_mean = np.stack([self.X[self.is_ctrl & (self.cell == i)].astype(np.float64).mean(0) for i in range(ct)]).astype(np.float32)
        self.ctrl_var = np.stack([self.X[self.is_ctrl & (self.cell == i)].astype(np.float64).var(0, ddof=1) for i in range(ct)]).astype(np.float32)


def models(cfg: dict, n_genes: int, n_cells: int, edges):
    m, k = cfg["model"], cfg["mask"]
    target = TargetModel(n_genes, hidden_dim=m["hidden_dim"], time_pos_dim=m["time_pos_dim"], embed_dim=m["gene_wise_embed_dim"],
                         mole_dim=m["mole_dim"], cell_type_num=n_cells, cell_type_embed_dim=m["cell_type_embed_dim"])
    mask = MaskModel(n_genes, edges, gene_init_dim=k["gene_init_dim"], mole_dim=m["mole_dim"], heads=k["gat_heads"],
                     cell_type_num=n_cells, cell_type_embed_dim=k["cell_type_embed_dim"])
    return target, mask


def gene_graph(cfg: dict, inp: Inputs):
    g = cfg["grn"]
    return graph_edges(coexpression_adjacency(inp.X[inp.is_ctrl], g["threshold_co"], g["n_cells"], g["seed"]))


def ckpts(run: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in run.glob("ckpt_*.pt"))


def train(seed: int, root: Path, path: str, mol_path: str, device: str = "cuda") -> None:
    cfg = json.loads(CONFIG.read_text())
    run = root / "runs" / "doloris" / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "training_complete.json").exists():
        print(f"seed {seed}: already trained")
        return
    data, mols = StdData(path), load_molecules(mol_path)
    inp = Inputs(data, mols)
    (run / "run.json").write_text(json.dumps({"seed": seed, "x_max": inp.x_max, "n_rows": len(inp.X),
                                              "device": torch.cuda.get_device_name(0)}, indent=1) + "\n")
    torch.manual_seed(seed)
    target, mask = models(cfg, inp.X.shape[1], len(data.cell_types), gene_graph(cfg, inp))
    X = torch.as_tensor(inp.X, device=device)
    mole = torch.as_tensor(mols.rows(inp.drug_idx), device=device)
    dosage = torch.as_tensor(inp.dosage, device=device)
    cell = torch.as_tensor(inp.cell, device=device)
    mu, var = torch.as_tensor(inp.ctrl_mean, device=device), torch.as_tensor(inp.ctrl_var, device=device)

    if not (run / "mask_model.pt").exists():  # main_mask_model.py: treated training cells, 2 epochs
        k = cfg["mask"]
        mask = mask.to(device).train()
        opt = torch.optim.AdamW(mask.parameters(), lr=k["lr"], weight_decay=k["weight_decay"])
        treated = torch.as_tensor(np.flatnonzero(~inp.is_ctrl), device=device)
        gen = torch.Generator().manual_seed(seed)
        t0 = time.time()
        for epoch in range(k["epochs"]):
            perm = treated[torch.randperm(len(treated), generator=gen).to(device)]
            for i in range(0, len(perm), k["batch_size"]):
                b = perm[i:i + k["batch_size"]]
                loss = mask.loss(X[b], mole[b], dosage[b], mu[cell[b]], var[cell[b]], cell_type=cell[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f"mask epoch {epoch + 1}: last BCE {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(mask.state_dict(), run / "mask_model.pt")

    t = cfg["train"]
    diffusion = SpacedDiffusion(cfg["diffusion"]["steps"], cfg["diffusion"]["respacing"], cfg["diffusion"]["clip_denoised"])
    target = target.to(device).train()
    opt = torch.optim.AdamW(target.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    gen, gpu_gen = torch.Generator().manual_seed(seed), torch.Generator(device=device).manual_seed(seed + 1)
    start, done = 0, ckpts(run)
    if done and (run / "resume.pt").exists():
        state = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        if state["step"] == done[-1]:
            target.load_state_dict(torch.load(run / f"ckpt_{done[-1]}.pt", map_location=device, weights_only=True))
            opt.load_state_dict(state["opt"])
            gen.set_state(state["gen"]); gpu_gen.set_state(state["gpu_gen"])
            torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
            start = state["step"]
            print(f"resumed at update {start}", flush=True)
    window, t0 = [], time.time()
    with open(run / "history.jsonl", "a") as hist:
        for k in range(start, t["steps"]):
            for g in opt.param_groups:
                g["lr"] = t["lr"] * (1 - max(k - 1, 0) / t["steps"])
            b = torch.randperm(len(inp.X), generator=gen)[:t["batch_size"]].to(device)
            loss = diffusion.training_loss(target, X[b], {"mole": mole[b], "dosage": dosage[b], "ctrl_mean": mu[cell[b]],
                                                          "ctrl_var": var[cell[b]], "cell_type": cell[b]}, generator=gpu_gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            window.append(loss.item())
            n = k + 1
            if n % 1000 == 0:
                hist.write(json.dumps({"step": n, "loss": float(np.mean(window)), "seconds": time.time() - t0}) + "\n")
                hist.flush()
                print(f"update {n}: loss {np.mean(window):.5f} ({time.time() - t0:.0f}s)", flush=True)
                window = []
            if n % t["save_every"] == 0 or n == t["steps"]:
                torch.save(target.state_dict(), run / f"ckpt_{n}.pt")
                torch.save({"step": n, "opt": opt.state_dict(), "gen": gen.get_state(), "gpu_gen": gpu_gen.get_state(),
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, run / "resume.pt.tmp")
                (run / "resume.pt.tmp").rename(run / "resume.pt")
    (run / "training_complete.json").write_text(json.dumps({"steps": t["steps"], "checkpoints": ckpts(run)}, indent=1) + "\n")
    (run / "resume.pt").unlink(missing_ok=True)


class Sampler:
    def __init__(self, cfg, data: StdData, inp: Inputs, mols, run: Path, seed: int, device: str, offset: int = 0):
        self.cfg, self.data, self.inp, self.mols, self.run, self.seed, self.device = cfg, data, inp, mols, run, seed, device
        self.n = cfg["inference"]["n_cells"]
        self.diffusion = SpacedDiffusion(cfg["diffusion"]["steps"], cfg["diffusion"]["respacing"], cfg["diffusion"]["clip_denoised"])
        target, mask = models(cfg, inp.X.shape[1], len(data.cell_types), gene_graph(cfg, inp))
        self.target, self.mask = target.to(device), mask.to(device)
        self.mask.load_state_dict(torch.load(run / "mask_model.pt", map_location=device, weights_only=True))
        self.mask.eval()
        rng = np.random.default_rng([seed, offset, 7])
        self.ctrl = []
        for i in range(len(data.cell_types)):
            pick = np.sort(rng.choice(np.flatnonzero(inp.is_ctrl & (inp.cell == i)), self.n, replace=False))
            self.ctrl.append(torch.as_tensor(inp.X[pick], device=device))
        keys = pd.unique(inp.key)  # mask pool: every training condition (cell line specific) and the vehicles
        self.pool_keys = list(keys)
        self.pool_masks = [inp.X[inp.key == k] != 0 for k in keys]
        self.pool_probs = np.stack([m.mean(0) for m in self.pool_masks])

    def load(self, step: int) -> None:
        self.target.load_state_dict(torch.load(self.run / f"ckpt_{step}.pt", map_location=self.device, weights_only=True))
        self.target.eval()
        self.latent = []
        for i, x in enumerate(self.ctrl):
            kw = {"dosage": torch.zeros((self.n, 1), device=self.device), "cell_type": torch.full((self.n,), i, device=self.device)}
            self.latent.append(self.diffusion.ddim_reverse(self.target, x, kw))

    @torch.no_grad()
    def condition(self, key: str, split: str) -> np.ndarray:
        c = self.data.cond.loc[key]
        i = self.data.cell_types.index(c.cell_type)
        mole = torch.as_tensor(self.mols.of(c.drug), device=self.device)[None].repeat(self.n, 1)
        dosage = torch.full((self.n, 1), float(c.dose_val), device=self.device)
        ct = torch.full((self.n,), i, device=self.device)
        x = self.diffusion.ddim_sample(self.target, self.latent[i], {"mole": mole, "dosage": dosage, "ctrl_mean": self.ctrl[i],
                                                                     "cell_type": ct})
        prob = self.mask.predict(self.ctrl[i], mole, dosage, cell_type=ct).mean(0).cpu().numpy()
        high, low = (0.95, 0.05) if split == "ood" else (0.99, 0.01)
        rng = np.random.default_rng([self.seed, zlib.crc32(key.encode())])
        masks, _ = sample_mask(prob, self.pool_keys, self.pool_probs, self.pool_masks, self.n, high, low,
                               self.cfg["inference"]["mask_feature_topk"], rng)
        return np.clip(x.cpu().numpy() * masks * self.inp.x_max, 0, None).astype(np.float32)


def generate_all(seed: int, root: Path, path: str, mol_path: str, device: str = "cuda") -> None:
    import anndata as ad
    cfg = json.loads(CONFIG.read_text())
    run = root / "runs" / "doloris" / f"seed_{seed}"
    if not (run / "training_complete.json").exists():
        raise SystemExit(f"{run} has not finished training")
    data, mols = StdData(path), load_molecules(mol_path)
    inp = Inputs(data, mols)
    s = Sampler(cfg, data, inp, mols, run, seed, device)
    sel_path = run / "selection.jsonl"
    curve = {r["step"]: r for r in map(json.loads, sel_path.open())} if sel_path.exists() else {}
    val_keys = data.keys("val")
    for step in ckpts(run):
        if step in curve:
            continue
        s.load(step)
        err = [np.sqrt(np.mean((s.condition(k, "val").mean(0, dtype=np.float64)
                                - data.X[data.idx(k)].mean(0, dtype=np.float64)) ** 2)) for k in val_keys]
        curve[step] = {"step": step, "val_rmse_all": float(np.mean(err))}
        with open(sel_path, "a") as f:
            f.write(json.dumps(curve[step]) + "\n")
        print(f"checkpoint {step}: val RMSE (All) {curve[step]['val_rmse_all']:.5f}", flush=True)
    steps = ckpts(run)
    rules = {"doloris_final": steps[-1], "doloris_bestval": min(steps, key=lambda k: (curve[k]["val_rmse_all"], k))}
    (run / "selected.json").write_text(json.dumps(rules, indent=1) + "\n")
    for method, step in rules.items():
        todo = [sp for sp in SPLITS if not (root / "preds" / method / f"seed_{seed}" / f"{sp}.h5ad").exists()]
        if not todo:
            continue
        s.load(step)
        for split in todo:
            keys = data.keys(split)
            cells = {k: s.condition(k, split) for k in keys}
            labels = np.concatenate([[k] * len(cells[k]) for k in keys])
            pred = ad.AnnData(np.vstack([cells[k] for k in keys]), obs=pd.DataFrame({"condition": labels},
                              index=[f"g{i}" for i in range(len(labels))]), var=pd.DataFrame(index=data.genes))
            dest = root / "preds" / method / f"seed_{seed}" / f"{split}.h5ad"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.stem + ".tmp.h5ad")
            pred.write_h5ad(tmp, compression="gzip")
            tmp.rename(dest)
            print(f"{method} (step {step}) seed {seed} {split}: {len(keys)} conditions", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "generate"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--molecules", default=str(ROOT / "unimol_embed.npz"))
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(ROOT))
    a = p.parse_args(argv)
    if a.stage == "train":
        train(a.seed, Path(a.root), a.file, a.molecules)
    else:
        generate_all(a.seed, Path(a.root), a.file, a.molecules)


if __name__ == "__main__":
    main()
