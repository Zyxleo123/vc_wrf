"""Train DOLORIS on the panel's training conditions: the mask model, then the diffusion model.

Settings come from the config (configs/doloris_v1.json, transcribed from the released
config/sciplex3/*.yaml and train.py defaults):

  mask       MaskModel, 2 epochs over the *treated* training cells, batch 32, AdamW(lr 1e-2,
             weight decay 1e-2), x_noisy = control mean + control SD * N(0, 1)  (main_mask_model.py)
  diffusion  TargetModel, `steps` updates of a random batch of 32 training cells (treated and
             vehicle), AdamW(lr 1e-4, no weight decay), learning rate annealed linearly to 0 over
             the run (TrainLoop), masked x0-MSE over the 50 retained diffusion steps

The released loop keeps only the last weights (model.pt, rewritten every 10k steps) and test.py
loads them. Here the weights are also saved every `save_every` updates, so generate.py can score
the released rule (final weights) and a validation-RMSE rule from the same run. Training resumes
from the last saved step.

usage: python -m sciplex_doloris.train --config configs/doloris_v1.json --seed 0 \
         --molecules results/doloris/unimol_embed.npz --out results/doloris/runs/v1/seed_0 [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from sciplex_interp.core import DEFAULT_PANEL, Data

from .data import coexpression_adjacency, load_config, load_molecules, train_set
from .diffusion import SpacedDiffusion
from .model import MaskModel, TargetModel, graph_edges


def build_target(cfg: dict, n_genes: int) -> TargetModel:
    m = cfg["model"]
    return TargetModel(n_genes, hidden_dim=m["hidden_dim"], time_pos_dim=m["time_pos_dim"],
                       embed_dim=m["gene_wise_embed_dim"], mole_dim=m["mole_dim"])


def build_mask(cfg: dict, n_genes: int, edges: torch.Tensor) -> MaskModel:
    m = cfg["mask"]
    return MaskModel(n_genes, edges, gene_init_dim=m["gene_init_dim"], mole_dim=cfg["model"]["mole_dim"], heads=m["gat_heads"])


def build_diffusion(cfg: dict) -> SpacedDiffusion:
    d = cfg["diffusion"]
    return SpacedDiffusion(d["steps"], d["respacing"], d["clip_denoised"])


def gene_edges(cfg: dict, ts) -> tuple[torch.Tensor, np.ndarray]:
    g = cfg["grn"]
    adj = coexpression_adjacency(ts.X[ts.is_ctrl], g["threshold_co"], g["n_cells"], g["seed"])
    return graph_edges(adj), adj


def ckpt_steps(run: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in run.glob("ckpt_*.pt"))


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    tmp.rename(path)


def train_mask(cfg, ts, mols, run: Path, seed: int, device: str) -> None:
    path = run / "mask_model.pt"
    if path.exists():
        print(f"{path} exists; skipping the mask model")
        return
    torch.manual_seed(seed)
    m = cfg["mask"]
    edges, adj = gene_edges(cfg, ts)
    model = build_mask(cfg, ts.X.shape[1], edges).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    treated = np.flatnonzero(~ts.is_ctrl)
    X = torch.as_tensor(ts.X[treated], device=device)
    mole = torch.as_tensor(mols.rows(ts.drug_idx[treated]), device=device)
    dosage = torch.as_tensor(ts.dosage[treated], device=device)
    mu = torch.as_tensor(ts.ctrl_mean, device=device)
    var = torch.as_tensor(ts.ctrl_var, device=device)
    gen = torch.Generator().manual_seed(seed)
    losses, t0 = [], time.time()
    for epoch in range(m["epochs"]):
        perm = torch.randperm(len(treated), generator=gen).to(device)
        for i in range(0, len(perm), m["batch_size"]):  # DataLoader(shuffle=True), last batch kept
            b = perm[i:i + m["batch_size"]]
            loss = model.loss(X[b], mole[b], dosage[b], mu.expand(len(b), -1), var.expand(len(b), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"mask epoch {epoch + 1}: mean BCE {np.mean(losses[-(len(perm) // m['batch_size'] + 1):]):.4f}")
    torch.save(model.state_dict(), path)
    write_json(run / "mask_info.json", {"n_updates": len(losses), "final_loss_mean_last100": float(np.mean(losses[-100:])),
                                        "n_genes": int(adj.shape[0]), "n_undirected_edges": int((adj.sum() - adj.shape[0]) // 2),
                                        "n_directed_edges_with_self_loops": int(edges.shape[1]), "seconds": time.time() - t0})


def train_diffusion(cfg, ts, mols, run: Path, seed: int, device: str) -> None:
    t = cfg["train"]
    steps, save_every, bs = t["steps"], t["save_every"], t["batch_size"]
    if (run / "training_complete.json").exists():
        print("diffusion training already complete")
        return
    torch.manual_seed(seed)
    model = build_target(cfg, ts.X.shape[1]).to(device).train()
    diffusion = build_diffusion(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    gpu_gen = torch.Generator(device=device).manual_seed(seed + 1)
    start = 0
    done = ckpt_steps(run)
    if done and (run / "resume.pt").exists():
        state = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)  # RNG states must stay on the CPU
        if state["step"] == done[-1]:
            model.load_state_dict(torch.load(run / f"ckpt_{done[-1]}.pt", map_location=device, weights_only=True))
            opt.load_state_dict(state["opt"])  # moves its state to the parameters' device
            gen.set_state(state["gen"])
            gpu_gen.set_state(state["gpu_gen"])
            torch.set_rng_state(state["torch_rng"])
            if device.startswith("cuda"):
                torch.cuda.set_rng_state(state["cuda_rng"])
            start = state["step"]
            print(f"resumed at update {start}")

    X = torch.as_tensor(ts.X, device=device)
    mole = torch.as_tensor(mols.rows(ts.drug_idx), device=device)
    dosage = torch.as_tensor(ts.dosage, device=device)
    mu = torch.as_tensor(ts.ctrl_mean, device=device).expand(bs, -1)
    var = torch.as_tensor(ts.ctrl_var, device=device).expand(bs, -1)
    n = len(ts.X)
    hist = open(run / "history.jsonl", "a")
    window, t0 = [], time.time()
    for k in range(start, steps):
        # TrainLoop: the first update uses lr; after update j the rate is set to lr * (1 - j / steps)
        for g in opt.param_groups:
            g["lr"] = t["lr"] * (1 - max(k - 1, 0) / steps)
        b = torch.randperm(n, generator=gen)[:bs].to(device)  # next(iter(shuffled DataLoader))
        loss = diffusion.training_loss(model, X[b], {"mole": mole[b], "dosage": dosage[b], "ctrl_mean": mu, "ctrl_var": var},
                                       generator=gpu_gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        window.append(loss.item())
        done_updates = k + 1
        if done_updates % 1000 == 0:
            hist.write(json.dumps({"step": done_updates, "loss": float(np.mean(window)), "lr": opt.param_groups[0]["lr"],
                                   "seconds": time.time() - t0}) + "\n")
            hist.flush()
            print(f"update {done_updates}: loss {np.mean(window):.5f}  ({time.time() - t0:.0f}s)")
            window = []
        if done_updates % save_every == 0 or done_updates == steps:
            torch.save(model.state_dict(), run / f"ckpt_{done_updates}.pt")
            torch.save({"step": done_updates, "opt": opt.state_dict(), "gen": gen.get_state(), "gpu_gen": gpu_gen.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if device.startswith("cuda") else None}, run / "resume.pt.tmp")
            (run / "resume.pt.tmp").rename(run / "resume.pt")
    hist.close()
    write_json(run / "training_complete.json", {"steps": steps, "checkpoints": ckpt_steps(run), "seconds_this_job": time.time() - t0,
                                                "n_parameters": sum(p.numel() for p in model.parameters())})
    (run / "resume.pt").unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--molecules", required=True)
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", choices=["mask", "diffusion", "both"], default="both")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    saved = run / "config.json"
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise SystemExit(f"{run} was trained with a different config; refusing to mix runs")
    write_json(saved, cfg)
    data, mols = Data(a.data), load_molecules(a.molecules)
    ts = train_set(data, mols)
    write_json(run / "run.json", {"seed": a.seed, "panel": a.data, "molecules": a.molecules, "x_max": ts.x_max,
                                  "n_train_cells": int(len(ts.X)), "n_controls": int(ts.is_ctrl.sum()),
                                  "n_train_conditions": int(len(set(ts.condition[~ts.is_ctrl]))),
                                  "torch": torch.__version__, "python": platform.python_version(),
                                  "device": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"})
    if a.stage in ("mask", "both"):
        train_mask(cfg, ts, mols, run, a.seed, a.device)
    if a.stage in ("diffusion", "both"):
        train_diffusion(cfg, ts, mols, run, a.seed, a.device)


if __name__ == "__main__":
    main()
