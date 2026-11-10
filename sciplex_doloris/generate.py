"""Generate held-out-dose cells with a trained DOLORIS run, under two checkpoint rules.

  final    the last weights: the released rule (test.py loads result/target_*/model.pt, the
           non-EMA weights after the last update)
  bestval  the saved checkpoint with the lowest validation "RMSE (All)" (DOLORIS's RMSE: between
           the per-gene means of generated and measured cells over all genes, averaged over the 9
           validation conditions). Not part of the released code; reported beside it.

Generation for one condition follows test_util.predict (molecular branch, "unseen covariate"
split, which is the held-out (drug, dose) setting here):
  1. draw N training control cells once per (seed, sampling seed) and DDIM-invert them with the
     control condition (dosage 0) into the shared latent
  2. DDIM-sample (eta 0, x0 clipped to [-1, 1]) from those latents under the drug's Uni-Mol
     embedding and dose_val, with ctrl_mean = the same control cells (x_noisy = x^c)
  3. sparsity mask (sample_topk_conditions): the mask model's probabilities from the control cells,
     averaged over cells; the training condition whose empirical non-zero rates are nearest (L2
     over the 300 genes with the highest predicted probability); every generated cell takes the
     zero pattern of a random cell of that condition; genes with probability >= 0.99 are set to 1
     and <= 0.01 to 0 (the thresholds the released code uses for this split)
  4. cells x mask x x_max, negatives set to 0 (test.py), i.e. back in the panel's log1p(CP10K)
The released code draws 100 control cells; here N = 512, as for CellFlow and Ours, so that every
method is scored on the same number of generated cells (the evaluator needs >= the measured count).

usage: python -m sciplex_doloris.generate --run results/doloris/runs/v1/seed_0 --molecules ... \
         --preds-root results/doloris/preds --name v1 [--splits val test] [--sampling-seed 0]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from sciplex_interp.core import DEFAULT_PANEL, Data, read_jsonl

from .data import dose_val, load_config, load_molecules, train_set
from .train import build_diffusion, build_mask, build_target, ckpt_steps, gene_edges, write_json

RULES = ("final", "bestval")


def sample_mask(prob, keys, pool_probs, pool_masks, n, high, low, feature_topk, rng):
    """sample_topk_conditions(topk=1, distance L2, feature_selection=True): the pool condition nearest
    to the predicted non-zero rates over the `feature_topk` most likely genes lends n random cells'
    zero patterns; confident genes are then forced on (>= high) or off (<= low)."""
    active = np.argsort(prob)[-feature_topk:] if len(prob) > feature_topk else np.arange(len(prob))
    dist = np.sqrt(((pool_probs[:, active] - prob[active][None]) ** 2).sum(1))
    k = int(np.argsort(dist)[0])
    masks = np.asarray(pool_masks[k])[rng.choice(len(pool_masks[k]), n, replace=True)].astype(np.float32)
    masks[:, prob >= high] = 1
    masks[:, prob <= low] = 0
    return masks, keys[k]


class Generator:
    def __init__(self, run: Path, data: Data, mols, device: str, seed: int, sampling_seed: int):
        self.cfg = load_config(run / "config.json")
        self.data, self.mols, self.device, self.run = data, mols, device, run
        self.seed, self.sampling_seed = seed, sampling_seed
        self.ts = train_set(data, mols)
        saved = json.loads((run / "run.json").read_text())
        if abs(saved["x_max"] - self.ts.x_max) > 1e-6:
            raise SystemExit(f"x_max {self.ts.x_max} differs from training ({saved['x_max']})")
        self.diffusion = build_diffusion(self.cfg)
        edges, _ = gene_edges(self.cfg, self.ts)
        self.mask_model = build_mask(self.cfg, self.ts.X.shape[1], edges).to(device)
        self.mask_model.load_state_dict(torch.load(run / "mask_model.pt", map_location=device, weights_only=True))
        self.mask_model.eval()
        inf = self.cfg["inference"]
        self.n = inf["n_cells"]
        rng = np.random.default_rng([seed, sampling_seed, 7])
        ctrl_rows = np.flatnonzero(self.ts.is_ctrl)
        self.ctrl_pick = np.sort(rng.choice(ctrl_rows, self.n, replace=False))
        self.ctrl_x = torch.as_tensor(self.ts.X[self.ctrl_pick], device=device)
        # candidate zero patterns: every training condition, vehicle included (test.py cond_dict)
        self.pool = {}
        for c in pd.unique(self.ts.condition):
            nz = self.ts.X[self.ts.condition == c] != 0
            self.pool[c] = (nz, nz.mean(0))
        self.pool_keys = list(self.pool)
        self.pool_probs = np.stack([self.pool[k][1] for k in self.pool_keys])
        self.model = None

    def load(self, step: int) -> None:
        self.model = build_target(self.cfg, self.ts.X.shape[1]).to(self.device)
        self.model.load_state_dict(torch.load(self.run / f"ckpt_{step}.pt", map_location=self.device, weights_only=True))
        self.model.eval()
        zero = torch.zeros((self.n, 1), device=self.device)
        self.latent = self.diffusion.ddim_reverse(self.model, self.ctrl_x, {"dosage": zero})

    def sample_mask(self, prob: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        inf = self.cfg["inference"]
        return sample_mask(prob, self.pool_keys, self.pool_probs, [self.pool[k][0] for k in self.pool_keys], self.n,
                           inf["mask_high"], inf["mask_low"], inf["mask_feature_topk"], rng)

    @torch.no_grad()
    def condition(self, drug: str, dose_nM: float, condition: str) -> tuple[np.ndarray, dict]:
        mole = torch.as_tensor(self.mols.of(drug), device=self.device)[None].repeat(self.n, 1)
        dosage = torch.full((self.n, 1), float(dose_val(dose_nM)), device=self.device)
        x = self.diffusion.ddim_sample(self.model, self.latent, {"mole": mole, "dosage": dosage, "ctrl_mean": self.ctrl_x})
        prob = self.mask_model.predict(self.ctrl_x, mole, dosage).mean(0).cpu().numpy()
        rng = np.random.default_rng([self.seed, self.sampling_seed, zlib.crc32(condition.encode())])
        masks, key = self.sample_mask(prob, rng)
        cells = np.clip(x.cpu().numpy() * masks * self.ts.x_max, 0, None).astype(np.float32)
        return cells, {"mask_source": key, "n_forced_on": int((prob >= self.cfg["inference"]["mask_high"]).sum()),
                       "n_forced_off": int((prob <= self.cfg["inference"]["mask_low"]).sum())}

    def split(self, split: str) -> tuple[dict, dict]:
        cells, info = {}, {}
        for t in self.data.tasks(split):
            cells[t.target], info[t.target] = self.condition(t.drug, t.dose, t.target)
        return cells, info


def rmse_all(data: Data, cells: dict) -> float:
    return float(np.mean([np.sqrt(np.mean((data.mean(c) - X.mean(0, dtype=np.float64)) ** 2)) for c, X in cells.items()]))


def select(gen: Generator, run: Path) -> dict:
    """Score every saved checkpoint on validation; resumable (selection.jsonl)."""
    path = run / f"selection_s{gen.sampling_seed}.jsonl"
    rows = {r["step"]: r for r in read_jsonl(path)} if path.exists() else {}
    for step in ckpt_steps(run):
        if step in rows:
            continue
        gen.load(step)
        cells, _ = gen.split("val")
        rows[step] = {"step": step, "val_rmse_all": rmse_all(gen.data, cells)}
        with open(path, "a") as f:
            f.write(json.dumps(rows[step]) + "\n")
        print(f"checkpoint {step}: validation RMSE (All) {rows[step]['val_rmse_all']:.5f}")
    steps = ckpt_steps(run)
    best = min(steps, key=lambda s: (rows[s]["val_rmse_all"], s))
    out = {"final": steps[-1], "bestval": best, "curve": [rows[s] for s in steps]}
    write_json(run / f"selected_s{gen.sampling_seed}.json", out)
    return out


def write_preds(path: Path, cells: dict, gene_ids: list[str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conds = list(cells)
    X = np.concatenate([cells[c] for c in conds])
    obs = pd.DataFrame({"condition": np.repeat(conds, [len(cells[c]) for c in conds])})
    obs.index = obs.index.astype(str)
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(gene_ids, name="gene_id")))
    a.uns["generation"] = json.dumps(meta)
    tmp = path.with_name(path.stem + ".tmp.h5ad")
    a.write_h5ad(tmp)
    tmp.rename(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--molecules", required=True)
    p.add_argument("--data", default=DEFAULT_PANEL)
    p.add_argument("--preds-root", required=True)
    p.add_argument("--name", required=True, help="method prefix: doloris_<name>_<rule>")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--sampling-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    run = Path(a.run)
    if not (run / "training_complete.json").exists():
        raise SystemExit(f"{run}: training is not complete")
    seed = json.loads((run / "run.json").read_text())["seed"]
    data = Data(a.data)
    gen = Generator(run, data, load_molecules(a.molecules), a.device, seed, a.sampling_seed)
    chosen = select(gen, run)
    print(f"final = {chosen['final']}, bestval = {chosen['bestval']}")
    for rule in RULES:
        step = chosen[rule]
        loaded = False
        for split in a.splits:
            out = Path(a.preds_root) / f"doloris_{a.name}_{rule}" / f"seed_{seed}" / f"{split}_sample_s{a.sampling_seed}" / "preds.h5ad"
            if out.exists():
                print(f"{out} exists")
                continue
            if not loaded:
                gen.load(step)
                loaded = True
            cells, info = gen.split(split)
            write_preds(out, cells, data.spec.gene_ids, {"rule": rule, "step": step, "seed": seed, "sampling_seed": a.sampling_seed,
                                                         "n_cells": gen.n, "run": str(run)})
            write_json(out.parent / "generation.json", {"rule": rule, "step": step, "conditions": info,
                                                        "control_rows": gen.ctrl_pick.tolist()})
            print(f"wrote {out} (step {step}, rule {rule})")


if __name__ == "__main__":
    main()
