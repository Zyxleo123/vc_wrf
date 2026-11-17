"""Apply DOLORIS's sparsity mask to Ours' generated cells (post-hoc), writing <base>_masked.

DOLORIS imposes sparsity with a separate module, not a loss inside its generator. A MaskModel
(binary cross-entropy over a gene-graph GAT, sciplex_standard.doloris.train) predicts a per-gene
P(non-zero) for a (drug, dose, cell line); at inference (sciplex_standard.doloris.Sampler) the
dense diffusion output is multiplied by a 0/1 mask built by sample_mask: the training condition
whose empirical non-zero rates are nearest (over the most likely genes) lends each generated cell
a real cell's zero pattern, and confident genes are forced on (>= high) or off (<= low). The
sparsity pattern depends only on the condition, not on which model drew the cells, so the very
same mask drops onto Ours' decoded cells.

For each seed we reuse that seed's trained mask model (results/standard/runs/doloris/seed_s/
mask_model.pt), the co-expression gene graph it was built on, the per-cell-line control draw and the
training-condition zero-pattern pool -- all exactly as sciplex_standard.doloris builds them -- and
mask the saved Ours cells (preds/<base>/seed_s/<split>.h5ad) condition by condition, writing
preds/<base>_masked/seed_s/<split>.h5ad. Ours' cells are already in the panel's log1p space and
non-negative, so the 0/1 mask multiplies them directly (no x_max: that factor only undoes DOLORIS's
internal 1/x_max scaling, which Ours never applied).

The masked cells are scored by sciplex_standard.evaluate exactly like any other method; <base>_masked
appears beside <base> in the report.

usage: python -m sciplex_standard.mask_ours --root results/standard
         [--bases ours ours_final] [--splits test ood val] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sciplex_doloris.data import load_molecules
from sciplex_doloris.generate import sample_mask
from sciplex_doloris.model import MaskModel

from .data import DEFAULT_FILE, REPO, StdData
from .doloris import Inputs, gene_graph

ROOT = REPO / "results" / "standard"
CONFIG = REPO / "configs" / "standard_doloris_v1.json"
BASES = ("ours", "ours_final")
SPLITS = ("test", "ood", "val")


def build_mask(cfg: dict, n_genes: int, n_cells: int, edges, device: str) -> MaskModel:
    k, m = cfg["mask"], cfg["model"]
    mask = MaskModel(n_genes, edges, gene_init_dim=k["gene_init_dim"], mole_dim=m["mole_dim"],
                     heads=k["gat_heads"], cell_type_num=n_cells, cell_type_embed_dim=k["cell_type_embed_dim"])
    return mask.to(device)


class Masker:
    """One seed's DOLORIS sparsity mask, ready to drop on any method's cells of a condition."""

    def __init__(self, cfg: dict, data: StdData, inp: Inputs, mols, run: Path, seed: int, device: str, offset: int = 0):
        self.cfg, self.data, self.inp, self.mols, self.device, self.seed = cfg, data, inp, mols, device, seed
        self.n_ctrl = cfg["inference"]["n_cells"]
        self.topk = cfg["inference"]["mask_feature_topk"]
        self.mask = build_mask(cfg, inp.X.shape[1], len(data.cell_types), gene_graph(cfg, inp), device)
        self.mask.load_state_dict(torch.load(run / "mask_model.pt", map_location=device, weights_only=True))
        self.mask.eval()
        rng = np.random.default_rng([seed, offset, 7])  # same control draw as doloris.Sampler
        self.ctrl = []
        for i in range(len(data.cell_types)):
            pick = np.sort(rng.choice(np.flatnonzero(inp.is_ctrl & (inp.cell == i)), self.n_ctrl, replace=False))
            self.ctrl.append(torch.as_tensor(inp.X[pick], device=device))
        keys = pd.unique(inp.key)  # mask pool: every training (cell-line) condition and the vehicles
        self.pool_keys = list(keys)
        self.pool_masks = [inp.X[inp.key == k] != 0 for k in keys]
        self.pool_probs = np.stack([m.mean(0) for m in self.pool_masks])
        self._prob: dict[str, np.ndarray] = {}  # P(non-zero) depends only on the condition, not the method

    @torch.no_grad()
    def predict_prob(self, key: str) -> np.ndarray:
        if key not in self._prob:
            c = self.data.cond.loc[key]
            i = self.data.cell_types.index(c.cell_type)
            mole = torch.as_tensor(self.mols.of(c.drug), device=self.device)[None].repeat(self.n_ctrl, 1)
            dosage = torch.full((self.n_ctrl, 1), float(c.dose_val), device=self.device)
            ct = torch.full((self.n_ctrl,), i, device=self.device)
            self._prob[key] = self.mask.predict(self.ctrl[i], mole, dosage, cell_type=ct).mean(0).cpu().numpy()
        return self._prob[key]

    def mask_condition(self, key: str, split: str, cells: np.ndarray) -> tuple[np.ndarray, dict]:
        """`cells`: Ours' generated cells of `key`, (n, genes) in data.genes order, log1p, >= 0."""
        prob = self.predict_prob(key)
        high, low = (0.95, 0.05) if split == "ood" else (0.99, 0.01)
        rng = np.random.default_rng([self.seed, zlib.crc32(key.encode())])
        masks, src = sample_mask(prob, self.pool_keys, self.pool_probs, self.pool_masks, len(cells),
                                 high, low, self.topk, rng)
        out = np.clip(cells * masks, 0, None).astype(np.float32)
        info = {"mask_source": src, "n_forced_on": int((prob >= high).sum()), "n_forced_off": int((prob <= low).sum()),
                "nz_frac_before": float((cells > 0).mean()), "nz_frac_after": float((out > 0).mean())}
        return out, info


def read_cells(path: Path, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import anndata as ad
    a = ad.read_h5ad(path)
    pos = {g: i for i, g in enumerate(map(str, a.var_names))}
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.asarray(X[:, [pos[g] for g in genes]], dtype=np.float32)
    return a.obs["condition"].astype(str).to_numpy(), X


def write_preds(path: Path, cells: dict, keys: list[str], genes: list[str], meta: dict) -> None:
    import anndata as ad
    labels = np.concatenate([[k] * len(cells[k]) for k in keys])
    a = ad.AnnData(np.vstack([cells[k] for k in keys]),
                   obs=pd.DataFrame({"condition": labels}, index=[f"g{i}" for i in range(len(labels))]),
                   var=pd.DataFrame(index=genes))
    a.uns["masking"] = json.dumps(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.h5ad")
    a.write_h5ad(tmp, compression="gzip")
    tmp.rename(path)


def run(root: Path, bases, splits, seeds, path: str, mol_path: str, device: str) -> None:
    cfg = json.loads(CONFIG.read_text())
    data = StdData(path)
    mols = load_molecules(mol_path)
    inp = Inputs(data, mols)
    for seed in seeds:
        run_dir = root / "runs" / "doloris" / f"seed_{seed}"
        if not (run_dir / "mask_model.pt").exists():
            raise SystemExit(f"{run_dir}/mask_model.pt missing; train DOLORIS (sciplex_standard.doloris) first")
        masker = Masker(cfg, data, inp, mols, run_dir, seed, device)
        for base in bases:
            for split in splits:
                src = root / "preds" / base / f"seed_{seed}" / f"{split}.h5ad"
                dest = root / "preds" / f"{base}_masked" / f"seed_{seed}" / f"{split}.h5ad"
                if not src.exists():
                    print(f"skip (no source cells): {src}", flush=True)
                    continue
                if dest.exists():
                    print(f"exists: {dest}", flush=True)
                    continue
                labels, X = read_cells(src, data.genes)
                keys = data.keys(split)
                missing = [k for k in keys if not (labels == k).any()]
                if missing:
                    raise SystemExit(f"{src}: no cells for {len(missing)} {split} conditions, e.g. {missing[:3]}")
                cells, info = {}, {}
                for k in keys:
                    cells[k], info[k] = masker.mask_condition(k, split, X[labels == k])
                before = float(np.mean([info[k]["nz_frac_before"] for k in keys]))
                after = float(np.mean([info[k]["nz_frac_after"] for k in keys]))
                write_preds(dest, cells, keys, data.genes,
                            {"base": base, "seed": seed, "split": split, "mask_run": str(run_dir),
                             "nz_frac_before": before, "nz_frac_after": after, "conditions": info})
                print(f"{base}_masked seed {seed} {split}: {len(keys)} conditions, "
                      f"non-zero fraction {before:.3f} -> {after:.3f} -> {dest}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--molecules", default=str(ROOT / "unimol_embed.npz"))
    p.add_argument("--bases", nargs="+", default=list(BASES))
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    run(Path(a.root), a.bases, a.splits, a.seeds, a.file, a.molecules, a.device)


if __name__ == "__main__":
    main()
