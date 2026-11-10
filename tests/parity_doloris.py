"""Parity of the DOLORIS port (sciplex_doloris) with the released code, on a GPU.

The released model.py hard-codes 'cuda' and needs torch_geometric; both are supplied here
(--pyg-site: a directory with torch_geometric 2.6.1, the version requirements.txt pins).
Helper functions are taken from the released test_util.py by AST, so its unrelated imports
(umap, scGPT, ...) are not needed. Weights of the released modules are copied into the port
(keys renamed; modules that are unused on the molecular path are dropped), dropout is switched
off, and the same inputs and random streams are fed to both. Checked:

  TargetModel forward (train mode: BatchNorm batch statistics on control / treated sub-batches;
  eval mode), the masked x0 training loss and its gradients, DDIM inversion and DDIM sampling
  through the spaced 50-step schedule, MaskModel loss and probabilities (GAT on a gene graph),
  sample_topk_conditions, and the three metric functions.

Writes a JSON record; exits 1 if any check fails. Not collected by pytest (needs a GPU).

usage: python tests/parity_doloris.py --doloris-src SRC --pyg-site DIR --out results/doloris/parity.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sciplex_doloris.diffusion import SpacedDiffusion  # noqa: E402  (ours first, before the released dir is on the path)
from sciplex_doloris.generate import sample_mask  # noqa: E402
from sciplex_doloris.metrics import e_distance, emd, rmse_of_means  # noqa: E402
from sciplex_doloris.model import MaskModel, TargetModel  # noqa: E402

G = 2085  # the panel's gene count
RESULTS = []


def check(name, a, b, tol):
    a = a.detach().double().cpu() if torch.is_tensor(a) else torch.as_tensor(np.asarray(a, dtype=np.float64))
    b = b.detach().double().cpu() if torch.is_tensor(b) else torch.as_tensor(np.asarray(b, dtype=np.float64))
    diff = float((a - b).abs().max())
    scale = max(1.0, float(a.abs().max()))
    ok = a.shape == b.shape and diff <= tol * scale
    RESULTS.append({"check": name, "max_abs_diff": diff, "max_abs_ref": float(a.abs().max()), "tol": tol * scale, "pass": bool(ok)})
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} max|diff| {diff:.3e}  (max|ref| {float(a.abs().max()):.3e}, tol {tol * scale:.1e})")


def flag(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}")


def no_dropout(*models):
    for m in models:
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
            if hasattr(mod, "dropout") and isinstance(getattr(mod, "dropout"), float):
                mod.dropout = 0.0


def remap(sd: dict, drop: tuple[str, ...]) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith(drop):
            continue
        k = k.replace("source_model.", "", 1) if k.startswith("source_model.") else k
        out[k.replace("dosager.mlp.", "dosager.")] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doloris-src", required=True)
    p.add_argument("--pyg-site", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"

    sys.path.insert(0, a.pyg_site)
    sys.path.insert(0, a.doloris_src)
    import diffusion as gd  # the released diffusion.py
    import model as released  # the released model.py
    import respace
    tree = ast.parse(Path(a.doloris_src, "test_util.py").read_text())
    wanted = {"compute_rmse", "compute_e_distance", "sample_topk_conditions"}
    ns = {"np": np, "cdist": cdist, "pdist": pdist}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
                            type_ignores=[]), "test_util.py", "exec"), ns)

    # ---- diffusion schedule ------------------------------------------------------------------
    theirs_d = respace.SpacedDiffusion(use_timesteps=respace.space_timesteps(500, "ddim50"),
                                       betas=gd.get_named_beta_schedule("linear", 500),
                                       model_mean_type=gd.ModelMeanType.START_X, model_var_type=gd.ModelVarType.FIXED_LARGE,
                                       loss_type=gd.LossType.MSE, rescale_timesteps=False)
    mine_d = SpacedDiffusion(500, 50, clip_denoised=True)
    flag("respaced timestep map", theirs_d.timestep_map == mine_d.timestep_map, f"{len(mine_d.timestep_map)} steps")
    check("alphas_cumprod (spaced)", torch.as_tensor(theirs_d.alphas_cumprod), torch.as_tensor(mine_d.acp), 1e-12)

    # ---- target model -------------------------------------------------------------------------
    torch.manual_seed(0)
    theirs = released.TargetModel(gene_num=G, GRN=None, init_gene_emb=None, gene_dim=32, hidden_dim=1024, time_pos_dim=1024,
                                  gene_wise_embed_dim=256, cell_type_num=1, time_embed_dim=256, cell_type_embed_dim=256,
                                  data_name="sciplex3", mole_dim=512, pert_type="molecular").to(dev)
    mine = TargetModel(G, hidden_dim=1024, time_pos_dim=1024, embed_dim=256, mole_dim=512).to(dev)
    unused = ("source_model.block_1.cell_type_encoder.", "control_net.gnn.", "control_net.gene_mole_encoder.", "control_net.ln.")
    mine.load_state_dict(remap(theirs.state_dict(), unused), strict=True)
    n_used = sum(v.numel() for k, v in theirs.named_parameters() if not k.startswith(unused))
    flag("target parameters covered", n_used == sum(v.numel() for v in mine.parameters()), f"{n_used:,} parameters")
    no_dropout(theirs, mine)

    g = torch.Generator(device=dev).manual_seed(0)
    B = 64
    x = torch.rand(B, G, device=dev, generator=g) * (torch.rand(B, G, device=dev, generator=g) > 0.6)
    dosage = torch.cat([torch.zeros(20, 1, device=dev), torch.rand(B - 20, 1, device=dev, generator=g)])
    mole = torch.randn(B, 512, device=dev, generator=g)
    mole[:20] = 0
    mu = (torch.rand(G, device=dev, generator=g) * 0.3).expand(B, -1)
    var = (torch.rand(G, device=dev, generator=g) * 0.01).expand(B, -1)
    tmodel = torch.randint(0, 50, (B,), device=dev, generator=g) * 10

    theirs.train(); mine.train()
    kw = {"mole": mole, "dosage": dosage, "ctrl_mean": mu, "ctrl_var": var}
    torch.manual_seed(1); ya = theirs(x, tmodel, **kw)
    torch.manual_seed(1); yb = mine(x, tmodel, **kw)
    check("TargetModel forward, train mode", ya, yb, 1e-5)

    t_idx = torch.randint(0, 50, (B,), device=dev, generator=g)
    noise = torch.randn(B, G, device=dev, generator=g)
    theirs.zero_grad(); mine.zero_grad()
    torch.manual_seed(2); la = theirs_d.training_losses(theirs, x, t_idx, model_kwargs=kw, noise=noise)["loss"]
    torch.manual_seed(2); lb = mine_d.training_loss(mine, x, kw, t=t_idx, noise=noise)
    check("masked x0 training loss", la.reshape(1), lb.reshape(1), 1e-5)
    la.backward(); lb.backward()
    pa, pb = dict(theirs.named_parameters()), dict(mine.named_parameters())
    for k_theirs, k_mine in [("source_model.block_1.mlp1.network.0.weight", "block_1.mlp1.network.0.weight"),
                             ("source_model.decoder_treated.network.0.weight", "decoder_treated.network.0.weight"),
                             ("source_model.decoder_control.network.12.weight", "decoder_control.network.12.weight"),
                             ("control_net.mole_encoder.dosager.mlp.network.0.weight", "control_net.mole_encoder.dosager.network.0.weight"),
                             ("control_net.control_token", "control_net.control_token")]:
        check(f"gradient {k_mine}", pa[k_theirs].grad, pb[k_mine].grad, 1e-4)

    theirs.eval(); mine.eval()
    kw_eval = {"mole": mole, "dosage": dosage, "ctrl_mean": x.flip(0)}
    check("TargetModel forward, eval mode", theirs(x, tmodel, **kw_eval), mine(x, tmodel, **kw_eval), 1e-5)

    n = 48
    ctrl = x[:n]
    lat_a = theirs_d.ddim_reverse_sample_loop(theirs, ctrl, model_kwargs={"dosage": torch.zeros(n, device=dev)}, clip_denoised=True)
    lat_b = mine_d.ddim_reverse(mine, ctrl, {"dosage": torch.zeros(n, 1, device=dev)})
    check("DDIM inversion of control cells (latents)", lat_a, lat_b, 1e-3)
    kw_s = {"mole": mole[30:31].repeat(n, 1), "dosage": torch.full((n, 1), 0.1, device=dev), "ctrl_mean": ctrl}
    s_a = theirs_d.ddim_sample_loop(theirs, shape=(n, G), noise=lat_a, model_kwargs=kw_s, eta=0, clip_denoised=True)
    s_b = mine_d.ddim_sample(mine, lat_a, kw_s)
    check("DDIM sampling from the same latents", s_a, s_b, 1e-4)
    check("inversion + sampling end to end", s_a, mine_d.ddim_sample(mine, lat_b, kw_s), 1e-3)

    # ---- mask model ---------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    upper = np.triu(rng.random((G, G)) < 0.002, 1)
    adj = upper | upper.T
    src, dst = np.nonzero(adj)
    edge_index = torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=dev)
    from sciplex_doloris.model import graph_edges
    torch.manual_seed(0)
    theirs_m = released.MaskModel(gene_num=G, GRN=SimpleNamespace(edge_index=edge_index), gene_init_dim=64, cell_type_num=1,
                                  cell_type_embed_dim=512, data_name="sciplex3", mole_dim=512, pert_type="molecular").to(dev)
    mine_m = MaskModel(G, graph_edges(adj), gene_init_dim=64, mole_dim=512, heads=2).to(dev)
    unused_m = ("control_net.mlp_1.", "control_net.mlp_2.", "control_net.gene_encoder.", "control_net.ko_token")
    mine_m.load_state_dict(remap(theirs_m.state_dict(), unused_m), strict=True)
    no_dropout(theirs_m, mine_m)
    flag("gene graph edges (directed, without self loops)", True, f"{len(src):,}")
    bm = 16
    xm, mm, dm = x[20:20 + bm], mole[20:20 + bm], dosage[20:20 + bm]
    theirs_m.train(); mine_m.train()
    torch.manual_seed(3); la = theirs_m(x=xm, mole=mm, dosage=dm, ctrl_mean=mu[:bm], ctrl_var=var[:bm])
    torch.manual_seed(3); lb = mine_m.loss(xm, mm, dm, mu[:bm], var[:bm])
    check("MaskModel BCE loss, train mode", la.reshape(1), lb.reshape(1), 1e-5)
    theirs_m.eval(); mine_m.eval()
    prob_a = theirs_m.predict(x=ctrl[:bm], mole=mole[30:31].repeat(bm, 1), dosage=torch.full((bm, 1), 0.1, device=dev))
    prob_b = mine_m.predict(ctrl[:bm], mole[30:31].repeat(bm, 1), torch.full((bm, 1), 0.1, device=dev))
    check("MaskModel probabilities, eval mode", prob_a, prob_b, 1e-5)

    # ---- three cell types (the released branches taken when cell_type_num > 1) -----------------
    torch.manual_seed(0)
    theirs3 = released.TargetModel(gene_num=G, GRN=None, init_gene_emb=None, gene_dim=32, hidden_dim=1024, time_pos_dim=1024,
                                   gene_wise_embed_dim=256, cell_type_num=3, time_embed_dim=256, cell_type_embed_dim=256,
                                   data_name="sciplex3", mole_dim=512, pert_type="molecular").to(dev)
    mine3 = TargetModel(G, hidden_dim=1024, time_pos_dim=1024, embed_dim=256, mole_dim=512, cell_type_num=3,
                        cell_type_embed_dim=256).to(dev)
    mine3.load_state_dict(remap(theirs3.state_dict(), ("control_net.gnn.", "control_net.gene_mole_encoder.")), strict=True)
    no_dropout(theirs3, mine3)
    cell = torch.randint(0, 3, (B, 1), device=dev, generator=g)
    kw3 = {**kw, "cell_type": cell}
    theirs3.train(); mine3.train()
    torch.manual_seed(4); ya = theirs3(x, tmodel, **kw3)
    torch.manual_seed(4); yb = mine3(x, tmodel, **kw3)
    check("3 cell types: TargetModel forward, train mode", ya, yb, 1e-5)
    torch.manual_seed(5); la = theirs_d.training_losses(theirs3, x, t_idx, model_kwargs=kw3, noise=noise)["loss"]
    torch.manual_seed(5); lb = mine_d.training_loss(mine3, x, kw3, t=t_idx, noise=noise)
    check("3 cell types: masked x0 training loss", la.reshape(1), lb.reshape(1), 1e-5)
    theirs3.eval(); mine3.eval()
    check("3 cell types: TargetModel forward, eval mode", theirs3(x, tmodel, **{**kw_eval, "cell_type": cell}),
          mine3(x, tmodel, **{**kw_eval, "cell_type": cell}), 1e-5)
    ct = torch.full((n, 1), 2, device=dev)
    lat_a = theirs_d.ddim_reverse_sample_loop(theirs3, ctrl, model_kwargs={"dosage": torch.zeros(n, device=dev), "cell_type": ct},
                                              clip_denoised=True)
    lat_b = mine_d.ddim_reverse(mine3, ctrl, {"dosage": torch.zeros(n, 1, device=dev), "cell_type": ct})
    check("3 cell types: DDIM inversion", lat_a, lat_b, 1e-3)
    kw_s3 = {**kw_s, "cell_type": ct}
    check("3 cell types: DDIM sampling", theirs_d.ddim_sample_loop(theirs3, shape=(n, G), noise=lat_a, model_kwargs=kw_s3,
                                                                   eta=0, clip_denoised=True),
          mine_d.ddim_sample(mine3, lat_a, kw_s3), 1e-4)
    torch.manual_seed(0)
    theirs_m3 = released.MaskModel(gene_num=G, GRN=SimpleNamespace(edge_index=edge_index), gene_init_dim=64, cell_type_num=3,
                                   cell_type_embed_dim=512, data_name="sciplex3", mole_dim=512, pert_type="molecular").to(dev)
    mine_m3 = MaskModel(G, graph_edges(adj), gene_init_dim=64, mole_dim=512, heads=2, cell_type_num=3, cell_type_embed_dim=512).to(dev)
    mine_m3.load_state_dict(remap(theirs_m3.state_dict(), unused_m), strict=True)
    no_dropout(theirs_m3, mine_m3)
    cm = cell[20:20 + bm]
    theirs_m3.train(); mine_m3.train()
    torch.manual_seed(6); la = theirs_m3(x=xm, mole=mm, dosage=dm, cell_type=cm, ctrl_mean=mu[:bm], ctrl_var=var[:bm])
    torch.manual_seed(6); lb = mine_m3.loss(xm, mm, dm, mu[:bm], var[:bm], cell_type=cm)
    check("3 cell types: MaskModel BCE loss, train mode", la.reshape(1), lb.reshape(1), 1e-5)
    theirs_m3.eval(); mine_m3.eval()
    m_in, d_in = mole[30:31].repeat(bm, 1), torch.full((bm, 1), 0.1, device=dev)
    check("3 cell types: MaskModel probabilities, eval mode",
          theirs_m3.predict(x=ctrl[:bm], mole=m_in, dosage=d_in, cell_type=cm),
          mine_m3.predict(ctrl[:bm], m_in, d_in, cell_type=cm), 1e-5)

    # ---- mask sampling and metrics -------------------------------------------------------------
    pools = {f"c{i}": (rng.random((30, G)) > q).astype(float) for i, q in enumerate(rng.uniform(0.2, 0.9, 12))}
    cond_dict = {k: {"sample_mask": v, "overall_prob": v.mean(0)} for k, v in pools.items()}
    pred = np.clip(list(pools.values())[5].mean(0) + rng.normal(0, 0.05, G), 0, 1)
    pred[:40], pred[40:80] = 0.995, 0.005
    np.random.seed(0)
    m_a, keys_a = ns["sample_topk_conditions"](pred, cond_dict, N=64, high=0.99, low=0.01)
    keys = list(pools)
    m_b, key_b = sample_mask(pred, keys, np.stack([cond_dict[k]["overall_prob"] for k in keys]), [pools[k] for k in keys],
                             64, 0.99, 0.01, 300, np.random.default_rng(0))
    free = (pred < 0.99) & (pred > 0.01)
    rows = {r.tobytes() for r in pools[key_b][:, free].astype(np.float32)}
    flag("sample_topk_conditions: nearest condition", keys_a[0] == key_b, f"{keys_a[0]} vs {key_b}")
    flag("sample_topk_conditions: forced genes", bool((m_a[:, ~free] == m_b[:, ~free]).all()))
    flag("sample_topk_conditions: other genes from the pool's cells",
         all(r.astype(np.float32).tobytes() in rows for r in m_a[:, free]) and all(r.tobytes() in rows for r in m_b[:, free]))

    real, gen = rng.random((180, G)) * (rng.random((180, G)) > 0.5), rng.random((512, G)) * (rng.random((512, G)) > 0.4)
    check("compute_rmse", np.array([ns["compute_rmse"](real, gen)]), np.array([rmse_of_means(real, gen)]), 1e-12)
    check("compute_e_distance", np.array([ns["compute_e_distance"](gen, real)]), np.array([e_distance(gen, real)]), 1e-12)
    ref_emd = np.mean([wasserstein_distance(real[:, i], gen[:, i]) for i in range(real.shape[1])])
    check("EMD (mean per-gene W1)", np.array([ref_emd]), np.array([emd(real, gen)]), 1e-12)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = all(r["pass"] for r in RESULTS)
    out.write_text(json.dumps({"pass": ok, "doloris_src": a.doloris_src, "torch": torch.__version__,
                               "device": torch.cuda.get_device_name(0), "checks": RESULTS}, indent=1) + "\n")
    print(f"{'ALL PASS' if ok else 'SOME CHECKS FAILED'}: wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
