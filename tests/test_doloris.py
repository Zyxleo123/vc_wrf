"""Unit tests for the DOLORIS port (sciplex_doloris). Parity with the released code itself is
checked separately on a GPU by tests/parity_doloris.py (needs torch_geometric and the source)."""
import numpy as np
import pytest
import torch

from sciplex_doloris.data import coexpression_adjacency, dose_val
from sciplex_doloris.diffusion import SpacedDiffusion, linear_betas
from sciplex_doloris.generate import sample_mask
from sciplex_doloris.metrics import doloris_metrics, e_distance, emd, rmse_of_means
from sciplex_doloris.model import GATConv, MaskModel, TargetModel, graph_edges
from sciplex_interp.core import energy_distance


def test_schedule_is_ddim50_of_linear500():
    d = SpacedDiffusion(500, 50)
    assert d.timestep_map == list(range(0, 500, 10))
    base = np.cumprod(1 - np.linspace(2e-4, 4e-2, 500))
    np.testing.assert_allclose(linear_betas(500), np.linspace(2e-4, 4e-2, 500))
    np.testing.assert_allclose(d.acp, base[::10])
    assert d.acp_prev[0] == 1.0 and d.acp_next[-1] == 0.0


def test_eps_from_x0_inverts_q_sample():
    d = SpacedDiffusion()
    x0, noise = torch.rand(8, 5), torch.randn(8, 5)
    t = torch.arange(8) * 6
    np.testing.assert_allclose(d.eps_from_x0(d.q_sample(x0, t, noise), t, x0), noise, atol=1e-4)


def test_ddim_with_exact_x0_predictor_returns_x0():
    d = SpacedDiffusion()
    x0 = torch.rand(4, 6) * 2 - 0.5  # some values outside [-1, 1] are clipped by the sampler
    model = lambda x, t, **kw: x0.expand_as(x)
    latent = d.ddim_reverse(model, x0.clamp(-1, 1), {})
    np.testing.assert_allclose(d.ddim_sample(model, latent, {}), x0.clamp(-1, 1), atol=1e-5)


def test_masked_loss_ignores_zero_entries():
    d = SpacedDiffusion()
    x0 = torch.tensor([[0.0, 0.5, 0.0, 0.2]])
    t, noise = torch.tensor([3]), torch.zeros(1, 4)
    a = lambda x, ts, **kw: torch.tensor([[9.0, 0.4, -7.0, 0.2]])
    b = lambda x, ts, **kw: torch.tensor([[0.0, 0.4, 0.0, 0.2]])
    la, lb = d.training_loss(a, x0, {}, t=t, noise=noise), d.training_loss(b, x0, {}, t=t, noise=noise)
    assert la.item() == pytest.approx(lb.item()) == pytest.approx(0.01 / 2)


def test_gat_matches_dense_attention():
    torch.manual_seed(0)
    n, heads, c = 7, 2, 3
    adj = np.zeros((n, n), bool)
    for i, j in [(0, 1), (1, 2), (2, 5), (3, 4), (0, 6)]:
        adj[i, j] = True
    edges = graph_edges(adj)
    conv = GATConv(4, c, heads)
    x = torch.randn(n, 4)
    out = conv(x, edges[0], edges[1])
    h = conv.lin(x).view(n, heads, c)
    s = (h * conv.att_src).sum(-1)
    t = (h * conv.att_dst).sum(-1)
    full = adj | adj.T | np.eye(n, dtype=bool)
    ref = torch.zeros(n, heads, c)
    for i in range(n):
        nb = np.flatnonzero(full[:, i])  # sources j with an edge j -> i
        e = torch.nn.functional.leaky_relu(s[nb] + t[i], 0.2)
        ref[i] = (torch.softmax(e, 0)[..., None] * h[nb]).sum(0)
    np.testing.assert_allclose(out.detach(), (ref.reshape(n, -1) + conv.bias).detach(), atol=1e-6)


def test_target_model_routes_controls_and_adds_the_control_prior():
    torch.manual_seed(0)
    g = 12
    m = TargetModel(g, hidden_dim=16, time_pos_dim=8, embed_dim=6, mole_dim=5).eval()
    x, ts = torch.rand(4, g), torch.tensor([0, 10, 20, 30])
    dosage = torch.tensor([[0.0], [0.1], [0.0], [1.0]])
    mole = torch.randn(4, 5)
    prior = torch.rand(4, g)
    a = m(x, ts, mole=mole, dosage=dosage, ctrl_mean=prior)
    b = m(x, ts, mole=mole, dosage=dosage, ctrl_mean=prior + 1.0)
    np.testing.assert_allclose(a[[0, 2]].detach(), b[[0, 2]].detach())          # controls ignore the prior
    np.testing.assert_allclose((b - a)[[1, 3]].detach(), np.ones((2, g)), atol=1e-5)  # treated: + x_noisy
    c = m(x, ts, mole=torch.randn(4, 5), dosage=dosage, ctrl_mean=prior)
    np.testing.assert_allclose(a[[0, 2]].detach(), c[[0, 2]].detach())          # controls ignore the drug


def test_mask_model_runs_and_gives_probabilities():
    torch.manual_seed(0)
    g = 10
    adj = coexpression_adjacency(np.random.default_rng(0).random((60, g)).astype(np.float32), 0.1, 50, 0)
    m = MaskModel(g, graph_edges(adj), gene_init_dim=8, mole_dim=5)
    x = torch.rand(3, g) * (torch.rand(3, g) > 0.5)
    loss = m.loss(x, torch.randn(3, 5), torch.rand(3, 1), torch.rand(3, g), torch.rand(3, g) * 0.1)
    loss.backward()
    p = m.eval().predict(torch.rand(3, g), torch.randn(3, 5), torch.rand(3, 1))
    assert p.shape == (3, g) and float(p.min()) >= 0 and float(p.max()) <= 1


def test_sample_mask_nearest_pool_and_forced_genes():
    rng = np.random.default_rng(0)
    pools = [rng.random((20, 6)) > 0.9, rng.random((20, 6)) > 0.1]  # non-zero rates ~0.1 vs ~0.9
    probs = np.stack([p.mean(0) for p in pools])
    prob = probs[1].copy()
    prob[0], prob[5] = 0.995, 0.001
    masks, key = sample_mask(prob, ["a", "b"], probs, pools, 50, 0.99, 0.01, 300, rng)
    assert key == "b" and masks.shape == (50, 6)
    assert (masks[:, 0] == 1).all() and (masks[:, 5] == 0).all()
    rows = {tuple(r) for r in pools[1][:, 1:5].astype(np.float32)}
    assert all(tuple(r) in rows for r in masks[:, 1:5])


def test_doloris_metrics_definitions():
    rng = np.random.default_rng(1)
    real, pred = rng.normal(size=(40, 50)), rng.normal(0.3, 1.2, size=(70, 50))
    assert e_distance(pred, real) == pytest.approx(np.sqrt(energy_distance(pred, real)), rel=1e-9)
    assert e_distance(pred, real) == pytest.approx(e_distance(real, pred))
    assert rmse_of_means(real, pred) == pytest.approx(np.sqrt(np.mean((real.mean(0) - pred.mean(0)) ** 2)))
    from scipy.stats import wasserstein_distance
    assert emd(real, pred) == pytest.approx(np.mean([wasserstein_distance(real[:, i], pred[:, i]) for i in range(50)]))
    de = list(rng.permutation(50))
    m = doloris_metrics(real, pred, de)
    assert m["rmse_de20"] == pytest.approx(rmse_of_means(real[:, de[:20]], pred[:, de[:20]]))
    assert m["emd_de40"] == pytest.approx(emd(real[:, de[:40]], pred[:, de[:40]]))
    assert m["edist_all"] == pytest.approx(e_distance(pred, real))


def test_coexpression_adjacency_and_dose_scale():
    rng = np.random.default_rng(0)
    base = rng.random((600, 1))
    X = np.hstack([base, base + 0.01 * rng.random((600, 1)), rng.random((600, 1)), np.zeros((600, 1))]).astype(np.float32)
    adj = coexpression_adjacency(X, 0.3, 500, 42)
    assert adj[0, 1] and adj[1, 0] and not adj[0, 2] and adj.diagonal().all()
    assert not adj[3, :3].any()
    np.testing.assert_allclose(dose_val([10, 100, 1000, 10000]), [0.001, 0.01, 0.1, 1.0])
