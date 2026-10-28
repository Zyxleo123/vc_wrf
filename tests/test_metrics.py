from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sciplex_eval import EvalSpec, evaluate_condition, mmd2_rbf, response_score, weak_responder_fraction
from sciplex_eval.metrics import summarize
from sciplex_eval.spec import program_activity

REPO = Path(__file__).resolve().parents[1]
PANEL = Path("/data/user_data/yixiz/sciplex3/processed/a549_hdaci_v1.h5ad")


def toy_spec(score_type="log_sum", G=6):
    rng = np.random.default_rng(0)
    comps = np.linalg.qr(rng.normal(size=(G, 3)))[0].T
    return EvalSpec(
        version="toy", cell_line="X", gene_ids=[f"g{i}" for i in range(G)],
        gene_symbols=[f"G{i}" for i in range(G)], program_name="toy", program_gene_ids=["g1", "g3"],
        score_type=score_type, mu0=1.0, s0=0.5, quantiles=[0.9, 0.95], taus=[0.5, 1.0], primary_q=0.95,
        deg_top_k=2, mmd_n_max=500, mmd_sigmas=[0.5, 1.0, 2.0],
        pca_mean=np.zeros(G), pca_components=comps, control_mean=np.zeros(G))


def cells(rng, n, G=6, shift=0.0):
    return np.clip(rng.normal(1.0 + shift, 0.5, size=(n, G)), 0, None)


def test_program_activity_and_response_score():
    spec = toy_spec("log_sum")
    X = np.array([[9, 1.0, 9, 2.0, 9, 9], [0, -3.0, 0, 0.0, 0, 0]])
    a = program_activity(X, spec)
    assert np.allclose(a, [np.log1p(np.expm1(1.0) + np.expm1(2.0)), 0.0])  # negatives clipped
    assert np.allclose(response_score(X, spec), (1.0 - a) / (0.5 + 1e-6))
    spec_m = toy_spec("mean_log")
    assert np.allclose(program_activity(X, spec_m), [1.5, 0.0])
    assert np.allclose(program_activity(sp.csr_matrix(X), spec), a)


def test_weak_responder_fraction_strict():
    r = np.array([0.0, 0.5, 1.0, 2.0])
    assert np.allclose(weak_responder_fraction(r, [0.5, 1.0, 5.0]), [0.25, 0.5, 1.0])


def test_mmd_same_vs_shifted():
    rng = np.random.default_rng(1)
    A, B, C = rng.normal(size=(300, 3)), rng.normal(size=(400, 3)), rng.normal(0.7, 1, size=(300, 3))
    same, shifted = mmd2_rbf(A, B, [1.0]), mmd2_rbf(A, C, [1.0])
    assert abs(same) < 0.01 and shifted > 10 * abs(same)


def test_identical_population_scores_zero():
    spec = toy_spec()
    X = cells(np.random.default_rng(2), 200)
    rec = evaluate_condition(spec, X, X_pred=X.copy())
    for k in ("w1_r", "rmse_mean_all", "rmse_mean_deg2", "rmse_mean_program", "wrf_err_pp_q0.95"):
        assert rec[k] == pytest.approx(0, abs=1e-12)
    assert rec["mmd2"] <= 1e-9  # unbiased estimator on identical samples is <= 0


def test_shift_increases_errors_and_wrf_only_path():
    spec = toy_spec()
    rng = np.random.default_rng(3)
    real, near, far = cells(rng, 300), cells(rng, 300), cells(rng, 300, shift=-0.6)
    a, b = evaluate_condition(spec, real, X_pred=near), evaluate_condition(spec, real, X_pred=far)
    for k in ("w1_r", "mmd2", "rmse_mean_all", "wrf_err_pp_q0.95"):
        assert b[k] > a[k]
    f = {q: b[f"wrf_pred_q{q:g}"] for q in spec.quantiles}
    c = evaluate_condition(spec, real, wrf_pred=f)
    assert c["wrf_err_pp_q0.95"] == pytest.approx(b["wrf_err_pp_q0.95"])
    assert "mmd2" not in c


def test_spec_roundtrip_and_align(tmp_path):
    spec = toy_spec()
    spec.save(tmp_path / "s")
    s2 = EvalSpec.load(tmp_path / "s")
    assert s2.gene_ids == spec.gene_ids and np.allclose(s2.pca_components, spec.pca_components)
    X = np.arange(12.0).reshape(2, 6)
    perm = [5, 0, 3, 1, 4, 2]
    Xp = X[:, perm]
    assert np.allclose(s2.align(Xp, [spec.gene_ids[i] for i in perm]), X)
    with pytest.raises(ValueError):
        s2.align(X[:, :5], spec.gene_ids[:5])


def test_summarize():
    recs = [{"wrf_err_pp_q0.95": v, "mmd2": 0.0} for v in (1.0, 2.0, 3.0)]
    s = summarize(recs)
    assert s["n_conditions"] == 3 and s["wrf_err_pp_q0.95"]["mean"] == pytest.approx(2.0)
    lo, hi = s["wrf_err_pp_q0.95"]["ci95"]
    assert 1.0 <= lo <= 2.0 <= hi <= 3.0


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built / data not mounted")
def test_real_spec_controls():
    import anndata as ad
    spec = EvalSpec.load(REPO / "artifacts" / "eval_spec_v1")
    p = ad.read_h5ad(PANEL)
    assert list(p.var_names) == spec.gene_ids
    ctrl = (p.obs["drug"] == "Vehicle").values
    f = weak_responder_fraction(response_score(p.X[ctrl], spec), spec.taus)
    for q, fq in zip(spec.quantiles, f):
        assert fq <= q + 1e-9  # equality unless tau_q falls on a tie (point mass at zero program UMIs)
    assert f[spec.quantiles.index(0.95)] == pytest.approx(0.95, abs=0.002)
    assert set(p.obs["split"]) == {"control", "train", "val", "test"}
    held = p.obs[p.obs.split.isin(["val", "test"])]
    assert set(held["dose"]) <= {100.0, 1000.0}
    assert held.groupby("drug", observed=True)["condition"].nunique().max() == 1
