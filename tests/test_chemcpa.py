"""Behavioral checks for training, chemical conditioning and held-out isolation."""
import argparse
import copy
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from sciplex_baselines.chemcpa import (
    FractionLoss, cell_inputs, decode, evaluate_run, fraction_conditions, generate, make_model,
    minibatches, n_features, prediction_inputs, train_one, training_arrays, validate_panel,
)
from sciplex_baselines.fraction_loss import expected_soft_indicator, supervised_taus
from sciplex_eval.spec import EPS
from sciplex_baselines.chemistry import molecular_inputs
from test_metrics import toy_spec


FRACTION = {"weight": 10.0, "h": 0.25, "conditions_per_step": 2, "cells_per_condition": 8, "max_count": 64}


def small_config(likelihood="gauss", fraction=None):
    extra = {} if fraction is None else {"fraction_loss": fraction}
    return {**extra, "likelihood": likelihood, "epochs": 2, "validate_every": 1, "batch_size": 16,
            "n_prediction_cells": 20, "generation": "sample", "selection_metric": "mmd2",
            "hparams": {"dim": 4, "autoencoder_width": 8, "autoencoder_depth": 1,
                        "adversary_width": 8, "adversary_depth": 1,
                        "embedding_encoder_width": 8, "embedding_encoder_depth": 1,
                        "dosers_width": 8, "dosers_depth": 1}}


def fixture_panel():
    rng = np.random.default_rng(7)
    obs, conditions = [], []
    for split, dose in (("control", 0), ("train", 10), ("val", 100), ("test", 1000)):
        cond = f"X|{'Vehicle' if split == 'control' else 'DrugA'}|{dose}"
        for i in range(20):
            obs.append({"condition": cond, "drug": "Vehicle" if split == "control" else "DrugA",
                        "split": split, "dose": dose, "replicate": f"rep{1 + i % 2}",
                        "culture_plate": "plate", "well_oligo": f"{cond}_{i % 2}"})
        conditions.append({"condition": cond, "drug": obs[-1]["drug"], "dose_nM": dose,
                           "split": split, "n_cells": 20})
    counts = rng.poisson(2, (80, 6)).astype(np.float32)
    library = counts.sum(axis=1) + rng.integers(20, 40, size=80)
    panel = ad.AnnData(np.log1p(counts / library[:, None] * 1e4).astype(np.float32),
                       obs=pd.DataFrame(obs, index=[f"cell{i}" for i in range(80)]).assign(n_counts_human=library),
                       var=pd.DataFrame(index=[f"g{i}" for i in range(6)]), layers={"counts": counts})
    manifest = {"conditions": conditions[1:], "controls": conditions[0]}
    return panel, manifest


def flat(cells, idx=None, doses=None):
    return [v for v in (cells.features, cells.expression, cells.counts, cells.library, idx, doses) if v is not None]


@pytest.mark.parametrize("likelihood", ["gauss", "nb"])
def test_training_and_generation_never_use_heldout_expression(likelihood):
    torch.set_num_threads(1)
    panel, manifest = fixture_panel()
    spec = toy_spec()
    validate_panel(panel, spec, manifest)
    drugs = ["DrugA", "Vehicle"]
    original = flat(*training_arrays(panel, spec, drugs, likelihood))
    controls, conds = prediction_inputs(panel, spec, "test", likelihood)
    changed = panel.copy()
    heldout = changed.obs.split.isin(["val", "test"]).to_numpy()
    changed.X[heldout] = np.nan
    changed.layers["counts"][heldout] = 1e6
    changed.obs.loc[heldout, "n_counts_human"] = 1
    altered = flat(*training_arrays(changed, spec, drugs, likelihood))
    assert len(original) == (4 if likelihood == "gauss" else 6)
    for a, b in zip(original, altered):
        np.testing.assert_array_equal(a, b)
    c2, conds2 = prediction_inputs(changed, spec, "test", likelihood)
    for a, b in zip(flat(controls), flat(c2)):
        np.testing.assert_array_equal(a, b)
    assert conds == conds2
    model = make_model(n_features(spec, likelihood), np.eye(2, dtype=np.float32), small_config(likelihood), 0, "cpu")
    a = generate(model, controls, conds, drugs, spec.gene_ids, n_cells=21, seed=3)
    b = generate(model, c2, conds2, drugs, spec.gene_ids, n_cells=21, seed=3)
    np.testing.assert_array_equal(a.X, b.X)
    assert a.shape == (21, 6) and np.isfinite(a.X).all()
    if likelihood == "nb":
        assert (a.X >= 0).all() and (a.X == 0).any()
    changed.obs.loc[changed.obs.split == "test", "split"] = "train"
    with pytest.raises(ValueError, match="frozen split"):
        validate_panel(changed, spec, manifest)


def test_nb_inputs_match_normalization_and_decoding():
    panel, _ = fixture_panel()
    spec = toy_spec()
    mask = panel.obs.split.eq("control").to_numpy()
    cells = cell_inputs(panel, spec, mask, "nb")
    assert cells.features.shape == (20, 7) and cells.counts.shape == (20, 7)
    np.testing.assert_allclose(cells.counts.sum(axis=1), cells.library)
    np.testing.assert_allclose(cells.features[:, :6], panel.X[mask])
    np.testing.assert_allclose(cells.features[:, 6], np.log1p(cells.counts[:, 6] / cells.library * 1e4), rtol=1e-6)
    broken = panel.copy()
    broken.obs["n_counts_human"] = broken.obs["n_counts_human"] * 2
    with pytest.raises(ValueError, match="log1p"):
        cell_inputs(broken, spec, mask, "nb")
    p = np.array([[0.1, 0.2, 0.7], [0.0, 0.0, 1.0]])
    mean, theta, rng = 50 * p, np.full_like(p, 1e3), np.random.default_rng(0)
    np.testing.assert_allclose(decode(mean, theta, "nb", "mean", rng), np.log1p(p[:, :2] * 1e4))
    draws = decode(mean, theta, "nb", "sample", rng)
    counts = np.expm1(draws[0]) / 1e4  # fractions of the generated total
    assert draws.shape == (2, 2) and np.isfinite(draws).all() and (draws[1] == 0).all()
    assert (counts >= 0).all() and counts.sum() < 1
    zero = decode(np.zeros((1, 3)), np.ones((1, 3)), "nb", "sample", rng)
    assert np.array_equal(zero, np.zeros((1, 2)))


def test_expected_soft_indicator_matches_count_sampling():
    spec = toy_spec()  # program genes g1, g3; mu0 1, s0 0.5, taus 0.5 / 1.0
    rng = np.random.default_rng(1)
    n, library, h = 4000, 1e4, 0.3
    theta = np.tile([2.0, 1.0, 5.0, 3.0, 1.0, 1.0, 50.0], (n, 1))
    # Equal mean/theta ratios across program genes make the program total exactly NB.
    ratio = rng.uniform(0.1, 0.8, size=(n, 1))
    mean = theta * ratio
    lib = np.full(n, library)
    got = expected_soft_indicator(torch.tensor(mean), torch.tensor(theta), torch.tensor(lib), spec, h, spec.taus).numpy()
    draws = rng.poisson(rng.gamma(theta[:, spec.program_idx], mean[:, spec.program_idx] / theta[:, spec.program_idx], size=(200, n, 2)))
    r = (spec.mu0 - np.log1p(draws.sum(-1) / library * 1e4)) / (spec.s0 + EPS)
    sampled = 1 / (1 + np.exp(-(np.array(spec.taus)[None, None] - r[..., None]) / h))
    np.testing.assert_allclose(got.mean(0), sampled.mean((0, 1)), atol=4e-3)
    assert got.shape == (n, 2) and ((got > 0) & (got < 1)).all()


def test_fraction_loss_targets_training_conditions_and_reaches_the_doser():
    torch.set_num_threads(1)
    panel, _ = fixture_panel()
    spec = toy_spec()
    drugs = ["DrugA", "Vehicle"]
    controls, _ = prediction_inputs(panel, spec, "test", "nb")
    np.testing.assert_array_equal(supervised_taus(spec, controls.expression), spec.taus)
    taus = supervised_taus(spec, controls.expression, [0.25, 0.5, 0.9])
    assert taus.shape == (3,) and np.all(np.diff(taus) >= 0)
    conditions = fraction_conditions(panel, spec, drugs, 0.25, taus)
    assert conditions[2].shape == (2, 3)
    changed = panel.copy()
    changed.X[changed.obs.split.isin(["val", "test"]).to_numpy()] = np.nan
    for a, b in zip(conditions, fraction_conditions(changed, spec, drugs, 0.25, taus)):
        np.testing.assert_array_equal(a, b)
    assert conditions[0].tolist() == [0, 1] and np.allclose(conditions[1], [np.log1p(0.01), 0])
    model = make_model(7, np.eye(2, dtype=np.float32), small_config("nb", FRACTION), 0, "cpu")
    # Only the drug condition: zero-dose vehicle rightly sends no gradient to the doser.
    loss = FractionLoss(model, spec, controls, tuple(v[:1] for v in conditions), FRACTION, taus, 0)()
    assert loss.ndim == 0 and loss.item() > 0
    loss.backward()
    for module in (model.decoder, model.dosers, model.drug_embedding_encoder, model.encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_real_chemical_features_and_missing_structure_failure():
    obs = pd.DataFrame({"drug": ["DrugA", "DrugB", "Vehicle"]})
    structures = pd.DataFrame({"name": ["DrugA ", "DrugB"], "SMILES": ["CCO", "CCN"]})
    drugs, fp, meta = molecular_inputs(obs, structures)
    assert fp.shape == (3, 2048) and fp.dtype == np.float32
    assert not np.array_equal(fp[0], fp[1]) and not fp[2].any()
    assert drugs == ["DrugA", "DrugB", "Vehicle"] and len(meta["molecules"]) == 3
    with pytest.raises(ValueError, match="DrugB"):
        molecular_inputs(obs, structures.iloc[:1])
    with pytest.raises(ValueError, match="found 2"):
        molecular_inputs(obs, pd.concat([structures, pd.DataFrame({"name": ["DrugA"], "SMILES": ["CCC"]})]))


@pytest.mark.parametrize("likelihood", ["gauss", "nb"])
def test_zero_dose_single_cell_and_training_updates(likelihood):
    torch.set_num_threads(1)
    model = make_model(6, np.eye(2, dtype=np.float32), small_config(likelihood), 2, "cpu")
    model.eval()
    assert not model.covariates_embeddings.training and not model.adversary_covariates.training
    zero = model.compute_drug_embeddings_(drugs_idx=torch.tensor([0]), dosages=torch.tensor([0.0]))
    assert zero.shape == (1, 4) and torch.equal(zero, torch.zeros_like(zero))
    before = copy.deepcopy(model.state_dict())
    model.train()
    counts = torch.poisson(torch.full((16, 6), 3.0))
    x = torch.log1p(counts) if likelihood == "nb" else torch.randn(16, 6)
    for _ in range(4):
        losses = model.update(x, drugs_idx=torch.arange(16) % 2, dosages=torch.ones(16),
                              counts=counts, library=counts.sum(1), covariates=[torch.ones(16, 1)])
        assert all(np.isfinite(v) for v in losses.values())
    after = model.state_dict()
    for prefix in ("encoder", "decoder", "adversary_drugs", "drug_embedding_encoder", "dosers"):
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix) and "weight" in k)
    torch.testing.assert_close(before["drug_embeddings.weight"], after["drug_embeddings.weight"])
    for n in (2, 17, 33, 48):
        batches = minibatches(n, 16, torch.Generator().manual_seed(0))
        assert min(map(len, batches)) >= 2
        assert sorted(torch.cat(batches).tolist()) == list(range(n))


@pytest.mark.parametrize("likelihood,fraction", [("gauss", None), ("nb", None), ("nb", FRACTION),
                                                 ("nb", {**FRACTION, "quantiles": [0.2, 0.5, 0.95]})])
def test_checkpoint_roundtrip_and_real_evaluator(tmp_path, likelihood, fraction):
    torch.set_num_threads(1)
    panel, manifest = fixture_panel()
    spec = toy_spec()
    spec.save(tmp_path / "spec")
    panel.write_h5ad(tmp_path / "panel.h5ad")
    (tmp_path / "split.json").write_text(json.dumps(manifest))
    args = argparse.Namespace(out=str(tmp_path / "runs"), device="cpu", spec=str(tmp_path / "spec"),
                              split_config=str(tmp_path / "split.json"), data=str(tmp_path / "panel.h5ad"),
                              split="test", generation=None, sampling_seed=0)
    train_one(panel, spec, small_config(likelihood, fraction), ["DrugA", "Vehicle"], np.eye(2, dtype=np.float32), {}, args, 0)
    evaluate_run(panel, spec, args, 0)
    out = tmp_path / "runs/seed_0/test_sample_s0"
    # Resubmission skips finished work; a different config refuses to reuse the run.
    stamp = (tmp_path / "runs/seed_0/model.pt").stat().st_mtime_ns, (out / "summary.json").stat().st_mtime_ns
    train_one(panel, spec, small_config(likelihood, fraction), ["DrugA", "Vehicle"], np.eye(2, dtype=np.float32), {}, args, 0)
    evaluate_run(panel, spec, args, 0)
    assert stamp == ((tmp_path / "runs/seed_0/model.pt").stat().st_mtime_ns, (out / "summary.json").stat().st_mtime_ns)
    with pytest.raises(ValueError, match="different config"):
        train_one(panel, spec, {**small_config(likelihood, fraction), "epochs": 3}, ["DrugA", "Vehicle"],
                  np.eye(2, dtype=np.float32), {}, args, 0)
    result = json.loads((out / "summary.json").read_text())
    assert result["n_conditions"] == 1 and result["split"] == "test"
    assert np.isfinite(result["wrf_err_pp_q0.95"]["mean"])
    assert len((out / "per_well.jsonl").read_text().splitlines()) == 2
    assert ad.read_h5ad(out / "predictions.h5ad").shape == (20, 6)
    from sciplex_baselines.report_chemcpa import main as report_main
    report_main(["--runs", args.out, "--data", args.data, "--spec", args.spec,
                 "--seeds", "0", "--out", str(tmp_path / "report")])
    report = json.loads((tmp_path / "report/summary.json").read_text())
    assert report["across_seeds"]["wrf_err_pp_q0.95"]["mean"] == pytest.approx(result["wrf_err_pp_q0.95"]["mean"])
    assert report["likelihood"] == likelihood and (tmp_path / "report/diagnostics.pdf").is_file()
    sensitivity = pd.read_csv(tmp_path / "report/threshold_sensitivity.csv")
    family = "nbwrf" if fraction else likelihood
    assert report["family"] == family and set(sensitivity.method) == {family, "vehicle", "replicate"}
    run = json.loads((tmp_path / "runs/seed_0/run.json").read_text())
    assert run["method"].endswith("_wrf") == bool(fraction)
    assert all(np.isfinite(json.loads(line)["loss_auxiliary"]) for line in (tmp_path / "runs/seed_0/history.jsonl").open())
    from sciplex_baselines.compare_chemcpa import main as compare_main
    compare_main(["--reports", str(tmp_path / "report"), "--out", str(tmp_path / "compare")])
    assert "Replicate" in (tmp_path / "compare/comparison.md").read_text()
    # An altered frozen score must fail before producing any new metrics.
    (tmp_path / "spec/spec.json").write_text((tmp_path / "spec/spec.json").read_text() + "\n")
    with pytest.raises(ValueError, match="spec differs"):
        evaluate_run(panel, spec, args, 0)
