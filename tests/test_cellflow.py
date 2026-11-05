"""CellFlow study code on a tiny synthetic panel (CPU JAX; run in the CellFlow environment)."""
import json

import numpy as np
import pandas as pd
import pytest

cellflow = pytest.importorskip("cellflow")
ad = pytest.importorskip("anndata")
jax = pytest.importorskip("jax")

from sciplex_interp import cellflow as cf_study  # noqa: E402

N_GENES = 12
CONFIG = {"batch_size": 16, "training_steps": 20, "hidden_dims": [16], "decoder_dims": [16], "time_encoder_dims": [8],
          "time_freqs": 4, "condition_embedding_dim": 8, "condition_mlp_dims": [8], "condition_output_dropout": 0.0,
          "constant_noise": 0.0, "ot_epsilon": 0.1, "weight_decay": 1e-6, "learning_rate_peak": 1e-3,
          "learning_rate_end": 1e-5, "warmup_steps": 2, "ema": 1.0, "validation_steps": [20]}


def tiny_panel(seed=0, per_condition=20):
    """Vehicle + two drugs at two training doses, in both replicates, plus labelled held-out cells."""
    rng = np.random.default_rng(seed)
    rows, X = [], []

    def add(drug, dose, split, shift):
        for rep in ("rep1", "rep2"):
            for _ in range(per_condition):
                rows.append({"drug": drug, "dose": float(dose), "split": split, "replicate": rep,
                             "condition": f"A549|{drug}|{dose:g}"})
                X.append(np.maximum(rng.normal(shift, 0.3, N_GENES), 0))

    add("Vehicle", 0, "control", 0.5)
    for i, drug in enumerate(("drugA", "drugB")):
        add(drug, 10, "train", 0.6 + 0.1 * i)
        add(drug, 10000, "train", 1.0 + 0.2 * i)
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    return ad.AnnData(np.asarray(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]))


class TinyData:
    def __init__(self, seed=0):
        self.adata = tiny_panel(seed)


@pytest.fixture(scope="module")
def model():
    train = cf_study.model_data(TinyData())
    cf, _ = cf_study.build(train, CONFIG, seed=0)
    return cf, train


def test_model_data_keeps_only_permitted_cells():
    a = tiny_panel()
    a.obs.loc[a.obs_names[:5], "split"] = "test"
    data = TinyData()
    data.adata = a
    train = cf_study.model_data(data)
    assert set(train.obs["split"]) <= {"train", "control"}
    assert train.n_obs == a.n_obs - 5


def test_sampler_keeps_the_published_random_stream(model):
    from cellflow.data._dataloader import TrainSampler
    cf, _ = model
    published = TrainSampler(data=cf.train_data, batch_size=8)
    ours = cf_study.make_sampler(cf, 8)
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(5):
        a, b = published.sample(r1), ours.sample(r2)
        np.testing.assert_array_equal(a["src_cell_data"], b["src_cell_data"])
        np.testing.assert_array_equal(a["tgt_cell_data"], b["tgt_cell_data"])
        for k in a["condition"]:
            np.testing.assert_array_equal(a["condition"][k], b["condition"][k])


def test_target_means_match_the_sampled_targets(model):
    cf, _ = model
    means = cf_study.target_means(cf)
    assert means.shape == (cf.train_data.n_perturbations, N_GENES)
    sampler = cf_study.make_sampler(cf, 4096)  # with replacement, so the batch mean approaches the target mean
    batch = sampler.sample(np.random.default_rng(1))
    np.testing.assert_allclose(batch["tgt_cell_data"].mean(0), means[batch["target_idx"]], atol=0.03)


def run_steps(aux, n=8, k=1):
    train = cf_study.model_data(TinyData())
    cf, _ = cf_study.build(train, CONFIG, seed=0)
    step = cf_study.make_step(cf.solver, aux, cf_study.target_means(cf), k)
    sampler = cf_study.make_sampler(cf, 16)
    rng, key, out = np.random.default_rng(0), jax.random.PRNGKey(0), []
    for _ in range(n):
        key, sub = jax.random.split(key)
        batch = sampler.sample(rng) if k == 1 else sampler.sample_many(rng, k)
        out.append([float(v) for v in step(sub, batch)])
    return cf, np.array(out)


def test_aux_step_schedule_and_weighting():
    _, rows = run_steps({"kind": "mean", "weight": 2.0, "every": 2, "start": 3, "batch": 8, "ode_steps": 2})
    total, fm, term, active = rows.T
    assert active.tolist() == [0, 0, 0, 1, 0, 1, 0, 1]  # updates 4, 6, 8: step >= 3 and (step + 1) % 2 == 0
    assert (term[active == 0] == 0).all() and (term[active == 1] > 0).all()
    np.testing.assert_allclose(total, fm + 2.0 * 2 * term, rtol=1e-5)


def test_zero_weight_term_equals_plain_flow_matching():
    """Same random stream: a weight-0 term leaves the published update unchanged."""
    cf_aux, _ = run_steps({"kind": "mean", "weight": 0.0, "every": 1, "start": 0, "batch": 8, "ode_steps": 2}, n=4)
    train = cf_study.model_data(TinyData())
    cf, _ = cf_study.build(train, CONFIG, seed=0)
    sampler = cf_study.make_sampler(cf, 16)
    rng, key = np.random.default_rng(0), jax.random.PRNGKey(0)
    for _ in range(4):
        key, sub = jax.random.split(key)
        cf.solver.step_fn(sub, sampler.sample(rng))
    for a, b in zip(jax.tree.leaves(cf_aux.solver.vf_state.params), jax.tree.leaves(cf.solver.vf_state.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_mean_term_survives_an_exploding_field():
    """Huge parameters overflow an unguarded rollout; the guarded update stays finite."""
    train = cf_study.model_data(TinyData())
    cf, _ = cf_study.build(train, CONFIG, seed=0)
    state = cf.solver.vf_state
    cf.solver.vf_state = state.replace(params=jax.tree.map(lambda p: p * 200.0, state.params))
    step = cf_study.make_step(cf.solver, {"kind": "mean", "weight": 1.0, "every": 1, "start": 0, "batch": 8, "ode_steps": 8},
                              cf_study.target_means(cf))
    batch = cf_study.make_sampler(cf, 16).sample(np.random.default_rng(0))
    step(jax.random.PRNGKey(0), batch)
    for leaf in jax.tree.leaves(cf.solver.vf_state.params):
        assert np.isfinite(np.asarray(leaf)).all()


@pytest.mark.parametrize("aux", [None, {"kind": "mean_velocity", "weight": 3.0},
                                 {"kind": "mean", "weight": 3.0, "every": 1, "start": 0, "batch": 4, "ode_steps": 2}])
def test_several_conditions_per_update(aux):
    _, rows = run_steps(aux, n=3, k=3)
    assert np.isfinite(rows).all()
    if aux:
        np.testing.assert_allclose(rows[:, 0], rows[:, 1] + 3.0 * rows[:, 2], rtol=1e-5)


def test_stacked_batches_keep_each_target_with_its_condition(model):
    cf, _ = model
    sampler = cf_study.make_sampler(cf, 512)
    batch = sampler.sample_many(np.random.default_rng(0), 4)
    assert batch["src_cell_data"].shape == (4, 512, N_GENES) and batch["target_idx"].shape == (4,)
    means = cf_study.target_means(cf)
    for i, target in enumerate(batch["target_idx"]):
        np.testing.assert_allclose(batch["tgt_cell_data"][i].mean(0), means[target], atol=0.05)
        for name, value in batch["condition"].items():
            np.testing.assert_array_equal(value[i], cf.train_data.condition_data[name][target])


def test_parameters_round_trip(tmp_path, model):
    cf, train = model
    path = tmp_path / "params.msgpack"
    cf_study.save_params(cf, path)
    other, _ = cf_study.build(train, CONFIG, seed=1)
    cf_study.load_params(other, path)
    for a, b in zip(jax.tree.leaves(cf.solver.vf_state_inference.params), jax.tree.leaves(other.solver.vf_state_inference.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert other.solver.is_trained


def test_generation_uses_the_replicate_proportions(model):
    cf, train = model
    cf.solver.is_trained = True
    conds = [{"condition": "A549|drugA|100", "drug": "drugA", "dose_nM": 100.0, "n_cells": 40, "n_rep1": 30}]
    cells, negative = cf_study.generate(cf, train, conds, seed=0, n=16)
    x = cells["A549|drugA|100"]
    assert x.shape == (16, N_GENES) and (x >= 0).all() and 0 <= negative <= 1


def test_teacher_covariate_reaches_a_held_out_dose(tmp_path):
    keys = [json.dumps([d, x]) for d in ("drugA", "drugB") for x in (10.0, 100.0, 10000.0)]
    (tmp_path / "teacher").mkdir()
    np.savez(tmp_path / "teacher" / "autodec.npz", keys=np.array(keys), coef=np.ones((6, 3), np.float32),
             basis=np.zeros((3, N_GENES), np.float32), control_mean=np.zeros(N_GENES, np.float32), scale=np.float32(2.0))
    cfg = {**CONFIG, "covariates": ["drug", "dose", "teacher"]}
    train = cf_study.model_data(TinyData(), cfg, tmp_path)
    assert cf_study.teacher_key("drugA", 100.0) in train.uns["teacher_embeddings"]  # never observed in training
    np.testing.assert_allclose(train.uns["teacher_embeddings"][cf_study.teacher_key("drugA", 10.0)], 0.5)
    np.testing.assert_array_equal(train.uns["teacher_embeddings"][cf_study.teacher_key("Vehicle", 0.0)], 0)
    cf, _ = cf_study.build(train, cfg, seed=0)
    cf.solver.is_trained = True
    conds = [{"condition": "A549|drugA|100", "drug": "drugA", "dose_nM": 100.0, "n_cells": 40, "n_rep1": 20}]
    assert cf_study.generate(cf, train, conds, seed=0, n=8)[0]["A549|drugA|100"].shape == (8, N_GENES)


def test_decode_residual_keeps_the_source_cell():
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.normal(size=(N_GENES, 3)))[0].T.astype(np.float32)
    centre = rng.normal(size=N_GENES).astype(np.float32)
    x0 = rng.normal(size=(5, N_GENES)).astype(np.float32)
    z0 = (x0 - centre) @ U.T
    np.testing.assert_allclose(cf_study.decode(z0, z0, x0, U, centre, "residual"), x0, atol=1e-6)  # no displacement
    move = rng.normal(size=(5, 3)).astype(np.float32)
    delta = cf_study.decode(z0 + move, z0, x0, U, centre, "residual") - x0
    np.testing.assert_allclose(delta @ U.T, move, atol=1e-5)  # the displacement stays inside the basis
    inverse = cf_study.decode(z0, z0, x0, U, centre, "inverse")
    np.testing.assert_allclose((inverse - centre) @ U.T, z0, atol=1e-5)


def test_latent_flow_generates_gene_space_cells():
    rng = np.random.default_rng(1)
    U = np.linalg.qr(rng.normal(size=(N_GENES, 4)))[0].T.astype(np.float32)
    train = cf_study.model_data(TinyData())
    train.obsm["X_latent"] = ((train.X - 0.5) @ U.T).astype(np.float32)
    train.uns["latent"] = {"basis": U, "centre": np.full(N_GENES, 0.5, np.float32), "decode": "residual"}
    cf, _ = cf_study.build(train, {**CONFIG, "latent": {"dims": 4}}, seed=0)
    cf.solver.is_trained = True
    conds = [{"condition": "A549|drugA|100", "drug": "drugA", "dose_nM": 100.0, "n_cells": 40, "n_rep1": 20}]
    assert cf_study.generate(cf, train, conds, seed=0, n=8)[0]["A549|drugA|100"].shape == (8, N_GENES)
