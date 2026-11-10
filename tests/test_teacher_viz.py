"""The statistics behind the teacher's coordinate figures, on synthetic data with known answers."""
import numpy as np
import pytest

pytest.importorskip("anndata")
torch = pytest.importorskip("torch")

from sciplex_interp import teacher_viz as tv  # noqa: E402


def wells(rng, signal, n_cond=40, n_genes=200, n_cells=100, noise=0.5):
    """Two replicate wells per condition sharing `signal`, with independent cell noise."""
    cells, means = [], []
    for c in range(n_cond):
        X = signal[c] + noise * rng.standard_normal((n_cells, n_genes))
        cells.append(X)
        means.append(X.mean(0))
    ctrl = noise * rng.standard_normal((500, n_genes))
    return np.array(means) - ctrl.mean(0), cells, ctrl


def test_crossfit_finds_the_real_axis_and_keeps_noise_axes_under_the_bar():
    rng = np.random.default_rng(0)
    direction = np.zeros(200)
    direction[:20] = 1 / np.sqrt(20)
    strength = rng.normal(0, 1.0, 40)
    signal = strength[:, None] * direction[None]
    D1, _, _ = wells(rng, signal)
    D2, cells2, _ = wells(rng, signal)
    share, V = tv.crossfit_signal(D1, D2, cells2, k=6)
    assert share[0] > 0.9
    assert abs(V[0] @ direction) > 0.95
    assert np.all(share[1:] < tv.REAL_SHARE), share


def test_crossfit_is_centred_on_noise_and_does_not_reach_the_bar():
    """Under noise one axis's share has SD about sqrt(2 / conditions), so the bar is about 6 SD away with 78 conditions."""
    rng = np.random.default_rng(1)
    shares = []
    for _ in range(30):
        signal = np.zeros((78, 100))
        D1, _, _ = wells(rng, signal, n_cond=78, n_genes=100)
        D2, cells2, _ = wells(rng, signal, n_cond=78, n_genes=100)
        shares.append(tv.crossfit_signal(D1, D2, cells2, k=6)[0])
    shares = np.array(shares)
    assert abs(shares.mean()) < 0.06, shares.mean()
    assert shares.std() == pytest.approx(np.sqrt(2 / 77), rel=0.35)
    assert (shares >= tv.REAL_SHARE).sum() == 0


def test_crossfit_ignores_a_large_shared_vehicle_error():
    """The vehicle mean's error is one draw shared by every condition; it must not lift a noise axis."""
    rng = np.random.default_rng(3)
    signal = np.zeros((78, 100))
    D1, _, _ = wells(rng, signal, n_cond=78, n_genes=100)
    D2, cells2, _ = wells(rng, signal, n_cond=78, n_genes=100)
    clean = tv.crossfit_signal(D1, D2, cells2, k=6)[0]
    shifted = tv.crossfit_signal(D1, D2 + 5.0 * rng.standard_normal(100), cells2, k=6)[0]
    assert np.allclose(clean, shifted)


def test_shuffled_null_erases_a_real_axis():
    rng = np.random.default_rng(2)
    direction = np.zeros(100)
    direction[:20] = 1 / np.sqrt(20)
    strength = rng.normal(0, 1.0, 40)
    parts, wells_of = [], {"rep1": [], "rep2": []}
    start = 0
    for r in wells_of:
        for c in range(40):
            parts.append(strength[c] * direction + 0.5 * rng.standard_normal((60, 100)))
            wells_of[r].append(np.arange(start, start + 60))
            start += 60
    X = np.vstack(parts)
    means = {r: np.array([X[w].mean(0) for w in wells_of[r]]) for r in wells_of}
    real = tv.crossfit_signal(means["rep1"], means["rep2"], [X[w] for w in wells_of["rep2"]], k=6)[0]
    assert real[0] > 0.9
    null = tv.shuffled_null(X, wells_of, 6, rng, shuffles=30)
    assert null.shape == (30, 6)
    assert null[:, 0].mean() < 0.3
    # Centred near 0, slightly below it: E[1 / chi2_39 * 39] = 39 / 37. SD of one share about 0.22 here, of the mean about 0.016.
    assert -0.13 < null.mean() < 0.03, null.mean()


def test_replicate_r_finds_real_axes_and_stays_near_zero_on_noise():
    """Axes fit on the same cells make two wells look like they agree even on pure noise; refitting without the fold removes that."""
    rng = np.random.default_rng(5)
    D1, D2 = 0.2 * rng.standard_normal((78, 300)), 0.2 * rng.standard_normal((78, 300))
    V = np.linalg.svd((D1 + D2) / 2, full_matrices=False)[2][:6]
    naive = tv.column_r(D1 @ V.T, D2 @ V.T)
    w1, w2 = tv.replicate_coords(D1, D2, V, rng)
    assert w1.shape == (5, 78, 6)
    honest = tv.replicate_r(w1, w2)
    assert naive.min() > 0.5  # the bias this guards against
    assert np.abs(honest).max() < 0.4, honest  # SD of one r about 0.09 here
    # Over many noise draws: centred on 0, and never near the bar.
    many = []
    for _ in range(20):
        A, B = 0.2 * rng.standard_normal((78, 300)), 0.2 * rng.standard_normal((78, 300))
        many.append(tv.replicate_r(*tv.replicate_coords(A, B, np.linalg.svd((A + B) / 2, full_matrices=False)[2][:6], rng)))
    many = np.array(many)
    assert abs(many.mean()) < 0.03, many.mean()
    assert 0.04 < many.std() < 0.14, many.std()
    assert (many >= tv.REAL_SHARE).sum() == 0
    direction = np.zeros(300)
    direction[:30] = 1 / np.sqrt(30)
    signal = rng.normal(0, 1.0, 78)[:, None] * direction[None]
    D1, D2 = signal + 0.2 * rng.standard_normal((78, 300)), signal + 0.2 * rng.standard_normal((78, 300))
    V = np.linalg.svd((D1 + D2) / 2, full_matrices=False)[2][:6]
    r = tv.replicate_r(*tv.replicate_coords(D1, D2, V, rng))
    assert r[0] > 0.9 and np.abs(r[1:]).max() < 0.4, r


def test_column_r_keeps_a_leading_axis():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3, 50, 4)), rng.normal(size=(3, 50, 4))
    assert np.allclose(tv.column_r(a, b)[1], tv.column_r(a[1], b[1]))
    assert np.allclose(tv.column_r(a[0], a[0]), 1)


def test_path_fit_holds_a_curve_and_leaves_what_is_off_it():
    x = np.linspace(0, 10, 100)
    C = np.c_[x, (x - 3) * (x - 8), np.zeros(100)]
    fit = tv.path_fit(C)
    assert tv.curve_share(C, fit) > 0.9999
    C[50, 2] = 5.0
    assert (C - tv.path_fit(C)(C))[50, 2] > 4.5


def test_anova_shares_separate_main_effects_and_sum_to_one():
    dose_only = np.tile(np.arange(7.0), (26, 1))
    drug, dose, inter = tv.anova_shares(dose_only)
    assert dose == pytest.approx(1) and drug == pytest.approx(0, abs=1e-9) and inter == pytest.approx(0, abs=1e-9)
    drug_only = dose_only.T[:7, :7]
    assert tv.anova_shares(drug_only)[0] == pytest.approx(1)
    rng = np.random.default_rng(0)
    assert sum(tv.anova_shares(rng.normal(size=(26, 7)))) == pytest.approx(1)


def test_curve_r2_sees_a_curve_and_not_noise():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    assert tv.curve_r2(x, x ** 2 - x) > 0.999
    assert tv.curve_r2(x, rng.normal(size=200)) < 0.1


def test_nearest_share_detects_clusters_and_matches_chance_without_them():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 4)
    strata = np.tile(np.arange(4), 10)
    clustered = labels[:, None] * 10.0 + rng.normal(size=(40, 3))
    rows = {r["group"]: r for r in tv.nearest_share(tv.pair_distances(clustered), labels, strata, rng, perms=200)}
    assert rows["all"]["observed"] == 1 and rows["all"]["p"] < 0.01
    noise = rng.normal(size=(40, 3))
    rows = {r["group"]: r for r in tv.nearest_share(tv.pair_distances(noise), labels, strata, rng, perms=200)}
    assert rows["all"]["null_lo"] <= rows["all"]["observed"] <= rows["all"]["null_hi"] + 0.1


def test_mantel_sign_and_covariate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    y = rng.normal(size=20)
    gap = np.abs(x[:, None] - x[None])
    assert tv.mantel(gap, -gap, rng, perms=200)["rho"] < -0.95
    # The distance also has an independent part. A similarity that only tracks the covariate looks related until the covariate is removed.
    D = gap + np.abs(y[:, None] - y[None])
    assert tv.mantel(D, -gap, rng, perms=50)["rho"] < -0.3
    assert abs(tv.mantel(D, -gap, rng, covariate=gap, perms=50)["rho"]) < 0.02
    # A similarity tracking the independent part survives removing the covariate.
    assert tv.mantel(D, -np.abs(y[:, None] - y[None]), rng, covariate=gap, perms=50)["rho"] < -0.3


def test_gene_set_direction_and_fdr():
    loadings = np.arange(100.0)
    top = np.zeros(100, bool)
    top[-10:] = True
    assert tv.gene_set_scores(loadings, top)["auc"] == pytest.approx(1)
    assert tv.gene_set_scores(-loadings, top)["z"] < -3
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(tv.benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.5])


def test_plate_effect_finds_a_plate_shift():
    rng = np.random.default_rng(0)
    n = 40
    plates = [f"p{i % 4}" for i in range(n)] + [f"q{i % 4}" for i in range(n)]
    base = rng.normal(size=n)
    shift = {"p0": 3.0, "p1": -3.0, "p2": 0.0, "p3": 1.0, "q0": 0.0, "q1": 2.0, "q2": -2.0, "q3": 0.0}
    with_plate = np.r_[base, base] + np.array([shift[p] for p in plates]) + 0.1 * rng.normal(size=2 * n)
    without = np.r_[base, base] + 0.1 * rng.normal(size=2 * n)
    rows = tv.plate_effect(np.c_[with_plate, without], plates, n)
    assert rows[0]["plate_partial_r2"] > 0.9 and rows[0]["plate_p"] < 1e-6
    assert rows[1]["plate_p"] > 1e-3


def test_structure_classes_on_known_molecules():
    pytest.importorskip("rdkit")
    assert tv.structure_class("CC1=C(CCNCC2=CC=C(C=CC(=O)NO)C=C2)C2=C(N1)C=CC=C2") == "Hydroxamate"  # panobinostat
    assert tv.structure_class("NC1=CC=CC=C1NC(=O)C1=CC=C(CNC(=O)OCC2=CN=CC=C2)C=C1") == "Benzamide"  # entinostat
    assert tv.structure_class("[Na+].CCCC(CCC)C([O-])=O") == "Carboxylate"  # valproate
    assert tv.structure_class("[Na]OC(=O)CCCC1=CC=CC=C1") == "Carboxylate"  # phenylbutyrate, as GEO writes it
    assert tv.structure_class("c1ccccc1") == "Other"


def test_autodecoder_seeds_average_to_autodecoder():
    from types import SimpleNamespace

    from sciplex_interp.stats import K_BASIS, autodecoder, autodecoder_seeds

    rng = np.random.default_rng(0)
    drugs = np.repeat(["a", "b", "c"], 3)
    dose = np.tile([10.0, 100.0, 1000.0], 3)
    genes = K_BASIS + 5
    D = rng.normal(size=(9, genes))
    ctx = SimpleNamespace(drug=drugs, dose=dose, D=D, V=np.linalg.svd(D, full_matrices=True)[2], m0=np.zeros(genes))
    queries = [("a", 31.6), ("c", 10.0)]
    per_seed, names, codes = autodecoder_seeds(ctx, queries, 4, 1e-4, seeds=2, steps=5)
    assert per_seed.shape == (2, 2, K_BASIS) and names == ["a", "b", "c"] and codes.shape == (2, 3, 4)
    means = autodecoder(ctx, queries, 4, 1e-4, seeds=2, steps=5)
    assert np.allclose(means[0], per_seed.mean(0)[0] @ ctx.V[:K_BASIS], atol=1e-3)


def test_short_names():
    assert tv.short("Quisinostat (JNJ-26481585) 2HCl") == "Quisinostat"
    assert tv.short("Tubastatin A HCl") == "Tubastatin A"
    assert tv.short("Valproic acid sodium salt (Sodium valproate)") == "Valproic acid"
    assert set(tv.SELECTIVITY.values()) == set(tv.CLASSES)
