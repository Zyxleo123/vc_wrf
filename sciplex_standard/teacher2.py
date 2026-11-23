"""Condition response embedding v2: analogue retrieval with a separately calibrated magnitude.

What v1 (`sciplex_standard.teacher`) does and why it fails on unseen drugs: a tanh MLP maps a
4-number drug code -- a learned linear map of 32 Uni-Mol principal components -- plus a cell-line
code and log dose to the 20 coordinates of the condition's mean shift, fit with unweighted MSE. A
drug whose structure is unlike the training drugs lands near the middle of the training codes, and
the MSE fit answers there with something close to the average shift, which is close to nothing. On
the 9 unseen drugs v1's predicted shift has cosine 0.41 with the measured one but only 0.41 of its
size, and for drugs whose direction it partly knows the size collapses (Belinostat: cos 0.27, size
0.07; Givinostat: cos 0.36, size 0.14). Under a squared-error fit that shrinkage is not a bug --
the error-minimising size is exactly the expected cosine -- but v1 shrinks several times further
than that, and the generated cells inherit it: they sit almost on top of the control cells.

v2 keeps the same interface (a per-condition vector in the same 20-direction basis, written to the
same file) and rebuilds the prediction out of three pieces that are fit, and calibrated, separately:

  direction   a similarity-weighted average of the measured shifts of the *other* drugs at the same
              (cell line, dose), or a ridge regression from the molecular representation, whichever
              wins the leave-one-drug-out comparison; a convex blend of the two is also allowed.
              An average of real responses cannot answer "no effect" the way a regression can.
  magnitude   a separate ridge on log ||shift|| from the same representation, dose and cell line, so
              how *much* a drug does is predicted by a model that is never asked to also say what it
              does. Combining them gives  shift = gain * magnitude * unit(direction).
  gain        one scalar per dose (optionally per cell line), fit by leave-one-drug-out over the
              training drugs to minimise the squared error of that product against the measured
              shifts. Holding a whole drug out makes the calibration set look like the unseen-drug
              task, so the gain that is right there is the gain used at test time.

Nothing of the held-out drugs (`ood`), the test conditions or the val conditions is read: the fit,
the hyper-parameter choice and the calibration all run on `train` conditions only.

usage: python -m sciplex_standard.teacher2 --molecules results/standard/unimol_embed.npz
Writes <root>/teacher/std_autodec_v2.npz in the format sciplex_standard.ours already reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_FILE, REPO, StdData
from .represent import annotations, representations, shift_table, similarity

K_BASIS = 20
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
TAUS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
# how the gain may grow with the size of the prediction: 1.0 is one scalar for the group,
# above 1.0 lets a large, confident prediction be scaled up more than a small one.
# Left at 1.0: a leave-one-drug-out fit over the training drugs does prefer a power above 1, but the
# held-out drugs' predictions are larger than anything in training, so the extrapolation overshoots
# badly (All RMSE 0.076 against 0.060 for no effect). Kept as a flag for the ablation only.
SIZE_POWERS = (1.0,)


GAIN_PRIOR = 0.25      # how much of the pooled calibration to keep in a query's local gain
LOCAL_GAIN = False     # default: one gain per (cell line, dose), the version with one free scalar
DENOISE = True         # default: shrink each neighbour by its own measurement error before averaging


def signal_norm(Y: np.ndarray, noise2: np.ndarray) -> np.ndarray:
    """How large each measured response really is, with the measurement error taken out.

    `denoise` returns the error-minimising estimate of the mean *vector*, whose norm is
    (1 - share)*||y||. The signal's own norm is sqrt(1 - share)*||y||, which is the quantity to carry
    over to an unseen drug: the question "how big is this drug's response" is answered by the
    analogue's biology, not by the error-minimising hedge on its direction. On this panel the two
    differ by 2.3x for a typical training condition (share 0.81), which is most of the magnitude
    deficit of a retrieval predictor.
    """
    n2 = (Y * Y).sum(1)
    return np.sqrt(np.maximum(n2 - noise2, 0.0))


def denoise(Y: np.ndarray, noise2: np.ndarray) -> np.ndarray:
    """Shrink each measured coordinate vector toward zero by its own measurement error.

    A retrieval predictor averages the *measured* shifts of a query's neighbours, and a measured
    shift is signal plus sampling noise: ||y||^2 = ||s||^2 + E||eps||^2. On sci-Plex3 the noise term
    is not small -- most training conditions are a few dozen cells and their shift is mostly noise --
    so averaging raw measurements produces an average whose size is set by how many cells happened
    to be sequenced rather than by biology, and it is systematically too small for the one thing we
    want to carry over: how large the response really is. The James-Stein factor
    max(0, 1 - E||eps||^2 / ||y||^2) estimates the signal, condition by condition.
    """
    n2 = (Y * Y).sum(1)
    return Y * np.maximum(0.0, 1.0 - noise2 / np.maximum(n2, 1e-12))[:, None]


def power_weight(D: np.ndarray, power: float, ref: float | None = None) -> np.ndarray:
    """Row factors that turn a gain into a gain that grows with the size of the prediction.

    With `power` 1 this is 1 everywhere, so the gain is a single scalar per group. Above 1 a
    prediction twice the typical size is scaled up by 2**(power-1) more, which is what the data ask
    for when the drugs whose response is large are also the ones whose direction is trustworthy.
    The reference size keeps the factor near 1 for a typical prediction, so `power` does not fight
    the gain for the overall scale; at query time it is the training rows' reference, not the query
    set's, so nothing about the held-out drugs enters.
    """
    n = np.linalg.norm(D, axis=1, keepdims=True)
    r = float(np.median(n)) if ref is None else ref
    if power == 1.0 or r <= 0:
        return np.ones_like(n)
    return (np.maximum(n, 1e-12) / r) ** (power - 1.0)


def deshrink(x: np.ndarray, y: np.ndarray, lo: float = 0.5, hi: float = 4.0) -> tuple[float, float]:
    """Slope and intercept that undo a ridge's pull toward the mean, on log magnitudes.

    A ridge fit of log ||shift|| has deliberately too little spread: the strongest drugs come out
    too weak and the weakest too strong. Regressing the measured log magnitude on the predicted one
    (both leave-one-drug-out) recovers the slope that puts the spread back, which matters here
    because the drugs this benchmark holds out are far stronger than the average training drug.
    """
    xm, ym = x.mean(), y.mean()
    var = float(((x - xm) ** 2).mean())
    b = float(((x - xm) * (y - ym)).mean() / var) if var > 1e-12 else 1.0
    b = float(np.clip(b, lo, hi))
    return b, float(ym - b * xm)


def gain_of(w: np.ndarray, D: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The error-minimising gain, averaged over the drugs a query actually retrieves.

    Calibrating one gain over every training drug answers the average drug, and the average drug in
    sci-Plex3 barely responds -- so the gain is far too timid for a strong, confidently retrieved
    query. Weighting the calibration by the same similarities used to retrieve the neighbours fixes
    the regime: a query whose neighbours are potent, well-predicted drugs inherits their gain. A
    `GAIN_PRIOR` share of the pooled calibration is kept so a query with no close neighbour falls
    back on the global answer.
    """
    n = (D * Y).sum(1)              # per drug: <D_j, Y_j>
    d = (D * D).sum(1)              # per drug: ||D_j||^2
    W = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    num = W @ n + GAIN_PRIOR * n.mean()
    den = W @ d + GAIN_PRIOR * d.mean()
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def local_gain(sim_dd: np.ndarray, drugs_in: np.ndarray, D: np.ndarray, Y: np.ndarray, tau: float) -> np.ndarray:
    """gain_of applied leave-one-drug-out inside a group (a drug never calibrates on itself)."""
    W = np.exp(sim_dd[np.ix_(drugs_in, drugs_in)] / tau)
    np.fill_diagonal(W, 0.0)
    return gain_of(W, D, Y)


def ridge_fit(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ridge with the penalty chosen by closed-form leave-one-out; returns (W, x_mean, y_mean, lam)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yc
    best, best_err = RIDGE_GRID[0], np.inf
    for lam in RIDGE_GRID:
        d = s / (s ** 2 + lam)
        H = (U * (s * d)).sum(1)
        r = (Yc - U @ (d[:, None] * UtY)) / np.maximum(1 - H[:, None], 1e-6)
        if (err := float((r ** 2).sum())) < best_err:
            best, best_err = lam, err
    W = Vt.T @ ((s / (s ** 2 + best))[:, None] * UtY)
    return W, xm, ym, best


def kernel_predict(sim: np.ndarray, Y: np.ndarray, tau: float) -> np.ndarray:
    W = np.exp(sim / tau)
    return (W / W.sum(1, keepdims=True)) @ Y


def loo_ridge(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Leave-one-row-out ridge predictions (the penalty is rechosen on each fold)."""
    out = np.zeros_like(Y)
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        W, xm, ym, _ = ridge_fit(X[keep], Y[keep])
        out[i] = (X[i] - xm) @ W + ym
    return out


def loo_kernel(sim_dd: np.ndarray, Y: np.ndarray, drugs_in: np.ndarray, tau: float) -> np.ndarray:
    """Leave-one-drug-out similarity-weighted averages of the other drugs' shifts in the group."""
    W = np.exp(sim_dd[np.ix_(drugs_in, drugs_in)] / tau)
    np.fill_diagonal(W, 0.0)
    return (W / np.maximum(W.sum(1, keepdims=True), 1e-12)) @ Y


def build(sh: dict, reps: dict, rep_name: str) -> dict:
    drugs = sorted(set(sh["drug"]))
    d_of = {d: i for i, d in enumerate(drugs)}
    S = sh["shift"].astype(np.float64)
    keys = [str(k) for k in sh["keys"]]
    drug_i = np.array([d_of[d] for d in sh["drug"]])
    group = np.array([f"{c}|{d:g}" for c, d in zip(sh["cell_type"], sh["dose_nM"])])
    is_train = sh["split"] == "train"

    V = np.linalg.svd(S[is_train], full_matrices=False)[2][:K_BASIS]       # basis from training shifts only
    Y_all = S @ V.T
    # the measurement error of each mean shift, carried into the 20 coordinates
    noise2 = (sh["sem2"].astype(np.float64) @ (V ** 2).T).sum(1) if "sem2" in sh else np.zeros(len(S))
    Y_signal = denoise(Y_all, noise2) if DENOISE else Y_all   # what a neighbour contributes when averaged
    S_norm = signal_norm(Y_all, noise2) if DENOISE else np.linalg.norm(Y_all, axis=1)   # and how big it is
    rep_d = np.asarray(reps[rep_name], np.float64)
    sim_dd = similarity(rep_d, rep_name)

    report = {"representation": rep_name, "groups": {}}
    pred = np.zeros_like(Y_all)
    for g in sorted(set(group)):
        in_g = group == g
        tr = np.flatnonzero(in_g & is_train)
        if len(tr) < 10:
            continue
        di = drug_i[tr]
        Y = Y_all[tr]
        mag = np.linalg.norm(denoise(Y_all[tr], noise2[tr]) @ V, axis=1)
        Xm, lm = rep_d[di], np.log(np.maximum(mag, 1e-6))
        raw = loo_ridge(Xm, lm[:, None]).ravel()                   # size, from a head of its own
        b_m, a_m = deshrink(raw, lm)                               # with the ridge's shrinkage undone
        m_loo = np.exp(a_m + b_m * raw)
        Ys = Y_signal[tr]                                          # neighbours, measurement error removed
        r_hat = loo_ridge(Xm, Ys)                                  # direction, regression arm
        best = None                                                # (tau, blend, size rule) by LOO
        for tau in TAUS:
            k_hat = loo_kernel(sim_dd, Ys, di, tau)
            for beta in BLENDS:
                u = (1 - beta) * r_hat + beta * k_hat
                nu = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
                m_nb = loo_kernel(sim_dd, S_norm[tr][:, None], di, tau).ravel()   # neighbours' own sizes
                for rule, D0 in (("head", (u / nu) * m_loo[:, None]), ("direct", u),
                                 ("neighbour", (u / nu) * m_nb[:, None])):
                    if float((D0 * D0).sum()) <= 0:
                        continue
                    for power in SIZE_POWERS:
                        D = D0 * power_weight(D0, power)
                        a_i = (local_gain(sim_dd, di, D, Y, tau) if LOCAL_GAIN
                               else np.full(len(D), float((D * Y).sum() / max((D * D).sum(), 1e-12))))
                        err = float(((a_i[:, None] * D - Y) ** 2).sum())
                        if best is None or err < best[0]:
                            best = (err, tau, beta, rule, power, a_i, k_hat, D)
        _, tau, beta, rule, power, gain_loo, k_hat, D = best
        gain = float((D * Y).sum() / (D * D).sum())                # the one-scalar version, for the record
        u_loo = (1 - beta) * r_hat + beta * k_hat
        report["groups"][g] = {
            "n_train_drugs": int(len(tr)), "tau": tau, "blend": round(beta, 2), "size": rule,
            "size_slope": round(b_m, 3), "size_power": power, "gain_pooled": round(gain, 4), "gain_local_mean": round(float(gain_loo.mean()), 4),
            "gain_local_range": [round(float(gain_loo.min()), 3), round(float(gain_loo.max()), 3)],
            "loo_cos": round(float(np.mean([(Y[i] @ u_loo[i]) / max(np.linalg.norm(Y[i]) * np.linalg.norm(u_loo[i]), 1e-12)
                                            for i in range(len(tr))])), 4),
            "loo_rmse_ratio": round(float(np.sqrt(((gain_loo[:, None] * D - Y) ** 2).sum() / (Y ** 2).sum())), 4),
            "loo_rmse_ratio_pooled_gain": round(float(np.sqrt(((gain * D - Y) ** 2).sum() / (Y ** 2).sum())), 4),
            "loo_rmse_ratio_v1_style": round(float(np.sqrt(((u_loo - Y) ** 2).sum() / (Y ** 2).sum())), 4)}
        # the fitted predictors, applied to every condition of this group (train, val, test, ood)
        Wd, xmd, ymd, _ = ridge_fit(Xm, Ys)
        Wm, xmm, ymm, _ = ridge_fit(Xm, lm[:, None])
        q = np.flatnonzero(in_g)
        dq = drug_i[q]
        r_q = (rep_d[dq] - xmd) @ Wd + ymd
        w = np.exp(sim_dd[np.ix_(dq, di)] / tau)
        same = dq[:, None] == di[None, :]                        # never let a drug retrieve itself
        w = np.where(same, 0.0, w)
        k_q = (w / np.maximum(w.sum(1, keepdims=True), 1e-12)) @ Ys
        u_q = (1 - beta) * r_q + beta * k_q
        wn = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        if rule == "direct":
            D_q = u_q
        else:
            m_q = (wn @ S_norm[tr] if rule == "neighbour"
                   else np.exp(a_m + b_m * ((rep_d[dq] - xmm) @ Wm + ymm).ravel()))
            D_q = (u_q / np.maximum(np.linalg.norm(u_q, axis=1, keepdims=True), 1e-12)) * m_q[:, None]
        D_q = D_q * power_weight(D_q, power, ref=np.median(np.linalg.norm(D, axis=1)))
        a_q = gain_of(w, D, Y) if LOCAL_GAIN else np.full(len(D_q), gain)
        pred[q] = a_q[:, None] * D_q
        # a training condition keeps its own leave-one-drug-out prediction, the one D already holds,
        # so the embedding the flow trains on is exactly as reliable as the one it meets on an unseen
        # drug -- otherwise the flow learns to trust a teacher that is better in training than in use
        pred[tr] = gain_loo[:, None] * D
    scale = float(Y_all[is_train].std())
    return {"keys": np.array(keys), "coef": pred.astype(np.float32),
            "basis": V.astype(np.float32), "scale": np.float32(scale),
            "control_means": sh["control_means"], "cell_types": sh["cell_types"],
            "n_train_conditions": np.int64(int(is_train.sum())), "report": json.dumps(report)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecules", required=True)
    p.add_argument("--smiles", default=None)
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--root", default=str(REPO / "results" / "standard"))
    p.add_argument("--rep", default="rdkit_morgan",
                   help="a representation name, 'structure' (SMILES-derived views only, what DOLORIS gets), "
                        "or 'all' (adds the annotated target and pathway); chosen by leave-one-drug-out")
    p.add_argument("--out", default=None)
    p.add_argument("--local-gain", action="store_true", help="ablation: calibrate the gain on the retrieved neighbours")
    p.add_argument("--size-powers", type=float, nargs="+", default=None, help="ablation: let the gain grow with size")
    p.add_argument("--raw-neighbours", action="store_true", help="ablation: average measured shifts, not denoised ones")
    a = p.parse_args(argv)
    global LOCAL_GAIN, SIZE_POWERS, DENOISE
    LOCAL_GAIN = bool(a.local_gain)
    SIZE_POWERS = tuple(a.size_powers) if a.size_powers else SIZE_POWERS
    DENOISE = not a.raw_neighbours
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "teacher" / "std_autodec_v2.npz"
    cache = root / "diagnose" / "shifts.npz"
    if cache.exists():                       # the cached mean shifts are all this needs
        sh = {k: v for k, v in np.load(cache, allow_pickle=True).items()}
    else:
        sh = shift_table(StdData(a.file))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **sh)
    sh = {k: (v.astype(str) if v.dtype.kind in "OU" else v) for k, v in sh.items()}
    ann = annotations(a.file)
    drugs = sorted(set(sh["drug"]))
    reps = representations(drugs, a.molecules, a.smiles or str(root / "smiles.csv"), ann)
    # which molecular view: chosen by the leave-one-drug-out error over the TRAINING drugs only,
    # so no held-out drug and no val/test condition takes part in the choice.
    built, scores = {}, {}
    structural = [r for r in reps if r not in ("target", "pathway2", "structure3_target")]
    candidates = {"all": list(reps), "structure": structural}.get(a.rep, [a.rep])
    for name in candidates:
        built[name] = build(sh, reps, name)
        groups = json.loads(str(built[name]["report"]))["groups"]
        scores[name] = float(np.mean([g["loo_rmse_ratio"] for g in groups.values()]))
        cos = float(np.mean([g["loo_cos"] for g in groups.values()]))
        print(f"{name:20s} leave-one-drug-out rmse/no-effect {scores[name]:.4f}  cos {cos:.3f}", flush=True)
    chosen = min(scores, key=scores.get)
    print(f"chosen representation: {chosen}", flush=True)
    t = built[chosen]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **t)
    (out.parent / (out.stem + "_selection.json")).write_text(json.dumps(
        {"chosen": chosen, "loo_rmse_ratio": scores, "report": json.loads(str(t["report"]))}, indent=1) + "\n")

    # how the chosen embedding scores against the measured mean shifts, per split (diagnostic only)
    pred = t["coef"].astype(np.float64) @ t["basis"].astype(np.float64)
    S = sh["shift"].astype(np.float64)
    for split in ("train", "val", "test", "ood"):
        m = sh["split"] == split
        nt = np.linalg.norm(S[m], axis=1)
        npd = np.linalg.norm(pred[m], axis=1)
        cos = (S[m] * pred[m]).sum(1) / np.maximum(nt * npd, 1e-12)
        rmse = np.sqrt(((pred[m] - S[m]) ** 2).mean(1))
        zero = np.sqrt((S[m] ** 2).mean(1))
        print(f"{split:6s} n {m.sum():4d}  cos {cos.mean():.3f}  gain {(npd / np.maximum(nt, 1e-12)).mean():.3f}  "
              f"rmse {rmse.mean():.4f} vs {zero.mean():.4f} for no effect "
              f"(ratio {np.mean(rmse / zero):.3f})", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
