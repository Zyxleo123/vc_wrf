"""Shape and behaviour checks for the v2 condition response embedding, on synthetic drugs.

Twenty drugs live on a circle in a 2-number "structure" space; a drug's response is a rotation of a
fixed pair of directions by its angle, scaled by dose and by a per-drug potency that is itself a
smooth function of the angle. A drug held out of training is therefore predictable from its
neighbours, and the test asks that build() recovers it far better than predicting nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciplex_standard import teacher2


def synthetic(n_drugs: int = 24, n_genes: int = 60, held_out: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_drugs, endpoint=False)
    B = rng.normal(size=(2, n_genes))
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    cells, doses = ["A", "B"], [10.0, 100.0, 1000.0, 10000.0]
    keys, shift, drug, cell, dose, split = [], [], [], [], [], []
    for i, a in enumerate(ang):
        potency = 0.5 + 0.5 * np.cos(2 * a)
        for c in cells:
            for d in doses:
                direction = np.cos(a) * B[0] + np.sin(a) * B[1]
                size = potency * np.log10(d)
                keys.append(f"{c}_drug{i}_{d:g}")
                shift.append(size * direction + 0.05 * rng.normal(size=n_genes))
                drug.append(f"drug{i}")
                cell.append(c)
                dose.append(d)
                split.append("ood" if i < held_out else "train")
    sh = {"keys": np.array(keys), "shift": np.stack(shift).astype(np.float32), "drug": np.array(drug),
          "cell_type": np.array(cell), "dose_nM": np.array(dose, float), "split": np.array(split),
          "control_means": np.zeros((2, n_genes), np.float32), "cell_types": np.array(cells)}
    order = np.argsort(np.array([f"drug{i}" for i in range(n_drugs)]))   # build() indexes drugs by sorted name
    reps = {"circle": np.stack([np.cos(ang), np.sin(ang)], 1)[order]}
    return sh, reps


def test_build_recovers_held_out_drugs():
    sh, reps = synthetic()
    t = teacher2.build(sh, reps, "circle")
    assert t["coef"].shape[0] == len(sh["keys"])
    assert np.isfinite(t["coef"]).all()
    pred = t["coef"].astype(np.float64) @ t["basis"].astype(np.float64)
    ood = sh["split"] == "ood"
    true = sh["shift"].astype(np.float64)
    cos = [float(true[i] @ pred[i] / (np.linalg.norm(true[i]) * np.linalg.norm(pred[i]))) for i in np.flatnonzero(ood)]
    err = np.sqrt(((pred[ood] - true[ood]) ** 2).mean())
    zero = np.sqrt((true[ood] ** 2).mean())
    print(f"held-out cos {np.mean(cos):.3f}, rmse {err:.4f} vs {zero:.4f} for no effect")
    assert np.mean(cos) > 0.8, f"held-out drugs not recovered: cos {np.mean(cos):.3f}"
    assert err < 0.6 * zero, f"no better than predicting nothing: {err:.4f} vs {zero:.4f}"


def test_gain_is_not_degenerate():
    sh, reps = synthetic()
    t = teacher2.build(sh, reps, "circle")
    pred = t["coef"].astype(np.float64) @ t["basis"].astype(np.float64)
    ood = sh["split"] == "ood"
    size = np.linalg.norm(pred[ood], axis=1) / np.linalg.norm(sh["shift"].astype(np.float64)[ood], axis=1)
    print(f"held-out size ratio {size.mean():.3f}")
    assert 0.5 < size.mean() < 1.5, f"predicted size collapsed or exploded: {size.mean():.3f}"


if __name__ == "__main__":
    test_build_recovers_held_out_drugs()
    test_gain_is_not_degenerate()
    print("ok")
