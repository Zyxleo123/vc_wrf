"""Registry of models, methods, references and grids shared by every analysis stage."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "artifacts" / "eval_spec_v1"
BASIS_PATH = REPO / "artifacts" / "analysis_v1" / "nonprogram_pca.npz"
DEFAULT_PANEL = os.environ.get("SCIPLEX_PANEL", "/zfsauton/scratch/yixiz/sciplex3/processed/a549_hdaci_v1.h5ad")
DEFAULT_ROOT = Path(os.environ.get("ANALYSIS_ROOT", REPO / "results" / "analysis"))
CHEMCPA_RUNS = REPO / "results" / "chemcpa"
CELLFLOW_DOCS = REPO / "docs" / "cellflow"

GRID13 = json.loads((REPO / "configs" / "eval_grid13_v1.json").read_text())
GRID13_QS = [float(q) for q in GRID13["quantiles"]]
SPEC_QS = [0.9, 0.95, 0.975]
PRIMARY_Q = 0.95
# Every 0.025 from 0.025 to 0.975; contains all 13 grid quantiles.
FINE_QS = [round(0.025 * k, 3) for k in range(1, 40)]
SPLITS = ("val", "test")
N_GENES, N_PROGRAM = 2085, 97

# Plot identity follows the role, identically in both models (dataviz reference
# palette, categorical slots 1-4 in fixed order).
ROLES = {
    "base": {"color": "#2a78d6", "label": "λ = 0 (no WRF loss)"},
    "wrf3": {"color": "#eb6834", "label": "+ WRF loss, 3 τ"},
    "wrf13": {"color": "#1baf7a", "label": "+ WRF loss, 13 τ"},
    "mmd": {"color": "#eda100", "label": "+ MMD loss"},
}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REFERENCES = {
    "vehicle": {"label": "Vehicle cells (no drug effect)", "dash": (0, (6, 3))},
    "interpolation": {"label": "Log-dose WRF interpolation", "dash": (0, (4, 1.5, 1, 1.5))},
    "nearest_dose": {"label": "Nearest dose (CellFlow CSV)", "dash": (0, (2, 2))},
    "adjacent_dose_mixture": {"label": "Adjacent-dose mixture (CellFlow CSV)", "dash": (0, (8, 2, 2, 2))},
    "replicate": {"label": "Replicate well (noise floor)", "dash": (0, (1, 2.5))},
}
# Reference rows in docs/cellflow/per_condition.csv and thresholds.csv.
CELLFLOW_REFERENCE_NAMES = {"unchanged_control": "vehicle", "replicate_reference": "replicate",
                            "nearest_dose": "nearest_dose", "adjacent_dose_mixture": "adjacent_dose_mixture"}


def _chemcpa_methods() -> dict:
    """Every chemCPA config with predictions: λ, h and supervised quantiles from the config."""
    selected = {"nb_v1": "base", "nbwrf_l10_h0.5": "wrf3", "nbwrfgrid_l10_h0.25": "wrf13"}
    methods = {}
    for path in sorted((REPO / "configs").glob("chemcpa_*.json")):
        name = path.stem.removeprefix("chemcpa_")
        config = json.loads(path.read_text())
        loss = config.get("fraction_loss")
        if loss is None:
            entry = {"lam": 0.0, "h": None, "train_qs": [], "likelihood": config["likelihood"]}
        else:
            qs = [float(q) for q in loss.get("quantiles", SPEC_QS)]
            entry = {"lam": float(loss["weight"]), "h": float(loss["h"]), "train_qs": qs, "likelihood": "nb"}
        entry["n_taus"] = len(entry["train_qs"])
        entry["role"] = selected.get(name, "other" if name == "gauss_v1" else "sweep")
        entry["checkpoint_rule"] = "min validation MMD²"
        methods[name] = entry
    return methods


MODELS = {
    "chemcpa": {
        "label": "chemCPA-NB",
        "methods": _chemcpa_methods(),
        "loss_aggregation": "mean over sampled conditions and τs (sciplex_baselines/chemcpa.py FractionLoss)",
    },
    "cellflow": {
        "label": "CellFlow",
        "methods": {
            "cellflow": {"role": "base", "lam": 0.0, "h": None, "train_qs": [], "n_taus": 0,
                         "checkpoint_rule": "val WRF q0.95 then W1; candidate steps 500–30000"},
            "cellflow_wrf": {"role": "wrf3", "lam": 0.1, "h": 0.3, "train_qs": SPEC_QS, "n_taus": 3,
                             "checkpoint_rule": "val WRF q0.95 then W1; candidate steps 5000–30000"},
            "cellflow_wrf13": {"role": "wrf13", "lam": 0.1, "h": 0.3, "train_qs": GRID13_QS, "n_taus": 13,
                               "checkpoint_rule": "val WRF q0.95 then W1; candidate steps 5000–30000"},
            "cellflow_mmd": {"role": "mmd", "lam": 1.0, "h": None, "train_qs": [], "n_taus": 0,
                             "checkpoint_rule": "val WRF q0.95 then W1; candidate steps 500–30000"},
        },
        "loss_aggregation": "jnp.mean over τs (experiments/cellflow/source/cellflow_population_loss.py)",
    },
}


def role_method(model: str, role: str) -> str | None:
    for name, entry in MODELS[model]["methods"].items():
        if entry["role"] == role:
            return name
    return None


def method_label(model: str, method: str) -> str:
    entry = MODELS[model]["methods"][method]
    if entry["role"] in ROLES:
        base = ROLES[entry["role"]]["label"]
    else:
        base = f"WRF loss, {entry['n_taus']} τ" if entry["lam"] else method
    if entry["lam"] and entry["role"] != "mmd":
        base += f" (λ={entry['lam']:g}, h={entry['h']:g})"
    elif entry["role"] == "mmd":
        base += f" (λ={entry['lam']:g})"
    return base


def extract_path(root: Path, model: str, method: str, seed: int, split: str, sampling: int = 0) -> Path:
    return Path(root) / "extracts" / model / method / f"seed{seed}_{split}_s{sampling}.npz"


def chemcpa_manifest(runs: Path = CHEMCPA_RUNS, methods=None) -> list[dict]:
    """Prediction entries in the CellFlow artifacts.json format, from the chemCPA run layout."""
    entries = []
    for method in methods or MODELS["chemcpa"]["methods"]:
        for path in sorted((Path(runs) / method).glob("seed_*/*_sample_s*/predictions.h5ad")):
            split, _, sampling = path.parent.name.split("_")
            entries.append({"method": method, "seed": int(path.parent.parent.name.split("_")[1]),
                            "split": split, "sampling": int(sampling.removeprefix("s")), "path": str(path)})
    return entries


def cellflow_manifest(path: Path = CELLFLOW_DOCS / "artifacts.json") -> list[dict]:
    return [{"sampling": 0, **entry} for entry in json.loads(Path(path).read_text())["predictions"]]


def short_condition(condition: str) -> str:
    _, drug, dose = condition.split("|")
    drug = drug.split(" (")[0]
    return f"{drug} {int(float(dose))} nM"


def fractions_below(r: np.ndarray, taus) -> np.ndarray:
    """F(t) = Pr[r < t] for each t (the WRF indicator of eq. 1 at every threshold)."""
    r = np.sort(np.asarray(r, dtype=np.float64))
    return np.searchsorted(r, np.asarray(taus, dtype=np.float64), side="left") / len(r)
