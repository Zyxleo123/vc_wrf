"""Train pinned chemCPA and score generated populations with the frozen WRF spec.

Training and test evaluation are separate commands. Only train/control expression
enters optimization, validation chooses the checkpoint, and test predictions are
made exclusively from permitted controls and condition metadata.

Two likelihoods are supported (config "likelihood"): "gauss", the paper-era decoder
over log1p(CP10K), and "nb", CPA's negative binomial over raw UMIs with the observed
library size (see vendor/chemCPA_NOTICE.md). An NB config may add "fraction_loss",
the project's response-fraction regularizer (fraction_loss.py), fitted on training
conditions only. Both commands skip finished work, so a job that hits its time limit
is resubmitted rather than restarted.

An NB config may also set "output_genes": "program". The encoder still reads every
spec gene, but the decoder generates only the program genes plus one feature for
all other human UMIs. The whole reconstruction loss then falls on the genes behind
r(x): a control for attributing the fraction loss's gain to extra supervision on
that subset. Such runs are scored on the program-only metrics (WRF, W1(r),
program-gene RMSE).

Two more opt-in config keys (docs/interp/README.md); without them training is unchanged.
"dose_transform": "log10_nM" feeds the doser log10(dose_nM) / 4 instead of chemCPA's
log1p(dose in µM), which nearly merges 10 and 100 nM. "mean_loss" (MeanLoss) adds
weight x the squared difference between the counterfactual population's average (permitted
controls pushed through a training condition, as at prediction time) and that training
condition's measured average, for both likelihoods.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from sciplex_eval import EvalSpec, evaluate_predictions, summarize
from sciplex_eval.evaluate import DEFAULT_PANEL, DEFAULT_SPEC, load_panel
from sciplex_eval.metrics import evaluate_condition
from sciplex_eval.spec import _dense

from .chemistry import molecular_inputs
from .fraction_loss import expected_soft_indicator, real_targets, supervised_taus
from .vendor.chemcpa_model import ComPert

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "a4a4ded0c3b949c64ff1ea51033be1b7c301c36b"
LIKELIHOODS = ("gauss", "nb")
OUTPUT_GENES = ("all", "program")
GENERATIONS = ("sample", "mean")
EVAL_OUTPUTS = ("predictions.h5ad", "summary.json", "per_condition.jsonl", "per_well.jsonl",
                "control_diagnostic.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_hash(path):
    return {name: sha256(Path(path) / name) for name in ("spec.json", "arrays.npz")}


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_panel(panel, spec, manifest):
    """Enforce the frozen condition split, sizes, and well isolation before fitting."""
    required = {"condition", "split", "drug", "dose", "replicate", "culture_plate", "well_oligo"}
    if required - set(panel.obs):
        raise ValueError(f"panel metadata missing {sorted(required - set(panel.obs))}")
    if not panel.var_names.is_unique or not panel.obs_names.is_unique:
        raise ValueError("panel cell and gene identifiers must be unique")
    spec.align(panel.X[:1], panel.var_names)
    expected = {c["condition"]: c for c in manifest["conditions"]}
    ctrl = manifest["controls"]
    expected[ctrl["condition"]] = {**ctrl, "split": "control", "drug": "Vehicle", "dose_nM": 0}
    if set(panel.obs.condition.astype(str)) != set(expected):
        raise ValueError("panel condition set differs from the frozen split")
    for cond, rows in panel.obs.groupby("condition", observed=True):
        item = expected[str(cond)]
        if (set(rows.split.astype(str)) != {item["split"]}
                or set(rows.drug.astype(str)) != {item["drug"]}
                or not np.allclose(rows.dose.astype(float), item["dose_nM"])
                or len(rows) != item["n_cells"]):
            raise ValueError(f"condition metadata/count does not match frozen split: {cond}")
    wells = panel.obs.groupby(["replicate", "culture_plate", "well_oligo"], observed=True)
    if (wells["split"].nunique() > 1).any():
        raise ValueError("an experimental well occurs in multiple splits")


@dataclass
class Cells:
    """Model inputs for a set of cells; `expression` is their spec log1p(CP10K) view
    and `counts` the NB reconstruction target over the decoder's output features."""
    features: np.ndarray
    expression: np.ndarray
    counts: np.ndarray | None = None
    library: np.ndarray | None = None


def n_features(spec, likelihood):
    """Encoder input features."""
    return spec.n_genes + (likelihood == "nb")


def output_genes(config):
    return config.get("output_genes", "all")


def output_gene_ids(spec, config):
    """Genes the decoder generates: every spec gene, or the program genes alone."""
    return spec.gene_ids if output_genes(config) == "all" else list(spec.program_gene_ids)


def n_outputs(spec, config):
    """Decoder output features: the generated genes, plus the NB rest feature."""
    return len(output_gene_ids(spec, config)) + (config["likelihood"] == "nb")


def output_program_idx(spec, config):
    """Positions of the program genes among the decoder's output features."""
    return spec.program_idx if output_genes(config) == "all" else np.arange(len(spec.program_gene_ids))


def method_name(config):
    return (f"chemcpa_morgan_{config['likelihood']}" + ("_program" if output_genes(config) == "program" else "")
            + ("_wrf" if "fraction_loss" in config else ""))


def cell_inputs(panel, spec, mask, likelihood, outputs="all"):
    """Gaussian: the spec genes' log1p(CP10K). NB: those plus one log1p(CP10K) feature
    for all other human UMIs, with raw counts over the output features and the
    human-UMI library size that normalizes X. With outputs="program" the counts are
    the program genes plus one feature for all other human UMIs; inputs are unchanged."""
    expression = _dense(spec.align(panel.X[mask], panel.var_names)).astype(np.float32)
    if not np.isfinite(expression).all() or (expression < 0).any():
        raise ValueError("expression must be finite nonnegative log1p(CP10K)")
    if likelihood == "gauss":
        return Cells(expression, expression)
    counts = _dense(spec.align(panel.layers["counts"][mask], panel.var_names)).astype(np.float64)
    library = panel.obs["n_counts_human"].to_numpy(dtype=np.float64)[mask]
    rest = library - counts.sum(axis=1)
    if (library <= 0).any() or (rest < 0).any():
        raise ValueError("n_counts_human must be positive and cover the panel's counts")
    if not np.allclose(np.log1p(counts / library[:, None] * 1e4), expression, atol=1e-3):
        raise ValueError("X is not log1p(counts / n_counts_human * 1e4)")
    features = np.hstack([expression, np.log1p(rest / library * 1e4)[:, None]])
    if outputs == "program":
        counts = counts[:, spec.program_idx]
    target = np.hstack([counts, (library - counts.sum(axis=1))[:, None]])
    return Cells(features.astype(np.float32), expression, target.astype(np.float32), library.astype(np.float32))


DOSE_TRANSFORMS = {
    # chemCPA's convention (µM, log1p): 10/100/1000/10000 nM -> 0.010/0.095/0.69/2.40
    "log1p_uM": lambda dose_nM: np.log1p(dose_nM / 1000),
    # evenly spaced decades: 10/100/1000/10000 nM -> 0.25/0.5/0.75/1; vehicle stays 0
    "log10_nM": lambda dose_nM: np.where(dose_nM > 0, np.log10(np.maximum(dose_nM, 1)) / 4, 0.0),
}


def dose_transform(config):
    return config.get("dose_transform", "log1p_uM")


def encode_dose(dose_nM, transform):
    """Doser input for doses in nM (vehicle = 0 maps to 0, which the doser gates off)."""
    return np.asarray(DOSE_TRANSFORMS[transform](np.asarray(dose_nM, dtype=np.float64)), dtype=np.float64)


def training_arrays(panel, spec, drugs, likelihood, outputs="all", transform="log1p_uM"):
    mask = panel.obs.split.isin(["train", "control"]).to_numpy()
    obs = panel.obs.loc[mask]
    cells = cell_inputs(panel, spec, mask, likelihood, outputs)
    lookup = {drug: i for i, drug in enumerate(drugs)}
    indices = np.array([lookup[d] for d in obs.drug.astype(str)], dtype=np.int64)
    # Frozen unit conversion; no statistics from withheld expression or doses.
    doses = encode_dose(obs.dose.to_numpy(dtype=np.float64), transform).astype(np.float32)
    if not np.isfinite(doses).all() or (doses < 0).any():
        raise ValueError("doses must be finite and nonnegative")
    return cells, indices, doses


def fraction_conditions(panel, spec, drugs, h, taus, transform="log1p_uM"):
    """Measured training conditions (train doses and vehicle) and their smoothed
    weak-responder fractions at `taus`; validation and test cells are never read."""
    allowed = panel.obs.split.isin(["train", "control"]).to_numpy()
    lookup = {drug: i for i, drug in enumerate(drugs)}
    drug_idx, doses, targets = [], [], []
    for cond, group in panel.obs[allowed].groupby("condition", observed=True):
        mask = allowed & panel.obs.condition.eq(cond).to_numpy()
        expression = _dense(spec.align(panel.X[mask], panel.var_names))
        drug_idx.append(lookup[str(group.drug.iloc[0])])
        doses.append(float(encode_dose(float(group.dose.iloc[0]), transform)))
        targets.append(real_targets(spec, expression, h, taus))
    return (np.array(drug_idx, dtype=np.int64), np.array(doses, dtype=np.float32),
            np.array(targets, dtype=np.float32))


class FractionLoss:
    """weight x mean over sampled training conditions and spec taus of the squared
    difference between generated and measured smoothed weak-responder fractions."""

    def __init__(self, model, spec, controls, conditions, settings, taus, seed, program_idx=None):
        self.model, self.spec, self.settings, self.taus = model, spec, settings, taus
        # Program genes' columns among the decoder outputs (spec order by default).
        self.program_idx = spec.program_idx if program_idx is None else program_idx
        device = model.device
        self.features = torch.from_numpy(controls.features).to(device)
        self.library = torch.from_numpy(controls.library).to(device)
        self.drug, self.dose, self.target = (torch.from_numpy(v).to(device) for v in conditions)
        self.generator = torch.Generator().manual_seed(seed + 30000)

    def __call__(self):
        n_cond, n_cells = self.settings["conditions_per_step"], self.settings["cells_per_condition"]
        device = self.model.device
        conds = torch.randint(len(self.target), (n_cond,), generator=self.generator).to(device)
        cells = torch.randint(len(self.library), (n_cond * n_cells,), generator=self.generator).to(device)
        per_cell = conds.repeat_interleave(n_cells)
        output, _ = self.model.predict(self.features[cells], drugs_idx=self.drug[per_cell],
                                       dosages=self.dose[per_cell], library=self.library[cells],
                                       covariates=[torch.ones((len(cells), 1), device=device)])
        mean, theta = output.chunk(2, dim=1)
        soft = expected_soft_indicator(mean, theta, self.library[cells], self.spec,
                                       self.settings["h"], self.taus, self.settings["max_count"],
                                       program_idx=self.program_idx)
        generated = soft.view(n_cond, n_cells, -1).mean(1)
        return self.settings["weight"] * ((generated - self.target[conds]) ** 2).mean()


def mean_conditions(panel, spec, drugs, likelihood, outputs="all", transform="log1p_uM"):
    """Every training condition and vehicle with its measured population mean over the decoder's
    output features: log1p(1e4 x mean count proportion) under NB (the rest feature included),
    mean log1p(CP10K) under the Gaussian likelihood. Validation and test cells are never read."""
    allowed = panel.obs.split.isin(["train", "control"]).to_numpy()
    lookup = {drug: i for i, drug in enumerate(drugs)}
    drug_idx, doses, targets = [], [], []
    for cond, group in panel.obs[allowed].groupby("condition", observed=True):
        mask = allowed & panel.obs.condition.eq(cond).to_numpy()
        cells = cell_inputs(panel, spec, mask, likelihood, outputs)
        if likelihood == "nb":
            target = np.log1p(1e4 * (cells.counts / cells.library[:, None]).mean(0))
        else:
            target = cells.expression.mean(0)
        drug_idx.append(lookup[str(group.drug.iloc[0])])
        doses.append(float(encode_dose(float(group.dose.iloc[0]), transform)))
        targets.append(target)
    return (np.array(drug_idx, dtype=np.int64), np.array(doses, dtype=np.float32),
            np.array(targets, dtype=np.float32))


class MeanLoss:
    """weight x mean over sampled training conditions and output features of the squared
    difference between the counterfactual population's mean (permitted controls pushed through
    the condition, as at prediction time) and the measured condition's mean. chemCPA's own
    objective only reconstructs each cell through its own encoding, so this is the only term
    that supervises the counterfactual path at the population level."""

    def __init__(self, model, controls, conditions, settings, seed):
        self.model, self.settings = model, settings
        device = model.device
        self.features = torch.from_numpy(controls.features).to(device)
        self.library = None if controls.library is None else torch.from_numpy(controls.library).to(device)
        self.drug, self.dose, self.target = (torch.from_numpy(v).to(device) for v in conditions)
        self.generator = torch.Generator().manual_seed(seed + 40000)

    def __call__(self):
        n_cond, n_cells = self.settings["conditions_per_step"], self.settings["cells_per_condition"]
        device = self.model.device
        conds = torch.randint(len(self.target), (n_cond,), generator=self.generator).to(device)
        cells = torch.randint(len(self.features), (n_cond * n_cells,), generator=self.generator).to(device)
        per_cell = conds.repeat_interleave(n_cells)
        library = None if self.library is None else self.library[cells]
        output, _ = self.model.predict(self.features[cells], drugs_idx=self.drug[per_cell],
                                       dosages=self.dose[per_cell], library=library,
                                       covariates=[torch.ones((len(cells), 1), device=device)])
        mean, _ = output.chunk(2, dim=1)
        if self.model.loss_ae == "nb":
            mean = torch.log1p(1e4 * (mean / library[:, None]).view(n_cond, n_cells, -1).mean(1))
        else:
            mean = mean.view(n_cond, n_cells, -1).mean(1)
        return self.settings["weight"] * ((mean - self.target[conds]) ** 2).mean()


class SumLoss:
    def __init__(self, parts):
        self.parts = parts

    def __call__(self):
        return sum(part() for part in self.parts)


def make_model(n_inputs, fingerprints, config, seed, device, n_outputs=None):
    # Upstream randomizes hyperparameters when seed != 0 unless a complete dict
    # is supplied. Hold architecture/optimizer settings fixed across seeds.
    hparams = dict(ComPert.defaults())
    hparams.update({"reg_adversary_cov": 0, "reg_multi_task": 0})
    hparams.update(config["hparams"])
    hparams["batch_size"] = config["batch_size"]
    if hparams["adversary_steps"] < 2:
        raise ValueError("adversary_steps must be >= 2 so the autoencoder is updated")
    seed_all(seed)
    embedding = torch.nn.Embedding.from_pretrained(torch.as_tensor(fingerprints), freeze=True)
    model = ComPert(
        num_genes=n_inputs, num_drugs=len(fingerprints), num_covariates=[1],
        drug_embeddings=embedding, device=str(device), seed=seed,
        doser_type="amortized", decoder_activation="linear", hparams=hparams,
        use_drugs_idx=True, loss_ae=config["likelihood"], num_outputs=n_outputs,
    )
    model.dose_transform = dose_transform(config)
    return model


def minibatches(n, batch_size, generator):
    """Keep every cell while preventing singleton training BatchNorm batches."""
    if n < 2 or batch_size < 2:
        raise ValueError("at least two training cells and batch_size >= 2 are needed")
    batches = list(torch.randperm(n, generator=generator).split(batch_size))
    if len(batches) > 1 and len(batches[-1]) == 1:
        last = batches.pop()
        batches[-1] = torch.cat([batches[-1], last])
    return batches


def prediction_inputs(panel, spec, split, likelihood, outputs="all"):
    """Separate permitted expression from held-out metadata before prediction."""
    controls = cell_inputs(panel, spec, panel.obs.split.eq("control").to_numpy(), likelihood, outputs)
    if len(controls.features) < 2:
        raise ValueError("at least two permitted control cells are required")
    # Grouping metadata is allowed; withheld expression is never passed to generate.
    rows = []
    for cond, group in panel.obs[panel.obs.split == split].groupby("condition", observed=True):
        rows.append({"condition": str(cond), "drug": str(group.drug.iloc[0]),
                     "dose_nM": float(group.dose.iloc[0]), "n_cells": len(group)})
    if not rows:
        raise ValueError(f"no {split} conditions")
    return controls, sorted(rows, key=lambda row: row["condition"])


def decode(mean, spread, likelihood, mode, rng):
    """Map decoder outputs to log1p(CP10K) cells over the generated genes.

    Gaussian samples keep their original support (no posthoc clipping or threshold
    calibration); the shared response score clips negatives itself. NB samples are
    Gamma-Poisson count draws normalized by their generated total, the rest-of-
    transcriptome feature included, exactly as real cells are normalized.
    """
    if likelihood == "gauss":
        if mode == "sample":
            mean = mean + rng.standard_normal(mean.shape) * np.sqrt(spread)
        return mean
    if mode == "sample":
        mean = rng.poisson(rng.gamma(spread, mean / spread)).astype(np.float64)
    total = np.maximum(mean.sum(axis=1, keepdims=True), 1)
    return np.log1p(mean[:, :-1] / total * 1e4)


@torch.inference_mode()
def generate(model, controls, conditions, drugs, gene_ids, *, n_cells, seed, mode="sample"):
    """Push permitted control cells through chemCPA and decode one cell per input.

    `sample` draws from the fitted likelihood; `mean` is a separately labelled
    diagnostic for the common decoder-mean usage. Under NB, each generated cell
    keeps the library size of the control cell it was generated from.
    """
    if mode not in GENERATIONS or n_cells < 2:
        raise ValueError(f"mode must be one of {GENERATIONS} and n_cells >= 2")
    model.eval()
    rng = np.random.default_rng(seed)
    lookup = {d: i for i, d in enumerate(drugs)}
    generated, rows = [], []
    for cond in conditions:
        if cond["drug"] not in lookup:
            raise ValueError(f"drug missing from molecular inputs: {cond['drug']}")
        n = max(n_cells, cond["n_cells"])
        picked = rng.integers(len(controls.features), size=n)
        chunks = []
        for start in range(0, n, 512):
            batch = picked[start:start + 512]
            x = torch.as_tensor(controls.features[batch], device=model.device)
            library = None if controls.library is None else torch.as_tensor(controls.library[batch], device=model.device)
            idx = torch.full((len(x),), lookup[cond["drug"]], dtype=torch.long, device=model.device)
            dose = torch.full((len(x),), float(encode_dose(cond["dose_nM"], model.dose_transform)),
                              device=model.device)
            output, _ = model.predict(x, drugs_idx=idx, dosages=dose, library=library,
                                      covariates=[torch.ones((len(x), 1), device=model.device)])
            mean, spread = np.split(output.cpu().numpy().astype(np.float64), 2, axis=1)
            cells = decode(mean, spread, model.loss_ae, mode, rng)
            if not np.isfinite(cells).all():
                raise RuntimeError("nonfinite chemCPA predictions")
            chunks.append(cells.astype(np.float32))
        generated.append(np.vstack(chunks))
        rows.extend([{"condition": cond["condition"]}] * n)
    return ad.AnnData(np.vstack(generated), obs=pd.DataFrame(rows, index=[f"pred_{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=gene_ids))


def save_scores(out, spec, panel, pred, split, meta, program_only=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = evaluate_predictions(spec, panel, pred, split=split, seed=meta["seed"], program_only=program_only)
    align = spec.align_program if program_only else spec.align
    write_json(out / "summary.json", {**meta, **summarize(records)})
    with (out / "per_condition.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps({**meta, **record}) + "\n")
    wells = []
    for cond in sorted(set(pred.obs.condition)):
        real_obs = panel.obs[panel.obs.condition == cond]
        xp = align(pred.X[(pred.obs.condition == cond).to_numpy()], pred.var_names)
        for key, group in real_obs.groupby(["replicate", "culture_plate", "well_oligo"], observed=True):
            real = spec.align(panel[group.index].X, panel.var_names)
            wells.append({**meta, "condition": cond, "replicate": str(key[0]),
                          "culture_plate": str(key[1]), "well_oligo": str(key[2]),
                          **evaluate_condition(spec, real, X_pred=xp, seed=meta["seed"], program_only=program_only)})
    with (out / "per_well.jsonl").open("w") as f:
        for record in wells:
            f.write(json.dumps(record) + "\n")
    return summarize(records)


def finished_training(out, config):
    """True for a completed seed run; refuses to reuse one trained with another config."""
    if not ((out / "run.json").is_file() and (out / "model.pt").is_file()):
        return False
    run = json.loads((out / "run.json").read_text())
    if "training_seconds" not in run:
        return False
    if run["config"] != config:
        raise ValueError(f"{out} was trained with a different config; use a new --out")
    return True


def train_one(panel, spec, config, drugs, fingerprints, molecular_meta, args, seed):
    out = Path(args.out) / f"seed_{seed}"
    if finished_training(out, config):
        print(f"seed={seed}: finished run exists, skipping {out}", flush=True)
        return
    if out.exists():
        print(f"seed={seed}: discarding unfinished run {out}", flush=True)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    likelihood, outputs = config["likelihood"], output_genes(config)
    program_only, gene_ids = outputs == "program", output_gene_ids(spec, config)
    model = make_model(n_features(spec, likelihood), fingerprints, config, seed, args.device, n_outputs(spec, config))
    transform = dose_transform(config)
    cells, idx, dose = training_arrays(panel, spec, drugs, likelihood, outputs, transform)
    tensors = {"x": cells.features, "idx": idx, "dose": dose}
    if likelihood == "nb":
        tensors.update(counts=cells.counts, library=cells.library)
    tensors = {k: torch.from_numpy(v) for k, v in tensors.items()}
    controls, val_conditions = prediction_inputs(panel, spec, "val", likelihood, outputs)
    fraction, taus = None, None
    if "fraction_loss" in config:
        settings = config["fraction_loss"]
        taus = supervised_taus(spec, controls.expression, settings.get("quantiles"))
        fraction = FractionLoss(model, spec, controls,
                                fraction_conditions(panel, spec, drugs, settings["h"], taus, transform),
                                settings, taus, seed, program_idx=output_program_idx(spec, config))
    aux = [fraction] if fraction is not None else []
    mean_loss = None
    if "mean_loss" in config:
        mean_loss = MeanLoss(model, controls, mean_conditions(panel, spec, drugs, likelihood, outputs, transform),
                             config["mean_loss"], seed)
        aux.append(mean_loss)
    aux_loss = None if not aux else aux[0] if len(aux) == 1 else SumLoss(aux)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {
        "method": method_name(config), "seed": seed, "config": config,
        "upstream_commit": UPSTREAM, "spec_hashes": spec_hash(args.spec),
        "split_sha256": sha256(args.split_config), "panel_sha256": sha256(args.data),
        "model_source_sha256": sha256(Path(__file__).parent / "vendor" / "chemcpa_model.py"),
        "adapter_source_sha256": sha256(__file__),
        "data": str(Path(args.data).resolve()), "n_train_and_control": len(idx),
        "n_controls": len(controls.features), "n_parameters": n_parameters,
        "device": args.device, "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "numpy", "pandas", "scipy", "anndata", "rdkit", "scikit-learn")},
        "molecular_inputs": molecular_meta,
        "selection": f"minimum equal-condition validation {config['selection_metric']}",
        "training_splits": ["train", "control"], "likelihood": likelihood,
        "output_genes": outputs, "n_inputs": n_features(spec, likelihood), "n_outputs": n_outputs(spec, config),
        "fraction_loss": config.get("fraction_loss"),
        "fraction_loss_conditions": None if fraction is None else len(fraction.target),
        "fraction_loss_taus": None if taus is None else [float(t) for t in taus],
        "generation": config["generation"],
        "dose_transform": {"log1p_uM": "log1p(dose_nM / 1000)", "log10_nM": "log10(dose_nM) / 4"}[transform],
        "mean_loss": config.get("mean_loss"),
        "mean_loss_conditions": None if mean_loss is None else len(mean_loss.target),
    }
    write_json(out / "run.json", meta)
    generator = torch.Generator().manual_seed(seed)
    best = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals, steps = {}, 0
        for batch in minibatches(len(idx), config["batch_size"], generator):
            b = {k: v[batch].to(args.device) for k, v in tensors.items()}
            stats = model.update(b["x"], drugs_idx=b["idx"], dosages=b["dose"],
                                 counts=b.get("counts"), library=b.get("library"), aux_loss=aux_loss,
                                 covariates=[torch.ones((len(batch), 1), device=args.device)])
            if not all(np.isfinite(v) for v in stats.values()):
                raise RuntimeError(f"nonfinite training loss at epoch {epoch}: {stats}")
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            steps += 1
        for scheduler in (model.scheduler_autoencoder, model.scheduler_adversary, model.scheduler_dosers):
            scheduler.step()
        history = {"epoch": epoch, **{k: v / steps for k, v in totals.items()},
                   "elapsed_seconds": time.monotonic() - started}
        if epoch % config["validate_every"] == 0 or epoch == config["epochs"]:
            # Common random numbers across checkpoints reduce simulation noise.
            pred = generate(model, controls, val_conditions, drugs, gene_ids,
                            n_cells=config["n_prediction_cells"], seed=seed + 10000, mode=config["generation"])
            summary = summarize(evaluate_predictions(spec, panel, pred, "val", seed=seed, program_only=program_only))
            history["validation"] = summary
            score = summary[config["selection_metric"]]["mean"]
            if score < best:
                best = score
                checkpoint = {"model": model.state_dict(), "config": config, "drugs": drugs,
                              "fingerprints": torch.from_numpy(fingerprints), "gene_ids": spec.gene_ids,
                              "output_gene_ids": gene_ids,
                              "epoch": epoch, "seed": seed, "spec_hashes": meta["spec_hashes"],
                              "split_sha256": meta["split_sha256"], "panel_sha256": meta["panel_sha256"]}
                torch.save(checkpoint, out / "model.pt")
                write_json(out / "selected.json", {"epoch": epoch, "selection_metric": config["selection_metric"],
                                                     "score": best, "validation": summary})
            shown = " ".join(f"val_{k}={summary[k]['mean']:.5f}" for k in ("mmd2", "w1_r") if k in summary)
            print(f"seed={seed} epoch={epoch} nll={history['loss_reconstruction']:.4f} "
                  f"aux={history['loss_auxiliary']:.5f} {shown} "
                  f"val_wrf_pp={summary[f'wrf_err_pp_q{spec.primary_q:g}']['mean']:.3f}", flush=True)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(history) + "\n")
    meta["training_seconds"] = time.monotonic() - started
    write_json(out / "run.json", meta)
    print(f"saved {out / 'model.pt'} ({meta['training_seconds']:.1f}s)", flush=True)


def evaluate_run(panel, spec, args, seed):
    run = Path(args.out) / f"seed_{seed}"
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["spec_hashes"] != spec_hash(args.spec) or checkpoint["gene_ids"] != spec.gene_ids:
        raise ValueError("evaluation spec differs from training")
    if checkpoint["split_sha256"] != sha256(args.split_config) or checkpoint["panel_sha256"] != sha256(args.data):
        raise ValueError("panel or frozen split differs from training")
    config = checkpoint["config"]
    likelihood, outputs = config["likelihood"], output_genes(config)
    program_only, gene_ids = outputs == "program", output_gene_ids(spec, config)
    if checkpoint.get("output_gene_ids", spec.gene_ids) != gene_ids:
        raise ValueError("checkpoint output genes differ from its config")
    mode = args.generation or config["generation"]
    suffix = f"{args.split}_{mode}_s{args.sampling_seed}"
    out = run / suffix
    if all((out / name).is_file() for name in EVAL_OUTPUTS):
        print(f"seed={seed} {suffix}: finished evaluation exists, skipping", flush=True)
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = make_model(n_features(spec, likelihood), checkpoint["fingerprints"], config, checkpoint["seed"],
                       args.device, n_outputs(spec, config))
    model.load_state_dict(checkpoint["model"])
    controls, conditions = prediction_inputs(panel, spec, args.split, likelihood, outputs)
    sample_seed = seed + (10000 if args.split == "val" else 20000) + args.sampling_seed
    pred = generate(model, controls, conditions, checkpoint["drugs"], gene_ids,
                    n_cells=config["n_prediction_cells"], seed=sample_seed, mode=mode)
    pred.uns["chemcpa"] = {"checkpoint_sha256": sha256(run / "model.pt"), "epoch": checkpoint["epoch"],
                            "seed": seed, "likelihood": likelihood, "generation": mode,
                            "sampling_seed": sample_seed, "output_genes": outputs}
    pred.write_h5ad(out / "predictions.h5ad", compression="gzip")
    meta = {"method": f"{method_name(config)}_{mode}", "seed": seed, "split": args.split,
            "spec_version": spec.version, "epoch": checkpoint["epoch"],
            "likelihood": likelihood, "generation": mode, "output_genes": outputs,
            "sampling_seed": sample_seed, "checkpoint_sha256": sha256(run / "model.pt")}
    summary = save_scores(out, spec, panel, pred, args.split, meta, program_only=program_only)
    # Decoder reconstruction diagnostics on controls only, separate from scores.
    n_controls = len(controls.features)
    ctrl_cond = [{"condition": "control_diagnostic", "drug": "Vehicle", "dose_nM": 0, "n_cells": n_controls}]
    diagnostic = {}
    for diagnostic_mode in GENERATIONS:
        control_pred = generate(model, controls, ctrl_cond, checkpoint["drugs"], gene_ids,
                                n_cells=n_controls, seed=sample_seed, mode=diagnostic_mode)
        xp = spec.align_program(control_pred.X, gene_ids) if program_only else control_pred.X
        diagnostic[diagnostic_mode] = evaluate_condition(spec, controls.expression, X_pred=xp, seed=seed,
                                                         program_only=program_only)
    write_json(out / "control_diagnostic.json", diagnostic)
    print(f"seed={seed} {suffix}: WRF MAE={summary[f'wrf_err_pp_q{spec.primary_q:g}']['mean']:.3f} pp", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--data", default=DEFAULT_PANEL)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--split-config", default=str(REPO / "configs/split_v1.json"))
    parser.add_argument("--config", default=str(REPO / "configs/chemcpa_nb_v1.json"))
    parser.add_argument("--chemicals", help="GEO hashTable_metadata.txt.gz or TSV with drug/name and SMILES")
    parser.add_argument("--out", required=True, help="run root, containing seed_N subdirectories")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--generation", choices=GENERATIONS, help="evaluation-only diagnostic override")
    parser.add_argument("--sampling-seed", type=int, default=0, help="additional evaluation sampling offset")
    args = parser.parse_args(argv)
    if args.command == "train" and (args.split != "val" or args.generation is not None):
        parser.error("training uses validation only; set generation in the training config")
    if args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("threads must be positive and seeds must be unique")
    torch.set_num_threads(args.threads)
    panel, spec = load_panel(args.data), EvalSpec.load(args.spec)
    manifest = json.loads(Path(args.split_config).read_text())
    validate_panel(panel, spec, manifest)
    if args.command == "train":
        config = json.loads(Path(args.config).read_text())
        if any(config[k] < 1 for k in ("epochs", "validate_every")) or config["batch_size"] < 2:
            parser.error("epochs/validate_every must be positive and batch size >= 2")
        if (config.get("likelihood") not in LIKELIHOODS or config["generation"] not in GENERATIONS
                or config["n_prediction_cells"] < 2):
            parser.error("invalid likelihood/generation configuration")
        settings = config.get("fraction_loss")
        if settings is not None and (
                config["likelihood"] != "nb" or settings["weight"] < 0 or settings["h"] <= 0
                or min(settings[k] for k in ("conditions_per_step", "cells_per_condition", "max_count")) < 1):
            parser.error("fraction_loss needs the nb likelihood, weight >= 0, h > 0 and positive sizes")
        if settings is not None and not all(0 < q < 1 for q in settings.get("quantiles", [0.5])):
            parser.error("fraction_loss quantiles must lie in (0, 1)")
        if dose_transform(config) not in DOSE_TRANSFORMS:
            parser.error(f"dose_transform must be one of {sorted(DOSE_TRANSFORMS)}")
        ml = config.get("mean_loss")
        if ml is not None and (ml["weight"] < 0 or min(ml["conditions_per_step"], ml["cells_per_condition"]) < 1):
            parser.error("mean_loss needs weight >= 0 and positive sizes")
        if output_genes(config) not in OUTPUT_GENES or (output_genes(config) == "program" and config["likelihood"] != "nb"):
            parser.error(f"output_genes must be one of {OUTPUT_GENES}; 'program' needs the nb likelihood")
        selectable = {"mmd2", "w1_r", f"wrf_err_pp_q{spec.primary_q:g}", "rmse_mean_all"}
        if output_genes(config) == "program":  # no non-program genes to score MMD² or all-gene RMSE on
            selectable -= {"mmd2", "rmse_mean_all"}
        if config["selection_metric"] not in selectable:
            parser.error(f"unsupported validation selection metric; choose from {sorted(selectable)}")
        obs = panel.obs[panel.obs.split.isin(["train", "control"])]
        chemicals = pd.read_csv(args.chemicals, sep="\t") if args.chemicals else None
        drugs, fingerprints, molecular_meta = molecular_inputs(obs, chemicals)
        for seed in args.seeds:
            train_one(panel, spec, config, drugs, fingerprints, molecular_meta, args, seed)
    else:
        for seed in args.seeds:
            evaluate_run(panel, spec, args, seed)


if __name__ == "__main__":
    main()
