"""DOLORIS's evaluation metrics (released test.py / test_util.py), for any method's generated cells.

Per held-out condition, generated cells vs all measured cells of that condition, in the panel's
log1p(CP10K) space:

  RMSE        compute_rmse: RMSE between the per-gene means (mean over cells, then over genes)
  E-distance  compute_e_distance: sqrt(max(2 E|x-y| - E|x-x'| - E|y-y'|, 0)), Euclidean distances
              in gene space, within-sample means over distinct pairs (pdist)
  EMD         mean over genes of the 1-D Wasserstein-1 distance between the gene's values
              (scipy.stats.wasserstein_distance)

each over all genes, DE20 and DE40. DE20/DE40 are the first 20/40 genes of the condition's DEG
list, built as chemCPA's `lincs_DEGs` that DOLORIS reads (chemCPA rank_genes_groups_by_cov):
scanpy rank_genes_groups, default t-test, the condition's cells against all vehicle cells of
the cell line, ranked by |score| (rankby_abs=True), 50 genes, on the expression matrix
(use_raw=False). The DEG list is computed from measured cells, as in chemCPA's processed data.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

GENE_SETS = ("all", "de20", "de40")
KINDS = ("rmse", "edist", "emd")
COLUMNS = [f"{k}_{g}" for g in GENE_SETS for k in KINDS]


def rmse_of_means(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((real.mean(0, dtype=np.float64) - pred.mean(0, dtype=np.float64)) ** 2)))


def e_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    n, m = len(X), len(Y)
    cross = cdist(X, Y).mean()
    x_term = pdist(X).sum() * 2 / (n * (n - 1))
    y_term = pdist(Y).sum() * 2 / (m * (m - 1))
    return float(np.sqrt(max(2 * cross - x_term - y_term, 0.0)))


def emd(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([wasserstein_distance(real[:, i], pred[:, i]) for i in range(real.shape[1])]))


def doloris_metrics(real: np.ndarray, pred: np.ndarray, de_idx: list[int]) -> dict:
    """All nine numbers of DOLORIS's tables for one condition; de_idx is the ranked DEG list (>= 40)."""
    out = {}
    for name, cols in (("all", slice(None)), ("de20", list(de_idx[:20])), ("de40", list(de_idx[:40]))):
        r, p = real[:, cols], pred[:, cols]
        out[f"rmse_{name}"] = rmse_of_means(r, p)
        out[f"edist_{name}"] = e_distance(p, r)
        out[f"emd_{name}"] = emd(r, p)
    return out


def de_genes(X: np.ndarray, labels: np.ndarray, gene_ids: list[str], reference: str, groups: list[str],
             n_genes: int = 50) -> dict[str, list[int]]:
    """chemCPA lincs_DEGs for `groups`: column indices into gene_ids, most significant first."""
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    keep = np.isin(labels, [reference, *groups])
    a = ad.AnnData(X=np.asarray(X[keep], dtype=np.float32),
                   obs=pd.DataFrame({"group": pd.Categorical(labels[keep])}, index=np.arange(keep.sum()).astype(str)),
                   var=pd.DataFrame(index=pd.Index(gene_ids)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)  # scanpy's frame inserts, harmless
        sc.tl.rank_genes_groups(a, groupby="group", groups=list(groups), reference=reference, method="t-test",
                                rankby_abs=True, n_genes=n_genes, use_raw=False)
    names = a.uns["rank_genes_groups"]["names"]
    pos = {g: i for i, g in enumerate(gene_ids)}
    return {g: [pos[str(x)] for x in names[g]] for g in groups}
