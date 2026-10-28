"""Convert the GEO sci-Plex3 download into one raw-count AnnData (.h5ad).

All 799,317 cells x 110,983 genes (human + mouse barnyard reference) are kept, with
pData as obs and a `species` column in var. X holds raw UMI counts (CSR, int32).
The ~920M-entry triplet file is streamed straight into the on-disk CSR, so peak
memory stays a few GB. Checks: entries sorted by (cell, gene), and every cell's
UMI total equals pData's n.umi.

usage: python -I scripts/build_sciplex3_h5ad.py [DATA_DIR]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

try:
    from anndata.io import write_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import write_elem

DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/user_data/yixiz/sciplex3")
RAW = DATA_DIR / "raw" / "GSE139944"
OUT = DATA_DIR / "processed" / "sciplex3_raw_counts.h5ad"
PREFIX = "GSM4150378_sciPlex3_A549_MCF7_K562_screen"
CHUNK = 1 << 22


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_obs_var():
    cells = pd.read_csv(RAW / f"{PREFIX}_cell.annotations.txt.gz", sep="\t", header=None,
                        names=["cell", "sample"])
    pdata = pd.read_csv(RAW / "GSM4150378_sciPlex3_pData.txt.gz", sep=" ", low_memory=False)
    assert (pdata["cell"].values == cells["cell"].values).all(), "pData not row-aligned with cell annotations"
    obs = pdata.set_index("cell")
    obs.index.name = None
    obs = obs.rename(columns=lambda c: c.replace(".", "_"))
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype("category")
    genes = pd.read_csv(RAW / f"{PREFIX}_gene.annotations.txt.gz", sep=" ")
    var = pd.DataFrame({
        "gene_id": genes["id"].values,
        "gene_short_name": genes["gene_short_name"].astype(str).values,
        "species": np.where(genes["id"].str.startswith("ENSMUSG"), "mouse", "human"),
    })
    var.index = var["gene_id"].values
    return obs, var


def stream_triplets():
    """Yield (gene0, cell0, count) int arrays from the 1-based gene/cell/count triplet file."""
    proc = subprocess.Popen(["pigz", "-dc", str(RAW / f"{PREFIX}_UMI.count.matrix.gz")],
                            stdout=subprocess.PIPE, bufsize=1 << 24)
    reader = pcsv.open_csv(
        proc.stdout,
        read_options=pcsv.ReadOptions(column_names=["g", "c", "n"], block_size=1 << 26),
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(column_types={"g": pa.int32(), "c": pa.int32(), "n": pa.int32()}),
    )
    for batch in reader:
        yield (batch.column("g").to_numpy() - 1, batch.column("c").to_numpy() - 1,
               batch.column("n").to_numpy())
    if proc.wait() != 0:
        raise RuntimeError("pigz failed")


def main():
    if OUT.exists():
        log(f"exists, skipping: {OUT}")
        return
    obs, var = load_obs_var()
    n_cells, n_genes = len(obs), len(var)
    log(f"obs {obs.shape}, var {var.shape}")

    tmp = OUT.with_suffix(".h5ad.tmp")
    nnz_per_cell = np.zeros(n_cells, dtype=np.int64)
    umi_per_cell = np.zeros(n_cells, dtype=np.int64)
    last_cell, last_gene, nnz = -1, -1, 0
    with h5py.File(tmp, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        X = f.create_group("X")
        X.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                        "shape": np.array([n_cells, n_genes])})
        kw = dict(shape=(0,), maxshape=(None,), dtype="int32", chunks=(CHUNK,),
                  compression="gzip", compression_opts=1)
        d_data = X.create_dataset("data", **kw)
        d_idx = X.create_dataset("indices", **kw)
        for g, c, n in stream_triplets():
            # CSR needs rows (cells) non-decreasing and columns (genes) increasing within a row
            prev_c = np.concatenate([[last_cell], c[:-1]])
            prev_g = np.concatenate([[last_gene], g[:-1]])
            same = c == prev_c
            assert (c >= prev_c).all(), "matrix not sorted by cell"
            assert (g[same] > prev_g[same]).all(), "genes not sorted within a cell"
            last_cell, last_gene = int(c[-1]), int(g[-1])
            nnz_per_cell += np.bincount(c, minlength=n_cells)
            umi_per_cell += np.bincount(c, weights=n, minlength=n_cells).astype(np.int64)
            k = len(g)
            for d, v in ((d_data, n), (d_idx, g)):
                d.resize((nnz + k,))
                d[nnz:nnz + k] = v
            if (nnz + k) // 100_000_000 > nnz // 100_000_000:
                log(f"  {(nnz + k) / 1e6:.0f}M entries, cell {last_cell + 1}/{n_cells}")
            nnz += k
        X.create_dataset("indptr", data=np.concatenate([[0], np.cumsum(nnz_per_cell)]).astype(np.int64))
        log(f"nnz={nnz}")

        mismatch = int((umi_per_cell != obs["n_umi"].values).sum())
        log(f"cells whose UMI total != pData n.umi: {mismatch}")
        assert mismatch == 0

        write_elem(f, "obs", obs)
        write_elem(f, "var", var)
        write_elem(f, "uns", {"source": json.dumps({
            "geo": "GSE139944 / GSM4150378 (sciPlex3 screen)",
            "files": [f"{PREFIX}_UMI.count.matrix.gz", f"{PREFIX}_cell.annotations.txt.gz",
                      f"{PREFIX}_gene.annotations.txt.gz", "GSM4150378_sciPlex3_pData.txt.gz"],
            "X": "raw UMI counts, cells x genes, human+mouse barnyard reference",
        })})
        for k in ("obsm", "varm", "layers", "obsp", "varp"):
            write_elem(f, k, {})

    a = ad.read_h5ad(tmp, backed="r")
    assert a.shape == (n_cells, n_genes)
    tmp.rename(OUT)
    log(f"wrote {OUT} ({OUT.stat().st_size / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
