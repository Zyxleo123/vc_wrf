"""Uni-Mol molecule embeddings for DOLORIS (Dataset/MoleEmb.py extract_mole_embed).

Runs in its own environment (unimol_tools + CPU torch; see slurm/doloris/defaults.sh), so it
imports nothing from this repository. Reads the drug -> SMILES table chemCPA recorded from the GEO
chemical metadata (run.json "molecular_inputs"), embeds each structure with the pretrained
Uni-Mol v1 all-hydrogen model (UniMolRepr(data_type="molecule", remove_hs=False), CLS token,
512 numbers), and writes drugs, SMILES and embeddings to an .npz plus a provenance .json.

usage: python -I sciplex_doloris/unimol_embed.py --run-json results/chemcpa/gauss_v1/seed_0/run.json \
         --weights-dir /path/to/unimol_weights --out results/doloris/unimol_embed.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-json", help="chemCPA run.json with molecular_inputs.molecules")
    src.add_argument("--smiles-csv", help="CSV with columns drug,smiles")
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        print(f"{out} exists; nothing to do")
        return
    os.environ["UNIMOL_WEIGHT_DIR"] = a.weights_dir  # read when unimol_tools is imported
    import numpy as np
    import unimol_tools
    from unimol_tools import UniMolRepr

    if a.run_json:
        molecules = json.loads(Path(a.run_json).read_text())["molecular_inputs"]["molecules"]
    else:
        import csv
        with open(a.smiles_csv) as f:
            molecules = [{"drug": r["drug"], "smiles": r["smiles"]} for r in csv.DictReader(f)]
    records = [m for m in molecules if m["smiles"]]  # the vehicle has no structure
    drugs, smiles = [m["drug"] for m in records], [m["smiles"] for m in records]
    clf = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv1", use_cuda=False)
    rep = clf.get_repr(smiles, return_atomic_reprs=True)
    emb = np.asarray(rep["cls_repr"], dtype=np.float32)
    if emb.shape != (len(drugs), 512) or not np.isfinite(emb).all():
        sys.exit(f"unexpected Uni-Mol output {emb.shape}")
    if len({e.tobytes() for e in emb}) != len(drugs):  # e.g. two names for one structure
        print(f"warning: {len(drugs) - len({e.tobytes() for e in emb})} drugs share an embedding with another drug")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, drugs=np.array(drugs), smiles=np.array(smiles), emb=emb)
    tmp.rename(out)
    weights = Path(a.weights_dir) / "mol_pre_all_h_220816.pt"
    source = Path(a.run_json or a.smiles_csv)
    meta = {"source": str(source), "source_sha256": sha256(source), "n_drugs": len(drugs),
            "model": "unimolv1 molecule_all_h (mol_pre_all_h_220816.pt), CLS token", "remove_hs": False,
            "weights_sha256": sha256(weights) if weights.exists() else None,
            "unimol_tools": getattr(unimol_tools, "__version__", None), "numpy": np.__version__}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}: {len(drugs)} drugs x {emb.shape[1]}")


if __name__ == "__main__":
    main()
