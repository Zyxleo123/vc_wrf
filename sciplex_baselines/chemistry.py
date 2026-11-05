"""Fixed molecular inputs; no expression values are used to build fingerprints."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator


def molecular_inputs(obs: pd.DataFrame, metadata: pd.DataFrame | None = None):
    """Return drug names, Morgan fingerprints and an auditable SMILES mapping.

    `obs` must contain only permitted train/control rows. GEO's hashTable_metadata
    is accepted as `metadata`; drug names are matched exactly after trimming.
    Unknown or ambiguous structures fail instead of falling back to drug IDs.
    Salts/disconnected fragments and stereochemistry are retained.
    """
    drugs = sorted(obs.drug.astype(str).unique())
    source = obs if "SMILES" in obs else metadata
    if source is None or "SMILES" not in source:
        raise ValueError("SMILES are missing; pass --chemicals GEO hashTable_metadata.txt.gz")
    name_col = "drug" if "drug" in source else "name"
    if name_col not in source:
        raise ValueError("chemical metadata needs a drug or name column and SMILES")
    names = source[name_col].astype(str).str.strip()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprints, records = [], []
    for drug in drugs:
        if drug == "Vehicle":
            fingerprints.append(np.zeros(2048, dtype=np.float32))
            records.append({"drug": drug, "smiles": None})
            continue
        values = source.loc[names == drug.strip(), "SMILES"].dropna().unique()
        canonical = set()
        for value in values:
            mol = Chem.MolFromSmiles(str(value))
            if mol is None:
                raise ValueError(f"invalid SMILES for {drug}: {value}")
            canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
        if len(canonical) != 1:
            raise ValueError(f"expected one structure for {drug}, found {len(canonical)}")
        smiles = canonical.pop()
        fingerprints.append(generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.float32))
        records.append({"drug": drug, "smiles": smiles})
    return drugs, np.stack(fingerprints), {
        "type": "Morgan", "radius": 2, "n_bits": 2048, "include_chirality": True,
        "rdkit_version": rdBase.rdkitVersion, "molecules": records,
    }
