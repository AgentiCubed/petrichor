"""
Keller 2016 / DREAM olfaction set — real molecule axis for valence experiments.

WHAT IS REAL
  - molecule descriptors : RDKit descriptors of published SMILES (cached CSV)
  - base(m)              : mean human pleasantness, centered/scaled to ~[-1, 1]
  - food gate            : mean human EDIBLE rating, binarized at median

WHAT IS STILL MODELLED
  - satiety coupling food(m) * A * (1 - 2s)
    No public set ships molecule x internal-state pleasantness.
"""

from __future__ import annotations

import os

import numpy as np

from petrichor.data.synthetic import (
    FLIP_AMPLITUDE,
    NOISE_SIGMA,
    SAMPLES_PER_MOLECULE,
)

_RDKIT_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "FractionCSP3", "NumHeteroatoms", "HeavyAtomCount",
    "NumValenceElectrons", "qed", "BalabanJ", "BertzCT", "HallKierAlpha",
    "LabuteASA", "Chi0", "Chi1", "Kappa1", "Kappa2", "MaxPartialCharge",
    "MinPartialCharge",
]

# Prefer package-local cache; fall back to the historical experiment path so
# existing committed dream_real.csv keeps working without duplication.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CANDIDATE_CACHES = [
    os.path.join(_HERE, "dream_real.csv"),
    os.path.join(_REPO_ROOT, "experiments", "state-flip", "dream_real.csv"),
]


def _cache_path() -> str:
    for path in _CANDIDATE_CACHES:
        if os.path.exists(path):
            return path
    return _CANDIDATE_CACHES[0]


def _build_real_table():
    """Pull Keller 2016 via pyrfume and compute RDKit descriptors (online path)."""
    import pandas as pd
    import pyrfume
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")

    mol = pyrfume.load_data("keller_2016/molecules.csv")
    st = pyrfume.load_data("keller_2016/stimuli.csv")
    beh = pyrfume.load_data("keller_2016/behavior.csv")

    hi = st[st["Concentration"] == 1e-3][["Stimulus", "CIDs"]].copy()
    hi["CID"] = pd.to_numeric(hi["CIDs"], errors="coerce")
    hi = hi.dropna(subset=["CID"])
    hi["CID"] = hi["CID"].astype(int)

    def per_cid(label):
        sub = beh[beh["MeasurementValue"] == label].merge(hi, on="Stimulus")
        sub["v"] = pd.to_numeric(sub["Value"], errors="coerce")
        return sub.dropna(subset=["v"]).groupby("CID")["v"].mean()

    pleas = per_cid("HOW PLEASANT IS THE SMELL?").rename("pleas")
    edible = per_cid("EDIBLE").rename("edible")

    mol = mol.copy()
    mol["CID"] = mol["CID"].astype(int)
    rows = {}
    desc_fns = {name: getattr(Descriptors, name) for name in _RDKIT_DESCRIPTORS}
    for cid, smi in zip(mol["CID"], mol["CanonicalSMILES"]):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        vals = {}
        ok = True
        for name, fn in desc_fns.items():
            try:
                x = float(fn(m))
            except Exception:
                ok = False
                break
            if not np.isfinite(x):
                ok = False
                break
            vals[name] = x
        if ok:
            rows[cid] = vals

    desc = pd.DataFrame.from_dict(rows, orient="index")
    desc.index.name = "CID"
    return desc.join(pleas, how="inner").join(edible, how="inner").dropna()


def _load_real_table_arrays():
    """Load cached CSV without pandas (offline, minimal deps).

    Returns (descriptor_matrix, pleas, edible) as numpy arrays.
    Falls back to pyrfume+pandas+rdkit only if the cache is missing.
    """
    path = _cache_path()
    if os.path.exists(path):
        # header: CID, <descriptors...>, pleas, edible
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        cols = {name: i for i, name in enumerate(header)}
        for name in _RDKIT_DESCRIPTORS + ["pleas", "edible"]:
            if name not in cols:
                raise RuntimeError(f"dream cache missing column {name}: {path}")
        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        desc = raw[:, [cols[c] for c in _RDKIT_DESCRIPTORS]].astype(float)
        pleas = raw[:, cols["pleas"]].astype(float)
        edible = raw[:, cols["edible"]].astype(float)
        return desc, pleas, edible

    table = _build_real_table()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table.to_csv(path)
    desc = table[_RDKIT_DESCRIPTORS].to_numpy(dtype=float)
    pleas = table["pleas"].to_numpy(dtype=float)
    edible = table["edible"].to_numpy(dtype=float)
    return desc, pleas, edible


def load_dream(
    seed: int = 0,
    amplitude: float = FLIP_AMPLITUDE,
    noise_sigma: float = NOISE_SIGMA,
    samples_per_molecule: int = SAMPLES_PER_MOLECULE,
) -> dict:
    """Real-data path. Same dict shape as make_dataset()."""
    rng = np.random.default_rng(seed)
    desc, pleas, edible = _load_real_table_arrays()

    mu = desc.mean(0)
    sd = desc.std(0)
    sd[sd == 0] = 1.0
    molecules = (desc - mu) / sd
    n_molecules, d = molecules.shape

    base = (pleas - 50.0) / 50.0
    mol_food = (edible > np.median(edible)).astype(int)

    X_desc, state, y, is_food, mol_id = [], [], [], [], []
    for i in range(n_molecules):
        s = rng.random(samples_per_molecule)
        flip = mol_food[i] * amplitude * (1.0 - 2.0 * s)
        v = base[i] + flip + rng.normal(0, noise_sigma, samples_per_molecule)
        X_desc.append(np.tile(molecules[i], (samples_per_molecule, 1)))
        state.append(s)
        y.append(v)
        is_food.append(np.full(samples_per_molecule, mol_food[i]))
        mol_id.append(np.full(samples_per_molecule, i))

    return {
        "X_desc": np.vstack(X_desc),
        "state": np.concatenate(state),
        "y": np.concatenate(y),
        "is_food": np.concatenate(is_food).astype(int),
        "mol_id": np.concatenate(mol_id).astype(int),
        "molecules": molecules,
        "mol_food": mol_food,
        "base": base,
        "meta": {
            "amplitude": amplitude,
            "noise_sigma": noise_sigma,
            "n_molecules": n_molecules,
            "samples_per_molecule": samples_per_molecule,
            "d_descriptors": d,
            "source": (
                "keller_2016 (DREAM): real RDKit descriptors + real human "
                "pleasantness (base) + real human edibility (food gate); "
                "satiety state-coupling modelled (no public molecule x state set)"
            ),
        },
    }
