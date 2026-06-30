"""
Synthetic odor-valence dataset for the state-flip demo.

WHY SYNTHETIC (read this before trusting any number):
-----------------------------------------------------
The thesis we are testing is *architectural*, not empirical: can a valence head
conditioned on an internal-state scalar represent the same molecule flipping
good <-> bad as the body's stake changes, where a state-blind descriptor model
provably cannot? That claim is about representational capacity, and the cleanest
way to test capacity is on data whose generative rule we control exactly, so we
can prove what the optimal state-blind model is allowed to do.

The real target is the DREAM/Keller olfaction set (per-molecule, per-subject
pleasantness) modulated by a context variable like satiety. DREAM gives you the
molecule->valence half for free; nobody has published the molecule x state half
because the standard release collapses context. So the *flip* itself has to be
modelled. Here we model it with a transparent rule and SAY SO. `load_dream()`
below is the documented seam where real descriptors/pleasantness drop in; the
models in models.py do not care whether features are synthetic or Mordred/POM.

UPDATE (Block F, 2026-06-29): `load_dream()` IS NOW WIRED to real data — the
Keller 2016 / DREAM set pulled live via pyrfume. Real RDKit descriptors, real
human pleasantness (base valence), and real human edibility (the food gate)
replace the synthetic molecule axis. Only the satiety state-coupling stays
modelled, because no public set ships molecule x internal-state pleasantness.
Run `python run.py --source dream`. The synthetic path below is kept because its
state-blind failure is a *provable theorem* (closed-form floor), which real
noisy data cannot show as cleanly.

THE GENERATIVE RULE (the ground truth both models are scored against):
    valence(m, s) = base(m) + food(m) * A * (1 - 2s) + noise

  - m : a molecule, represented by D synthetic "descriptor" dims (stand-in for a
        POM-style embedding / Mordred descriptor vector).
  - s : internal state scalar in [0, 1]. Read it as satiety: 0 = starving,
        1 = fully sated.
  - base(m) : intrinsic, state-independent valence (smooth fn of descriptors).
  - food(m) : food-association in {0, 1}. Only food odorants are state-modulated.
  - A : flip amplitude. With A > |base(m)| for food molecules, the sign of
        valence flips as s goes 0 -> 1: appetising when starving (s=0, +A),
        aversive when stuffed (s=1, -A). This is the satiety-flip every animal
        with a gut shows, and the single fact a state-blind model has nowhere
        to put.

PROVABLE CONSEQUENCE (the whole point):
  A state-blind model sees only m, so for a fixed molecule it must emit ONE
  number. The number minimising its MSE over s ~ Uniform[0,1] is
      E_s[valence(m,s)] = base(m)               (the (1-2s) term has mean 0)
  so its irreducible MSE on a food molecule is
      Var_s[A(1-2s)] = A^2 * 4 * Var(s) = A^2 * 4 * (1/12) = A^2 / 3.
  That floor is not a training artefact; it is a capacity limit. The
  state-conditioned model has s as an input and can drive its error to the
  noise floor. run.py measures both against this prediction.
"""

import numpy as np

# --- generative-rule constants (the "physics" of this toy world) -------------
D_DESCRIPTORS = 32      # molecular descriptor dimensionality (POM-embedding stand-in)
FLIP_AMPLITUDE = 1.5    # A: how hard satiety swings food valence (> base spread => sign flip)
NOISE_SIGMA = 0.10      # observation noise on valence ratings
FOOD_FRACTION = 0.5     # share of molecules that are food-associated (state-modulated)
SAMPLES_PER_MOLECULE = 16   # (state, valence) observations sampled per molecule


def irreducible_baseline_food_mse(amplitude=FLIP_AMPLITUDE):
    """Provable lower bound on a state-blind model's MSE for food molecules.

    Var over s~U[0,1] of A*(1-2s) = A^2 * 4 * Var(s) = A^2 / 3.
    """
    return (amplitude ** 2) / 3.0


def _base_valence(descriptors, weights, food):
    """Intrinsic, state-independent valence. Food molecules are kept near 0 so
    the +/-A satiety swing reliably crosses zero (a clean sign flip)."""
    raw = np.tanh(descriptors @ weights)          # smooth, bounded in (-1, 1)
    # shrink intrinsic valence for food molecules so |base| < A => guaranteed flip
    shrink = np.where(food == 1, 0.25, 1.0)
    return raw * shrink


def make_dataset(n_molecules=1500, seed=0, amplitude=FLIP_AMPLITUDE,
                 noise_sigma=NOISE_SIGMA):
    """Build the molecule x state valence dataset.

    Returns a dict with:
      X_desc   : (N_obs, D)  molecule descriptors (repeated per observation)
      state    : (N_obs,)    internal-state scalar s in [0,1]
      y        : (N_obs,)    observed valence (noisy)
      is_food  : (N_obs,)    food-association flag {0,1}
      mol_id   : (N_obs,)    which molecule each observation came from
      molecules: (N, D)      unique molecule descriptor table
      mol_food : (N,)        per-molecule food flag
      base     : (N,)        per-molecule intrinsic valence
      meta      : constants used (amplitude, noise, etc.)
    """
    rng = np.random.default_rng(seed)

    molecules = rng.standard_normal((n_molecules, D_DESCRIPTORS))

    # Food-association is a STRUCTURAL property: in reality food odorants have
    # characteristic chemistry, so whether a molecule is state-modulated must be
    # readable from its descriptors. We make it a linear region of descriptor
    # space (projection past its median => ~FOOD_FRACTION are food). This is what
    # lets the state-conditioned model learn *which* molecules to modulate; if
    # food-ness were an unobservable latent, no model could gate the flip.
    w_food = rng.standard_normal(D_DESCRIPTORS) / np.sqrt(D_DESCRIPTORS)
    food_score = molecules @ w_food
    thresh = np.quantile(food_score, 1.0 - FOOD_FRACTION)
    mol_food = (food_score > thresh).astype(int)

    weights = rng.standard_normal(D_DESCRIPTORS) / np.sqrt(D_DESCRIPTORS)
    base = _base_valence(molecules, weights, mol_food)

    X_desc, state, y, is_food, mol_id = [], [], [], [], []
    for i in range(n_molecules):
        s = rng.random(SAMPLES_PER_MOLECULE)                 # states in [0,1]
        flip = mol_food[i] * amplitude * (1.0 - 2.0 * s)     # +A at s=0, -A at s=1
        v = base[i] + flip + rng.normal(0, noise_sigma, SAMPLES_PER_MOLECULE)
        X_desc.append(np.tile(molecules[i], (SAMPLES_PER_MOLECULE, 1)))
        state.append(s)
        y.append(v)
        is_food.append(np.full(SAMPLES_PER_MOLECULE, mol_food[i]))
        mol_id.append(np.full(SAMPLES_PER_MOLECULE, i))

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
            "samples_per_molecule": SAMPLES_PER_MOLECULE,
            "d_descriptors": D_DESCRIPTORS,
        },
    }


def split_by_molecule(data, test_frac=0.3, seed=1):
    """Split into train/test by MOLECULE (never the same molecule on both sides),
    so we measure generalisation to unseen odorants, not memorisation."""
    rng = np.random.default_rng(seed)
    n = data["meta"]["n_molecules"]
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_mols = set(perm[:n_test].tolist())
    test_mask = np.array([m in test_mols for m in data["mol_id"]])
    return ~test_mask, test_mask


# --- REAL DATA: Keller 2016 / DREAM olfaction set (Block F) ------------------
# A fixed panel of interpretable RDKit descriptors. This is the REAL molecular
# feature vector that replaces the synthetic `molecules` matrix. Chosen for
# robustness (all defined for small odorant molecules) and interpretability.
_RDKIT_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "FractionCSP3", "NumHeteroatoms", "HeavyAtomCount",
    "NumValenceElectrons", "qed", "BalabanJ", "BertzCT", "HallKierAlpha",
    "LabuteASA", "Chi0", "Chi1", "Kappa1", "Kappa2", "MaxPartialCharge",
    "MinPartialCharge",
]

CACHE_CSV = "dream_real.csv"   # per-molecule real table, committed for offline reproduction


def _build_real_table():
    """Pull Keller 2016 (DREAM) via pyrfume, compute RDKit descriptors, and
    aggregate per-molecule REAL pleasantness and REAL edibility. Returns a
    pandas DataFrame indexed by CID with descriptor columns + 'pleas' + 'edible'.

    REAL components:
      - descriptors : RDKit descriptors from the published CanonicalSMILES.
      - pleas       : mean human 'HOW PLEASANT IS THE SMELL?' rating (0-100),
                      high-concentration (1/1000) stimuli, averaged over subjects.
      - edible      : mean human 'EDIBLE' descriptor rating (0-100), same stimuli.
    Everything here is real measured data; nothing is modelled.
    """
    import pyrfume
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    mol = pyrfume.load_data("keller_2016/molecules.csv")
    st = pyrfume.load_data("keller_2016/stimuli.csv")
    beh = pyrfume.load_data("keller_2016/behavior.csv")

    # high-concentration (1/1000), single-CID stimuli -> CID
    hi = st[st["Concentration"] == 1e-3][["Stimulus", "CIDs"]].copy()
    hi["CID"] = pd.to_numeric(hi["CIDs"], errors="coerce")
    hi = hi.dropna(subset=["CID"]); hi["CID"] = hi["CID"].astype(int)

    def per_cid(label):
        sub = beh[beh["MeasurementValue"] == label].merge(hi, on="Stimulus")
        sub["v"] = pd.to_numeric(sub["Value"], errors="coerce")
        return sub.dropna(subset=["v"]).groupby("CID")["v"].mean()

    pleas = per_cid("HOW PLEASANT IS THE SMELL?").rename("pleas")
    edible = per_cid("EDIBLE").rename("edible")

    mol = mol.copy(); mol["CID"] = mol["CID"].astype(int)
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
                ok = False; break
            if not np.isfinite(x):
                ok = False; break
            vals[name] = x
        if ok:
            rows[cid] = vals

    desc = pd.DataFrame.from_dict(rows, orient="index")
    desc.index.name = "CID"
    table = desc.join(pleas, how="inner").join(edible, how="inner").dropna()
    return table


def _load_real_table():
    """Load the cached real per-molecule table, building (and caching) it from
    pyrfume on first run so the repo reproduces offline thereafter."""
    import os
    import pandas as pd
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CACHE_CSV)
    if os.path.exists(path):
        return pd.read_csv(path, index_col="CID")
    table = _build_real_table()
    table.to_csv(path)
    return table


def load_dream(seed=0, amplitude=FLIP_AMPLITUDE, noise_sigma=NOISE_SIGMA):
    """REAL-DATA path. Returns the SAME dict shape as make_dataset(), so the
    models in models.py and the scoring in run.py are unchanged.

    WHAT IS REAL (measured, from Keller 2016 / DREAM, ~400 odorants):
      - molecule descriptors : RDKit descriptors of the published SMILES.
      - base(m) (intrinsic valence) : mean human PLEASANTNESS rating, centered
        and scaled to ~[-1, 1]. This is real human valence, not invented.
      - food gate : mean human EDIBLE rating, binarized at its median. Whether a
        molecule is state-modulated is therefore decided by a real human
        food-association judgement, not a synthetic linear region.

    WHAT IS STILL MODELLED (and why):
      - the satiety coupling  food(m) * A * (1 - 2s).  No public olfaction set
        ships per-molecule x internal-state pleasantness; the standard DREAM
        release collapses context. So the *flip with state* is overlaid on the
        REAL base/edibility with a transparent rule (identical to make_dataset),
        and we SAY SO. The architectural claim being tested is exactly about
        this term: a state-blind model has nowhere to put it. Real chemistry and
        real human valence now stand in for the molecule axis; only the state
        axis remains a declared model.
    """
    rng = np.random.default_rng(seed)
    table = _load_real_table()

    desc = table[_RDKIT_DESCRIPTORS].to_numpy(dtype=float)
    # standardize descriptors (z-score); guard zero-variance columns
    mu = desc.mean(0); sd = desc.std(0); sd[sd == 0] = 1.0
    molecules = (desc - mu) / sd
    n_molecules, d = molecules.shape

    # REAL base valence: center pleasantness on 50 (scale midpoint) -> ~[-1,1].
    pleas = table["pleas"].to_numpy(dtype=float)
    base = (pleas - 50.0) / 50.0

    # REAL food gate: binarize edibility at its median.
    edible = table["edible"].to_numpy(dtype=float)
    mol_food = (edible > np.median(edible)).astype(int)

    X_desc, state, y, is_food, mol_id = [], [], [], [], []
    for i in range(n_molecules):
        s = rng.random(SAMPLES_PER_MOLECULE)
        flip = mol_food[i] * amplitude * (1.0 - 2.0 * s)   # MODELLED state coupling
        v = base[i] + flip + rng.normal(0, noise_sigma, SAMPLES_PER_MOLECULE)
        X_desc.append(np.tile(molecules[i], (SAMPLES_PER_MOLECULE, 1)))
        state.append(s)
        y.append(v)
        is_food.append(np.full(SAMPLES_PER_MOLECULE, mol_food[i]))
        mol_id.append(np.full(SAMPLES_PER_MOLECULE, i))

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
            "samples_per_molecule": SAMPLES_PER_MOLECULE,
            "d_descriptors": d,
            "source": "keller_2016 (DREAM): real RDKit descriptors + real human "
                      "pleasantness (base) + real human edibility (food gate); "
                      "satiety state-coupling modelled (no public molecule x state set)",
        },
    }
