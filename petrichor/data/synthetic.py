"""
Synthetic odor-valence dataset for capacity proofs.

WHY SYNTHETIC
-------------
The architectural claim: a valence head conditioned on an internal-state scalar
can represent the same molecule flipping good <-> bad as the body's stake
changes, where a state-blind descriptor model provably cannot. Capacity is
tested cleanest on data whose generative rule we control exactly.

THE GENERATIVE RULE
-------------------
    valence(m, s) = base(m) + food(m) * A * (1 - 2s) + noise

  - m : molecule, D descriptor dims (POM-embedding stand-in)
  - s : internal state in [0, 1] (satiety: 0 = starving, 1 = sated)
  - base(m) : intrinsic state-independent valence
  - food(m) : food-association in {0, 1}; only food odorants are state-modulated
  - A : flip amplitude; with A > |base(m)| for food, sign flips as s: 0 -> 1

PROVABLE CONSEQUENCE
--------------------
A state-blind model sees only m, so for fixed m it emits one number.
MSE-optimal over s ~ U[0,1] is E_s[valence] = base(m), and irreducible MSE on
food molecules is Var_s[A(1-2s)] = A^2 / 3.
"""

from __future__ import annotations

import numpy as np

D_DESCRIPTORS = 32
FLIP_AMPLITUDE = 1.5
NOISE_SIGMA = 0.10
FOOD_FRACTION = 0.5
SAMPLES_PER_MOLECULE = 16


def irreducible_baseline_food_mse(amplitude: float = FLIP_AMPLITUDE) -> float:
    """Provable lower bound on a state-blind model's MSE for food molecules.

    Var over s~U[0,1] of A*(1-2s) = A^2 * 4 * Var(s) = A^2 / 3.
    """
    return (amplitude ** 2) / 3.0


def true_valence(base: float, food: int, s: float, amplitude: float = FLIP_AMPLITUDE) -> float:
    """Noise-free ground-truth valence at internal state s."""
    return float(base + food * amplitude * (1.0 - 2.0 * s))


def _base_valence(descriptors: np.ndarray, weights: np.ndarray, food: np.ndarray) -> np.ndarray:
    """Intrinsic valence. Food molecules kept near 0 so +/-A crosses zero."""
    raw = np.tanh(descriptors @ weights)
    shrink = np.where(food == 1, 0.25, 1.0)
    return raw * shrink


def make_dataset(
    n_molecules: int = 1500,
    seed: int = 0,
    amplitude: float = FLIP_AMPLITUDE,
    noise_sigma: float = NOISE_SIGMA,
    d_descriptors: int = D_DESCRIPTORS,
    samples_per_molecule: int = SAMPLES_PER_MOLECULE,
    food_fraction: float = FOOD_FRACTION,
) -> dict:
    """Build molecule x state valence dataset.

    Returns dict with X_desc, state, y, is_food, mol_id, molecules, mol_food,
    base, meta.
    """
    rng = np.random.default_rng(seed)

    molecules = rng.standard_normal((n_molecules, d_descriptors))

    # Food-association is structural (readable from descriptors), as in real
    # chemistry: food odorants cluster. A linear region of descriptor space.
    w_food = rng.standard_normal(d_descriptors) / np.sqrt(d_descriptors)
    food_score = molecules @ w_food
    thresh = np.quantile(food_score, 1.0 - food_fraction)
    mol_food = (food_score > thresh).astype(int)

    weights = rng.standard_normal(d_descriptors) / np.sqrt(d_descriptors)
    base = _base_valence(molecules, weights, mol_food)

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
            "d_descriptors": d_descriptors,
            "source": "synthetic",
        },
    }
