"""Train/test splits that never leak molecules across the cut."""

from __future__ import annotations

import numpy as np


def split_by_molecule(data: dict, test_frac: float = 0.3, seed: int = 1):
    """Split by molecule so metrics measure generalisation to unseen odorants."""
    rng = np.random.default_rng(seed)
    n = data["meta"]["n_molecules"]
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_mols = set(perm[:n_test].tolist())
    test_mask = np.array([m in test_mols for m in data["mol_id"]])
    return ~test_mask, test_mask
