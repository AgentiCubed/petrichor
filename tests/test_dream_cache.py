"""Real DREAM cache loads and preserves dict contract."""

from __future__ import annotations

import os

import pytest

from petrichor import load_dream, split_by_molecule


def test_dream_cache_exists():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "experiments", "state-flip", "dream_real.csv")
    assert os.path.exists(path), "committed dream_real.csv required for offline runs"


def test_load_dream_shape():
    data = load_dream(seed=0)
    n = data["meta"]["n_molecules"]
    assert n >= 100
    assert data["molecules"].shape[0] == n
    assert data["X_desc"].shape[0] == n * data["meta"]["samples_per_molecule"]
    assert set(data["mol_food"]).issubset({0, 1})
    assert "source" in data["meta"]
    train, test = split_by_molecule(data, seed=1)
    assert train.sum() > 0 and test.sum() > 0
