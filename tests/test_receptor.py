"""Lane C: receptor-routed valence preserves flip capacity; blind ablation does not."""

from __future__ import annotations

import numpy as np
import pytest

from petrichor import (
    ReceptorRoutedHead,
    evaluate_flip,
    make_dataset,
    split_by_molecule,
)


@pytest.fixture(scope="module")
def results():
    data = make_dataset(n_molecules=400, seed=0)
    train_mask, test_mask = split_by_molecule(data, seed=1)
    Xtr = data["X_desc"][train_mask]
    str_ = data["state"][train_mask]
    ytr = data["y"][train_mask]

    routed = ReceptorRoutedHead(n_receptors=64, seed=0).fit(Xtr, str_, ytr)
    blind = ReceptorRoutedHead(n_receptors=64, seed=0, state_blind=True).fit(Xtr, str_, ytr)
    return {
        "routed": evaluate_flip(routed, data, test_mask),
        "blind": evaluate_flip(blind, data, test_mask),
        "model": routed,
        "data": data,
    }


def test_receptor_activity_nonnegative(results):
    act = results["model"].receptor_activity(results["data"]["molecules"][:10])
    assert act.shape == (10, 64)
    assert np.isfinite(act).all()
    assert (act >= -1e-6).all()  # softplus >= 0


def test_receptor_routed_recovers_flip(results):
    assert results["routed"]["flip_rate_food"] >= 0.85


def test_receptor_blind_ablation_no_flip(results):
    assert results["blind"]["flip_rate_food"] == 0.0


def test_receptor_routed_beats_ablation_mse(results):
    assert results["routed"]["test_mse_food"] < results["blind"]["test_mse_food"] * 0.5
