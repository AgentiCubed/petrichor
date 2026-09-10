"""Capacity claims for the state-flip experiment — must hold every run."""

from __future__ import annotations

import numpy as np
import pytest

from petrichor import (
    StateBlindBaseline,
    StateConditionedHead,
    evaluate_flip,
    irreducible_baseline_food_mse,
    make_dataset,
    split_by_molecule,
)


@pytest.fixture(scope="module")
def flip_results():
    data = make_dataset(n_molecules=400, seed=0)
    train_mask, test_mask = split_by_molecule(data, seed=1)
    Xtr = data["X_desc"][train_mask]
    str_ = data["state"][train_mask]
    ytr = data["y"][train_mask]
    blind = StateBlindBaseline(seed=0).fit(Xtr, str_, ytr)
    cond = StateConditionedHead(seed=0).fit(Xtr, str_, ytr)
    return {
        "data": data,
        "blind": evaluate_flip(blind, data, test_mask),
        "cond": evaluate_flip(cond, data, test_mask),
        "floor": irreducible_baseline_food_mse(data["meta"]["amplitude"]),
    }


def test_provable_floor_formula():
    assert abs(irreducible_baseline_food_mse(1.5) - 0.75) < 1e-12
    assert abs(irreducible_baseline_food_mse(3.0) - 3.0) < 1e-12


def test_blind_flip_rate_is_zero(flip_results):
    assert flip_results["blind"]["flip_rate_food"] == 0.0


def test_blind_food_mse_near_floor(flip_results):
    # Allow slack for finite samples + model approx; must be near the theorem.
    food_mse = flip_results["blind"]["test_mse_food"]
    floor = flip_results["floor"]
    assert food_mse > floor * 0.7
    assert food_mse < floor * 1.5


def test_conditioned_recovers_flip(flip_results):
    assert flip_results["cond"]["flip_rate_food"] >= 0.85


def test_conditioned_beats_blind_on_food_mse(flip_results):
    assert flip_results["cond"]["test_mse_food"] < flip_results["blind"]["test_mse_food"] * 0.5


def test_conditioned_means_straddle_zero(flip_results):
    assert flip_results["cond"]["mean_v_starving"] > 0.5
    assert flip_results["cond"]["mean_v_sated"] < -0.5


def test_blind_means_identical(flip_results):
    assert flip_results["blind"]["mean_v_starving"] == pytest.approx(
        flip_results["blind"]["mean_v_sated"], abs=1e-9
    )


def test_split_never_leaks_molecules():
    data = make_dataset(n_molecules=100, seed=2)
    train_mask, test_mask = split_by_molecule(data, seed=3)
    train_mols = set(data["mol_id"][train_mask].tolist())
    test_mols = set(data["mol_id"][test_mask].tolist())
    assert train_mols.isdisjoint(test_mols)
    assert train_mask.sum() + test_mask.sum() == len(data["y"])
