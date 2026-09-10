"""Falsifiable core: same sensors, wrong objective must not recover the flip."""

from __future__ import annotations

import pytest

from petrichor import (
    StateAsFeatureWrongObjective,
    StateBlindBaseline,
    StateConditionedHead,
    evaluate_flip,
    make_dataset,
    split_by_molecule,
)
from petrichor.metrics import build_pairs, score_choices


@pytest.fixture(scope="module")
def results():
    data = make_dataset(n_molecules=400, seed=0)
    train_mask, test_mask = split_by_molecule(data, seed=1)
    Xtr = data["X_desc"][train_mask]
    str_ = data["state"][train_mask]
    ytr = data["y"][train_mask]
    mol = data["mol_id"][train_mask]

    blind = StateBlindBaseline(seed=0).fit(Xtr, str_, ytr)
    cond = StateConditionedHead(seed=0).fit(Xtr, str_, ytr)
    same = StateAsFeatureWrongObjective(seed=0).fit(Xtr, str_, ytr, mol_id=mol)

    flip_pairs, _ = build_pairs(data, test_mask, seed=2)
    return {
        "blind_flip": evaluate_flip(blind, data, test_mask),
        "cond_flip": evaluate_flip(cond, data, test_mask),
        "same_flip": evaluate_flip(same, data, test_mask),
        "blind_choice": score_choices(blind, data, flip_pairs),
        "cond_choice": score_choices(cond, data, flip_pairs),
        "same_choice": score_choices(same, data, flip_pairs),
    }


def test_same_info_flip_rate_near_zero(results):
    # State bits present; averaged target => should not learn the sign flip.
    assert results["same_flip"]["flip_rate_food"] <= 0.15


def test_conditioned_flip_high(results):
    assert results["cond_flip"]["flip_rate_food"] >= 0.85


def test_same_info_choice_near_ceiling(results):
    # Behaviourally behaves like the blind chooser on flip pairs.
    assert results["same_choice"]["accuracy"] <= 0.60


def test_conditioned_choice_high(results):
    assert results["cond_choice"]["accuracy"] >= 0.85


def test_same_info_not_sneaking_past_blind(results):
    # The control must not beat the conditioned head; gap is the claim.
    assert results["cond_flip"]["flip_rate_food"] - results["same_flip"]["flip_rate_food"] >= 0.6
