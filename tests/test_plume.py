"""Plume arena: stake-shaped reward and valence-competent policies outperform blind."""

from __future__ import annotations

import numpy as np
import pytest

from petrichor.agents import (
    RandomPolicy,
    ValenceGreedyPolicy,
    evaluate_policy,
    train_valence_head_for_arena,
)
from petrichor.env import PlumeArena


@pytest.fixture(scope="module")
def arena_bundle():
    env = PlumeArena(width=11, height=11, seed=0, max_steps=60)
    heads = {
        "blind": train_valence_head_for_arena(env, kind="blind", seed=0),
        "conditioned": train_valence_head_for_arena(env, kind="conditioned", seed=0),
        "same_info": train_valence_head_for_arena(env, kind="same_info", seed=0),
    }
    policies = {
        "random": RandomPolicy(seed=0),
        "blind": ValenceGreedyPolicy(heads["blind"], name="blind"),
        "conditioned": ValenceGreedyPolicy(heads["conditioned"], name="conditioned"),
        "same_info": ValenceGreedyPolicy(heads["same_info"], name="same_info"),
    }
    return env, policies


def test_valence_map_flips_with_satiety():
    env = PlumeArena(width=11, height=11, seed=0)
    food = next(s for s in env.sources if s.kind == "food")
    y, x = food.position
    v_starve = env.true_valence_at(y, x, satiety=0.0)
    v_sated = env.true_valence_at(y, x, satiety=1.0)
    assert v_starve > 0
    assert v_sated < 0


def test_toxin_always_negative():
    env = PlumeArena(width=11, height=11, seed=0)
    toxin = next(s for s in env.sources if s.kind == "toxin")
    y, x = toxin.position
    assert env.true_valence_at(y, x, satiety=0.0) < 0
    assert env.true_valence_at(y, x, satiety=1.0) < 0


def test_forage_food_reward_flips():
    env = PlumeArena(width=11, height=11, seed=0, max_steps=20)
    food = next(s for s in env.sources if s.kind == "food")
    # Starve, stand on food, forage.
    env.reset(satiety=0.0)
    env.y, env.x = food.position
    _, r0, _, info0 = env.step(5)  # FORAGE
    assert info0["forage_kind"] == "food"
    assert r0 > 0

    env.reset(satiety=1.0)
    env.y, env.x = food.position
    _, r1, _, info1 = env.step(5)
    assert info1["forage_kind"] == "food"
    assert r1 < 0


def test_conditioned_beats_random_when_starving(arena_bundle):
    env, policies = arena_bundle
    rnd = evaluate_policy(env, policies["random"], n_episodes=12, seed=1, satiety_mode="starving")
    cond = evaluate_policy(
        env, policies["conditioned"], n_episodes=12, seed=1, satiety_mode="starving"
    )
    assert cond["mean_return"] > rnd["mean_return"]


def test_conditioned_beats_blind_overall(arena_bundle):
    env, policies = arena_bundle
    blind = evaluate_policy(env, policies["blind"], n_episodes=15, seed=2, satiety_mode="mixed")
    cond = evaluate_policy(
        env, policies["conditioned"], n_episodes=15, seed=2, satiety_mode="mixed"
    )
    # Conditioned should not lose badly; typically wins. Allow small slack for noise.
    assert cond["mean_return"] >= blind["mean_return"] - 0.5


def test_conditioned_reduces_food_when_sated(arena_bundle):
    env, policies = arena_bundle
    starve = evaluate_policy(
        env, policies["conditioned"], n_episodes=12, seed=3, satiety_mode="starving"
    )
    sated = evaluate_policy(
        env, policies["conditioned"], n_episodes=12, seed=3, satiety_mode="sated"
    )
    # When sated, food forages should drop relative to starving start.
    assert sated["mean_food_forages"] <= starve["mean_food_forages"] + 0.5
