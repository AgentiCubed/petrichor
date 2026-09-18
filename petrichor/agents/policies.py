"""
Policies for the plume arena.

ValenceGreedyPolicy: at each step, evaluate predicted valence of the five
reachable cells (N/S/E/W/stay) under current satiety; move to the best; forage
if local predicted valence exceeds a threshold.

The valence head is the entire difference between policies — architecture and
training objective decide whether the agent can track the satiety flip.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from petrichor.env.plume import (
    DOWN,
    FORAGE,
    LEFT,
    N_ACTIONS,
    RIGHT,
    STAY,
    UP,
    PlumeArena,
)
from petrichor.models import (
    ReceptorRoutedHead,
    StateBlindBaseline,
    StateConditionedHead,
    StateAsFeatureWrongObjective,
)


MOVE_ACTIONS = (UP, DOWN, LEFT, RIGHT, STAY)
_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    STAY: (0, 0),
}


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: PlumeArena, obs: dict) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


class ValenceGreedyPolicy:
    """Climb predicted valence; forage when local stake is worth it."""

    def __init__(
        self,
        model,
        forage_threshold: float = 0.15,
        name: Optional[str] = None,
    ):
        self.model = model
        self.forage_threshold = forage_threshold
        self.name = name or getattr(model, "name", "valence-greedy")

    def _predict_cell(self, env: PlumeArena, y: int, x: int, satiety: float) -> float:
        desc = env.smell_descriptors(y, x).reshape(1, -1)
        return float(self.model.predict(desc, np.array([satiety]))[0])

    def act(self, env: PlumeArena, obs: dict) -> int:
        s = float(obs["satiety"][0])
        y0, x0 = obs["position"]

        # If current cell smells worth consuming, forage.
        local_v = self._predict_cell(env, y0, x0, s)
        if local_v >= self.forage_threshold:
            return FORAGE

        best_a, best_v = STAY, local_v
        for a in MOVE_ACTIONS:
            dy, dx = _DELTA[a]
            y = int(np.clip(y0 + dy, 0, env.height - 1))
            x = int(np.clip(x0 + dx, 0, env.width - 1))
            v = self._predict_cell(env, y, x, s)
            if v > best_v:
                best_v, best_a = v, a
        return best_a


def _collect_arena_supervision(env: PlumeArena, n_samples: int = 4000, seed: int = 0):
    """Offline dataset: smell descriptors x satiety -> true valence, across the grid."""
    rng = np.random.default_rng(seed)
    X, S, Y = [], [], []
    # Dense coverage of the grid x satiety, plus random jitter samples.
    satiety_grid = np.linspace(0, 1, 11)
    for y in range(env.height):
        for x in range(env.width):
            for s in satiety_grid:
                X.append(env.smell_descriptors(y, x))
                S.append(s)
                Y.append(env.true_valence_at(y, x, satiety=s))
    # Extra random samples for intensity diversity.
    for _ in range(n_samples):
        y = int(rng.integers(0, env.height))
        x = int(rng.integers(0, env.width))
        s = float(rng.random())
        X.append(env.smell_descriptors(y, x))
        S.append(s)
        Y.append(env.true_valence_at(y, x, satiety=s))
    return (
        np.asarray(X, dtype=float),
        np.asarray(S, dtype=float),
        np.asarray(Y, dtype=float),
    )


def train_valence_head_for_arena(
    env: PlumeArena,
    kind: str = "conditioned",
    seed: int = 0,
):
    """Fit a valence head on the arena's true stake field.

    kind:
      - "blind"        : StateBlindBaseline
      - "conditioned"  : StateConditionedHead
      - "receptor"     : ReceptorRoutedHead (Lane C)
      - "same_info"    : StateAsFeatureWrongObjective (sees state, averaged target)
    """
    X, S, Y = _collect_arena_supervision(env, seed=seed)
    if kind == "blind":
        model = StateBlindBaseline(seed=seed)
        model.fit(X, S, Y)
    elif kind == "conditioned":
        model = StateConditionedHead(seed=seed)
        model.fit(X, S, Y)
    elif kind == "receptor":
        model = ReceptorRoutedHead(n_receptors=64, seed=seed)
        model.fit(X, S, Y)
    elif kind == "same_info":
        model = StateAsFeatureWrongObjective(seed=seed)
        # Group by discretised descriptor hash is messy; use global+local means
        # via a coarse grid id so the target is state-averaged per cell smell.
        # Smell varies by cell; average Y over satiety for each (y,x) group.
        # Reconstruct groups from the deterministic grid portion of the dataset.
        n_grid = env.height * env.width * 11
        mol_id = np.zeros(len(Y), dtype=int)
        # First n_grid rows are ordered (y,x,s_idx); group by (y,x).
        idx = 0
        gid = 0
        for y in range(env.height):
            for x in range(env.width):
                for _ in range(11):
                    mol_id[idx] = gid
                    idx += 1
                gid += 1
        # Remaining random samples: assign nearest grid cell id via round-robin skip;
        # approximate by fitting without fine groups (use cell-averaged from grid only).
        if len(Y) > n_grid:
            # map extras to random existing groups
            rng = np.random.default_rng(seed)
            mol_id[n_grid:] = rng.integers(0, gid, size=len(Y) - n_grid)
        model.fit(X, S, Y, mol_id=mol_id)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return model


def run_episode(
    env: PlumeArena,
    policy,
    satiety: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    obs = env.reset(seed=seed, satiety=satiety)
    rewards = []
    forages = []
    while True:
        a = policy.act(env, obs)
        obs, r, done, info = env.step(a)
        rewards.append(r)
        if info.get("foraged"):
            forages.append(info.get("forage_kind"))
        if done:
            break
    return {
        "total_reward": float(sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "n_steps": len(rewards),
        "forages": forages,
        "final_satiety": float(env.satiety),
        "n_food_forages": sum(1 for k in forages if k == "food"),
        "n_toxin_forages": sum(1 for k in forages if k == "toxin"),
    }


def evaluate_policy(
    env: PlumeArena,
    policy,
    n_episodes: int = 20,
    seed: int = 0,
    satiety_mode: str = "mixed",
) -> dict:
    """Evaluate average return under different initial satiety regimes.

    satiety_mode:
      - "mixed"    : uniform random initial satiety
      - "starving" : always start at s=0
      - "sated"    : always start at s=1
    """
    rng = np.random.default_rng(seed)
    returns = []
    food_hits = []
    toxin_hits = []
    for i in range(n_episodes):
        if satiety_mode == "starving":
            s0 = 0.0
        elif satiety_mode == "sated":
            s0 = 1.0
        else:
            s0 = float(rng.random())
        ep = run_episode(env, policy, satiety=s0, seed=seed + 1000 + i)
        returns.append(ep["total_reward"])
        food_hits.append(ep["n_food_forages"])
        toxin_hits.append(ep["n_toxin_forages"])
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_food_forages": float(np.mean(food_hits)),
        "mean_toxin_forages": float(np.mean(toxin_hits)),
        "n_episodes": n_episodes,
        "satiety_mode": satiety_mode,
        "policy": getattr(policy, "name", type(policy).__name__),
    }
