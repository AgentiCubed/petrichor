"""Choice / regret metrics — when valence must drive an action."""

from __future__ import annotations

import numpy as np

from petrichor.data.synthetic import true_valence


def true_valence_of(data: dict, m: int, s: float) -> float:
    return true_valence(
        data["base"][m],
        int(data["mol_food"][m]),
        s,
        amplitude=data["meta"]["amplitude"],
    )


def _desc(data, m):
    return data["molecules"][m].reshape(1, -1)


def choose(model, data, a, b, s):
    va = float(model.predict(_desc(data, a), np.array([s]))[0])
    vb = float(model.predict(_desc(data, b), np.array([s]))[0])
    return a if va >= vb else b


def best_choice(data, a, b, s):
    return a if true_valence_of(data, a, s) >= true_valence_of(data, b, s) else b


def build_pairs(data, test_mask, seed: int = 0):
    """Held-out FOOD x NON-FOOD flip pairs, plus random held-out pairs."""
    rng = np.random.default_rng(seed)
    test_mols = [int(m) for m in np.unique(data["mol_id"][test_mask])]
    food = [m for m in test_mols if data["mol_food"][m] == 1]
    nonfood = [m for m in test_mols if data["mol_food"][m] == 0]
    k = min(len(food), len(nonfood))
    rng.shuffle(food)
    rng.shuffle(nonfood)
    flip_pairs = list(zip(food[:k], nonfood[:k]))

    perm = rng.permutation(test_mols)
    rand_pairs = [
        (int(perm[i]), int(perm[i + 1])) for i in range(0, len(perm) - 1, 2)
    ]
    return flip_pairs, rand_pairs


def score_choices(model, data, pairs, states=(0.0, 1.0)):
    correct, n, regret = 0, 0, 0.0
    for (a, b) in pairs:
        for s in states:
            pick = choose(model, data, a, b, s)
            truth = best_choice(data, a, b, s)
            if pick == truth:
                correct += 1
            regret += abs(true_valence_of(data, truth, s) - true_valence_of(data, pick, s))
            n += 1
    return {
        "accuracy": correct / n if n else 0.0,
        "mean_regret": regret / n if n else 0.0,
        "n_trials": n,
    }


def accuracy_vs_state(model, data, pairs, s_grid):
    accs = []
    for s in s_grid:
        c = sum(
            1
            for (a, b) in pairs
            if choose(model, data, a, b, s) == best_choice(data, a, b, s)
        )
        accs.append(c / len(pairs) if pairs else 0.0)
    return np.array(accs)
