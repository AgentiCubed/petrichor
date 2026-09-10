"""Flip-rate and MSE evaluation for state-conditioned valence."""

from __future__ import annotations

import numpy as np


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def evaluate_flip(model, data: dict, test_mask) -> dict:
    """Score a fitted model on held-out molecules: MSE + food flip rate."""
    Xd = data["X_desc"][test_mask]
    st = data["state"][test_mask]
    y = data["y"][test_mask]
    food = data["is_food"][test_mask]
    pred = model.predict(Xd, st)

    out = {
        "test_mse_all": mse(pred, y),
        "test_mse_food": mse(pred[food == 1], y[food == 1]) if (food == 1).any() else None,
        "test_mse_nonfood": mse(pred[food == 0], y[food == 0]) if (food == 0).any() else None,
    }

    test_food_mols = [
        m for m in np.unique(data["mol_id"][test_mask]) if data["mol_food"][m] == 1
    ]
    flips, examples = 0, []
    amp = data["meta"]["amplitude"]
    for m in test_food_mols:
        desc = data["molecules"][m].reshape(1, -1)
        v0 = float(model.predict(desc, np.array([0.0]))[0])
        v1 = float(model.predict(desc, np.array([1.0]))[0])
        true0 = data["base"][m] + amp
        true1 = data["base"][m] - amp
        if np.sign(v0) != np.sign(v1) and abs(v0) > 1e-9 and abs(v1) > 1e-9:
            flips += 1
        elif np.sign(v0) != np.sign(v1):
            flips += 1
        examples.append((int(m), v0, v1, float(true0), float(true1)))

    out["n_food_test"] = len(test_food_mols)
    out["flip_rate_food"] = flips / len(test_food_mols) if test_food_mols else 0.0
    out["examples"] = examples
    if examples:
        out["mean_v_starving"] = float(np.mean([e[1] for e in examples]))
        out["mean_v_sated"] = float(np.mean([e[2] for e in examples]))
    return out
