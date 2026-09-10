"""
State-flip demo via the petrichor package.

Keeps the original run.py for backward compatibility; this entry point is the
supported one going forward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from petrichor import (  # noqa: E402
    StateBlindBaseline,
    StateConditionedHead,
    evaluate_flip,
    irreducible_baseline_food_mse,
    load_dream,
    make_dataset,
    split_by_molecule,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["synthetic", "dream"], default="synthetic")
    args = ap.parse_args()

    data = load_dream(seed=args.seed) if args.source == "dream" else make_dataset(seed=args.seed)
    train_mask, test_mask = split_by_molecule(data, seed=args.seed + 1)
    Xtr, str_, ytr = data["X_desc"][train_mask], data["state"][train_mask], data["y"][train_mask]

    baseline = StateBlindBaseline(seed=args.seed).fit(Xtr, str_, ytr)
    conditioned = StateConditionedHead(seed=args.seed).fit(Xtr, str_, ytr)

    res_b = evaluate_flip(baseline, data, test_mask)
    res_c = evaluate_flip(conditioned, data, test_mask)
    floor = irreducible_baseline_food_mse(data["meta"]["amplitude"])

    print("=" * 68)
    print("STATE-FLIP  —  grounded valence (molecule x internal state)")
    print("=" * 68)
    print(f"data source: {data['meta'].get('source', 'synthetic')}")
    print(f"provable state-blind food-MSE floor (A^2/3) = {floor:.3f}")
    print("-" * 68)
    print(f"{'metric':<26}{'state-blind':>16}{'state-cond.':>16}")
    print("-" * 68)

    def row(label, kb, kc):
        print(f"{label:<26}{kb:>16.3f}{kc:>16.3f}")

    row("test MSE (all)", res_b["test_mse_all"], res_c["test_mse_all"])
    row("test MSE (food)", res_b["test_mse_food"], res_c["test_mse_food"])
    row("test MSE (non-food)", res_b["test_mse_nonfood"], res_c["test_mse_nonfood"])
    row("flip rate (food mols)", res_b["flip_rate_food"], res_c["flip_rate_food"])
    print("=" * 68)

    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    if args.plot:
        _plots(baseline, conditioned, data, res_b, res_c, floor, suffix)
        print(f"figures saved to {FIG_DIR}/")

    summary = {
        "source": data["meta"].get("source", "synthetic"),
        "n_molecules": data["meta"]["n_molecules"],
        "floor_food_mse": floor,
        "noise_floor": data["meta"]["noise_sigma"] ** 2,
        "baseline": {k: res_b[k] for k in res_b if k != "examples"},
        "conditioned": {k: res_c[k] for k in res_c if k != "examples"},
    }
    with open(os.path.join(HERE, f"last_run{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _plots(baseline, conditioned, data, res_b, res_c, floor, suffix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    m = res_c["examples"][0][0]
    desc = data["molecules"][m].reshape(1, -1)
    s_grid = np.linspace(0, 1, 50)
    true = data["base"][m] + data["meta"]["amplitude"] * (1 - 2 * s_grid)
    pb = np.array([baseline.predict(desc, np.array([s]))[0] for s in s_grid])
    pc = np.array([conditioned.predict(desc, np.array([s]))[0] for s in s_grid])
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color="0.7", lw=1)
    plt.plot(s_grid, true, "k--", label="true valence")
    plt.plot(s_grid, pb, label="state-blind baseline")
    plt.plot(s_grid, pc, label="state-conditioned head")
    plt.xlabel("internal state s  (0 = starving, 1 = sated)")
    plt.ylabel("valence  (+ approach / − recoil)")
    plt.title(f"Same molecule, flipping valence (food molecule #{m})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"flip_curve{suffix}.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["state-blind", "state-cond."], [res_b["test_mse_food"], res_c["test_mse_food"]],
            color=["#c0504d", "#4f81bd"])
    plt.axhline(floor, color="k", ls="--", label=f"provable floor A²/3 = {floor:.2f}")
    plt.axhline(data["meta"]["noise_sigma"] ** 2, color="0.4", ls=":",
                label=f"noise floor = {data['meta']['noise_sigma']**2:.2f}")
    plt.ylabel("test MSE on food molecules")
    plt.title("Cost of having nowhere to put state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"food_mse{suffix}.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
