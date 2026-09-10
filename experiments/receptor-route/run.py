"""
Lane C — receptor-routed valence.

Claim: routing valence through a receptor-binding layer (descriptors ->
receptor population -> valence x state) preserves the state-flip capacity
result, and makes the thesis claim in the *architecture* itself: the valence
head never sees raw descriptors, only contact-shaped receptor activity.

Compares:
  - state-blind baseline (descriptors only)
  - state-conditioned head (descriptors + state)
  - receptor-routed head (bind -> receptors + state -> valence)
  - receptor-routed state-blind ablation (receptors, no state)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from petrichor import (  # noqa: E402
    ReceptorRoutedHead,
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
    ap.add_argument("--receptors", type=int, default=64)
    args = ap.parse_args()

    data = load_dream(seed=args.seed) if args.source == "dream" else make_dataset(seed=args.seed)
    train_mask, test_mask = split_by_molecule(data, seed=args.seed + 1)
    Xtr = data["X_desc"][train_mask]
    str_ = data["state"][train_mask]
    ytr = data["y"][train_mask]

    models = {
        "blind": StateBlindBaseline(seed=args.seed).fit(Xtr, str_, ytr),
        "conditioned": StateConditionedHead(seed=args.seed).fit(Xtr, str_, ytr),
        "receptor": ReceptorRoutedHead(n_receptors=args.receptors, seed=args.seed).fit(Xtr, str_, ytr),
        "receptor_blind": ReceptorRoutedHead(
            n_receptors=args.receptors, seed=args.seed, state_blind=True
        ).fit(Xtr, str_, ytr),
    }
    results = {k: evaluate_flip(m, data, test_mask) for k, m in models.items()}
    floor = irreducible_baseline_food_mse(data["meta"]["amplitude"])

    print("=" * 78)
    print("RECEPTOR-ROUTE  —  Lane C (valence through a binding layer)")
    print("=" * 78)
    print(f"data source: {data['meta'].get('source', 'synthetic')}")
    print(f"n_receptors: {args.receptors}")
    print(f"provable blind food-MSE floor A^2/3 = {floor:.3f}")
    print("-" * 78)
    hdr = f"{'metric':<22}" + "".join(f"{k:>14}" for k in models)
    print(hdr)
    print("-" * 78)

    def row(label, key, fmt="{:.3f}"):
        cells = "".join(
            f"{fmt.format(results[k][key]):>14}" if results[k][key] is not None else f"{'—':>14}"
            for k in models
        )
        print(f"{label:<22}{cells}")

    row("MSE all", "test_mse_all")
    row("MSE food", "test_mse_food")
    row("MSE non-food", "test_mse_nonfood")
    row("flip rate food", "flip_rate_food")
    print("-" * 78)
    print("Read it:")
    print("  Receptor-routed + state should match conditioned flip capacity.")
    print("  Receptor-routed WITHOUT state should collapse to the blind floor.")
    print("  That isolates the state channel from the binding layer.")
    print("=" * 78)

    if args.plot:
        _plot(results, floor, data, args)
        print(f"figures saved to {FIG_DIR}/")

    summary = {
        "source": data["meta"].get("source", "synthetic"),
        "n_receptors": args.receptors,
        "floor_food_mse": floor,
        "results": {
            k: {kk: vv for kk, vv in r.items() if kk != "examples"}
            for k, r in results.items()
        },
    }
    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    with open(os.path.join(HERE, f"last_run{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _plot(results, floor, data, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    labels = list(results.keys())
    flips = [results[k]["flip_rate_food"] for k in labels]
    foods = [results[k]["test_mse_food"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, flips, color=["#c0504d", "#4f81bd", "#2e8b57", "#a0a0a0"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("flip rate (food)")
    axes[0].set_title("Sign-flip recovery")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, foods, color=["#c0504d", "#4f81bd", "#2e8b57", "#a0a0a0"])
    axes[1].axhline(floor, color="k", ls="--", label=f"floor A²/3={floor:.2f}")
    axes[1].set_ylabel("food MSE")
    axes[1].set_title("Food MSE vs floor")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Lane C — receptor-routed valence")
    fig.tight_layout()
    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    fig.savefig(os.path.join(FIG_DIR, f"receptor_route{suffix}.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
