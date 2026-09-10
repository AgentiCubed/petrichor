"""State-choice probe via the petrichor package."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from petrichor import (  # noqa: E402
    StateBlindBaseline,
    StateConditionedHead,
    load_dream,
    make_dataset,
    split_by_molecule,
)
from petrichor.metrics import accuracy_vs_state, build_pairs, score_choices  # noqa: E402

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

    flip_pairs, rand_pairs = build_pairs(data, test_mask, seed=args.seed + 2)
    b_flip = score_choices(baseline, data, flip_pairs)
    c_flip = score_choices(conditioned, data, flip_pairs)
    b_rand = score_choices(baseline, data, rand_pairs)
    c_rand = score_choices(conditioned, data, rand_pairs)

    print("=" * 70)
    print("STATE-CHOICE  —  when valence must drive the action")
    print("=" * 70)
    print(f"flip pairs: {len(flip_pairs)}   random pairs: {len(rand_pairs)}")
    print(f"provable state-blind accuracy ceiling on flip pairs = 0.500")
    print("-" * 70)
    print(f"{'metric':<34}{'state-blind':>16}{'state-cond.':>16}")
    print("-" * 70)
    print(f"{'choice accuracy — FLIP pairs':<34}{b_flip['accuracy']:>16.3f}{c_flip['accuracy']:>16.3f}")
    print(f"{'choice accuracy — random pairs':<34}{b_rand['accuracy']:>16.3f}{c_rand['accuracy']:>16.3f}")
    print(f"{'mean regret — FLIP':<34}{b_flip['mean_regret']:>16.3f}{c_flip['mean_regret']:>16.3f}")
    print("=" * 70)

    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    if args.plot:
        _plots(baseline, conditioned, data, flip_pairs, b_flip, c_flip, b_rand, c_rand, suffix)
        print(f"figures saved to {FIG_DIR}/")

    summary = {
        "source": data["meta"].get("source", "synthetic"),
        "n_flip_pairs": len(flip_pairs),
        "n_random_pairs": len(rand_pairs),
        "blind_accuracy_ceiling_flip": 0.5,
        "baseline": {"flip": b_flip, "random": b_rand},
        "conditioned": {"flip": c_flip, "random": c_rand},
    }
    with open(os.path.join(HERE, f"last_run{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _plots(baseline, conditioned, data, flip_pairs, b_flip, c_flip, b_rand, c_rand, suffix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(6.2, 4.2))
    x = np.arange(2)
    w = 0.36
    plt.bar(x - w / 2, [b_flip["accuracy"], b_rand["accuracy"]], w, label="state-blind", color="#c0504d")
    plt.bar(x + w / 2, [c_flip["accuracy"], c_rand["accuracy"]], w, label="state-conditioned", color="#4f81bd")
    plt.axhline(0.5, color="k", ls="--", lw=1, label="provable blind ceiling (flip) = 0.50")
    plt.xticks(x, ["flip pairs", "random pairs"])
    plt.ylim(0, 1.05)
    plt.ylabel("choice accuracy")
    plt.title("Choosing right depends on having the state")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"choice_accuracy{suffix}.png"), dpi=130)
    plt.close()

    s_grid = np.linspace(0, 1, 41)
    ab = accuracy_vs_state(baseline, data, flip_pairs, s_grid)
    ac = accuracy_vs_state(conditioned, data, flip_pairs, s_grid)
    plt.figure(figsize=(6.2, 4.2))
    plt.axhline(0.5, color="0.7", lw=1)
    plt.plot(s_grid, ab, label="state-blind", color="#c0504d")
    plt.plot(s_grid, ac, label="state-conditioned", color="#4f81bd")
    plt.xlabel("internal state s  (0 = starving, 1 = sated)")
    plt.ylabel("choice accuracy on flip pairs")
    plt.ylim(-0.02, 1.05)
    plt.title("The blind chooser is right in only one state at a time")
    plt.legend(loc="center right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"accuracy_vs_state{suffix}.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(5.6, 4.2))
    plt.bar(["state-blind", "state-cond."], [b_flip["mean_regret"], c_flip["mean_regret"]],
            color=["#c0504d", "#4f81bd"])
    plt.ylabel("mean regret on flip pairs  (true valence lost)")
    plt.title("The felt cost of choosing without state")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"regret{suffix}.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
