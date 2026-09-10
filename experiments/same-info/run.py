"""
Falsifiable core (03 §5.2) — same-information control.

Question: does a grounded valence channel measurably change what a model does
versus a channel that has the *same sensors* but a non-stake learning signal?

Three heads, identical capacity, identical inputs where noted:

  1. state-blind            — descriptors only; target = y(m,s)
  2. state-conditioned       — descriptors + state; target = y(m,s)   [stake]
  3. same-info control      — descriptors + state; target = E_s[y|m]
                              (state bits present; objective ignores the flip)

If (3) fails to recover the flip while (2) succeeds, the result is not "you
needed more features" — it is "the learning signal must be stake-shaped."
Having internal-state bits available is not enough.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from petrichor import (  # noqa: E402
    StateAsFeatureWrongObjective,
    StateBlindBaseline,
    StateConditionedHead,
    evaluate_flip,
    irreducible_baseline_food_mse,
    load_dream,
    make_dataset,
    split_by_molecule,
)
from petrichor.metrics import build_pairs, score_choices  # noqa: E402

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
    Xtr = data["X_desc"][train_mask]
    str_ = data["state"][train_mask]
    ytr = data["y"][train_mask]
    mol_tr = data["mol_id"][train_mask]

    blind = StateBlindBaseline(seed=args.seed).fit(Xtr, str_, ytr)
    cond = StateConditionedHead(seed=args.seed).fit(Xtr, str_, ytr)
    same = StateAsFeatureWrongObjective(seed=args.seed).fit(Xtr, str_, ytr, mol_id=mol_tr)

    models = {"blind": blind, "conditioned": cond, "same_info": same}
    flip = {k: evaluate_flip(m, data, test_mask) for k, m in models.items()}

    flip_pairs, rand_pairs = build_pairs(data, test_mask, seed=args.seed + 2)
    choice = {
        k: {
            "flip": score_choices(m, data, flip_pairs),
            "random": score_choices(m, data, rand_pairs),
        }
        for k, m in models.items()
    }
    floor = irreducible_baseline_food_mse(data["meta"]["amplitude"])

    print("=" * 78)
    print("SAME-INFO CONTROL  —  falsifiable core (03 §5.2)")
    print("=" * 78)
    print(f"data source: {data['meta'].get('source', 'synthetic')}")
    print(f"provable blind food-MSE floor A^2/3 = {floor:.3f}")
    print(f"provable blind choice ceiling on flip pairs = 0.500")
    print("-" * 78)
    print(f"{'metric':<28}{'blind':>12}{'conditioned':>14}{'same-info':>12}")
    print("-" * 78)

    def row(label, key_path, src):
        vals = []
        for name in ("blind", "conditioned", "same_info"):
            d = src[name]
            for k in key_path:
                d = d[k]
            vals.append(d)
        print(f"{label:<28}{vals[0]:>12.3f}{vals[1]:>14.3f}{vals[2]:>12.3f}")

    row("food MSE", ["test_mse_food"], flip)
    row("flip rate (food)", ["flip_rate_food"], flip)
    row("choice acc (flip pairs)", ["flip", "accuracy"], choice)
    row("choice regret (flip)", ["flip", "mean_regret"], choice)
    row("choice acc (random)", ["random", "accuracy"], choice)
    print("-" * 78)
    print("Read it:")
    print("  conditioned  = stake-shaped target, state in input  -> should flip")
    print("  same-info    = state in input, state-averaged target -> should NOT flip")
    print("  blind        = no state                              -> cannot flip")
    print("  If same-info ≈ blind on flip metrics, presence needs a valenced signal,")
    print("  not merely access to internal-state bits.")
    print("=" * 78)

    if args.plot:
        _plot(flip, choice, floor, args)
        print(f"figures saved to {FIG_DIR}/")

    summary = {
        "source": data["meta"].get("source", "synthetic"),
        "floor_food_mse": floor,
        "flip": {k: {kk: vv for kk, vv in r.items() if kk != "examples"} for k, r in flip.items()},
        "choice": choice,
    }
    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    with open(os.path.join(HERE, f"last_run{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _plot(flip, choice, floor, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    names = ["blind", "conditioned", "same_info"]
    labels = ["blind", "conditioned", "same-info"]
    colors = ["#c0504d", "#4f81bd", "#e6b800"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [flip[n]["flip_rate_food"] for n in names], color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("flip rate (food)")
    axes[0].set_title("Representational flip")

    axes[1].bar(labels, [choice[n]["flip"]["accuracy"] for n in names], color=colors)
    axes[1].axhline(0.5, color="k", ls="--", label="blind ceiling 0.50")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("choice accuracy (flip pairs)")
    axes[1].set_title("Behavioural choice")
    axes[1].legend(fontsize=8)
    fig.suptitle("Same sensors, wrong objective — the falsifiable core")
    fig.tight_layout()
    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    fig.savefig(os.path.join(FIG_DIR, f"same_info{suffix}.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
