"""
Plume forage — valence must drive behaviour in a world with stakes.

A 2D chemical arena emits food, toxin, and neutral plumes. Reward is
homeostatic (satiety flip on food; damage on toxin), not label-shaped.

Policies compared (identical greedy actuator; only the valence head changes):

  random
  state-blind valence head
  same-info control (state in input, averaged target)
  state-conditioned valence head
  receptor-routed valence head (Lane C)

Success metric: mean episodic return, plus food/toxin forage counts, under
starving and sated initial conditions. The state-conditioned / receptor heads
should approach food when starving and avoid it when sated; blind/same-info
cannot track that flip, so they either over-eat when full or under-eat when
hungry (or both).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from petrichor.agents import (  # noqa: E402
    RandomPolicy,
    ValenceGreedyPolicy,
    evaluate_policy,
    train_valence_head_for_arena,
)
from petrichor.env import PlumeArena  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    env = PlumeArena(width=15, height=15, seed=args.seed)

    heads = {
        "blind": train_valence_head_for_arena(env, kind="blind", seed=args.seed),
        "same_info": train_valence_head_for_arena(env, kind="same_info", seed=args.seed),
        "conditioned": train_valence_head_for_arena(env, kind="conditioned", seed=args.seed),
        "receptor": train_valence_head_for_arena(env, kind="receptor", seed=args.seed),
    }
    policies = {"random": RandomPolicy(seed=args.seed)}
    for name, model in heads.items():
        policies[name] = ValenceGreedyPolicy(model, name=name)

    modes = ("starving", "sated", "mixed")
    table = {p: {} for p in policies}
    for pname, policy in policies.items():
        for mode in modes:
            table[pname][mode] = evaluate_policy(
                env, policy, n_episodes=args.episodes, seed=args.seed + 7, satiety_mode=mode
            )

    print("=" * 78)
    print("PLUME FORAGE  —  grounded valence drives action in a world with stakes")
    print("=" * 78)
    print(f"arena {env.width}x{env.height}, {len(env.sources)} sources, "
          f"{args.episodes} episodes/condition")
    print("-" * 78)
    print(f"{'policy':<14}{'ret starve':>12}{'ret sated':>12}{'ret mixed':>12}"
          f"{'food|starve':>12}{'food|sated':>12}{'toxin':>8}")
    print("-" * 78)
    for pname in policies:
        rs = table[pname]["starving"]
        rf = table[pname]["sated"]
        rm = table[pname]["mixed"]
        print(
            f"{pname:<14}"
            f"{rs['mean_return']:>12.2f}"
            f"{rf['mean_return']:>12.2f}"
            f"{rm['mean_return']:>12.2f}"
            f"{rs['mean_food_forages']:>12.2f}"
            f"{rf['mean_food_forages']:>12.2f}"
            f"{rm['mean_toxin_forages']:>8.2f}"
        )
    print("-" * 78)
    print("Read it:")
    print("  Conditioned/receptor should eat food when starving and back off when sated.")
    print("  Blind/same-info commit to one appetite and bleed return on the flip side.")
    print("  Toxin forages should stay low for any valence-competent head.")
    print("=" * 78)

    if args.plot:
        _plot(env, table, args)
        print(f"figures saved to {FIG_DIR}/")

    # JSON-friendly dump
    summary = {
        "episodes": args.episodes,
        "seed": args.seed,
        "results": table,
    }
    with open(os.path.join(HERE, "last_run.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _plot(env, table, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    names = list(table.keys())
    starve = [table[n]["starving"]["mean_return"] for n in names]
    sated = [table[n]["sated"]["mean_return"] for n in names]

    x = np.arange(len(names))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x - w / 2, starve, w, label="start starving", color="#4f81bd")
    axes[0].bar(x + w / 2, sated, w, label="start sated", color="#c0504d")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel("mean episodic return")
    axes[0].set_title("Return under opposite internal states")
    axes[0].legend(fontsize=8)

    # Valence maps at s=0 and s=1
    v0 = env.valence_map(satiety=0.0)
    v1 = env.valence_map(satiety=1.0)
    vmax = max(abs(v0).max(), abs(v1).max(), 1e-6)
    im = axes[1].imshow(v0 - v1, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[1].set_title("True valence map: starving − sated")
    for src in env.sources:
        yy, xx = src.position
        axes[1].scatter([xx], [yy], c="k", s=40)
        axes[1].text(xx, yy, src.kind[0].upper(), color="k", fontsize=8,
                     ha="left", va="bottom")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("Plume forage — stake-shaped world")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "plume_forage.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
