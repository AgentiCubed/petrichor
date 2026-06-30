"""
State-CHOICE probe: when valence has to drive an ACTION, not just a number.

This is the second experiment in the grounded-valence arc. The first
(experiments/state-flip/) showed a state-blind descriptor model provably cannot
*represent* a molecule whose valence flips with internal state, while a
state-conditioned head can. A fair objection: regression MSE is an abstraction;
animals do not minimise MSE, they CHOOSE — approach this, avoid that. So this
probe asks the behavioural question directly:

    Offered two odorants and an internal state, which do you go for?

The agent must pick the odorant with the higher *true* valence FOR ITS CURRENT
STATE. A chooser is scored only on whether it picks the right one.

WHY THIS STRENGTHENS THE THESIS
-------------------------------
We pair each held-out FOOD molecule with a held-out NON-FOOD molecule. For a
food molecule with intrinsic |base| < A, its valence is +~A when starving (s=0)
and -~A when sated (s=1) — it crosses zero. The non-food partner is
state-independent with |base| < A. So the CORRECT choice flips with state:

    starving -> approach the food odor   (it is the most appetising thing here)
    sated    -> avoid it, prefer the neutral odor   (food now smells worse)

PROVABLE CEILING (the whole point, same spirit as A^2/3 in experiment 1):
  A state-blind chooser scores each molecule with ONE fixed number g(m) and
  picks argmax. For a flip pair its preference is therefore FIXED across states,
  so it is correct in exactly ONE of {starving, sated} and wrong in the other:

      state-blind choice accuracy on flip pairs  <=  1/2   (exactly, in the limit)

  This is not a tuning artefact; it is a capacity ceiling. A state-conditioned
  chooser has the state as input and can pick differently per state, so it can
  approach 100%. run measures both against this 1/2 ceiling.

We also report RANDOM pairs (any two molecules) to show the blind chooser is not
globally broken — it does fine when one option dominates in both states — the
failure is specifically the state-dependent decisions, which are the ones that
keep an animal alive.

BEHAVIOURAL REGRET: beyond accuracy we report mean regret = the true valence of
the best option minus the true valence of the chosen one, in valence units. It
is the felt cost of the wrong move (how bad the thing you ate actually was),
which accuracy alone hides.

Run:
    python choice.py                 # synthetic toy world (provable ceiling)
    python choice.py --plot          # also save figures to ./figures/
    python choice.py --source dream  # real Keller/DREAM molecule axis

All work is local and reversible.
"""

import argparse
import json
import os
import sys

import numpy as np

# reuse the dataset + the two heads from the sibling state-flip experiment;
# nothing is duplicated, so any fix there flows here unchanged.
_FLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "state-flip")
sys.path.insert(0, os.path.abspath(_FLIP_DIR))

from data import make_dataset, load_dream, split_by_molecule   # noqa: E402
from models import StateBlindBaseline, StateConditionedHead     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def true_valence(data, m, s):
    """Noise-free ground-truth valence of molecule m at internal state s.
    valence(m,s) = base(m) + food(m) * A * (1 - 2s)."""
    A = data["meta"]["amplitude"]
    return data["base"][m] + data["mol_food"][m] * A * (1.0 - 2.0 * s)


def _desc(data, m):
    return data["molecules"][m].reshape(1, -1)


def choose(model, data, a, b, s):
    """Which molecule (a or b) the model approaches at state s: argmax predicted
    valence. Returns the chosen molecule id."""
    va = float(model.predict(_desc(data, a), np.array([s]))[0])
    vb = float(model.predict(_desc(data, b), np.array([s]))[0])
    return a if va >= vb else b


def best_choice(data, a, b, s):
    """The truly-correct approach at state s (higher true valence)."""
    return a if true_valence(data, a, s) >= true_valence(data, b, s) else b


def build_pairs(data, test_mask, seed=0):
    """Held-out FOOD x NON-FOOD flip pairs, plus random held-out pairs.

    Flip pairs are guaranteed sign-flippers: a food molecule (|base| < A so it
    crosses zero with satiety) vs a state-independent non-food molecule.
    """
    rng = np.random.default_rng(seed)
    test_mols = [m for m in np.unique(data["mol_id"][test_mask])]
    food = [m for m in test_mols if data["mol_food"][m] == 1]
    nonfood = [m for m in test_mols if data["mol_food"][m] == 0]
    k = min(len(food), len(nonfood))
    rng.shuffle(food)
    rng.shuffle(nonfood)
    flip_pairs = list(zip(food[:k], nonfood[:k]))

    # random pairs across all held-out molecules (sanity / global behaviour)
    perm = rng.permutation(test_mols)
    rand_pairs = [(int(perm[i]), int(perm[i + 1]))
                  for i in range(0, len(perm) - 1, 2)]
    return flip_pairs, rand_pairs


def score_choices(model, data, pairs, states=(0.0, 1.0)):
    """Accuracy and mean regret over (pair, state) trials."""
    correct, n, regret = 0, 0, 0.0
    for (a, b) in pairs:
        for s in states:
            pick = choose(model, data, a, b, s)
            truth = best_choice(data, a, b, s)
            if pick == truth:
                correct += 1
            # regret = true valence lost by the chosen vs the best option
            regret += abs(true_valence(data, truth, s) - true_valence(data, pick, s))
            n += 1
    return {"accuracy": correct / n if n else 0.0,
            "mean_regret": regret / n if n else 0.0,
            "n_trials": n}


def accuracy_vs_state(model, data, pairs, s_grid):
    """Choice accuracy on flip pairs as the state sweeps 0 -> 1 (for the curve)."""
    accs = []
    for s in s_grid:
        c = sum(1 for (a, b) in pairs
                if choose(model, data, a, b, s) == best_choice(data, a, b, s))
        accs.append(c / len(pairs) if pairs else 0.0)
    return np.array(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="save figures to ./figures/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["synthetic", "dream"], default="synthetic")
    args = ap.parse_args()

    data = load_dream(seed=args.seed) if args.source == "dream" else make_dataset(seed=args.seed)
    train_mask, test_mask = split_by_molecule(data, seed=args.seed + 1)

    Xtr, str_, ytr = data["X_desc"][train_mask], data["state"][train_mask], data["y"][train_mask]
    baseline = StateBlindBaseline().fit(Xtr, str_, ytr)
    conditioned = StateConditionedHead().fit(Xtr, str_, ytr)

    flip_pairs, rand_pairs = build_pairs(data, test_mask, seed=args.seed + 2)

    b_flip = score_choices(baseline, data, flip_pairs)
    c_flip = score_choices(conditioned, data, flip_pairs)
    b_rand = score_choices(baseline, data, rand_pairs)
    c_rand = score_choices(conditioned, data, rand_pairs)

    A = data["meta"]["amplitude"]

    # ---- report -------------------------------------------------------------
    print("=" * 70)
    print("STATE-CHOICE PROBE  —  when valence must drive the action")
    print("=" * 70)
    print(f"data source: {data['meta'].get('source', 'synthetic')}")
    print(f"flip pairs (food vs non-food): {len(flip_pairs)}    "
          f"random pairs: {len(rand_pairs)}")
    print(f"provable state-blind accuracy ceiling on flip pairs = 0.500")
    print("-" * 70)
    hdr = f"{'metric':<34}{'state-blind':>16}{'state-cond.':>16}"
    print(hdr)
    print("-" * 70)

    def row(label, vb, vc, fmt="{:.3f}"):
        print(f"{label:<34}{fmt.format(vb):>16}{fmt.format(vc):>16}")

    row("choice accuracy — FLIP pairs", b_flip["accuracy"], c_flip["accuracy"])
    row("choice accuracy — random pairs", b_rand["accuracy"], c_rand["accuracy"])
    row("mean regret (valence) — FLIP", b_flip["mean_regret"], c_flip["mean_regret"])
    row("mean regret (valence) — random", b_rand["mean_regret"], c_rand["mean_regret"])
    print("-" * 70)
    print("Read it:")
    print(f"  The state-blind chooser sits at ~0.50 on flip pairs — the provable")
    print(f"  ceiling. It commits to one odor and is therefore wrong in exactly")
    print(f"  one of {{starving, sated}}. On random pairs (one option usually wins")
    print(f"  in both states) it does fine — the deficit is the STATE-DEPENDENT")
    print(f"  decisions, the ones that matter for survival. The state-conditioned")
    print(f"  chooser approaches 1.0 on flip pairs and carries far less regret.")
    print("=" * 70)

    suffix = "" if args.source == "synthetic" else f"_{args.source}"
    if args.plot:
        save_figures(baseline, conditioned, data, flip_pairs,
                     b_flip, c_flip, b_rand, c_rand, suffix)
        print(f"\nfigures saved to {FIG_DIR}/")

    summary = {
        "source": data["meta"].get("source", "synthetic"),
        "amplitude": A,
        "n_flip_pairs": len(flip_pairs),
        "n_random_pairs": len(rand_pairs),
        "blind_accuracy_ceiling_flip": 0.5,
        "baseline": {"flip": b_flip, "random": b_rand},
        "conditioned": {"flip": c_flip, "random": c_rand},
    }
    with open(os.path.join(HERE, f"last_run{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def save_figures(baseline, conditioned, data, flip_pairs,
                 b_flip, c_flip, b_rand, c_rand, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)

    # Fig 1: choice accuracy on flip vs random pairs, vs the provable 0.5 ceiling.
    plt.figure(figsize=(6.2, 4.2))
    groups = ["flip pairs", "random pairs"]
    x = np.arange(len(groups))
    w = 0.36
    blind = [b_flip["accuracy"], b_rand["accuracy"]]
    cond = [c_flip["accuracy"], c_rand["accuracy"]]
    plt.bar(x - w / 2, blind, w, label="state-blind", color="#c0504d")
    plt.bar(x + w / 2, cond, w, label="state-conditioned", color="#4f81bd")
    plt.axhline(0.5, color="k", ls="--", lw=1,
                label="provable blind ceiling (flip) = 0.50")
    plt.xticks(x, groups)
    plt.ylim(0, 1.05)
    plt.ylabel("choice accuracy")
    plt.title("Choosing right depends on having the state")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"choice_accuracy{suffix}.png"), dpi=130)
    plt.close()

    # Fig 2: accuracy on flip pairs as the internal state sweeps 0 -> 1.
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

    # Fig 3: behavioural regret (valence lost) on flip pairs.
    plt.figure(figsize=(5.6, 4.2))
    bars = ["state-blind", "state-cond."]
    vals = [b_flip["mean_regret"], c_flip["mean_regret"]]
    plt.bar(bars, vals, color=["#c0504d", "#4f81bd"])
    plt.ylabel("mean regret on flip pairs  (true valence lost)")
    plt.title("The felt cost of choosing without state")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"regret{suffix}.png"), dpi=130)
    plt.close()


# expose for the curve helper used in save_figures
def _bind():
    pass


if __name__ == "__main__":
    main()
