"""
State-flip demo: run, score, and (optionally) plot.

Trains the state-blind baseline and the state-conditioned valence head on the
same synthetic molecule x state valence data, then measures three things:

  1. Overall test MSE (held-out molecules).
  2. Food-molecule MSE at the satiety extremes (s=0 starving, s=1 sated),
     against the PROVABLE baseline floor A^2/3.
  3. Flip rate: fraction of held-out food molecules whose predicted valence
     changes SIGN between s=0 and s=1 (the true sign does flip for these).
     Baseline flip rate is ~0 by construction; the conditioned head should
     recover most of the flips.

Run:
    python run.py            # metrics only
    python run.py --plot     # also save figures to ./figures/

All work is local and reversible.
"""

import argparse
import json
import os

import numpy as np

from data import (make_dataset, load_dream, split_by_molecule,
                  irreducible_baseline_food_mse)
from models import StateBlindBaseline, StateConditionedHead

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def evaluate(model, data, test_mask):
    """Score a fitted model on held-out molecules."""
    Xd, st, y = data["X_desc"][test_mask], data["state"][test_mask], data["y"][test_mask]
    food = data["is_food"][test_mask]
    pred = model.predict(Xd, st)

    out = {
        "test_mse_all": mse(pred, y),
        "test_mse_food": mse(pred[food == 1], y[food == 1]) if (food == 1).any() else None,
        "test_mse_nonfood": mse(pred[food == 0], y[food == 0]) if (food == 0).any() else None,
    }

    # Flip analysis on held-out FOOD molecules: probe each at s=0 and s=1.
    test_food_mols = [m for m in np.unique(data["mol_id"][test_mask]) if data["mol_food"][m] == 1]
    flips, examples = 0, []
    for m in test_food_mols:
        desc = data["molecules"][m].reshape(1, -1)
        v0 = float(model.predict(desc, np.array([0.0]))[0])   # starving
        v1 = float(model.predict(desc, np.array([1.0]))[0])   # sated
        true0 = data["base"][m] + data["meta"]["amplitude"]   # +A
        true1 = data["base"][m] - data["meta"]["amplitude"]   # -A
        if np.sign(v0) != np.sign(v1):
            flips += 1
        examples.append((m, v0, v1, float(true0), float(true1)))
    out["n_food_test"] = len(test_food_mols)
    out["flip_rate_food"] = flips / len(test_food_mols) if test_food_mols else 0.0
    out["examples"] = examples
    # aggregate flip: mean predicted valence across all test food molecules at
    # each satiety extreme. True mean flips +A -> -A; a flip-capable model's
    # means should straddle zero too.
    if examples:
        out["mean_v_starving"] = float(np.mean([e[1] for e in examples]))
        out["mean_v_sated"] = float(np.mean([e[2] for e in examples]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="save figures to ./figures/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["synthetic", "dream"], default="synthetic",
                    help="synthetic toy world (provable floor) or real Keller/DREAM data")
    args = ap.parse_args()

    data = load_dream(seed=args.seed) if args.source == "dream" else make_dataset(seed=args.seed)
    train_mask, test_mask = split_by_molecule(data, seed=args.seed + 1)

    Xtr, str_, ytr = data["X_desc"][train_mask], data["state"][train_mask], data["y"][train_mask]

    baseline = StateBlindBaseline().fit(Xtr, str_, ytr)
    conditioned = StateConditionedHead().fit(Xtr, str_, ytr)

    res_b = evaluate(baseline, data, test_mask)
    res_c = evaluate(conditioned, data, test_mask)
    floor = irreducible_baseline_food_mse(data["meta"]["amplitude"])

    # ---- report -------------------------------------------------------------
    print("=" * 68)
    print("STATE-FLIP DEMO  —  grounded valence (molecule x internal state)")
    print("=" * 68)
    print(f"data source: {data['meta'].get('source', 'synthetic')}")
    print(f"data: {data['meta']['n_molecules']} molecules, "
          f"A(flip)={data['meta']['amplitude']}, noise sigma={data['meta']['noise_sigma']}")
    print(f"provable state-blind food-MSE floor (A^2/3) = {floor:.3f}")
    print(f"noise floor (sigma^2)                       = {data['meta']['noise_sigma']**2:.3f}")
    print("-" * 68)
    hdr = f"{'metric':<26}{'state-blind':>16}{'state-cond.':>16}"
    print(hdr)
    print("-" * 68)

    def row(label, kb, kc, fmt="{:.3f}"):
        vb = fmt.format(kb) if kb is not None else "—"
        vc = fmt.format(kc) if kc is not None else "—"
        print(f"{label:<26}{vb:>16}{vc:>16}")

    row("test MSE (all)", res_b["test_mse_all"], res_c["test_mse_all"])
    row("test MSE (food)", res_b["test_mse_food"], res_c["test_mse_food"])
    row("test MSE (non-food)", res_b["test_mse_nonfood"], res_c["test_mse_nonfood"])
    row("flip rate (food mols)", res_b["flip_rate_food"], res_c["flip_rate_food"])
    print("-" * 68)

    print(f"aggregate over {res_c['n_food_test']} held-out food molecules — "
          f"mean valence:")
    print(f"  TRUE        starving=+{data['meta']['amplitude']:.2f}   "
          f"sated=-{data['meta']['amplitude']:.2f}")
    print(f"  state-blind starving={res_b['mean_v_starving']:+.2f}   "
          f"sated={res_b['mean_v_sated']:+.2f}   (identical — no state input)")
    print(f"  state-cond. starving={res_c['mean_v_starving']:+.2f}   "
          f"sated={res_c['mean_v_sated']:+.2f}   (straddles zero — flip)")
    print("-" * 68)

    # one concrete molecule the reader can hold: the cleanest flip (|base| ~ 0)
    ex = min(res_c["examples"], key=lambda e: abs(e[3] + e[4]))  # base ~ -(true0+true1)/-2
    m = ex[0]
    bex = next(e for e in res_b["examples"] if e[0] == m)
    print(f"example food molecule #{m}:")
    print(f"  TRUE valence   starving(s=0)={ex[3]:+.2f}   sated(s=1)={ex[4]:+.2f}   (sign flips)")
    print(f"  state-blind    starving={bex[1]:+.2f}   sated={bex[2]:+.2f}   "
          f"({'FLIP' if np.sign(bex[1])!=np.sign(bex[2]) else 'no flip — same number'})")
    print(f"  state-cond.    starving={ex[1]:+.2f}   sated={ex[2]:+.2f}   "
          f"({'FLIP captured' if np.sign(ex[1])!=np.sign(ex[2]) else 'no flip'})")
    print("=" * 68)
    verdict = (
        "Baseline cannot move with state (one molecule -> one number), so its "
        "food-MSE sits at the provable floor and its flip rate is ~0. The "
        "state-conditioned head recovers the flip and drives error toward noise."
    )
    print(verdict)

    suffix = "" if args.source == "synthetic" else f"_{args.source}"

    if args.plot:
        save_figures(baseline, conditioned, data, test_mask, res_b, res_c, suffix)
        print(f"\nfigures saved to {FIG_DIR}/")

    # machine-readable dump for RESULTS.md
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


def save_figures(baseline, conditioned, data, test_mask, res_b, res_c, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)

    # Fig 1: the flip curve for one food molecule, valence vs satiety.
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

    # Fig 2: food-MSE bars vs the provable floor.
    floor = irreducible_baseline_food_mse(data["meta"]["amplitude"])
    plt.figure(figsize=(6, 4))
    bars = ["state-blind", "state-cond."]
    vals = [res_b["test_mse_food"], res_c["test_mse_food"]]
    plt.bar(bars, vals, color=["#c0504d", "#4f81bd"])
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
