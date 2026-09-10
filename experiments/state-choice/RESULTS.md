# RESULTS — state-choice

**The claim:** when valence must drive an *action* (pick the better odorant for
the current internal state), a state-blind chooser is provably capped at 0.50
accuracy on food/non-food flip pairs. A state-conditioned head is not.

## Headline numbers (synthetic, seed 0)

| metric (flip pairs) | state-blind | state-conditioned |
|---------------------|------------:|------------------:|
| choice accuracy | **0.500** | **0.963** |
| mean regret (valence lost) | 0.490 | **0.024** |
| choice accuracy — random pairs | 0.658 | 0.864 |

Blind sits exactly on the ceiling for flip pairs and is fine on random pairs —
the deficit is specifically the state-dependent decisions.

Figures: `figures/choice_accuracy.png`, `accuracy_vs_state.png`, `regret.png`.

*Reproduce: `python experiments/state-choice/run_pkg.py --plot`*
