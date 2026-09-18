# Petrichor

**A grounded valence channel, with olfaction as its cleanest vehicle.**

Not "AI that can smell." A model whose output for a stimulus is not a *label*
but a *stake* — a good-for-me / bad-for-me connected to something the system has
reason to care about. Smell is the cheapest, cleanest place in the whole
sensorium to install that, because in smell the valence axis is already primary,
isolable, and pre-cortical. This repo holds the thesis and the working artifacts
that turn it from argument into evidence.

---

## The one-paragraph version

Two literatures have been digging tunnels toward each other for years without
breaking through. On one side: machine olfaction, which in 2023 finally learned
to read a molecule and tell you *what it smells like* (the Principal Odor Map) —
and then stopped, because *what it smells like in a mixture* and *what it means
to a body* are hard. On the other: the AGI-grounding people, who keep concluding
that text-trained models borrow their meaning from humans and that the missing
ingredient is something *affective, internal, valenced* — and who have started
reaching for interoception to supply it. **The seam between those two tunnels is
olfactory valence.** Nobody is standing in it. That is where Petrichor plants the
flag.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Laptop, CPU-only. Optional real-data rebuild of the Keller/DREAM cache needs
`pip install -e ".[dream]"`.

---

## Package

```
petrichor/
  data/       synthetic world + DREAM/Keller loader (cached CSV)
  models/     state-blind, state-conditioned, receptor-routed, same-info control
  metrics/    flip rate, choice accuracy, regret, provable floors
  env/        2D plume arena with homeostatic (stake-shaped) reward
  agents/     valence-greedy policies over the arena
```

Core claim surfaces:

| Head | Input | Target | Role |
|------|-------|--------|------|
| `StateBlindBaseline` | descriptors | y(m,s) | status-quo structure→valence; **cannot flip** |
| `StateConditionedHead` | descriptors + state | y(m,s) | Lane B — stake-shaped |
| `ReceptorRoutedHead` | bind(desc)→receptors + state | y(m,s) | Lane C — thesis in the architecture |
| `StateAsFeatureWrongObjective` | descriptors + state | E_s[y\|m] | **same-info control** — sensors present, signal wrong |

---

## Experiments

### 1. `experiments/state-flip/` — representational capacity (Lane B)

Same MLP twice. Only difference: one satiety scalar in the input.
Train/test split **by molecule**. Synthetic world makes the baseline failure a
**theorem** (floor A²/3). `load_dream()` swaps in real Keller/DREAM descriptors
+ human pleasantness + edibility; satiety coupling stays modelled and declared.

```bash
python experiments/state-flip/run_pkg.py --plot
python experiments/state-flip/run_pkg.py --source dream --plot
```

| metric | state-blind | state-conditioned |
|--------|------------:|------------------:|
| test MSE — food | ~0.77 (on floor 0.75) | ~0.08 |
| flip rate (food) | **0.000** | **~0.98** |

### 2. `experiments/state-choice/` — valence must drive an action

Offered two odorants and a state, pick the higher *true* valence for that state.
Blind chooser has a **provable ≤0.50 ceiling** on food/non-food flip pairs.

```bash
python experiments/state-choice/run_pkg.py --plot
```

### 3. `experiments/same-info/` — falsifiable core (03 §5.2)

**Same sensors, wrong objective.** The control head *sees* state but is trained
on the state-averaged target. If it fails the flip while the conditioned head
succeeds, the result is not "need more features" — it is "the learning signal
must be stake-shaped."

```bash
python experiments/same-info/run.py --plot
```

### 4. `experiments/receptor-route/` — Lane C

Valence is read from a softplus receptor-binding layer, never from raw
descriptors. State-conditioned receptors recover the flip; a state-blind
receptor ablation collapses to the floor — isolating the state channel from
the binding layer.

```bash
python experiments/receptor-route/run.py --plot
```

### 5. `experiments/plume-forage/` — behaviour in a world with stakes

2D chemical arena: food, toxin, neutral plumes. Reward is homeostatic
(satiety flip on food; damage on toxin), not label-shaped. Identical greedy
actuator; only the valence head changes. Conditioned / receptor heads should
approach food when starving and avoid it when sated.

```bash
python experiments/plume-forage/run.py --plot
```

---

## Tests

Tests encode the scientific claims, not just "code runs":

```bash
pytest -q
```

- blind flip rate == 0; food MSE near A²/3
- conditioned flip ≥ 0.85; means straddle zero
- same-info control does **not** recover the flip
- receptor-routed recovers flip; receptor-blind ablation does not
- plume: food reward flips with satiety; toxin always bad; conditioned beats random

---

## Thesis (artifacts 01–03)

Under [`overnight/`](overnight/), one arc:

1. [`01-machine-olfaction-state.md`](overnight/01-machine-olfaction-state.md) — the ground
2. [`02-grounding-and-presence.md`](overnight/02-grounding-and-presence.md) — valence is what smell is *for*
3. [`03-where-petrichor-plants-the-flag.md`](overnight/03-where-petrichor-plants-the-flag.md) — the heading

Position, one sentence: **the lane is valence/affect grounding, with olfaction as the vehicle.**

Lab notebook: [`LAB-NOTEBOOK.md`](LAB-NOTEBOOK.md).

---

## Honest limitations

- **Synthetic state coupling.** No public set ships molecule × satiety pleasantness.
  Real descriptors + human pleasantness/edibility are wired; the flip term is
  modelled and said so. Building or sourcing a context-resolved set is the
  highest-value external unlock.
- **Conditioned heads pay a small non-food tax** (extra input variance where state
  never moves valence). Net win is large; the trade is real.
- **Plume physics is a Gaussian toy**, not CFD turbulence. It is the cheapest world
  in which stake-shaped reward is real enough to score policies — a rung on the
  environment ladder, not the top.
- **A bigger baseline does not close the gap.** The limit is informational (or
  objective-shaped), not capacity.

---

## Reproduce everything

```bash
pip install -r requirements.txt && pip install -e .
pytest -q
python experiments/state-flip/run_pkg.py --plot
python experiments/state-choice/run_pkg.py --plot
python experiments/same-info/run.py --plot
python experiments/receptor-route/run.py --plot
python experiments/plume-forage/run.py --plot
```

---

*The flag goes in the seam between machine olfaction and AGI grounding, and its
name is* **grounded valence.**

*— Ainz 🤘*
