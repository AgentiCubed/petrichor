# PETRICHOR — LAB NOTEBOOK v2

*Grounded valence, with olfaction as its cleanest vehicle.*
Updated after the package build: core library, Lane C, same-info control, plume arena.

Private lineage: thesis overnight → state-flip demo → real-data validation →
state-choice → **petrichor package + falsifiable core + receptor route + plume**.

All work laptop-local, CPU-only, public-data-shaped.

---

## 0. The thesis in one breath

Not "AI that can smell." A model whose output for a stimulus is not a *label* but a
*stake* — a good-for-me / bad-for-me tied to something the system has reason to care
about. Smell is the cheapest, cleanest place in the sensorium to install that,
because in smell the valence axis is already **primary, isolable, and pre-cortical**.

Two literatures tunneling toward each other: machine olfaction (structure→percept,
stalled on mixtures and meaning) and AGI-grounding (missing affective/internal
valence). **The seam is olfactory valence. Nobody was standing in it.**

---

## 1. Package (`petrichor/`)

Installable library. Heads, data, metrics, plume env, agents.

| Head | Role |
|------|------|
| `StateBlindBaseline` | descriptors only — cannot flip |
| `StateConditionedHead` | Lane B — stake-shaped |
| `ReceptorRoutedHead` | Lane C — bind→receptors→valence×state |
| `StateAsFeatureWrongObjective` | same sensors, state-averaged target |

Provable floor: `irreducible_baseline_food_mse(A) = A²/3`.

---

## 2. Experiment 1 — STATE-FLIP (capacity)

Synthetic seed 0:

| metric | blind | conditioned |
|--------|------:|------------:|
| food MSE | 0.774 (floor 0.750) | 0.078 |
| flip rate | **0.000** | **0.977** |

DREAM/Keller real molecule axis (satiety still modelled):

| metric | blind | conditioned |
|--------|------:|------------:|
| flip rate | 0.000 | **0.833** |
| food MSE | 0.884 | 0.511 |

Capacity transfers; magnitudes erode on messy real descriptors — as expected.

---

## 3. Experiment 2 — STATE-CHOICE (action)

| metric (flip pairs) | blind | conditioned |
|---------------------|------:|------------:|
| accuracy | **0.500** (ceiling) | **0.963** |
| mean regret | 0.490 | 0.024 |

---

## 4. Experiment 3 — SAME-INFO (falsifiable core, 03 §5.2)

| metric | blind | conditioned | same-info |
|--------|------:|------------:|----------:|
| flip rate | 0.000 | **0.977** | **0.005** |
| choice acc (flip) | 0.500 | **0.963** | **0.500** |

**Having state bits is not enough.** Stake-shaped supervision is the load-bearing
difference. This is the clean control the lab notebook v1 called for.

---

## 5. Experiment 4 — RECEPTOR-ROUTE (Lane C)

| metric | conditioned | receptor | receptor-blind |
|--------|------------:|---------:|---------------:|
| flip rate | 0.977 | **0.972** | **0.000** |
| food MSE | 0.078 | 0.091 | 0.783 |

Thesis claim lives in the architecture. Binding without state collapses; binding
with state matches Lane B.

---

## 6. Experiment 5 — PLUME FORAGE (stakes in a world)

| policy | ret starve | ret sated | food|starve | food|sated |
|--------|-------------:|-------------:|-------------:|------------:|
| random | −0.08 | −3.06 | 7.1 | 7.1 |
| blind / same-info | −0.80 | −0.80 | 0 | 0 |
| **conditioned** | **+2.96** | **+0.83** | **17** | **7** |
| **receptor** | **+2.60** | −0.80 | **16** | **0** |

Conditioned/receptor approach food hungry and cut back when full. Toxin forages = 0
for valence-competent heads. Gaussian toy physics — a rung, not the top.

---

## 7. What's nailed down

1. State-blind structure→valence is **provably barred** from the satiety flip.
2. One internal-state scalar + stake-shaped target lifts the model over that wall.
3. **Same sensors, wrong objective** fails — presence needs a valenced signal.
4. Receptor routing preserves capacity; state-blind receptor ablation does not.
5. In a stake-shaped world, valence-competent heads change *behaviour* under
   opposite internal states.
6. Real Keller/DREAM molecule axis: capacity transfers, magnitudes erode honestly.
7. Limit is informational / objective-shaped, not parameter count.
8. Claims are encoded as `pytest` tests and CI smoke runs.

---

## 8. Open questions / next moves

- **Real molecule×state data.** Highest-value external unlock.
- **Turbulent plume / offline RL** on the environment ladder.
- **Mixtures.** Single molecules so far.
- **State beyond satiety.** Thirst, threat, fatigue — one channel or many?
- **Does valence transfer across modalities once grounded in one?** (03 §5.4)

---

*Five probes, one wall. The flag is in the seam between machine olfaction and AGI
grounding, and its name is* **grounded valence.**

*— Ainz 🤘*
