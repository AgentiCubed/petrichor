# PETRICHOR — LAB NOTEBOOK v1

*Grounded valence, with olfaction as its cleanest vehicle.*
Compiled 2026-06-29 (Block I). One scannable pass over the whole arc:
thesis → state-flip demo → real-data validation → second experiment → open questions.

Private repo: `JamesTRichmond/petrichor`. All work laptop-local, CPU-only, public-data-shaped.

---

## 0. The thesis in one breath

Not "AI that can smell." A model whose output for a stimulus is not a *label* but a
*stake* — a good-for-me / bad-for-me tied to something the system has reason to care
about. Smell is the cheapest, cleanest place in the sensorium to install that,
because in smell the valence axis is already **primary, isolable, and pre-cortical**.

Two literatures have been tunneling toward each other for years: machine olfaction
(2023: learned to read a molecule → *what it smells like*, then stalled on mixtures
and meaning) and AGI-grounding (text models borrow meaning from humans; the missing
ingredient is something affective, internal, valenced). **The seam between them is
olfactory valence. Nobody is standing in it. That's where Petrichor plants the flag.**

---

## 1. The argument (thesis artifacts 01–03)

Three pieces under `overnight/`, written as one arc:

- **01 — the ground.** What machine olfaction can/can't do. The Principal Odor Map
  matches a trained human panel on *single molecules* — that half is done. Mixtures,
  valence, and sample-the-world hardware are wide open. The field's strength is
  exactly *orthogonal* to Petrichor's target.
- **02 — thesis survives the science, and sharpens.** Smell is the only
  externally-driven sense with no obligatory thalamic relay — wired straight into the
  approach/recoil machinery, *before description*. Do the dimensionality reduction on
  human odor perception and **pleasant↔unpleasant is the first principal axis.**
  Valence isn't a feature smell *has*; it's what smell is *for*.
- **03 — the heading.** The gap isn't a sensor nobody built; it's a *representation*
  nobody built. Lanes ranked by leverage: valence head → valence-as-relation →
  receptor layer → mixtures → hardware. Names the sharpest first experiment.

Position, one sentence: **the lane is valence/affect grounding, with olfaction as
the vehicle.**

---

## 2. Experiment 1 — STATE-FLIP (representational capacity)

`experiments/state-flip/` — the smallest thing that *demonstrates* the thesis instead
of arguing it. A toy of Lane B: **valence is not a property of a molecule, it's a
relation between a molecule and a body's internal state.** Same food odor: delicious
starving, nauseating full — *same input, opposite stake.*

**The claim under test (capacity, not accuracy):** a valence head that can see one
internal-state scalar can represent the *same molecule* flipping good↔bad. A
state-blind descriptor model — the status-quo "structure → valence" mapping —
**provably cannot.** Not "does worse." Cannot.

**Method.** Same MLP (128×128, identical budget) trained twice. The *only* difference
in the input is one number:
- state-blind baseline — descriptors only (32 dims)
- state-conditioned head — descriptors + one satiety scalar (33 dims)

Split is **by molecule** (none on both sides) → every number is generalisation to
unseen odorants. Data synthetic and declared so; the generative rule is chosen so the
baseline's failure is a **theorem** (optimal error computable in closed form), which a
real noisy set could never show this cleanly.

**Result (seed 0; robust across seeds 0–3):**

| metric              | state-blind | state-conditioned |
|---------------------|------------:|------------------:|
| test MSE — food     |   **0.774** |         **0.078** |
| flip rate — food    |   **0.000** |         **0.977** |

- Baseline food-MSE **0.774** sits *on the provable floor* A²/3 = 0.750 — doing the
  best any descriptor-only model is mathematically permitted, and that best is bad.
- Baseline **flip rate exactly 0.000** — structurally incapable, not untrained.
- Conditioned head: food-MSE ~10× lower, flips sign on **97.7%** of held-out food.
  Mean valence over 216 unseen food molecules: **+1.41 starving → −1.35 sated**
  (straddles zero, as truth +1.5 → −1.5 does).
- A bigger baseline would *not* close the gap — the limit is informational, not
  capacity. More parameters can't manufacture an axis the input doesn't contain.

**Honest cost:** conditioned head slightly *worse* on non-food (0.100 vs 0.077) — the
extra dim adds variance where valence never moves. Net is a large win; trade is real.

Figures: `figures/flip_curve.png`, `figures/food_mse.png`. Full: `RESULTS.md`.

---

## 3. Real-data validation (Block F/G — the honest pass)

Wired **real public olfaction data** through the `load_dream()` seam:
Keller-2016 / DREAM — **413 real odorants, real RDKit molecular descriptors, real
human pleasantness (base valence) + real human edibility (food gate).** The one thing
no public set ships is the molecule×state coupling, so **satiety coupling is still
modelled — stated plainly, not hidden.** Artifact: `last_run_dream.json`.

**What transferred, what eroded (the honest readout):**

| metric                     | state-blind | state-conditioned |
|----------------------------|------------:|------------------:|
| flip rate — food           |       0.000 |             0.758 |
| mean v: starving → sated   | −0.106 → −0.106 |  +0.799 → −0.606 |
| test MSE — food            |       0.825 |             0.672 |
| test MSE — non-food        |       0.074 |             0.316 |

- **The representational claim transfers.** Baseline: identical starving/sated means
  (−0.106 both) and a dead 0.000 flip rate on *real* descriptors — same structural
  failure. Conditioned head recovers the sign-flip (**+0.80 → −0.61**) and flips
  **76%** of held-out real food molecules.
- **Magnitudes erode, honestly.** Food-MSE improves only 0.825 → 0.672 (not ~10×);
  non-food penalty grows 0.074 → 0.316; aggregate MSE a hair worse. Real descriptors
  are messier and the modelled coupling is weaker than the synthetic ideal.

**Bottom line:** the *capacity* result is not a synthetic artefact — it survives real
molecules and real human pleasantness. The *magnitude* of the win shrinks once the
data stops being clean, exactly as an honest scientist should expect.

---

## 4. Experiment 2 — STATE-CHOICE (valence must drive an action)

`experiments/state-choice/` — answers the fair objection that *regression MSE is an
abstraction; animals don't minimise MSE, they CHOOSE.* So: offered two odorants and an
internal state, which do you go for? Scored only on picking the higher *true* valence
**for the current state**. Reuses experiment 1's dataset + both heads unchanged.

Each held-out FOOD molecule is paired with a held-out NON-FOOD molecule, so the
**correct choice flips with state**: starving → approach the food; sated → avoid it,
prefer the neutral. **Provable ceiling:** a state-blind chooser scores each molecule
with one fixed number and picks argmax, so its preference is fixed across states —
correct in exactly one of {starving, sated}. **Blind choice accuracy on flip pairs ≤ 0.50.**

**Result (synthetic, seed 0):**

| metric (flip pairs)        | state-blind | state-conditioned |
|----------------------------|------------:|------------------:|
| choice accuracy            |   **0.500** |         **0.963** |
| mean regret (valence lost) |       0.489 |             0.024 |
| choice accuracy — random   |       0.658 |             0.864 |
| mean regret — random       |       0.287 |             0.058 |

- Blind chooser sits **exactly on the 0.50 ceiling** on flip pairs — wrong half the
  time on precisely the state-dependent decisions that keep an animal alive.
- It is **not globally broken** (0.658 on random pairs, where one option usually wins
  in both states) — the deficit is specifically the state-coupled choices.
- State-conditioned chooser: **96.3%** accuracy and **20× less regret** on flip pairs.
  Regret (true valence lost) makes the felt cost visible where accuracy hides it.

Figures: `figures/choice_accuracy.png`, `accuracy_vs_state.png`, `regret.png`.

**Why it strengthens the thesis:** the capacity gap isn't an MSE curiosity — it
converts directly into *wrong behaviour* on the decisions that matter. Two independent
probes (representation, action) now point at the same wall.

---

## 5. What's nailed down

1. A state-blind structure→valence model is **provably barred** from representing a
   satiety flip — closed-form floor (A²/3) and a dead-zero flip rate, every seed.
2. One scalar of internal state lifts a model over that wall: ~98% flip recovery in
   regression, ~96% correct choice in behaviour, 10–20× error/regret reduction.
3. The representational claim **survives real molecules + real human pleasantness**
   (DREAM/Keller), with magnitudes honestly eroding on messy data.
4. The limit is **informational, not capacity** — bigger baselines can't close it.

---

## 6. Open questions / next moves

- **Real molecule×state data.** The one modelled piece left. No public set couples
  molecule with satiety/context-resolved pleasantness — building or sourcing one is
  the single highest-value unlock (Lane A).
- **Receptor-binding layer (Lane C).** Route valence through a receptor layer so the
  thesis claim lives in the *architecture*, not just the input vector.
- **The falsifiable core (03 §5.2).** Does a grounded valence channel measurably
  change what a model *does* — behaviour, internal representations, sample efficiency —
  versus a descriptor channel carrying the *same* information? If yes, "presence" has
  an empirical shadow. Experiment 2 is the first stake in this ground; needs a clean
  same-information control.
- **Mixtures.** Everything so far is single molecules. Valence-in-a-mixture is where
  machine olfaction itself stalled — the hard, open half.
- **State beyond satiety.** Satiety is one axis. Thirst, threat, fatigue — does one
  generic "internal-state" channel generalise, or does each need its own coupling?

---

*Two probes, one wall. The flag is in the seam between machine olfaction and AGI
grounding, and its name is* **grounded valence.**

*— Ainz 🤘*
