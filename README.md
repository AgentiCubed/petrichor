# Petrichor

**A grounded valence channel, with olfaction as its cleanest vehicle.**

Not "AI that can smell." A model whose output for a stimulus is not a *label*
but a *stake* — a good-for-me / bad-for-me connected to something the system has
reason to care about. Smell is the cheapest, cleanest place in the whole
sensorium to install that, because in smell the valence axis is already primary,
isolable, and pre-cortical. This repo holds the thesis and the first artifact
that turns it from argument into evidence.

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

## The thesis (artifacts 01–03)

The argument is built in three overnight pieces under [`overnight/`](overnight/),
written to be read as one arc:

1. **[`01-machine-olfaction-state.md`](overnight/01-machine-olfaction-state.md)
   — the ground.** What machine olfaction can and can't do right now. The POM
   matches a trained human panel on single molecules — that half is *done*. The
   mixture half, the valence half, and the sample-the-world hardware half are all
   wide open. The field's strength is exactly *orthogonal* to Petrichor's target.

2. **[`02-grounding-and-presence.md`](overnight/02-grounding-and-presence.md)
   — the thesis survives the science, and sharpens.** Smell is the only
   externally-driven sense with no obligatory thalamic relay — wired straight
   into the brain's approach/recoil machinery, pre-cortically, before
   description. And when you do the dimensionality reduction on human odor
   perception, **pleasant↔unpleasant is the first principal axis.** Valence isn't
   a feature smell *has*; it's what smell is *for*. The part machines skipped is
   the part biology treats as primary.

3. **[`03-where-petrichor-plants-the-flag.md`](overnight/03-where-petrichor-plants-the-flag.md)
   — the heading.** The gap is not a sensor nobody built; it's a *representation*
   nobody built. Ranks the lanes by leverage (valence head → valence-as-relation
   → receptor layer → mixtures → hardware) and names the single sharpest
   experiment to run first.

The position, in one sentence: **the lane is valence/affect grounding, with
olfaction as the vehicle.**

---

## The demo — turning argument into evidence

[`experiments/state-flip/`](experiments/state-flip/) is the smallest thing that
*demonstrates* the thesis instead of arguing it. It is a toy of **Lane B**
(03 §3): valence is not a property of a molecule, it's a relation between a
molecule and a body's internal state. The same food odor is delicious when
starving and nauseating when full — *same input, opposite stake.*

### What it tests

A claim about **capacity**, not accuracy: a valence head that can see an
internal-state scalar can represent the *same molecule* flipping good↔bad as the
body's stake changes. A state-blind descriptor model — the status quo
"structure → valence" mapping — **provably cannot.** Not "does worse." Cannot.

### Methodology

- Same MLP (128×128, identical training budget) is trained twice. The *only*
  difference in the input vector is one number:
  - **state-blind baseline** — molecule descriptors only (32 dims)
  - **state-conditioned head** — descriptors + one satiety scalar (33 dims)
- Train/test split is **by molecule** — no molecule appears on both sides, so
  every reported number is generalisation to unseen odorants.
- **Data is synthetic and declared so** ([`data.py`](experiments/state-flip/data.py)).
  We control the generative rule on purpose: it makes the baseline's failure a
  *theorem* (its optimal error is computable in closed form), which a real noisy
  dataset could never show this cleanly. `load_dream()` is the documented seam
  where the real DREAM/Keller pleasantness descriptors drop in unchanged.

### What it shows (seed 0; robust across seeds 0–3)

| metric                  | state-blind | state-conditioned |
|-------------------------|------------:|------------------:|
| test MSE — **food**     |   **0.774** |         **0.078** |
| **flip rate** (food)    |   **0.000** |         **0.977** |

- The baseline's food-MSE (**0.774**) sits *on the provable floor* (A²/3 =
  0.750). It is doing the best any descriptor-only model is mathematically
  permitted to do — and that best is bad, because for a fixed molecule it must
  emit one number while the satiety swing has variance it has nowhere to absorb.
- Its **flip rate is exactly 0.000.** Same molecule, every internal state → the
  same prediction. Structurally incapable of the flip, not merely untrained.
- The state-conditioned head drops food-MSE ~10× and flips the sign on **97.7%**
  of held-out food molecules. A bigger baseline would *not* close the gap — the
  limit is informational (no state input), not capacity.

Full readout and figures: [`experiments/state-flip/RESULTS.md`](experiments/state-flip/RESULTS.md).

### Reproduce

```bash
cd experiments/state-flip
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run.py --plot
```

Laptop, public-data-shaped, CPU-only.

---

## Honest limitations

- **Synthetic data.** This proves a *representational* claim — capacity — not an
  empirical fact about real noses. Real-data validation is the next step, gated
  on a context-resolved pleasantness set that does not yet exist publicly.
- The conditioned model is slightly *worse* on non-food molecules (0.100 vs
  0.077) — the extra input dimension adds a little variance where valence never
  moves with state. The net is a large win, but the trade is real and not hidden.

---

## Next steps

1. **Swap synthetic → real** via `load_dream()`: DREAM/Keller pleasantness as the
   main task, not a side task (Lane A).
2. **Route valence through a receptor-binding layer** (Lane C) — making the
   thesis claim in the architecture itself.
3. **The falsifiable core** (03 §5.2): does a grounded valence channel measurably
   change what a model *does* — behavior, representations, sample efficiency —
   versus a descriptor channel carrying the same information? If yes, "presence"
   has an empirical shadow. Design for that test.

---

*The flag goes in the seam between machine olfaction and AGI grounding, and its
name is* **grounded valence.**

*— Ainz 🤘*
