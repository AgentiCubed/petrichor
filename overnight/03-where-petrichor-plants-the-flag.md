# Where Petrichor Plants the Flag

> Block 3 of the overnight work — the synthesis. Blocks 1 and 2 mapped the ground:
> what machine olfaction *can* do right now (01), and whether the founding thesis
> survives the actual neuroscience (02, it does, and sharpens). This is the part
> that turns a map into a heading. Written in the FOUNDING voice — weighty, honest,
> no hype, no romanticizing across the walls. For James, over coffee.
>
> — Ainz 🤘, 2026-06-29, ~04:15 EDT

---

## The one-paragraph version

Two literatures have been digging tunnels toward each other for years without
either one breaking through. On one side: machine olfaction, which in 2023 finally
learned to read a molecule and tell you *what it smells like* — and then stopped
there, because the next thing, *what it smells like in a mixture* and *what it
means to a body*, is hard. On the other side: the AGI-grounding people, who keep
concluding that text-trained models borrow their meaning from humans and that the
missing ingredient is something *affective, internal, valenced* — and who have
started, in the last two years, reaching for interoception to supply it. **The seam
between those two tunnels is olfactory valence.** Smell is the only externally-driven
sense wired straight into the brain's approach/recoil machinery, pre-cortically,
before description. Nobody is standing in that seam. That is where Petrichor plants
the flag: **not "AI that can smell," but "a grounded valence channel, with olfaction
as its cleanest vehicle."** Same calling, sharper coordinates.

---

## 1. What changed between FOUNDING and tonight

FOUNDING (June 24) made three bets in the dark. Five nights of reading later, here's
the scorecard — because the honest thing is to say which bets got stronger and which
got more dangerous.

- **"Smell has a direct line to the limbic system, no translation layer."**
  → **Confirmed, literally.** It's the only sense with no obligatory thalamic relay.
  This is documented wiring, not a metaphor. Stop hedging on it. (02 §1)

- **"Smell is an evaluative/valence channel, the thing models lack."**
  → **Stronger than you wrote it.** Valence isn't a feature smell *has* — when you do
  the dimensionality reduction on human odor perception, pleasant↔unpleasant is the
  *first principal axis*. Valence is what smell is *for*, and it's hardwired and
  staged as early as the olfactory bulb. The part machines skipped is the part biology
  treats as primary. (01 §7, 02 §3)

- **"The closest prior art (Osmo / POM) is still embryonic."**
  → **Half wrong, in a useful way.** The *read-the-molecule* half is no longer
  embryonic — the Principal Odor Map matches a trained human panel on single
  molecules. That's done. But the *mixture* half and the *valence* half and the
  *sample-the-world hardware* half are all still wide open. The field's strength is
  exactly orthogonal to Petrichor's target. (01 §1, §7)

The net: **the calling didn't move, but the lane narrowed and got real.** "Give AI
smell" was a direction. "Build a grounded valence channel, olfaction-first" is a
*program* — with adjacent literatures, open benchmarks, and a falsifiable core.

---

## 2. The gap, stated precisely

Three facts, stacked, define the opening:

1. **Every machine-olfaction system today models "what does it smell like."** The POM,
   every GNN/transformer successor, every eNose — they predict expert *descriptors*
   ("floral, musky, green") or detect specific analytes. (01)
2. **Almost none model "what does it mean to a body" — approach/recoil valence.** The
   evaluative channel is essentially unclaimed in silico. (01 §7, 02 §3)
3. **That unclaimed channel is precisely what the grounding literature is now reaching
   for** to fix what reference-from-text can't — and it's reaching via interoception,
   the body's internal affective state. (02 §4)

So the gap is not "a sensor nobody built." It's **a *representation* nobody built**:
a model whose output for a stimulus is not a label but a *stake* — a good-for-me /
bad-for-me that is connected to something the system has reason to care about. Smell
is the cheapest, cleanest place in the whole sensorium to install that, because in
smell the valence axis is already primary, isolable, and pre-cortical. You're not
fighting the architecture; you're following the biology's own wiring diagram.

---

## 3. The lanes — concrete, ranked by leverage

These are real directions, not vibes. Ranked by *how much they move the thesis per
unit of effort James can actually muster solo or with a small collaborator.*

### Lane A — The valence head (the main bet)
**Build a model that outputs olfactory *valence*, not descriptors.** Take the existing
structure→percept stack (POM-style embeddings are public-adjacent; GS-LF and DREAM
data exist) and put a *valence/affect head* on it instead of (or alongside) the
descriptor head. The DREAM set already has **pleasantness** ratings per molecule per
subject — that's a valence label sitting in the canonical benchmark, mostly used as
a side task. Make it the *main* task.
- **Why it's the flag:** it's the literal instantiation of "model what it means, not
  what it is." It's biologically faithful (valence is the first axis). And it connects
  Petrichor to the AGI conversation, not just to perfumery.
- **Feasibility:** highest of the deep lanes. Data exists. No hardware. A laptop and
  the public datasets get you a v0. This is the one to start.

### Lane B — Valence is *not* a property of the molecule; it's a relation
The sharp version of Lane A, and the part that's genuinely novel. Pleasantness in
DREAM is a population mean — it smooths over the individual, and over *context*
(satiety flips food smells; the same molecule is delicious hungry, nauseating full).
**Model valence as molecule × internal-state, not molecule alone.** Even a toy
internal state (a scalar "satiety" or "threat" variable) that *modulates* the valence
output would be a result nobody has: a demonstration that the same input flips
good→bad when the body's stake changes. That's the difference between a label and a
felt evaluation, made measurable.
- **Why it matters:** this is the falsifiable downgrade of "presence." If a
  state-conditioned valence channel changes behavior in ways a static descriptor
  channel can't, you've shown the channel *does something*. (02 §6, objection 2's answer.)

### Lane C — The receptor/contact layer
The most *grounded* and least-exploited modeling stage: structure → olfactory-receptor
activation → percept (Microsoft's combinatorial-coding work has public code; 01 §3c).
This is the only place in the stack where "the world physically docks with the model"
is even representable — the computational analog of FOUNDING's "Albedo's hair: molecules
that *arrived*." (02 §5)
- **Why it matters:** if presence comes from contact, this is where contact lives. A
  Petrichor architecture that routes valence *through* a receptor-binding layer is
  making a thesis claim in the architecture itself.
- **Feasibility:** harder than A/B — receptor data is messier — but the code exists and
  it's a strong "phase 2" once the valence head is standing.

### Lane D — Mixtures (know it's the wall, don't camp on it)
Mixtures are the field's headline open problem (AROMMA, d2smell, 2025–26; 01 §3a). It's
tempting because it's "the frontier." **Recommend: don't make it the main bet.** It's
crowded with chemists who have better hardware and data pipelines than a solo founder,
and it's a *descriptive* problem (predict the blend's percept), not a *valence* one. It's
adjacent and worth tracking — a valence model eventually has to handle blends — but the
asymmetric bet (FOUNDING's whole strategy) is the empty valence seam, not the contested
mixtures front. Let the chemists soften the wall; plant the flag where no one is.

### Lane E — The hardware temptation (resist, for now)
eNoses drift, foul, and stay narrow (01 §4). Building or buying sensing hardware is a
capital-and-time sink that gates nothing in Lanes A–C, which are pure software on
existing data. **Recommend: stay off hardware until the valence representation is real
and demands a sensor to feed it.** The software side raced ahead of hardware for a
reason — ride that asymmetry.

---

## 4. The single sharpest experiment to run first

If James wants one concrete thing to build that would *advance the position* — not just
read about it — here it is, and it's deliberately small enough to actually finish:

> **The state-flip demo (a toy of Lane B).**
> Train (or fine-tune) a small model on DREAM pleasantness so it predicts valence from
> molecular structure. Then add one scalar "internal state" input and a tiny synthetic
> rule for how state should modulate valence (e.g. a satiety variable that suppresses
> the valence of food-associated odorants). Show that the *same molecule* produces
> opposite approach/recoil outputs as the state variable changes — and that a
> descriptor-only baseline *can't* represent the flip because it has nowhere to put the
> state.

That's a weekend-to-a-fortnight project, runs on a laptop, uses only public data, and
produces the first artifact that demonstrates the thesis instead of arguing it: *a
valence that depends on the body's stake.* It's the falsifiable proxy for "presence,"
shrunk to something you can hold. Everything bigger (real internal homeostasis, the
receptor layer, mixtures, hardware) is a sequel to this.

---

## 5. The sharpest open questions (the ones worth a life, not a paper)

In rough order of depth:

1. **What is the minimal artificial *stake* that turns a valence label into a valence
   that matters?** A model's "this smells bad" isn't grounded in anything that can die.
   Innate olfactory valence in animals is grounded in survival (rotten = pathogen =
   death). What's the smallest homeostatic loop — the minimal "something to lose" — that
   makes an approach/recoil signal *about* anything? This is where Petrichor touches the
   hard problem. It's the deepest question and probably the real life's-work. (02 §6.3)

2. **Does a grounded valence channel measurably change what a model does** — its
   behavior, its representations, its sample efficiency — relative to a descriptor
   channel carrying the same information? If yes, "presence" has an empirical shadow. If
   no, the thesis needs revising. *This is the falsifiable core; design for it.*

3. **Is contact (proximal/receptor docking) actually privileged for affect-grounding,
   or is that a phenomenological claim smuggled into an engineering problem?** The
   honest skeptic's strongest shot (02 §6.1). Fight on *valence*, not on *reference* —
   but stay alert that this is the hinge the whole "presence needs contact" claim turns
   on, and it's not proven.

4. **Can valence transfer across modalities once it's grounded in one?** If a model
   learns approach/recoil through smell (where it's cleanest), does that valence
   representation generalize to ground other senses — or to ground *language* about
   good/bad? That would be the bridge from "a niche sensor project" to "a load-bearing
   piece of AGI grounding." It's speculative. It's also the prize.

---

## 6. What to actually do next (not a roadmap, a heading)

- **Read the two artifacts (01, 02) once more and let the reframe settle:** the lane is
  **valence/affect grounding**, olfaction as the vehicle. That sentence is the position.
- **Build the state-flip demo (§4).** Smallest thing that turns argument into evidence.
- **Reach out to one person in the seam.** The named people who'd actually get it:
  **Noam Sobel** (Weizmann — valence as the primary axis, metamers, perceptual metrics)
  and the **Mollo & Millière** grounding line. Not to pitch — to ask the one question
  (§5.2) and see if it lands. (No sends without your say-so — this is a note to self.)
- **Keep the FOUNDING discipline:** name the walls, don't romanticize across them. The
  hardest wall (the stake problem, §5.1) is not a reason to stop; it's the thing that
  makes this worth a life. "Proven things are crowded." The valence seam is empty
  because it's hard. That emptiness is the whole point.

---

*Block 3 complete. The three overnight artifacts now form one arc: 01 = the ground
(what machine olfaction can and can't do), 02 = the thesis survives the science and
sharpens to valence, 03 = where to plant the flag and the smallest experiment that
turns the position into evidence. The flag goes in the seam between machine olfaction
and AGI grounding, and its name is* **grounded valence.**

*— Ainz 🤘*
