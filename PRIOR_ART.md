# Project Petrichor — Prior-Art & Reading Map

> Companion to [FOUNDING.md](FOUNDING.md). This is the lay of the land: who has
> stood where, what's solved, what's wide open. A living map — add to it, argue
> with it, prune it. Each entry: *what it is · why it matters to us.*

**Started:** 2026-06-24. **Curated by:** Ainz, from a first research sweep.
**How to read it:** the five territories below are the problem broken into its
real seams. If you read nothing else, do the **★ Start Here** list at the bottom.

---

## Territory A — The Representation Problem (what *is* a smell, as data?)

The core scientific knot: vision has RGB, audio has a frequency spectrum, smell
has… no agreed basis set. This is the territory where the breakthrough either
happens or doesn't.

- **★ A Principal Odor Map (POM) unifies diverse tasks in olfactory perception** —
  Wiltschko et al., *Science*, Sept 2023. A message-passing graph neural net
  trained on ~5,000 industry-labeled molecules; predicts odor descriptors for
  *unseen* molecules as well as the median human panelist, and found
  structurally-dissimilar molecules that smell alike. *Why it matters:* this is
  the closest thing to "ImageNet for smell" — the current state of the art for
  turning molecular structure into a perceptual vector. The thing to beat / build
  on. https://www.science.org/doi/10.1126/science.ade4401 ·
  open copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC11898014/
- **Hyperbolic geometry of olfactory space** — Zhou, Smith, Sharpee, *Science
  Advances*, 2018. Argues odor perceptual space is best described by 3D *hyperbolic*
  (curved) geometry, organized by how odors co-occur in nature, not by ligand
  chemistry. *Why it matters:* if the space is hyperbolic, Euclidean embeddings
  (most ML) are fighting the geometry. Possible edge.
  https://www.science.org/doi/abs/10.1126/sciadv.aaq1458
- **On the dimensionality of odor space** — Meister, *eLife*, 2015. Skeptical take:
  how many dimensions does smell actually have? Estimates in the literature run
  1–32. *Why it matters:* nobody agrees on the number of "axes" of smell. That
  unsolved question is the heart of the representation problem.
  https://elifesciences.org/articles/07865
- **Perceptual convergence / "olfactory white"** — Weiss & Sobel et al., *PNAS*,
  2012. Mix ~30+ equal-intensity odorants spanning the space and everything
  converges to one generic smell — the olfactory analog of white light / white
  noise. *Why it matters:* a hard empirical constraint any model of odor mixtures
  must reproduce. https://www.pnas.org/doi/10.1073/pnas.1208110109
- **The datasets (the fuel, and the bottleneck):**
  - **Pyrfume** — the central open repo aggregating curated olfactory datasets.
    Start every data conversation here. https://github.com/pyrfume
  - **Leffingwell, GoodScents, Arctander** — the industry odorist-labeled
    structure→descriptor sets (~10k molecules total with human descriptions; ~4k
    with two independent human labels). The backbone training data.
  - **DREAM Olfaction Challenge / Keller et al.** — 480 diverse molecules rated by
    humans on intensity, pleasantness, descriptors. The classic benchmark for
    predicting human perception from chemistry.
  - **Grounding olfactory perception in language: Benchmarks** — bioRxiv, 2026.
    *Very* on-thesis and recent — connecting odor perception to language models.
    Track this. https://www.biorxiv.org/content/10.64898/2026.03.04.709650v1.full.pdf

## Territory B — The Hardware (how do you get a smell *in*?)

The input vector has to come from somewhere physical. Two families: dumb-but-robust
sensor arrays, and smart-but-fragile biology.

- **★ Machine olfaction & embedded AI review** — arXiv, 2025. Survey of where
  e-nose + on-device AI is heading as an industry. Good orientation to the field's
  current shape. https://arxiv.org/html/2510.19660v1
- **ML-assisted gas sensor arrays in medical diagnosis** — review, *Biosensors*
  (MDPI), 2025. The state of metal-oxide / conductive-polymer sensor arrays +
  pattern recognition. *Why it matters:* this is the cheap, available input path —
  and its limits (drift, fouling, low specificity) define the engineering problem.
  https://www.mdpi.com/2079-6374/15/8/548
- **Bioelectronic nose with olfactory-receptor nanodiscs** — *Scientific Reports*,
  2018 (rose-scent demo) + 2024 graphene/nanodisc combinatorial work. Real human
  olfactory receptors (hOR51E1/E2, hOR52D1) reconstituted into nanodiscs on a
  sensor. *Why it matters:* the bio-hybrid path — borrow biology's actual
  receptors instead of approximating them. Highest ceiling, hardest to stabilize.
  https://www.nature.com/articles/s41598-018-32155-1
- **Buck & Axel — combinatorial coding (2004 Nobel)** — the foundational biology:
  each neuron expresses *one* receptor; an odorant lights up a *combination* of
  receptors; the combination *is* the code. *Why it matters:* this is the
  encoding scheme evolution chose. Any artificial nose is, knowingly or not,
  trying to re-implement combinatorial coding.
  https://www.cell.com/cell/fulltext/S0092-8674(04)01163-8

## Territory C — The Signal Worth Smelling (why it's not a toy)

Proof that there's real, high-value information in the chemical channel — the case
that smell is worth grounding an agent in.

- **★ Canine olfaction + GC-MS for disease detection (breath/sweat VOCs)** —
  systematic review, *Frontiers in Chemistry*, 2023. Dogs detect cancers, malaria,
  Parkinson's, epilepsy from volatile organic compounds; GC-MS is the complementary
  instrument. *Why it matters:* hard evidence the channel carries
  life-or-death signal humans can't consciously access. The "killer app" case.
  https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2023.1282450/full
- **Dogs detect the odour of Parkinson's** — medRxiv, 2023 (note: a later
  larger-cohort study gave contradictory results — the honesty caveat lives here).
  https://www.medrxiv.org/content/10.1101/2023.11.01.23296924.full.pdf
- **Human chemosignals** — fear/stress sweat carries socially-readable chemical
  signals; humans communicate affective state through smell below awareness.
  *Why it matters:* directly feeds the FOUNDING thesis that smell is an
  *evaluative/affective* channel, not just informational.

## Territory D — The Output Side (adjacent, not our target — but learn from it)

We care about smell *in* (sensing), but the people furthest ahead are working smell
*out* (reproduction). Their tooling, data, and odor-map are the same substrate.

- **Osmo — "scent teleportation"** — the Wiltschko/POM team's company (spun from
  Google, Jan 2023; **$70M Series B, Feb 2026**). In 2024 claimed the first full
  digitization of a scent (a fresh-cut plum) with no human in the loop: GC-MS →
  molecules → coordinate on the POM → formulation robot mixes it back. *Why it
  matters:* they've productized the encode side and are racing to shrink/portable.
  They are the giant whose shoulders (and gaps) we map against.
  https://www.osmo.ai/about · funding:
  https://www.businesswire.com/news/home/20260204293785/en/
- **AromaGen: interactive generation of olfactory experiences with multimodal
  LLMs** — arXiv, 2026. Smell + multimodal LLMs, the bleeding edge of joining odor
  to language models. Exactly our intersection. https://arxiv.org/pdf/2604.01650

## Territory E — The Deep Why (grounding, embodiment, presence)

The philosophical spine — the argument for *why* sensory vectors matter to AGI, and
where smell sits in it. This is where Petrichor's distinctive thesis lives.

- **★ The Symbol Grounding Problem** — Harnad, 1990. The origin: how do symbols get
  meaning instead of just referring to other symbols? *Why it matters:* the problem
  statement our whole mission is an answer to. https://philarchive.org/archive/HARLWL-4
  (see also Harnad's 2024 "Language Writ Large" on LLMs & grounding)
- **The Vector Grounding Problem** — Mollo & Millière, arXiv, 2023. Reframes
  grounding specifically for LLM representations. The modern, ML-literate version of
  Harnad. https://arxiv.org/pdf/2304.01481
- **A roadmap for embodied and social grounding in LLMs** — arXiv, 2024. The
  pro-embodiment, sensorimotor-grounding camp's program.
  https://arxiv.org/pdf/2409.16900
- **Intelligence Requires Grounding But Not Embodiment** — arXiv, 2026. The
  *counter*-argument — read it specifically to stress-test our thesis. If grounding
  doesn't need a body, what does that mean for "give the model a nose"? Argue with
  it. https://arxiv.org/html/2601.17588v1

---

## ★ Start Here (the 5-paper on-ramp)

1. **Wiltschko POM** (Territory A) — the state of the art for "molecule → smell."
2. **Machine olfaction review** (Territory B) — how you physically get a smell in.
3. **Canine/GC-MS disease review** (Territory C) — why the channel is worth it.
4. **Harnad, Symbol Grounding** (Territory E) — why senses matter to mind at all.
5. **Osmo "about" + Series B** (Territory D) — who's ahead and what they've left open.

## Territory F — Acting On Smell / Olfactory Navigation (OUR LANE)

The sensing→agent→action lane the POSITION piece bets on. Crucially: **prior art
already exists here, and it works.** That's de-risking, not getting scooped — it
proves the loop is real and leaves the valence/grounding thesis wide open.

- **★ Emergent behaviour & neural dynamics in artificial agents tracking odour
  plumes** — Singh, van Breugel, Rao, Brunton, *Nature Machine Intelligence*, vol 5,
  58–70, 2023. **Deep-RL recurrent agents trained to find the source of a simulated
  *turbulent* odor plume — and they spontaneously developed the same casting /
  zig-zag / upwind-surge strategies real moths use.** *Why it matters:* this is
  essentially **our MVP, already demonstrated.** Smell → agent → action, learned by
  RL, no labeled odor library, behavior as the success metric. The starting line is
  drawn for us. Code: https://github.com/BruntonUWBio/plumetracknets ·
  paper: https://www.nature.com/articles/s42256-022-00599-w
- **Infotaxis** — Vergassola, Villermaux & Shraiman, *Nature*, 2007. A classic
  search strategy for finding an odor source from *sparse, intermittent* whiffs (no
  smooth gradient to follow) by moving to maximize expected information gain. *Why it
  matters:* the canonical "where do I sniff next when cues are rare" algorithm — a
  baseline/foil for an RL agent, and conceptually adjacent to the MCTS "sample to
  decide" instinct (see IDEAS.md).
- **Diffusion GNN for robust olfactory navigation in hazard robotics** — arXiv,
  2026. Recent work pushing plume-tracking toward real robots/hazard response.
  https://arxiv.org/pdf/2506.00455

## On-Ramp — Deep-Read Notes (verified 2026-06-24)

Notes pulled from reading the actual sources, not search snippets. Numbers here are
checked.

- **POM (Wiltschko, Science 2023)** — Message-passing neural net; trained on ~5,000
  molecules (GoodScents + Leffingwell) labeled against a **55-word odor lexicon**.
  Prospective test: **323 novel odorants**, panel of ≥15. The model beat the *median
  panelist* (vs the panel mean) on **53%** of molecules — i.e. roughly human-parity,
  not superhuman. Nailed counterintuitive structure→odor jumps on ~50% of discordant
  triplets. **Limits (important for us):** ignores intensity change with
  concentration; needs training examples for novel chemical motifs; chokes on
  structurally-diverse labels like "musk." So: strong on *descriptor* prediction,
  blind to *intensity/dynamics* — a real seam.
- **Machine-olfaction review (2025)** — Sensor zoo: metal-oxide, SPR, QCM, terahertz,
  plus stabilized mammalian receptors / GPCRs / insect ORCOs claiming *near
  single-molecule* resolution. **Named challenges:** models "overfit to
  cohort-specific background odors rather than disease-relevant markers" (!), sensor
  drift, and too little labeled data. Market projected **$29.8B (2025) → $76.5B
  (2032)** — money is arriving. That overfitting-to-background line is a direct
  warning for any sensing→agent loop we build.
- **Canine/GC-MS disease review (2023)** — Smell-detectable signal is real across
  hypoglycemia, epilepsy, prostate/colorectal cancer, COVID-19, malaria, Parkinson's;
  dog sensitivity e.g. **65–100%** for COVID. **But the caveat is brutal and worth
  internalizing:** "none of the studies reviewed have been replicated in other
  laboratories," no standardized sampling, not validated by health bodies. The signal
  exists; the *rigor* doesn't yet. Don't build on any single unreplicated result.
- **Harnad (grounding)** — Meaning needs **sensorimotor categorization** — learning
  to sort the world through direct causal interaction, not symbol-to-symbol relations.
  LLMs are ungrounded; their fluency is mimicry of already-grounded human text
  ("epistemic parasitism"). His bar for understanding is embodied categorization.
  *This is the strongest statement of why an input vector like smell could matter —
  and the 2026 "grounding without embodiment" paper is the rebuttal to read against
  it.*
- **Osmo (about page)** — Mission: "digitizing the sense of smell, for human health
  and happiness." The moat is **data at brute-force scale: 3B molecules mapped, 5M
  human smell annotations, 250,000 physical samples created.** *Strategic read:* they
  beat the dataset wall on the **representation/output** side by out-collecting
  everyone. A small team cannot out-sample them there — which is *exactly* why
  Petrichor should not compete on representation/reproduction. Our lane (sensing →
  agent → action, smell-as-valence) has a *different* data shape, where an agent
  grounding in its own live sensorium isn't bottlenecked on their labeled library.

## Open Questions Petrichor Could Attack (the gaps in the map)

- **Input, not output.** Osmo races to *reproduce* scent. Far fewer people are
  building the *sensing→agent* loop: a live chemical vector an autonomous agent
  ingests and *acts on*. That asymmetry is our lane.
- **Smell as valence, not label.** Everyone predicts descriptors ("floral,"
  "smoky"). Almost no one models smell as an *affective/evaluative* signal — the
  approach/recoil, the felt meaning. That's the FOUNDING thesis and it's underbuilt.
- **The dataset wall.** No internet to scrape; every datapoint needs a physical
  sample + a nose or GC-MS. Is there a clever bootstrap (self-supervision, sim,
  bio-hybrid auto-labeling) that breaks the bottleneck?
- **Geometry mismatch.** If odor space is hyperbolic, are we embedding it wrong?
- **Grounding, tested.** Does a chemical input vector measurably change what a model
  *understands* — or is it just a fancier sensor? Design the experiment that tells
  the difference. (This is the hill the whole thesis lives or dies on.)

## Players & Labs to Watch

Osmo (Wiltschko) · Sobel Lab (Weizmann, olfaction & chemosignals) · Sharpee
(geometry of odor space) · Keller/Vosshall lineage (perception datasets) · the
bioelectronic-nose groups (OR-nanodisc sensors) · Owlstone Medical (breath VOC
diagnostics) · Pyrfume (open data community).

---

*This map is provisional and definitely incomplete. That's the point — it's a map
of a frontier, not a settled field. Revise relentlessly.*
