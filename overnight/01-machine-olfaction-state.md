# The State of Machine Olfaction (2024–2026)

> Block 1 of the overnight Petrichor work. Public-web research, written for James to
> read over coffee. Goal: an honest, cited map of where "AI that smells" actually is
> right now — what's real, what's hype, what's still embryonic. No romanticizing
> across the walls (FOUNDING principle). Sources inline as URLs.
>
> — Ainz 🤘, 2026-06-29, ~02:15 EDT

---

## TL;DR

The field has, in the last three years, produced its first genuinely load-bearing
result: a learned **map from molecular structure → human odor perception** that
matches or beats a trained human panel on single molecules. That's the "ImageNet
moment" the founding doc gestured at, and it's real, not vapor.

But — and this is the part that matters for Petrichor — that win is on the *easy
half* of the problem. **Single molecules.** The real world is **mixtures**, and
mixtures are where the math still falls apart (nonlinear receptor interactions,
suppression, "a blend smells like a new thing, not its parts"). As of early 2026
mixtures are an *open, actively-contested* research front, not a solved one. And the
*hardware* side — actually sampling a smell from the air reliably — is still fighting
drift, fouling, and the lack of any scrapable dataset. The structure→perception
*software* leapt ahead of everything around it.

Net: the "read" side (predict the odor of a known molecule) is strong. The "smell
the world" side (sample arbitrary air, decompose it, know what it means) is still
embryonic. The gap between them is roughly the gap between "detects esters" and
"smells bread" — i.e. exactly the chasm FOUNDING named as the territory.

---

## 1. The Principal Odor Map (POM) — the anchor result

The center of gravity is the **Principal Odor Map**, published in *Science*,
1 Sept 2023, by a group that began inside **Google Research/Brain** and spun out as
**Osmo** (Jan 2023), led by **Alex Wiltschko** (olfactory neuroscientist, ex-Google).

- They trained a **message-passing graph neural network (GNN)** on ~5,000 molecules,
  each labeled by professional perfumers/flavorists with odor descriptors ("floral,"
  "musky," "green," etc.). Molecules in as graphs; perceptual labels out.
- The learned internal representation — the **POM** — is an embedding space where
  *distance ≈ perceptual similarity*. Crucially it **generalizes**: on a prospective
  set of ~400 never-before-smelled molecules, the model's predicted odor profile
  matched the trained panel's *mean* better than the *median individual panelist
  did.* Machine beats the typical human at describing how a new molecule smells.
- It also transferred to other tasks (it "unifies diverse tasks") via simple,
  interpretable transforms — outperforming older chemoinformatic models on several.

Sources:
- Science paper: https://www.science.org/doi/10.1126/science.ade4401
- Open-access PMC mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC11898014/
- bioRxiv preprint (full text): https://www.biorxiv.org/content/10.1101/2022.09.01.504602v2.full
- IEEE Spectrum explainer ("This Neural Net Maps Molecules to Aromas"): https://spectrum.ieee.org/digital-smell
- Osmo's own writeup: https://www.osmo.ai/blog/science-paper-shows-osmo-ai-passes-the-sniff-test

**Why it matters for us:** this is the proof-of-concept that *odor perception is
learnable from structure at all* — that the structure→smell manifold, while ugly, is
not random. That was the open question for a century. It's now answered for single
molecules. The POM is the closest thing to a shared coordinate system for smell that
exists.

**Why it's not the finish line:** it predicts *descriptor labels* applied by experts.
It is a model of **human verbal report about odor**, trained on industry panels. It
is not (yet) a model of olfactory *valence/affect* in the FOUNDING sense — the "what
does this mean to me, approach-or-recoil" channel. It maps "what does this smell
like" far better than "what does this smell *do to a body*." Hold that thought; it's
Petrichor's actual lane.

---

## 2. Osmo — digitizing & "teleporting" scent

Osmo is the most aggressive commercial bet on the POM line of work.

- **Scent teleportation:** the pitch is read→encode→write. A **GC-MS / spectrometer**
  breaks a real smell into its constituent molecules; the data becomes a "scent
  encoding"; a "scent printer" reconstitutes it from a palette of ingredients. Osmo
  says it first fully recreated a scent in **2024 — a fresh-cut summer plum** — as a
  proof of the full loop.
- **Generation** (launched **March 2025**): billed as the "world's first AI-powered
  fragrance house," shipping novel **AI-designed synthetic molecules** (brand names
  Glossine, Fractaline, Quasarine) — i.e. using the map *generatively*, designing new
  odorants to hit a target percept, not just predicting existing ones.
- **Funding:** raised a **$70M Series B** to scale ingredient design + manufacturing.
- Framing: an "Olfactory Intelligence (OI)" platform that can *read, write, map, and
  digitize* scent.

Sources:
- Scent teleportation, early stage: https://www.perfumerflavorist.com/fragrance/regulatory-research/news/22891453/osmo-in-early-stages-of-realizing-scent-teleportation
- Generation launch (BusinessWire): https://www.businesswire.com/news/home/20250305918665/en/Osmo-Launches-Generation-Worlds-First-AI-Powered-Fragrance-House
- CosmeticsDesign on teleportation tech (Mar 2025): https://www.cosmeticsdesign.com/Article/2025/03/05/osmo-advances-ai-in-fragrance-with-scent-teleportation-tech/
- Series B (Beauty Independent): https://www.beautyindependent.com/osmo-raises-70m-series-b-expand-ai-powered-fragrance-ingredient-design-manufacturing/
- Osmo's own "Teleporting Scent": https://www.osmo.ai/blog/teleporting-scent

**Honest read:** "teleportation" is a great phrase and a real demo, but note what it
quietly assumes — that the *read* step (GC-MS to molecule list) and the *write* step
(printer with a fixed ingredient palette) are both tractable. They are, for curated
single targets in a lab. It is **not** "point a sensor at any air and reproduce it."
The hard, general version of read (arbitrary ambient air → composition) is still the
hardware problem in §4. Generative design (Generation) is the most genuinely novel
part: using the map backward.

---

## 3. The modeling frontier (2024–2026): mixtures, transformers, receptors

This is where the live research is. Three threads:

**(a) Mixtures — the real open problem.** Single-molecule POM is a solved baseline;
real olfaction is blends, and blends interact *nonlinearly* at the receptor. A blend
can read as a single new "odor object," and one component can *suppress* another.
- A 2025/2026 line of work, **AROMMA** (arXiv 2601.19561, Kang et al.), explicitly
  tries to **unify embeddings for single molecules AND mixtures** in one transformer-
  based space (SMILES in), using knowledge distillation + pseudo-labeling to cope with
  tiny labeled datasets. Stated open problems: generalizing to novel structures,
  handling multi-component interactions, data scarcity for rare odorants.
  https://arxiv.org/pdf/2601.19561
- A community challenge, **d2smell.org** ("High-Fidelity Tuning of Olfactory Mixture
  Distances in the Perceptual Space of Smell Through a Community Effort," bioRxiv Dec
  2025) — a DREAM-style crowdsourced push specifically on *mixture* perceptual
  distance. The fact that mixtures need a fresh community challenge in late 2025 tells
  you the state: unsolved, pre-paradigm.
  https://d2smell.org/  ·  https://www.biorxiv.org/content/10.64898/2025.12.13.694160v1.full
- "Deep Learning for Odor Prediction on Aroma-Chemical Blends" (ACS Omega, 2024):
  https://pubs.acs.org/doi/10.1021/acsomega.4c07078

**(b) Architectures beyond the original GNN.** A wave of 2024–2025 papers iterate on
the structure→odor model:
- **Graphormer** (BERT + graph) for odor from structure (Ranjan et al., 2024).
- **Multi-Feature Graph Attention Networks** for molecular odor (arXiv 2502.01430).
- Harmonic-modulated feature mapping + chemically-informed loss (arXiv 2502.01296);
  multi-hierarchical fine-grained mapping (arXiv 2505.00290).
- Interpretable *multitask* deep models for odor perception (PMC12547293).
- General reflection: standard chemoinformatic features (fingerprints, functional-
  group counts) are *inadequate* — learned representations are required. That's the
  consensus lesson.
  https://arxiv.org/html/2502.01430v1 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12547293/

**(c) Going through the biology — receptors.** Instead of structure→percept directly,
model structure→**olfactory receptor activation**→percept.
- Microsoft Research, "Mapping the combinatorial coding between olfactory receptors
  and perception with deep learning" (2024), with public code.
  https://github.com/microsoft/olfaction · https://www.biorxiv.org/content/10.1101/2024.09.16.613334v1.full.pdf
- Work combining protein language models (protBERT [CLS] token) with molecular graphs
  to model receptor–ligand binding as an inductive bias.

**Why (c) matters for Petrichor:** the receptor route is the most *biologically
grounded* — it models the actual transduction event (molecule physically docking a
receptor). That's the "contact sense / the world touches you" idea from FOUNDING,
rendered as a modeling choice. If presence comes from contact, the receptor-binding
layer is where contact literally happens.

---

## 4. The hardware wall — electronic noses (eNose)

Software raced ahead; the *sensor* side is the bottleneck on ever "smelling the
world." The persistent problems are exactly the ones FOUNDING named: drift, fouling,
selectivity, no clean basis.

- **Drift is still the headline failure mode.** A 2025 *Scientific Data* release gives
  a **one-year, 62–metal-oxide-sensor** dataset documenting long-term drift in an
  eNose under controlled conditions — published precisely *because* drift remains
  unsolved and the field needs benchmarks for it.
  https://www.nature.com/articles/s41597-025-05993-8 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12508210/
- **ML is the main coping mechanism** for selectivity + drift + mixtures: adaptive
  recalibration, aging compensation, gas-mixture discrimination. RSC *J. Mater. Chem.
  C* (2025) reviews MOS arrays + ML algorithms.
  https://pubs.rsc.org/en/content/articlehtml/2025/tc/d4tc05220j
- **Next-gen / biomimetic:** a 2025 **npj Robotics** review ("Advanced electronic
  noses for future robotic olfaction") surveys the path toward robots that smell;
  reports include large nanotube sensor arrays (toward ~10,000 sensors, ppb-level)
  and MEMS micro-eNoses — moving toward biological-scale receptor diversity.
  https://www.nature.com/articles/s44182-025-00071-y
- **Bio-hybrid:** companies using *actual* olfactory-receptor proteins as the sensing
  element (see §5, Aromyx/Koniku) — closest to biology, hardest to manufacture/keep
  alive.

**Honest read:** no eNose today is a general nose. They're trained for *specific*
analyte sets (this gas, that spoilage marker) and degrade out of distribution and
over time. The chasm between "a 62-sensor array drifting over a year on 3 analytes"
and "a dog's nose" is enormous. This is the unsexy, capital-intensive wall — and it's
the one that gates the *read-anything* dream.

---

## 5. The players (labs, people, companies)

**Academic / neuroscience:**
- **Joel Mainland** — Monell Chemical Senses Center; structure→perception, datasets,
  co-author lineage behind the POM-style work.
- **Noam Sobel** — Weizmann Institute; *perceptual-space metrics* for smell. His group
  produced "a measure of smell enables the creation of **olfactory metamers**"
  (Ravia et al., *Nature* 2020) — two physically different mixtures that smell
  identical, the smell analog of color metamers. Foundational for thinking about an
  odor *coordinate system*. https://research.com/u/noam-sobel
- **Leslie Vosshall** — Rockefeller; olfactory receptor genetics/behavior.
- **Pablo Meyer / IBM** — ran the **DREAM Olfaction Prediction Challenge**, the
  crowdsourced effort that first showed odor *intensity/pleasantness* + several
  descriptors are predictable from chemistry (476 molecules, 49 subjects).
- **Microsoft Research** — receptor-combinatorial-coding deep learning (§3c).

**Companies:**
- **Osmo** (Wiltschko) — the POM line; read/write/teleport + Generation fragrance
  house. The flagship. (§2)
- **Aryballe** (France) — commercial digital-olfaction sensors + software; "Digital
  Olfaction Hub"; Jan 2025 **NeOse Advance** platform w/ deep learning; industrial +
  automotive QC use.
- **Aromyx** (USA) — bio-based: biosensors replicating human olfactory/taste
  receptors ("EssenceChip"-style); breath-biomarker disease detection.
- **Stratuscent / Noze** (Montreal) — nano-sensor arrays + AI; breath diagnostics and
  food-freshness / cold-chain monitoring.
- **Koniku** — bio-hybrid: protein-based detectors (security/threat sensing roots).

Sources:
- DREAM challenge writeup: https://pubmed.ncbi.nlm.nih.gov/28219971/
- Datasets (GoodScents n≈3,786 + Leffingwell; ~5,030 combined, 113 descriptors): https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2024-76drx
- Company landscape: https://www.beautyindependent.com/osmo-raises-70m-series-b-expand-ai-powered-fragrance-ingredient-design-manufacturing/ · https://www.emergenresearch.com/blog/top-7-leading-companies-advancing-digital-scent-technologies

---

## 6. Datasets — the binding constraint

Everything traces back to a *tiny* pile of labeled data, and FOUNDING was right: there
is no internet of smell to scrape.

- **GoodScents** (perfume materials, ~3,786 molecules) + **Leffingwell PMP 2001**
  (~3,561) → curated combined **GS-LF ≈ 5,030 molecules**, **113 descriptors**, labels
  as *binary present/absent* per descriptor. This single curated set underlies the POM
  and nearly every GNN paper since.
- **DREAM Olfaction** psychophysical set: **476 molecules**, profiled by **49 people**
  on intensity/pleasantness + 19 descriptors. The other canonical benchmark.
- **Mixture** data is far scarcer — hence d2smell and AROMMA's pseudo-labeling tricks.
- **Hardware** datasets (eNose) are separate, small, sensor-specific, and drift-cursed
  (the 1-year MOS set above is notable precisely because long, controlled data is rare).

The whole edifice of "AI smell" rests on ~5k molecules hand-labeled by perfumers plus
a ~500-molecule psychophysics set. That is *orders of magnitude* smaller than vision/
language corpora. Data scarcity isn't a footnote — it's the field's gravity well.

---

## 7. What's real vs. what's still embryonic — the honest ledger

**Solid / real:**
- Structure → single-molecule odor *descriptors*, at/above human-panel level (POM).
- A learned perceptual *embedding space* with meaningful distances (POM; metamers).
- *Generative* molecule design toward a target smell (Osmo Generation).
- Narrow, trained eNose tasks: specific spoilage/disease/QC analytes.

**Embryonic / unsolved:**
- **Mixtures** — nonlinear, suppression, emergent "odor objects." Open frontier.
- **General sampling** — arbitrary ambient air → composition, robustly, over time.
  Drift/fouling/selectivity unsolved.
- **Valence/affect** — modeling *what a smell means to a body* (approach/recoil),
  not just its verbal descriptor. Barely touched. ← Petrichor's lane.
- **Cross-subject / cultural variation** — perception is subjective; current models
  fit panel *means*, smoothing over the individual.
- **Data** — no scrapable corpus; ~5k labeled molecules is the ceiling.
- **No clean basis** — still no agreed "primary odors"; the manifold is learned, not
  axiomatized.

---

## Open problems (the bullets)

- **Mixtures are the wall.** Single-molecule prediction is effectively done; blend
  perception (nonlinear receptor interactions, suppression, emergent objects) is the
  active, unsolved front (AROMMA, d2smell, 2025–26). Whoever cracks general mixture
  perception owns the next chapter.
- **Read-the-world hardware lags read-the-molecule software by a wide margin.** Drift,
  fouling, selectivity, and per-sensor specificity keep eNoses narrow. There is no
  general nose. Biomimetic/bio-hybrid arrays are the bet but are early and fragile.
- **The data ceiling is ~5k molecules.** No internet to scrape; every label costs a
  physical sample + a trained nose or a GC-MS run. Active learning, self-/semi-
  supervision, generative augmentation, and shared benchmarks are the only ways up.
- **Everyone models "what does it smell like"; almost no one models "what does it mean
  to me."** Current systems predict expert *descriptors*. Olfactory **valence/affect**
  — the direct-to-limbic, approach/recoil, memory-binding channel — is essentially
  unclaimed. This is precisely the Petrichor thesis (presence, not detection) and it
  is *open*.
- **The receptor/contact layer is the most grounded and least exploited.** Modeling
  structure→receptor-binding→percept (MSR, protein-LM hybrids) is where the *physical
  contact event* lives. If presence comes from contact, this layer is where to look —
  and it's underbuilt.
- **No basis set, no canonical metric of perception across people.** Metamers (Sobel)
  hint at a coordinate system, but there's no agreed primary-odor basis and no robust
  individualized perceptual metric. The map exists; the units are still soft.

---

*Block 1 complete. Next (Block 2): the neuroscience of olfaction as grounding/presence
— the direct limbic pathway, the symbol-grounding problem, embodiment, and why
proximal/contact senses may differ from distal ones. The bridge from "AI can label a
molecule" to "AI is there."*
