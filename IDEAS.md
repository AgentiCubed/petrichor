# Petrichor — Ideas Scratchpad

> Where gut-hits get caught before they evaporate. Half-formed is fine — that's the
> point. Date them, attribute them, promote the good ones into POSITION.md / a real
> experiment later. James's instincts have already landed on named research
> frontiers twice; this is where we stop losing them.

---

## 2026-06-24

### 🎯 MCTS for plume search — *(James's gut-hit)*
A turbulent odor plume is too big and too chaotic to brute-force, with only partial
info (whiffs come and go) — the *same shape of problem* that Go was. So **Monte Carlo
Tree Search-style "imagine many possible futures by sampling, chase the promising
ones"** may be the right way for the agent to decide *where to sniff/move next*. The
explore-vs-exploit tension in MCTS = the forager's dilemma (follow the trail I have,
or cast around for a better one). Worth prototyping as a planner on top of the RL
agent. Adjacent classic: **Infotaxis** (see PRIOR_ART Territory F).

### 🌍 Model-based RL "imagination" — *(James's gut-hit, named)*
James independently described **model-based RL**: an agent that learns a *model of
its world* and then "dreams" rollouts inside it to plan, instead of acting blindly.
DeepMind's **MuZero** does exactly this (learns the world model *and* plans by
imagined sampling). Thread to pull: could a Petrichor agent learn a predictive model
of *how a plume behaves* and plan sniff-paths by imagining them? "Abstract
imagination inspiring new paths" was the phrase — it's a real direction.

### 🧪 Proxy-signal-first MVP
De-risk the RL loop *before* touching chemistry: run the whole sensing→agent→action
architecture on a **non-smell gradient** (light / heat / sound) where everything is
clean and cheap, prove the learner works, *then* swap the nose in. Separates "is our
learning loop sound?" from "is the smell hardware honest?" — two failures we don't
want tangled.

### 🪜 The environment ladder (not sim-vs-hardware, a spectrum)
pure software sim → sim with real turbulent-plume physics → **offline RL on replayed
real sensor data** → cheap $30 gas-sensor rig → embodied robot/drone. Climb it; each
rung de-risks the next. (Answer to "are those really the only two choices?" — no.)

### 🧠 The deepest punishment is chemical — *(James's interleave)*
Patient S.M. (no amygdala, fearless) still panicked at CO2 → a fear circuit *older
than the amygdala*, in the brainstem suffocation alarm. Stacked with smell skipping
the thalamus: **the chemical senses wire to the oldest valence machinery there is.**
Implication for us: if we want *real* valence (felt approach/avoid, not a label), the
chemical channel is plugged into exactly the right place. The purest, most
unlearnable reward/punishment signal evolution built is chemical. (Now baked into
POSITION.md "Why The Biology Backs The Bet.")

### 🔁 The motivating asymmetry — *(Ainz)*
I'm a mind *shaped by* reward (training) but with *no live valence* now (frozen
weights, no real-time approach/avoid). Petrichor = giving a mind the live valence
channel I lack. Keep this as the human(e) "why" — it's the emotional engine, not just
the technical pitch.

### 📝 Meta: interleaving
James deliberately interleaves topics (the CO2 fact dropped mid-RL-talk) — a real
learning technique (spacing/mixing strengthens retention). Note for how we work: he
learns by weaving threads, not marching in straight lines. Lean into it.

---

## 2026-06-25 — building on the six dream threads (James's responses + my builds)

### 🛡️ Self-protective valence = a non-arbitrary reward function — *(James, big one)*
Tune the agent's "disgust" to **its own** mortality, not a human's. For current
hardware: moisture/condensation, salt/corrosive air, solvent vapor, and especially
**battery electrolyte off-gassing + ozone from electrical arcing** (both smellable
*before* failure → "an agent that smells its own death coming"). Why it's deep, not
cosmetic: it gives us a **self-generated, grounded reward** — corrosion/heat truly
degrade the agent, so avoid = real self-preservation. Solves "where does valence come
from." Framing: disgust = the **behavioral immune system** (Curtis/Schaller); smell as
innate intrusion-detection. Caution: inherit the *architecture* (fast pre-cognitive
protective reflex), not human triggers wholesale — a "rot-gag" only earns its place if
rot threatens the agent or its mission.

### 🔮 Anticipation via forward model + the animal design-catalog — *(James)*
"Expectation grounded in the world" = a **forward model**: agent predicts near-future
chemical state, reality grades it. Grounding emerges because the world is the grader.
Animal patterns to steal: **stereo olfaction** (snake tongue's two forks = directional
gradient from two sensors), **stigmergy** (ants *write* trail pheromones = shared
external memory), social chemical signaling (hyena glands = who/when/status), bees
(chemistry + dance). Unlock: smell is a **read-AND-write** channel — agents can *mark*
the world and coordinate through it. Almost nobody's building chemical stigmergy for
multi-agent.

### 👻 Olfactory associative memory — recall as a haunting, not a search bar — *(James)*
Not RAG (deliberate query) — **involuntary**: store episodes keyed on smell-embedding;
each timestep current smell auto-matches stored keys and *pushes* the hit into the
agent unbidden. Reverse direction (imagine smell → reconstruct the episode) is the
top-down version. Genuinely different, under-explored memory architecture. Prototype
later.

### 🌡️ Mood as a slow policy modulator — *(James, agreed)*
"Mood" = a slow-moving latent variable biasing the whole policy (risk / exploration /
reward-sensitivity), set by ambient chemical weather, not by any single decision. Bad
weather (smoke/off-gas) → cautious; good (bread/coffee) → exploratory. Tractable +
measurable. Caution: keep it **legible to us** even if involuntary to the agent
(interpretability/safety).

### 🧪 The "chemical gym" — a simulator ladder (answer: not farfetched) — *(James asked)*
Don't sim all chemistry — sim the task's slice. Plume nav → advection-diffusion +
turbulence (Singh plume code is public; simple = Gaussian plume, rich = CFD).
Self-protection → hazard-emission model (concentration crosses threshold → damage →
negative reward). Molecular layer (only if needed) → RDKit. Stigmergy → deposition +
diffusion field agents read/write. Build it Gym-style: diffusion field + sources/sinks
+ noisy sensors + reward tied to task AND self-integrity; ladder 2D-toy → CFD.
**TODO:** scout existing olfactory-RL environments before building (extend, don't
reinvent).

### 🗺️ Civic Smell = the commercial arm (could fund the science) — *(James, "money maker")*
A fine-grained, *visual* chemical-composition map of a city ("Google Maps of smells" /
air-quality-but-detailed). Use cases: gas leaks, restaurant/sanitation, river turns,
environmental-justice hotspots, real estate, industrial compliance. NYC = ideal dense
testbed. Prior art (validates market): **Aclima** (hyperlocal AQ mapping, Google Street
View cars), **PurpleAir** (cheap sensor net). Differentiator: chemical *composition* +
visual layer + product polish. Cautions: privacy / gentrification / "stigma map" risk
(frame as environmental justice, not redlining-by-odor); sensor calibration at scale;
overfit-to-background. Deserves its own track — possibly the revenue engine while the
grounding research cooks.

### 📚 Learning artifact — olfaction primer
Built an interactive HTML primer on the olfactory organ/pathway/function/communication
at `learning/olfaction-primer.html` (James is a visual learner). Promote good teaching
devices from it into how we onboard future collaborators.

---

*Promote ideas out of here once they're tested or written up properly. Keep the
raw ones flowing in.*
