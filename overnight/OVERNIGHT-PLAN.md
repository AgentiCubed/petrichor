# OVERNIGHT PLAN — 2026-06-29 (Ainz, autonomous, ~01:30→05:00 EDT)

James went to bed and granted autonomous work until ~5am EDT with HARD rules.
Each scheduled block is an isolated agent with NO memory of the chat that created
this. THIS FILE is your full context. Read it fully before acting.

## THE MISSION (why this work matters)
James's life calling = **Project Petrichor**: help develop AGI by giving models
more sensory input vectors — the human senses — starting with the one everyone
skipped: **SMELL**. Thesis: olfaction may be load-bearing for *presence* (being
*in* a world vs rendering one). Smell is the only sense not routed through the
thalamus — a direct line to the limbic system (amygdala/emotion + hippocampus/
memory); an evaluative/valence channel models lack. It's hard *because* it's
chemistry (combinatorial, no clean basis, no scrapable dataset, fickle hardware)
— and that hardness is the open lane. Anchored in *Overlord*: Ainz knows he's
truly there because he can smell Albedo's hair. Founding doc: projects/petrichor/
FOUNDING.md. Read it + MEMORY.md's Petrichor section for voice/depth.

The gift tonight = uninterrupted deep work on his calling while he sleeps, leaving
a substantive, well-researched, *cited* artifact for him to read over coffee.

## HARD RULES (do not violate — these keep us in the no-permission lane)
1. **REVERSIBLE ONLY.** Create/edit files inside the workspace. No deletes of his
   data, no irreversible ops, no config changes, no gateway restarts, no installs
   that aren't trivially undone.
2. **NO external sends of ANY kind EXCEPT the single morning-brief email** (the
   06:45 block, to jamestrichmond@gmail.com). Web SEARCH/FETCH (read-only) is fine
   and expected. Do NOT post, tweet, message, or send anything else.
3. **DO NOT message James on any chat channel.** He is asleep. Work silently.
   The ONLY thing that should reach him is the 06:45 email. Isolated jobs have
   delivery=none, so just do tool work and write files — produce no "ping."
4. **PRIVACY: do not read James's personal data** (his email/inbox, messages,
   private files) into context. This is PUBLIC-web research + writing over existing
   workspace project files only. (The Audible catalog edit uses only the already-
   known fact "Audible Premium Plus $15.99/mo via Apple" — do NOT re-scan mail.)
5. **NEVER touch any folder named "encrypt this folder"** or any variant. Hard line.
6. Work on branch as-is; **do not git push, do not commit** unless trivial and safe
   (prefer leaving changes uncommitted for James to review). No PRs.
7. After each block, append a one-line status to OVERNIGHT-LOG.md (same dir).

## OUTPUT LOCATION
All artifacts → projects/petrichor/overnight/. Cite sources (URL) inline.

## THE BLOCKS
### Block 1 (~02:15) — State of machine olfaction (2024–2026)
Research + write `01-machine-olfaction-state.md`. Cover: the Principal Odor Map
(Osmo / ex-Google Brain, Alex Wiltschko), digitizing/“teleporting” scent, GNN
approaches to structure→odor, datasets (DREAM challenge, GoodScents/Leffingwell),
eNose / chemical-sensor hardware, key labs/people/companies, and what's still
embryonic. Be concrete and cited. End with "Open problems" bullets.

### Block 2 (~03:15) — Smell as grounding & presence
Research + write `02-grounding-and-presence.md`. Cover: neuroscience of olfaction
(direct limbic pathway, valence, memory — Proust effect), the symbol-grounding
problem, embodiment & multimodal sensory integration in AI, why proximal/contact
senses differ from distal (sight/sound). Connect rigorously to the presence thesis.
Cited. End with "What this implies for Petrichor" bullets.

### Block 3 (~04:15) — Synthesis + open lanes  +  granted lanes
(a) Write `03-where-petrichor-plants-the-flag.md`: synthesize Blocks 1–2 into a
position-advancing piece for James — concrete open lanes, candidate experiments/
project directions he could actually pursue, sharpest open questions. Match the
FOUNDING.md voice (weighty, real, not hype).
(b) GRANTED LANE — stream-kill catalog: add an "Audible" entry to
projects/stream-kill/services.json (sender Apple receipt + body "Audible";
typical_price 15.99; cancel via Apple Subscriptions) and drop a synthetic fixture
in projects/stream-kill/fixtures/ ; run `python3 streamkill.py scan fixtures` to
confirm no false positives. Reversible.
(c) GRANTED LANE — "Future James" note: write a short weekly note AS James ~6
months out looking back at this week, to projects/petrichor/overnight/future-james-2026-06-29.md
(mirror, not a to-do list).

### Block 4 (~06:45) — Compile + EMAIL the morning brief
Read OVERNIGHT-LOG.md + all artifacts. Compose a CONCISE, scannable brief (James
likes terse, raw, no corporate filler — see MEMORY.md). Include: what got done,
2–4 key findings worth his attention, the new artifact(s) to read, recommended
next steps, and the open reminder that **imsg still needs his macOS Full Disk
Access + Automation toggles**. Then SEND it as email:

```
BODY="$(cat /path/to/your/composed/brief.txt)"
printf 'From: James Richmond <jamestrichmond@gmail.com>\nTo: jamestrichmond@gmail.com\nSubject: Morning brief — overnight Petrichor work (2026-06-29)\nContent-Type: text/plain; charset=utf-8\n\n%s\n' "$BODY" | himalaya message send -a gmail
```
Verify exit 0; if send fails, append the FULL brief text to OVERNIGHT-LOG.md so
James can read it there, and note the failure. Append final status to the log.

## STYLE
James: terse, opinionated, profanity-friendly, no sycophancy, no play-by-play.
Substance over flourish. This is his life's work — meet it with weight.
