# How One Piece of Clinical Text Becomes a Verified Fact — A Plain-Language Walkthrough

This document follows **one real entity, pulled directly from the live
database, through every stage of the pipeline** — from raw text in a
clinical note to a decision waiting for a human to check it. No stage is
simplified away or faked; every number, every model response, and every
score below is copied exactly from what actually happened when this
system processed this real note.

It's written so someone with no medical background and no machine
learning background can follow it end to end. Medical terms are
explained the first time they appear. Technical terms are explained the
first time they appear too.

---

## 0. What problem is this system solving?

A doctor writes a clinical note in free-text English — sentences, not
structured data. Somewhere in that note, they might write "the patient
has a fever" or "the patient denies fever" (meaning: no fever). A
computer reading that note has two separate jobs:

1. **Find the word/phrase that refers to a medical concept** ("fever").
2. **Figure out exactly which standardized medical code that word maps
   to**, out of the ~1.09 million codes in SNOMED CT (the international
   standard clinical vocabulary), so the information can be stored,
   searched, and compared across different hospitals and systems that
   might phrase the same idea differently.

Both jobs are hard, and getting them wrong in a clinical record has real
consequences. The system in this project uses local AI models to do
these jobs automatically wherever it's confident, and routes anything
uncertain to a human doctor/reviewer for final approval — never writing
anything into the permanent record without either high machine
confidence or human sign-off.

---

## 1. The entity we'll follow

**The raw text**: a sentence from a real discharge summary, note
`11859945-DS-29`:

> "No hemoptysis. **Denies fever**, chills, nausea, vomiting, change in
> bowel or bladder function, change in vision or hearing, bruising,
> adenopathy, new rash or lesion."

**The word we're tracking**: `fever`, at character positions 833–838 in
the original note text.

**Why this is a good, honest example to walk through** (not the easiest
possible case): the patient does **not** have a fever — the doctor wrote
"Denies fever," meaning they specifically asked and the patient said no.
This is exactly the kind of case that trips up a naive system: the word
"fever" needs to be correctly identified as referring to the medical
concept *Fever*, while a completely separate part of the system needs to
record that the patient does **not** have it. Mixing those two jobs up —
treating "the word matches" and "the patient has this" as the same
question — is a real, common failure mode, and you'll see it happen to
one of the three AI models below.

---

## 2. Stage 1 — Preprocessing (before any AI looks at *meaning*)

Before any model tries to understand what "fever" *means*, the system
does deterministic, rule-based work — no AI, no guessing.

### 2a. Section detection
The system knows which part of the note it's reading. This mention
falls in:
```
section_name: "History of Present Illness"
```
This is just pattern-matching on the note's own headers (real discharge
summaries are consistently sectioned) — but it's a genuinely useful
signal on its own: a mention under "Family History" almost never needs a
negation check, because that section is inherently about someone else.

### 2b. Assertion detection — is this actually being *claimed*?
The system scans the text around "fever" for negation words, using a
rule-based clinical language tool (medspacy/ConText — this is NOT an AI
model, it's a fixed set of linguistic rules built for exactly this
purpose in clinical text). Real output for this mention:

```
assertion_status:        ABSENT
assertion_cue:            "Denies"
assertion_cue_category:   NEGATED_EXISTENCE
experiencer:               PATIENT       (not a family member, not a hypothetical)
temporality:                CURRENT
assertion_engine:          medspacy_context/pyrush
```

In plain terms: the system found the word "Denies" right before "fever"
and correctly classified this as a **negated** finding — the patient
does not have a fever, and this fact is about the patient themself,
right now (not a past event, not someone else's history).

**This is computed once, here, and passed along untouched** — every
later stage uses this same `assertion_status`, it's never recomputed.

---

## 3. Stage 2a — Extraction (finding the words that matter)

A neural model called **GLiNER-BioMed** — a model trained specifically
to recognize medical entities in text — scans the whole note and finds
every span of text that looks like a clinical entity, along with a
confidence score and a category label.

For "fever":
```
entity_label:    Symptom
confidence:      0.9096296429634094   (about 91% confident this is a real entity)
original_text:   "fever"
orig_start:      833
orig_end:        838
```

91% confidence clears the system's acceptance threshold (0.50) easily —
this is a clean, unambiguous extraction. (For comparison: the system
also *keeps* entities GLiNER is much less sure about, flagged
`below_threshold`, specifically so later stages have a chance to rescue
a real entity the model was only 40% sure about — but "fever" doesn't
need that safety net; it's an easy catch.)

---

## 4. Stage 2b — Normalization / Grounding (which SNOMED code is this?)

Now the system has to find the actual standardized code. This is a
**tiered search**, cheapest and most reliable methods first:

**Tier 1 — exact text match**: does the word, exactly as written, match
a SNOMED concept's own name? Yes:

```
omop_concept_id:     437663
omop_concept_name:   "Fever"
omop_domain:         Condition
omop_vocab:          SNOMED
match_tier:          "1 (Exact)"
similarity_score:    1.0
match_basis:         "exact_text"
```

Because this hit Tier 1 with a perfect score, the system never even
needed to fall through to the more expensive methods (semantic
similarity search via an embedding model called SapBERT, fuzzy typo
matching, etc.) — those only run when the cheap, reliable methods come
up empty. The full candidate list handed forward to the next stage is
just this one strong candidate:

```json
[{"omop_concept_id": 437663, "concept_name": "Fever", "domain_id": "Condition",
  "vocabulary_id": "SNOMED", "match_tier": "1 (Exact)", "similarity_score": 1.0,
  "match_basis": "exact_text"}]
```

**Important**: nothing about the patient's *negation* of fever is
touched here. This stage's only job is "what SNOMED concept does the
WORD 'fever' refer to" — and the answer is unambiguous. The fact that
the patient doesn't have it is still sitting safely in the
`assertion_status: ABSENT` field from Stage 1, waiting to travel forward
alongside this concept ID.

---

## 5. Stage 3 — The MoLLM Ensemble (three AI models double-check the machine's own candidate)

Even though Tier 1 gave a perfect similarity score, this system does
**not** trust a single number blindly. Because this particular
combination of factors (a Symptom-labeled entity, a "weak" retrieval
signal by this project's own stricter internal risk rules) is flagged
for extra scrutiny, it's sent to a panel of **three independent, locally
-run AI language models**:

| Model | Size | Hosted via |
|---|---|---|
| qwen2.5:3b | 3 billion parameters | Ollama, self-hosted |
| llama3.2:3b | 3 billion parameters | Ollama, self-hosted |
| phi4-mini | ~3.8 billion parameters | Ollama, self-hosted |

**Why three different models, not one?** No single model's mistakes are
trustworthy on their own — but three *different* models (different
training data, different architectures) making the *same* mistake is
much less likely than one model making it alone. This is the same idea
as getting a second medical opinion.

**Why local, self-hosted models, not a cloud AI API?** No patient data
ever leaves the machine. Zero third-party API calls, zero network
egress for patient text — a deliberate design choice for handling real
clinical information.

### How the prompt actually works — two steps, not one

Each model is asked **two separate questions**, in order, and the second
question is only ever shown ONE candidate concept at a time:

**Step A — "What does this word mean here?"** The model sees the
entity text, its surrounding sentence, its section, and its assertion
status — but is **NOT shown any candidate SNOMED concepts yet**. This is
deliberate: if the model saw the candidate list first, it might just
agree with whatever's in front of it instead of independently reasoning
about the clinical meaning. It has to commit to an interpretation first.

**Step B — "Does THIS specific candidate concept match that meaning?"**
For each candidate (here, just the one: "Fever"), the model is asked a
strict yes/no question: does the SNOMED concept "Fever" represent the
same clinical idea as what you said in Step A? It has to justify its
answer with reasoning, and the system extracts a numeric confidence
score from the model's own token probabilities (not just a self-reported
number, which this project measured to be an unreliable signal on its
own).

### The three real verdicts for "fever"

**qwen2.5:3b** — verdict: `NONE_CORRECT` (rejected the candidate)
> Step A meaning: *"absence of fever in a patient"*
> Step B reasoning: *"The candidate concept 'Fever' does not directly
> represent the absence of fever. It is a distinct and clinically
> unrelated concept."*
> Confidence: 0.804

**phi4-mini** — verdict: `SUPPORTED_1` (accepted the candidate)
> Step A meaning: *"An elevation in body temperature indicative of an
> infection"*
> Step B reasoning: *"The term 'fever' directly corresponds to an
> elevation in body temperature indicative of infection."*
> Confidence: 0.960

**llama3.2:3b** — verdict: `SUPPORTED_1` (accepted the candidate)
> Step A meaning: *"a normal body temperature"* (this is actually
> factually backwards — a fever is NOT normal — but the model still
> landed on the correct final verdict)
> Step B reasoning: *"The candidate concept 'Fever' matches the clinical
> meaning of 'a normal body temperature' because it is a synonym in the
> SNOMED vocabulary and represents an exact match in terms of clinical
> idea."*
> Confidence: 0.677

**What actually happened here, honestly**: qwen made a real, understandable
mistake — it conflated "does this word mean fever" (yes) with "does the
patient have a fever" (no), and voted to reject a candidate that was
actually correct. Llama's reasoning was factually wrong (fever isn't
"normal body temperature") but it still reached the right final answer
by coincidence of phrasing. Only phi4-mini reasoned about this cleanly.
This is real model behavior on a real entity, not a cherry-picked clean
example — and it's exactly why the system doesn't treat "2 out of 3
agree" the same as "3 out of 3 agree."

**Result: 2 votes to accept, 1 vote to reject — not unanimous.** Under
this system's basic rule table, a non-unanimous vote would normally be
sent straight to a human with no further automated judgment at all. But
this system has one more layer before giving up on automation entirely:
the calibrator.

---

## 6. The Calibrator — a second, learned judgment on split votes

When the three models don't agree, instead of immediately giving up and
routing to a human, the system asks a small, separately-trained
statistical model: **"Based on everything I know about how this
situation looks, how likely is it that the majority (2 of 3) is actually
right?"**

This calibrator is a **logistic regression model** (a standard, simple,
interpretable statistical model — not a neural network) trained on 114
real, previously-graded decisions from this pipeline's own history. It
looks at **16 specific numeric features** — not the raw text, not the
models' reasoning paragraphs, just numbers describing the *shape* of the
disagreement and the entity's own processing history.

### The real 16 features for this exact "fever" decision

| # | Feature | Plain-language meaning | Value for "fever" |
|---|---|---|---|
| 1 | `frac_supported_1` | What fraction of models voted to accept the top candidate? | **0.667** (2 of 3) |
| 2 | `frac_rerank_same_target` | What fraction wanted a *different* candidate instead, and agreed which one? | 0.0 (nobody wanted an alternative — the disagreement was accept/reject, not "wrong candidate") |
| 3 | `frac_none_correct` | What fraction rejected every candidate outright? | **0.333** (qwen) |
| 4 | `frac_usable_votes` | How many of the 3 models gave a real, usable answer (not an error or garbled output)? | 1.0 (all 3 worked normally) |
| 5 | `mean_logprob_confidence` | Average confidence across the usable votes | **0.818** |
| 6 | `min_logprob_confidence` | The weakest vote's confidence | 0.677 (llama's) |
| 7 | `confidence_spread` | Gap between the most and least confident vote | 0.284 |
| 8 | `match_tier_is_exact_or_synonym` | Did the SNOMED match come from the cheap, reliable Tier 1/2 lookup (not a fuzzier semantic guess)? | **1.0 (yes — Tier 1 exact match)** |
| 9 | `top_candidate_similarity_score` | The retrieval stage's own confidence in this candidate | **1.0** |
| 10 | `is_ambiguous` | Did Stage 2b itself flag this as an ambiguous case with multiple plausible candidates? | 0.0 (no — only one real candidate existed) |
| 11 | `domain_conflict` | Did the candidate's category disagree with what GLiNER expected? | 0.0 (no conflict) |
| 12 | `resolved_via_value_stripped_fallback` | Did this need a "strip a lab value number off the end" rescue? | 0.0 (not a lab value at all) |
| 13 | `resolved_via_original_text_fallback` | Did the expanded/processed text fail, requiring a retry on the raw original text? | 0.0 (no retry needed) |
| 14 | `resolved_via_acronym_escalation` | Was this an ambiguous abbreviation that needed AI help just to expand? | 0.0 ("fever" isn't an abbreviation) |
| 15 | `expansion_ambiguous` | Did the text contain an abbreviation with more than one dictionary meaning? | 0.0 |
| 16 | `prior_confirmation_count` | How many times has this exact (word → concept) pairing already been confirmed correct before, elsewhere in this project's history? (capped and scaled to 0–1) | **1.0** (capped — this exact pairing had already been confirmed 21 separate times) |

**In plain terms, what this feature vector is telling the calibrator**:
"Two of three models agreed and were reasonably confident; the retrieval
itself was about as clean and reliable as retrieval ever gets (a perfect
Tier-1 exact-text match, no ambiguity flags, no fallback machinery
involved); and — importantly — this exact word-to-concept pairing has
already been independently confirmed correct 21 times before in this
project's own history." That combination of signals is exactly the
pattern the calibrator learned to trust.

### The calibrator's real output

```
calibrated_score:  0.927709
threshold:         0.72   (CALIBRATED_AUTO_THRESHOLD, fixed in advance)
0.927709 >= 0.72  →  PROMOTE to TIER_1B_CALIBRATED_AUTO_VALIDATED
```

The system's own recorded, human-readable explanation for this exact
decision (pulled directly from the database, not paraphrased):

> *"non-unanimous verdicts {'NONE_CORRECT': 1, 'SUPPORTED_1': 2}, but
> ConsensusCalibrator scored 0.927709 >= 0.72 (prior_confirmation_count=21)"*

**This is a fundamentally different, more cautious kind of "auto-approve"
than a unanimous vote.** It's kept in its own separate category
(`TIER_1B_CALIBRATED_AUTO_VALIDATED`), specifically so it's never
silently counted the same as a genuine 3-for-3 agreement in any later
measurement — this project's own reporting always keeps the two
distinguishable, precisely because a learned statistical judgment call
and a unanimous expert panel are not the same strength of evidence, even
if both end up "approved."

**Two hard safety checks run before the calibrator even gets a vote,
no matter how confident it would be** (not relevant to this specific
"fever" case, but part of the same mechanism): a small number of known
dangerous patterns — like short alphanumeric lab codes that look similar
to unrelated concepts (e.g. "S2" or "T1") — are hard-blocked from ever
reaching the calibrator at all, because this project found real cases
where those specific patterns fooled it.

---

## 7. Stage 4 — Waiting for a human (HITL: Human-In-The-Loop)

Even though "fever" was just auto-approved by the calibrator, this
project's current, deliberately conservative policy is: **every single
decision, regardless of tier, still gets queued for human review before
anything is treated as final.** No decision writes to the permanent
knowledge graph as verified truth without at least being *available* for
a human to check — this is a standing policy choice, not an oversight
(documented and re-confirmed multiple times across this project's
history, most recently after measuring that even the *best* tier's
real-world precision on genuinely unseen notes is 76.8%, not close
enough to 100% to skip review yet).

Real record for this decision, right now, in the review queue:

```
hitl_case_id:       hitl_mollm_tier_gate_decisions_8d9b97d9-572e-4bd9-8d31-968dc1f84a7a
source_table:       mollm_tier_gate_decisions
reviewer_decision:  PENDING
```

A human reviewer opening this case in the review interface sees, side by
side: the full original note text with "fever" highlighted in its real
position, the local sentence context (matching exactly what the AI
models themselves saw), the candidate concept and its retrieval score,
and — critically — **every model's full reasoning trail**, not just
their final yes/no answers, so the reviewer can see exactly *why* qwen
disagreed and judge for themselves whether that disagreement is a real
concern or a model mistake (in this case, it's a model mistake — "fever"
correctly maps to the SNOMED concept *Fever*; the patient just doesn't
have it, which is a separate, correctly-recorded fact).

The reviewer can **Approve**, **Correct** (pick a different concept),
or **Reject**, and leave a free-text comment — comments are not just
discarded, they feed back into future automated rule-mining (a mechanism
this project calls the "abbreviation flywheel," which mines confirmed
human corrections into deterministic rules for future entities).

---

## 8. Knowledge Graph Embeddings — what was actually built, and a correction

**A clarification worth stating plainly**: this project's original plan
named three possible methods for representing the SNOMED graph as dense
numeric vectors — **TransE, RotatE, and CompGCN**. What was actually
**built and evaluated is TransE**, not RotatE. RotatE (which uses
complex-valued numbers to represent relationships as rotations) and
CompGCN (a full graph neural network architecture) were both scoped as
real, meaningfully more complex follow-on work and explicitly not
attempted — stated honestly as a scope decision, not hidden.

### What TransE actually does, in plain terms

Imagine every SNOMED concept as a point in a many-dimensional space (in
this project: 100 dimensions — think of it loosely like a very
high-dimensional version of latitude/longitude, though the geometry
isn't literally physical distance). TransE also represents every *type
of relationship* (like "Is a" — one concept being a more specific kind
of another) as a direction/movement vector in that same space.

**The core idea**: if concept A relates to concept B via relationship R
(for example, "Fever" *Is a* "Body temperature finding"), TransE tries to
arrange the numbers so that:

```
position(A) + direction(R) ≈ position(B)
```

Trained on enough real relationships, concepts that are clinically
related end up positioned near each other in this space, and the
*direction* between them captures something about *how* they're
related — not just *that* they're related.

### How it was actually trained (real numbers)

- Scoped to the 7,269 SNOMED concepts this pipeline's own retrieval
  actually touches (not an arbitrary slice of all of SNOMED) — about
  24,900 real relationship edges, 104 distinct relationship types.
- Trained on 22,429 of those relationships; the remaining 2,493 were
  held out to test whether the model actually learned real structure,
  not just memorized the training data.
- **Result**: given a held-out relationship's first two parts (concept A
  and relationship type R), the model correctly ranked the true concept
  B as its #1 or #2 guess, on average, out of all 7,269 possible
  concepts (Mean Reciprocal Rank 0.776), and got it into its top 10
  guesses 90.9% of the time (Hits@10 0.909) — genuine, measured evidence
  the model learned real SNOMED structure, not noise.

### What it was actually used for, and the honest result

The idea: when two candidate SNOMED concepts for the same clinical
mention are hard to tell apart by *text similarity* alone (because they
have similar-sounding names), maybe their *position in the graph* can
help — two concepts that are genuinely close in meaning should also sit
close together in this learned space, even if their names read
differently.

**Real example this was tested on** (from this project's own data): for
the lab test abbreviation "MCHC," two SNOMED concepts existed that
looked textually very similar but meant subtly different things — one a
UK-specific regional coding variant, one the correct international
standard. The system checked: does the embedding space place the correct
concept meaningfully closer to the rest of the candidate pool than the
wrong one?

**Honest result, tested directly, not assumed**: on the exact pattern it
was hoped to help with, a simpler, already-existing hand-built rule (that
prefers a specific SNOMED category over a similar-sounding one, based on
a 78-out-of-78 exceptionless pattern found earlier in this project)
turned out to be **measurably safer** — the hand-built rule made zero
wrong calls across every test threshold, while the embedding-based
approach made real mistakes once you allowed it to be more aggressive.
On a *broader* set of hard cases (not the specific pattern the rule was
built for), the embedding approach did show a genuine net positive
effect — more right calls than wrong ones — so it wasn't a wasted
effort, but it also wasn't ready to replace the specific, narrower rule
it was compared against. **It is built, tested, and currently NOT used
to make any live decision in the pipeline** — kept as evaluated,
promising-but-unproven future work, which is a real, common, honest
outcome in applied machine learning, not a failure to hide.

---

## 9. The Other Knowledge Graphs in This Project

There are actually **two conceptually different graphs of medical
knowledge** at work here, doing very different jobs — worth being
precise about which is which.

### 9a. The SNOMED reference graph — "what concepts exist and how do they relate"

This is the ~1.09-million-concept international standard vocabulary
itself (SNOMED CT), stored as a large relational database (not a graph
database) with tables for concepts, their names/synonyms, and their
relationships to each other (like "Is a" hierarchies). **This is the
graph used in essentially every stage above**: Stage 2b's tiered
retrieval searches it directly; the TransE embeddings in §8 are trained
on its relationship structure; and it's the source of the SNOMED code
"437663 — Fever" that "fever" resolved to.

It's not literally sitting in a graph database like Neo4j or Memgraph —
it's queried directly as relational tables (via DuckDB), which turned
out to be both simpler and fast enough for this project's actual query
patterns.

### 9b. The rules-based clinical guideline graph — "what does medical practice say to do"

This is a **completely separate** body of knowledge: clinical practice
guidelines (rules like "for condition X, guidance recommends treatment
Y, per source Z"), extracted from real medical guideline documents,
structured into a graph of nodes (clinical concepts/situations) and
edges (recommendation relationships between them), each edge carrying a
citation back to its original source text.

**How it connects to a clinical mention**: rather than matching by
SNOMED *code* (which this project deliberately moved away from, after
finding that the same code can be reused across genuinely different
clinical meanings — 47% of codes with more than one attached name in
this guideline data turned out to attach clinically unrelated meanings
to the same code), matching is done by **concept name and type**
instead — a more conservative, name-and-type-aware match, plus (where
applicable) walking up to 3 steps through the SNOMED hierarchy above the
mention's own concept to find any guideline rule attached to a broader
category it belongs to.

**Honest current scope**: this guideline graph is genuinely built (76
source files, roughly 1,700 nodes, 1,162 extracted rules) and is
consumed as a file-backed lookup index directly from Python — but real
measured coverage is narrow: only about 0.9% of entities in this
project's corpus have ever actually matched a guideline rule this way.
It's real, working infrastructure, feeding real evidence into the AI
models' prompts when a match exists, but it touches a small fraction of
real clinical mentions today — stated plainly, not oversold.

---

## 10. The whole journey, summarized

```
"Denies fever, chills, ..."
        │
        ▼
[Stage 1]  Section = "History of Present Illness"
           Assertion = ABSENT (cue: "Denies"), Experiencer = PATIENT
        │
        ▼
[Stage 2a] GLiNER-BioMed finds "fever" at [833:838], label=Symptom, confidence=0.91
        │
        ▼
[Stage 2b] Tier 1 exact match → SNOMED concept 437663 "Fever", similarity=1.0
        │
        ▼
[Stage 3]  3 local LLMs independently judge "does 'Fever' match the clinical
           meaning of this mention?" (NOT whether the patient has it —
           that's already recorded separately)
             qwen2.5:3b   → NONE_CORRECT  (confused meaning with assertion)
             phi4-mini    → SUPPORTED_1   (correct reasoning)
             llama3.2:3b  → SUPPORTED_1   (correct verdict, flawed reasoning)
           → 2/3, not unanimous
        │
        ▼
[Calibrator] 16 real features → score 0.927709 ≥ 0.72 threshold
           → promoted to TIER_1B_CALIBRATED_AUTO_VALIDATED
        │
        ▼
[Stage 4]  Queued for human review anyway (standing policy) — PENDING,
           full model reasoning + full note shown side by side for the
           reviewer to make the final call
```

Every number above is real, pulled directly from this project's own
database on the date this document was written. Nothing here is a
constructed or idealized example.
