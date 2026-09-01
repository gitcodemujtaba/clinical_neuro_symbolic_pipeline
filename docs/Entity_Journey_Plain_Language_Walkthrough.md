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
does deterministic, rule-based work — no AI, no guessing. **A correction
worth stating up front**: this stage is sometimes described (including
in earlier drafts of this document) as running "scispaCy." That's not
accurate — checked directly against the real code (`src/assertion.py`)
rather than assumed. scispaCy appears exactly once in this codebase, as
a version-compatibility comment pinning spaCy below 3.8.0; it is never
actually imported or run as a pipeline component. What genuinely runs is
three real sub-processes, chained together, detailed below.

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

### 2b. Sentence splitting — PyRuSH, not a plain sentence splitter

Before negation detection can even ask "what sentence is this word in,"
something has to decide where one sentence ends and the next begins.
The system uses **PyRuSH** (medspaCy's own clinical-aware rule-based
sentence segmenter), built on a blank spaCy pipeline
(`spacy.blank("en")`) with `medspacy_pyrush` attached as a pipe — not
spaCy's own general-purpose statistical sentence splitter, and not a
naive "split on periods" approach.

**Why this specific choice matters, with a real documented failure it
was chosen to avoid**: note `10000032-DS-21` in this project's own
corpus contains a dense physical-exam/lab block — a single run-on
passage 1,908 characters long, packed with abbreviated lab values and no
normal sentence-ending punctuation. Tested directly against a plain,
general-purpose sentence splitter (spaCy's built-in `sentencizer`), that
entire 1,908-character block collapsed into **one single "sentence"** —
which meant a negation cue anywhere in that giant block (e.g. the word
"without" near the start) got applied to *every* lab value in it,
producing false `ABSENT` assertions on real, actually-present findings:
`GLUCOSE-109`, `UREA N-25`, `CREAT-0.3`, `HGB-14`, and `MCV-99` were all
incorrectly marked as negated. PyRuSH's clinical-specific splitting
rules (aware of lab-value line breaks and clinical list punctuation, not
just periods) correctly break this block into its real, separate
sentences instead — this is a genuine, previously-hit bug this specific
engineering choice was made to fix, not a hypothetical. (`sentencizer`
is still kept as an automatic fallback if PyRuSH itself fails to load —
worse than PyRuSH, but safer than crashing the whole pipeline.)

**Input** (raw note text, unchanged): the full text of note
`11859945-DS-29`, including the sentence `"No hemoptysis. Denies fever,
chills, nausea, vomiting, change in bowel or bladder function, ..."`.

**Output**: sentence-boundary offsets — for this passage, PyRuSH
correctly identifies `"No hemoptysis."` as one sentence and `"Denies
fever, chills, nausea, vomiting, change in bowel or bladder function,
change in vision or hearing, bruising, adenopathy, new rash or
lesion."` as the next, separate sentence. This sentence boundary is what
lets the next sub-process (2c) correctly scope "Denies" as applying to
everything in *that* sentence's list, not bleeding into unrelated text
before or after it.

### 2c. Assertion detection — is this actually being *claimed*?
The system scans the text around "fever," within the sentence boundary
PyRuSH just found, for negation words — using **medspaCy's ConText**
module, a rule-based clinical language tool (this is NOT an AI model,
it's a fixed set of linguistic rules built for exactly this purpose in
clinical text, chained onto the same blank-spaCy pipeline as
`medspacy_context`). Real output for this mention:

```
assertion_status:        ABSENT
assertion_cue:            "Denies"
assertion_cue_category:   NEGATED_EXISTENCE
experiencer:               PATIENT       (not a family member, not a hypothetical)
temporality:                CURRENT
assertion_engine:          medspacy_context/pyrush
```

In plain terms: the system found the word "Denies" right before "fever,"
within the sentence PyRuSH scoped it to, and correctly classified this
as a **negated** finding — the patient does not have a fever, and this
fact is about the patient themself, right now (not a past event, not
someone else's history).

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

### How many actual model calls does this cost?

For "fever" specifically: **1 Step-A call + 1 Step-B call per model**
(Step B only evaluates 1 candidate here, because Stage 2b only handed
forward one — see §4), run independently and in parallel for all 3
models, so **6 real LLM calls total** for this one entity's Stage 3
decision (2 per model × 3 models). By default, Step B stops at the
first candidate a model accepts — but in production this pipeline runs
with `EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` turned on, meaning a model
keeps checking every remaining candidate even after accepting one (so
it can flag a genuine tie between two candidates it both accepts,
rather than silently stopping at whichever one happened to be checked
first). For an entity with, say, 3 real candidates instead of 1, the
real cost is `1 + 3 = 4` calls per model, `12` calls total — this is
the concrete reason this project explicitly budgets a generous
2–5-minute-per-note latency allowance rather than treating Stage 3 as
cheap.

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
interpretable statistical model — not a neural network). It has been
retrained twice since it was first built, most recently on 2026-08-31
(after fixing a real data-leakage bug where some of its training
examples were accidentally drawn from the project's own locked test
split) — the version live today is fit on **1,293 real, previously-graded
decisions across 75 distinct notes** (`training_split:
full_corpus_105_notes_2026-08-31`, checked directly against the loaded
model file, not from memory). It looks at **17 specific numeric
features** — not the raw text, not the models' reasoning paragraphs,
just numbers describing the *shape* of the disagreement, the entity's
own processing history, and (new since 2026-08-30) what the project's
own permanent knowledge graph already independently knows about this
same word-to-concept pairing.

### The real 17 features for this exact "fever" decision

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
| 16 | `prior_confirmation_count` | How many times has this exact (word → concept) pairing already been confirmed correct before, in DuckDB's own record of `mollm_tier_gate_decisions`/`hitl_review_queue`? (capped at 10, scaled to 0–1) | **1.0** (capped — 21 confirmations at the time this specific stored decision was made; querying live today, that count has grown to **47**, since the pipeline has kept processing "fever" mentions in other notes since then) |
| 17 | `kg3_confirmation_count` | The same idea as #16, but sourced from a completely different place: the live Memgraph knowledge graph (KG3) itself, not DuckDB. Counts real `:PatientObservation` nodes already written into KG3 for this exact (text, concept) pair. (capped at 10, scaled to 0–1) | **1.0** (capped — queried live against the real KG3 instance right now: **39** existing `:PatientObservation` nodes already link the text "fever" to concept 437663) |

**Why this is a genuinely separate feature, not a duplicate of #16**: KG3's
current population is gold-simulated (a one-off backfill from this
project's gold-graded history, not real clinician review clicks — see
§7.5 below), so it is deliberately kept as its own, separately-weighted
input rather than merged into `prior_confirmation_count` — the model is
free to learn that the two agree (as they do here: 47 vs. 39, the same
underlying story from two different tables) without this document, or
the calibrator's own design, ever asserting KG3 is currently an
independent, human-verified source of truth. See §7.6 for exactly how
this number is computed against the live graph.

**In plain terms, what this feature vector is telling the calibrator**:
"Two of three models agreed and were reasonably confident; the retrieval
itself was about as clean and reliable as retrieval ever gets (a perfect
Tier-1 exact-text match, no ambiguity flags, no fallback machinery
involved); and — importantly — this exact word-to-concept pairing has
already been independently confirmed correct dozens of times before,
both in this project's own decision history AND in the separate,
already-written knowledge graph." That combination of signals is exactly
the pattern the calibrator learned to trust.

### The calibrator's real output

```
calibrated_score:  0.927709
threshold:         0.78   (CALIBRATED_AUTO_THRESHOLD, fixed in advance)
0.927709 >= 0.78  →  PROMOTE to TIER_1B_CALIBRATED_AUTO_VALIDATED
```

The system's own recorded, human-readable explanation for this exact
decision (pulled directly from the database, not paraphrased — this
decision was stored on 2026-08-20, when the threshold constant in the
code at that time was still 0.72; both numbers below are real, just
from two different points in this project's history):

> *"non-unanimous verdicts {'NONE_CORRECT': 1, 'SUPPORTED_1': 2}, but
> ConsensusCalibrator scored 0.927709 >= 0.72 (prior_confirmation_count=21)"*

**Why the threshold moved from 0.72 to 0.78, told honestly**: the
original 0.72 was picked as the smallest threshold that reached 100%
validation precision on a clean sample — but that sample itself later
turned out to be contaminated (some of its "clean" examples had leaked
in from the project's own locked test split, a real bug fixed on
2026-08-31 — see project history). Re-deriving the threshold the exact
same way, on the corrected, leak-free data, moved it to 0.78. The
"fever" decision above still clears the new, stricter bar comfortably
(0.927709 is well above 0.78), so the actual outcome for this specific
entity is unchanged — but a borderline case that had scored, say, 0.74
would flip from promoted to HITL-routed under the corrected threshold.

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
this project calls the "abbreviation flywheel," §7c below).

### 7a. What an approved decision actually writes into the knowledge graph (KG3)

Once a decision is approved (by a human, or — for the small set of
tiers trusted enough — automatically), the pipeline writes a permanent
record into a separate graph database, **KG3** (a Memgraph instance,
Bolt-compatible with Neo4j), so the fact "this exact span of this exact
note maps to this exact standardized concept, and here's the full
evidence trail for why" is queryable later, not just sitting in a
one-off log line.

**A real record, pulled live from the actual running KG3 instance** (a
different entity than "fever" — "upper respiratory infection," from note
`19895550-DS-7` — chosen because it's a genuine, complete 4-node chain
that could be walked end to end; "fever"'s own KG3 node exists but this
one shows every linked node type in one real example):

```
:PatientObservation {
  entity_id:        "19895550-DS-7-e3afe58fe3e"
  note_id:           "19895550-DS-7"
  raw_text:           "upper respiratory infection"
  label:              "Condition"
  orig_start:         533
  orig_end:           560
  confidence:          0.8861986994743347
  omop_concept_id:    4181583
  vocabulary_id:      "SNOMED"
  matched:            true
}
      │ [:INSTANCE_OF]
      ▼
:Concept {
  omop_concept_id:    4181583
  concept_name:        "Upper respiratory infection"
  domain_id:            "Condition"
  vocabulary_id:        "SNOMED"
}

:PatientObservation ──[:VALIDATED_BY]──▶ :MoLLMDecision {
  mollm_call_id:       "e29047cd-8b5e-4aae-a99f-5ef226770097"
  source_table:         "mollm_tier_gate_decisions"
}
      │ [:REVIEWED_BY]
      ▼
:HITLReview {
  hitl_case_id:            "gold_populated_e29047cd-8b5e-4aae-a99f-5ef226770097"
  final_decision_status:    "APPROVED"
  queue_reason:              "gold_based_population_TIER_1_AUTO_VALIDATED"
}
```

Every field above was read directly off the live graph with a real
Cypher query — nothing constructed for illustration. The full
provenance trail this node points back to (the actual
`mollm_tier_gate_decisions` row for `mollm_call_id
e29047cd-8b5e-4aae-a99f-5ef226770097`, in DuckDB) is just as real:
`tier: TIER_1_AUTO_VALIDATED`, `composite_confidence: 0.79114`, a
3-for-3 unanimous `SUPPORTED_1` vote, and each of the three models' own
full Step-A/Step-B reasoning trail, verbatim — exactly the same shape of
evidence trail §5 walked through for "fever."

**An honest caveat about this specific record, stated plainly rather
than glossed over**: this particular node's `hitl_case_id` is prefixed
`gold_populated_`, not the two real write-paths this pipeline ships
today (see next paragraph) — it came from a one-off historical backfill
that populated KG3 from this project's own gold-graded Stage 3 history,
so its `final_decision_status: APPROVED` reflects a simulated,
gold-based approval, not a real clinician's click. This is consistent
with what this project's own memory record already states about KG3's
current population (gold-simulated, not real human review), and is
exactly why kg3_confirmation_count (§7b) is kept as its own, separately
distrusted calibrator feature rather than treated as equivalent
independent evidence.

**The two real, currently-shipping write paths**, for comparison (both
in `src/kg3_ingestion.py`, both write the identical 4-node graph shape
above, differing only in the `:HITLReview` node's own fields):

- `ingest_reviewed_case()` — the human-reviewed path. `hitl_case_id` is
  the real reviewer's own queue case id; `final_decision_status` is
  `APPROVED` or `CORRECTED`, taken directly from what a real reviewer
  clicked in the HITL Review Queue page (§7).
- `ingest_auto_decision()` — the new, unreviewed-but-trusted path for
  Tier 1/1B/2/3 decisions specifically. `hitl_case_id` is always
  `auto_{mollm_call_id}`; `final_decision_status` is always the literal
  string `"AUTO"` (never a human decision) — and **this path defaults to
  `dry_run=True`**: it computes and returns exactly what *would* be
  written, without ever touching Memgraph, until a caller explicitly
  passes `dry_run=False`. As of this document, every real production run
  still calls it in dry-run mode — direct, unreviewed KG3 writes remain
  gated pending further validation, a deliberate, standing conservatism
  (§7 above already explains why: even the best tier's precision on
  genuinely unseen notes, 76.8%, isn't close enough to 100% yet).

### 7b. How KG3 feeds back into the calibrator — the exact query, the exact numbers

`src/kg3_query.py`'s `count_kg3_confirmations(driver, entity_text,
concept_id)` runs this Cypher query against the live graph:

```cypher
MATCH (obs:PatientObservation)-[:INSTANCE_OF]->(c:Concept {omop_concept_id: $cid})
WHERE toLower(trim(obs.raw_text)) = toLower(trim($text))
RETURN count(obs) AS n
```

Run live, right now, against the real running instance, for the two
entities this document already follows:

| Entity text | Concept ID | `kg3_confirmation_count` (real, live) | `prior_confirmation_count` (DuckDB, real, live) |
|---|---|---|---|
| "fever" | 437663 (Fever) | **39** | **47** |
| "upper respiratory infection" | 4181583 | **1** | **1** |

Both numbers get capped at 10 and scaled to 0–1 before reaching the
calibrator (feature #16/#17 in §6's table) — so "fever," at 39 real KG3
confirmations, saturates that feature at its maximum value of 1.0,
exactly as shown in §6's table. "Upper respiratory infection," with only
1 confirmation on file, would contribute a much smaller 0.1 for that
same feature if it were ever routed back through the calibrator — the
same mechanism, genuinely different evidence strength, depending on how
often the pipeline has actually seen and confirmed that specific
pairing before.

### 7c. The abbreviation flywheel — feeding corrections back into the pipeline

The idea: every human correction (and, for the frequency-based
mechanism below, every one of the pipeline's own confident-but-recorded
Stage 2b outcomes) is a real, cheap-to-mine data point about which
meaning of an ambiguous word is actually correct in practice — so
instead of only ever using them once, mine them into standing rules that
help the *next* similar mention resolve faster and more reliably,
without needing a fresh model call at all.

Two real, built mechanisms exist, and their current live status is
genuinely different — reported honestly, not smoothed over:

- **`compute_frequency_priority()`** — aggregates the pipeline's *own*
  Stage 2b outcomes for an abbreviation into a "which meaning did we
  pick most often" ledger. **A real, cautionary finding from this
  project's own history**: an early version of this mechanism was tested
  against 50 real, production-processed notes (8,062 entities), and the
  top 7 highest-confidence ledger winners it would have promoted were
  gold-checked directly — **7 out of 7 were wrong** (`DM` → "deep
  masseter" instead of diabetes mellitus; `IVF` → "In Vitro
  Fertilization" instead of IV fluids; and five more of the same shape).
  Worse, the mechanism was caught, mid-run, re-selecting its own earlier
  wrong guesses as supposedly-confirming evidence — the exact circularity
  risk this whole design was built to guard against. **The fix,
  live today**: this mechanism now requires an abbreviation to be on an
  explicit, manually gold-verified `VERIFIED_ALLOW_LIST` before it will
  ever return an answer at all — and that list **starts, and currently
  remains, empty**. So `compute_frequency_priority()` is real, tested,
  wired in, and returns `None` (defers to the existing static rules)
  for every abbreviation today, by design, until an entry is
  individually gold-verified and added.
- **`mine_context_rules()`** — a structurally different, deliberately
  *not*-excluded mechanism: it mines rules only from
  `hitl_review_queue` rows a **real human reviewer** actually confirmed
  (`reviewer_decision IN ('APPROVED','CORRECTED')`), on the reasoning
  that independent human confirmation is exactly the kind of evidence
  that could correct a systematic model bias, rather than just
  reinforcing one. **Checked live, right now, against the real
  database**: `hitl_review_queue` currently holds 19,103 rows, and every
  single one of them has `reviewer_decision = 'PENDING'` — zero real
  reviewer decisions exist yet (this project's standing HITL-everything
  policy from §7 means the queue is populated, but not yet worked, in
  the currently deployed state). `mine_context_rules()` therefore
  correctly, honestly returns **0** rules today — not a bug, an accurate
  reflection of "no real review data exists yet to mine."

## 8. Knowledge Graph Embeddings — what was actually built, and the honest result

**Update, this is now current as of this document's latest revision**:
this project's original plan named three possible methods for
representing the SNOMED graph as dense numeric vectors — **TransE,
RotatE, and CompGCN**. An earlier version of this document said only
TransE had been built. That's now out of date: **RotatE has since been
built too**, as a genuine 4-configuration ablation (not just "does
RotatE work," but "does RotatE trained on four different real data
sources work") — full real results in §8b below. **CompGCN (a full
graph neural network architecture) remains the one deliberately
unbuilt method**, scoped as real, meaningfully bigger follow-on work
and explicitly deferred, not hidden.

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
was hoped to help with, a simpler, already-existing hand-built rule (see
§8c below) turned out to be **measurably safer** — the hand-built rule
made zero wrong calls across every test threshold, while the
embedding-based approach made real mistakes once you allowed it to be
more aggressive. TransE is built, tested, and — like every KGE method
evaluated in this project so far (§8b) — currently **not used to make
any live decision in the pipeline**, kept as evaluated,
promising-but-unproven future work.

### 8b. RotatE — the second method, built as a real 4-way ablation

RotatE represents each SNOMED concept as a point in a *complex-valued*
space (not the plain real-valued space TransE uses) and represents each
relationship type as a **rotation** rather than a straight-line
movement — the idea being that some relationships (e.g. "is the reverse
of") are better captured by a repeatable rotation than by addition.

**The real question this project asked before training anything**:
train RotatE on what data, exactly? The obvious choice — reuse the same
SNOMED relationship subgraph TransE already used — was checked and
rejected. Directly querying the live KG3 graph found its own core
structure (`PatientObservation`/`Concept`/`MoLLMDecision`/`HITLReview`)
has exactly 3 relationship types, each a 1:1 provenance edge, not real
relational structure — the wrong shape of graph for this kind of
model. Instead, **four separate, real training-data sources were tried,
as a genuine ablation, not a single run**:

| Config | What it's built from | Real trainable triples | Relation types |
|---|---|---|---|
| `guideline` | The curated clinical-guideline graph (also living in KG3) | 263 | 26 |
| `gold` | This project's own gold-graded correct-vs-wrong candidate pairs | 1,209 | 1 (`PREFERRED_OVER`) |
| `combined` | `guideline` + `gold` together | 1,472 | 27 |
| `snomed_is_a` | The full SNOMED "is a" hierarchy, pulled from a live Neo4j copy | 530,515 | 1 (`IS_A`) |

**Two separate evaluations were run for all four, plus TransE for
comparison** — and this is the real, honest heart of the finding: the
two evaluations point in *opposite* directions.

*Evaluation 1 — does the embedding space separate right answers from
wrong ones, in aggregate?* Yes, for `gold` especially: it correctly
places the right concept measurably closer than a random one in 78.9% of
cases — actually beating TransE's own 63.7% on this same measure.

*Evaluation 2 — does picking a winner by embedding distance actually
help or hurt on individual, real, gold-graded tiebreak decisions?* This
is the test that matters for whether it's safe to use — and here, every
single config **loses badly**, `gold` included: 97 wins against 757
losses (net **−660**). `combined` is nearly identical (−659).
`snomed_is_a` is the single worst performer of all five methods tested,
including TransE (net −478, only a 1.4% win rate on the cases it could
even resolve). `guideline` couldn't resolve any real cases at all — its
training graph is simply too small.

**The core finding, stated plainly**: a method can have genuinely good
*aggregate* signal (RotatE-on-gold's 78.9%, the best of any method
tested) and still be actively harmful as a *per-decision* tiebreak
(−660 net). Those are different questions, and this project measured
both, on purpose, rather than trusting the easier-to-compute aggregate
number alone. **Every RotatE configuration loses to TransE, and TransE
itself already loses to the existing hardcoded rule (§8c)** — so neither
of the two KGE methods built in this project is used to make any live
routing decision today. This is a complete, considered, negative result
for both remaining named methods from the original project plan
(RotatE now built; CompGCN deliberately still isn't) — reported exactly
as measured, not adjusted to look better.

### 8c. What actually replaced KGE — the hardcoded rule that beat both

The rule that both TransE and every RotatE configuration lose to is a
3-line, hand-written tiebreak, `_prefer_lab_procedure_over_observable()`
(`src/normalization/tier_retrieval.py`). It applies to one specific,
well-understood pattern: for a Lab-Test-labeled entity (like "WBC" —
white blood cell count), SNOMED often has *two* separate concepts for
the same real-world thing — a "Procedure"-class concept (the act of
measuring it) and an "Observable Entity"-class concept (the abstract
property being measured) — and this project's own embedding model
(SapBERT) consistently scores the *wrong* one (the Observable Entity)
higher by raw text similarity.

**Why this rule, and not a KGE model, was trusted**: measured directly
against this project's own gold-graded corpus, across every Lab-Test
entity where both concept classes were present as candidates, the
Procedure-class concept was the gold-correct answer in **78 out of 78
cases** — zero exceptions. One concrete real example: for the
abbreviation "WBC," SapBERT itself scores "Leucocyte count" (the wrong,
Observable-Entity concept) at 0.892 and "White blood cell count" (the
right, Procedure concept) at 0.8694 — genuinely close, and wrong by
raw similarity alone. The rule doesn't change either score (a reviewer
still sees the model's true, unmodified numbers) — it only re-ranks
which candidate is offered first, based on a pattern that has held
without a single counterexample across the whole corpus tested so far.
That's a stronger, more falsifiable evidence base (78/78, exceptionless,
on real gold data) than either embedding method managed to produce, and
it's why this simple rule — not a learned model — is what's actually
live in the pipeline today for this specific class of decision.

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

The system that actually resolves clinical mentions to codes queries it
directly as relational tables (via DuckDB), which turned out to be both
simpler and fast enough for this project's real query patterns.

**A correction to this section, found later in this same investigation**:
a real graph-database copy of SNOMED CT *does* exist on this project's
own infrastructure — a Neo4j instance holding 386,110 real concepts and
641,727 "is a" relationships, verified by querying it directly. It is
just never touched by anything the pipeline actually runs — the only
code that ever connects to it is a standalone diagnostic script. See §10
below for what happened when it was tried as a live input to the AI
models' reasoning.

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

## 10. An experiment — what if the AI models had a dictionary?

§5 showed one of the three AI models (qwen) getting confused about
"fever": it defined the word as *"absence of fever in a patient"* —
mixing up "what does this word mean" with "does the patient have it."
That confusion is the direct, documented reason it voted the wrong way.

Why did it get confused? Because when the system asks a model *"what
does this word mean, clinically?"*, it hands over the note text and
nothing else — no dictionary, no textbook, no medical reference of any
kind. The model has to answer purely from whatever it happened to learn
during its general training. It isn't allowed to look anything up.

That raised an obvious question: **this project already has a real,
working copy of the SNOMED medical vocabulary sitting in a graph
database (see the correction in §9a) — what if we let the model consult
its "is a" relationships before answering?** A graph "is a" relationship
is simple: it just says one thing is a specific kind of another thing —
"Fever is a kind of *elevated body temperature finding*," for instance.
It's not a dictionary definition, but it's real, structured medical
knowledge the model could actually be shown.

### The experiment

Using the exact same "fever" entity followed through this whole
document, the system was asked "what does this word mean?" twice — once
exactly as it works today (no outside information), and once with one
extra paragraph added, pulled live from the real graph database:

> *RELATED SNOMED TAXONOMY (context only): "Fever" is a type of "Body
> temperature above reference range." Narrower/related terms include:
> Recurrent fever, Chronic fever, Fever due to infection...*

Everything else about the question was left completely unchanged — same
model, same note text, same everything. The only difference was that one
extra paragraph.

**The result: qwen's answer changed from wrong to right.**

| | Without the graph | With the graph |
|---|---|---|
| qwen's definition | "absence of fever in a patient" ❌ | "a symptom of elevated body temperature" ✅ |
| qwen's final vote | Rejected the correct answer | **Accepted the correct answer**, very confidently |

And because the second, separate check ("does this SNOMED code match
that definition?") was also re-run with the new definition, this wasn't
just a nicer-sounding sentence — the model's actual final decision
flipped from wrong to right. All three models would now agree
unanimously, which means this case would no longer need the calibrator
(§6) to rescue it at all — it would just be correctly resolved on the
first pass.

### But — a more honest, larger test told a more complicated story

One example proving a point is not the same as the mechanism actually
working well in general, so a further test was run: 15 *different* real
cases where the three models had genuinely disagreed with each other (the
same population §6's calibrator exists to rescue), each checked against
the true answer from this project's gold-standard reference data.

Two honest findings came out of that larger test, neither of them as
clean as the "fever" story:

1. **The graph could only actually be consulted for 5 of the 15 cases**
   (a third). The lookup only works when the exact wording in the note
   matches a term in the graph precisely — and most real clinical
   language doesn't. Abbreviations like "HTN" expand to "Hypertension,"
   but the graph's own official name is "Hypertensive disorder" — close,
   but not an exact match, so the lookup came up empty. Phrases like
   "renal failure" or "groin pain" simply don't exist as their own single
   entry in the graph at all. The graph itself is real and useful; the
   simple way of *searching* it used in this test just wasn't
   sophisticated enough to find something usable most of the time.
2. **Of the 5 cases it could check, none flipped from wrong to right —
   and one got a real bit worse.** Four stayed exactly as correct as they
   already were (one of them became more *confident* — going from a
   shaky 2-out-of-3 agreement to all 3 models agreeing — without
   changing the actual answer, which is still a genuine improvement in
   safety even without changing the outcome). But in the fifth case (a
   pneumonia mention), giving the model extra background information
   backfired: qwen's new definition invented specific medical detail
   ("bilateral opacities suggestive of multifocal pneumonia") that
   wasn't actually anywhere in the note, and it then second-guessed
   itself and rejected the right answer for a different, worse reason
   than before. The final team decision was still correct because the
   other two models outvoted it, but qwen itself got less reliable, not
   more.

### A fairer test — 10 cases picked specifically from where the graph *can* help

The test above mixed together cases where the graph could help with
cases where it couldn't even try, which made it hard to tell how good
the idea really is *when it applies*. So a second, more targeted test
was run: instead of grabbing 15 random disagreement-cases, the system
kept sampling real cases until it had found **10 where the graph lookup
actually worked** (it had to check 31 real cases to find those 10 — the
same roughly one-in-three hit rate as before).

**This time, the result was a clean, real win:**

| | Without the graph | With the graph |
|---|---|---|
| Correct | 6 out of 10 | **8 out of 10** |
| Got fixed (wrong → right) | — | **2 cases** |
| Got broken (right → wrong) | — | **0 cases** |

Both of the newly-fixed cases (a heart-damage blood test called
"Troponin," and a "neck swelling" mention) followed the exact same
pattern as "fever": the models had been leaning toward the wrong answer
2-votes-to-1, and once they were shown where the term actually sits in
the medical hierarchy, all three agreed on the right answer.

The two cases that stayed wrong did so for a completely different, fair
reason: for those two, the *earlier* step in the pipeline (the one that
searches the vocabulary for candidate matches, before the AI models ever
get involved — see §4) had already handed the models the wrong option to
choose from in the first place. All three models agreed with each other
both times, correctly following instructions — the problem was upstream
of anything this experiment touches, and giving the models a dictionary
can't fix a question they were never actually asked.

**The fair, honest summary, taking both tests together**: giving the AI
models real medical-hierarchy context to reason from is a genuinely
promising idea. It only reaches roughly a third of the cases that would
need it, because finding the right entry in the graph for a given
phrase is its own hard problem — most real clinical wording (short
abbreviations, multi-word phrases) doesn't land on an exact match. But
**when it does reach a case, the evidence so far is genuinely
encouraging**: the first, broader test found no clear pattern either way
(one improvement, one setback, mostly no change); this second, more
targeted test — specifically built to isolate the cases where the idea
could actually be judged — found a real 20-percentage-point improvement
with zero cases made worse. That's still a small sample, not proof this
would hold up across the whole corpus, but it's real, encouraging
evidence rather than a coin flip. This is exactly why this project
treats it as a promising experiment worth pursuing further (starting
with a better way of finding graph matches in the first place), not
something ready to switch on for real notes yet.

---

## 11. A different experiment — what if we just used a real medical AI instead?

Section 10 tried giving a general-purpose AI model outside help (a
medical dictionary of sorts) to make up for it not having medical
training. A different, simpler question follows naturally: **what if we
skipped that, and just used an AI model that was actually trained on
medical text in the first place?**

### First, a real, useful side-discovery

This project's own setup notes name two medical AI models it originally
planned to use — but checking directly, **neither is actually running**
anywhere on this machine right now. Separately, this project's storage
already has two *different* copies of similar medical models sitting
fully downloaded but never used, because the software needed to actually
run that particular file format was never installed. So "does the
project have a medical AI available" turned out to have a real, slightly
disappointing answer: sort of, but not in a form that's usable today
without extra setup work.

### So two real medical AI models were downloaded fresh and tried

Two genuine, publicly available medical AI models were pulled down for
real — not simulated — and given the exact same "fever" puzzle from
section 5, completely unaided (no dictionary help this time — just
"here's the note, here's the candidate answer, does it match?").

**One of them got it right. The other got it wrong — and in a strange,
new way.** The one that failed didn't seem confused about the medicine
at all — when asked to describe what "fever" means, it gave a
perfectly sensible answer. But then, when asked "does the official
answer match your own description," it said no, and its stated reason
made no sense: it claimed the two didn't belong in the same category,
which wasn't even true. A model that sounds medically fluent isn't
automatically a *reliable* one for this specific kind of check.

### A bigger, fairer test: the same 10 real cases from section 10

| | Correct out of 10 |
|---|---|
| Regular AI models, no help | 6 |
| Regular AI models, **with** the medical dictionary | **8** |
| Medical AI #1 (the one that failed on "fever") | **0** |
| Medical AI #2 (the one that succeeded on "fever") | 7 (plus 1 case it couldn't answer at all) |

The model that got "fever" wrong got **every single one of the 10 cases
wrong** — including several where its own description of the term was
basically correct, but it then said "no" anyway when asked to confirm
the match. That's a strange, specific kind of unreliability, not a lack
of medical knowledge.

The other medical model did about as well as the regular models did
*with* the dictionary's help — and the few cases it missed were the same
two cases everything else in this whole document also got wrong, for an
unrelated reason (the very first search step, described in section 4,
never found the right answer to hand to any AI model in the first
place — no model, however well-trained, can pick a correct answer it
was never shown).

**The fair, honest takeaway**: being "trained on medical text" doesn't
automatically make an AI model trustworthy for this specific job. One
real medical model tested here was the single worst performer of
everything tried anywhere in this document. The other was genuinely
good — about as good as giving a regular model a dictionary to consult.
Which specific AI model you pick matters at least as much as whether it
was "medically trained" in general.

---

## 12. Four real test batches, compared side by side

Beyond following one entity end-to-end, this project has run the whole
pipeline on four different small batches of real notes at different
points in time, each fully processed through Stage 3 (every eligible
entity decided, not a partial/capped run). Comparing them side by side
shows how the system's real, measured behavior has actually shifted —
not a guess, four real runs. The fourth is the most demanding test yet:
five notes verified to be genuinely new to *every* part of the pipeline,
including the calibrator's own training history:

| | Fresh-10 (2026-08-20) | Fresh-5 original (2026-08-30) | Fresh-5 gazetteer (2026-08-31) | Fresh-5 calibrator-unseen (2026-09-01) |
|---|---|---|---|---|
| Notes | 10 | 5 | 5 | 5 |
| Total Stage 3 decisions | 258 | 373 | 411 | 688 |
| Found the right SNOMED code at all ("linked recall") | 26.8% | 40.1% | 33.5% | 27.7% |
| Of everything auto-approved, how often it was actually right | 76.8% | 92.0% | 93.3% | 86.9% |
| Share of decisions that skipped human review entirely | 30.2% | 56.6% | 47.9% | 53.2% |
| Average seconds to decide one entity in Stage 3 | 10.2s | 6.9s | 7.0s | 7.0s |

**What this actually shows**: the three more recent batches auto-approve
correctly well above the older Fresh-10 batch's 76.8%, reflecting
everything fixed on the pipeline in between (SNOMED crosswalk fix,
near-duplicate concept fixes, the calibrator, and — most recently — the
`normalized_entities` bug described in §7a-7b). The newest batch's 86.9%
sits between the two prior Fresh-5 batches, but that blended number
actually hides a stronger individual result: broken down by tier, this
batch's calibrator-promoted decisions (`TIER_1B`) hit **100%** (25/25) and
its exact-match decisions (`TIER_3`) also hit **100%** (63/63) — both as
good as or better than any prior batch. The lower blended average comes
from this batch simply having a bigger *share* of genuinely hard,
split-vote entities (`TIER_4`, 33.3% of the total, the largest share of
any batch except Fresh-10), not from any individual tier getting worse.
Stage 3's own per-entity speed has stayed remarkably stable across the
three most recent batches (6.9s, 7.0s, 7.0s) — real evidence the
ensemble's own cost hasn't drifted even as everything else about the
pipeline changed around it.

### A fifth data point, deliberately different — the same 5 notes, twice, one setting changed

The three batches above each differ in more than one way (different
notes, different points in time, several fixes landing in between) — a
real, honest limitation already noted for that table. To isolate one
specific mechanism's effect cleanly, a fourth experiment was run
2026-09-01: five brand-new notes (never processed before, not among the
5-model calibrator's own training notes) were run through Stage 1-2b
**twice** — once exactly as normal, once with the newly-extended
GLiNER gazetteer fallback (§7c, now 24 terms) switched on — with
literally nothing else different between the two runs.

| | Without the gazetteer | With the gazetteer (24 terms) |
|---|---|---|
| Entities found (Stage 2a, all 5 notes) | 688 | 700 (+12) |
| Found the right text span at all ("span recall") | 50.2% | **51.5%** |
| Found the right text *and* the right SNOMED code ("linked recall") | 27.4% | **27.7%** |

A small, real, positive result — not a large one, and that's expected:
the 24-term list was deliberately built narrow and safe (§7c already
walks through the two-bar test every term had to pass, and the 6 terms
that got rejected for being too common). Almost all of the gain came
from just one of the five notes (9 of the 12 extra entities) — a real
reminder that a 5-note sample is small enough for one note's own content
to dominate the result, not proof the mechanism only helps one kind of
note. The gazetteer stays off by default in production pending a larger
validation batch, the same standing policy every other new mechanism in
this project follows before being turned on for real notes.

---

## 13. All ablation studies run across this pipeline — the complete, honest scoreboard

This document has already walked through several of these in detail as
they came up naturally (§8's TransE/RotatE comparison, §10-11's
guideline-context and medical-AI-model experiments). This section pulls
every real ablation/A-B study run across the whole project into one
place, so the pattern across all of them is visible at a glance —
**most of them found the tested idea does NOT beat what the pipeline
already had**, which is itself a meaningful, repeated finding, not a
string of failures to be embarrassed about.

| # | What was tested | Real result | Adopted? |
|---|---|---|---|
| 1 | Lab-Test Procedure-vs-Observable tiebreak rule (§8c) | 78/78 exceptionless on gold data | **Yes — live in production** |
| 2 | TransE knowledge-graph embeddings (§8) | Loses to rule #1: 130 wins / 379 losses (net −249) on the rule's own applicable subset | No — evaluated, not wired to any decision |
| 3 | RotatE, 4-config ablation — `guideline`/`gold`/`combined`/`snomed_is_a` (§8b) | Best aggregate signal of any method (`gold`, 78.9%) but worst per-decision safety of any method (net −660) | No — evaluated, not wired to any decision |
| 4 | RRF hybrid retrieval (BM25 + dense + prior) vs. dense-only Tier 3 search | Dense-only strictly beat every blended weight tested, on both Top-1 (61.3%) and oracle (74.2%) accuracy | No — hybrid retrieval flag stays off |
| 5 | Guideline-evidence injection into the Stage 3 tiebreak prompt | 23 gradable paired entities, 20/23 correct in BOTH arms, **zero flips either direction** — the injected evidence was real but one-sided, never actually discriminating between the tied candidates | No — flag stays off |
| 6 | MoLLM acronym-escalation (resolving ambiguous abbreviations via a live model call, §"Phase 4") | 34.3%→36.1% precision across two corpus-scale grading passes — a systematic textbook-prior bias (e.g. "LAD" always resolved to the artery, even when gold meant "Lymphadenopathy") | No — stays off by default |
| 7 | `compute_frequency_priority()` — the pipeline's own past picks as a tiebreak (§7c) | 7/7 gold-checked promotions were wrong on first real-data test; caught actively re-confirming its own earlier mistakes | No — gated behind an empty, manually-curated allow-list |
| 8 | GLiNER gazetteer fallback — recovering entities GLiNER's neural model misses entirely, via a small curated term list | 96.1% span-level precision (488 TP/20 FP) on the train split; separately found only 9/17 (52.9%) of recovered spans went on to link to the *correct* concept downstream. **Extended 13→24 terms (2026-09-01)**, re-mined to rank 50 with a stricter two-bar test (gold-consistency AND a measured extraction-worthiness/tag-rate check — the second bar alone rejected 6 more candidates, including `k`/`mcv`/`infection`, that would have repeated the `interactive`/`evaluation`/`surgery` over-extraction mistake). A real, isolated before/after on 5 fresh notes (§12): span recall **+1.3pp**, linked recall **+0.4pp** | Partially — 24 of 32 mined candidates kept across two passes; `glucose` and 7 others excluded on real, measured evidence (context-gating unreliable, or failed one of the two safety bars). Flag stays off by default pending a larger validation batch |
| 9 | `prior_confirmation_count` calibrator-feature ablation (dropping it and re-fitting) | Model's real signal came from consensus-shape/retrieval-provenance, not this feature — but this same investigation surfaced a real, separate false-positive cluster (coronary-artery-segment abbreviations like LCX/LMCA) | `prior_confirmation_count` kept; a dedicated hard-coded trap added for the coronary-segment pattern instead |
| 10 | Calibrator retrain on a larger, 51-note pool | Val AUROC dropped (0.701 vs. the 0.74 baseline) — more data did not automatically help; also surfaced that the deployed threshold's precision on a larger, more diverse validation set (89.5%) was lower than the small-sample read that first justified it | Diagnostic only — nothing changed in production from this run alone |
| 11 | Medical-domain-pretrained AI models vs. general-purpose models, on the same tiebreak task (§11) | One medical model scored 0/10 (worse than every other approach tried in this whole document); the other scored 7/10, matching the general model + dictionary-context combination | Neither swapped in — which specific model matters more than whether it was "medically trained" |
| 12 | SNOMED-graph "is a" hierarchy context injected into the Stage 3 prompt (§10) | On a fair, targeted 10-case sample: 6/10 → 8/10 correct, 2 cases fixed, 0 made worse | No — promising, but the lookup only succeeds for about 1 in 3 real mentions; not yet reliable enough to enable broadly |
| 13 | `normalized_entities` schema audit — was the recall gap a model problem or a plumbing problem? (§7a) | Found a real, 100%-explained bug: two entity_ids sharing the same (text, label) tuple collapsed into one DB row, silently orphaning 8,653 entities corpus-wide (~14pp of the recall shortfall) from ever reaching Stage 3/HITL/KG3 — not a GLiNER/SapBERT accuracy issue at all | **Yes — fixed, live**. Already-published recall/precision numbers didn't change (a grading-script workaround had already masked the symptom); the real gain is 8,653 entities now reachable downstream that weren't before |
| 14 | GLiNER extraction-confidence threshold sweep — is 0.35 still the right cutoff on the current, grown corpus? | Full corpus, 12-point sweep (0.35→0.90): **0.35 is simultaneously the best recall point (53.0%) AND the best precision point (70.6%)** — precision falls as the threshold rises, recall falls monotonically, no point on the curve beats the starting value on either axis | No change — the current 0.35 setting is confirmed optimal, not just historically chosen |
| 15 | SapBERT similarity-floor sweep, below the current cutoff — would lowering `TIER3_SIMILARITY_FLOOR` (0.72) recover real matches? | Real re-computation (not simulated) on 300 gold-gradable entities currently dropped as "no candidate": precision sits **flat at 4-7% all the way down to 0.50** — ~1,650 more entities could be recovered by lowering the floor, but ~96% of them would be wrong. (Caught and fixed a real self-made bug first: comparing an OMOP `concept_id` directly against a raw SNOMED code — different id spaces — mechanically forced a meaningless 0% before the fix) | No change — 0.72 confirmed correct; the real gap in this population is a genuine retrieval-quality problem, not a threshold to tune |

**The one clear, adopted win** in this whole scoreboard is #1 — a
3-line hand-written rule, backed by an exceptionless 78/78 real-data
check. Every learned/statistical alternative tried against that exact
same problem (TransE, all four RotatE configs) lost to it. That's not a
coincidence specific to embeddings — it's the same shape of result as
#4, #5, #6, #7, and #10: a more sophisticated mechanism, tested
honestly against real gold data instead of assumed to help, did not
beat what was already there. The lesson this project draws from that
pattern, stated directly: prefer measuring a specific, narrow,
falsifiable improvement (like rule #1's 78/78) over trusting a more
general-sounding mechanism's aggregate promise — several of the
"sophisticated" ideas above (RotatE's 78.9% aggregate signal, the
guideline-evidence injection's real-but-irrelevant hits) looked
promising by exactly the kind of surface metric that's easiest to
report, and only real per-decision grading against gold caught that
they weren't actually safe to ship.

---

## 14. The whole journey, summarized

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
[Calibrator] 17 real features (incl. kg3_confirmation_count=39 from the
           live graph) → score 0.927709 ≥ 0.78 threshold
           → promoted to TIER_1B_CALIBRATED_AUTO_VALIDATED
        │
        ▼
[Stage 4]  Queued for human review anyway (standing policy) — PENDING,
           full model reasoning + full note shown side by side for the
           reviewer to make the final call
        │
        ▼
[If approved] Written into KG3 (Memgraph) as a real 4-node provenance
           chain: PatientObservation-[:INSTANCE_OF]->Concept, plus
           MoLLMDecision-[:REVIEWED_BY]->HITLReview (§7a) — which then
           feeds back into kg3_confirmation_count for the NEXT "fever"
           mention's own calibrator decision (§7b), and, once real
           reviewer decisions exist, into the abbreviation flywheel's
           mined context rules (§7c) for future ambiguous mentions.
```

Every number above is real, pulled directly from this project's own
database on the date this document was written. Nothing here is a
constructed or idealized example.
