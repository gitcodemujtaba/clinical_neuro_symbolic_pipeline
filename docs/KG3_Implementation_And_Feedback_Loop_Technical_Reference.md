# KG3 — Implementation & the HITL Feedback Loop, Technical Reference

Deep-dive companion to `docs/Implementation_Methodology.md`'s Stage 4
section. This document covers exactly how KG3 (the dynamic,
patient-instance knowledge graph, in Memgraph) is written to, how a
human reviewer's decision on a queued case gets there, and — the part
this document goes into the most depth on — the two concrete mechanisms
that **repurpose** reviewed labels to make the pipeline itself better
over time, not just to populate a graph. Every claim is grounded in the
real, current source (`src/kg3_ingestion.py`, `src/hitl_queue.py`,
`src/kg3_query.py`, `src/abbreviation_flywheel.py`,
`src/mollm_tier_calibrator.py`) and live queries against the production
DuckDB, run while writing this document.

**Headline, upfront, stated honestly**: the write paths, the review
queue, and both repurposing mechanisms below are fully built and unit
tested. The repurposing mechanisms have **not yet run on real human
review data** — live-queried while writing this document,
`hitl_review_queue` currently holds 19,103 cases, **all still
`PENDING`** (zero `APPROVED`/`CORRECTED`/`REJECTED`), and
`abbreviation_context_rules` (§5's mined-rule table) has **zero rows**
as a direct consequence. This is not a bug — it is the expected,
by-design state before any real reviewer has clicked Approve/Correct —
but it means this document describes a mechanism that is primed and
ready, not one with measured impact yet.

---

## 1. KG3 in context — which graph is which

This project has three distinct data stores that are easy to conflate,
so this is worth stating precisely before anything else:

- **KG2 (`athena_concept` + related tables, DuckDB)** — the static
  **reference vocabulary**: SNOMED/RxNorm/OMOP concepts, already fully
  populated (see `docs/SapBERT_Technical_Reference.md` §3 for how it got
  its embedding column). This never changes as a result of anything
  this pipeline does.
- **The guideline KG (file-backed, `GuidelineIndex`)** — clinical
  guideline triplets, also static reference data (see
  `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`).
- **KG3 (Memgraph, Bolt-compatible)** — the **dynamic** graph this
  pipeline itself writes: one `:PatientObservation` node per resolved
  clinical entity, linked to the `:Concept` it was grounded to, the
  `:MoLLMDecision` that grounded it, and the `:HITLReview` (if any) that
  verified it. This is the graph this document is about, and the only
  one of the three whose content depends on what this pipeline — and,
  eventually, human reviewers — actually do.

## 2. Two write paths into KG3, both gated, neither optional

`src/kg3_ingestion.py`'s own docstring is explicit that **nothing
writes to KG3 without going through one of exactly two gated paths** —
there is no third, ungated route anywhere in this codebase:

1. **`ingest_reviewed_case()`** — the original, **human-reviewed** path.
   Every case it writes has already passed through `src/hitl_queue.py`'s
   review workflow (`reviewer_decision` in `APPROVED`/`CORRECTED`). This
   path exists specifically because of a real, named risk: an early
   measurement found `AUTO_VALIDATED` precision at only 39.4%/52.6%
   under the older binary `route()` gate — writing unfiltered
   high-confidence Stage 3 output straight into KG3 at that precision
   would have baked silent errors into the graph as "verified," a
   pseudo-labeling feedback-loop risk the project's own implementation
   checklist named explicitly before this module was built.
2. **`ingest_auto_decision()`** — a deliberately **narrower, newer**
   unreviewed path, added for `src/mollm_tier_gate.py`'s Tier 1-5 gate.
   It only accepts Tier 1/2/3 decisions (`AUTO_TIERS` — the tiers whose
   entire purpose is to be trustworthy enough to skip human review), and
   defaults to `dry_run=True`, writing nothing until that trust has
   actually been validated on real data at scale (see
   `docs/Implementation_Methodology.md`'s tier-gate section and
   `docs/ConsensusCalibrator_Technical_Reference.md` for how that trust
   is measured).

**Both paths write structurally identical graph shapes** — a
`:HITLReview` node either way, distinguished only by
`final_decision_status` (`'APPROVED'`/`'CORRECTED'` vs. the literal
string `'AUTO'`) — so any query walking the graph never needs a special
case for whether a given observation came through human review or the
auto-write gate. This is a deliberate design choice, not an accident:
§6 and §7's repurposing mechanisms both benefit from being able to treat
"confirmed" uniformly regardless of which path confirmed it.

### Idempotent by construction

Every write is a Cypher `MERGE` keyed on each node's natural id
(`entity_id`, `mollm_call_id`, `hitl_case_id`, `omop_concept_id`), never
`CREATE`. This matters concretely for the batch driver
(`scripts/run_kg3_ingestion.py`), which is designed to be safely
re-runnable — it only re-attempts cases whose
`hitl_review_queue.ingested_at` is still `NULL`. Without `MERGE`, a
transaction that failed after creating some nodes but before the caller
marked the case ingested would duplicate those nodes on the next retry;
with it, a retry is a no-op for whatever already landed.

## 3. The HITL review queue — where a decision waits for a human

`hitl_review_queue` (`src/hitl_queue.py`) is a **separate table**, not a
column bolted onto the decision tables it draws from — deliberately.
`mollm_decisions`, `mollm_review_decisions`, and
`mollm_tier_gate_decisions` each record what a Stage 3 decision *was*;
this table records what happened to it *afterward* — a human reviewer's
verdict — which is a genuinely different lifecycle with its own states
(`PENDING`/`APPROVED`/`CORRECTED`/`REJECTED`) and its own writer (the
Streamlit reviewer UI, not a batch script). Keeping it separate means no
source table's schema has to anticipate a review workflow that may never
touch a given row.

### 3.1 Three source tables feed one queue

`enqueue_pending_cases()` pulls from all three Stage 3 decision tables —
`mollm_decisions` (citation-gated), `mollm_review_decisions`
(confidence-driven, all-tier), `mollm_tier_gate_decisions` (the current
production Tier 1-5 gate) — each via its own
`_presented_suggestion_from_*()` builder that normalizes that source's
particular shape (verdict strings, candidate-index conventions, or a
free-text `proposed_concept_name` needing resolution back to a concrete
`omop_concept_id`, §3.2) into one common reviewer-facing payload:
original text, candidate list, model verdicts, routing decision,
**and**, since 2026-08-15 reviewer feedback, the surrounding
`local_context`/`section_name`/`assertion_status`/`experiencer` — added
specifically because an earlier version of the queue showed model
reasoning with no note context at all, giving a reviewer nothing to
actually judge the decision against.

### 3.2 `suggested_omop_concept_id` — resolved once, at enqueue time, per source

Each source table records "what was decided" in a different shape, and
each builder resolves it to one canonical `omop_concept_id` a reviewer's
Approve click can unambiguously confirm — computed **once, at enqueue
time**, not re-derived under time pressure at ingestion time:

- `mollm_decisions`: parses `RESOLVED_TO_CANDIDATE_<N>` verdicts,
  majority vote among models that resolved to the same N (ties broken
  toward the lower index); falls back to the top-1 candidate for
  contradiction-mode verdicts (`SUPPORTED`/`CONTRADICTED`/
  `INSUFFICIENT_EVIDENCE`), since those checks validate Stage 2b's own
  top-1 rather than choosing among alternatives.
- `mollm_review_decisions`: has no `concept_id` column at all, only a
  free-text `proposed_concept_name` string. Resolution is two-pronged —
  first an explicit `"(OMOP <id>"` annotation, validated against the
  entity's real candidate-id set before being trusted (never trust an
  arbitrary number in free text on its own); then a fallback exact-match
  against `candidates[].concept_name`. A real, checked finding behind
  this design: of 20 rows with a non-null `proposed_concept_name` at the
  time this was built, 6 weren't a clean name at all — the model had
  echoed the full candidate description it was shown — which a bare
  exact-match could never catch without the id-annotation path.
- `mollm_tier_gate_decisions`: simplest of the three — `route_tier()`
  already records exactly which candidate it picked as
  `final_candidate_index` (1-based), so this is a direct list index,
  not text parsing.

### 3.3 Why *every* decision is queued, not just the uncertain ones

`enqueue_pending_cases()` queues every clean (non-error) decision from
all three sources — **including Tier 1/2/3 `AUTO_VALIDATED`/
`AUTO_RESOLVED` ones**, not just the ones that failed to reach an
auto-tier. This is recorded explicitly as **deliberate, temporary
conservatism**, not the final design: until a calibrated confidence
threshold is validated against real reviewed data at scale, the code
does not trust its own routing tier enough to skip queuing a case just
because it says "auto." `queue_reason` records the source row's own
tier/routing reason for the reviewer's context, but never gates whether
a case gets queued at all.

### 3.4 The review verdict, and what it unlocks

`submit_review()` accepts exactly one of `APPROVED`/`CORRECTED`/
`REJECTED`:

```python
final_ingestion_path = "HUMAN_VERIFIED" if reviewer_decision in ("APPROVED", "CORRECTED") else None
```

`APPROVED`/`CORRECTED` set `final_ingestion_path='HUMAN_VERIFIED'` — the
only value `load_ingestible_cases()` (§4) will ever read as ready for
KG3. `REJECTED` leaves it `NULL` — **structurally**, not just by
convention, excluded from ever reaching KG3: a rejected case has no code
path into `load_ingestible_cases()`'s `WHERE final_ingestion_path =
'HUMAN_VERIFIED'` filter, so there is nothing to accidentally forget to
check later.

`reviewer_comment` — added 2026-08-17, free text available regardless of
decision (unlike `rejection_reason`, which only makes sense on a
`REJECTED` case) — is the **real ground truth**
`mine_context_rules()` (§5) and any future analysis reads back from this
table. A reviewer explaining *why*, not just *what*, is what eventually
lets the pipeline improve from real cases instead of a guess — and,
honestly, until reviewers actually populate this field, that specific
input has nothing to mine, by design.

## 4. From a reviewed case to a KG3 write

`ingest_reviewed_case()` (`src/kg3_ingestion.py:125-169`) takes one
`load_ingestible_cases()` row plus a caller-supplied `entity_fields`
dict (the DuckDB-side lookup of `orig_start`/`orig_end`/`confidence`,
kept as a separate parameter specifically so this module's *only*
external dependency is Memgraph, not also DuckDB — `scripts/run_kg3_ingestion.py`
owns that join instead).

`resolve_concept_id()` is where a `CORRECTED` case's real value shows
up: for `CORRECTED`, it trusts the reviewer's own
`corrected_concept_id` directly — the concept Stage 2b/3 originally
proposed is irrelevant, the human's answer is authoritative. For
`APPROVED`, it uses the `suggested_omop_concept_id` computed at enqueue
time (§3.2). Either shape that reaches this function without a
resolvable id — e.g. an `APPROVED` case sourced from
`mollm_review_decisions` whose `proposed_concept_name` never resolved —
raises `UningestibleCase` **loudly**, not silently: a case dropped here
would otherwise look identical to one that was simply never queued,
exactly the kind of silent gap this project's own evaluation work has
repeatedly found expensive to diagnose after the fact.

The five-`MERGE` Cypher transaction (`_ingest_tx()`) writes, in one
atomic `execute_write()` call:

```
(:Concept {omop_concept_id}) <-[:INSTANCE_OF]- (:PatientObservation {entity_id})
                                                        -[:VALIDATED_BY]-> (:MoLLMDecision {mollm_call_id})
                                                                                  -[:REVIEWED_BY]-> (:HITLReview {hitl_case_id})
```

— either all five `MERGE`s commit, or none do (the neo4j driver wraps
the whole function in one transaction), so a partial write can never
leave the graph in an inconsistent state.

## 5. Repurposing mechanism 1 — mined context rules feed straight back into Stage 1

This is the mechanism that most directly answers "how does reviewing
labels improve the pipeline" — it closes a real loop from a human's
`CORRECTED` verdict back into how *future* notes get processed, before
Stage 3 ever runs on them.

### 5.1 The problem it solves

Stage 1's abbreviation expansion (`src/preprocessing.py`'s
`expand_text_and_track_offsets()`) resolves a multi-meaning abbreviation
via a static tiebreak chain — numeric context, then an observed-
frequency prior, then OMOP groundability — that **knows nothing about
which meaning the rest of the pipeline actually confirms**, note after
note. A concrete, directly-measured failure this exposed: `src/acronym_escalation.py`'s
own investigation found that even showing a model *clean, uncorrupted*
context, "PDA" still resolved to the wrong meaning across all three
MoLLM models — the models' own prior toward the more textbook-famous
reading overrode good context. A deterministic rule that never asks a
model to judge sidesteps that exact failure mode.

### 5.2 `mine_context_rules()` — turning reviewer verdicts into deterministic rules

`mine_context_rules()` (`src/abbreviation_flywheel.py:345-465`) reads
**exactly** the population this document's headline caveat describes —
real, human-confirmed resolutions:

```sql
SELECT e.original_text, h.corrected_concept_id, e.orig_start, e.orig_end, e.note_id
FROM hitl_review_queue h
JOIN extracted_entities e ON e.entity_id = h.entity_id
WHERE h.reviewer_decision IN ('APPROVED', 'CORRECTED')
  AND e.expansion_ambiguous = TRUE
```

— i.e. every case where a human approved or corrected a resolution for
an entity Stage 1 itself had already flagged as an ambiguous-expansion
candidate. For each one, it slices a small pre/post word window around
the entity's span **from the raw, unexpanded note text** — deliberately
not Stage 1's own expanded text, reusing the exact reasoning
`src/acronym_escalation.py`'s `raw_local_context()` fix already
established: showing a rule-miner Stage 1's own already-expanded text
would mean training on a prior guess dressed up as literal wording, the
same circularity risk this whole mechanism exists to avoid.

**Scoring**: for each `(abbreviation, meaning, trigger_word, position)`
combination, `_log_odds()` computes the add-alpha-smoothed log-odds of
that word appearing near *this* meaning versus every *other* meaning of
the same abbreviation — a word that appears near one meaning and never
the others is a strong, real signal; a word appearing near all of them
equally is not. Two support floors before a rule is ever written, both
guarding against building a rule from too little evidence:
`MIN_CONTEXT_RULE_SUPPORT = 5` independent confirmed examples per
abbreviation (summed across its meanings) before *any* rule for it is
attempted, and per-word support `>= 2` within that. Rules are written to
`abbreviation_context_rules`, upserted idempotently (safe to re-run this
miner repeatedly as more review data accumulates).

### 5.3 Where the mined rule gets consumed — closing the loop

`select_by_context_pattern()` (`src/abbreviation_flywheel.py:287-324`)
is consulted by Stage 1's own expansion tiebreak chain — the **first**
tiebreak tried, checked before any numeric-context heuristic or model
call, the same "skip the judgment call entirely when a deterministic
answer exists" shape as `src.mollm_tier_gate.tier3_fast_path()`. Given
an abbreviation's real pre/post context words at a *new* occurrence, it
looks up matching trigger-word rules and returns the meaning with the
higher summed score — but explicitly refuses to force a pick when two
meanings' matched rules score a genuine tie, falling through to the
existing chain instead of guessing.

**The loop, stated end to end**: a human reviewer corrects one ambiguous
abbreviation resolution → `mine_context_rules()` turns that correction
(plus enough others like it) into a trigger-word rule →
`select_by_context_pattern()` applies that rule to every *future*
occurrence of the same abbreviation in a *similar* context, at Stage 1,
before GLiNER or any MoLLM model ever runs on that note. A correction
made once, on one note, structurally improves every future note that
shares the same pattern — this is the concrete mechanism, not a
metaphor.

### 5.4 The asymmetry that makes this safe — and why it's deliberate

This module sits alongside a **first**, separate mechanism
(`compute_frequency_priority()`) that aggregates the pipeline's *own*
Stage 2b outcomes — not reviewer-confirmed ones — into a frequency
prior. That mechanism was found, on real production data, to be
dangerous: a 50-note production run gold-checked 7/7 of its
highest-confidence non-excluded picks as **wrong** (`DM`→"deep masseter"
instead of diabetes mellitus, `IVF`→"In Vitro Fertilization" instead of
IV fluids, and 5 more), and — worse — the mechanism was caught
**actively re-selecting its own earlier wrong guesses within the same
run**, exactly the circularity this document's calibrator section (§6)
also has to guard against. That mechanism was inverted from a block-list
to a strict, currently-**empty** allow-list as a direct result.

`mine_context_rules()` (this section) is **not** subject to that same
restriction, and the code's own comment is explicit about why: its
input is real, independent, human-confirmed ground truth from
`hitl_review_queue`, not the pipeline's own confident-but-possibly-wrong
guess. Aggregating the pipeline's own guesses formalizes whatever bias
already exists in them; aggregating a human's corrections is exactly the
kind of independent evidence that *can* correct a systematic bias rather
than reinforce it — including, eventually, for the exact
coronary-artery-segment abbreviations (`LAD`/`LCX`/`LMCA`/...) currently
hard-blocked elsewhere in the pipeline (see
`docs/ConsensusCalibrator_Technical_Reference.md`'s trap-gate section).
This mechanism is the **intended eventual path to relaxing those hard
traps for real**, once enough real review volume exists — which is
precisely why it does not pre-emptively exclude them the way the
frequency-priority mechanism does.

## 6. Repurposing mechanism 2 — confirmed resolutions as a calibrator trust signal

A second, structurally different way reviewed (and auto-confirmed) data
feeds back into the pipeline: `count_prior_confirmations()`
(`src/mollm_tier_calibrator.py:160-204`), one of the 16 input features
to the `ConsensusCalibrator` that decides whether a *split-vote* Stage 3
decision can still be promoted to an auto-write tier
(`TIER_1B_CALIBRATED_AUTO_VALIDATED` — full mechanism in
`docs/ConsensusCalibrator_Technical_Reference.md`).

```sql
-- half 1: how many times has this (entity text, concept) pairing already
-- reached an AUTO tier via the tier gate...
SELECT count(*) FROM mollm_tier_gate_decisions d
JOIN extracted_entities e ON e.entity_id = d.entity_id
JOIN normalized_entities n ON n.entity_id = d.entity_id
WHERE lower(trim(e.original_text)) = lower(trim(?))
  AND n.omop_concept_id = ?
  AND d.mollm_routing_decision IN ('AUTO_VALIDATED', 'AUTO_RESOLVED')

-- half 2: ...or been explicitly APPROVED/CORRECTED by a human reviewer
-- to this exact concept
SELECT count(*) FROM hitl_review_queue h
JOIN extracted_entities e ON e.entity_id = h.entity_id
WHERE lower(trim(e.original_text)) = lower(trim(?))
  AND h.corrected_concept_id = ?
  AND h.reviewer_decision IN ('APPROVED', 'CORRECTED')
```

The two halves are summed into one `prior_confirmation_count` feature —
"how many times has this same resolution already been confirmed,
one way or the other." Unlike §5's mechanism, this one deliberately
**includes** the pipeline's own AUTO-tier confirmations alongside human
ones, not just human ones — and the code is explicit that this is a
real, acknowledged risk, not an oversight: since real KG3 ingestion runs
`dry_run=True` throughout this pipeline today, this feature is currently
built from the *same* dry-run tier-gate decisions it might end up
reinforcing, not from independent live-graph evidence. The calibrator's
own held-out-split validation discipline (never train and score on the
same notes) is the safeguard against this specific feature encoding
circularity rather than real signal — an ablation run during that
calibrator's own development (dropping this feature entirely) confirmed
the model's real predictive signal comes from vote-consensus shape and
retrieval provenance, not from this feature, which is exactly the
honest, checked answer to "is this feature doing something real or just
echoing its own training data."

**How this repurposes reviewed data concretely**: once real
`APPROVED`/`CORRECTED` review volume exists (currently zero, per this
document's headline), every future entity sharing the same
`(entity text, concept)` pairing as an already-reviewed one gets a
nonzero `prior_confirmation_count` — direct, measurable evidence the
calibrator can use to promote a *future* split-vote decision on that
same pairing to an auto-write tier, without needing a human to re-review
an already-settled case every single time it recurs.

## 7. The read side — built ahead of having data to read

`src/kg3_query.py` is deliberately **read-only**, with no
ranking/scoring logic of its own — it answers "what is currently in
KG3," not "what should Stage 2b do about it" (that second question is
the two repurposing mechanisms above, which live in `src/`, not here).
Its own docstring states plainly why it exists *before* the feedback
loop it will eventually support: the real feedback mechanisms
(GLiNER prompt/search-space feedback, a CompGCN/TransE/RotatE
re-ranking layer — see `docs/KG_Embedding_Technical_Reference.md`
for what was actually built of that) need a meaningful volume of
accumulated `HUMAN_VERIFIED` corrections to mean anything, which does
not exist yet. Building the read interface now, against whatever the
write side actually produces, means it's ready the moment real data
exists rather than being designed blind alongside the write side later.

Three functions: `count_by_label()` (cheapest possible sanity check —
node counts by `:PatientObservation.label`), `get_observations_for_concept()`
(full provenance chain for one concept, for audit or a future
"how often has a human confirmed this" query), and
`get_accepted_triples()` (every fully-provenanced, `APPROVED`/`CORRECTED`
observation, optionally domain-filtered — the exact population a future
re-ranker or prompt-feedback mechanism would train or query against).

## 8. Regression safety net, built before there's anything to regress

`scripts/regression_check_kg3.py` snapshots gold-recall plus KG3 coverage
and flags regressions against the previous snapshot — reusing
`scripts/score_gold_recall.py`'s scoring directly rather than
reimplementing it (this codebase's repeated "two copies of one idea
drifting apart" discipline). Its own docstring is candid that its
central number **cannot move on its own today**: nothing currently feeds
KG3 signal back into Stage 2b, so there is nothing yet for this script
to actually detect a regression *from*. It exists so that the very first
change that *does* close that loop has something to diff against
immediately — the tool and the habit built ahead of the mechanism, not
bolted on afterward once something breaks silently.

## 9. Current real status (live-queried, not assumed)

| | Count |
|---|---|
| `hitl_review_queue` total cases | 19,103 |
| ...`PENDING` | 19,103 (100%) |
| ...`APPROVED` / `CORRECTED` / `REJECTED` | 0 / 0 / 0 |
| `hitl_review_queue.ingested_at IS NOT NULL` (real KG3 writes via the reviewed path) | 0 |
| `abbreviation_context_rules` rows (§5's mined rules) | 0 |

Every real KG3 write anywhere in this pipeline today, via either path
(§2), is `dry_run=True` or simply hasn't happened — there is no live
Memgraph content this document can report numbers from yet. This is
stated as plainly here as everywhere else in this project's
documentation: the machinery in §2-§7 is built, tested (§10), and
verified against synthetic/dry-run data; its real-world improvement to
the pipeline (§5's rule mining, §6's calibrator feature) is currently
**zero in practice**, pending actual reviewer throughput — a scale
problem, not a code problem.

## 10. Testing & reproducibility

- **`tests/test_kg3_ingestion.py`** — 6 checks, all passing
  (`python3 -m pytest tests/test_kg3_ingestion.py -q -s`).
- **A real, confirmed gap**: no test file anywhere in `tests/` references
  `src/hitl_queue.py`'s functions (`load_ingestible_cases`,
  `submit_review`, `enqueue_pending_cases`, ...) — confirmed via direct
  search. The review-queue lifecycle itself has no dedicated unit-test
  coverage today, unlike the ingestion module it feeds.
- **To exercise the batch driver**:
  `python3 scripts/run_kg3_ingestion.py --dry-run` — resolves and prints
  every ingestible case without touching Memgraph or marking anything
  ingested, safe to run anytime against the real DB.
- **To check regression-tracking machinery**:
  `python3 scripts/regression_check_kg3.py` — read-only, safe to run
  anytime; will currently report an empty/first-run baseline given §9's
  numbers.

## 11. Simulated-review impact experiment — fresh-5 held-out notes

**Context and method, stated upfront.** §9 established that real review
throughput is zero, so §5-§6's mechanisms have no *measured* impact yet
— only a description of how they're supposed to work. This section
closes that gap with a genuine, one-off computational experiment: since
19,103 real reviewer clicks aren't available, the queue was
**gold-based auto-approved** instead — for every queued case, the
entity's already-known gold SNOMED annotation (from this project's own
evaluation corpus) stands in for a reviewer's verdict: a matching
suggestion is treated as `APPROVED`, a mismatch is treated as
`CORRECTED` to the gold concept, and anything without a clean,
gradable gold match is left `PENDING` rather than guessed at. This is
**not real HITL data** — it substitutes gold-corpus lookup for genuine
clinical judgment — and is reported here as exactly that: a controlled
simulation of "what if the queue were fully reviewed," not a claim
about real reviewer behavior. Run entirely against a **throwaway copy**
of the production DB (`kg3_impact_experiment.duckdb`, in the session
scratchpad, never committed) — the real production `hitl_review_queue`
(§9's numbers) was never touched by this experiment and remains 100%
`PENDING`.

**Notes measured**: the first 5 of `ui/components/fresh10_notes.py`'s
`FRESH10_NOTE_IDS` (`13538696-DS-11`, `19895550-DS-7`,
`11516225-DS-20`, `14652764-DS-17`, `12298181-DS-9`), held out from
approval so nothing about their own outcome could leak into the pool
that trains the mechanisms being measured. The approval pool was every
*other* queued case: 139 notes, 18,978 cases.

### 11.1 Approval pass — real numbers

| Outcome | Count | % of pool |
|---|---|---|
| `APPROVED` (suggestion matched gold) | 5,889 | 31.0% |
| `CORRECTED` (mismatch, gold concept resolvable) | 4,881 | 25.7% |
| Left `PENDING` — no clean single gold-span overlap | 8,043 | 42.4% |
| Left `PENDING` — gold SNOMED code not a resolvable standard OMOP concept | 165 | 0.9% |
| **Total reviewed** | **10,770** | **56.7%** |

The 42.4% "no clean gold overlap" figure is not a flaw in the
experiment — it reflects the same clean-span-only grading discipline
every other measurement in this project's documentation uses (a
prediction span must cleanly cover exactly one gold annotation to be
graded at all), applied honestly here rather than loosened to inflate
the approved count.

### 11.2 Mechanism 1 (mined context rules) — real output

`mine_context_rules()` (§5) ran against the newly-reviewed pool and
wrote **27 rules**, covering exactly **3 distinct abbreviations**:
`lad` (25 rules, split across two meanings — concept `315085`
"Lymphadenopathy" and concept `4245039`, the coronary-artery-segment
sense), `cta` (1 rule), `ttp` (1 rule). The `lad` result is a
genuine, substantive finding on its own: 21 independent confirmed
examples of the word `"no"` immediately preceding `lad` predicting the
Lymphadenopathy sense (e.g. "no lymphadenopathy" in a physical-exam
line) — this is real, mined evidence bearing directly on the
coronary-artery-segment systematic-bias problem §5.4 already discusses,
the population this mechanism was specifically built to eventually
help correct.

**Fresh-5 Stage 1 impact: zero.** Of fresh5's 22 ambiguous-expansion
entities, **none** use `lad`, `cta`, or `ttp` — a real, checked null
result, not an assumption. The mined rules are real and substantive,
they simply don't happen to touch this particular 5-note sample.

### 11.3 Mechanism 2 (calibrator confirmation count) — real output, with a real complication found along the way

Re-scoring fresh5's non-auto tier-gate decisions against the
now-populated `hitl_review_queue` surfaced a genuine, unplanned finding
**before** any impact number could even be computed: loading the
production calibrator with `scoring_note_ids=FRESH5` triggered its own
leakage guard —

```
! leakage: trained on 3 of the note(s) about to be scored
  (11516225-DS-20, 12298181-DS-9, 19895550-DS-7).
  refusing to use it; falling back to existing routing.
```

**3 of the 5 "fresh" notes used for this experiment were actually in
the production `ConsensusCalibrator`'s own training set** — contradicting
`ui/components/fresh10_notes.py`'s own docstring claim that this note
set is "genuinely held-out." That claim is accurate with respect to the
SNOMED near-duplicate retrieval fix and the KGE evaluation (what it was
originally written for), but **not** with respect to this specific
calibrator artifact. This is reported here, plainly, as a real
contamination finding this experiment surfaced — not fixed or corrected
elsewhere, per this section's scope.

Given that, results below are reported **per-note contamination
status**, never blended:

| Entity | Note | Status | Prior tier | `prior_confirmation_count` (new) | Calibrated score | Result |
|---|---|---|---|---|---|---|
| `Abd` | `13538696-DS-11` | clean | (tier=`None`, an error/edge-case decision) | 0 | 0.6646 | Below `CALIBRATED_AUTO_THRESHOLD=0.72` — not promoted |
| `VS` | `19895550-DS-7` | **contaminated** | `TIER_2_AUTO_RESOLVED` | 44 | 0.1786 | Not a clean result (calibrator trained on this note) — but note the score is low regardless |
| `RRR` | `14652764-DS-17` | clean | `TIER_2_AUTO_RESOLVED` | — | — | `_is_short_alphanumeric_code()` hard trap fired; calibrator never consulted |

Of fresh5's 125 total tier-gate decisions, only **3** were even
calibrator-eligible (already-auto tiers and split-vote decisions with
no `final_candidate_index` — i.e. no single candidate reached plurality
among the models — are structurally never consulted). **Net result on
the clean population: 0 of 0 eligible clean entities newly promoted**
(`Abd` is the only clean, calibrator-consulted case, and its score
stayed below threshold).

One number worth flagging on its own merit despite the contamination:
`VS`'s `prior_confirmation_count` reached **44** from a single
approval pass — real evidence the mechanism accumulates volume exactly
as designed — yet its calibrated score was still low (0.18). This is
consistent with, not contradicting, this project's own earlier ablation
finding (§6): `prior_confirmation_count` is a real but **secondary**
signal in the fitted model, not one that can push a score to promotion
on its own regardless of magnitude.

### 11.4 Honest summary of this experiment

**Measured impact of KG3's repurposing mechanisms on this specific
fresh-5 note set, after gold-based-simulated review of 56.7% of the
queue: zero routing changes.** Not because either mechanism is broken —
§11.1-§11.3 show both firing exactly as designed on real data (27 real
rules mined, a real 44-count confirmation signal accumulated) — but
because of three compounding, honestly-reported reasons specific to
this sample: (1) the abbreviations that reached mining volume don't
overlap this note set's ambiguous entities, (2) most of this note set's
split-vote entities structurally never reach the calibrator at all
(no plurality candidate), and (3) the one genuinely clean,
calibrator-consulted entity scored meaningfully below the promotion
threshold. A larger approval pass, a different 5-note sample, or simply
more accumulated volume over time could all plausibly change this —
this experiment measures one honest data point, not a ceiling on what
the mechanisms can do.

**Reproducing this experiment**: the two scripts used
(`kg3_impact_experiment.py`, `kg3_impact_experiment_part2.py`) were
written as one-off session scratchpad scripts, not committed to this
repository — they are not part of the maintained codebase. Reproducing
this result requires a scratch copy of the production DB, gold
annotations and raw note text (both PhysioNet-restricted, see the
README's Data Access section), and the same gold-based-approval
methodology described in this section.

## 12. Honest limitations

- **Zero real review throughput to date** (§9) — real reviewer clicks
  remain at zero. §11's numbers come from a gold-based *simulation* of
  review, not genuine human judgment — a real, useful data point, but
  not the same evidence real HITL volume would provide.
- **§11's simulated experiment measured zero routing changes** on the
  specific fresh-5 note sample tested — both mechanisms fired
  correctly on real data (27 mined rules, a real 44-count confirmation
  signal), but neither happened to change an outcome for these
  particular 5 notes; §11.4 explains why in detail. Do not read this as
  evidence the mechanisms don't work — the sample is small and the null
  result is well-explained, not mysterious.
- **A real note-set contamination was found while running §11's
  experiment**: 3 of the 5 "fresh10" notes are actually inside the
  production `ConsensusCalibrator`'s own training set, despite
  `ui/components/fresh10_notes.py`'s docstring describing the set as
  "genuinely held-out." That claim holds for what it was originally
  written about (the SNOMED near-duplicate fix, the KGE evaluation) but
  not for this calibrator specifically. Flagged here, not corrected
  elsewhere, per this document's scope.
- **`mine_context_rules()` needs real volume, not just any review
  activity** — `MIN_CONTEXT_RULE_SUPPORT=5` independent examples *per
  abbreviation* before any rule is written at all; a handful of reviewed
  cases spread across many different abbreviations would still yield
  zero rules for a long time.
- **`count_prior_confirmations()`'s circularity risk is real, not fully
  closed** (§6) — mitigated by the calibrator's held-out-split
  discipline and confirmed low-importance via ablation, but the feature
  itself is still built partly from the pipeline's own dry-run
  decisions, not exclusively from independent human confirmation.
- **The HITL review UI's throughput is the actual bottleneck** — nothing
  in §2-§8 is blocked on more engineering; it is blocked on 19,103
  queued cases actually being reviewed by a human, which this document's
  mechanisms cannot accelerate on their own.
- **No dedicated test coverage for `src/hitl_queue.py`** (§10) — a real,
  stated gap, not glossed over.
