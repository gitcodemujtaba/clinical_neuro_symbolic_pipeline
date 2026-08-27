# Phase 5/6 Build, Closeout Items, and Corpus-Scale Validation

This continues `docs/2026-08-16_Shadow_Run_Precision_At_Scale.md` (allergy
domain-restriction fix) and `docs/2026-08-15_Stage4_Stage5_Build.md` (HITL
queue / KG3 ingestion foundations). It covers the session stretch that
closed the three carried-over items from Phase 4, built Phase 5 (sliding-
window chunking) and Phase 6 build-order steps 1-2 (`ConsensusCalibrator`),
and ran the first corpus-scale validation of acronym escalation plus a real
end-to-end pipeline check. Every number below is quoted from an actual run
in this session, not estimated.

Commits covered: `6e8e0ff`, `ed173b4`, `93207d4`, `f4e370e`, `99f91be`.

---

## 1. Cache-entrenchment fix (`src/acronym_escalation.py`, commit `6e8e0ff`)

**Why.** Phase 4's corpus-scale build-out surfaced a real bug during its own
live testing: the `acronym_priors` cache upserts `hit_count += 1` whenever
an MoLLM-sourced resolution's search subsequently reaches Tier 1/2/3 —
"only count a success," per the original design. But "reached a tier" and
"reached the *correct* concept" are different things. One case (`PDA` in a
coronary-stenosis context) was confidently resolved wrong by all 3 models,
still cleared Tier 1/2/3 cleanly, and would have been cached and reused for
every future `PDA` mention in a similar context with zero further model
calls to ever reconsider it — a single wrong LLM answer entrenching itself
permanently.

**Fix.** `MIN_CACHE_HIT_COUNT = 2`. `lookup_acronym_prior()`'s SQL gained a
`hit_count >= ?` filter, so a cache row is only trusted (served without a
model call) after it has been independently confirmed twice, not once. The
module docstring's "CACHE-ENTRENCHMENT RISK" section was updated to
"PARTIALLY CLOSED" — this raises the bar from "any single wrong resolution
entrenches" to "must be wrong twice in a row," which is a real mitigation
but explicitly not protection against a *systematic* model bias that would
reliably reproduce the same wrong answer on repeat encounters (see §6 —
this is exactly what happened with LAD/NAD at corpus scale).

**Test evidence.** `tests/test_acronym_escalation.py` rewritten: a single
upsert of a triple is asserted NOT yet a cache hit; a second upsert of the
same triple IS; an explicit "hit_count=1 not trusted" / "hit_count=2
trusted" pair of checks; "cache hit skips model call" now upserts twice
before asserting zero calls.

---

## 2. Lab Value Suffix Fallback optimization (`src/normalization/orchestrator.py`, commit `93207d4`)

**Why.** This was explicitly deferred earlier in the project ("Do not
optimize the Lab Test fallback right now") because the fallback loop calls
the full, expensive `normalize_entity()` (SapBERT embedding + Tier 1-4
cascade) once per candidate string produced by `strip_lab_value_suffix()`,
for every `Lab Test`-labeled entity — a real, measured performance sink.
The user explicitly reversed the deferral this session ("do it") once it
became a blocker for get through the corpus faster.

**Fix.** Added a cheap `_lookup_tier12()` pre-check (pure SQL, no SapBERT
call) across all suffix-stripped candidates (both `expanded_text` and
`orig_text` forms) *before* paying for `normalize_entity()`. If any
candidate clears Tier 1/2 via the pre-check, only that one candidate gets a
full `normalize_entity()` call. The original exhaustive loop (calling
`normalize_entity()` on every candidate) is preserved unchanged as a
fallback for when nothing clears Tier 1/2 via the cheap pre-check — needed
for the "keep the best near-miss" Tier 3/4 logic that depends on comparing
multiple full resolutions.

**Test evidence (live, not synthetic).** `WBC-7` → "White blood cell count"
in 3.65s, one SapBERT call (down from N calls, one per candidate).
`RBC-3` → "Red blood cell count" in 2.03s, one SapBERT call. Both previously
paid for a `normalize_entity()` call per candidate regardless of whether an
early one already matched cleanly.

---

## 3. Brand-to-generic allergy fallback (`src/normalization/orchestrator.py`, commit `93207d4`)

**Why.** `docs/2026-08-16_Shadow_Run_Precision_At_Scale.md`'s open items
list "7/16 allergy entities still get zero candidates" — brand and
combination-drug names never match the synthesized "Allergy to {text}"
exact-match pattern, since RxNorm brand names aren't SNOMED allergy
concepts.

**Fix.** New `_brand_to_generic_names()` wraps the already-existing 3-hop
KG walk `_alias_expand_brand_to_generic()` (`src/normalization/tier_retrieval.py`,
originally built for the "Lasix problem") to return generic concept *names*
instead of just IDs. `_apply_allergy_nonstandard_exact_override()` gained a
`drug_text` parameter: on a direct exact-match miss, it now retries against
every brand→generic name found, tagged with a distinct
`matched_via="allergy_nonstandard_exact_brand_to_generic"` so the source is
auditable separately from a direct match.

**Test evidence (live).** `Elavil` → "Allergy to amitriptyline" (via
brand-to-generic). `Reglan` → "Allergy to metoclopramide" (via
brand-to-generic). `Spiriva` stayed `Unmapped` — it isn't an exact RxNorm
Brand Name concept-class match, so the KG walk finds nothing to expand.
Documented honestly as "closes one real gap, not all of them" — the
combination-drug-name portion of the original 7/16 gap is untouched.

---

## 4. Phase 5 — sliding-window GLiNER chunking (`src/entity_extraction.py`, commit `ed173b4`)

**Why.** No chunking logic existed anywhere in the repo; GLiNER ran on the
whole note in a single `predict_entities()` call. Confirmed live:
`model.config.max_len = 2048` word-tokens is GLiNER-BioMed's real ceiling,
tokenized via `model.data_processor.words_splitter()` — notes longer than
that silently truncate, and a `possibly_truncated` flag already existed
from an earlier session to *detect* this but nothing acted on it.

**Fix.** `_build_chunks(sentences, words, budget=1800, overlap=128)`: pure
function, sentence-boundary-snapped sliding windows (never splits an entity
or separates a negation cue from its finding), covers the whole note with
overlap between consecutive windows. `_extract_entities_chunked()`: runs
`predict_entities()` per chunk, remaps local offsets to global, merges
exact-match duplicates keeping the higher score. The word-count check
(cheap, no model inference) happens *before* deciding chunked-vs-single, so
notes already known to need chunking never pay for a wasted full-text
forward pass first.

**Test evidence (live, on the exact note the original truncation-detection
comment cited).** Note `11532659-DS-11`, 24,858 chars: single-call
extraction → 124 entities, `truncated=True` (confirmed). Chunked → **399
entities, `possibly_truncated=False`** — recovering 282 real clinical
entities the single pass silently dropped, including pleural effusions,
right-upper-lobe pneumonia, a full lab panel, and several procedures. Cost:
85.6s chunked vs 35.8s single-call on this one note — a real latency
tradeoff, well inside the project's 2-5 min/note budget. 12 new unit tests
(`tests/test_entity_chunking.py`), full suite passing.

---

## 5. Phase 6 build-order steps 1-2 — `ConsensusCalibrator` (`src/mollm_tier_calibrator.py`, commits `f4e370e`, `99f91be`)

**Why now.** `mollm_tier_gate_decisions` already holds hundreds of real
(dry-run) graded decisions from this session. The current Tier 1-5 table is
a **blunt rule**: 3/3 unanimous vote = trust, *any* split = HITL. The
corpus-scale numbers measured this session (§6-7 below) show what that
costs — `TIER_4_ENSEMBLE_SPLIT` is roughly 44.8% of all decisions
(175/391, from the plan's own running tally), and this session's own
allergy-domain work already proved a 2-1 split isn't uniformly wrong:
several splits had a demonstrably correct top candidate, just not unanimous
agreement.

**Explicit design constraint.** The user gave two rounds of explicit
correction on this: first, "don't look at the old [superseded]
calibrator's mechanics while designing this," then a stronger version —
"write calibrator from scratch without any reference to old one." A
near-identical mechanism (`MoLLMCalibrator` in the now-superseded
`src/mollm_calibrator.py`) already existed for the old gate, and the first
draft of the new module — even after removing every literal textual
reference — still implicitly mirrored the old module's method names and
docstring framing. The final version was fully rewritten: fresh docstrings
framed purely around the *current* gate's consensus-disagreement problem,
verified via `grep` that zero `MoLLMCalibrator`/`mollm_ensemble`/
`mollm_calibrator` references remain anywhere in the file.

**What it is.** `ConsensusCalibrator`: a `LogisticRegression(class_weight=
"balanced")` scoring `P(correct)` from 16 features in 3 groups — (a) vote
consensus shape (fraction supported/re-ranked-same-target/none-correct,
mean/min/spread of per-model logprob confidence, computed only over
*usable* non-degenerate votes), (b) retrieval/provenance (match tier, top
candidate similarity, ambiguity/domain-conflict flags, and fragile-fallback
flags parsed from `normalized_from` — did this resolution come through the
Lab Value suffix-stripped path or the acronym-escalation path, both of
which carry measurably lower trust than a direct match), and (c) a new
`prior_confirmation_count` feature.

**The "reference already-approved labels" requirement.** The user
explicitly asked that this calibrator "can also reference our already
approved labels which becomes part of KG." Confirmed via direct code
research first (per this project's own standing discipline of verifying
before building — see project memory) that KG3 writes are *all* still
`dry_run=True`, and that no function anywhere in the codebase — DuckDB or
Memgraph — already computes "how many times has this resolution been
confirmed before." `count_prior_confirmations()` is genuinely new: it sums
confirmed-outcome counts from `mollm_tier_gate_decisions`
(`AUTO_VALIDATED`/`AUTO_RESOLVED`) and `hitl_review_queue`
(`APPROVED`/`CORRECTED`), scoped to the same `(entity_text, concept_id)`
pairing, capped at 10 and scaled to `[0,1]` — the same pattern as
`acronym_priors`' `hit_count`, generalized beyond abbreviations to every
resolution. Because real KG3 has no data yet, this is necessarily sourced
from DuckDB's own shadow-decision history today, not real graph traversal
— documented as a known limitation, not hidden.

**Integration — additive, never a weakening.** `route_tier()` gained
optional `calibrator=None, conn=None` params. Confirmed via the full
existing test suite that passing neither reproduces prior behavior
byte-for-byte (a genuine no-op). The calibrator is only ever consulted for
entities that already fail every hard Tier 1/2/3 rule, and — a specific
guard tested directly — is never invoked at all when the plurality verdict
is `NONE_CORRECT` (no candidate to promote). A score
`>= CALIBRATED_AUTO_THRESHOLD` (0.90, explicitly marked CALIBRATION-PENDING
— not yet fit or validated) routes to a new, distinctly-labeled
`TIER_1B_CALIBRATED_AUTO_VALIDATED`, never silently merged into a genuine
unanimous Tier 1 in any downstream count. 8 new tests cover: default no-op,
high-score promotion (checked against `tier`/`mollm_routing_decision`/
`final_candidate_index`/`routing_basis`), low-score staying Tier 4, either
param alone still being a no-op, and the `NONE_CORRECT` guard via a fake
calibrator whose call log proves it's never invoked for that shape.

**Status.** Untrained — build-order steps 3-4 (a training-data pipeline via
`evaluation/tier_gate_cal_eval.py`, then fit + held-out-split validation)
are explicitly blocked on DB write-lock access, since the overnight corpus
run (§8) holds it.

---

## 6. Corpus-scale acronym-escalation grading — the real number

**Prior state.** `ACRONYM_ESCALATION_ENABLED` had only been validated on a
small sample: 15/15 correct on one broad sample, then 13/14 on a second
end-to-end check (one deliberate miss: "PDA," logged as a genuine model
reasoning/verdict-mismatch case). Both readings were optimistic small
samples, not a corpus-scale measurement.

**What was run.** All 31 usable test notes, `CNSP_ACRONYM_ESCALATION=1`,
Stage 1→2b. Killed and restarted twice mid-run: once because the process
had the *old* orchestrator code loaded in memory before the Lab Value/
brand-generic fixes existed (Python doesn't hot-reload — caught directly
because the user asked "should we not load new orchestration," which
prompted a `ps`/thread check confirming the stale in-memory state), and
once per the user's explicit "finish the background task" request, after
5/21 remaining notes had graded.

**Result, first 10 completed notes graded:** 182 ambiguous entities, 61
escalated by the model, 35 gradable against gold. **12/35 = 34.3%
precision** — well below both earlier small-sample reads (15/15, 13/14). A
second grading pass over a slightly different completed subset read 36.1%
(22/61) — same order of magnitude, same conclusion.

**Root cause, confirmed via direct SNOMED lookup, not assumed.** The
"PDA" reasoning/verdict-mismatch failure mode generalizes to other common
abbreviations, all sharing the same shape: a small local LLM's prior toward
the more textbook-famous meaning overrides correct in-context evidence.
"LAD" consistently resolved to "left anterior descending artery" even in a
note where gold's own SNOMED code (`30746006`) is "Lymphadenopathy" — a
completely different clinical concept. "NAD" consistently resolved to the
biochemistry reading (NAD/NADH ratio) instead of "no acute distress" across
3 separate notes.

**Decision.** `ACRONYM_ESCALATION_ENABLED` stays off. This corpus-scale
finding, not the earlier optimistic small samples, is the number this
decision is based on. This is *not* a production regression — the feature
was never on by default — it's a legitimate downward revision of a feature
whose real-world precision turned out much lower than early testing
suggested. Fixing it needs a mechanism change (e.g. a reasoning/verdict-
mismatch guard that catches the model contradicting its own stated
reasoning), not more data volume — more notes alone would keep reproducing
the same systematic bias, not average it away.

---

## 7. End-to-end full-pipeline validation (2 notes)

**Why.** Before committing to a large overnight run, the accumulated day's
fixes (allergy domain restriction, Lab Value optimization, brand-to-generic
fallback, acronym escalation off, chunking) needed to be proven working
*together* in a real, unmocked, full pipeline pass — not just in isolated
unit/smoke tests.

**What was run.** `scripts/test_pipeline_e2e.py` against 3 requested note
IDs; one (`10000032-DS-21`) wasn't present in `gold_notes.csv` (silently
skipped by the script, printed a warning), so the run proceeded on the 2
that resolved: `12314513-DS-16`, `12545016-DS-17`. Full Stage 1→2a→2b, then
Stage 3 tier-gate, against freshly-cleared `mollm_tier_gate_decisions`/
`hitl_review_queue` rows for these two notes (57 stale rows from an earlier
session pass were deleted first, for a clean read).

**Result.** 226 entities processed, 0 errors. Stage 3 grading: 22/30
gradable = **73.3% precision**. Specifically re-confirmed NSAIDS resolves
correctly end-to-end in the real production flow (not just in the isolated
allergy-domain-fix testing from `docs/2026-08-16_Shadow_Run_Precision_At_Scale.md`).

---

## 8. Overnight run

**Scope decision.** Given a choice between validating on just the 31-note
test corpus vs. scaling up into the larger 272-note gold corpus, the
smaller, already-known-good 31-note scope was chosen — it directly answers
the still-open "what's the real corpus-scale acronym-escalation number"
question from §6 (interrupted mid-run twice) and produces enough freshly-
graded `mollm_tier_gate_decisions` volume to eventually fit the
`ConsensusCalibrator` from §5, without the added runtime and unknowns of a
much larger, less-vetted corpus on the first unattended run.

**What's running.** Stage 1→2b (`CNSP_ACRONYM_ESCALATION=1`) then Stage 3
tier-gate, chained sequentially, across all 31 usable test notes, writing a
`DONE` marker on completion. This holds DuckDB's single-writer lock for its
full duration — no other DB read/write is possible until it finishes or is
killed, which is why this document and the plan-file update were written
entirely from numbers already gathered and quoted verbatim during the
session, without querying the DB again.

---

## Decisions and their causes — summary table

| Decision | Caused by (actual test/data) |
|---|---|
| `MIN_CACHE_HIT_COUNT=2` added to acronym cache | Live PDA case: one wrong resolution cleared Tier 1/2/3 and would have entrenched in cache permanently |
| Lab Value fallback given a cheap Tier1/2 pre-check | WBC-7 3.65s / RBC-3 2.03s, one SapBERT call each (was N calls) |
| Brand-to-generic retry added to allergy exact-match | Elavil→amitriptyline, Reglan→metoclopramide now resolve; Spiriva still doesn't (no RxNorm Brand Name match) |
| Sliding-window chunking built for notes >budget | Note 11532659-DS-11: 124→399 entities (282 recovered), single-call `truncated=True` confirmed |
| `ConsensusCalibrator` built fully from scratch | Explicit repeated user instruction; verified via grep zero references to superseded module remain |
| `ACRONYM_ESCALATION_ENABLED` stays off | Corpus-scale grading: 34.3%/36.1% precision, well below small-sample 15/15 and 13/14; root cause is a systematic, reproducible model bias (LAD/NAD), not noise |
| Overnight run scoped to 31 notes, not the 272-note corpus | Directly resolves the still-open §6 question; produces enough graded volume for Phase 6 steps 3-4 without first-unattended-run risk of a much larger corpus |
| `TIER_1B_CALIBRATED_AUTO_VALIDATED` kept distinct from Tier 1 | So a calibrator-assisted decision is always separable from a genuine unanimous vote in every downstream measurement, never silently inflating Tier 1's reported precision |
