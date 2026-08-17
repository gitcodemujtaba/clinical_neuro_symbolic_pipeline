# Pipeline Walkthrough: One Real Entity Per Tier, All Stages

Every example below is a real entity from this session's actual runs (the
overnight 31-note corpus run, the 5-note and 20-note calibrator validation
runs) — not constructed. Where a specific field value (vote counts,
calibrator score, candidate list, routing_basis text) is quoted, it was
directly observed via a grading query or the pipeline's own log during this
session, not reconstructed. Where a stage's *mechanism* is described but the
exact field values for that specific entity weren't captured in this
session's transcript (Stage 1/2a's precise `section_name`/`local_context`
for some examples), that's noted explicitly rather than invented.

## The four stages, in one paragraph each

**Stage 1 — extraction** (`src/entity_extraction.py`). GLiNER-BioMed reads
the raw note text and produces spans labeled `Condition`, `Symptom`,
`Medication`, `Procedure`, `Anatomy`, or `Lab Test` (`Qualifier` is a
relabel-only target used later, never a GLiNER output), each with a
confidence score and a character span. `src/assertion.py` then tags each
span's assertion status (`PRESENT`, `ABSENT`, `CONDITIONAL`, `ALLERGY`, ...)
and experiencer (`PATIENT`/`FAMILY`) via a ConText-style negation/hedging
scan. Notes longer than GLiNER's real word-token ceiling get sliding-window
chunked (Phase 5) so nothing past the ceiling silently truncates.

**Stage 2a — abbreviation expansion** (`src/preprocessing.py`). Every
extracted span is checked against the `kg2a_abbreviations` dictionary. An
unambiguous abbreviation just expands. An abbreviation with more than one
dictionary meaning gets `expansion_ambiguous=TRUE` and a stored
`candidate_expansions` list — a *prior* pick is still made (via a tiebreak
chain), but the entity carries the ambiguity flag forward rather than
hiding it.

**Stage 2b — normalization** (`src/normalization/orchestrator.py`'s
`normalize_entity()`). A Tier 1→4 cascade against the OMOP/SNOMED
vocabulary: Tier 1 exact lexical match, Tier 2 synonym match, Tier 3
SapBERT dense semantic search (domain-restricted per `GLINER_LABEL_TO_DOMAIN`,
floor `TIER3_SIMILARITY_FLOOR=0.72`), Tier 4 fuzzy edit-distance as a last
resort. Returns a ranked `candidates` list plus the chosen top candidate,
`match_tier`, and an `is_ambiguous`/`domain_conflict` flag pair.

**Stage 3 — the tier gate** (`src/mollm_tier_gate.py`'s `route_tier()`).
Three free pre-checks first (no model call): a standalone `Qualifier` span
routes straight to HITL, a graph-verified brand-alias hit routes straight to
AUTO, a candidate pool with nothing above the similarity floor (or an
unresolved ambiguous abbreviation) routes straight to HITL. Everything else
goes to the full **two-step ensemble**: qwen2.5:3b, llama3.2:3b, and
phi4-mini each independently run **Step A** ("state the clinical meaning of
this span from context alone, with *no* candidate list shown") then **Step
B** ("given that independently-stated meaning, does candidate #1 match? If
not, candidate #2? ...", stopping at the first accept). Three verdicts come
back — `SUPPORTED_1`, `RE_RANK_TO_CANDIDATE_N`, `NONE_CORRECT`, or `ERROR`
— and the Tier 1-5 table below decides what happens next.

---

## TIER_1_AUTO_VALIDATED — 3/3 unanimous, high confidence

**Trigger** (`route_tier()`): all three models independently return
`SUPPORTED_1` (candidate #1 matches), and the mean of their three
verdict-token confidences is `>= TIER1_CONFIDENCE_FLOOR` (0.70).

**Real example**: `warfarin` (Medication), overnight corpus run.
- **Stage 1**: extracted as `Medication`, text as written `"warfarin"`.
- **Stage 2b**: Tier 1 exact lexical match — `athena_concept.concept_name`
  has a literal `"warfarin"` row in the `Drug`/`Ingredient` vocabulary.
  `match_tier = "1 (Exact)"`, `similarity_score = 1.0`, one candidate.
- **Stage 3**: `tier3_fast_path()` doesn't fire here (no brand-alias
  candidate to bypass on), so the full ensemble runs. All three models'
  Step A independently describe the span as the anticoagulant medication
  warfarin; Step B checks candidate #1 (`warfarin`, `Drug`/`Ingredient`)
  against that meaning and all three accept.
- **Outcome**: `routing_basis = "3/3 unanimous SUPPORTED_1, composite_confidence
  <value>"`. `mollm_routing_decision = AUTO_VALIDATED`. Confirmed correct
  against gold in the fresh-note grading pass.

---

## TIER_2_AUTO_RESOLVED — 3/3 unanimous, but not candidate #1

**Trigger**: all three models return the *same* `RE_RANK_TO_CANDIDATE_N`
(N > 1) — unanimous agreement that the retrieval-ranked #1 candidate is
wrong, but a specific lower-ranked one is right.

**Real example**: `papilledema` (Symptom), note `16991646-DS-11`, overnight
corpus run.
- **Stage 2b**: multiple SNOMED candidates existed for the term; the
  retrieval ranking's #1 pick was not what the ensemble settled on.
- **Stage 3**: all three models' Step B rejected candidate #1, moved to the
  next candidate, and unanimously accepted `"Papilledema - optic disc edema
  due to raised intracranial pressure"` (candidate #2) as the correct
  concept — a genuine re-rank, not a retrieval failure.
- **Outcome**: `tier = TIER_2_AUTO_RESOLVED`, `final_candidate_index = 2`,
  `mollm_routing_decision = AUTO_RESOLVED`. Graded against gold: this
  specific case turned out to be a **miss** (gold wanted the more specific
  "Optic disc edema" reading) — included here deliberately to show the
  mechanism honestly, not cherry-picked for a correct outcome. `TIER_2` is
  a genuinely small population (1 decision in the 20-note proof run so far,
  8 across the full 31-note overnight corpus) — too small to have a
  separately reliable precision number of its own yet.

---

## TIER_3_AUTO_VALIDATED — the free fast path

**Trigger** (`tier3_fast_path()`, runs *before* any model call): exactly one
candidate has `match_basis == "verified_brand_alias"` (a graph-verified
brand→generic KG walk, not a fuzzy string match), the entity's abbreviation
expansion wasn't ambiguous, and its assertion status is `PRESENT` (or
unset). All three conditions together are the "zero contradiction cues"
the fast path requires — skips the two-step ensemble entirely.

**Real example**: `Coumadin` (Medication), note `10860165-DS-24`.
- **Stage 2b**: `Coumadin` is a brand name, not itself a standard SNOMED/RxNorm
  concept. `_alias_expand_brand_to_generic()` walks the KG (brand →
  ingredient) and surfaces `warfarin` as a verified alias — the *only*
  candidate, tagged `match_basis="verified_brand_alias"`.
- **Stage 3**: `tier3_fast_path()` fires immediately: `routing_basis =
  "Tier 3 fast path: candidate [1] (warfarin) is a graph-verified brand
  alias, the sole such hit, with no ambiguous expansion or non-PRESENT
  assertion -- skipped the two-step ensemble entirely."` Zero model calls.
- **Outcome**: `AUTO_VALIDATED`, `final_candidate_index = 1`. Note: graded
  against gold this specific SNOMED code choice didn't match gold's exact
  concept (a real, separately-tracked drug-vs-therapy-concept granularity
  gap) — the *brand identification* (Coumadin = warfarin) was correct; the
  mechanism worked as designed even where the final code choice needs
  more work elsewhere in the stack.

---

## TIER_1B_CALIBRATED_AUTO_VALIDATED — the calibrator's rescue

**Trigger**: the vote was **not** unanimous, but a fitted
`ConsensusCalibrator` (`src/mollm_tier_calibrator.py`) scores
`P(top candidate correct) >= CALIBRATED_AUTO_THRESHOLD` (currently 0.72),
from 16 features spanning vote-consensus shape, retrieval provenance, and
prior-confirmation history — and neither hard safety trap (coronary-segment
or short-alphanumeric-code) fires first.

**Real example**: `chest pain` (Symptom), note `19895550-DS-7`, 5-note fresh
validation run (a note outside the calibrator's own training set).
- **Stage 3 ensemble**: votes came back `{'SUPPORTED_1': 2, 'NONE_CORRECT': 1}`
  — two models independently confirmed candidate #1 (`"Chest pain"`), one
  rejected every candidate. Not unanimous, so the hard Tier 1/2 rules don't
  fire.
- **Calibrator consultation**: `count_prior_confirmations()` looks up how
  many times this exact (text, concept) pairing has already been confirmed
  elsewhere in `mollm_tier_gate_decisions`/`hitl_review_queue`; combined
  with the 2-1 vote shape and retrieval provenance, `ConsensusCalibrator.score()`
  returned **0.90479** — well above threshold.
- **Outcome**: `routing_basis = "non-unanimous verdicts {'SUPPORTED_1': 2,
  'NONE_CORRECT': 1}, but ConsensusCalibrator scored 0.90479 >= 0.72
  (prior_confirmation_count=...)"`. `tier = TIER_1B_CALIBRATED_AUTO_VALIDATED`
  — deliberately never merged into `TIER_1_AUTO_VALIDATED` in any count, so
  a calibrator-assisted decision stays auditable and separable. Confirmed
  correct against gold.

---

## TIER_4_ENSEMBLE_SPLIT — split vote, not rescued

**Trigger**: non-unanimous, and either no calibrator/conn was supplied, the
calibrator scored below threshold, or a hard safety trap fired first.

**Real example**: `LCX` (Anatomy), note `10124346-DS-4`, held-out validation
set.
- **Stage 3 ensemble**: votes came back `{'SUPPORTED_1': 2, 'NONE_CORRECT': 1}`
  — the same 2-1 shape as the `chest pain` example above, but this time on
  a coronary-artery-segment abbreviation.
- **Why it didn't get rescued**: `LCX` matches
  `CORONARY_SEGMENT_TRAP_ABBREVIATIONS` — a hard-coded quarantine added
  after this *exact* entity (across several notes) kept split-voting wrong,
  a confirmed SapBERT embedding-collapse pattern (the model can't reliably
  separate "Left circumflex coronary artery" from the generic parent
  concept "Coronary artery structure"). The trap bypasses
  `calibrator.score()` entirely — it's never even called.
- **Outcome**: `routing_basis = "non-unanimous verdicts {...}; calibrator
  bypassed -- coronary-artery-segment trap..."`, `queue_reason =
  "coronary_segment_trap"`, stays `TIER_4_ENSEMBLE_SPLIT` → `HITL_REQUIRED`.
  Measured against gold: the plurality candidate here actually *was* wrong
  (resolved to the generic parent, gold wanted the named branch) — the trap
  correctly kept a genuinely bad case out of AUTO tier.

---

## TIER_5_TRUE_AMBIGUITY — no ensemble verdict is trustworthy

Three different ways to land here, all free (no model call) except the
last:

1. **`no_candidates`** (`tier5_precheck()`): Stage 2b produced zero
   candidates at all — nothing to evaluate.
2. **`below_similarity_floor`**: the best candidate in the pool scored under
   `TIER3_SIMILARITY_FLOOR` (0.72) — retrieval itself wasn't confident
   enough to hand the ensemble anything worth judging.
3. **`unresolved_acronym`**: `expansion_ambiguous=TRUE` and Stage 2a's
   ambiguity was never resolved (Phase 4's MoLLM acronym escalation exists
   but is gated off by default after its corpus-scale precision came in at
   34-45%, well below a safe bar — see the Phase 4 closeout).
4. **`standalone_qualifier_span`** (`qualifier_fragment_precheck()`, checked
   even before the above): a bare `Qualifier`-labeled span (a laterality or
   generic modifier word) is removed from the ensemble's job entirely —
   confirmed via a real DB query against the exact spans that caused an
   earlier precision collapse: `'left'`, `'right'`, `'multiple'`, `'third'`,
   `'R'`, `'Cranial'`.
5. **Full-ensemble path**: all three models independently, unanimously
   return `NONE_CORRECT` — genuine three-way agreement that nothing in the
   candidate pool is right.

All five routes land at the same `TIER_5_TRUE_AMBIGUITY` /
`HITL_REQUIRED`, each with a distinct `queue_reason` so a human reviewer
(or a later analysis pass) can tell which failure shape produced it,
without averaging five different problems into one bucket.

---

## `tier=None`, `below_confidence_threshold` — unanimous, but not confident enough

**Trigger**: 3/3 unanimous `SUPPORTED_1`, but the mean confidence is
*below* 0.70 — the one case where full agreement still isn't trusted.

**Real example**: `ciprofloxacin` (Medication), overnight corpus run.
- **Stage 3**: all three models accepted candidate #1 as a genuine match —
  no disagreement at all — but their own token-level confidence in that
  match averaged under the floor.
- **Outcome**: `routing_basis = "unanimous SUPPORTED_1 but
  composite_confidence <value> < 0.70"`, `tier = None`,
  `mollm_routing_decision = HITL_REQUIRED`, `queue_reason =
  "below_confidence_threshold"`. This is a distinct failure shape from
  everything above: it's not disagreement, and it's not a missing
  candidate — it's the models converging on an answer they're each
  individually unsure of.

---

## The safety traps — a special case worth naming on its own

Both `_is_coronary_segment_trap()` and `_is_short_alphanumeric_code()` run
*inside* the `TIER_4_ENSEMBLE_SPLIT` branch, strictly before
`calibrator.score()` is ever called — not a score override applied after
the fact. Real trigger seen this session: `S2` (Anatomy) scored **0.704**
by the calibrator on a genuinely fresh note (above the *old* 0.65
threshold, would have been promoted) — but `S2` matches the
short-alphanumeric-code shape (`^[A-Za-z]{1,2}[0-9]{1,2}$`), the same
embedding-collapse pattern as `LCX` just with numbers instead of named
branches (heart sound S2 vs. second sacral vertebra). Trapped before
scoring, stays `TIER_4_ENSEMBLE_SPLIT`.

---

## Summary table

| Tier | Model calls | Real example | Trigger |
|---|---|---|---|
| `TIER_1_AUTO_VALIDATED` | 3 models, full 2-step | `warfarin` | 3/3 unanimous `SUPPORTED_1`, confidence ≥0.70 |
| `TIER_2_AUTO_RESOLVED` | 3 models, full 2-step | `papilledema` | 3/3 unanimous re-rank to the same candidate N |
| `TIER_3_AUTO_VALIDATED` | 0 (fast path) | `Coumadin`→warfarin | sole verified-brand-alias candidate, no ambiguity |
| `TIER_1B_CALIBRATED_AUTO_VALIDATED` | 3 models + calibrator | `chest pain` (score 0.905) | non-unanimous, calibrator ≥0.72, no trap |
| `TIER_4_ENSEMBLE_SPLIT` | 3 models (+ calibrator attempt) | `LCX` (coronary trap) | non-unanimous, not rescued |
| `TIER_5_TRUE_AMBIGUITY` | 0-3 (often free) | `'left'`, `'right'` (qualifier precheck) | 5 distinct sub-reasons, see above |
| `None` / `below_confidence_threshold` | 3 models, full 2-step | `ciprofloxacin` | 3/3 unanimous but under-confident |
