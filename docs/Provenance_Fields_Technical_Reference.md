# Provenance Fields — Complete Technical Reference

**What this document covers**: every provenance field this pipeline
writes, stage by stage — where it is computed, exactly what it means,
and every real downstream location that reads it. "Provenance" here
means specifically: a field recording *how* a decision was reached, not
*what* the decision was — the audit trail, not the answer.

**Running example**: the same real entity used in
`docs/Entity_Journey_Plain_Language_Walkthrough.md` — "fever," note
`11859945-DS-29`, entity_id `11859945-DS-29-ebf08ad5f49`. Every real
value quoted below is that entity's actual stored value, pulled live
from the database, so the same entity can be cross-referenced across
all four technical documents.

**Companion documents**: `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`
(how Stage 2b provenance selects which prompt rule text is shown),
`docs/ConsensusCalibrator_Technical_Reference.md` (how 8 of these fields
become calibrator features 8–15), `docs/Code_Flow.md` (the execution
order these fields are written in).

---

## 1. The general pattern

Every stage **writes its own provenance once, and every later stage
reads it without recomputing it.** This is a deliberate, repeated design
choice in this codebase — assertion detection happens once in Stage 1
and is never re-derived; a candidate's retrieval tier is decided once in
Stage 2b and is never re-judged by Stage 3. The cost of this discipline
is that a field's meaning is fixed at write time and any consumer must
trust it; the benefit is that no two stages can silently disagree about
the same fact, and every decision has exactly one place to look for why
it was made.

**One real, load-bearing exception, worth knowing up front**: a handful
of fields exist **only as in-memory dict keys on an entity object
during a single pipeline run** — they are never columns in any table.
Reconstructing an entity by `SELECT *`-ing it back out of the database
silently drops them. This has caused two real, live bugs this session
(§9) and is the single most important gotcha in this document.

---

## 2. Stage 1 — Preprocessing provenance

### 2.1 Abbreviation-expansion provenance

| Field | Meaning | "fever" value |
|---|---|---|
| `expansion_ambiguous` | the abbreviation dictionary had >1 meaning for this token | `False` — "fever" isn't an abbreviation at all |
| `candidate_expansions` | the full list of alternative meanings, when ambiguous | `None` |
| `selection_basis` | **which tiebreak mechanism picked the winning expansion**, when ambiguous | `None` |

`selection_basis` is the richest field here — it records exactly which
rung of the four-tiebreak ladder (`docs/2026-08-20_Session_Results_And_Status.md`
§14 / the Decisions Log §2) resolved an ambiguous abbreviation:
`context_pattern_rule`, `numeric_context`, `observed_frequency_priority`,
`omop_groundability`, or `unvetted_ambiguous_unexpanded` (left unexpanded
entirely). **Real consumers**: `src/abbreviation_flywheel.py` (mines
confirmed-correct resolutions back into `mine_context_rules()`),
`scripts/score_gold_recall.py` (breaks recall down by which tiebreak won,
to answer "is the flywheel actually helping"), and every cold-start
builder (`src/physexam_shorthand.py`, `src/lab_abbrev_coldstart.py`,
`src/narrative_state_word_coldstart.py`) sets its own distinct
`selection_basis` string so a cold-start-injected entity is
distinguishable from a dictionary-expanded one downstream.

### 2.2 Section provenance

| Field | Meaning | "fever" value |
|---|---|---|
| `section_name` | which note section this span falls in | `"History of Present Illness"` |

Computed once by pattern-matching the note's own headers. **Real
consumers**: shown directly in both MoLLM prompts (§2, §4 of the prompts
reference document); read by `src/normalization/orchestrator.py` for
allergy-context detection (an entity under an "Allergies" section
combined with `assertion_status=PRESENT` triggers the allergy-search
override); displayed in the HITL review UI as one of the meta-caption
fields a reviewer sees.

### 2.3 Assertion provenance

| Field | Meaning | "fever" value |
|---|---|---|
| `assertion_status` | PRESENT / ABSENT / HISTORICAL / POSSIBLE / CONDITIONAL / ALLERGY / FAMILY | **`ABSENT`** |
| `experiencer` | PATIENT / FAMILY / OTHER | `PATIENT` |
| `temporality` | CURRENT / HISTORICAL | `CURRENT` |
| `assertion_cue` | the literal trigger word found | `"Denies"` |
| `assertion_cue_start` / `_end` | character offsets of that cue word | 1003 / 1009 |
| `assertion_cue_category` | the ConText rule category that fired | `NEGATED_EXISTENCE` |
| `assertion_engine` | which tool computed this | `medspacy_context/pyrush` |

Computed once, deterministically, by a rule-based tool (medspacy/ConText
— **not an LLM**) in Stage 1, and never recomputed by any later stage.
**Real consumers, and this is the field with the widest reach in the
whole pipeline**:

- **Both MoLLM prompts** (`src/mollm_tier_gate.py`) — `_clinical_meaning_prompt()`
  shows it directly and switches in `ALLERGY_MEANING_INSTRUCTION` when
  `assertion_status == "ALLERGY"`; `_binary_match_prompt()` shows it and
  rule 3 explicitly tells the model to ignore it for concept-matching
  purposes, except when `ALLERGY_CONTEXT_CLAUSE` is added for the same
  `ALLERGY` status (full detail: prompts reference document §2, §4, §7).
- **`src/normalization/orchestrator.py`** — `is_allergy_context` reads
  it directly to switch the Stage 2b search domain from
  Medication/RxNorm to Condition+Observation (the allergy-context
  retrieval override).
- **`src/kg3_ingestion.py`** — `ABSENT`/`FAMILY` records are never
  written to KG3 regardless of the MoLLM verdict, a hard rule
  independent of model confidence.
- **HITL display** (`ui/pages/2_🩺_HITL_Review_Queue.py`) — shown as a
  meta-caption alongside the entity so a reviewer sees it without
  opening the raw note.

For "fever" specifically: `ABSENT` is what makes this a genuinely
instructive example rather than a trivial one — the word still maps to
the concept *Fever* (Stage 2b's job), while this field separately and
correctly records that the patient does not have it. See the
plain-language walkthrough §1–§2 for the full explanation of why that
separation matters.

---

## 3. Stage 2a — Extraction provenance

| Field | Meaning | "fever" value |
|---|---|---|
| `entity_label` | GLiNER's own category guess | `Symptom` |
| `confidence` | GLiNER's extraction confidence | `0.9096296429634094` |
| `orig_start` / `orig_end` | character offsets in the **raw** note text | 833 / 838 |
| `exp_start` / `exp_end` | character offsets in the **expanded** (abbreviation-substituted) text | 1010 / 1015 |
| `below_threshold` | did this clear the 0.50 promotion gate, or only the 0.35 retention floor | `False` |
| `extraction_threshold` | the floor actually used for this run | `0.3499999940395355` |
| `flat_ner` | flat (non-nested) span mode, project-wide constant | `True` |
| `crosses_sentence_boundary` | did the span straddle two sentences | `False` |
| `sentence_id` / `sentence_ids_spanned` | which sentence(s) this falls in | `26` / `[26]` |
| `local_context` | the sentence-bounded text window | *(the "Denies fever, chills..." sentence)* |
| `compound_split_of` | if this entity resulted from splitting a longer span, the parent's id | `None` |
| `superseded_by_split` | this entity was later replaced by a compound split | `False` |
| `grown_from` | if this entity's span was widened, the original narrower entity's id | `None` |
| `superseded_by_growth` | this entity was later replaced by a grown span | `False` |
| `possibly_truncated` | GLiNER's own tokenizer truncated the input before reaching this span | `False` |
| `gliner_input_token_count` | the actual token count GLiNER saw for this note | `3276` |
| `gliner_model_version` | which checkpoint extracted this | `Ihor/gliner-biomed-large-v1.0` |

**`orig_*` vs `exp_*`, and why both are kept**: an abbreviation
expansion can change span length (e.g. "MI" → "myocardial infarction"),
so a single offset pair can't correctly index both the original clinical
note text and the expanded text GLiNER actually ran on. `map_offsets_to_original()`
(`src/entity_extraction.py`) is the one function trusted to convert
between the two coordinate systems; every consumer that needs to
highlight text in the *raw* note (the HITL page, the Troubleshooting
page, the plain-language walkthrough's highlighter) uses `orig_start`/`orig_end`
specifically for that reason.

**`superseded_by_split` / `superseded_by_growth`, and why every grading
script filters on them**: when compound-span splitting or span-growth
promotes a *better* version of an entity, the original row is kept (for
audit trail) but flagged superseded rather than deleted. **Every single
evaluation script in this codebase excludes superseded rows** — a
replaced entity and its replacement would otherwise both contribute a
data point for what is really one underlying decision, silently
inflating any count. Confirmed real consumers: `evaluation/iou_metrics.py`,
`evaluation/stage2a_cal_eval.py`, `evaluation/stage2b_cal_eval.py`,
`evaluation/stage2b_hybrid_ab.py`, `evaluation/kg_tiebreak_validation.py`,
`evaluation/stage1_disambiguation_eval.py`, `evaluation/stage_calibration.py`
— nine separate scripts all independently applying the same filter,
because a superseded row left in any one of them would silently
double-count.

**`possibly_truncated`**: surfaced specifically so a corpus-wide query
could measure real impact before building the chunking fix — see
`docs/Implementation_Decisions_Log.md` §3 for the real before/after
(124 vs. 399 entities recovered on the corpus's longest note) that
measurement produced.

---

## 4. Stage 2b — Normalization provenance (the richest set)

This is the stage with the most downstream reach — its fields feed
directly into MoLLM prompt construction, the fast-path bypass logic, and
8 of the calibrator's 16 features.

| Field | Meaning | "fever" value |
|---|---|---|
| `match_tier` | which retrieval tier found the winning candidate | `"1 (Exact)"` |
| `similarity_score` | that tier's own confidence in the match | `1.0` |
| `match_basis` | **finer-grained than match_tier** — the exact mechanism, per-candidate | `"exact_text"` |
| `candidates` | the full ranked candidate list (JSON) | 1 candidate: concept 437663 "Fever" |
| `candidates_hash` | a hash of the candidate list, for change detection | `34b9090ad1da6faa` |
| `is_ambiguous` | Stage 2b's own flag that the pool is genuinely contested | `False` |
| `ambiguity_reason` | why, when `is_ambiguous` | `None` |
| `domain_conflict` | the candidate's OMOP domain disagreed with the extracted label | `None` |
| `normalized_from` | **the exact route by which this concept was reached** | `"expanded"` |
| `tier_trace` | a full log of which tiers were attempted and how many hits each got | `[{"tier": "1 (Exact)", "attempted": true, "hits": 1}]` |
| `tier_reasons` | why this entity was flagged for extra Stage 3 scrutiny at all | `["high_gliner_risk", "weak_match_tier"]` |
| `confidence_tier_in` | the coarse routing bucket Stage 2b assigned before Stage 3 ever runs | `"LOW"` |
| `domain_id_queried` / `vocab_queried` | which OMOP domains/vocabularies retrieval was restricted to | `["Condition", "Observation"]` / `["SNOMED"]` |
| `sapbert_pooling_method` | which embedding pooling strategy was used | `"cls"` |
| `matched` | did retrieval find anything at all | `True` |
| `athena_vocabulary_release` | which exact SNOMED/OMOP data snapshot was queried | `signature:snomed_n=1093147,latest_valid_start=20250409` |
| `tier12_rank_basis` | how Tier 1/2 candidates were ordered when more than one exact/synonym hit exists | `"concept_id_asc"` |
| `run_id` / `code_version` / `created_at` | which pipeline execution, which git state, and when | `run_bb6274bb27704c46` / `1826d7e-dirty` / `2026-08-19 07:08:21` |
| `is_stale` | was this row computed by code that has since changed (see §8) | `False` |

### 4.1 `match_basis` — the field that changes what a model is told to believe

Six real values seen in this codebase, each triggering different
downstream behavior:

| `match_basis` value | Meaning | Special handling |
|---|---|---|
| `exact_text` | literal Tier 1 name match | none — the default case |
| `synonym` | Tier 2 synonym-table match | none |
| `semantic_similarity` | Tier 3 SapBERT match (the default) | none |
| `verified_brand_alias` | a curated brand→generic drug lookup, not a similarity guess | Step B rule 2 fires (prompts reference §4); `_CURATED_MATCH_BASES` wins hierarchy-collapse ties unconditionally |
| `verified_lab_test_alias` | a curated abbreviation→concept lookup (the 26 `_LAB_TEST_ALIASES` entries) | same as above, plus eligible for `tier3_fast_path()`'s zero-LLM-call bypass |
| `lab_procedure_preferred` | tagged by `_prefer_lab_procedure_over_observable()`'s rank penalty | eligible for `_lab_procedure_fast_path()`'s zero-LLM-call bypass |

**Real consumers**: `_binary_match_prompt()` (conditionally injects rule
2, prompts reference §4), `tier3_fast_path()`/`_lab_procedure_fast_path()`
(`src/mollm_tier_gate.py`, deterministic bypasses that skip the LLM
ensemble entirely for these bases), `_collapse_hierarchy_duplicates()`
(`src/normalization/tier_retrieval.py`, curated bases win union-find
ties unconditionally regardless of raw score).

### 4.2 `normalized_from` — the "how fragile was this route" field

This single string field is parsed by **substring match** in three
completely different places for three different purposes, and is the
direct source of calibrator features 12–14:

- **`src/mollm_ensemble.py`** (the superseded old ensemble): if it
  starts with `"value_stripped_from_"`, a unanimous vote is capped at
  `MOLLM_RESOLVED` instead of promoted to `AUTO_VALIDATED` — the
  "Fragile Concept Gate" (`docs/Implementation_Decisions_Log.md` §4).
- **`src/mollm_tier_calibrator.py`**: three separate substring checks
  become three separate features — `"value_stripped_from_"` (feature
  12), `"original_after_expanded_failed"` (feature 13),
  `"+acronym_mollm"` or `"+acronym_cache"` (feature 14). Full detail:
  the calibrator reference document §3.2.
- **`evaluation/grade_overnight_corpus_run.py`**: filters specifically
  for `"+acronym_mollm"`/`"+acronym_cache"` to isolate and separately
  grade the acronym-escalation population — this is literally how the
  34.3–36.1% acronym-escalation precision figure (Decisions Log §5) was
  computed.

Real values observed in the live system, beyond `"fever"`'s plain
`"expanded"`: `"original_after_expanded_failed"` (the expanded text
found nothing, raw text retried), `"value_stripped_from_MCHC-31"` (a
lab-value suffix had to be stripped first), and compound strings like
`"expanded+acronym_mollm"` (an ambiguous abbreviation was resolved by
the MoLLM acronym-escalation model before normal retrieval ran on it).

### 4.3 `is_ambiguous` — the field with a 100%-exceptionless finding attached to it

**The single most consequential provenance fact in this whole
document**: every one of the 259 `TIER_2_AUTO_RESOLVED` decisions ever
made in this database has `is_ambiguous = True`. This is the direct,
measured evidence behind the decision to exclude Tier 2 from
`AUTO_TIERS` (Decisions Log §5) — a "3/3 unanimous re-rank" on a pool
Stage 2b itself already flagged as contested more likely reflects three
models sharing a bias than three models independently verifying. It's
also calibrator feature 10, and is read directly by
`_lab_procedure_sibling_check()` (`src/normalization/tier_retrieval.py`)
to decide whether an exact/synonym-tier hit needs a supplementary
Procedure-class semantic lookup.

---

## 5. Stage 3 — MoLLM Tier Gate provenance

| Field | Meaning | "fever" value (real, from §6 of the prompts reference) |
|---|---|---|
| `tier` | the final routing tier | `TIER_1B_CALIBRATED_AUTO_VALIDATED` |
| `mollm_routing_decision` | AUTO_VALIDATED / HITL_REQUIRED | `AUTO_VALIDATED` |
| `queue_reason` | why, when routed to HITL | `None` (not applicable here) |
| `final_candidate_index` | which candidate (1-based) the gate is committing to | `1` |
| `composite_confidence` | the ensemble's own mean confidence over the winning verdict | `0.818419` |
| `calibrated_score` | the calibrator's P(correct), when consulted | `0.927709` |
| `routing_basis` | **a human-readable sentence explaining the exact decision path** | `"non-unanimous verdicts {'NONE_CORRECT': 1, 'SUPPORTED_1': 2}, but ConsensusCalibrator scored 0.927709 >= 0.72 (prior_confirmation_count=21)"` |
| `models` | the full per-model record: verdict, clinical_meaning, reasoning, logprob_confidence, degenerate_generation, eval_trail | *(see prompts reference §3–4 for the full real text)* |
| `mollm_call_id` | unique id for this specific gate call | `8d9b97d9-572e-4bd9-8d31-968dc1f84a7a` |

**`routing_basis` deserves special mention**: it is the one field in the
entire system whose entire purpose is human-readability. It exists
specifically so a reviewer (or a document like this one) doesn't have to
reconstruct "why did this get promoted" by re-deriving it from the raw
`models` JSON and the tier logic — it's written once, in plain English,
at decision time, by the exact code path that made the decision.

**`models[i].eval_trail`**: the fullest provenance record in the whole
system — not just a model's final verdict, but every intermediate
judgment (Step A's `clinical_meaning`, then one accept/reject entry per
Step B candidate, in order, plus a `tiebreak: true` entry if the
comparative call ran). This is what makes the HITL page's "MoLLM
decisions" panel possible (§6 below) and what let this document's
companion prompts reference show qwen's actual reasoning error rather
than just its final wrong verdict.

---

## 6. Stage 4 — HITL provenance

| Field | Meaning |
|---|---|
| `hitl_case_id` | deterministic id, `source_table + source_call_id`, so re-enqueueing is idempotent |
| `source_table` | which of the (up to 3) decision tables this case came from |
| `presented_suggestion` | **a JSON snapshot** of everything the reviewer needs, captured at enqueue time |
| `reviewer_decision` | PENDING / APPROVED / CORRECTED / REJECTED |
| `corrected_concept_id` | if CORRECTED, the reviewer's chosen concept |
| `rejection_reason` | free text, if REJECTED |
| `reviewer_comment` | free text, available on **every** decision, not just REJECTED |
| `review_duration` | how long the reviewer spent, tracked in the UI via `st.session_state.hitl_case_started_at` |

**`presented_suggestion` is a snapshot, not a live join** — this is why
it needed `orig_start`/`orig_end` added explicitly (§9 below) for the
full-note highlighter to work; anything not captured into this JSON blob
at enqueue time is invisible to the reviewer UI, however available it
might be elsewhere in the database.

**`reviewer_comment` is the one field this entire provenance chain feeds
*back into***: `src/abbreviation_flywheel.py`'s `mine_context_rules()`
reads confirmed human corrections from this queue and mines them into
deterministic pre/post trigger-word rules for future entities — the
only place in the system where a human's free-text explanation becomes
machine-usable input again, closing the loop the rest of this document
describes as one-directional.

---

## 7. Cross-cutting provenance — versioning and staleness

| Field | Present on | Meaning |
|---|---|---|
| `run_id` | `normalized_entities`, `mollm_tier_gate_decisions` | which specific execution produced this row — lets two rows from the same pipeline run be distinguished from two rows produced days apart |
| `code_version` | same tables | a git short hash (or a descriptive retrain label, see the calibrator reference §10.9) — the exact code state that produced this decision |
| `created_at` | same tables | wall-clock timestamp |
| `is_test` | every stage's tables | smoke-test rows vs. real corpus rows, purgeable independently |
| `is_stale` | `normalized_entities` | **the field that makes re-normalization detectable** — set by `scripts/mark_notes_stale.py`, read by the Streamlit note selectors (`ui/pages/1`, `3`, `4`) to hide notes whose candidates predate the current code |

**`is_stale` is the field behind this session's single most consequential
process discovery**: `scripts/run_stage3_tier_gate.py` never recomputes
Stage 2b candidates — it only reads what's already stored. When the
SNOMED namespace-exclusion fix landed, restarting Stage 3 did **not**
exercise the fix at all, because the stored `normalized_entities.candidates`
rows were untouched and still `is_stale`-eligible under the old code.
Fixing this required a dedicated re-normalization pass
(`scripts/refix_uk_extension_lab_candidates.py`), not just a Stage 3
restart — see `docs/Code_Flow.md` §4's closing note and
`docs/Implementation_Decisions_Log.md` §5 for the full story.

---

## 8. The in-memory-only marker fields — a real, repeated gotcha

**Three fields exist only as keys on an entity `dict` during a single
pipeline run, and are never written as columns to any table:**

| Marker | Set by | Consumed by |
|---|---|---|
| `physexam_shorthand` | `build_physexam_shorthand_entities()` | `_cold_start_mapping()` (`src/normalization/orchestrator.py`), same run only |
| `narrative_coldstart` | `build_narrative_state_word_entities()` | same |
| `mollm_resolved_expansion` | `resolve_ambiguous_acronyms()` (when `ACRONYM_ESCALATION_ENABLED`) | the allergy-context-style interceptor in `process_and_normalize_entities()`, same run only |

**Why this matters, concretely**: if a script reconstructs an entity
dict via `SELECT * FROM extracted_entities WHERE ...` — the natural,
obvious way to "re-process an existing entity" — these markers are
silently absent, because they were never persisted anywhere to select.
The entity looks structurally identical to a normal one and processes
without error, just with the WRONG code path (the ordinary retrieval
path instead of the cold-start bypass it actually needs).

**This caused two real, live bugs in this session, both caught only by
directly inspecting live post-fix data, not by any test passing**:

1. `scripts/refix_coldstart_hierarchy_collapse.py` re-normalized
   physexam/lab-abbreviation cold-start entities correctly (their
   markers are cheap to reconstruct from `gliner_model_version`), but a
   **second-order regression**: the SAME script also happened to touch
   narrative-state-word entities, which it reconstructed via a raw DB
   select — stripping their `narrative_coldstart` marker and silently
   routing them through ordinary Tier 1-3 search instead of the
   deterministic bypass they need.
2. The fix for bug 1 (`scripts/refix_narrative_coldstart_marker.py`)
   had to call the **real builder function fresh, per note** — not
   reconstruct entities from a DB read at all — specifically because
   the marker cannot survive a round-trip through storage.

**The general lesson, stated for reuse**: any future re-processing
script that reconstructs entities from a DB `SELECT` rather than calling
the original extraction/cold-start builder functions will silently drop
these three fields. This is not a bug to be fixed once — it is a
structural property of how these markers are designed (deliberately
in-memory-only, to avoid a schema migration for what's meant to be a
same-run signal), and every future script touching these entity
populations needs to know it going in.

---

## 9. A field added mid-session for a specific, documented reason

`orig_start` / `orig_end` on `presented_suggestion` (§6) did not exist
in the HITL queue snapshot until 2026-08-20 — added specifically because
the full-note highlighter (`ui/pages/2_🩺_HITL_Review_Queue.py`'s
rebuild) needed a real character offset to locate the entity inside the
raw note text, and the pre-existing snapshot only carried
`local_context` (the sentence-bounded window, not an offset pair). Rows
enqueued before this change lack the field entirely and degrade
gracefully to the local-context-only view — a real, live example of
`presented_suggestion`'s snapshot-at-enqueue-time nature (§6) actually
mattering: the underlying `extracted_entities.orig_start` value had been
sitting in the database the whole time, just never captured into this
particular JSON blob until something needed it there.

---

## 10. KG-embedding checkpoints — a provenance artifact outside the DB row model

Every other field in this document lives inside a DuckDB row and follows
§1's pattern (written once by one stage, read without recomputation by
later stages). The KG-embedding checkpoints (`src/kg_embedding.py`'s
TransE, `src/kg_embedding_rotate.py`'s RotatE) are a different shape of
provenance artifact — a versioned **file**, not a row — worth recording
here specifically because it's the one place this pipeline's provenance
discipline extends outside the database.

**What `save_model()` bundles, and why the bundle matters as provenance**
(`src/kg_embedding.py` / `src/kg_embedding_rotate.py`, both identical in
shape):

| Field in the `.pt` checkpoint | Meaning |
|---|---|
| `state_dict` | the trained weights |
| `entity2idx` | which OMOP `concept_id`s were in THIS training run's subgraph, and their embedding-table row |
| `relation2idx` | which relation types were in THIS run's subgraph |
| `dim` | embedding width used (RotatE: per-component width, doubled internally for the packed real+imaginary table) |

The vocab is data-dependent — a state_dict alone can't be reused without
knowing which concept_id maps to which embedding row, so `entity2idx`/
`relation2idx` are carried in the checkpoint itself rather than assumed
stable across runs. This is the file-based analog of this document's own
"vocab is data-dependent" discipline (§4.1's `match_basis`, §7's
`code_version`) — a checkpoint is only interpretable together with the
exact vocab it was trained on, so the two are never separated.

**Checkpoint inventory, as of 2026-08-31** (all read-only evaluation
artifacts — none is loaded by any live pipeline code path):

| File | Training data | Real triple count |
|---|---|---|
| `models/kg_transe_v1.pt` | SNOMED CT relationship subgraph, restricted to this pipeline's own touched concepts | 24,922 |
| `models/kg_rotate_guideline_v1.pt` | Curated clinical-guideline graph (Memgraph `:GuidelineNode`), OMOP-grounded subset | 263 |
| `models/kg_rotate_gold_v1.pt` | This project's own gold-confirmed candidate-competition signal (`gather_tp_records()`) | 1,593 |
| `models/kg_rotate_combined_v1.pt` | guideline + gold, concatenated | 1,856 |
| `models/kg_rotate_snomed_is_a_v1.pt`† | Full SNOMED IS_A hierarchy, separate KG1 Neo4j instance | 530,515 |

†**Not tracked in git** — 319,557 entities makes this checkpoint ~247MB,
over GitHub's 100MB per-file limit (no LFS configured for this repo).
Every other checkpoint above is small enough (a few KB–3MB) to commit
directly. Regenerate locally with `python3 scripts/build_kg_embeddings_
rotate.py --config snomed_is_a` — same "large, script-regenerable, not
committed" convention already used for `db/` (31GB, gitignored).

**Why none of this shows up anywhere else in this document yet**: as of
2026-08-31, no entity-level provenance field (`routing_basis`,
`match_basis`, `normalized_from`, or any `mollm_tier_gate_decisions`
column) is ever populated from a KGE checkpoint. `src/kg_embedding_
tiebreak.py`'s functions exist and are unit-tested, and
`evaluation/kg_tiebreak_validation.py` validates every checkpoint above
against real gold data — but that validation is an offline evaluation
harness, not a wired-in decision path. Both TransE and RotatE were
measured to lose head-to-head to the existing hardcoded
`_prefer_lab_procedure_over_observable()` rule at every threshold tested
(0 losses for the rule vs. 105+ for every KGE variant) — the honest,
current reason this document has no "KGE-sourced" row to add to §5's
table. If a KGE signal is ever added as a `ConsensusCalibrator` feature
(proposed, not yet built — see `docs/RotatE_KG_Embedding_Technical_
Reference.md`'s closing section), that would be the first time a KGE
checkpoint's output becomes real, DB-persisted, entity-level provenance,
and this section should be updated to add it to §5's table at that
point.
