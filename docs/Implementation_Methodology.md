# docs/Implementation_Methodology.md — Architecture & Methodology

> **Status: current as of 2026-08-27.** This doc describes the pipeline as it
> is actually built and running, not the original proposal. It supersedes
> its own earlier version (which described a never-built MedGemma/OpenBioLLM
> two-model ensemble) and consolidates what was otherwise scattered across
> ~25 dated point-in-time docs, now moved to `docs/archive/`. Those docs
> remain as the historical record of *how* each piece was found/fixed/
> measured — this doc is the *current-state* reference.
>
> **For numbers, start with `docs/FINAL_RESULTS_Single_Source_Of_Truth.md`**,
> not this doc. For deeper detail than this file covers: `docs/Code_Flow.md`
> (execution order), `docs/Code_Reference_Stages_And_Metrics.md` (core code +
> metric formulas), `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`
> (every prompt verbatim), `docs/ConsensusCalibrator_Technical_Reference.md`
> (the 16-feature calibrator in full), `docs/Provenance_Fields_Technical_Reference.md`
> (every provenance field, where computed, every consumer), and
> `docs/Entity_Journey_Plain_Language_Walkthrough.md` (one real entity
> traced end to end). `docs/2026-08-20_Session_Results_And_Status.md` and
> `docs/Implementation_Decisions_Log.md` carry the narrative and the
> decision-by-decision rationale, respectively.

## Objective

A neuro-symbolic pipeline that extracts clinical entities from MIMIC-IV
discharge notes, grounds them to OMOP/SNOMED CT concepts, and validates
that grounding with an LLM ensemble — aiming for a high share of fully
autonomous, high-precision concept writes to a knowledge graph (KG3), with
the remainder routed to human review rather than written unreviewed.
Real KG3 writes are gated behind `dry_run=True` everywhere in the code
today; nothing has been switched to write live yet.

## Data Sources

* **MIMIC-IV-Note** discharge summaries — the input corpus.
* **PhysioNet SNOMED CT Entity Linking Challenge v1.2.0** — the gold
  evaluation set: 272 notes, 75,491 annotation rows, `train`/`test` split
  preserved via the archive's own `annotation_type` column. `start`/`end`
  are stored as floats in the CSV (`"180.0"`) — cast with `int(float(x))`.
* **imantsm/medical_abbreviations** — a static abbreviation dictionary used
  by Stage 1's expansion step.
* **Athena/OHDSI OMOP vocabulary tables**, loaded into DuckDB
  (`db/kg2_lexical_store.duckdb`) — `athena_concept`,
  `athena_concept_synonym`, `athena_concept_relationship`,
  `athena_concept_ancestor`. This is also where the SNOMED CT IS_A
  hierarchy itself lives and is queried — see the correction below.
* **A curated clinical-guideline triplet corpus**
  (`data/local_triplets_db2_v6_cleaned_grounded_rules_added/`, 76 source
  documents, ~1,700 nodes, 1,162 extracted recommendation rules) — the
  rules-based KG behind the Stage 3 guideline-evidence bridge (see
  "Rules-based guideline KG" under Stage 3 below). File-backed, loaded by
  `GuidelineIndex` in `src/retrieval.py`, not a graph database.

**Correction to an earlier version of this doc**: SNOMED hierarchy
queries and KG3 write-back are **not** the same graph, and neither goes
through Memgraph for hierarchy lookups. Memgraph (Bolt,
`bolt://localhost:7688`) holds **only** the dynamic patient-instance KG3
graph (`:PatientObservation` nodes, dry-run writes) — structurally
separate from the static reference vocabulary on purpose (see Stage 4).
SNOMED IS_A traversal for guideline-rule matching queries DuckDB's
`athena_concept_ancestor`/`athena_concept_relationship` tables directly,
the same as every other retrieval query in this pipeline. There is no
Neo4j deployment in the current system — see the main `README.md`.

## Pipeline Architecture (as built)

Five stages, each with its own module and its own notion of "correct" —
see `evaluation/stage_calibration.py`'s docstring for why ECE/IoU numbers
should never be compared directly across stages.

### Stage 1 — Preprocessing (`src/preprocessing.py`)
Segments the note into sections (`segment_sections()`), expands known
abbreviations via dictionary lookup, and tracks character-offset mapping
back to the original raw text throughout. Ambiguous abbreviations (a token
with multiple possible expansions, e.g. "ED", "RA", "MS") are flagged
`expansion_ambiguous=TRUE` with their `candidate_expansions` list stored —
this population feeds the (currently off-by-default) acronym-escalation
mechanism, see below.

### Stage 2a — Extraction (`src/entity_extraction.py`, `src/extraction.py`)
* **GLiNER-BioMed** (`Ihor/gliner-biomed-large-v1.0`) does zero-shot span
  extraction over 6 clinical labels (Condition, Medication, Procedure, Lab
  Test, Anatomy, Symptom). Runs on **GPU** (explicit `map_location="cuda"`
  with a CPU fallback on failure — `GLiNER.from_pretrained()`'s own default
  is CPU-only, confirmed live via `nvidia-smi --query-compute-apps` and a
  10+ minute single-note CPU-inference regression before the fix).
* **`EXTRACTION_THRESHOLD = 0.35`** (lowered from an unexamined 0.5 default
  after a corpus-wide threshold sweep, n=21,653 predictions, showed
  GLiNER's own confidence is *inverted* against correctness in this corpus
  — precision falls monotonically as confidence rises from 0.35 to 0.95).
  Everything down to `SUBTHRESHOLD_FLOOR = 0.35` is stored and flagged
  `below_threshold`, not silently dropped.
* **Sliding-window chunking** (`_build_chunks()`/`_extract_entities_chunked()`)
  activates automatically for notes over `CHUNK_WORD_BUDGET=1800` words
  (128-word sentence-boundary-snapped overlap) — confirmed to recover 282
  real entities GLiNER silently dropped via single-pass truncation on the
  corpus's longest note (24,858 chars). Full technical detail — model
  choice, threshold calibration evidence, the chunking algorithm, and the
  three false-positive filters built around GLiNER's specific failure
  modes — in `docs/GLiNER_Models_Technical_Reference.md` Part 1.
* **PhysExam shorthand cold-start** (`src/physexam_shorthand.py`) —
  telegraphic exam notation ("Abd: S/NT/ND", bare section headers like
  "HEENT"/"NAD") that GLiNER never proposes as a candidate at *any*
  confidence (corpus-wide: 1,788 gold annotations in Physical-Exam
  sections, 98.9% short abbreviation-shaped spans). Injected directly from
  raw text via an evidence-mined (not guessed) text→concept dictionary,
  after the sub-threshold filter, skipping any span GLiNER already covers.
* **GLiNER-relex** (`knowledgator/gliner-relex-large-v1.0`, `src/extraction.py`)
  — relation extraction, same GPU-placement fix applied. Clinical-T5 was
  removed from the live pipeline (MIMIC-III/IV pretraining contamination
  risk against the eval set) and is kept only as an external baseline.
  Extracted relations feed Channel E of guideline-evidence retrieval
  (`src/retrieval.py`'s `channel_e_relation()`) — real, live-measured
  reach: 34/300 relations (11.3%) clear the both-endpoints-linked-and-
  SNOMED-anchored bar. Full technical detail — the GLiREL fallback
  investigation, offset-overlap endpoint linking, the shared-`FLAT_NER`
  measurement, and the full predicate-coverage numbers — in
  `docs/GLiNER_Models_Technical_Reference.md` Part 2.
* **Span growth** then **compound-span splitting** run after extraction,
  before normalization.

### Stage 2b — Normalization / Grounding (`src/normalization/`)
* **SapBERT** (`src/normalization/sapbert_model.py`) embeds spans; also
  GPU-placed with a CPU fallback — the highest-volume model call in the
  pipeline (every Tier 3 candidate needs an embedding).
* **Tiered candidate retrieval** (`src/normalization/tier_retrieval.py`,
  `compound_span.py`): Tier 1 (exact concept-name match), Tier 2 (exact
  synonym match), Tier 3 (SapBERT dense semantic similarity, gated at
  `TIER3_SIMILARITY_FLOOR = 0.72`), plus curated verified-alias tables for
  brand→generic drug names and common lab-test shorthand
  (`_LAB_TEST_ALIASES`). `CANDIDATE_LIMIT = 5` (widened from an original 3
  after measuring oracle-in-top-5 accuracy).
* **`CONTEXTUAL_CANDIDATES_ENABLED`** (default **ON** since 2026-08-18):
  widens Condition-label search to also include the Observation domain —
  fixes a real class of "Condition vs Observation" duplicate-concept
  confusions (e.g. wound dehiscence). Also drives
  `_prefer_lab_procedure_over_observable()` / `_lab_procedure_sibling_check()`
  in Tier 3 ranking, which re-ranks Procedure-class lab concepts ahead of
  their Observable-Entity siblings and tags the winning candidate's
  `match_basis` so downstream tier-gating can trust it deterministically
  (see Stage 3's Tier 3 fast path below).
* **Hybrid BM25+dense retrieval** (`bm25_index.py`,
  `_tier3_hybrid_rows()`) exists and was A/B tested via real RRF weight
  sweeps — **stays OFF** (`CNSP_HYBRID_RETRIEVAL` unset): dense-only
  strictly beat every blended weight on both Top-1 and oracle accuracy in
  the actual measurement. A validated non-default, not an unfinished
  feature.
* **Acronym escalation** (`src/acronym_escalation.py`) — a local-Ollama
  single-model call that picks among an ambiguous abbreviation's
  `candidate_expansions`, schema-constrained (hallucination structurally
  impossible), with a reconfirm-before-trust cache
  (`acronym_priors`, `MIN_CACHE_HIT_COUNT=2`). **Stays OFF**
  (`ACRONYM_ESCALATION_ENABLED`/`CNSP_ACRONYM_ESCALATION` unset):
  corpus-scale validation (31 notes) found only 34.3–36.1% precision, root-
  caused to a systematic textbook-prior bias (e.g. "LAD"→"left anterior
  descending artery" even when gold means "Lymphadenopathy"). A considered,
  data-grounded decision to keep off, not an oversight.
* **Abbreviation flywheel** (`src/abbreviation_flywheel.py`) — mines the
  pipeline's own confirmed outcomes and real HITL-reviewer-confirmed
  resolutions into deterministic pre/post context-word rules
  (`mine_context_rules()`), checked before any model call. The
  frequency-prior half of this mechanism is gated behind an explicit,
  currently-empty `VERIFIED_ALLOW_LIST` (inverted from an earlier
  block-list design after a live check found 7/7 of its highest-confidence
  ledger entries were wrong — the mechanism was re-confirming its own
  earlier mistakes).
* **SNOMED regional-extension exclusion** (2026-08-20,
  `_UK_EXTENSION_EXCLUSION` in `tier_retrieval.py`) — the SCTID
  namespace-identifier block (`concept_code NOT LIKE '%1000000___'`)
  spliced into every open-ended Tier 1-4 retrieval query. `vocabulary_id`
  cannot make this distinction (only one distinct value exists in the
  whole `athena_concept` table) and `concept_class_id` alone is not
  reliable either (23,842 of the excluded concepts are themselves
  `'Procedure'` class) — the SCTID namespace block is the actual robust
  signal, verified against zero overlap with the corpus's 4,522
  gold-standard SNOMED codes before shipping. Paired with an extension of
  `_prefer_lab_procedure_over_observable()`'s rank penalty to also cover
  `'Qualifier Value'`-class near-duplicates (22 SNOMED "X calculation
  technique" concepts), a second, distinct duplicate pattern the exclusion
  alone surfaced. Verified against gold on the MCHC/RDW subset: 100%
  (28/28) of post-fix candidate pools now surface the gold-correct
  concept as top SapBERT candidate.
* **KG embedding (TransE) topological tiebreak** (`src/kg_embedding.py`,
  `src/kg_embedding_tiebreak.py`) — a real TransE model (Bordes et al.
  2013, plain PyTorch) trained on the SNOMED CT relationship subgraph this
  pipeline's own candidate pools touch (7,269 concepts, ~24,900 edges;
  MRR 0.776, Hits@10 0.909 on held-out link prediction). Its tiebreak
  signal is mean embedding distance from a tied candidate to the rest of
  that entity's OWN SapBERT-proposed candidate pool, not a GLiNER-label
  centroid or neighboring-entity anchor (both considered and rejected:
  the former blurs distinct concepts sharing one coarse label, the latter
  risks error-cascading from an already-wrong neighbor).
  **Evaluated against gold, not deployed**: head-to-head against
  `_prefer_lab_procedure_over_observable()` on the rule's own pattern
  showed the hardcoded rule strictly safer (0 losses at every threshold
  0.01-0.08 vs. KGE's 63 losses past 0.02) — the rule was kept, KGE was
  not wired into production as its replacement. On the broader tied-pair
  population beyond the rule's scope, KGE showed a genuine positive net
  (265 win / 181 loss at threshold 0.03) — real value as a generalist
  secondary signal, not yet integrated pending a calibrated gating
  mechanism (the raw win/loss rate is not risk-free enough for auto-write
  decisions on its own). Full technical detail — architecture, training
  procedure, both evaluations, the gold-validated threshold sweep, and
  the specific falsified claim — in
  `docs/TransE_KG_Embedding_Technical_Reference.md`.

### Stage 3 — MoLLM Tier Gate (`src/mollm_tier_gate.py`)
Three local Ollama models — **qwen2.5:3b, llama3.2:3b, phi4-mini** — each
independently run a two-step chain-of-thought: Step A defines the
clinical meaning first, with no candidate list visible; Step B then
judges candidates one at a time, starting at #1. **Whether Step B stops
at the first accepted candidate or evaluates every candidate is itself a
flag** (`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED`, default **ON** — see below);
the historical "stop at first accept" behavior is now the non-default
case. Full prompt text, every conditional rule, and worked real examples:
`docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`.

`route_tier()` combines their verdicts into one of:

| Tier | Meaning | Routing |
|---|---|---|
| `TIER_3_AUTO_VALIDATED` | fast path: exact lexical / curated alias / lab-procedure-preferred, no LLM call needed | AUTO |
| `TIER_1_AUTO_VALIDATED` | 3/3 unanimous `SUPPORTED_1` on candidate #1 | AUTO |
| `TIER_1B_CALIBRATED_AUTO_VALIDATED` | fails hard Tier 1/3 rules, but the calibrator scores a split vote above threshold (call site 1) | AUTO (separate audit trail from genuine Tier 1) |
| `TIER_2_AUTO_RESOLVED` | 3/3 unanimous re-rank to the same candidate N≠1 | **currently excluded from `AUTO_TIERS`, routes to HITL** (see below) |
| `TIER_2B_CALIBRATED_AUTO_RESOLVED` | a Tier 2 (unanimous re-rank) decision the calibrator scores above threshold (call site 2) | AUTO by routing decision, but **also excluded from `AUTO_TIERS`** — see below |
| `TIER_4_ENSEMBLE_SPLIT` | any 2-1 or 1-1-1 disagreement | HITL_REQUIRED |
| `TIER_5_TRUE_AMBIGUITY` | low top-candidate similarity / degenerate context / no candidate at all | HITL_REQUIRED |

`AUTO_TIERS = {TIER_1_AUTO_VALIDATED, TIER_3_AUTO_VALIDATED, TIER_1B_CALIBRATED_AUTO_VALIDATED}`
(`src/mollm_tier_gate.py`) — the only tiers whose decisions
`kg3_ingestion.ingest_auto_decision()` will write (still `dry_run=True`
everywhere in production).

**`TIER_2_AUTO_RESOLVED` is deliberately NOT in `AUTO_TIERS` as of
2026-08-19.** Root cause of its measured ~20% precision was found and
fixed (a genuinely dead `CONDITION_VS_OBSERVATION_PRIOR` gate that had
never fired in production, relaxed from a `concept_class_id` check no
`_candidate()` call ever populated, to a `domain_id`-only check; plus the
lab-procedure-vs-observable-entity confusion above). Rather than trust
that the fix recovers precision without re-measurement, Tier 2 is held out
of AUTO pending a fresh-note evaluation — see the latest-results doc for
where that stands.

**Consensus calibrator** (`src/mollm_tier_calibrator.py`,
`ConsensusCalibrator`) — a 16-feature logistic regression (vote-consensus
shape, retrieval provenance, fragile-fallback flags, prior-confirmation
count) scoring `P(correct)` for entities the ensemble did not agree on
unanimously. `CALIBRATED_AUTO_THRESHOLD = 0.72`. Full feature-by-feature
detail, the exact learned coefficients, and where 0.72 came from:
`docs/ConsensusCalibrator_Technical_Reference.md`.

**Two distinct call sites, sharing one helper (`_score_with_calibrator()`)
but not the same validation history**: call site 1 scores a genuine
split vote (`TIER_1B`, above) and is the population the calibrator was
actually fitted and validated on. Call site 2 scores a *unanimous*
Tier-2 re-rank (`TIER_2B`, above) — a materially different feature
distribution (100% `is_ambiguous`) the calibrator was never fitted on;
`TIER_2B` is promoted by routing decision but **deliberately excluded
from `AUTO_TIERS`** pending its own shadow validation.

Two hard "trap" gates bypass the calibrator entirely — checked before
`score()` is ever called, at both call sites — for known-fragile
patterns found via direct false-positive investigation:
`_is_coronary_segment_trap()` (LCX/LMCA-style SapBERT embedding
collapse) and `_is_short_alphanumeric_code()` (two regexes: S1-4/T1-2/V1-6
shaped codes, and bare 3-4-letter uppercase abbreviations like LAD/RCA).
`ConsensusCalibrator.load(..., scoring_note_ids=...)` has a leakage
guard: it silently degrades to untrained if any scoring note overlaps
its own training notes.

**`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED`'s broader population, measured
2026-08-20**: this flag (paired with `CONTEXTUAL_CANDIDATES_ENABLED`)
keeps Step B's per-candidate evaluation running past the first "yes"
instead of stopping there, specifically to fix the wound-dehiscence-class
SNOMED duplicate-concept pattern it was built for (verified working on
that narrow pattern). Its cost was already known (~34% more LLM calls);
its accuracy on the broader population it also touches (any entity where
a model independently accepts 2+ candidates, not just the one pattern)
was an open question until measured this session:
`evaluation/exhaustive_candidate_eval_impact.py` found that population's
precision at only **14.3%** (3/21), vs. **84.7%** (265/313) for entities
that don't trigger it — a 70pp gap, on a 5-note test scope. The flag
remains default-on (its narrow verified win is real); a proposed
mitigation (route the 2+-accept population straight to HITL instead of
paying for the comparative tiebreak call) is identified but **not
implemented**.

**Rules-based guideline KG bridge** (`src/guideline_evidence.py`,
2026-08-20) — closes a real proposal gap (Objective 2: "deterministic
context injection from established medical guidelines") that was never
actually wired into the production gate before this session. When 2+
candidates are independently accepted (the same tiebreak population as
`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` above), `_guideline_evidence_block()`
splices real evidence from the guideline corpus into the tiebreak prompt
— matched by **concept name + soft type compatibility, not SNOMED
code** (the corpus's own curators flagged several codes
`same_snomed_type_mismatch_not_merged`, so code-based matching was
judged less trustworthy than the corpus's own names). Additive only,
never a deterministic override — the model still has to reason about it.
**`GUIDELINE_EVIDENCE_ENABLED` defaults OFF**, same validate-before-flip
discipline as the other flags on this page: real, measured coverage is
67 of 7,151 distinct candidate concept names (0.9%) — modest, reported
honestly, pending its own validation batch before touching the live
tiebreak prompt. Full mechanism, matching rules, and a real evidence
block computed live against the actual index:
`docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md` §8.2.

**Escalation experiments (built, measured, deliberately shelved as of
2026-08-20)** — three distinct 8B-model (`llama3.1:8b`) architectures for
hard-case resolution were built and tested against real gold: from-scratch
KG-grounded re-evaluation (`src/tier4_kg_escalation.py`, 27.8% precision),
Stage 2b generate-then-verify candidate augmentation
(`src/tier2b_llm_candidate_generation.py`, 10.7% recall recovery), and an
end-of-pipeline arbiter shown the 3B ensemble's own reasoning
(`src/tier4_arbiter_8b.py`, 38.0%→51.0% precision on a properly
note-diversified N=100 sample — the best-performing of the three). None
are wired into `AUTO_TIERS` or the production routing path; all stay as
standalone, independently-runnable modules pending further validation.

### Stage 4 — Routing & Ingestion (`src/hitl_queue.py`, `src/kg3_ingestion.py`, `src/kg3_query.py`)
`AUTO_TIERS` decisions attempt a direct KG3 write via
`ingest_auto_decision()` (imports `AUTO_TIERS` directly from
`mollm_tier_gate` so the two can never silently drift apart — a real bug
of exactly that shape was found and fixed once already). Everything else
enqueues into `hitl_review_queue` via `enqueue_pending_cases()`, reviewed
through `ui/pages/2_🩺_HITL_Review_Queue.py`. Patient-instance facts are
modeled as `:PatientObservation` nodes linked via `:INSTANCE_OF` to
`:Concept` nodes in Memgraph, keeping patient data structurally distinct
from the reference ontology. **All writes remain `dry_run=True`** — no
code path writes to KG3 unreviewed and live today.

### Stage 5 — Active Learning (foundations only)
Scoped to foundations in this build: the abbreviation flywheel and the
consensus calibrator's `prior_confirmation_count` feature are the two
concrete feedback-loop mechanisms actually running today. A full dynamic
KG3-driven feedback loop into GLiNER's own prompt/search space (as
originally proposed) is not built.

## Concurrency & Operational Infrastructure

DuckDB is single-writer-exclusive. Two real production incidents (data
loss from a too-short retry timeout; full starvation from parallel
Stage1-2b/Stage3 execution) led to the current discipline:

* **`src/db_utils.connect_with_retry()`** — connections are opened right
  before a DB operation and closed right after (not held for an entire
  multi-minute batch), with bounded retry on a lock-conflict specifically
  (any other DuckDB error raises immediately, unretried).
* **`scripts/snapshot_refresh.py`** keeps a separate snapshot file in sync
  so Streamlit can browse continuously while a batch job holds the live
  DB's write lock, rather than being locked out for the batch's entire
  duration.
* **`is_stale`** column on `normalized_entities` (migrated via
  `scripts/mark_notes_stale.py`) — a durable DB-level flag for "processed
  by current code," replacing a hand-maintained cutoff timestamp in
  application code. The UI pages filter on it directly.
* Streamlit pages deliberately do **not** `@st.cache_resource` their DB
  connections — a cached connection lives for the whole server process,
  which would exclude every background batch job for as long as the page
  has ever been visited, not just while actively in use.

## Evaluation Methodology

* **`scripts/score_gold_recall.py`** — the shared gold-loading/matching
  primitives (`load_gold()`, `overlaps()`, `attach_snomed_codes()`,
  `best_tier()`) every other evaluation script builds on. Character-offset
  overlap (`overlaps()`) is "any overlap," not exact-span match.
* **Three different "correct"s, one per stage** — `evaluation/stage_calibration.py`'s
  own docstring: span-is-real (2a), concept-is-right (2b),
  AUTO-tier-decision-is-right (3). ECE/IoU values are never compared
  directly across stages, only each stage's curve against its own
  threshold.
* **`evaluation/iou_metrics.py`** — two genuinely different IoU
  definitions, not variants of one formula:
  - **Set IoU** = `TP / (TP + FP + FN)` at the decision level, defined per
    stage as above — the generalization of Jaccard to any binary
    correct/incorrect decision set.
  - **Benchmark char IoU** (`benchmark_char_iou()`) — the DrivenData
    SNOMED-CT entity-linking benchmark's own definition
    (https://www.drivendata.org/benchmarks/310/benchmark-snomed-ct/page/983/),
    confirmed 2026-08-20 by reading the metric section directly: "class" =
    SNOMED concept ID (not an entity-type taxonomy — gold's own
    `concept_id` column IS the class field), and a predicted span's
    characters only count toward a class's set if that span's own resolved
    concept exactly matches the class ("relationships between concepts are
    not taken into account for scoring"). Character positions are keyed by
    `(note_id, offset)`, not raw offset alone, so scoring several notes at
    once can't spuriously collide two unrelated notes' spans that share a
    numeric offset. Computed at Stage 2b (the first stage with a resolved
    concept at all) using each entity's top candidate as "our answer" — a
    real benchmark submission has no HITL-deferral option. Stage 2a's own
    `span_only_char_iou` is a deliberately concept-blind diagnostic, not
    a stand-in for this metric.
* **`evaluation/tier_gate_grading.py`** — general-purpose per-tier grading
  (`grade_by_tier()`), used by both the UI and batch scripts. Its default
  tier list does NOT automatically track `AUTO_TIERS` — pass an explicit
  tier list when grading a tier that was deliberately excluded from
  `AUTO_TIERS` (like Tier 2 currently), or it silently disappears from the
  report.
* Every batch/eval script scopes to `is_test=TRUE` and excludes
  `superseded_by_split`/`superseded_by_growth` rows (a replaced entity and
  its replacement would otherwise double-count).
* **`evaluation/kg_tiebreak_validation.py`** (2026-08-20) — sweeps a
  `TIE_THRESHOLD` over SapBERT top1/top2 score deltas, grades the KGE
  tiebreak's pick against gold (win/loss/neutral, defined as: baseline
  wrong→mechanism right = win, baseline right→mechanism wrong = loss,
  anything else = neutral), and reports a head-to-head against the
  hardcoded rule specifically on the subset where both apply. Reads
  Stage 2b's already-stored candidate scores — zero live SapBERT calls,
  deliberately, after an earlier live-recompute attempt measurably
  stalled a concurrently-running Stage 3 batch via model-load contention.
* **`evaluation/exhaustive_candidate_eval_impact.py`** (2026-08-20) —
  detects the `EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` tiebreak-eligible
  population directly from a stored decision's `models[i].eval_trail`
  (any model with `sum(1 for t in trail if t.get("match")) >= 2`), grades
  it against gold, and compares precision to the non-eligible population
  — the net-impact half of that flag's cost/accuracy tradeoff, which its
  cost-only measurement (2026-08-19) had left open.

## UI (`ui/`, Streamlit)

Four pages, all reading live or snapshot DuckDB state, never
authoritative — they visualize what the pipeline already computed:

1. **Pipeline Runner** — run Stage 1→2b (and optionally Stage 3) on a note
   interactively, inspect the entity trace.
2. **HITL Review Queue** — the human-review interface for everything not
   in `AUTO_TIERS`; the only page that opens a write connection. Rebuilt
   2026-08-20: shows the **full raw note** with the entity highlighted at
   its real character offset (not just the sentence-bounded local
   context), side by side with **every model's full reasoning trail**
   (Step A's clinical-meaning judgment, then every Step B candidate
   accept/reject in order, plus any tiebreak resolution) — not just the
   final verdict, so a reviewer can catch a model that reasoned correctly
   then contradicted itself in its own final answer. Approve / Correct /
   Reject, plus a free-text comment on every decision (not just
   rejections) that feeds back into the abbreviation flywheel.
3. **Troubleshooting** — step-by-step input/output walkthrough for one
   note, including a character-level diff highlighter
   (`render_highlighted_note()`/`_split_overlap_spans()`): green = our
   entities, blue = abbreviations, gold/amber = gold annotations, and
   where our span and gold's span disagree on start/end, grey marks the
   exact overlapping characters with each side's non-overlapping
   extension kept in its own color.
4. **Evaluation Metrics** — ECE + IoU (including the benchmark char IoU
   above) per stage, plus static calibrator metadata.

## Current Headline Result — Fresh-Note Held-Out Validation

**AUTO-tier precision: 76.8% (43/56)**, **deflection rate: 31.2%
(78/250)**, **linked concept-level F1: 33.7%** -- measured 2026-08-20 on
10 notes from the official locked test split
(`data/splits/note_splits.csv`) that were not used to develop or debug
this session's SNOMED near-duplicate retrieval fix or the KGE evaluation,
and were (for 7 of the 10) also outside the tier-gate calibrator's own
training set. These are the recommended numbers to cite as the
pipeline's current performance -- not the higher corpus-wide figures
(86.9% precision, 57.0% deflection, 40.1% F1, measured on all 144
`is_test` notes) which are inflated by including notes used during
active development; the ~10pp precision gap and ~26pp deflection gap
between the two is itself a real, reportable measure of how much of the
corpus-wide number is genuine generalization vs. fitting to what was
debugged on. Full breakdown (including the annotation-velocity/
cost-effectiveness comparison) in
`docs/2026-08-20_Session_Results_And_Status.md` §13 and §15; the note
list lives in `ui/components/fresh10_notes.py`, also used to scope the
Streamlit demo pages to this validated population.

## Baseline Comparison

**Corrected 2026-08-27**: an earlier version of this section described a
three-longitudinal-checkpoint Clinical-T5 comparison (KG3 empty,
~half-processed, fully processed) as if it were completed methodology.
No record of that comparison actually being run exists in this session
or in any results doc, and no numbers for it are cited anywhere — it
appears to be planning text carried over from the original proposal
draft, not a description of completed work. Stated honestly rather than
implied as done:

Clinical-T5 was removed from the live pipeline entirely (Stage 2a
relation extraction) due to a real data-contamination risk — it was
pretrained on MIMIC-III/MIMIC-IV, which may include the evaluation
notes themselves. It is intended to run as an **independent baseline**
on the same notes, through the same SapBERT/DuckDB grounding step so
both systems are scored at the concept-ID level rather than on raw text
overlap — but this comparison has **not been executed**. It remains
planned future work, not a reported result. See
`docs/FINAL_RESULTS_Single_Source_Of_Truth.md` for what has actually
been measured.
