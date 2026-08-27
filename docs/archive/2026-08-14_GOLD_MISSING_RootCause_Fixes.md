# GOLD_MISSING Root-Cause Fixes and Near-Synonym Ranking — Session Report

**Date:** 2026-08-14 (later the same day as `docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md`)
**Scope:** Stage 2b (OMOP concept-linking, `src/normalization/*.py`), Stage 3 routing/prompting (`scripts/experiment_3b_voting.py`, `src/mollm_ensemble.py`)
**Status:** All fixes implemented, regression-tested (47+33 test suites), and validated against a genuine full Stage 1→2a→2b→3 corpus re-run.

---

## 1. What triggered this session

The prior session (`docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md`) measured **GOLD_MISSING at 63.3%** of gradable Stage 3 failures — the gold SNOMED concept never appearing in Stage 2b's candidate list at all — and flagged it as "the single biggest lever in the whole pipeline... NOT addressed by anything this session." This session root-caused it directly, plus a companion problem (a specific entity, `WBC-13`, initially theorized as an "ontology drift" case where SapBERT retrieves a plausible-but-wrong concept) that on direct investigation turned out to be a distinct, more precise bug: a near-synonym ranking problem, not a coverage gap.

The investigative discipline throughout: verify every hypothesis against real data (DB queries, direct `normalize_entity()` calls) before writing a fix, including hypotheses proposed by the user — several plausible-sounding theories were directly disproven this way before any code changed. See the `feedback-empirical-validation-before-fixing` memory entry for the pattern.

## 2. Fixes implemented (`src/normalization/tier_retrieval.py` unless noted)

### Fix 1 — `_collapse_hierarchy_duplicates()` discarding valid compound concepts
Built 2026-08-13 to collapse near-duplicate SNOMED parent/child concepts (the ALT-measurement trio) into one representative, keeping the highest-scored member. Found silently discarding a legitimate *combined* concept — "Intermaxillary fixation of mandible **and** maxilla" (gold) — in favor of a higher-scored single-component sibling ("...of mandible" alone), because a parent/child Subsumes edge doesn't always mean "same fact, different specificity" (the case this function was built for); sometimes it means "genuinely different scope." Fixed with `_is_compound_concept_name()`: any candidate whose name contains a coordinating conjunction ("and"/"&") is exempted from the collapse and always survives. Verified this doesn't regress the original ALT case (still correctly collapses to one, since none of that trio's names are compound).

### Fix 2 — `_detect_domain_conflict()` surfacing SNOMED Organism-hierarchy false positives
This function (built prior session) re-runs Tier 1-3 without the domain restriction after a filtered miss, to catch genuine GLiNER mislabeling. Found it could also surface SNOMED's biological Organism hierarchy purely on embedding proximity to a short/polysemous word — "galea" (Latin: helmet) scored 0.90 against "Genus Galea" (a rodent genus) vs 0.61 for the correct anatomy concept "Structure of galea aponeurotica". Fixed by excluding `concept_class_id = 'Organism'` from conflict candidates. Explicitly checked and did NOT exclude `concept_class_id = 'Event'` — confirmed all ~3,561 such concepts in this vocab are legitimate COVID-exposure/contact-tracing Observations, not noise.

### Fix 3 — same function, vocab restriction too narrow for generic drug-class mentions
The domain-conflict retry relaxed `domains` but kept the entity label's own `vocabs` restriction, so a `Medication`-labeled generic mention ("diuretics") could never find gold's answer when gold's concept is a SNOMED Procedure-domain "X therapy" concept (RxNorm has no equivalent class-level concept, only ingredients/products). Fixed by retrying with `vocabs=DEFAULT_VOCAB` when the first (domain-relaxed, vocab-restricted) attempt still fails. No-op for every other label, which already defaults to `DEFAULT_VOCAB`.

### Fix 4 — Procedure-vs-Observable-Entity near-synonym re-rank (`_prefer_lab_procedure_over_observable()`)
The actual root cause of the `WBC-13` "ontology drift" case: for a lab test, SapBERT consistently scores the SNOMED Observable-Entity-class concept (e.g. "Leucocyte count", the abstract property) higher than the Procedure-class concept for the same test ("White blood cell count", the act of measuring) — 0.892 vs 0.8694 for WBC. Measured directly against real gold data across every Lab-Test entity in the 27-note corpus with both classes present: **Procedure-class is gold-correct 78/78 times, zero exceptions**, whenever either is correct at all. A real, exceptionless DrivenData annotation convention, not a heuristic guess. Implemented as a rank-only penalty (never touches the displayed `similarity_score`).

**Implementation trap hit here, worth remembering:** `normalize_entity()` in `orchestrator.py` has its own separate, duplicated inline Tier-3 code block — it does not call the shared `_tier_queries()` helper (only `_detect_domain_conflict()`'s fallback path uses that). The fix was first wired only into `_tier_queries()` and silently did nothing on the main path; caught only by testing the actual entity through the real call path, not the helper in isolation. Both call sites now patched.

### Decided against: raising `CANDIDATE_LIMIT`
Checked empirically before deciding: raising the limit from 3 recovers only 14.6% of remaining GOLD_MISSING cases at limit=10, 34.3% at limit=50, steeply diminishing. Rejected — conflicts with this same session's Stage 3 finding that longer candidate lists measurably hurt 3B model accuracy (see §4).

### Ruled out: SNOMED CT edition/version mismatch
100% of the 6,595 distinct gold SNOMED codes exist in the local `athena_concept` dump (96.5% as `standard_concept='S'`). If gold used a different edition, some meaningful fraction would be completely absent; none are.

### Two more pre-existing bugs found (unrelated to any of this session's own changes)
`src/normalization/compound_span.py` used `_tokens_with_offsets`/`_trim_connectors` (from `text_utils.py`) and `orchestrator.py` used `_LAB_TIER_RANK` (from `compound_span.py`)/`SAPBERT_POOLING` (from `sapbert_model.py`) — all four without importing them. A latent gap from the 2026-08-14 morning module split, only surfaced because this session's full pipeline re-run was the first genuine end-to-end exercise of that code path since the split. All four imports fixed; an AST-based undefined-name checker (written ad hoc, works around `pyflakes`'s blind spot on `from .constants import *` wildcard imports) confirmed zero remaining gaps across `src/normalization/*.py` afterward.

## 3. GOLD_MISSING root-cause breakdown (what's fixed vs. explicitly deferred)

| Cause | Size (of remaining misses after this session) | Status |
|---|---|---|
| Hierarchy-dedup discarding valid compound concepts | ~20% of "contradiction" (RANK≤3) bucket | **Fixed** |
| SNOMED Organism false positives | smaller, real | **Fixed** |
| Vocab too narrow for generic drug-class mentions | real | **Fixed** |
| CANDIDATE_LIMIT cutoff | 26-34% of remainder | **Declined** — diminishing returns |
| Genuine SapBERT embedding-coverage gaps | ~50% of RANK>50 bucket | **No safe fix** — needs better retrieval/model, not a logic bug |
| Compound spans (symptom+anatomy, etc.) | 19.5% of remaining misses | **Confirmed real, deferred** — Stage 2a semantic-splitting feature, not a Stage 2b fix |
| Context-dependent span completeness (e.g. "cancer" vs gold's "colon cancer") | — | **Deferred** — Stage 2a extraction-completeness problem |
| Case-sensitivity collision (`CTA` vs `cTa`) | — | **Deferred** — real trade-off, fixing risks more regressions than it solves without corpus-wide A/B data |

## 4. Full-corpus validated results

Genuine Stage 1→2a→2b→3 re-run on the 27-note baseline corpus (753→759 entities after the fixes), not just direct-call spot checks:

| Metric | Original baseline | Stage-3-fixes only | **All fixes combined** |
|---|---|---|---|
| AUTO_VALIDATED precision | 33.8% | 37.4% | **39.4%** |
| MOLLM_RESOLVED precision | 40.5% | 40.6% | **53.7%** |
| HITL_REQUIRED precision | 48.5% | 51.7% | 43.3% |
| AUTO_VALIDATED coverage | 41.0% | 33.6% | 41.2% |
| Deterministic bypass rate | 0.0% | 0.0% | **7.2%** |
| GOLD_MISSING | 63.1% | (unchanged) | **60.7%** |
| Lab-value-entity precision | 34.5% | (not isolated) | **64.1%** |

The "Stage-3-fixes only" column reflects three companion Stage 3 fixes made the same session: a lab-value display note (candidate generation for e.g. "WBC-13.0" was already correct via `strip_lab_value_suffix()`'s existing fallback — the gap was purely that the raw value-bearing text reached the LLM prompt unexplained), an entity-type prompt anchor, and a "Fragile Concept Gate" (a 3-0 unanimous vote is capped at MOLLM_RESOLVED, not promoted to AUTO_VALIDATED, when the winning candidate was only resolved via the lab-value-suffix salvage fallback — gated on that precise `normalized_from` provenance field, not the broader `gliner_label=='Lab Test' AND match_tier==3` heuristic, which was checked and would have wrongly caught 40% of clean, legitimate resolutions). Also fixed the same session: a production bug where superseded (already split/grown) entity rows still reached Stage 3 validation alongside their own split children (`src/mollm_ensemble.py::load_validation_records()`).

## 5. Regression testing

All changes pass the full test suite (47 pytest-collected + 33-test `test_tier12_ranking.py`), re-run after every fix.

## 6. Where to pick up next

1. **GOLD_MISSING is still 60.7%**, dominated by the two deferred causes (embedding-coverage gaps, compound-span splitting). Neither has a safe, bounded fix — the embedding gap needs a better retrieval/model, and compound-span splitting needs new Stage 2a semantic logic (the existing `find_compound_split()` only handles lexical/whitespace-tokenizable compounds).
2. **AUTO_VALIDATED-precision-inversion is improved but not fully closed** in the experimental-harness measurement (§4). The production Stage 3 batch run done immediately after this session (see `docs/2026-08-15_Stage4_Stage5_Build.md`) shows the *correct* tier ordering (AUTO_VALIDATED highest) at 52.6% precision — still far from safe-to-skip-review, and the reason Stage 4 queues every decision regardless of tier for now.
3. `evaluation/stage_calibration.py`'s own `FROM normalized_entities` query still lacks the `superseded_by_split`/`superseded_by_growth` filter applied everywhere else this session — confirmed low-priority (offline measurement script, not a live decision path) and explicitly deferred, not forgotten.
