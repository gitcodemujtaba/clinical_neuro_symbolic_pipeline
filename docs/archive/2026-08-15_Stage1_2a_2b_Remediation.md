# Stage 1/2a/2b Remaining-Gaps Remediation — Session Report

**Date:** 2026-08-15
**Scope:** `src/preprocessing.py` (Stage 1), `src/entity_extraction.py`/`src/extraction.py` (Stage 2a), `src/normalization/*.py` (Stage 2b)
**Status:** All 7 checklist items resolved — 4 were already implemented (checklist was stale), 3 needed real code changes, all now done and verified. Plus a related, separately-decided item (Tier 3 hard cutoff) empirically validated.

---

## 1. Why this session happened

A summary of "what's left in Stage 1/2" was requested, sourced from `docs/Implementation_Checklist.md`. Before answering, that checklist was cross-checked against the actual current code rather than trusted as-is — it's dated 2026-08-07/08-11 and had already proven stale multiple times this session (domain filtering, the Tier-3 floor). The cross-check found **4 of 7 "open" items were already fully implemented**, just never checked off; only 3 were real gaps. This doc records what was verified, what was actually built, and what was audited-and-closed.

## 2. Already done — checklist was stale, no code change needed

- **`abbreviation_dict_version` field** — `src/preprocessing.py:109-121` (`_dict_version()`, sha256 hash of the abbreviation table), landed 2026-08-14 (commit `6bc4b6f`).
- **Per-expansion `ambiguous` flag** — `src/preprocessing.py:372,395-406`, same commit. Stage 2a's `expansion_ambiguous`/`candidate_expansions` are copied forward from this same field, not an independent gap.
- **`extracted_relations` entity_id FKs** — `relation_id`/`head_entity_id`/`tail_entity_id` are real, populated columns (`src/extraction.py:220-252`), resolved via deterministic character-offset overlap (`_link_endpoint()`/`_overlap_ratio()`, `MIN_ENDPOINT_OVERLAP=0.50`), not fuzzy text matching.
- **OMOP `domain_id` filtering on Tier 3, and separately on Tier 2** — both already implemented (`domain_clause`/`domain_clause2` in `src/normalization/tier_retrieval.py` and `orchestrator.py`), built out across the 08-13/08-14 sessions after these checklist items were written.

## 3. Real fixes implemented this session

### GLiNER truncation — surfaced, not fixed
`src/entity_extraction.py`'s `extract_and_store_entities()` now wraps `model.predict_entities()` in a scoped `warnings.catch_warnings()` and parses GLiNER's own truncation `UserWarning` ("Sentence of length N has been truncated to 2048") directly — previously silently swallowed by the module's blanket `warnings.filterwarnings("ignore")`. Two new columns on every `extracted_entities` row: `possibly_truncated`, `gliner_input_token_count`. **Deliberately does not chunk/re-extract** long notes to recover the truncated tail — that's a real, separate feature with real span-boundary risk, the same category already deferred for compound-span splitting in `docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md`. Query `SELECT DISTINCT note_id FROM extracted_entities WHERE possibly_truncated` to measure real impact before deciding whether to build it.

### Pure-function extraction refactor
`extract_and_store_entities()` now takes `provenance: dict` as a required parameter instead of re-querying `note_expansions` internally. `run_pipeline()` (`src/clinical_pipeline.py`) passes the `stage1_provenance` it already holds from `process_and_store_note()`; `scripts/test_extraction.py` updated the same way. `store_entities()` remains the sole, separate persist step — it already was one, this was the only remaining impurity. Verified via a real end-to-end smoke test (fresh note through `run_pipeline()`, confirmed correct output including the new truncation columns) plus `pytest tests/` (47/47).

### Stage 2b dedup-key canonicalization
`process_and_normalize_entities()`'s cache key now collapses whitespace and folds case **for the key only** — `normalize_entity()` itself still receives the unmodified `expanded_text`. Deliberately does not touch Tier 1/2's actual SQL matching (confirmed case-sensitive, no `LOWER()`/`ILIKE`), since `docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md` already documents a real case-sensitivity collision (`CTA`/`cTa`) as a **deliberately deferred** trade-off — folding case in the matching logic itself would silently re-decide that separate, already-weighed decision. Verified directly: `"abd distension"` / `"ABD distension"` / `"abd  distension"` now all resolve to the same concept (`4012079`, "Abdominal distension symptom").

## 4. Audited and closed — no fix needed

**Brand-name `athena_concept_synonym` coverage.** Checked 15 common brand names directly against real data. `athena_concept_synonym` genuinely has near-zero brand-name coverage, but it's functionally irrelevant: every real brand name tested resolves at **Tier 1** directly (RxNorm's `concept_class_id='Brand Name'` concepts use the brand name as `concept_name` itself), and `_alias_expand_brand_to_generic()` (built 2026-08-13) already crosswalks to the generic ingredient. The checklist's own motivating example — `lasix` "failing Tier 1/2" — does not reproduce (2 direct Tier 1 hits today). The other two named examples (`spirnolactone`, `bioplar`) are misspellings with zero matches anywhere; that's a fuzzy-matching problem (already handled by `_merge_fuzzy()`), not a synonym gap.

## 5. Related, separately-decided item: Tier 3 hard cutoff (0.72)

Not part of this remediation pass originally, but decided and validated the same session, so recorded here for completeness. `src/normalization/orchestrator.py`'s Tier 3 floor branch now returns `NO_CANDIDATE` (was: forwarded weak candidates flagged `failed`) when top score < `TIER3_SIMILARITY_FLOOR=0.72` and no domain-conflict rescue applies — implemented at the user's explicit direction after being shown the 08-13 calibration finding that 0.72 doesn't cleanly separate signal from noise at any point on the curve.

**Full 27-note corpus re-check** (2,303 entities, fresh `normalize_entity()` calls vs. stored production output, both graded against gold):

| | OLD (forwarded) | NEW (hard cutoff) |
|---|---|---|
| Precision (of gradable) | 47.9% | 50.7% |
| Coverage (gradable at all) | 71.9% (1656/2303) | 59.7% (1376/2303) |

Of 369 entities that flipped to `NO_CANDIDATE`: 190 were genuine garbage (correctly removed), but **90 were genuinely gold-correct matches now lost outright** (e.g. `WBC-13.0`→`White blood cell count` at 0.8694). GOLD_MISSING — already this pipeline's dominant remaining gap at 60.7% — grows by roughly this many entities, with zero remaining downstream chance of recovery (previously Stage 3's resolution mode at least had the weak candidate to reason over). Kept as implemented pending final user decision (keep / revert / soften to "forward but harder-flag").

## 6. Where to pick up next

1. **Decide the Tier 3 hard-cutoff trade-off** (§5) — keep as-is, revert, or soften.
2. **Measure real truncation impact**: run the `possibly_truncated` query across the full note corpus, not just the 27-note baseline, to know if chunk-and-merge extraction is worth building.
3. Everything else in this doc is closed — no further action needed on the 7 original checklist items.
