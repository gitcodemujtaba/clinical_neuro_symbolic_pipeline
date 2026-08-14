# Stage 2 Alias/Dedup Fixes and Stage 3 Provenance-Gated Reasoning — Session Report

**Date:** 2026-08-13 → 2026-08-14 (overnight session)
**Scope:** Stage 2b (OMOP concept-linking, `src/normalization.py`), Stage 3 (MoLLM ensemble, `src/mollm_ensemble.py` and the experimental smaller-model harness `scripts/experiment_3b_voting.py`)
**Status:** All code changes implemented and regression-tested. NOT committed to git. NOT re-run through the full corpus pipeline since the last code change (see "Critical caveat for tomorrow" below — this is the most important thing to read before trusting any test against live DB data).

---

## 1. What triggered this session

Reviewing `3b_voting_results.json` (a prior 3-model MoLLM voting experiment) surfaced two dominant Stage 2b failure patterns:

1. **Recall gap ("the Lasix problem")**: brand-name drug mentions (Lasix, Levophed, Aldactone, Prilosec...) never had their generic active ingredient in the Stage 2b candidate list at all, because SapBERT's embedding space doesn't reliably place a brand name near its ingredient. All three voting models could correctly *reason* "Lasix is a brand name for furosemide" and still vote `NONE_CORRECT`, because furosemide simply wasn't an option.
2. **Near-duplicate candidate splitting**: SNOMED `Is a`/`Subsumes` parent/child concepts (e.g. three different "ALT measurement" concepts) got retrieved together and fractured MoLLM voting into artificial 2-1 or 1-1-1 splits.

## 2. Stage 2b fixes implemented (`src/normalization.py`)

### Fix 1 — `_alias_expand_brand_to_generic(conn, search_text)`
Deterministic KG traversal for brand→generic resolution. Verified empirically there is **no 1-hop** relationship from a brand concept to its ingredient; the real path is 3 hops:
```
brand -[Brand name of]-> branded product (Branded Drug Comp)
      -[Tradename of]-> generic Clinical Drug Comp (standard, RxNorm)
      -[RxNorm has ing]-> Ingredient (standard, RxNorm)
```
Returns only the final Ingredient-level concept_ids (not the hundreds of intermediate dose-specific SKUs a well-established brand like Lasix has — that was an early bug, fixed: returning the full `generic` CTE exploded to 143 ids for Lasix before narrowing to just `ingredient` collapsed it to the correct single id, 956874 furosemide).

### Fix 2 — `_collapse_hierarchy_duplicates(conn, cands)`
Union-find over direct `Is a`/`Subsumes` edges within a candidate set; collapses each connected component to its single highest-similarity member. Confirmed empirically: the ALT-measurement trio (4095055, 4146380, 44810789) is a real SNOMED parent/child hierarchy, not literal duplicate concept_ids — a plain `GROUP BY concept_id` would not have caught it.

### Wiring
Both fixes are wired into `_tier3_semantic_rows()` (shared by `normalize_entity()`'s main path and `_tier_queries()`'s domain-conflict-recovery path). Alias candidates are force-included in the Tier 3 SQL via a `UNION` with the natural top-K, scored by their own real cosine similarity (not pinned to 1.0).

### Bug found and fixed — the "Aldactone problem"
`process_and_normalize_entities()` only persists the full `candidates[]` array when `ambiguous=True`; a confident top-1 pick ships only itself (line ~2211, `mapping["candidates"][:1]` otherwise — a pre-existing space-saving optimization). Aldactone → propiolactone (0.83 similarity, confidently wrong) cleared both the floor and margin checks, so the alias-injected spironolactone (0.56, correct) got computed by Fix 1 and then silently discarded one function later. **Fixed** by forcing `ambiguous=True` with a new reason `alias_candidate_outranked` whenever an alias candidate exists in `cands` but isn't `cands[0]` (`normalize_entity()`, end of the Tier 3 block). This was caught by direct DB inspection of a completed pipeline run, not by unit tests — worth remembering as a class of bug (silent downstream truncation) that only inspection catches.

## 3. Validation performed and results

Two full pipeline re-runs on the same 28-note baseline set used in earlier sessions (`10043750-DS-6, 10371195-DS-9, 10848570-DS-12, 10860165-DS-24, 10912090-DS-33, 11134545-DS-21, 11532659-DS-11, 11649745-DS-4, 11838076-DS-20, 11997336-DS-3, 12128814-DS-15, 12247014-DS-9, 12314513-DS-16, 12545016-DS-17, 12962702-DS-14, 12970259-DS-4, 13164440-DS-18, 14102739-DS-16, 14280440-DS-8, 14975962-DS-19, 15853461-DS-4, 16393593-DS-5, 16991646-DS-11, 17739994-DS-31, 18570237-DS-10, 14490470-DS-11, 17751158-DS-19, 19442119-DS-15`), via `scripts/test_pipeline_e2e.py --note-ids <28 notes>` (~1.5–2h each, GPU-accelerated). Second run was needed specifically to pick up the Aldactone truncation fix.

### Gold-recall benchmark (`scripts/score_gold_recall.py`)
| | linked_recall | span_recall | linked_correct |
|---|---|---|---|
| Pre-session baseline | 10.14% | 21.05% | 963 |
| Post-fix | 10.53% | 21.80% | 1000 |

**Caveat**: `predicted_entities` also shifted (2824→2996) despite these fixes never touching Stage 2a extraction — some of this movement is confounded by other already-uncommitted Stage 2a work in the tree, not cleanly attributable to this session alone. Also: **this benchmark structurally excludes Medications** (the SNOMED CT Entity Linking Challenge's gold set only annotates Procedures/Body Structures/Clinical Findings), so it is *blind to Fix 1's actual target* — the real validation of Fix 1 came from direct DB/candidate inspection, not this number.

### Direct candidate verification (the real Fix 1 test)
Confirmed via `normalized_entities.candidates` inspection: Lasix→furosemide, Levophed→norepinephrine, Prilosec→omeprazole, Aldactone→spironolactone (only after the truncation fix) all now present in the candidate list for every occurrence in the 28-note corpus.

### ECE / calibration (`evaluation/stage2a_cal_eval.py`, `stage2b_cal_eval.py`, `cal_eval.py`)
| Stage | ECE | Freshness |
|---|---|---|
| 2a (GLiNER extraction confidence) | 0.1446 | Fresh — unaffected by this session (Stage 2a untouched) |
| 2b Tier-3 (SapBERT similarity_score) | **0.4986** | Fresh — the smoking gun: top similarity bin (0.90–1.00) is only 55% accurate, 0.70–0.80 bin only 19% accurate. Direct quantitative confirmation that raw cosine similarity is a bad confidence signal, motivating everything in §4 below. |
| 3 (MoLLM composite_confidence) | 0.762 | **Stale** — from `mollm_decisions` predating this session's fixes entirely; not re-measurable without a fresh Stage 3 corpus run |

### 3-model voting re-run + audit (`scripts/experiment_3b_voting.py`, small local Ollama models: qwen2.5:3b, llama3.2:3b, phi4-mini — NOT the production BioMistral/OpenBioLLM stack)
Ran all 768 LOW-tier entities across the 28 notes (~38 min, GPU). Audited against gold with a 3-category breakdown (script: scratchpad `audit_3b_voting.py`, not yet copied into the repo):

| Category | n | % of gradable (510) |
|---|---|---|
| GOLD_PRESENT_CORRECT | 113 | 22.2% |
| GOLD_PRESENT_REJECTED (model saw gold, said NONE_CORRECT) | 33 | 6.5% |
| GOLD_PRESENT_OUTVOTED (model saw gold, picked a distractor) | 41 | 8.0% |
| **GOLD_MISSING (gold never in candidate list — Stage 2 recall gap)** | **323** | **63.3%** |

**Key finding**: the Stage 3 prompt/provenance problem (REJECTED+OUTVOTED) is real but modest — 14.5% of gradable cases. GOLD_MISSING dominates at 63.3%, even after this session's Fix 1/2. This is now the single biggest lever in the whole pipeline and is **NOT addressed by anything in this session** — see §6.

### GOLD_MISSING root-cause diagnostic (scratchpad `diagnose_gold_missing.py`)
For all 323 missing cases, re-ran Tier 3 with LIMIT 50 (unrestricted) to see if gold was nearby but cut off, or genuinely absent from the embedding neighborhood:

| | n | % |
|---|---|---|
| CANDIDATE_LIMIT cutoff (gold in top-50, just not top-3) | 100 | 31.2% |
| **Genuine embedding-space miss (gold not even in top-50)** | **220** | **68.8%** |

Raising `CANDIDATE_LIMIT` to 10 would only recover 54/320 (16.9%) of all missing cases — not the fix. Reading the actual misses, four distinct patterns:
1. **Compound spans** (`'right EVD placement'` — gold wants separate codes for right/EVD/placement; Stage 2a's job, not Stage 2b's) — confirmed only ~3.1% overlap with the existing `compound_spans` tracker, so this is a real but minor slice.
2. **Rare/specialized single terms poorly embedded** (`'galea'` → `Genus Galea` the rodent; `'tracheostomy'`, 3/3 occurrences missing).
3. **Lay-phrasing → clinical-term gaps** (`'bloody cough'` → gold's real code, likely hemoptysis-related, isn't in top-50 at all).
4. **Route/qualifier noise burying the ingredient** (`'subcutaneous heparin'` → all shown candidates are branded injectable SKUs; same shape as the Lasix problem but no brand name, so Fix 1 doesn't reach it).

## 4. Stage 3 provenance-gated reasoning (implemented, NOT yet corpus-validated)

Motivated by direct evidence: llama3.2:3b's own reasoning on Lasix said *"lasix is a common brand name for furosemide, but none of the candidate options match this exact term"* — the model had the clinical knowledge, furosemide was in its candidate list, and it still voted NONE_CORRECT, because nothing in the prompt distinguished "this is a KG-verified fact" from "this merely sounds alike."

### `match_basis` field (`src/normalization.py`, `_candidate()`)
Every candidate dict now carries `match_basis`, defaulting per tier (`exact_text` / `synonym` / `semantic_similarity` / `fuzzy_edit_distance`), overridden to `verified_brand_alias` for Fix-1-injected candidates.

**⚠️ CRITICAL CAVEAT FOR TOMORROW**: this field was added to the code *after* the second full pipeline re-run already completed and wrote `normalized_entities.candidates` to the DB. **The DB's stored candidates for all 28 notes have NO `match_basis` key at all right now.** Any test that reads candidates via `load_entities()` / the DB (rather than calling `normalize_entity()` fresh) will silently default every candidate to `semantic_similarity` and the deterministic bypass (§4.4) will never fire. This caused significant confusion mid-session — several "attention dilution" and "hallucination" findings were partly or wholly artifacts of testing against this stale, tag-less data, not genuine model failures. **A fresh pipeline re-run is required before any corpus-scale Stage 3 test is trustworthy.**

The final validation of the night used fresh, directly-computed `normalize_entity()` calls (bypassing the DB) for exactly this reason, and got a clean result — see §4.5.

### Production prompt updates (`src/mollm_ensemble.py`)
- `_format_candidates()` now renders `basis {match_basis}` per candidate line.
- `SYSTEM_PROMPT` RULES gained three bullets: **PROVENANCE OVERRIDES SPELLING** (trust `verified_brand_alias`/`exact_text` over lexical judgment), **SCORE WARNING** (high similarity ≠ reliable evidence), **SYNONYM TOLERANCE** (don't reject valid paraphrases, e.g. "Right atrial structure" for "right atrium" — this exact case was in the GOLD_PRESENT_REJECTED audit list).

### Experimental script restructure (`scripts/experiment_3b_voting.py` — untracked in git, never committed)
- `load_entities()` now also pulls `assertion_status`.
- `PROMPT_TEMPLATE` rewritten to a structured ENTITY/SECTION/ASSERTION/CONTEXT/CANDIDATES layout with 6 rules: PROVENANCE OVERRIDES SPELLING, SCORE WARNING, SYNONYM TOLERANCE, **SAFETY FIRST** (added after a regression — see below), IGNORE NEGATION, and the original NONE_CORRECT fallback.
- `_format_candidates()` (new helper) renders each candidate as a 3-line block with the Basis visually isolated as `>>> VERIFIED_BRAND_ALIAS <<<` — a fix for a measured "attention dilution" bug where a dense single-line format let 3B models detach the Basis tag from its own `[i]` index and misattribute it to the highest-scored candidate instead.
- New **`--sequential` mode**: `evaluate_candidates_sequentially()` / `run_vote_sequential()` — a 1-to-1 binary evaluation loop (one candidate per LLM call, "is this a match, yes/no", stop at first yes) instead of 1-to-N multiple choice, eliminating bracket/index-tracking entirely. Costs up to `len(candidates)×` more calls per model.
- New **`check_deterministic_bypass()`**: skips the LLM ensemble entirely when exactly one candidate has `match_basis == "verified_brand_alias"`. Deliberately **excludes `exact_text`** — measured this corpus's own Tier 1 (Exact) accuracy at 52.48% (766 entities, 402 correct) via `stage2b_cal_eval.py`, essentially a coin flip, so string-exact-match is not a safe auto-validate criterion. Also deliberately requires **exactly one** alias hit, not "at least one" — a combination brand (Aldactone) can legitimately produce several alias candidates (one per active ingredient), and picking the first by list order would be arbitrary; multiple hits correctly fall through to the ensemble as genuine ambiguity.

### Final validation (fresh data, bypassing the stale DB)
Direct `normalize_entity()` calls for the four known brand cases:

| Case | `verified_brand_alias` candidates | Result |
|---|---|---|
| Lasix | 1 (furosemide) | **Bypass fired** → correct, 0 LLM calls |
| Prilosec | 1 (omeprazole) | **Bypass fired** → correct, 0 LLM calls |
| Levophed | 1 (norepinephrine) | **Bypass fired** → correct, 0 LLM calls |
| Aldactone | 6 (spironolactone, furosemide, buthiazide, canrenoate, canrenoic acid, phosphoric acid — real combo-formulation ingredients) | Correctly deferred to ensemble (genuine ambiguity, not a bug) |

Aldactone's ensemble fallback also reconfirmed a genuine (not data-artifact) finding: llama3.2:3b claimed `propiolactone` (a plain `semantic_similarity` candidate) was `verified_brand_alias` — a real, model-specific hallucination tendency, worth remembering when weighting/trusting this model's votes specifically.

## 5. Regression testing

All changes pass the existing suite: `test_tier12_ranking.py` (33), `test_confidence_tier_reasons.py` (27), `test_degenerate_generation.py` (34), `test_override_gate.py` (37), `test_reasoning_verdict_mismatch.py` (21), `test_offset_mapping.py`, `test_stage3_safety_rules.py` (32) — all passing, re-run multiple times through the session.

## 6. Model choice decision (2026-08-14, end of session)

**The three small local Ollama models (qwen2.5:3b, llama3.2:3b, phi4-mini) are now the standing choice for the MoLLM ensemble going forward**, not a side experiment being compared against the production BioMistral-7B-AWQ/OpenBioLLM-Llama3-8B-AWQ stack. This changes how the rest of this doc should be read: `scripts/experiment_3b_voting.py`'s architecture (provenance-gated prompt, `match_basis`, the deterministic bypass, `--sequential` mode) is the direction to build out, not a throwaway comparison harness. Practically this means:
- `src/llm_client.py`/`src/mollm_ensemble.py`'s model configuration should move toward qwen2.5:3b/llama3.2:3b/phi4-mini rather than the BioMistral/OpenBioLLM pair.
- The stale `cal_eval.py` Stage 3 ECE (0.762, §3) was measured against the OLD model pair and `mollm_decisions` — it is not just stale on data freshness now, it's measuring the wrong models entirely. A fresh Stage 3 ECE run post-pipeline-re-run will be the first real number for the models actually in use.
- llama3.2:3b's basis-hallucination tendency (confirmed twice on real data, §4.5) is now a live concern for the actual production ensemble, not a footnote about an experiment — worth deciding whether to keep it in the 3-vote ensemble, drop it, or add a self-verification step, once more evidence accumulates.

## 7. Where to pick up tomorrow

1. **Re-run the full pipeline** (`scripts/test_pipeline_e2e.py --note-ids <28 notes>`, ~1.5–2h) so `normalized_entities.candidates` in the DB actually carries `match_basis`. Nothing corpus-scale can be trusted until this happens.
2. **Re-run `scripts/score_gold_recall.py` and the 768-entity voting + audit** with the deterministic bypass and provenance prompt active, to get real corpus-scale numbers (today's numbers in §3 predate all of §4).
3. **GOLD_MISSING (63.3% of Stage 3 failures) is the biggest unaddressed problem** and nothing this session did touches it. Worth its own investigation: compound-span splitting (Stage 2a), rare-term embedding coverage, lay-phrasing→clinical-term gaps, and qualifier/route-noise stripping (the `'subcutaneous heparin'` pattern — structurally similar to Fix 1 but not brand-name-triggered).
4. **Port `check_deterministic_bypass()`, `run_vote_sequential()`, and the provenance-gated prompt architecture to production** (`src/mollm_ensemble.py`/`src/llm_client.py`), now that qwen2.5:3b/llama3.2:3b/phi4-mini are the standing model choice rather than an experiment — currently these only exist in the experimental script.
5. **llama3.2:3b's basis-hallucination tendency** (confirmed twice on real data) may warrant dropping it from the ensemble, or adding a self-verification step to its prompt — now a production concern, see §6.
6. **Nothing in this session has been committed.** `git status` shows `src/normalization.py` and `src/mollm_ensemble.py` modified, `scripts/experiment_3b_voting.py` still untracked (never been git-added in any prior session either).
