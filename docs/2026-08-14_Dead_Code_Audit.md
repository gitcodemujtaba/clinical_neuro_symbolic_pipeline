# Dead Code / Cleanup Audit — Report Only, Nothing Deleted

**Date:** 2026-08-14
**Scope:** Full repo survey (`src/`, `scripts/`, `evaluation/`, `tests/`, root-level artifacts, `requirements.txt`) at the user's request, immediately after committing all Stage 2/3 session work (commit `6bc4b6f`). **By explicit agreement, this audit reports candidates with evidence — it makes NO deletions and NO restructuring.** Every finding below should be independently reviewed before acting; grep-based evidence is a starting point, not proof, and this audit already caught and discarded two of its own false positives (see §4).

---

## 1. CORRECTION (post-audit) — the zero-byte files are NOT dead code

The original draft of this audit flagged 4 zero-byte files as high-confidence dead code on file-age evidence alone (all trace to the initial commit, never touched since). Before acting, `docs/Implementation_Checklist.md` was checked for references and it turned out **all four are open, explicitly tracked `- [ ]` TODO items**, not abandoned scaffolding:
- `evaluation/eval_suite.py` — checklist item #99, with a specific plan ("worth pointing `eval_suite.py` at [`score_gold_recall.py`'s existing logic] rather than rebuilding it").
- `scripts/init_memgraph_snomed.py` — checklist item #60, still-planned SNOMED IS_A → Neo4j ingestion.
- `scripts/init_memgraph_guidelines.py` — actively recommended as the next build step in `docs/Rules_LLM_Triplets_Review.md` §"practical path."
- `tests/test_pipeline_integration.py` — checklist item #118, explicitly flagged as a gap ("0 lines despite the offset-mapping logic being the most bug-sensitive part of the working code so far").

**These were left in place.** This is recorded here as a caution for any future cleanup pass: file age and zero line count are not sufficient evidence of dead code on their own — always cross-check the planning docs before moving/deleting anything that looks like a stub.

## 2. Moved to `dormant/` (2026-08-14, via `git mv` — full history preserved, nothing deleted)

### `scripts/diagnose_glirel.py`
One-time diagnostic that isolated why GLiREL (`jackboyla/glirel-large-v0`) produced near-zero relation-extraction scores. Its conclusion is already fully preserved as a permanent record in `src/extraction.py`'s own module docstring ("GLiREL... was tried first and abandoned... Replaced rather than chasing a downgrade that risked breaking GLiNER-BioMed/SapBERT"). The finding survives without the script; GLiREL isn't even a current dependency. No checklist references found.

### `scripts/check_pluralization_gap.py` and `scripts/test_hyphen_preprocessing_hypothesis.py`
Both self-described as one-time investigations ("quick, targeted check... NOT a blind fix — this only looks"; "EMPIRICAL TEST, not a pipeline change"), with no references anywhere else in the codebase or docs, and no checklist entries. Unlike the `measure_*`/`diagnose_*` scripts in §4 (which ARE actively cited as provenance for tuned constants), these two aren't load-bearing documentation.

Now at `dormant/scripts/` — see `dormant/README.md`.

## 3. Unused dependencies (`requirements.txt`)

Verified by checking actual import usage, not just package-name string matching (a naive check falsely flagged `scikit-learn` and `scispacy` — see §4 for why those are real dependencies):

- **`streamlit`** — zero references anywhere in the codebase.
- **`pyvis`** — zero references anywhere in the codebase.

Both sound like remnants of a planned interactive dashboard/graph-visualization feature that was never built (or was removed) — worth checking with whoever added them before dropping, in case there's a work-in-progress branch or notebook elsewhere that uses them.

- **`pandas`** — zero `import pandas` anywhere in the currently committed `.py` files. Common enough (and cheap enough) that it may be intentionally kept for ad-hoc analysis outside the repo; lower confidence than streamlit/pyvis that it's truly droppable.

## 4. Checked and ruled OUT — false positives worth recording so they aren't re-flagged later

- **`src/extraction.py`** — looks like a stray duplicate of `src/entity_extraction.py` by name alone. It is not: `extraction.py` is Stage 2a *relation* extraction (GLiNER-relex), a distinct and actively-used module (imported by `src/clinical_pipeline.py`, `evaluation/stage_calibration.py`, `scripts/test_extraction.py`). Confusable naming, not dead code.
- **`scikit-learn`** — naive grep for the literal string missed it because the import name (`sklearn`) differs from the package name. Genuinely used in `src/mollm_calibrator.py`, `scripts/fit_mollm_calibrator.py`, `evaluation/metrics.py`, `tests/test_calibrator_fit.py`.
- **`scispacy`** — never directly `import`ed, but required as a transitive dependency: `src/preprocessing.py` loads the `en_core_sci_sm` model (`spacy.load("en_core_sci_sm")` / `import en_core_sci_sm`), which is a scispacy-distributed model package that needs scispacy installed to function.
- **`scripts/inspect_review_decisions.py`** — zero references elsewhere, but this is by design: it's a general-purpose interactive CLI inspection tool (`--note-id`, `--routing` flags), meant to be run ad hoc, not cited as provenance like the `measure_*` scripts. Not dead.
- **The `measure_*`/`diagnose_guard_suppression.py`/`diagnose_citation_quotes.py` family** (`measure_channel_b_coverage.py`, `measure_relation_coverage.py`, `measure_heuristic_and_boundary.py`, `measure_gliner_risk_vs_match_tier.py`, `diagnose_guard_suppression.py`, `diagnose_citation_quotes.py`) — these ARE actively referenced, but as **provenance comments in `src/` docstrings** ("the measured hit rate that shaped its bound" style citations in `src/normalization.py`, `src/retrieval.py`, `src/assertion.py`), not as imports. This is a deliberate, valuable pattern — a tuned constant's docstring points back to the exact script that measured the number — and removing these scripts would silently break that audit trail even though nothing `import`s them. **Do not remove these.**

## 5. Repo hygiene — DONE (2026-08-14, via `git mv`)

**18 JSON report/artifact files** (17 counted in the original pass, plus `score_gold_recall_AFTER.json`'s sibling; ~3.1MB total) moved from repo root into `reports/`: `3b_voting_results.json`, `benchmark_68note_report.json`, `cal_report.json`, `channel_b_coverage.json`, `cov_cleaned.json`, `gliner_risk_report.json`, `heuristic_report.json`, `score_gold_recall_AFTER.json`, `score_gold_recall_AFTER_stage2_fixes.json`, `score_gold_recall_AFTER_stage2_fixes_v2.json`, `score_gold_recall_BEFORE.json`, `score_report.json`, `stage2a_cal.json`, `stage2a_ece.json`, `stage2b_cal.json`, `stage2b_cal_fresh.json`, `stage2b_ece.json`, `stage3_ece.json`.

Verified before moving: every reference to these filenames elsewhere in the codebase was a `--out filename.json` usage example in a docstring, not a hardcoded `default=` argparse value — moving the files doesn't break anything. Updated the 5 docstrings that showed a bare filename (`evaluation/cal_eval.py`, `evaluation/stage2a_cal_eval.py`, `evaluation/stage2b_cal_eval.py`, `scripts/measure_gliner_risk_vs_match_tier.py`, `scripts/measure_heuristic_and_boundary.py`) to show `--out reports/filename.json` instead, so future runs following the example land in the same place. `reports/` is not gitignored — these 18 files are already committed history (`6bc4b6f`), so keeping them tracked preserves that record; future runs can still write elsewhere if a given output isn't meant to be kept.

Note for later: `scripts/experiment_3b_voting.py` still writes its own output to a hardcoded absolute scratchpad path outside the repo (`/tmp/claude-1000/.../scratchpad/3b_voting_results.json`) rather than accepting an `--out` flag like the other scripts — inconsistent with the rest of the tooling, not touched here since it's a behavior change, not a file move.

## 6. Modularization (component/module-wise split, per the original ask)

### `src/normalization.py` — DONE (2026-08-14)
Split into a package, `src/normalization/`, with the single-file module's public surface fully preserved — every existing `from src.normalization import X` and `import src.normalization as N; N.X` call site across the codebase (`src/clinical_pipeline.py`, `evaluation/stage_calibration.py`, `tests/test_tier12_ranking.py`, `tests/test_confidence_tier_reasons.py`, etc.) continues to work completely unchanged, verified by re-running every test file plus a live functional check against the real DB (Lasix→furosemide `verified_brand_alias` tagging, Aldactone's 6-way combo ambiguity, both identical to pre-split behavior).

| File | Lines | Contents |
|---|---|---|
| `__init__.py` | 140 | Re-exports all 65 original top-level names (including underscore-prefixed "private" helpers, since tests and diagnostic scripts reach into them directly) |
| `constants.py` | 298 | Module-wide config constants (`CANDIDATE_LIMIT`, `TIER3_SIMILARITY_FLOOR`, `VOCAB_BY_LABEL`, etc.) |
| `sapbert_model.py` | 49 | Model loading, `get_sapbert_embedding`, `_cosine` |
| `vocab_release.py` | 58 | Athena vocabulary release lookup |
| `text_utils.py` | 65 | Tokenization/SQL-clause helpers shared across tiers |
| `compound_span.py` | 642 | Compound-span splitting, span-growth detection, lab-value-suffix stripping |
| `tier_retrieval.py` | 565 | Tier 1-4 candidate retrieval/ranking, alias expansion, hierarchy collapse (tonight's Fix 1/Fix 2) |
| `orchestrator.py` | 812 | `normalize_entity()`, `process_and_normalize_entities()`, `compute_confidence_tier()` |

Method: extracted every top-level def/class/const by exact line range (`ast`-derived, byte-identical to the original — no code was retyped) rather than manually reconstructing content, specifically to eliminate transcription-error risk at this size. Caught two real issues before they shipped:
1. **A bare top-level `print(...)` statement** (announcing SapBERT model loading) isn't a `def`/`class`/assignment, so the AST-based extraction script didn't know to track it and it landed in the wrong file (`constants.py` instead of `sapbert_model.py`) by accident of line-range adjacency. Fixed by explicitly auditing for every top-level statement type the extraction script *doesn't* handle before trusting the split — worth doing again for `mollm_ensemble.py`/`retrieval.py` if those get split too.
2. **`from .constants import *` silently drops underscore-prefixed names** (Python's default `import *` behavior) — broke `_ALNUM_MIX_RE` at runtime with a `NameError`, only caught by actually running the test suite, not by import-checking alone. Fixed with an explicit `__all__` in `constants.py`.
3. **Cross-module monkey-patching breaks silently on a naive split.** `tests/test_tier12_ranking.py` does `src.normalization.get_sapbert_embedding = fake_embedding` to stub the model out. In the single-file module this worked because everything shared one namespace; after the split, `tier_retrieval.py` and `orchestrator.py` had each done `from .sapbert_model import get_sapbert_embedding`, which binds their OWN independent reference at import time — a later patch on the package's re-export never reaches those bindings, so the real (expensive, un-stubbed) model gets called instead. Fixed by having both files define a local `get_sapbert_embedding` that does a *lazy* `import src.normalization as _pkg; return _pkg.get_sapbert_embedding(text)` inside the function body — a live lookup through the package's public namespace at call time, so any patch applied there is honored. **This is the one finding most worth remembering before splitting `mollm_ensemble.py` or `retrieval.py`** — check every test file for `module.function = ...`-style monkey-patching first, and expect to need the same lazy-lookup pattern for whatever they patch.

Old file removed via `git rm` (full history preserved) plus a backup copy kept in the session scratchpad during the work.

### `src/mollm_ensemble.py` (1548 lines) and `src/retrieval.py` (1349 lines) — NOT yet split
Natural seams identified but not executed:
- `src/mollm_ensemble.py`: prompt assembly (`build_prompt`, `_format_candidates`, `_format_evidence`, `_format_suppressed`) vs. routing/safety-gate logic (confidence thresholds, override gate) vs. DB load/persist (`load_records_for_stage3`, `store_decision`).
- `src/retrieval.py`: `GuidelineIndex`, three hierarchy backends (`DuckDBHierarchy`/`Neo4jHierarchy`/`NullHierarchy`), `VocabularyRetriever`, `GroundingRetriever` — four already-separate classes, straightforward to become four files.

Same method (exact-range extraction, full regression + live functional testing, explicit check for monkey-patching in the test suite) should be used if/when these are split.

## 7. What this audit did NOT cover

Given the scope was explicitly "report only," this pass did not: run a full unused-function/dead-import linter (e.g. `vulture`/`pyflakes`) across every file, check the `docs/*.md` files themselves for staleness, or verify the `data/` directory's contents against what's actually loaded. Worth a follow-up pass if a deeper audit is wanted.
