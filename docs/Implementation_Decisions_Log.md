# Implementation Decisions Log — Every "Why We Chose X" With Its Evidence

Comprehensive decision log spanning the whole project (2026-08-07 through
2026-08-20), mined from all 32 dated/named docs in `docs/` plus this
session's own verified work. Organized by stage/topic, not chronology, so
a specific "why did we do X" question can be found directly. Every entry
states **Decision → Evidence → Source → Date**; evidence figures are
copied exactly as recorded in their source document, not rounded or
paraphrased. This is the "why," paired with proof — for the "what
changed" narrative, see `docs/2026-08-20_Session_Results_And_Status.md`;
for final numbers, see `docs/FINAL_RESULTS_Single_Source_Of_Truth.md`.

**One real contradiction found while compiling this, flagged rather than
silently resolved**: two docs from the same date (2026-08-07) state
opposite conclusions about which corpus was chosen as the primary
guideline-KG source (`local_triplets_db2_v6`/MedCAT vs. `rules-llm`) — see
§7, both entries kept, not reconciled here.

---

## 1. Data, Splits & Evaluation Methodology

### Locked train/val/test split materialized as a file
- **Decision**: Created a fixed, seeded `data/splits/note_splits.csv` (272 notes: 70 test / 60 val / 142 train) so evaluation scripts default to `--split val`, with `--split test` requiring an explicit unlock flag.
- **Evidence**: Matches `docs/Evaluation_Criteria.md`'s ~70-locked design; SHA256 recorded in the file header for integrity tracking.
- **Source**: docs/2026-08-13_Implementation_Verification.md (§3)
- **Date**: 2026-08-13

### Stratified 5-fold CV replaces single 80/20 split for calibrator evaluation
- **Decision**: Switched calibrator evaluation from one unstratified 80/20 split to stratified 5-fold CV, covering all 140 rows.
- **Evidence**: The single 80/20 split (n_test=28, only 20 positives across 140 rows) could hand the test fold 1–2 positives, producing a base-rate accuracy artifact.
- **Source**: docs/2026-08-13_Implementation_Verification.md (§2.1)
- **Date**: 2026-08-13

### Accuracy demoted, AUROC/AP promoted as calibrator headline metric
- **Decision**: `scripts/fit_mollm_calibrator.py` now reports AUROC and average precision as primary discrimination metrics, accuracy demoted, null-model baseline printed alongside.
- **Evidence**: Prior held-out accuracy of 85.71% exactly matched the base rate on a 14.3% positive rate (n_test=28), mean predicted P(correct)=0.1274 — a majority-class-prediction artifact, not genuine discrimination.
- **Source**: docs/2026-08-13_Implementation_Verification.md (§2.1)
- **Date**: 2026-08-13

### Clinical-T5 removed from in-pipeline role, kept only as external baseline
- **Decision**: Removed Clinical-T5 from live Stage 2a (ReLEx) and used it exclusively as an external accuracy/hallucination baseline.
- **Evidence**: Clinical-T5 was pretrained on MIMIC-III/MIMIC-IV, creating data-contamination risk (it may have seen the actual evaluation notes during pretraining).
- **Source**: docs/Proposal_Alignment_Review.md (§3.2)
- **Date**: 2026-08-07

### Fresh25 batch used to validate against calibrator train/val leakage
- **Decision**: Ran a 25-genuinely-fresh-note validation batch (outside every calibrator train/val note) as the standard for validating tier precision, rather than trusting development-set numbers.
- **Evidence**: Combined AUTO_TIERS precision on fresh25 = 967/1,162 = 83.2%; Tier 1B (calibrator) 100/108 = 92.6%; Tier 1 (unanimous) 544/690 = 78.8% — lower than earlier development-set checkpoints, partly explained by a newly-quantified Lab-Test SNOMED near-duplicate ceiling (64/94 = 68% wrong).
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§6)
- **Date**: 2026-08-20

### Fresh-10-note held-out validation (this session's own, distinct from fresh25)
- **Decision**: Ran a second, independent held-out check — 10 notes from the official locked test split, mostly outside the calibrator's training set — as the basis for the paper's headline precision figure.
- **Evidence**: 76.8% (43/56) AUTO-tier precision, vs. 86.9% corpus-wide — the corpus-wide figure is inflated by notes used during development (root-caused to `TIER_3`'s curated dictionaries not generalizing, accounting for 17.2 of a 25.8pp deflection-rate gap).
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§13, §14, §15)
- **Date**: 2026-08-20

### Dead code kept, not deleted, when tracked by an open checklist item
- **Decision**: 4 zero-byte files flagged as high-confidence dead code (by file-age evidence alone) were kept, since all four are open, explicitly tracked `- [ ]` TODO items in `docs/Implementation_Checklist.md`/`docs/Rules_LLM_Triplets_Review.md`.
- **Evidence**: `evaluation/eval_suite.py`, `scripts/init_memgraph_snomed.py`, `scripts/init_memgraph_guidelines.py`, `tests/test_pipeline_integration.py` — each cross-checked against a live tracked item before deciding.
- **Source**: docs/2026-08-14_Dead_Code_Audit.md (§1)
- **Date**: 2026-08-14

### One-off diagnostic scripts moved to `dormant/`, not deleted
- **Decision**: `scripts/diagnose_glirel.py`, `scripts/check_pluralization_gap.py`, `scripts/test_hyphen_preprocessing_hypothesis.py` moved via `git mv` rather than removed.
- **Evidence**: No references elsewhere in code or docs, no checklist entries; GLiREL's own finding is already preserved in `src/extraction.py`'s docstring and GLiREL is not even a current dependency.
- **Source**: docs/2026-08-14_Dead_Code_Audit.md (§2)
- **Date**: 2026-08-14

### `measure_*`/`diagnose_*` provenance scripts explicitly kept despite zero imports
- **Decision**: Kept `measure_channel_b_coverage.py`, `measure_relation_coverage.py`, `measure_heuristic_and_boundary.py`, `measure_gliner_risk_vs_match_tier.py`, `diagnose_guard_suppression.py`, `diagnose_citation_quotes.py`.
- **Evidence**: Actively referenced as provenance comments in `src/normalization.py`/`src/retrieval.py`/`src/assertion.py` docstrings — removing them would silently break the audit trail behind a tuned constant.
- **Source**: docs/2026-08-14_Dead_Code_Audit.md (§4)
- **Date**: 2026-08-14

### `src/normalization.py` split into a package via exact AST line-range extraction
- **Decision**: Split into `src/normalization/` (8 files) using AST-derived exact line-range extraction rather than manual retyping.
- **Evidence**: Verified against every dependent test plus a live functional check (Lasix→furosemide alias tagging, Aldactone's 6-way combo ambiguity, identical to pre-split behavior). Caught and fixed 3 real issues in the process: a stray top-level `print()` landing in the wrong file, `from .constants import *` silently dropping underscore-prefixed names, cross-module monkey-patching breaking silently.
- **Source**: docs/2026-08-14_Dead_Code_Audit.md (§6)
- **Date**: 2026-08-14

### `mollm_ensemble.py` and `retrieval.py` left unsplit
- **Decision**: Natural split seams identified in both large files (1548 and 1349 lines) but not executed.
- **Evidence**: Deferred as future work using the same exact-range-extraction method, with an explicit note to check every test file for monkey-patching first (the pattern found during the normalization split).
- **Source**: docs/2026-08-14_Dead_Code_Audit.md (§6)
- **Date**: 2026-08-14

---

## 2. Stage 1 — Preprocessing (Abbreviation Expansion, Sectioning, Assertion)

### Abbreviation dictionary loaded as multi-expansion map with an ambiguity flag
- **Decision**: Replaced the single-value `{abbr: meaning}` dict comprehension with `{abbr: [all_meanings]}`; set `ambiguous: true` + `candidate_expansions[]` when an abbreviation has >1 known meaning, routing it toward `confidence_tier_in = LOW`.
- **Evidence**: The old comprehension silently dropped all but the last-loaded expansion per key (`MS`, `PT`, `DC` each have multiple real clinical meanings), poisoning both extraction and normalization with no detectability.
- **Source**: docs/Stage1_2_Completeness_Audit.md (Severity 3)
- **Date**: 2026-08-08

### Note section segmentation added to Stage 1/2a
- **Decision**: Persist `(section_name, char_start, char_end)` spans and stamp each Stage 2a entity with its containing section.
- **Evidence**: MIMIC-IV notes are strongly, consistently sectioned (e.g. "Past Medical History" in 274/272 notes); section membership is a near-free, deterministic prior — often stronger than local negation cues (anything under "Family History" needs no negation cue).
- **Source**: docs/Stage1_2_Completeness_Audit.md (Severity 2)
- **Date**: 2026-08-08

### Rule-based clinical assertion detection (medspacy/ConText) added
- **Decision**: Added a deterministic assertion-detection pass producing `assertion_status`, `experiencer`, `temporality`, and matched cue text/offset, rather than relying on the LLM to infer negation.
- **Evidence**: 20.9% (15,794/75,491) of gold-annotated spans sit in a non-assertive context — 14.2% negated, 4.3% historical, 2.0% hypothetical, 1.1% family.
- **Source**: docs/Stage1_2_Completeness_Audit.md (Severity 1)
- **Date**: 2026-08-08

### Third-tier abbreviation tiebreak declines rather than guesses on genuine ambiguity
- **Decision**: `_select_by_groundability()` only overrides the alphabetical-default tiebreak when exactly one candidate meaning grounds to a real OMOP concept; falls through to alphabetical default (flagged) when zero or both ground.
- **Evidence**: `"fx"` → {"fracture","fractions"}, both real grounded OMOP concepts (45876626/4217808 and 4081834) — correctly declines rather than guessing. 6 stub + 2 integration tests passed; macro IoU 0.0816 vs 0.079 baseline.
- **Source**: docs/2026-08-13_Calibration_Diagnostics_And_Fixes.md (§3.1)
- **Date**: 2026-08-13

### Original-form (unexpanded) fallback added when expansion fails to map
- **Decision**: Retry normalization on the raw surface form when the expanded form fails, recording `normalized_from` for auditability.
- **Evidence**: `NAD` had expanded to "nicotinamide adenine dinucleotide" instead of "no acute distress"; `TTP` to "Thrombotic Thrombocytopenic Purpura" instead of "tenderness to palpation" — the deterministic alphabetical tiebreak had no clinical basis for either pick.
- **Source**: docs/Stage1_2_Completeness_Audit.md (item 17)
- **Date**: 2026-08-09

### Abbreviation flywheel: frequency-priority mechanism inverted from block-list to allow-list
- **Decision**: `compute_frequency_priority()` now requires explicit membership in a new, initially-empty `VERIFIED_ALLOW_LIST` before ever returning a ledger-derived answer — inverted from an earlier design that excluded only a known-bad list.
- **Evidence**: Gold-checking the 7 highest-confidence non-excluded ledger winners came back 7/7 wrong (e.g. `DM`→"deep masseter" vs. gold "Diabetes mellitus"; `air`→"autoimmune retinopathy" 10/10 vs. gold "Breathing room air"); `selection_basis` showed the mechanism re-selecting its own earlier wrong guesses mid-batch. Post-inversion, `DM`/`CAD`/`SBP` all correctly return `None`.
- **Source**: docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md (§7)
- **Date**: 2026-08-17

### Abbreviation flywheel's context-rule mining kept separate from the frequency-priority exclusion list
- **Decision**: `mine_context_rules()` (mines real HITL-reviewer-confirmed data into deterministic pre/post trigger-word rules) deliberately does NOT apply the same bias-exclusion list as the frequency-priority mechanism.
- **Evidence**: Reasoning: independent human-confirmed evidence is exactly what can correct a systematic model bias, and excluding it the same way would permanently block the mechanism from ever fixing the entities it exists to fix.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (context from Phase 7 build, referenced via §14 discipline pattern)
- **Date**: 2026-08-17

### Abbreviation flywheel scoped to train-split notes only
- **Decision**: First flywheel production run scoped to 50 notes from the `train` split, explicitly avoiding the `test` split.
- **Evidence**: 31 of 57 already-processed `is_test=TRUE` notes were from the locked test split — populating a ledger that live-influences pipeline decisions from benchmark-set observations was judged the same leakage risk `ConsensusCalibrator.assert_not_trained_on()` already guards against.
- **Source**: docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md (§7)
- **Date**: 2026-08-17

---

## 3. Stage 2a — Extraction (GLiNER, Relations, Chunking)

### GLiNER-BioMed swapped in for general-domain GLiNER
- **Decision**: Replaced `urchade/gliner_medium-v2.1` with `Ihor/gliner-biomed-large-v1.0`.
- **Evidence**: A domain-fit decision — GLiNER-BioMed is trained on biomedical NER benchmarks including clinical narratives, closing an earlier checklist requirement for a clinical/biomedical checkpoint.
- **Source**: docs/Implementation_Checklist.md (Stage 2a)
- **Date**: 2026-08-07

### `flat_ner` unified to flat (not nested) across both Stage 2a models
- **Decision**: Standardized both extraction models on a single shared `FLAT_NER = True`, resolving a previously silent library-default inconsistency.
- **Evidence**: On note `10000032-DS-21`, unifying to flat held relation count at 3 while endpoint linking went from 33% to 100% (flat spans from both models agree on boundaries, clearing the overlap threshold). Nested spans would also have produced duplicate `:PatientObservation` nodes for one clinical fact in Stage 4's design.
- **Source**: docs/Stage1_2_Completeness_Audit.md (Severity 6)
- **Date**: 2026-08-09

### Sub-threshold GLiNER entities retained (flagged), not discarded
- **Decision**: Lowered the extraction run threshold to 0.35, retaining everything below the 0.50 promotion gate, flagged `below_threshold`, excluded from every downstream stage by one filter.
- **Evidence**: On note `10000032-DS-21`, 33 sub-threshold spans were retained — 22% on top of 117 accepted entities — creating the population needed to measure whether Stage 3 can recover near-misses and whether 0.50 is set correctly.
- **Source**: docs/Stage1_2_Completeness_Audit.md (item 19)
- **Date**: 2026-08-09

### GLiNER-relex chosen over GLiREL for relation extraction
- **Decision**: Replaced GLiREL (`jackboyla/glirel-large-v0`) with GLiNER-relex (`knowledgator/gliner-relex-large-v1.0`), rather than downgrading `transformers`/`torch` to fix GLiREL.
- **Evidence**: After fixing a real label-format bug (which had silently produced 0 relations across 10 test notes), GLiREL still scored near-zero — root-caused to `transformers==4.57.6`/`torch==2.8.0` being far newer than what the unmaintained GLiREL checkpoint was validated against (reproducing GLiREL's own README example: documented score 0.9923, actual 0.0028). Downgrading was judged to risk breaking GLiNER-BioMed/SapBERT elsewhere.
- **Source**: docs/Implementation_Checklist.md (Stage 2a)
- **Date**: 2026-08-07

### MedGemma 4B deliberately not reused for relation extraction
- **Decision**: MedGemma 4B (committed to Stage 3's ensemble) was not reused for Stage 2a relation extraction.
- **Evidence**: Doing so would mean Stage 3 partially validates its own Stage 2a output, undermining ensemble independence.
- **Source**: docs/Implementation_Checklist.md (Stage 2a)
- **Date**: 2026-08-07

### Relation endpoint linking via character-offset overlap, not text/label matching
- **Decision**: Persist head/tail character offsets on extracted relations, resolve entity IDs by maximum overlap (≥50% required, else `unresolved`) — supersedes an originally-proposed text+label matching approach.
- **Evidence**: Both GLiNER-BioMed and GLiNER-relex run on the same `expanded_text`, sharing a coordinate system — offset-overlap is deterministic and roughly ten lines of code vs. a fuzzy cross-model text-matching heuristic.
- **Source**: docs/Stage1_2_Completeness_Audit.md (Severity 4)
- **Date**: 2026-08-08

### Sliding-window chunking built for GLiNER-truncated long notes
- **Decision**: Built `_build_chunks()`/`_extract_entities_chunked()` (1800-word budget, 128-word overlap, sentence-boundary-snapped) rather than leaving long notes silently truncated.
- **Evidence**: `model.config.max_len=2048` confirmed as GLiNER-BioMed's real ceiling. Note `11532659-DS-11` (24,858 chars): single-call → 124 entities, truncated; chunked → 399 entities, 282 additionally recovered. Cost: 85.6s chunked vs. 35.8s single-call, accepted within the 2-5 min/note budget.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§4)
- **Date**: 2026-08-17

### GLiNER truncation surfaced but chunk-and-merge re-extraction NOT built (earlier deferral, later reversed above)
- **Decision**: An earlier pass added `possibly_truncated`/`gliner_input_token_count` columns to surface truncation, but deliberately did not build chunk-and-merge recovery at that time.
- **Evidence**: Framed as "a real, separate feature with real span-boundary risk" — deferred pending a corpus-wide query of `possibly_truncated` to measure real impact first. (Superseded by the chunking decision above once that measurement was done.)
- **Source**: docs/2026-08-15_Stage1_2a_2b_Remediation.md (§3)
- **Date**: 2026-08-15

### Lab abbreviation cold-start injection built to close the dominant extraction-recall gap
- **Decision**: Built `src/lab_abbrev_coldstart.py` (21 case-sensitive bare lab-abbreviation terms, gold-mined text→concept injection), reusing the existing `_LAB_TEST_ALIASES`/`tier3_fast_path` mechanism rather than a new bypass.
- **Evidence**: Corpus-wide miss analysis (140 notes, 38,689 gold annotations): Stage 2a extraction recall only 46.1%; 100% of misses were true zero-shot GLiNER blind spots; 97.4% of misses ≤25 chars, dominated by 15 bare abbreviations (Creat, Hgb, RBC, Na, Hct, Cl, MCH, MCHC, RDW, HCO3, WBC, UreaN, Phos, Calcium, AnGap) accounting for 2,097 misses. Simulated impact: recall 46.4%→51.9% (+5.5pp); verified 101/101 on the worst-affected note.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§8)
- **Date**: 2026-08-20

### Narrative state-word cold-start added, with explicit polysemy screening
- **Decision**: Added a small cold-start dictionary for single-word state descriptors (`alert`, `improved`, `baseline`, `warm`, `clinic`), explicitly screening OUT tempting-but-polysemous candidates.
- **Evidence**: Screened-in terms were all ≥95% gold-consistent to one concept; screened-out terms (`pain` 76.7% across 4 concepts, `stable` 88.4%, `negative` 64.5% split, `procedure`/`support`/`tender`/`masses`/`wound`) fell below that bar. Combined with the lab fix: recall 46.4%→53.2% (+6.8pp total).
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§8)
- **Date**: 2026-08-20

---

## 4. Stage 2b — Normalization / Retrieval / Grounding

### Deterministic ORDER BY tiebreak added to every normalization tier query
- **Decision**: Added explicit `ORDER BY concept_id ASC` (Tier 1/2) / `ORDER BY similarity DESC, concept_id ASC` (Tier 3) to every tier query in `normalize_entity()`.
- **Evidence**: Two consecutive runs of the same note produced different outputs for the same entity (`ED` → "Ed District" vs. "Erectile dysfunction") because DuckDB's row order under parallel execution wasn't stable — undercutting the "deterministic, white-box" claim.
- **Source**: docs/Proposal_Alignment_Review.md (§7)
- **Date**: found 2026-08-07, fixed 2026-08-08

### Hierarchy-duplicate collapse via union-find over `Is a`/`Subsumes` edges
- **Decision**: `_collapse_hierarchy_duplicates()` uses union-find over direct SNOMED hierarchy edges, collapsing each connected component to its single highest-similarity member.
- **Evidence**: Confirmed the ALT-measurement trio (4095055, 4146380, 44810789) is a real SNOMED parent/child hierarchy, not literal duplicate IDs — a plain `GROUP BY concept_id` would not have caught it.
- **Source**: docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md (§2)
- **Date**: 2026-08-13/14

### Curated `match_basis` candidates given unconditional priority in hierarchy collapse
- **Decision**: Modified `_collapse_hierarchy_duplicates()` so a candidate with a curated `match_basis` (`verified_lab_test_alias`/`verified_brand_alias`/`lab_procedure_preferred`) wins its hierarchy root unconditionally, regardless of raw similarity.
- **Evidence**: Confirmed live on "HCO3": the curated concept (4194291, "Blood bicarbonate measurement") was losing to its own SNOMED parent (4227915, 0.8857 similarity) purely on cosine score. All 2,899 already-written cold-start entities re-normalized under the fix.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§10)
- **Date**: 2026-08-20

### Force `ambiguous=True` when an alias candidate exists but isn't ranked first
- **Decision**: `normalize_entity()` forces `ambiguous=True` (reason `alias_candidate_outranked`) whenever an alias candidate exists but isn't `cands[0]`.
- **Evidence**: Aldactone → propiolactone (0.83, confidently wrong) cleared both floor and margin checks, silently discarding the correct alias-injected spironolactone (0.56) one function later. Caught by direct DB inspection.
- **Source**: docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md (§2)
- **Date**: 2026-08-13/14

### Deterministic ensemble-bypass requires exactly one alias hit, excludes exact-text matches
- **Decision**: `check_deterministic_bypass()` skips the LLM ensemble only when exactly one candidate has `match_basis=="verified_brand_alias"`; deliberately excludes `exact_text` and requires exactly one, not "at least one."
- **Evidence**: Tier 1 (Exact) accuracy measured 52.48% (766 entities, 402 correct) — essentially a coin flip, not safe to auto-skip. Combination brands (e.g. Aldactone) can legitimately produce several alias candidates; picking the first by list order would be arbitrary.
- **Source**: docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md (§4)
- **Date**: 2026-08-13/14

### Brand-to-generic alias resolution: 3-hop KG traversal, ingredient-level only
- **Decision**: `_alias_expand_brand_to_generic()` returns only final Ingredient-level concept_ids (brand → Branded Drug Comp → Tradename of Clinical Drug Comp → RxNorm has ing → Ingredient), not intermediate dose-specific SKUs.
- **Evidence**: Returning the full intermediate CTE exploded to 143 ids for Lasix before narrowing to the single correct id (956874, furosemide).
- **Source**: docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md (§2)
- **Date**: 2026-08-13/14

### Candidate rendering isolates the match-basis tag to fix LLM attention dilution
- **Decision**: `_format_candidates()` renders each candidate as a 3-line block with Basis visually isolated (`>>> VERIFIED_BRAND_ALIAS <<<`), replacing a dense single-line format.
- **Evidence**: Measured "attention dilution" where the dense format let 3B models detach the Basis tag from its own index and misattribute it to the highest-scored candidate.
- **Source**: docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md (§4)
- **Date**: 2026-08-14

### Procedure-class ranked over Observable-Entity-class for lab tests (78/78 exceptionless)
- **Decision**: `_prefer_lab_procedure_over_observable()` applies a rank-only penalty (never touching displayed `similarity_score`) so Procedure-class concepts outrank Observable-Entity-class siblings for lab tests.
- **Evidence**: SapBERT consistently scores Observable-Entity ("Leucocyte count") higher than Procedure ("White blood cell count") for WBC — 0.892 vs. 0.8694. Measured across every Lab-Test entity in the 27-note corpus with both classes present: Procedure-class is gold-correct 78/78 times, zero exceptions.
- **Source**: docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md (§2, Fix 4)
- **Date**: 2026-08-14

### Lab-procedure preference extended to also cover Qualifier Value class (2026-08-20)
- **Decision**: Extended the above rule to also penalize `'Qualifier Value'`-class near-duplicates, not just Observable Entity.
- **Evidence**: RDW's Qualifier-Value duplicate (concept 42536363, "calculation technique") outscored the correct concept by 0.077 — exceeding the existing 0.1 bonus's margin. 22 such concepts found across common lab tests, zero ever gold-correct.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§12, §14)
- **Date**: 2026-08-20

### TIER_2 fix: deterministic fast-path instead of prompt persuasion
- **Decision**: Rather than continue tuning `_binary_match_prompt()`'s wording, built `_lab_procedure_fast_path()` bypassing the ensemble entirely (0 model calls) when a Lab-Test entity's top candidate is tagged `lab_procedure_preferred`.
- **Evidence**: Corpus-scale grading (68 notes, 8,815 decisions) found `TIER_2_AUTO_RESOLVED` at 20.0% precision, all sharing the RDW/MCV/MCHC pattern. Two prompt-wording fix attempts (soft, then "sledgehammer") produced no change and one regression. Root-caused to an architectural mismatch: `_binary_match_prompt()` shows one candidate at a time, so a model can't compare against the alternative it's told to resist.
- **Source**: docs/2026-08-19_Lab_Procedure_Vs_Observable_Entity_Finding.md
- **Date**: 2026-08-19

### SNOMED regional-extension concepts filtered by SCTID namespace pattern (2026-08-20)
- **Decision**: Filter `concept_code NOT LIKE '%1000000___'` into every open-ended Tier 1–4 query — rejecting a collaborator-proposed `vocabulary_id`-based filter.
- **Evidence**: `vocabulary_id` has only one distinct value (`'SNOMED'`) in the 1,093,147-row table — no vocabulary-level filter possible. `concept_class_id` unreliable: 23,842 of 98,487 extension-pattern concepts are themselves `'Procedure'` class. Zero overlap with the corpus's 4,522 gold-standard codes.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§12, §14)
- **Date**: 2026-08-20

### Compound-span splitter given a whole-phrase Tier 1/2 guard
- **Decision**: `find_compound_split()` returns `None` immediately if the full entity phrase already resolves confidently at Tier 1/2, before attempting to split.
- **Evidence**: The unguarded version wrongly decomposed already-correct merged entities ("aspiration pneumonia," "retroperitoneal hematoma"), dropping linked_recall from 15/144 to 13/144 on a test note. Guard restored recall past baseline to 18/144.
- **Source**: docs/Stage2_Compound_And_Qualifier_Gaps.md (Background)
- **Date**: pre-2026-08-10

### Compound splitter generalized to exhaustive 2–4 token-group partition search
- **Decision**: Generalized from a single binary cut point to an exhaustive partition search over 2–4 contiguous word groups.
- **Evidence**: `right EVD placement`/`Right EVD removal` etc. are gold-annotated as 3 separate concepts each, but the binary splitter only tried one cut point.
- **Source**: docs/Stage2_Compound_And_Qualifier_Gaps.md (Gap 1)
- **Date**: 2026-08-10

### Span-growing detector added (mirror-image of compound splitting)
- **Decision**: Added `find_span_growth()` to absorb 1–5 adjacent words from context and prefer the grown phrase when it resolves at Tier 1/2.
- **Evidence**: Gold repeatedly annotated a longer qualified phrase as a distinct, clinically different concept from the shorter predicted span (e.g. "congestive heart failure" 42343007 vs. predicted "heart failure" 84114007) — 8 such rows documented.
- **Source**: docs/Stage2_Compound_And_Qualifier_Gaps.md (Gap 2)
- **Date**: 2026-08-10

### Domain override threaded from split/growth detection into re-normalization
- **Decision**: Added a `domain_override` parameter to `normalize_entity()`, populated from the domain the split/growth detector's own unrestricted lookup confirmed — rejecting the simpler alternative of restricting split-detection to the label's default domain.
- **Evidence**: "Left craniectomy" split into "Left" (Unmapped) + "craniectomy" (correct) because Stage 2b's domain restriction excluded the domain "Left" actually lives in. The simpler fix was checked and rejected because it would also make "Left" fail during detection, losing the "craniectomy" win too.
- **Source**: docs/Stage2_Compound_And_Qualifier_Gaps.md (Gap 3)
- **Date**: 2026-08-10

### "At least one deferred partition part" relaxation implemented, then explicitly reverted
- **Decision**: A relaxation allowing one unresolved partition part per compound split was implemented, then reverted back to requiring every part resolve at Tier 1/2.
- **Evidence**: The relaxation dropped combined linked_recall 10.78%→9.72% (a spurious exact Tier-1 match of "heart" against an unrelated concept triggered bad splits). Reverting restored recall to 11.14%, above the original baseline.
- **Source**: docs/Stage2_Compound_And_Qualifier_Gaps.md
- **Date**: 2026-08-10

### Lab-value suffix stripping added as a last-resort, Lab-Test-only retry tier
- **Decision**: Added `strip_lab_value_suffix()`, gated to `gliner_label=="Lab Test"` only, stripping trailing `-NUMBER` suffixes.
- **Evidence**: `WBC-13.0`, `PTT-29.0`, `Glucose-117`, `ALT-736`, `AST-956` all failed every tier (MIMIC flowsheet notation glues results to test names). Post-fix: combined linked_recall rose 11.14%→11.26%.
- **Source**: docs/Stage2_Compound_And_Qualifier_Gaps.md (Gap 4)
- **Date**: 2026-08-10

### Lab Value Suffix Fallback given a cheap Tier 1/2 pre-check
- **Decision**: Added a pure-SQL `_lookup_tier12()` pre-check across suffix-stripped candidates before paying for full `normalize_entity()`, after an earlier session explicitly deferred this optimization.
- **Evidence**: `WBC-7`→"White blood cell count" in 3.65s (one SapBERT call, down from N); `RBC-3` in 2.03s. User explicitly reversed the earlier deferral once it became a throughput blocker.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§2)
- **Date**: 2026-08-17

### Fuzzy (Levenshtein) candidate channel added, conditional only
- **Decision**: Added a fuzzy-match supplement firing only when Tier 3 is already uncertain (below floor or margin-ambiguous), capped at 6 candidates, tagged "4 (Fuzzy)", never overriding a confident Tier 1–3 match.
- **Evidence**: "spirnolactone" (typo for spironolactone) had no correct candidate among Tier 1–3 results at all; fuzzy matching (edit distance 1) surfaced it, and re-running Stage 3 produced the correct resolution at 0.9635 confidence (vs. the original wrong-but-confident 0.916).
- **Source**: docs/Stage3_Open_Issues.md (Issue 3)
- **Date**: 2026-08-11

### Fragile Concept Gate caps unanimous votes narrowly, gated on precise provenance
- **Decision**: A 3-0 unanimous vote is capped at MOLLM_RESOLVED (not AUTO_VALIDATED) specifically when the winning candidate came from the lab-value-suffix salvage fallback, gated on the exact `normalized_from` field rather than a broader `gliner_label=='Lab Test' AND match_tier==3` heuristic.
- **Evidence**: The broader heuristic was checked and would have wrongly caught 40% of clean, legitimate resolutions.
- **Source**: docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md (§4)
- **Date**: 2026-08-14

### Compound-concept names exempted from hierarchy-dedup collapse
- **Decision**: `_collapse_hierarchy_duplicates()` exempts any candidate whose name contains a coordinating conjunction ("and"/"&") via `_is_compound_concept_name()`.
- **Evidence**: Was silently discarding gold-correct "Intermaxillary fixation of mandible and maxilla" in favor of a higher-scored single-component sibling; verified this doesn't regress the original ALT-trio collapse case.
- **Source**: docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md (§2, Fix 1)
- **Date**: 2026-08-14

### Organism concept class excluded from domain-conflict retry; Event class kept
- **Decision**: `_detect_domain_conflict()`'s domain-relaxed retry excludes `concept_class_id='Organism'` but explicitly keeps `'Event'`.
- **Evidence**: "galea" (Latin: helmet) scored 0.90 against "Genus Galea" (a rodent genus) vs. 0.61 for the correct anatomy concept. Confirmed all ~3,561 Event-class concepts in the vocab are legitimate COVID-exposure/contact-tracing Observations, not noise.
- **Source**: docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md (§2, Fix 2)
- **Date**: 2026-08-14

### Vocab restriction relaxed on a second domain-conflict retry for generic drug classes
- **Decision**: When the domain-relaxed, vocab-restricted retry still fails, `_detect_domain_conflict()` retries again with the full `DEFAULT_VOCAB`.
- **Evidence**: A Medication-labeled generic mention like "diuretics" could never find gold's SNOMED Procedure-domain "X therapy" concept, since RxNorm has no class-level equivalent, only ingredients/products.
- **Source**: docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md (§2, Fix 3)
- **Date**: 2026-08-14

### Declined to raise `CANDIDATE_LIMIT` (from 3) as a GOLD_MISSING fix — later raised anyway, for a different reason
- **Decision (2026-08-14)**: Rejected raising `CANDIDATE_LIMIT` as a fix for GOLD_MISSING cases.
- **Evidence**: Recovers only 14.6% of remaining GOLD_MISSING at limit=10, 34.3% at limit=50 — steeply diminishing, and conflicted with the finding that longer candidate lists measurably hurt 3B-model Stage 3 accuracy.
- **Decision (2026-08-16, superseding)**: `CANDIDATE_LIMIT` bumped 3→5 anyway, as part of the Phase 3 hybrid-retrieval build, matching the spec's "Top-5" target — a different rationale from the rejected 2026-08-14 proposal.
- **Source**: docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md (§2); docs/2026-08-16_Phase3_HybridRetrieval_Validation.md
- **Date**: 2026-08-14 (rejected) / 2026-08-16 (bumped for a different reason)

### Tier 3 hard cutoff (`NO_CANDIDATE`) kept despite a real precision/coverage trade-off
- **Decision**: Stage 2b's Tier 3 floor branch returns `NO_CANDIDATE` (not a flagged weak match) below `TIER3_SIMILARITY_FLOOR=0.72`, per explicit user direction, despite the measured cost — not yet finally decided whether to keep/revert/soften.
- **Evidence**: Full 27-note corpus (2,303 entities): precision 47.9%→50.7% (+2.8pt), coverage 71.9%→59.7% (−12.2pt). Of 369 flipped entities, 190 were genuine garbage but 90 were genuinely gold-correct matches now lost outright (e.g. `WBC-13.0`→"White blood cell count" at 0.8694).
- **Source**: docs/2026-08-15_Stage1_2a_2b_Remediation.md (§5); docs/Implementation_Checklist.md (Stage 2b)
- **Date**: 2026-08-15

### Dedup-key case-folding restricted to the cache key only, not to Tier 1/2 SQL matching
- **Decision**: `process_and_normalize_entities()`'s cache key collapses whitespace/case for the key only; Tier 1/2 SQL matching remains case-sensitive.
- **Evidence**: A prior session already documented a real case-sensitivity collision (CTA/cTa) as a deliberately deferred trade-off; folding case in the matching logic itself would silently re-decide that already-weighed decision.
- **Source**: docs/2026-08-15_Stage1_2a_2b_Remediation.md (§3)
- **Date**: 2026-08-15

### Brand-name synonym-coverage: no fix needed, audited and closed
- **Decision**: A perceived `athena_concept_synonym` brand-name coverage gap was checked and closed without a code fix.
- **Evidence**: 15 common brand names checked directly — every real brand name resolves at Tier 1 (RxNorm `concept_class_id='Brand Name'` uses the brand as `concept_name`); the checklist's own motivating "lasix failing" example did not reproduce.
- **Source**: docs/2026-08-15_Stage1_2a_2b_Remediation.md (§4)
- **Date**: 2026-08-15

### SNOMED crosswalk target-selection logic fixed with explicit preference ordering
- **Decision**: `VocabularyRetriever.snomed_code_for_concept()` fixed to prefer (1) `RxNorm - SNOMED eq` over `Mapped from`/`Value mapped from`, (2) domain match, (3) lowest `concept_id` — replacing an unordered `ORDER BY concept_id ASC LIMIT 1`.
- **Evidence**: RxNorm "warfarin" has 18 relationship rows spanning Drug/Observation/Procedure domains; old logic landed on a Procedure concept instead of the drug product. 153/314 (48.7%) of distinct Medication-domain crosswalks changed after the fix.
- **Source**: docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md (§5)
- **Date**: 2026-08-17

### Medication gold-annotation mismatch: not chased via crosswalk tuning or architecture change
- **Decision**: Decided not to chase the residual Medication precision gap via further crosswalk tuning or by making the pipeline resolve to administration-event/Procedure concepts to match the gold convention.
- **Evidence**: 11/14 gold codes for gradable Medication entities were Procedure-domain "Administration of X" concepts, not drug substance — a genuine annotation-schema difference (corpus-wide gradable Medication precision measured 31.5%, 17/54), not a pipeline defect; judged clinically questionable to architecturally chase one dataset's convention.
- **Source**: docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md (§6)
- **Date**: 2026-08-17

### Allergy-context override: search Condition domain instead of Medication domain
- **Decision**: Built `STATUS_ALLERGY` + `apply_allergy_context_override()`: for allergy-asserted Medication-labeled entities, search becomes `"Allergy to {text}"` against the Condition domain instead of RxNorm/Medication.
- **Evidence**: A precisely-sized, 100%-affected population: 19 entities across 32 notes in an Allergies section, 100% Medication-labeled with `assertion_status=PRESENT`. Root cause: `morphine`/`trazodone` resolved to drug PRODUCT concepts instead of gold's "Allergy to X" Clinical Finding concepts.
- **Source**: docs/2026-08-16_Shadow_Run_Precision_At_Scale.md
- **Date**: 2026-08-16

### OMOP standard-concept-only convention narrowly relaxed, only for the allergy exact-match pattern
- **Decision**: Of three options (general non-standard↔standard crosswalk, narrow relaxation, or accept the generic ceiling), implemented `_apply_allergy_nonstandard_exact_override()`: an exact case-insensitive match without the `standard_concept='S'` filter, scoped only to this one search pattern.
- **Evidence**: SNOMED's own "Allergy to morphine" (id 4164683, `standard_concept=NULL`) was invisible because every tier filters standard-only; `athena_concept_relationship` showed it "Maps to" only the generic "Allergy to drug." Verified correct post-fix (concept_id 4164683 returned both in isolation and full pipeline).
- **Source**: docs/2026-08-16_Shadow_Run_Precision_At_Scale.md
- **Date**: 2026-08-16

### Allergy domain restriction widened to include Observation domain, plus a narrow tiebreak
- **Decision**: `search_domain_override` widened from `['Condition']` to `['Condition','Observation']`; added `_apply_allergy_domain_tiebreak()` (promote a non-top Observation candidate within 0.03 of the top score).
- **Evidence**: "Allergy to Penicillins" scored 0.9880 (Observation, gold-correct) vs. 0.9165 (Condition, previously chosen) — widening alone sufficient. "Allergy to NSAIDS" scored 0.8078 (Condition) vs. 0.7867 (Observation, gold-correct) — a genuine near-tie needing the tiebreak. Final: 8/8 = 100% on the 6-note allergy population (up from 6/8).
- **Source**: docs/2026-08-16_Shadow_Run_Precision_At_Scale.md
- **Date**: 2026-08-16

### Brand-to-generic retry added to the allergy exact-match override
- **Decision**: `_apply_allergy_nonstandard_exact_override()` gained a retry via brand→generic name expansion on a direct exact-match miss.
- **Evidence**: `Elavil`→"Allergy to amitriptyline", `Reglan`→"Allergy to metoclopramide" now resolve; `Spiriva` remains Unmapped (not an exact RxNorm Brand Name concept-class match) — documented as closing one real gap, not all.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§3)
- **Date**: 2026-08-17

### RRF hybrid retrieval built and evaluated, kept OFF
- **Decision**: `CNSP_HYBRID_RETRIEVAL` stays off in production; dense-only (`w_dense=1.0`) kept as default.
- **Evidence**: Grid search (155-160 clean-span entities, 32 notes): dense-only → Top-1 61.3%, oracle 74.2%; every grid point declines monotonically as sparse weight increases (down to 37.7%/48.1% at pure sparse).
- **Source**: docs/2026-08-16_Phase3_HybridRetrieval_Validation.md
- **Date**: 2026-08-16

### Tier 3 similarity-floor check fixed to use pool max, not `candidates[0]`
- **Decision**: Both floor checks now compare against the pool's best dense score, not the (possibly RRF-reordered) first candidate.
- **Evidence**: An A/B showed hybrid producing 10 more zero-candidate rejections than dense-only despite identical top-5 correctness counts — a shrinking-denominator artifact. Post-fix dense-mode was byte-identical to the pre-fix run on every metric.
- **Source**: docs/2026-08-16_Phase3_HybridRetrieval_Validation.md
- **Date**: 2026-08-16

### No extension of `VOCAB_BY_LABEL` to additional non-SNOMED vocabularies
- **Decision**: Concluded no extension was warranted for Anatomy/Procedure/Measurement domains.
- **Evidence**: Empirical check of real `vocabulary_id` distribution per OMOP domain: Anatomy already 100% SNOMED; Procedure/Measurement's larger non-SNOMED vocabularies (ICD10PCS, LOINC) aren't this pipeline's target crosswalk vocabulary.
- **Source**: docs/2026-08-16_Phase3_HybridRetrieval_Validation.md
- **Date**: 2026-08-16

### GLiNER-Linker (bi-encoder and cross-encoder rerankers) evaluated and rejected
- **Decision**: Declined to integrate `gliner-linker-large-v1.0` or `gliner-linker-rerank-v1.0` as a Stage 2b reranking signal for the Condition/Observation duplicate problem; kept the existing MoLLM-ensemble-plus-prior mechanism.
- **Evidence**: Scored against all 10 gradable cases of the known-duplicate population: `gliner-linker-large` 4/10, `gliner-linker-rerank` 5/10 — the existing MoLLM tiebreak (strengthened prior) got 7/7 on the same population.
- **Source**: docs/2026-08-18_GLiNER_Linker_Reranker_Evaluation.md
- **Date**: 2026-08-18

### Isolated Python 3.11 venv used to evaluate GLiNER-Linker, main environment untouched
- **Decision**: Used a separate `.venv_gliner_linker` (Python 3.11) via subprocess/JSON, rather than upgrading the main pipeline's Python/`transformers`.
- **Evidence**: `glinker` requires Python ≥3.10 (main pipeline pinned 3.9.25 for scispaCy/medspacy); checkpoints saved in `transformers==5.0.0` format, incompatible with the pipeline's pinned 4.57.6.
- **Source**: docs/2026-08-18_GLiNER_Linker_Reranker_Evaluation.md
- **Date**: 2026-08-18

---

## 5. Stage 3 — MoLLM Ensemble & Tier Gate

### Model stack evolution: MedGemma → BioMistral/OpenBioLLM → qwen2.5:3b/llama3.2:3b/phi4-mini
- **Decision (step 1)**: Replaced MedGemma 4B with BioMistral 7B (AWQ), preserving architectural diversity against OpenBioLLM (Llama-3 base).
  - **Evidence**: MedGemma (Gemma-3-based) could not run on the project's Tesla T4 under any dtype — vLLM refuses `float16` for gemma3 (instability), `bfloat16` needs compute capability ≥8.0 vs. the T4's 7.5, `float32` needs ~16GB against 15.36GB VRAM.
  - **Source**: docs/MoLLM_Stage3_Retrieval_Design.md
  - **Date**: 2026-08-09
- **Decision (step 2)**: Standing ensemble became `qwen2.5:3b`/`llama3.2:3b`/`phi4-mini` via Ollama, replacing the vLLM BioMistral/OpenBioLLM pair.
  - **Evidence**: Section 3 voting audit on 768 LOW-tier entities (28 notes): GOLD_PRESENT_CORRECT 22.2%, established as a workable harness; BioMistral separately carried a measured 23.8%-of-verdicts decoding-degeneration defect.
  - **Source**: docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md (§6)
  - **Date**: 2026-08-14

### Sentence-bounded, capped `local_context` instead of note-scoped context
- **Decision**: `local_context` for prompts is sentence-bounded, capped ~800 chars; the smaller of the two models' context windows (OpenBioLLM's 8K, not BioMistral's 32K) treated as the binding budget, operationally 4,096 tokens.
- **Evidence**: Both models must see identical input for votes to be comparable; a long note alone is ~6,200 tokens, exceeding OpenBioLLM's window.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§3)
- **Date**: 2026-08-08

### `frequency_penalty` added instead of altering temperature, to fix degeneration
- **Decision**: `FREQUENCY_PENALTY=0.4` added to both models' kwargs; `TEMPERATURE=0.0` kept for reproducibility.
- **Evidence**: 475/3990 verdicts corpus-wide (11.9%) showed degenerate repetition, 100% (475/475) from BioMistral, ~23.8% of every BioMistral verdict — always paired with `raw_confidence_label=None`, terminating at `MAX_OUTPUT_TOKENS=800`.
- **Source**: docs/2026-08-13_Calibration_Diagnostics_And_Fixes.md (§3.3)
- **Date**: 2026-08-13

### Standing policy: every Stage 3 run must use the calibrator, never raw `composite_confidence`
- **Decision**: Policy set that any Stage 3 batch run must be invoked with `--calibrator`.
- **Evidence**: Accuracy on raw `composite_confidence`: 20/140 = 14.29%, ECE=0.773; threshold sweep flat at 14.3% precision from t=0.05 to t=0.85. Self-reported HIGH/LOW confidence carried no discriminative signal (BioMistral HIGH 13.9% correct; OpenBioLLM HIGH 14.0%).
- **Source**: docs/2026-08-13_Calibration_Diagnostics_And_Fixes.md (§5.4, §8)
- **Date**: 2026-08-13

### System prompt's "no outside clinical knowledge" rule narrowed, scoped to the labeling task
- **Decision**: Rewrote the unconditional instruction to permit terminology/domain knowledge when no guideline evidence is retrieved, scoped strictly to "is this the correct label," keeping the anti-fabricated-citation guard.
- **Evidence**: Models were deliberately chosen for clinical fine-tuning, but the old instruction suppressed that expertise in ~91.5% of records with no guideline evidence.
- **Source**: docs/MoLLM_Redesign_Proposal.md (§9)
- **Date**: 2026-08-11

### `grounding_basis` field added for auditability
- **Decision**: Added `grounding_basis ∈ {guideline_rule, ontology_only, model_terminology_knowledge}`, with the last tier required to carry the highest evidentiary bar.
- **Evidence**: Keeps the "white box, traceable" claim intact while using trained clinical knowledge, rather than one undifferentiated confidence number; a prior case showed both models agreeing confidently (0.916) on a wrong answer even with candidates and citation guard active.
- **Source**: docs/MoLLM_Redesign_Proposal.md (§9)
- **Date**: 2026-08-11

### MoLLIA/MoLAM literature: partial adoption only
- **Decision**: Adopted 3 mechanisms (trained calibrator replacing hand-tuned thresholds; annotation-discrepancy signal repurposed against Stage 2b's own judgment; negative-candidate bookkeeping as logging) — explicitly rejected a full architectural port (iterative AL retraining, negative learning loss, T-fold self-consistency, 5-member ensemble).
- **Evidence**: This project's own ECE=0.7529 finding (flat across correct/incorrect) matched MoLLIA's own documented failure mode for hand-combined vs. trained aggregators. T-fold rejected because measured throughput (5.32s/record, T=1) on a budgeted single T4 GPU meant T=5 likely wouldn't fit the KV-cache budget.
- **Source**: docs/MoLLM_Redesign_Proposal.md (§1, §3-5)
- **Date**: 2026-08-11

### Calibrator score can only substitute for the threshold comparison, never bypass hard safety rules
- **Decision**: The three hard gates (model disagreement, citation failure, CONTRADICTED/INSUFFICIENT_EVIDENCE/NONE_CORRECT) stay unchanged before any calibrator score is consulted.
- **Evidence**: A real case (spirnolactone→SPIRAPRILAT) had `composite_confidence=0.916` and `ensemble_agreement=True` both wrong; only `citation_verified=False` caught it — direct evidence a learned score is less trustworthy than a symbolic check when they conflict. Verified via stub tests.
- **Source**: docs/MoLLM_Redesign_Proposal.md (§4, §11)
- **Date**: 2026-08-11

### `d_anno` (annotation discrepancy) adopted as a calibrator feature, not a standalone gate
- **Decision**: Computed `d_anno = I[MoLLM's candidate ≠ Stage 2b's top-ranked candidate]`, added as a soft feature, not a fourth hard gate.
- **Evidence**: MoLLIA's own ablation found it the single best uncertainty signal; kept soft because Stage 2b-vs-3 disagreement is exactly the case the resolution mechanism exists to adjudicate.
- **Source**: docs/MoLLM_Redesign_Proposal.md (§4)
- **Date**: 2026-08-11

### `cited_evidence` schema field left free-form, not constrained to known rule IDs
- **Decision**: Deliberately did not constrain the field to an enum of known rule IDs.
- **Evidence**: Constraining would make fabrication structurally impossible to occur but also impossible to observe — a test confirmed fabrication is genuine model behavior (5/7 unchanged after making the field schema-optional), not a schema artifact.
- **Source**: docs/Stage3_Open_Issues.md (Issue 2)
- **Date**: 2026-08-10/11

### Resolution-mode prompt rewritten to explicitly prefer NONE_CORRECT over a nearest-spelled wrong candidate
- **Decision**: Rewrote the resolution-mode prompt to name the specific failure mode (nearest-spelled ≠ nearest-meaning) and direct the model toward `NONE_CORRECT`.
- **Evidence**: `spirnolactone` — both models resolved to SPIRAPRILAT despite BioMistral's own reasoning correctly identifying it as a diuretic; `NONE_CORRECT` was available but unused.
- **Source**: docs/Stage3_Open_Issues.md (Issue 3)
- **Date**: 2026-08-10/11

### `raw_confidence_label` recorded but never used for routing
- **Decision**: Self-reported HIGH/LOW confidence stored in provenance, not used to route decisions.
- **Evidence**: BioMistral returned HIGH on 10/10 records; OpenBioLLM returned LOW on 7/10 including records where it agreed with BioMistral — a per-model bias, not a per-record signal.
- **Source**: docs/Stage3_Open_Issues.md (Issue 4)
- **Date**: 2026-08-10

### `verify_citations()` ANDs all models' checks — kept, not loosened
- **Decision**: A record is only `citation_verified` if EVERY model's citation check passes.
- **Evidence**: On a real record, BioMistral cited perfectly (containment 1.0) but OpenBioLLM's flawed citation still forced HITL — "the ensemble's guarantee should be as strong as its weakest member."
- **Source**: docs/Stage3_Issue1_Rule_Backfill.md
- **Date**: 2026-08-11

### Devil's-advocate / conceptual-firewall prompt rewrite to de-anchor from Stage 2 confidence
- **Decision**: Rewrote both `mollm_review.py` and `mollm_ensemble.py` SYSTEM_PROMPTs with explicit anti-anchoring instructions (ignore score, adopt devil's-advocate posture, context supreme over string similarity).
- **Evidence**: Recall was inversely correlated with Stage 2 confidence — exact matches (score=1.0) caught only 24.0% of real errors vs. 63.2% for score<0.7. After the fix: CONFIRM→FLAG flips 16/20 (80%), per-model flag rate 14/60→41/60.
- **Source**: docs/2026-08-15_Contradiction_Detection_Analysis.md
- **Date**: 2026-08-15

### Objective 2's hard evidence-gating rule not loosened despite a zero-flip result
- **Decision**: Kept the rule "cannot return CONTRADICTED without guideline evidence" despite 0/20 flips to CONTRADICTED after the anti-anchoring fix.
- **Evidence**: Per-model confirm rate still dropped 81.7%→61.7% on the same 20 cases (2 flipped to unanimous INSUFFICIENT_EVIDENCE) — none of the 20 had guideline evidence retrieved from the narrow 1,700-node KG, so CONTRADICTED was structurally unreachable, not evidence the rule failed. Loosening it "would reopen exactly the citation-hallucination risk the override gate exists to prevent."
- **Source**: docs/2026-08-15_Contradiction_Detection_Analysis.md
- **Date**: 2026-08-15

### Blanket human review reaffirmed via isolated recall measurement
- **Decision**: The decision to gate all Stage 3 output behind human review, regardless of tier, was reaffirmed rather than revisited.
- **Evidence**: Contradiction-detection recall (of genuinely wrong top-1s caught) measured 31.2% (mollm_decisions) / 29.7% (mollm_review_decisions) — the ensemble rubber-stamps ~7 of 10 real Stage 2 errors shown to it.
- **Source**: docs/2026-08-15_Contradiction_Detection_Analysis.md
- **Date**: 2026-08-15

### Qualifier-fragment precheck routes standalone qualifiers to HITL, zero model calls
- **Decision**: `qualifier_fragment_precheck()` routes any `gliner_label=="Qualifier"` entity straight to HITL before any ensemble call.
- **Evidence**: Confirmed Qualifier is exactly the label covering fragment entities ('left','right','multiple') that caused an earlier precision collapse.
- **Source**: docs/2026-08-15_Phase2_TierGate_Validation.md
- **Date**: 2026-08-15

### Loosened "match core concept, not every detail" 5th rule rejected and reverted
- **Decision**: Tested and reverted a proposed 5th matching rule applied to all three models.
- **Evidence**: Dropped Tier 1 precision to 5.9% (1/17) by letting all three models unanimously rubber-stamp wrong matches on bare qualifier fragments.
- **Source**: docs/2026-08-15_Phase2_TierGate_Validation.md
- **Date**: 2026-08-15

### Subsumption clause applied only to qwen2.5:3b, not the other two models
- **Decision**: `QWEN_SUBSUMPTION_CLAUSE` applied only to qwen's Step B prompt.
- **Evidence**: Stored model reasoning showed qwen was the consistent dissenting vote on obviously-correct cases, demanding contextual specificity the candidate's bare name never carried — the other two models weren't over-rejecting.
- **Source**: docs/2026-08-15_Phase2_TierGate_Validation.md
- **Date**: 2026-08-15

### Phase 2 locked at 50.0% coverage / 94.4% precision rather than pushed further
- **Decision**: Accepted the two-step CoT + Tier 1-5 gate as the milestone rather than continuing to force consensus on remaining split cases.
- **Evidence**: 17/18 precision on a curated atomic-only sample, 50.0% coverage (up from 2.8%); the one error traced to upstream retrieval, not gate judgment. Pushing further "risks reproducing the 2026-08-15 precision-collapse failure mode for no real gain," since Tier 4/5 already route safely to review.
- **Source**: docs/2026-08-15_Phase2_TierGate_Validation.md
- **Date**: 2026-08-15/16

### Curated atomic-only sample built instead of a larger unfiltered batch
- **Decision**: Built `select_atomic_entities()` (Condition/Procedure/Medication, exactly one gold overlap, span ≥ gold) instead of a bigger unfiltered sample.
- **Evidence**: A prior unfiltered 36-entity batch produced zero clean data points — every gradable decision carried a `compound_span` or `narrower_than_gold` confound. The curated batch (pool of 549) yielded 18 confound-free gradable decisions.
- **Source**: docs/2026-08-15_Phase2_TierGate_Validation.md
- **Date**: 2026-08-15/16

### `mollm_tier_gate.py` frozen pending Phase 3/4 data
- **Decision**: Froze further prompt changes, deferring to real data from hybrid retrieval and acronym escalation.
- **Evidence**: Remaining coverage gap attributed to `ensemble_split` (36.1%) and `unresolved_acronym` volume, expected to convert into Tier 1/2 coverage from retrieval/acronym improvements, not further prompt tuning.
- **Source**: docs/2026-08-15_Phase2_TierGate_Validation.md
- **Date**: 2026-08-15/16

### Conservative dry-run deployment chosen over full production cutover
- **Decision**: `route_tier()` run on live data, but `ingest_auto_decision()` kept `dry_run=True`, every decision still queued regardless of tier.
- **Evidence**: The 94.4% precision figure was based on only 18 gradable entities — "not yet the scale this project's own standards require before writing unreviewed."
- **Source**: docs/2026-08-16_Phase2_Gate_Deployed_DryRun.md
- **Date**: 2026-08-16

### Tier-gate decisions stored in a new, separate table
- **Decision**: `store_tier_decision()` persists to `mollm_tier_gate_decisions`, kept separate from `mollm_decisions`.
- **Evidence**: The two-step CoT artifact lacks the `ensemble_agreement`/`citation_verified`/`mode` fields the older contradiction-audit-style table uses.
- **Source**: docs/2026-08-16_Phase2_Gate_Deployed_DryRun.md
- **Date**: 2026-08-16

### `dry_run=False` withheld after a natural-population shadow run
- **Decision**: Did not enable real KG3 writes after shadow-testing at scale.
- **Evidence**: 155-entity shadow run: raw 72.5% (37/51), clean-span 76.1%→78.3% after one gold-error adjudication — far below the curated 94.4% the Phase 2 lock was based on.
- **Source**: docs/2026-08-16_Shadow_Run_Precision_At_Scale.md
- **Date**: 2026-08-16

### Every AUTO-tier decision still queued for human review, kept despite tempting exclusion
- **Decision**: `enqueue_pending_cases()` continued queuing every AUTO-tier decision.
- **Evidence**: Validated directly by the shadow-run precision drop (94.4%→~78%) — "most errors explainable" was judged not equivalent to "safe to write unreviewed."
- **Source**: docs/2026-08-16_Shadow_Run_Precision_At_Scale.md
- **Date**: 2026-08-16

### ALLERGY exception carved into ensemble prompt rule 3
- **Decision**: Added `ALLERGY_MEANING_INSTRUCTION`/`ALLERGY_CONTEXT_CLAUSE`, carving an allergy exception into the "ignore assertion status" rule.
- **Evidence**: Root-caused via stored model trails: Step B rule 3 (correct for negation) pushed models to reject the exactly-correct "Allergy to X" candidate. Aspirin/fluconazole/morphine moved TIER_4→TIER_1 after the fix.
- **Source**: docs/2026-08-16_Shadow_Run_Precision_At_Scale.md
- **Date**: 2026-08-16

### Retroactive reprocessing scheduled sequentially, not parallel with the in-flight batch
- **Decision**: `scripts/retroactive_lab_procedure_fix.py` scheduled only after the in-flight batch's Stage 3 completes.
- **Evidence**: Two earlier attempts at parallel Stage 1-2b/Stage 3 execution caused real DB-lock contention — one causing actual data loss, one full starvation.
- **Source**: docs/2026-08-19_Lab_Procedure_Vs_Observable_Entity_Finding.md
- **Date**: 2026-08-19

### `ConsensusCalibrator` rewritten from scratch, per explicit instruction
- **Decision**: Rewritten with no reference to the superseded `MoLLMCalibrator`.
- **Evidence**: A first draft still implicitly mirrored the old module's naming/framing even after removing literal references; final version verified via grep to have zero references.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§5)
- **Date**: 2026-08-17

### `TIER_1B_CALIBRATED_AUTO_VALIDATED` kept structurally distinct from genuine Tier 1
- **Decision**: A calibrator-promoted decision routes to a separate tier, never merged into Tier 1 in any count.
- **Evidence**: Design constraint verified via tests: default no-op reproduces prior behavior byte-for-byte; calibrator never consulted when the plurality verdict is NONE_CORRECT.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§5)
- **Date**: 2026-08-17

### `ACRONYM_ESCALATION_ENABLED` kept OFF after corpus-scale grading, reversing an optimistic small-sample read
- **Decision**: Stays off in production.
- **Evidence**: Small-sample reads (15/15, 13/14) did not hold — corpus-scale grading on 31 notes: 34.3% then 36.1% precision, root-caused to systematic textbook-prior bias ("LAD"→"left anterior descending artery" even when gold means "Lymphadenopathy"; "NAD"→biochemistry reading instead of "no acute distress").
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§6)
- **Date**: 2026-08-17

### `MIN_CACHE_HIT_COUNT=2` added to acronym-priors cache
- **Decision**: A cached resolution now requires 2 independent successes, not 1, before being trusted without a model call.
- **Evidence**: "PDA" (coronary-stenosis context) was confidently resolved wrong by all 3 models, cleared Tier 1/2/3 cleanly, and would have entrenched permanently under the old rule.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§1)
- **Date**: 2026-08-17

### Overnight run scoped to the 31-note test corpus, not the full 272
- **Decision**: First unattended batch scoped to 31 notes.
- **Evidence**: Resolved the interrupted acronym-escalation corpus-scale question and produced enough volume to fit `ConsensusCalibrator`, without added runtime/unknowns of a much larger first unattended run.
- **Source**: docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md (§8)
- **Date**: 2026-08-17

### Calibrator threshold-lowering rejected (No-Go) after a 51-note retrain diagnostic
- **Decision**: Declined to lower `CALIBRATED_AUTO_THRESHOLD` below 0.72; the diagnostic retrain not deployed.
- **Evidence**: Val AUROC 0.701, below the 0.74 baseline. Threshold sweep found no band reaching the pre-agreed 98%+ precision bar; the production 0.72 threshold itself measured only 89.5% on this larger set vs. ~100% on the original smaller val set.
- **Source**: docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md (§8)
- **Date**: 2026-08-17

### `TIER_2_AUTO_RESOLVED` held out of `AUTO_TIERS`, confirmed correct by later data
- **Decision**: Kept Tier 2 excluded from auto-write despite fixing its two identified root causes; a third-party suggestion to eliminate Tier 2 outright was reviewed and declined in favor of this narrower gate.
- **Evidence**: Fresh25 grading confirmed the decision: Tier 2 precision on fresh notes was 11/68=16.2%, not recovered by the fixes — 100% (259/259) of Tier 2 decisions are `is_ambiguous=True` by construction, so "3/3 unanimous" reflects shared model bias.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§1, §6)
- **Date**: 2026-08-20

### Calibrator retrained on a larger, note-disjoint pool and adopted as new baseline
- **Decision**: Retrained and adopted, keeping the prior model as `.bak` (script's own adoption gate: only overwrite if new AUROC beats current).
- **Evidence**: Val AUROC 0.74→0.845; fresh25 confirmed it holds at 92.6% (100/108) precision on genuinely fresh notes.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§2, §6)
- **Date**: 2026-08-20

### Three shelved 8B hard-case architectures — all built, measured, kept out of production
- **Decision**: `tier4_kg_escalation.py`, `tier2b_llm_candidate_generation.py`, `tier4_arbiter_8b.py` all kept unwired; prioritized doubling down on already-validated components instead.
- **Evidence**: KG-escalation scored 27.8% (5/18, 58% of cases had only one candidate — a retrieval bottleneck). Candidate-generation scored 10.7% recall recovery. Arbiter was strongest (38.0%→51.0% precision, N=100) but still judged not statistically airtight enough for the active gate.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§3)
- **Date**: 2026-08-20

### KG search-loop (multi-round 8B escalation) tried, not adopted
- **Decision**: The one-shot pre-fetch arbiter architecture (51.0% precision) remains the reference design; a multi-round search-loop alternative was smoke-tested and not pursued.
- **Evidence**: Even after fixing a real bug, 7/9 smoke-tested entities still declined to commit to a verdict after 2-4 search rounds — judged evidence a 3-8B-class model lacks reliable planning/executive function to drive its own search strategy.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§7)
- **Date**: 2026-08-20

### Hardcoded lab-procedure-preference rule kept over the KGE topological tiebreak
- **Decision**: Kept `_prefer_lab_procedure_over_observable()`; declined to replace it with the KGE tiebreak despite a plausible argument that KGE should generalize better.
- **Evidence**: Head-to-head on the rule's own pattern: rule 0 losses at every threshold (0.01-0.08, n up to 380); KGE 63 losses past 0.02. A specific claim that KGE would "naturally" resolve a third near-duplicate was directly falsified against the actual retrained model (identical wrong pick, 0.0018 margin — noise).
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§12, §14)
- **Date**: 2026-08-20

### KGE tiebreak built, evaluated, left unwired from production
- **Decision**: Not wired into routing.
- **Evidence**: On the broader population beyond the rule's scope, KGE showed a genuine positive net (265 win/181 loss at threshold 0.03) — real signal, but a ~9.9% loss rate judged not risk-free without a calibrated gating mechanism that doesn't yet exist.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§12, §14)
- **Date**: 2026-08-20

### `EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` left default-on; HITL-routing mitigation deferred
- **Decision**: Flag stays on; a proposed mitigation (route tiebreak-eligible entities straight to HITL) documented, not implemented.
- **Evidence**: Tiebreak-eligible population precision: 14.3% (3/21) vs. 84.7% (265/313) non-eligible, on a 5-note/989-decision scope. The flag's one verified win (wound-dehiscence pattern) remains real; cost already known (~34% more LLM calls).
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§12, §14)
- **Date**: 2026-08-20

### Deflection-rate formula corrected: ALL `AUTO_TIERS` decisions over ALL Stage 3 decisions
- **Decision**: Fixed a real bug where deflection was computed as a gradable-restricted AUTO count over an unrestricted total.
- **Evidence**: Earlier chat-reported figures of 35.0%/22.4% were wrong; corrected to 57.0%/31.2% (corpus-wide/fresh-10) once reconciled against the raw tier distribution.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§15)
- **Date**: 2026-08-20

### 76.8% fresh-10-note AUTO-tier precision adopted as the paper's headline number
- **Decision**: Report 76.8% instead of earlier higher development-set figures (94.4%, 98.0%).
- **Evidence**: The earlier figures were measured on notes used to debug the exact mechanisms being scored — the same circularity the project's own `prior_confirmation_count` ablation had already demonstrated for a different mechanism. 76.8% (43/56) measured on the official locked test split, mostly outside calibrator training, none used to develop the fixes being reported.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§13, §14)
- **Date**: 2026-08-20

### Unsourced "45.3%" figure dropped from a draft table
- **Decision**: Dropped rather than included with a guessed source.
- **Evidence**: No record exists of this number being measured in this session; could not be verified.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§14)
- **Date**: 2026-08-20

---

## 6. Stage 4/5 — HITL Routing & KG3 Ingestion

### Every Stage 3 decision queued for human review, regardless of tier
- **Decision**: `enqueue_pending_cases()` queues everything from `mollm_decisions` and `mollm_review_decisions`, regardless of routing tier, rather than writing high-confidence output straight to KG3.
- **Evidence**: AUTO_VALIDATED precision measured 39.4% (experimental harness) and 52.6% (real production path, 532 gradable, 280 correct) — confirming the checklist's own pre-existing pseudo-labeling feedback-loop warning.
- **Source**: docs/2026-08-15_Stage4_Stage5_Build.md
- **Date**: 2026-08-14/15

### Both legacy decision tables unified into one HITL queue, tagged by source
- **Decision**: `enqueue_pending_cases()` reads and unifies `mollm_decisions` (Objective 2) and `mollm_review_decisions` (Objective 3), tagged by `source_table`.
- **Evidence**: The two tables use different designs (citation-gated vs. confidence-driven all-tier); Objective 3's own docstring states it exists to feed a future Stage 4 job.
- **Source**: docs/2026-08-15_Stage4_Stage5_Build.md
- **Date**: 2026-08-14/15

### Real production Stage 3 paths run to completion before trusting the experimental harness
- **Decision**: Both production batch runners run to completion before building Stage 4, rather than relying on the experimental harness's own measurement.
- **Evidence**: The experimental harness (`experiment_3b_voting.py`) never touches `mollm_decisions`/`mollm_review_decisions`; both tables held only stale data or none. Production run: 2247+2297 processed, 0 errors.
- **Source**: docs/2026-08-15_Stage4_Stage5_Build.md
- **Date**: 2026-08-14/15

### `mollm_review_decisions` APPROVED cases with no concept_id raise loudly, never guessed
- **Decision**: `_suggested_omop_concept_id()` sets `None` for these cases; ingestion raises `UningestibleCase` rather than inferring a concept from `proposed_concept_name`.
- **Evidence**: `mollm_review_decisions` rows carry no concept_id anywhere, only a proposed-name string.
- **Source**: docs/2026-08-15_Stage4_Stage5_Build.md
- **Date**: 2026-08-14/15

### KG3 ingestion uses `MERGE`, not `CREATE`, for idempotency
- **Decision**: `ingest_reviewed_case()` writes exclusively via `MERGE`-keyed Cypher.
- **Evidence**: Verified against a real running Memgraph instance: full chain readable after write, node counts unchanged after a deliberate re-ingestion of the same case.
- **Source**: docs/2026-08-15_Stage4_Stage5_Build.md
- **Date**: 2026-08-14/15

### Stage 5 active-learning feedback mechanisms deliberately not built
- **Decision**: The GLiNER prompt-feedback mechanism and CompGCN/TransE/RotatE re-ranking layer were not built this phase — only read-interface foundations.
- **Evidence**: 0 cases had been human-reviewed at session end (4,763 queued, pending) — "building the feedback mechanism now would mean building it against no data."
- **Source**: docs/2026-08-15_Stage4_Stage5_Build.md
- **Date**: 2026-08-14/15

### HITL pre-review queue stored in DuckDB, not Memgraph
- **Decision**: `hitl_review_queue` implemented as a DuckDB table; only reviewed cases (APPROVED/CORRECTED) get written to Memgraph as a durable `:PatientObservation→:MoLLMDecision→:HITLReview` chain — REJECTED cases never reach the graph.
- **Evidence**: Stated reasoning: the queue churns constantly (everything queued regardless of tier) and reuses DuckDB's proven read/write conventions; Memgraph reserved for a durable, queryable provenance ledger of finished decisions.
- **Source**: docs/Databases.md (§3)
- **Date**: 2026-08-15

### Stage 4 write-back gated behind blanket human review, not a calibrated threshold
- **Decision**: Every row queued regardless of tier, until the calibrator is fit against real reviewed outcomes.
- **Evidence**: Full-corpus AUTO_VALIDATED precision measured 52.6% (532 gradable, 280 correct) — better-ordered than other paths but far short of safe-to-skip-review.
- **Source**: docs/Implementation_Checklist.md (Stage 4)
- **Date**: 2026-08-15

---

## 7. Knowledge Graph / Guideline Infrastructure

### ⚠️ Unresolved contradiction, flagged not silently resolved
Two docs, both dated 2026-08-07, state opposite conclusions about which
corpus is the primary guideline-KG source:
- `docs/Guideline_Triplets_KG_Review.md` §6 states **`local_triplets_db2_v6` (MedCAT pipeline)** was selected as final, over `rules-llm`, citing `rules-llm`'s 14.4% dangling-target rate and weaker structural integrity despite better raw grounding (96.5% vs. 50.5%).
- `docs/Rules_LLM_Triplets_Review.md` §§1,4 states **`rules-llm`** was adopted as primary, citing 96.5% SNOMED grounding vs. 50.5%, 0% duplicate nodes vs. 63%, 71% high-containment citations vs. 47%.
Both cite similar grounding numbers but reach opposite conclusions — likely reflects a decision that evolved over the course of that day, or two docs written from different vantage points that were never reconciled. **Whichever is current should be verified directly against `data/` before citing either in the paper.**

### Neo4j/Memgraph split collapsed into unified KG1
- **Decision**: Collapsed the original design (SNOMED in Neo4j, guidelines in Memgraph) into one unified graph.
- **Evidence**: The core traversal (walk IS_A ancestors, collect guideline rules attached to any of them) would be impossible in one query across two separate graph databases.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§4)
- **Date**: 2026-08-08

### SNOMED code rejected as sole join key for guideline retrieval — name-agreement guard required
- **Decision**: Every code-based guideline match requires a name-agreement guard (token_set_ratio ≥0.75 agree / 0.45-0.75 weak ×0.6 / <0.45 reject); a `MERGE ... ON snomed`-style cross-file ingestion was explicitly ruled out.
- **Evidence**: Of 91 guideline codes with >1 distinct node name, 43 (47%) attach clinically unrelated names to one code (e.g. `24484000`="Severe" shared by GOLD 3 severe, major bleeding, severe AKI). Cross-file, `25876001` spans 14 nodes/11 files.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§2, §4)
- **Date**: 2026-08-08/09

### Guideline node identity keyed on `(source_file, @id)`, never SNOMED code
- **Decision**: Grounding expressed as an `ASSERTS_CODE` edge (only when the name-agreement guard passes), not a node property.
- **Evidence**: Merging `24484000`'s three differently-meaning nodes on code would fuse major bleeding with severe AKI into one graph node — an unrecoverable data-integrity error.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§4)
- **Date**: 2026-08-08

### Hierarchy traversal (Channel B) kept load-bearing, capped at 3 hops upward-only, stop-node list imposed
- **Decision**: Channel B (SNOMED IS_A traversal, `0.9^hops` decay) kept as a primary channel, not an optimization; downward traversal excluded; near-root concepts barred as stop-nodes.
- **Evidence**: Guarded combined coverage = 8.53% of 75,491 gold annotations; Channel B alone supplies 3.61 of the 8.53 points (42%). Beyond ~3 hops, SNOMED's upper hierarchy converges on near-root concepts subsuming most of the corpus, making `match_confidence` meaningless.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§2, §5)
- **Date**: 2026-08-08/09

### Ungrounded name-match channel (Channel D) demoted to a constrained fallback
- **Decision**: Runs only when `confidence_tier_in=LOW` and Channels A/B return nothing.
- **Evidence**: Standalone contribution measured at only 0.78% coverage — insufficient to justify a SapBERT pass over 982 node names per entity every time.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§2, §5)
- **Date**: 2026-08-09

### Evidence cap of 5 rules per validation record
- **Decision**: Retrieved guideline rules capped at 5, ranked by match_confidence then citation-type then predicate affinity.
- **Evidence**: Each rule costs ~80-150 tokens; 5 rules ≈400-750 tokens fits budget. Some nodes (e.g. AKI, 62 rules) would otherwise blow the context window alone.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§5)
- **Date**: 2026-08-08

### Empty EVIDENCE block structurally disallows CONTRADICTED
- **Decision**: When no guideline evidence retrieved (~87.5-91.5% of entities), HIGH-tier records may only return SUPPORTED or INSUFFICIENT_EVIDENCE.
- **Evidence**: A contradiction with no retrieved guideline to contradict is definitionally a hallucination — the single most important guardrail given how common the empty-evidence case is.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§6)
- **Date**: 2026-08-08

### Assertion status symbolically gates guideline retrieval before the model sees the record
- **Decision**: `ABSENT`/wrong-experiencer skips guideline retrieval entirely; `HISTORICAL` requires extra justification; `POSSIBLE`/`CONDITIONAL` downgrades CONTRADICTED to INSUFFICIENT_EVIDENCE; `ABSENT`/`FAMILY` records never written to KG3.
- **Evidence**: ~21% of spans are non-assertive (14.2% negated) — a high-frequency path; applying present-tense guideline rules to a negated/hypothetical mention is a category error better caught by rule than delegated to the LLM.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§6)
- **Date**: 2026-08-08

### Strict vs. loose citation verification branching on `citation_type`
- **Decision**: `verbatim`/`paraphrase_with_recovered_excerpt` checked by strict LCS containment ≥0.8; `paraphrase` gets loose rule-existence-only verification; `pointer_unverifiable` not offered as citable.
- **Evidence**: Reuses the exact metric `clean_local_triplets.py` used to originally assign `citation_type`, so ground truth and audit share methodology.
- **Source**: docs/MoLLM_Stage3_Retrieval_Design.md (§7)
- **Date**: 2026-08-08

### Node consolidation limited to matching `@type`, not blind merge-by-SNOMED-code
- **Decision**: Duplicate nodes sharing a SNOMED code only auto-merge when they also agree on `@type`; type-mismatched groups (41 found) left unmerged, tagged for review.
- **Evidence**: An initial merge-by-code-only version incorrectly combined "Systolic BP ≥160" and "Systolic BP <90" into one node sharing a generic parent code despite opposite clinical meaning.
- **Source**: docs/Guideline_Triplets_KG_Review.md (§6)
- **Date**: 2026-08-07

### Citation field split into verifiable-quote vs. curator-note classification
- **Decision**: Every citation reclassified into `citation_type` (`verbatim`/`paraphrase_with_recovered_excerpt`/`paraphrase`/`pointer_unverifiable`) rather than treating all as equally checkable.
- **Evidence**: LCS containment check on ≥15-char citations (1,114 total): only 47% strong verbatim matches, 27% low-containment/non-quote meta-references.
- **Source**: docs/Guideline_Triplets_KG_Review.md (§3, §6)
- **Date**: 2026-08-07

### Predicate/type vocabulary canonicalized conservatively, ambiguous pairs left alone
- **Decision**: 38 predicates and 26 types hand-mapped; direction-ambiguous pairs (`RECOMMENDS` vs `RECOMMENDED_FOR`) deliberately left unmerged.
- **Evidence**: 30+ distinct predicates with schema drift (e.g. `quantitative_threshold` used as both `@type` on 26 nodes and a predicate name on 26 edges).
- **Source**: docs/Guideline_Triplets_KG_Review.md (§3, §6)
- **Date**: 2026-08-07

### Three mechanical normalization passes required before ingesting `rules-llm`
- **Decision**: Normalize list-vs-dict schema, normalize `snomed`/`icd10` field values to plain strings, validate every rule's target UUID against the file's own node IDs before ingestion.
- **Evidence**: 14.4% of rules (102/706) point to a nonexistent target UUID; 154 nodes (16%) store `snomed` as a dict instead of a string.
- **Source**: docs/Rules_LLM_Triplets_Review.md (§3, §4)
- **Date**: 2026-08-07

### KG3 write-back: store triplets first, compute embeddings as a derived batch layer
- **Decision**: Store high-confidence extractions as real graph triples first; compute KG embeddings as a periodic batch job over accumulated triples, not the storage format itself.
- **Evidence**: A dense embedding vector can't be shown to a reviewer as evidence and can't be traced to a source sentence — going straight to embeddings would lose the "white-box, auditable" pitch.
- **Source**: docs/Proposal_Alignment_Review.md (§6)
- **Date**: 2026-08-07

### Guideline-derived KG (Objective 2) never wired into the production tier gate — acknowledged as a real gap
- **Decision**: Not implemented as a gap in this session's proposal-alignment review, later built and wired in via a name+type match (not SNOMED code, per explicit user redirect).
- **Evidence**: Confirmed via direct code search that `scripts/init_memgraph_guidelines.py` was never actually run against production before this session; the production tier gate did not consume the guideline KG.
- **Source**: docs/2026-08-20_Session_Results_And_Status.md (§9)
- **Date**: 2026-08-07 (gap identified) → 2026-08-20 (closed)

---

## 8. Notes on This Log's Own Methodology

Mined by three parallel research agents, each reading a disjoint subset
of the 32 source docs in full (not grepped/skimmed) and extracting
decision+evidence pairs verbatim. Synthesized and deduplicated here.
Entries that appeared to conflict were kept and flagged (§7) rather than
silently resolved in either direction. Anything mined that turned out to
be pure narration or a code description with no real "chose X over Y"
content was excluded by the mining agents per their own instructions.

This log does not claim to be perfectly exhaustive — a handful of very
small or purely internal refactoring decisions may not appear in the
source docs at all — but it covers every decision that was written up
with its own supporting evidence anywhere in this project's documentation.
