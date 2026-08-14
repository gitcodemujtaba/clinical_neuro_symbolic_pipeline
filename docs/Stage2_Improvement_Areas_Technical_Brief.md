# Stage 2 Improvement Areas — Technical Brief for Second Opinion

Prepared 2026-08-11, based on two real runs against the DrivenData SNOMED CT
Entity Linking Challenge's 25-note stratified test-split sample:
`scripts/score_gold_recall.py` (span/concept accuracy against gold) and
`evaluation/stage_calibration.py` (per-stage confidence calibration). All
numbers below are measured, not estimated. Pipeline scope: **Stage 1 →
2a → 2b only** — Stage 3 (MoLLM) is built but not wired into this run, so
none of the errors below are currently being caught by anything downstream.

Purpose of this document: lay out where the pipeline is currently weak, with
enough technical detail (exact numbers, code locations, root-cause
hypotheses, and candidate fixes with tradeoffs) that a second reviewer can
evaluate which fix is highest-leverage, rather than just "there's a
problem." Root-cause explanations below are flagged explicitly as
**hypothesis** vs. **confirmed** — several plausible causes have not yet
been isolated from each other with a targeted experiment.

---

## Headline number

25-note official-test-split run, official DrivenData character-IoU metric
(faithful reimplementation of the benchmark's own `scoring.py`):

| | Macro char-IoU | Support-weighted char-IoU |
|---|---|---|
| This pipeline (in-scope, excl. Medication) | **0.0787** | **0.0947** |
| Weakest DrivenData leaderboard baseline (FAISS+Qwen3) | 0.2321 | 0.2160 |
| DrivenData leaderboard #1 | 0.4657 | 0.6165 |

Combined span/concept accuracy on the same 25 notes: **8,655 gold
annotations**, 2,460 predictions, **span recall 20.83%**, **linked (concept)
recall 10.03%**, 73 compound-span cases, 16 uncrosswalked medication misses.

---

## Area 1 — Span recall: Stage 2a misses ~79% of gold-annotated spans

**Evidence.** Of 8,655 gold spans across 25 notes, only 20.83% have any
overlapping GLiNER-BioMed prediction at all. This is measured before
normalization ever runs — it is purely an extraction-coverage problem, and
it caps every downstream metric: a concept can't be scored correct if the
span was never proposed.

Representative missed spans (real, from note `10043750-DS-6`): `No Known
Allergies`, `Adverse Drug Reactions`, `clinic`, `cecal cancer`,
`investigations`, `metastatic disease`, `right colectomy`, `procedure`,
`Lives alone`, `support`, `Gen`, `OP clear`, `MMM`, `Resp`, `CTAB`.

**Pattern in the misses.** Two distinct shapes: (a) short exam-shorthand /
review-of-systems tokens (`Gen`, `MMM`, `CTAB`, `Resp`, `OP clear`) that
read as fragments outside their tabular/list context, and (b) section-header
or narrative phrases (`No Known Allergies`, `Lives alone`) that may fall
outside whatever fixed zero-shot label set is currently passed to GLiNER.

**Root-cause hypotheses (not yet isolated from each other):**
1. **[hypothesis]** The GLiNER zero-shot label schema passed at inference
   time (`src/entity_extraction.py`) under-covers the DrivenData annotation
   guideline's actual scope (Procedures, Body Structures, Clinical
   Findings are broad categories; the concrete label list used at
   inference may be narrower in practice).
2. **[hypothesis]** Structured/tabular content (vitals lines, exam
   shorthand) doesn't read as natural-language entities to a span model —
   directly analogous to the already-solved tabular-negation problem in
   `src/assertion.py` (`is_structured_result()`), but for *extraction*
   rather than *assertion status*. No equivalent carve-out exists for
   extraction yet.
3. **[hypothesis]** Possible silent truncation on long notes — GLiNER has a
   token-length ceiling and no explicit `max_length` gate has been
   confirmed set (flagged as an open item in `Implementation_Checklist.md`;
   not verified against the current model).
4. **[unlikely but unruled-out]** The extraction confidence floor itself
   (`EXTRACTION_THRESHOLD = 0.5`) discarding real spans — see Area 2, this
   interacts with the floor question directly.

**Candidate solutions, for a reviewer to weigh:**
- **(a) Broaden/audit the GLiNER label schema** against the DrivenData
  annotation guideline text directly, closing any gap between what's
  labeled and what's asked for.
- **(b) Add a rule-based extraction channel for structured/tabular
  content** (vitals lines, lab panels, terse exam shorthand), mirroring
  `is_structured_result()`'s tabular detection but feeding entities
  forward instead of just suppressing negation logic.
- **(c) Confirm and fix any silent truncation** on the longest notes
  (2,374–24,858 chars per `Implementation_Checklist.md`); chunk with
  overlap if a hard limit exists.
- **(d) Lower or remove the acceptance floor for extraction** (see Area 2
  — since confidence doesn't reliably predict correctness anyway, a lower
  floor may recover real spans at an acceptable precision cost, provided
  something downstream — ideally Stage 3 — can catch the added noise).

**Priority note.** This is very likely the single highest-leverage fix
available: even a perfect Stage 2b (normalization) cannot recover concept
credit for a span Stage 2a never proposed. Fixing normalization without
fixing recall puts a low ceiling on the total achievable score.

---

## Area 2 — GLiNER extraction confidence is inversely related to correctness

**Evidence.** `evaluation/stage_calibration.py`, n=3,277 extracted spans
(includes below-threshold spans retained down to `SUBTHRESHOLD_FLOOR=0.35`),
graded by whether the span overlaps any gold annotation (any concept —
precision on span reality, not concept correctness):

| Confidence bin | n | Accuracy |
|---|---|---|
| [0.3, 0.4) | 295 | 86.10% |
| [0.4, 0.5) | 522 | 80.84% |
| [0.5, 0.6) | 480 | 80.63% |
| [0.6, 0.7) | 440 | 80.00% |
| [0.7, 0.8) | 449 | 75.50% |
| [0.8, 0.9) | 525 | 65.14% |
| [0.9, 1.0) | 566 | 68.20% |

ECE = 0.2378. At the current threshold (0.5): coverage 75.07%,
precision-if-admitted 73.41%.

**This is not simple miscalibration (systematic over/under-confidence) —
accuracy *falls* as confidence *rises* through most of the range.** The
model is more often wrong when it reports higher confidence. This means
raising `EXTRACTION_THRESHOLD` to be "more selective" would plausibly make
precision *worse*, not better — the opposite of the usual fix for a
low-precision extractor.

**Root-cause hypotheses:**
1. **[hypothesis]** Boundary mismatch inflating apparent errors: GLiNER may
   be confidently correct about *which entity* it found but disagree with
   gold on exact span boundaries (e.g., "R colon" vs. gold's "colon
   cancer" — arguably the same clinical referent, scored as wrong here
   because grading is overlap + exact-concept, not partial credit). Not
   yet separated from genuine hallucinated-span errors.
2. **[hypothesis]** Domain/annotation-convention shift: GLiNER-BioMed's own
   fine-tuning data may define "confident" differently than this specific
   benchmark's annotation guidelines reward.

**Candidate solutions:**
- **(a) Re-grade with partial-credit / IoU-based correctness** instead of
  binary overlap, to test the boundary-mismatch hypothesis directly before
  assuming the confidence signal is truly broken.
- **(b) Post-hoc recalibration** (isotonic regression / Platt scaling)
  fit on this validation slice — cheap, but the relationship isn't cleanly
  monotonic across the whole range, so this may only partially help.
- **(c) Stop using GLiNER's own confidence as an accept/reject gate**;
  use it only to rank candidates, and gate acceptance on a different
  downstream signal (e.g., whether Stage 2b's Tier 1/2 finds anything at
  all for the span).
- **(d) Fine-tune GLiNER-BioMed on this project's own 204-note training
  split** rather than relying on off-the-shelf zero-shot confidence.

---

## Area 3 — Tier 1/2 "exact" matches are only ~57% correct

**Evidence.** `evaluation/stage_calibration.py`:

| Tier | n gradable | Accuracy | Excluded (no gold span / uncrosswalkable) |
|---|---|---|---|
| 1 (Exact concept-name match) | 637 | **56.5%** | 325 / 0 |
| 2 (Exact synonym match) | 482 | **58.7%** | 40 / 0 |

**Why this is surprising.** Tier 1/2 are deterministic string matches — if
the entity text exactly equals a concept's name or a known synonym, one
would expect near-100% correctness by construction. ~57–59% means a
substantial fraction of "exact" matches are landing on the wrong concept.

**Root-cause hypotheses (domain filtering already exists here — see below
— so the earlier documented "no domain filter at Tier 1/2" gap is
resolved and is *not* the explanation):**
1. **[confirmed present, coverage not verified]** `normalize_entity()` in
   `src/normalization.py` *does* apply a domain restriction to Tier 1 and
   Tier 2 queries (`domain_clause` built from `GLINER_LABEL_TO_DOMAIN.get(gliner_label)`),
   contradicting an earlier internal doc claim that domain filtering was
   Tier-3-only. What's unverified: whether `GLINER_LABEL_TO_DOMAIN`
   actually has an entry for every GLiNER label in use — a missing entry
   means `domains` is `None` and the filter silently doesn't apply for
   that label.
2. **[hypothesis]** Granularity mismatch, not domain collision: e.g. gold
   annotation `CKD III` (`433144002`) vs. the pipeline's `CKD` →
   `Chronic kidney disease` (`709044004`) — both are legitimate CKD
   concepts, but gold wants the staged/specific version. A domain filter
   cannot fix this; it needs a qualifier/severity-detection step.
3. **[hypothesis]** Within-domain collisions: two SNOMED concepts in the
   *same* OMOP domain can still share an exact synonym string (domain
   filtering only removes cross-domain collisions like the documented
   `ED` → `Ed District` case, not same-domain ambiguity).
4. **[hypothesis]** Upstream GLiNER mislabeling: if the entity's
   `gliner_label` itself is wrong, the domain restriction constrains to
   the *wrong* domain and can still return a confidently wrong exact
   match, or miss the right one entirely.
5. **[confirmed in code]** Tiebreak among multiple exact-match candidates
   is `ORDER BY concept_id ASC` — an arbitrary, non-semantic tiebreak.
   When Tier 1/2 returns >1 candidate, the result is flagged
   `ambiguous=True`, but nothing currently *uses* that flag to pick more
   carefully — the lowest `concept_id` is still what gets reported as
   the top pick.

**Candidate solutions:**
- **(a) Audit `GLINER_LABEL_TO_DOMAIN` coverage** against every label
  GLiNER actually emits; close any gaps.
- **(b) Route Tier 1/2's already-flagged `ambiguous=True` cases to
  context-aware disambiguation** (local sentence context, or Stage 3)
  instead of defaulting to lowest `concept_id`.
- **(c) Add qualifier/severity detection** (numbers, stage indicators,
  laterality) near the entity span, and prefer the more specific concept
  when one is available — addresses the `CKD` vs. `CKD III` class of
  error directly.
- **(d) Improve GLiNER label accuracy** itself, since a wrong label
  cascades into a wrong domain restriction even when the restriction
  mechanism is working correctly.

---

## Area 4 — Tier 3 (SapBERT semantic fallback) is unreliable even above its floor

**Evidence.** `evaluation/stage_calibration.py`, n=1,738 individual Tier-3
candidates (not just each entity's top-1 pick), graded per-candidate by
SNOMED crosswalk match against gold:

| Similarity bin | n | Accuracy |
|---|---|---|
| [0.4, 0.5) | 28 | 0.00% |
| [0.5, 0.6) | 150 | 1.33% |
| [0.6, 0.7) | 166 | 6.02% |
| [0.7, 0.8) | 504 | 11.51% |
| [0.8, 0.9) | 537 | 23.84% |
| [0.9, 1.0) | 353 | **42.78%** |

ECE = 0.5916. At the current production floor (`TIER3_SIMILARITY_FLOOR =
0.72`): coverage 80.21%, **precision-if-admitted only 24.18%.**

**Why this matters more than it looks.** The 0.72 floor is implemented and
enforced in the current code (`src/normalization.py`, below-floor results
are marked `"0 (Failed)"`) — this is *not* the "no threshold at all" gap an
earlier internal doc described. But the calibration data shows the floor
isn't the real problem: even at 0.9–1.0 similarity, fewer than half of
matches are correct. **No single cutoff on this signal gets precision much
above ~40%** — this needs a better signal, not a stricter cutoff.

**Concrete failure examples (real, from note `10043750-DS-6`):** `HLD` →
"Lateral deviation of tongue on protrusion" (gold wants hyperlipidemia);
`NAD` → "Flavin adenine dinucleotide measurement" (gold wants "no acute
distress"); `HEENT` → "Structure of head and/or neck" (overly generic);
`EOMI` → "Extraocular eye movement" (close but not the exam-finding
concept gold wants). **Pattern: short abbreviations are disproportionately
represented in the worst failures** — less surface context for the
embedding to disambiguate.

**Root-cause hypotheses:**
1. **[hypothesis]** SapBERT's off-the-shelf embedding space, while good at
   coarse biomedical clustering, may lack fine-grained discriminative
   power at the exact granularity OMOP concept names require, particularly
   for short/abbreviated input where there's minimal context to
   disambiguate.
2. **[hypothesis]** Same domain-filter-coverage question as Area 3 applies
   here too (Tier 3 shares the same `domain_clause` construction).
3. **[hypothesis]** No context awareness: Tier 3 embeds only the isolated
   entity string, never the surrounding sentence/section — unlike Stage 3,
   which already has a context-window mechanism (`build_local_context()`
   in `src/entity_extraction.py`) that Tier 3 does not use.

**Candidate solutions:**
- **(a) Raise the floor further** — cheap, but caps recall hard and, per
  the data, still leaves substantial residual error even at the top of
  the range; a partial fix at best.
- **(b) Route short/abbreviated tokens through a dedicated abbreviation
  lookup before Tier 3** — Stage 1 already maintains an abbreviation
  dictionary for expansion; extending or reusing it specifically at
  normalization time for short-token entities may resolve exactly the
  `NAD`/`HEENT`/`HLD`-shaped failures better than embedding search can.
- **(c) Add context-aware re-ranking** of Tier 3's top-K candidates using
  the entity's local sentence/section context (a cross-encoder step, or
  routing to an LLM call) — this overlaps substantially with what Stage 3
  is already designed to do; an argument that **wiring Stage 3 into the
  main pipeline may be a higher-leverage fix than further Stage 2b
  tuning**, since Stage 3 already has the context-window and candidate-
  reasoning machinery this problem calls for.
- **(d) Fine-tune SapBERT (or evaluate an alternative embedding model)
  on this project's own OMOP vocabulary and training split**, rather than
  using it off-the-shelf.

---

## Area 5 — Tier 4 (fuzzy-typo channel): inconclusive, watch not act

**Evidence.** n=32 graded candidates, 0% accuracy across all three
populated confidence bins (26 at [0.6,0.7), 5 at [0.7,0.8), 1 at
[0.8,0.9)). Contrasts with the one earlier documented success case
(`spirnolactone` → `spironolactone`, correctly resolved).

**Assessment.** n=32 is too small to draw a conclusion from — flagging as
a real data point, not a confirmed problem. Worth re-checking once a
larger corpus has run through Stage 2b; not a priority action item yet.

---

## Cross-cutting observation: compound spans (73 cases)

Beyond the four areas above, 73 predictions each overlap ≥2 distinct gold
annotations — Stage 2a merges what gold treats as separate entities (e.g.
`UreaN-13 Creat-1.2` as one span vs. gold's separate `UreaN` and `Creat`;
`hip pain` vs. gold's separate `left hip` and `pain`). This is a
**structural extraction-granularity problem**, not a normalization defect
— no Tier 1–4 fix addresses it. Candidate fix: a compound-span-splitting
post-processor for GLiNER output (lab-panel-shorthand and multi-concept
noun-phrase patterns specifically), separate from the accuracy work above.

---

## Suggested priority order for review

1. **Span recall (Area 1)** — hard ceiling on every other metric; highest
   expected leverage per unit of engineering effort.
2. **Tier 3 reliability (Area 4)** — largest single contributor to wrong
   *concepts* among spans that ARE found; the abbreviation-lookup and
   Stage-3-integration candidate solutions both look more promising than
   further threshold tuning.
3. **Tier 1/2 domain-mapping audit (Area 3)** — comparatively cheap to
   investigate (`GLINER_LABEL_TO_DOMAIN` coverage check) relative to
   potential accuracy recovery.
4. **Extraction confidence signal (Area 2)** — worth the partial-credit
   re-grading experiment before committing to a recalibration approach,
   since the root cause (boundary mismatch vs. genuine miscalibration) is
   still ambiguous.
5. **Compound-span splitting** — structural, independent of the above,
   moderate effort.

---

## Addendum, 2026-08-11 — Second-opinion review: points of agreement and disagreement

A second review of this brief proposed a ranked fix plan across all four
areas. Recorded here is where that plan holds up against the codebase and
where it needed correcting, so the record shows an independently
cross-checked recommendation rather than an unexamined one.

### Where we agree

- **Area 1 — rule-based extraction for structured/tabular content is the
  right direction**, and cheap: `src/assertion.py`'s `is_structured_result()`
  already has the regex machinery (`_LAB_NAME_VALUE`, `_LAB_PANEL_TOKEN`)
  for NAME-VALUE lab/vitals patterns, currently used only to suppress
  negation — extending it to emit entities is close to free.
- **Area 1 — silent truncation is real, not speculative.** Confirmed
  directly in code: `model.predict_entities(expanded_text, CLINICAL_LABELS,
  threshold=floor, flat_ner=FLAT_NER)` is a single call over the whole note
  with no `max_length` override. Given notes run up to 24,858 characters,
  this is very likely capping recall on long notes far more severely than
  "missing the back half" — priority-zero to quantify (log actual token
  count vs. the model's default window on a long note).
- **Areas 3/4 — wiring Stage 3 in for context-aware disambiguation is the
  right direction**, and is architecturally available now: Stage 3's
  resolution mode triggers on `confidence_tier_in == "LOW" and
  len(candidates) > 1` and reasons from the candidate list plus local
  context — it does **not** require guideline evidence, so the 8.53%
  guideline-coverage gap (Stage3_Open_Issues.md Issue 1) does not block
  this specific fix. That's a stronger case for the fix than originally
  stated.
- **Area 2 — testing whether this is a measurement artifact before
  touching the model is the right sequencing.** Cheap diagnostic before
  expensive fix is the correct order of operations regardless of which
  specific re-grade turns out to be informative.

### Where we disagree or refined the proposal

- **Area 1 — "skip (d), lowering the floor" is not supported by our own
  calibration data.** The stated reasoning (lower threshold floods the
  pipeline with noise) assumes normal monotonic calibration. Ours is
  inverted: the currently-discarded 0.3–0.4 confidence bin has **86.10%**
  accuracy, higher than the currently-kept 0.8–0.9 bin's 65.14%. The
  threshold is discarding some of the *most* reliable spans. Recommend a
  cheap experiment (re-score with the floor at 0.35, no new extraction
  needed — this data already exists under `below_threshold=TRUE`) rather
  than skipping outright.
- **Area 1 — "(b) rule-based extraction" is actually three separate fixes
  bundled together, not one:** (i) NAME-VALUE lab/vitals patterns
  (regex-friendly, extends existing code), (ii) bare exam-shorthand tokens
  like `Gen`/`MMM`/`CTAB` (no numeric delimiter — a finite-vocabulary
  gazetteer problem, not regex), and (iii) `CLINICAL_LABELS` itself is only
  six labels (`Condition, Symptom, Medication, Procedure, Anatomy, Lab
  Test`) with nothing covering allergy status, social history, or
  administrative findings — several missed spans (`No Known Allergies`,
  `Lives alone`) are a label-schema gap, not a tabular-parsing gap.
- **Areas 3/4 — wiring in Stage 3 is not a free accuracy win.** The
  citation-verification guard fires regardless of whether citation is
  required (Issue 2's finding: fabrication rate unchanged at
  `require_citation=False`), so a real fraction of newly-routed decisions
  will land in `HITL_REQUIRED` rather than resolve automatically — trading
  automatic coverage for precision, not adding pure accuracy. This is
  exactly what `evaluation/ablations.py`'s error-catching matrix and
  citation-guard-value functions were built to quantify; recommend running
  them on the routed sample before reporting a number from this fix.
- **Areas 3/4 — "(b) implement the abbreviation dictionary" should first
  check the cheaper fix locus.** Stage 1 already expands abbreviations
  *before* GLiNER runs. If `HLD`/`NAD`/`HEENT`/`EOMI` reached Tier 3 still
  abbreviated, that's evidence they're simply missing from the *existing*
  Stage 1 dictionary — curation (add entries) is cheaper than new lookup
  infrastructure at Stage 2b, and should be ruled out first.
- **Areas 3/4 — the routing rule needs one verification before it can be
  assumed to work.** Tier 1/2 already sets `ambiguous=True` with its own
  reason (`multiple_exact_concept_name_matches` /
  `multiple_exact_synonym_matches`) — that signal exists today. Unconfirmed:
  whether `confidence_tier_in` (what `build_prompt()` actually gates
  routing on) is computed FROM that Tier 1/2 signal, or only from Tier 3/4's
  own ambiguity reasons. If the latter, "wire in Stage 3" alone will not
  route these cases — the tier-assignment logic needs a change first. The
  "short/abbreviated token" trigger is confirmed new logic either way.
- **Area 2 — the proposed diagnostic tests the wrong hypothesis against the
  wrong example.** `grade_stage2a()` already grades by ANY character
  overlap, not exact-boundary match — it's already the leniently-graded
  case, so relaxing it further to IoU/partial-credit is unlikely to move
  the curve much. The cited example (`"R colon"` vs. gold `"colon
  cancer"`) is not from Area 2's extraction-confidence calibration at all —
  it's a Tier 2 concept-linking failure from `score_gold_recall.py`'s
  wrong-concept table (the span WAS found; Tier 2 linked it to the wrong
  SNOMED concept). Corrected version of this diagnostic: (i) for Area 2
  specifically, test whether confidence correlates better with
  *exact*-boundary match (tightening, not loosening); (ii) for the Area 3/4
  wrong-concept examples, use SNOMED hierarchy distance between predicted
  and gold concept as the partial-credit lens, since concept_id matching is
  categorical, not spatial — character IoU doesn't apply to it at all.
- **Area 2 — temper the "curve will likely normalize significantly"
  prediction.** Several already-observed wrong-concept examples are not
  boundary disagreements by any reading — `NAD` → "Flavin adenine
  dinucleotide measurement" is a category-level hallucination, not a
  near-miss of "no acute distress"; `S2`/`Abd` → `Unmapped` is a
  compound-span structural failure. Expect a hierarchy-distance re-grade to
  recover some credit (plausibly the `"R colon"`/`"colon cancer"` and
  `hip pain`/"Pain of hip region" shapes) but not to explain away the
  catastrophic cases — plan for "the number improves, and the remaining gap
  is the real signal," not "the problem was measurement all along."
