# Stage 2 — Compound-Span and Qualifier Gaps

**Date:** 2026-08-10
**Status:** Compound-span splitting (binary) is implemented and verified on the small gold note. This document consolidates what the median and large gold notes then revealed, and scopes the three fixes tracked as follow-on work.

## Run that produced this

```
python3 scripts/test_pipeline_e2e.py --note-ids 19442119-DS-15   # median, 269 gold
python3 scripts/score_gold_recall.py --note-ids 19442119-DS-15

python3 scripts/test_pipeline_e2e.py --note-ids 14490470-DS-11   # large, 431 gold
python3 scripts/score_gold_recall.py --note-ids 14490470-DS-11
```

Both runs used the current (whole-phrase-guarded, binary) `find_compound_split()`.

---

## Background: the compound-split fix and its regression

`find_compound_split()` (`src/normalization.py`) exists because a one-entity-one-concept design cannot satisfy gold annotations like "gunshot wound to abdomen", where gold links "gunshot wound" (56768003) and "abdomen" (818983003) as two separate concepts. The first implementation had no guard on whether the whole phrase already had a confident answer, and wrongly decomposed already-correct merged entities ("aspiration pneumonia", "retroperitoneal hematoma") into generic parts — linked_recall on the small note dropped 15/144 → 13/144. Adding a whole-phrase Tier 1/2 guard (return `None` immediately if the full text already resolves on its own) fixed the regression and pushed linked_recall past the original baseline: 18/144 (12.50%), with official IoU improving 0.1136/0.1042 → 0.1308/0.1250.

## Results at three note sizes

| Note | Gold | Pred | Span R | Linked R | Compound | Macro IoU | Weighted IoU |
|---|---|---|---|---|---|---|---|
| 17751158-DS-19 (small) | 144 | 90 | — | 12.50% | 4 | 0.1308 | 0.1250 |
| 19442119-DS-15 (median) | 269 | 90 | 23.79% | 12.64% | **0** | 0.0873 | 0.1094 |
| 14490470-DS-11 (large) | 431 | 93 | 22.74% | 7.42% | **20** | 0.0671 | 0.0739 |

Linked recall is consistent between the small and median notes (~12.5%), a positive sign the fix generalizes. The large note is a different regime: COMPOUND jumps from single digits to 20, and linked recall drops to the lowest of the three. Two genuinely new failure patterns emerged, plus one confirmed-safe finding.

---

## Gap 1 — Compound-span splitting is binary; gold sometimes wants 3+ concepts

**Evidence.** The large note is dense with laterality + device + action procedures, each coded by gold as three separate concepts: `right EVD placement` → 'right' + 'EVD' + 'placement' (×3 occurrences across the note, case variants included), `Right EVD removal`, `Right EVD replacement`, `right brachial, basilic thrombus` (×2) → 'right brachial' + 'basilic' + 'thrombus'. The current splitter only tries a single cut point (2 parts), so all of these stay merged and land in the COMPOUND=20 bucket untouched — this is the dominant driver of that count, not a new bug, just the ceiling of what a binary splitter can ever fix.

**Why it matters.** This is the same structural limitation as the earlier `Pneumonia`/`Gunshot wound` case, just far more common in procedure-heavy notes. A corpus-wide run will hit this disproportionately hard on surgical/trauma notes.

**Known fix, scoped.** Generalize the splitter to an exhaustive partition search over 2–4 contiguous token groups, preferring the fewest parts that fully resolve (so a genuinely 2-concept phrase is never forced into 3). See "Fixes Applied" below.

---

## Gap 2 — Qualifier-prefix under-extraction (the mirror image of compound spans)

**Evidence, median note.** Gold repeatedly annotates a *longer, qualified* phrase as one specific concept, while Stage 2a's predicted span is the *shorter, unqualified* sub-phrase, landing on a more generic — and wrong — concept:

| Gold span (concept) | Predicted span (concept) |
|---|---|
| congestive heart failure (42343007) | heart failure (84114007) |
| acute pulmonary edema (40541001) | pulmonary edema (19242006) |
| pulmonary hypertension (70995007) | hypertension (different, systemic-HTN concept) |
| valvular heart disease (368009) | heart disease |
| occlusion of the LAD (840608004) | LAD (anatomy only) |
| stenosis of the midcircumflex (1255257002) | midcircumflex (Unmapped — not a standalone term) |
| Mid LAD (91748002) | LAD |
| acute on chronic systolic heart failure (443253003) | heart failure |

**Why it matters.** `pulmonary hypertension` is not a narrower `hypertension` — it is a different disease in a different clinical category. Every row above is a case where the missing qualifier isn't decoration, it changes which concept is correct. COMPOUND=0 on this note confirms the splitter is behaving conservatively as designed (it never over-merges), but conservatism here just means it declines to help with a problem it wasn't built for — this is under-extraction, not over-merging, and needs the opposite mechanism.

**Known fix, scoped.** A span-growing detector: try absorbing 1–5 adjacent words from the sentence-bounded left/right context, and prefer the grown phrase whenever it resolves at Tier 1/2 (same confidence bar the splitter already requires, so this doesn't trade in shaky semantic guesses either). See "Fixes Applied" below.

---

## Gap 3 — Qualifier words (Left/Right/laterality) don't fit the 6-label domain schema

**Evidence, large note.** `Left craniectomy` — previously `Unmapped` entirely, even in the very first pre-fix run of this note — was split into `Left` (still Unmapped) + `craniectomy` (now Tier 1 Exact, correct). Net: a total failure became a partial success, not a regression. But `Left` staying Unmapped is a real, traceable bug: split-detection's Tier 1/2 lookup (`_lookup_tier12`) is domain-unrestricted, so it can find *some* match for "Left" (a genuine SNOMED qualifier-value concept). Stage 2b's real `normalize_entity()` then re-runs the lookup restricted to the parent's label domain (`Procedure` → domain `Procedure`), which excludes whatever domain that qualifier concept actually lives in — so the split-detector's own confirmed match is thrown away by the very next stage.

**Why a naive fix is wrong.** The obvious fix — restrict split-detection to the same domain Stage 2b will use — was checked and rejected: it would make "Left" fail *during detection too* (same restrictive domain excludes it there as well), so the split would never fire at all, and the `craniectomy` win disappears along with it. The 6-label schema (Condition/Symptom/Medication/Procedure/Anatomy/Lab Test) simply has no slot for standalone qualifier/laterality words — they're modifiers, not clinical entities in their own right, and this is a structural gap, not a quick patch.

**Known fix, scoped.** Thread the domain the split-detector *already confirmed a match in* through to Stage 2b's re-lookup as an explicit override, instead of re-deriving a domain from the label. Self-consistent by construction: no need to know or guess the exact OMOP domain string ("Qualifier Value", "Meas Value", or otherwise). See "Fixes Applied" below.

---

## Fixes Applied (2026-08-10)

All three gaps are implemented. No EC2/real-DB access exists in the environment these were written in, so verification here is: `py_compile` on every touched file, an `ast`-level check that `entity_extraction.py`'s INSERT column list / `?` placeholders / row-tuple / `ON CONFLICT DO UPDATE SET` list all agree in count, and 25 logic tests run against the real `find_compound_split()` / `find_span_growth()` / `normalize_entity()` / `split_compound_entities()` / `grow_entity_spans()` functions with DuckDB, GLiNER, SapBERT and spaCy stubbed out (fake `conn.sql()` returning canned rows). **The real-data test on EC2 (command below) is still the one that matters — this only rules out mechanical/logic errors.**

**Gap 1 — n-way splitting.** `find_compound_split()` (`src/normalization.py`) now does an exhaustive partition search over 2–4 contiguous word groups instead of a single cut point, trying 2-way partitions before 3-way before 4-way so it always prefers the fewest concepts the evidence supports. `_partition_token_ranges()` generates the candidate partitions. Verified: the previously-working 2-way cases (`gunshot wound to abdomen`, whole-phrase guard on `aspiration pneumonia`) reproduce byte-identical results; a synthetic `right EVD placement` case that has no valid 2-way split now correctly finds the 3-way `right`/`EVD`/`placement` split; a case where BOTH a 2-way and a 3-way split are available correctly prefers the 2-way one.

**Gap 2 — qualifier-prefix span growing.** New `find_span_growth()` (`src/normalization.py`) tries absorbing 1–5 adjacent words from the sentence-bounded left or right context (single-sided only — every real case measured was left-side) and returns the first widened phrase that resolves at Tier 1/2. New `grow_entity_spans()` (`src/clinical_pipeline.py`) wires this into the pipeline, bounding context to the entity's own PyRuSH sentence via `find_sentence()` so growth can never absorb a word from an unrelated statement. Runs before compound-splitting in `run_pipeline()`; a grown entity handed into the splitter next is provably a no-op there (the splitter's own whole-phrase guard re-checks the now-grown text and always passes). New `grown_from`/`superseded_by_growth` columns on `extracted_entities`, mirroring `compound_split_of`/`superseded_by_split`. `scripts/score_gold_recall.py` excludes `superseded_by_growth=TRUE` rows from scoring, same as it already did for splits. Verified: single-word (`heart failure` → `congestive heart failure`) and multi-word (`LAD` → `occlusion of the LAD`) absorption both resolve correctly; a case with nothing resolvable returns `None` rather than forcing a bad widen.

**Gap 3 — qualifier-word/laterality domain gap.** `normalize_entity()` (`src/normalization.py`) gained a `domain_override` parameter that, when given, replaces the `GLINER_LABEL_TO_DOMAIN`-derived domain restriction entirely. `src/clinical_pipeline.py`'s split and growth builders (`_build_split_or_grown_entity()`, shared by both) now populate `domain_override` directly from the domain the detector's own unrestricted Tier 1/2 lookup already confirmed a match in — so Stage 2b's re-normalization is guaranteed to search the same domain the split/growth step found the concept in, without ever needing to know or guess the actual OMOP `domain_id` string (`"Qualifier Value"` vs `"Meas Value"` vs anything else). This directly fixes the `Left craniectomy` → `Left` Unmapped case: `Left`'s split-detection match and Stage 2b's re-lookup now search the identical domain by construction. `process_and_normalize_entities()`'s cache key was extended to include `domain_override`, since two entities can share (text, label) but carry different overrides. Verified with a synthetic domain-restriction test asserting the SQL-bound domain parameter is exactly the override, not the label-derived default.

### Confirmed on EC2 (2026-08-10)

```
python3 scripts/test_pipeline_e2e.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
python3 scripts/score_gold_recall.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
```

| Note | Linked R (before) | Linked R (after) | Compound (before) | Compound (after) |
|---|---|---|---|---|
| small | 12.50% | 13.89% | 4 | 3 |
| median | 12.64% | 13.01% | 0 | 0 |
| large | 7.42% | 8.35% | 20 | 16 |
| **combined** | — | **10.78%** | — | 19 |

Linked recall improved on all three notes — no regression on any of them, including the small note's already-verified baseline. Official IoU combined: macro 0.0939 / in-scope 0.0998, weighted 0.1027 (roughly in line with a straight average of the three notes' individually-measured IoUs, so nothing anomalous there).

**Gap 3 confirmed working directly.** `Left`/`Right` now resolve instead of `Unmapped` — e.g. `gold 'Left' (64237003) -> predicted 'Left' = Left (7771000), tier 1 (Exact)`. Still shows up in "wrong-concept" because SNOMED has *multiple* distinct Left/Right qualifier concepts and the tier-1 lookup's `ORDER BY concept_id ASC LIMIT 1` tie-break doesn't necessarily pick the one gold wants — but this is now a solvable disambiguation problem instead of a structurally-impossible one, which is exactly what the fix was for.

**Two refinements surfaced, not yet acted on:**

1. **Gap 1 over-atomizes when a compound's coarser form is only Tier-3-resolvable.** `Right chest tube placement` — one of the original 20 large-note compound cases — did get split, but into 4 parts (`Right`+`chest`+`tube`+`placement`) instead of the 2 gold actually wants (`Right` + `chest tube placement`, gold concept 264957007). The 3 generic single-word parts each resolve at Tier 1/2 individually, so the fewest-parts-first search accepts that partition — but it never considers that leaving `chest tube placement` intact and letting it fall through to Tier 3 might land closer to gold's actual compound concept. This is why the `right EVD placement`-style 3-way cases (the ones gap 1 was originally built for) still don't split at all: `EVD` alone has no Tier 1/2 match, so no partition containing it as its own part ever succeeds — gap 1 fixed a case it wasn't aimed at and didn't fix the case that motivated it.
2. **Left/Right qualifier-concept ambiguity, now visible because gap 3 unblocked it.** Multiple distinct SNOMED qualifier-value concepts all mean "left" or "right" (context-dependent — laterality vs directional vs positional), and the current tier-1 tie-break has no way to pick the gold-aligned one. Previously invisible behind `Unmapped`.

Neither is a regression — combined linked_recall is up and nothing broke — but both are real, now-measured limits of the current design worth a decision on before going further.

## Refinements Applied (2026-08-10, same day)

**Gap 1 refinement — deferred parts.** `find_compound_split()` no longer requires every partition part to resolve at Tier 1/2. It now accepts a partition with AT MOST ONE unresolved ("deferred") part, provided at least one other part does resolve (an unanchored, all-guesswork partition is rejected — no better than not splitting). Within a part count, a fully-resolving (zero-deferred) partition still always wins over one with a deferred part; across part counts, fewer parts still always wins. A deferred part (`candidate: None` in the returned dict) is NOT looked up at Tier 3 by the splitter — it is handed to `src/clinical_pipeline.py`'s `_build_split_or_grown_entity()` unchanged, which now keeps the parent's own `entity_label` and sets no `domain_override` for it, so Stage 2b normalizes it exactly like an ordinary never-split entity (its own Tier 1/2 → Tier 3 pipeline, not a guess made by the splitter). This directly fixes the `Right chest tube placement` over-atomization: the 2-way partition `["Right", "chest tube placement"(deferred)]` is now found and accepted at k=1, before the search ever reaches the 4-way all-resolving partition. It also means `right EVD placement` now splits into `right` + `EVD placement`(deferred) rather than not splitting at all (`EVD` alone has no Tier 1/2 match in the real vocabulary, so the original gold-motivating 3-way case still doesn't fully decompose — but the laterality qualifier is at least separated out correctly now, which is what gap 3 needs to be able to do its job).

**Gap 3 refinement — multi-domain ambiguity surfaced.** `_lookup_tier12()` gained an `include_domains` option; when set, it attaches every DISTINCT `domain_id` across ALL Tier 1/2 matches for the text (via new `_tier12_domains()`), not just the single top-ranked row's domain. Previously, `domain_override` was populated from only the top-1 candidate's domain (picked via `ORDER BY concept_id ASC LIMIT 1`), which silently collapsed real cross-domain ambiguity — e.g. "Left" has multiple distinct exact SNOMED concepts in different qualifier categories — into one arbitrary winner, so Stage 2b's restricted re-lookup only ever saw one of them and reported a confident-but-often-wrong HIGH-tier match. `domain_override` is now the FULL domain set, so when more than one applies, Stage 2b's own existing ambiguity machinery (`ambiguous = len(candidates) > 1`) correctly fires and routes to Stage 3/LOW instead of guessing.

**Verification.** Both refinements verified the same way as the original three gaps: `py_compile` clean, and logic tests against the real `find_compound_split()`/`_lookup_tier12()`/`_build_split_or_grown_entity()` functions with dependencies stubbed — including a direct reproduction of the `Right chest tube placement` case confirming it now returns `["Right", "chest tube placement"(deferred)]` instead of 4 parts, and a direct reproduction of the multi-domain case confirming `include_domains=True` returns all domains rather than one. **Not yet run on real EC2 data — that's the next step.**

### To confirm on EC2

```
python3 scripts/test_pipeline_e2e.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
python3 scripts/score_gold_recall.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
```

What to look for against the "Confirmed on EC2" table above: `Right/Left chest tube placement` should now show as 2 entities (`Right`/`Left` + the compound remainder resolving via Tier 3, same as before splitting existed) instead of 4 generic Tier-1/2 parts — check the wrong-concept examples no longer show `predicted 'tube' = Tube ... gold 'chest tube placement'`. `right EVD placement` and its variants should now show `right`/`Right` split off cleanly (Qualifier, correct) with `EVD placement`/`EVD removal`/`EVD replacement` remaining as their own Tier-3-resolved entity, rather than fully merged. `Left`/`Right` wrong-concept examples should either resolve correctly now or shift from a confident wrong HIGH-tier match to an `ambiguous`/LOW one routed toward Stage 3 — worth checking `tier_reasons`/`ambiguity_reason` on a few of those rows directly if the printed table doesn't make it obvious. Combined linked_recall should not regress below 10.78%.

## Gap 1 Refinement Reverted (2026-08-10, later same day)

**Measured regression, not confirmed.** The EC2 run above showed combined linked_recall drop 10.78% → 9.72% (median 13.01% → 11.15%, large 8.35% → 7.42%). Root cause, directly visible in the printed entity table: `heart | Qualifier | Initial | 1 (Exact)`. `heart` has a spurious exact Tier-1 string match against an unrelated SNOMED concept (`Initial`, almost certainly a vocabulary data-quality artifact rather than a real clinical meaning) — and the "allow one deferred part" rule treated that single collision as sufficient evidence to split phrases like `chronic systolic heart failure`, peeling `heart` off and leaving a context-stripped remainder (`systolic heart`, missing `chronic` and `failure`) for Tier 3 to guess at — which then landed on the wrong disease category (`Systolic hypertension` instead of `Chronic systolic heart failure`). This is the direct opposite of what gap 2 (span growing) exists to prevent: less context reaching Tier 3, not more.

**"At least one part resolves" was too weak a bar.** This SNOMED/OMOP dump apparently has enough single-token exact-string collisions that one resolved token isn't reliable evidence a split boundary is real — and there's no real-DB access in this development environment to characterize which collisions are trustworthy versus spurious, so a targeted fix (e.g. an anchor-quality gate) can't be safely designed without another blind EC2 round-trip.

**Reverted, not tuned further.** `find_compound_split()` is back to requiring EVERY partition part resolve at Tier 1/2 — the version already measured net-positive across all three notes (10.78% combined, no regressions). This means the `Right chest tube placement` over-atomization (4 generic parts instead of gold's 2) is back too, as a known, documented, but *unconfirmed-severity* limitation — better than trading a measured win for a measured loss. `right EVD placement` goes back to not splitting at all (`EVD` alone never resolves in the real vocabulary, so no all-resolving partition exists for it under any grouping) — also a known limitation, not a regression, since it never split before gap 1 existed either.

**Gap 3's multi-domain surfacing was kept** (not implicated in the regression — it only affects `domain_override` on parts that already resolve) but did not visibly change the `Left`/`Right` wrong-concept cases in this run; they're still confident HIGH-tier misses, not ambiguous/LOW. Likely explanation: gold's alternate "Left"/"Right" concepts (e.g. `64237003` vs our `7771000`) are reached via a *different exact string* (a synonym, or a distinct primary name) rather than sharing the same domain — i.e. this is a same-domain, different-exact-match precision gap, not the cross-domain ambiguity gap 3 was built to surface. Confirmed harmless, not confirmed useful; left in place since it's still the theoretically-correct behavior for the case it *was* built for.

**Verified:** `py_compile` clean on both files; the exact regression scenario (a single spurious token match) reproduced against a synthetic vocab and confirmed to no longer trigger a split; the previously-verified 2-way/n-way/whole-phrase-guard cases re-confirmed unaffected by the revert. **Not yet re-run on real EC2 data — that's the next step**, to confirm linked_recall returns to (or above) the 10.78% baseline.

### To re-confirm on EC2

```
python3 scripts/test_pipeline_e2e.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
python3 scripts/score_gold_recall.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
```

Expect combined linked_recall back at ~10.78% (small 13.89%, median ~13.01%, large ~8.35%), and no more `heart | Qualifier | Initial`-style spurious splits in the printed table.

### Confirmed on EC2 (2026-08-10, revert verified)

| Note | Linked R (regression) | Linked R (reverted) |
|---|---|---|
| small | 13.89% | 13.89% |
| median | 11.15% | 13.38% |
| large | 7.42% | 8.82% |
| **combined** | **9.72%** | **11.14%** |

Revert confirmed clean: no regression, and combined linked_recall (11.14%) actually landed above the original pre-refinement baseline (10.78%) — the `Right chest tube placement`-style 4-way over-atomization is back (a known, documented, lower-severity limitation) and `EVD`-containing phrases stay merged (never split, before or after gap 1 existed), but nothing shows the `heart`-style spurious single-token collision anymore. Official IoU: macro 0.0919/0.0982 (in-scope), weighted 0.1066.

Status: gap 1 and gap 3 are closed for this session at their currently-verified, safe designs. Remaining known limitations (documented, not fixed): `Right chest tube placement`-style over-atomization when a compound's coarser form is Tier-3-only; `EVD`-containing 3-way gold decompositions that never split at all; same-domain Left/Right concept-selection precision (gap 3's multi-domain fix doesn't reach these since the ambiguity isn't cross-domain). All would need real vocabulary inspection (not available in this environment) to fix safely rather than by further blind trial-and-error.

## Gap 4 — Lab Value Suffix Stripping (2026-08-10, new)

**Evidence.** `WBC-13.0`, `PTT-29.0`, `Glucose-117` (small note), `ALT-736`, `AST-956` (median note) all showed `0 (Failed)`/`Unmapped`. MIMIC discharge summaries report labs in compact flowsheet notation — test name and numeric result glued together by a hyphen with no space. GLiNER extracts the whole hyphenated token as one Lab Test span (no whitespace boundary to stop at), so normalization tries to match `"wbc-13.0"` against SNOMED/LOINC concept names, which obviously never include the numeric result, and fails at every tier — Tier 3's semantic embedding on the polluted string doesn't clear the similarity floor either.

**Already known on the assertion side.** `src/assertion.py`'s `is_structured_result()` names the exact same pattern (`GLUCOSE-109`, `UREA N-25`, `CREAT-0.3`) as the reason lab-panel lines must not be treated as negatable narrative prose. Nobody had closed the loop on normalization actually being able to link the test name.

**Fix.** New `strip_lab_value_suffix()` in `src/normalization.py`: regex-strips a trailing `-NUMBER` suffix (`WBC-13.0` → `WBC`, `UREA N-25` → `UREA N`), returning `None` if the pattern doesn't match or the remaining name would be too short to be meaningful. Wired into `process_and_normalize_entities()` as a third, last-resort retry tier — after both the expanded-text and original-text forms already failed — gated to `gliner_label == "Lab Test"` only, so no other entity type is ever affected by a coincidental hyphen-number pattern. `normalized_from` records `value_stripped_from_expanded`/`_original` so this is auditable, matching the existing fallback-tracking convention. The numeric value itself is discarded, not carried forward — flagged in the docstring as a real, separate gap (no lab-value-fact table exists yet to hand it to), not silently dropped without a trace.

**Verified:** `py_compile` clean; 15 unit tests against real MIMIC-observed strings (`WBC-13.0`, `UREA N-25`, `CREAT-0.3`, etc., all correctly stripped; `LD(LDH)`, `CK(CPK)`, and non-matching text correctly left untouched); a full integration test through `process_and_normalize_entities()` confirming `WBC-13.0` now resolves to the correct concept via the new fallback tier, and confirming a non-Lab-Test entity with the same hyphen-number shape does NOT get the fallback. **Not yet run on real EC2 data.**

### To confirm on EC2

```
python3 scripts/test_pipeline_e2e.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
python3 scripts/score_gold_recall.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
```

Check that `WBC-13.0`, `PTT-29.0`, `Glucose-117`, `ALT-736`, `AST-956` no longer show `Unmapped`/`0 (Failed)` in the printed table, and that linked_recall doesn't regress below 11.14%.

### Confirmed on EC2 (2026-08-10)

All five now resolve via Tier 3: `WBC-13.0` → Leucocyte count, `PTT-29.0` → Prothrombin time, `Glucose-117` → Glucose measurement, `ALT-736` → Alanine transaminase activity, `AST-956` → Aspartate aminotransferase. (`ASA-NEG Ethanol-67...` stays `Unmapped` — a compound multi-result string, not the single `Name-Value` pattern this fix targets; expected, not a bug.)

Combined linked_recall: 11.26% (up from 11.14%) — small 13.89% (flat), median 13.75% (up from 13.38%), large 8.82% (flat). Official IoU: macro 0.0916/0.0978 (in-scope), weighted 0.1070. No regressions. Closed.
