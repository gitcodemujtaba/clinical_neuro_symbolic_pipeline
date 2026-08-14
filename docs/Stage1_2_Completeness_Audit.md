# Stage 1 & 2 Completeness Audit — What Stage 3 Needs and Isn't Getting

**Date:** 2026-08-08. **Purpose:** before implementing Stage 3, identify every piece of information that Stages 1–2 either compute and discard, or never compute, that the MoLLM validation gate needs for maximum visibility. Companion to `MoLLM_Stage3_Retrieval_Design.md`.

Findings are ordered by impact on Stage 3's correctness, not by implementation cost. Measurements are against the 272-note / 75,491-annotation gold set.

---

## Severity 1 — Clinical assertion status is never computed. This is the largest gap in the pipeline.

**Measured:** of 75,491 gold-annotated spans, **15,794 (20.9%) sit in a non-assertive context** — measured by scanning the 90 characters preceding each span for standard clinical assertion cues:

| Context | Instances | Share |
|---|---|---|
| Asserted present (no cue found) | 59,697 | 79.1% |
| **Negated** (`no`, `denies`, `without`, `negative for`, `ruled out`) | **10,689** | **14.2%** |
| Historical (`history of`, `h/o`, `s/p`, `status post`) | 3,209 | 4.3% |
| Hypothetical / uncertain (`if`, `concern for`, `suspected`, `monitor for`) | 1,504 | 2.0% |
| Family (`family history`, `mother`, `father`, `maternal`) | 803 | 1.1% |

(Regex proxy, so approximate and categories can overlap — but the order of magnitude is not in doubt, and it matches the published negation rates for clinical narrative.)

**Why this is severity 1:** Stage 3's entire job is to decide whether an extracted clinical fact is valid and whether it contradicts guideline evidence. Right now a note reading `denies chest pain` produces an entity `chest pain` → OMOP `29857009` → and Stage 3 is asked, in complete sincerity, whether the patient's chest pain contradicts the ACS guidelines. There is no field anywhere in the current schema that could tell it otherwise. The local context window from `MoLLM_Stage3_Retrieval_Design.md` §3 helps — the model *may* notice the `denies` — but relying on the LLM to re-derive negation from raw text is precisely the black-box inference this pipeline exists to replace with deterministic symbolic signal.

It also directly corrupts three downstream claims:

- **KG3 write-back** (Objective 3) would assert negated and family-history findings as patient facts. A patient whose note says `no history of MI` gets an MI observation node. That is a data-integrity failure that then feeds the active-learning loop as a pseudo-label.
- **Evaluation** (Objective 5): the DrivenData gold set annotates the span regardless of assertion, so this doesn't hurt F1 — which makes it *worse*, not better, because the metric is blind to an error class that matters clinically. Worth stating explicitly in the dissertation.
- **The contradiction check itself** is the most sensitive: negation inverts the clinical meaning, so a missed negation doesn't degrade the verdict gracefully, it flips it.

**Recommendation:** add a deterministic assertion-detection pass in Stage 2a, after span extraction, writing four fields per entity: `assertion_status` (`PRESENT` / `ABSENT` / `POSSIBLE` / `CONDITIONAL`), `experiencer` (`PATIENT` / `FAMILY` / `OTHER`), `temporality` (`CURRENT` / `HISTORICAL`), and the matched `assertion_cue` text + offset so a reviewer can see *why*. The standard tool is `medspacy` (the maintained successor to NegEx/ConText, built on spaCy, already a dependency); it is rule-based and deterministic, which fits the white-box claim far better than asking an LLM. Stage 3 then receives assertion status as a stated fact in the prompt, and `ABSENT`/`FAMILY` entities can be routed differently (or excluded from KG3 write-back) by explicit rule rather than model judgment.

---

## Severity 2 — Note section structure is discarded, though it is highly regular and clinically decisive

**Measured:** MIMIC-IV discharge notes are strongly and consistently sectioned. Across 272 notes: `Past Medical History` present in 274 occurrences, `History of Present Illness` 272, `Family History` 271, `Discharge Medications` 269, `Chief Complaint` 269, `Physical Exam` 267, `Medications on Admission` 258, `Brief Hospital Course` 233. Gold annotations distribute across them: `Brief Hospital Course` 2,981, `History of Present Illness` 2,194, `Pertinent Results` 1,927, `Discharge Instructions` 944, `Past Medical History` 748 (first 20k annotations).

**Why it matters to Stage 3:** section membership is a near-free, deterministic prior on exactly the questions assertion detection is trying to answer, and it is often *stronger* than the local cue:

- Anything under `Family History` is about a relative — no negation cue required, the section header settles it.
- `Past Medical History` implies `HISTORICAL`; `History of Present Illness` implies current.
- `Discharge Instructions` is largely hypothetical/anticipatory (`return if you develop fever`) — a major source of the 2.0% hypothetical bucket.
- `Allergies` entities are drug mentions that are emphatically *not* administered medications — currently indistinguishable from `Discharge Medications` entries once extracted.

Stage 1 already runs the full note through scispaCy and has everything needed. Sections are computed nowhere and stored nowhere.

**Recommendation:** add section segmentation to Stage 1, persisted as `(section_name, char_start, char_end)` spans in the note provenance; Stage 2a stamps each entity with its containing `section_name`. Cheap, deterministic, and it feeds both Stage 3's prompt and the assertion logic above.

---

## Severity 3 — Stage 1 silently collapses ambiguous abbreviations, and cannot report that it did

`load_abbreviations_dict()` builds `{abbr.lower(): meaning for abbr, meaning in rows}`. A dict comprehension keeps the **last** value for a duplicated key. The `imantsm/medical_abbreviations` source is explicitly a multi-expansion dictionary — `MS` maps to multiple sclerosis, mitral stenosis, morphine sulfate, mental status; `PT` to prothrombin time, physical therapy, patient; `DC` to discontinue, discharge.

So today: all but one expansion per abbreviation is silently dropped at load time, the surviving one is chosen by row order, and it is then applied unconditionally to every occurrence in the note. The wrong expansion propagates into `expanded_text`, which is what Stage 2a extracts from *and* what Stage 2b normalizes against — so a single bad expansion poisons the entity and its concept mapping, with nothing downstream able to detect it.

This is also why `Provenance_Schema.md`'s specified `ambiguous` flag is not merely unimplemented but currently *uncomputable*: by the time expansion runs, the alternatives no longer exist in memory.

**Recommendation:**
1. Load the dictionary as `{abbr: [all_meanings]}`, preserving every expansion.
2. Set `ambiguous: true` and record `candidate_expansions[]` whenever an abbreviation has >1 known meaning.
3. Add `abbreviation_dict_version` (the missing `Provenance_Schema.md` Stage 1 field).
4. Forward ambiguous expansions to Stage 3 as an explicit uncertainty signal — this is a textbook case for MoLLM disambiguation using local context, and arguably a better demonstration of Objective 2 than concept disambiguation alone.
5. Consider routing ambiguous abbreviations to `confidence_tier_in = LOW` automatically.

---

## Severity 4 — Relation extraction discards the offsets that would make entity linking trivial

`extract_and_store_relations()` calls `model.inference(...)` and assigns the returned entity list to `_entities` — **discarded**. It then stores only `head_entity_text` / `head_entity_label` per relation. GLiNER-relex's relation objects carry character offsets for head and tail spans; these are dropped.

The consequence is self-inflicted: `extraction.py`'s docstring correctly identifies that cross-model span alignment is hard and defers it — but the hard part (fuzzy text matching across two models' tokenizations) is only hard *because the offsets were thrown away*. With head/tail offsets retained, linking a relation endpoint to a canonical `extracted_entities` row is an **offset-overlap test** against spans in the same coordinate system (both models run on `expanded_text`), not a text-similarity heuristic. Overlap-based linking is deterministic, explainable, and roughly ten lines of code.

**Recommendation:** persist `head_start`/`head_end`/`tail_start`/`tail_end` (expanded-text offsets) on `extracted_relations`, map them back through the same `map_offsets_to_original()` "time machine" used for entities, and resolve `head_entity_id`/`tail_entity_id` by maximum character overlap with `extracted_entities` (require ≥50% overlap; otherwise `entity_link_status = unresolved`). This supersedes the text+label matching proposed earlier in the design discussion — offsets are strictly better and already available. Also persist the discarded `_entities` list, or at minimum record how many relex entities failed to link, as a data-quality metric.

---

## Severity 5 — Stage 2b throws away the shape of its own search

Beyond the top-k candidates already addressed in the Stage 3 design, `normalize_entity()` discards:

- **Which tiers were attempted and what they returned.** "Tier 1 and 2 both found nothing, Tier 3 scored 0.71" is a materially different situation from "Tier 1 hit immediately," and only the latter is currently distinguishable. For a LOW-tier record, the tier trace is exactly the provenance Stage 3 is supposed to reason over.
- **`domain_id_queried` and the vocabulary filter applied** — both now materially affect results (the `GLINER_LABEL_TO_DOMAIN` / `VOCAB_BY_LABEL` restrictions added 2026-08-07) and both are named in `Provenance_Schema.md`. If a lookup failed *because* the domain filter excluded the right concept, nothing downstream can tell.
- **`sapbert_pooling_method`, `athena_vocabulary_release`** — specified in `Provenance_Schema.md`, never written. The vocabulary release matters for reproducibility and for the concept-ID-stability check flagged in `Proposal_Alignment_Review.md` §3.8.4.
- **The original (unexpanded) surface form is never tried as a fallback.** Normalization runs on `expanded_text` only. Given severity 3, a wrong expansion means normalization has no path to recover even when the raw abbreviation would have matched cleanly. Recommend attempting both and recording which succeeded (`normalized_from: expanded | original`).

---

## Severity 6 — Miscellaneous discarded signal and unwritten schema fields

- **Sub-threshold GLiNER entities are dropped** (`threshold=0.5`). Spans scoring 0.35–0.5 are exactly the population where a KG-grounded second opinion has the most value; keeping them (flagged `below_threshold`, not promoted) would let you measure Stage 3's recall recovery — a strong Objective 2/5 result that is currently impossible to obtain.
- **`flat_ner` is inconsistent between the two Stage 2a models** — `entity_extraction.py` uses `predict_entities()` defaults (flat), `extraction.py` passes `flat_ner=False` (nested). Nested spans are legitimate in clinical text (`acute kidney injury` contains `kidney`), and the two models disagreeing on this makes their outputs harder to reconcile. Decide deliberately and document.
- **Unwritten `Provenance_Schema.md` fields:** `entity_id`, `gliner_model_version`, `extraction_threshold`, `relation_id`, `head_entity_id`, `tail_entity_id`, `matched`, `abbreviation_dict_version`, `ambiguous`, `domain_id_queried`, `sapbert_pooling_method`, `athena_vocabulary_release`. Model version and threshold matter beyond bookkeeping: the GLiNER checkpoint changed on 2026-08-07, and without a version stamp there is no way to tell which rows came from which model.
- **Sentence boundaries are computed and discarded.** Stage 1 runs `nlp(text)` and uses only `doc` tokens. `doc.sents` is exactly what the sentence-bounded `local_context` window needs — currently it would be re-derived from scratch.
- **No note-level metadata** (character length, token count, truncation-occurred flag). The tokenizer truncation risk flagged in `Implementation_Checklist.md` is still unmeasurable per-note.

---

## Consolidated change list

**Stage 1 (`preprocessing.py`)**
1. Multi-expansion dictionary + `ambiguous` flag + `candidate_expansions[]` *(sev 3)*
2. `abbreviation_dict_version` *(sev 3)*
3. Section segmentation persisted as spans *(sev 2)*
4. Persist sentence boundaries from the existing spaCy doc *(sev 6)*
5. Note-level metadata *(sev 6)*

**Stage 2a entities (`entity_extraction.py`)**
6. Mint `entity_id` *(prerequisite)*
7. Assertion/experiencer/temporality via medspacy + cue text/offset *(sev 1)*
8. Stamp `section_name` and `sentence_id` per entity *(sev 2)*
9. `gliner_model_version`, `extraction_threshold` *(sev 6)*
10. Optionally retain sub-threshold entities, flagged *(sev 6)*
11. Extract and persist `local_context` window *(Stage 3 prerequisite)*

**Stage 2a relations (`extraction.py`)**
12. Persist head/tail offsets; resolve `head_entity_id`/`tail_entity_id` by overlap; `relation_id` *(sev 4)*
13. Persist or count unlinkable relex entities *(sev 4)*

**Stage 2b (`normalization.py`)**
14. `LIMIT 3` + ambiguity gate + candidate list *(Stage 3 prerequisite)*
15. Tier trace *(sev 5)*
16. `domain_id_queried`, `vocab_queried`, `sapbert_pooling_method`, `athena_vocabulary_release` *(sev 5)*
17. Fallback to original surface form; record `normalized_from` *(sev 5)*
18. Deterministic `ORDER BY` tiebreak on all tiers *(pre-existing open bug — non-determinism)*

**Orchestrator (`clinical_pipeline.py`)**
19. Dedup-fanout: compute normalization once per distinct text, write one row per `entity_id` *(prerequisite)*

Items 6, 11, 14, 19 are hard prerequisites for the Stage 3 design as written. Items 1, 7, 8, 12 are the ones that materially change what Stage 3 can *reason about* rather than just what it can *reference*.

---

## Status — all 19 items closed (2026-08-09)

Severity 1–4 landed 2026-08-08; severity 5–6 on 2026-08-09. Two of the late items turned out to matter more than their severity ranking implied, and both were promoted on that basis rather than done as bookkeeping:

**Original-form fallback (item 17).** Normalisation runs on `expanded_text`, so a wrong abbreviation expansion poisoned it with no recovery path — and wrong expansions are real, not hypothetical. Measured on note `10000032-DS-21`: `NAD` expanded to *"nicotinamide adenine dinucleotide"* rather than *"no acute distress"*, and `TTP` to *"Thrombotic Thrombocytopenic Purpura"* rather than *"tenderness to palpation"*, because the deterministic tiebreak picks the alphabetically-first meaning and alphabetical order has no clinical basis. Stage 2b now retries on the raw surface form when the expanded form fails to map, and records `normalized_from` so the bypass is visible rather than inferred. Fires only on failure, so it costs nothing on the common path.

**Sub-threshold retention (item 10).** Extraction now runs at 0.35 and flags everything below the 0.50 gate as `below_threshold`. These are stored and **excluded from every downstream stage** by a single filter in `clinical_pipeline.run_pipeline()`. The point is measurement: *"Stage 3 recovered N entities the extractor nearly discarded"* is one of Objective 2's most direct claims and it is unevidenceable if the near-misses were thrown away at extraction time. It also gives an empirical basis for the 0.50 threshold, which until now was an unexamined default.

Also closed: `athena_vocabulary_release` (content-signature fallback when no `vocabulary` table exists — a load timestamp would say when the file was read, not what was in it), `matched`, `tier_trace`, `domain_id_queried`/`vocab_queried`/`sapbert_pooling_method`, and `flat_ner` consistency.

**`flat_ner` deserves a note**, because it was a silent inconsistency rather than a missing field: `entity_extraction.py` used `predict_entities()`' default (flat) while `extraction.py` passed `flat_ner=False` (nested). The two Stage 2a models disagreed about whether overlapping spans were permitted, which was a library-default accident, not a decision. Now a single shared `FLAT_NER = True` constant. Flat was chosen because `extracted_entities` is the canonical table — Stage 4 writes one `:PatientObservation` per entity, so nested spans (`kidney` inside `acute kidney injury`) would produce two observations for one clinical fact, and the offset-overlap relation linking would face several equally valid targets. Nested extraction is defensible on its own terms but needs a deduplication policy that does not exist yet.

`docs/Provenance_Schema.md` is now fully implemented for Stages 1–2b. The only unwritten fields left are Stage 4's, which await the HITL queue.

### Measured effect on note `10000032-DS-21`

| Signal | Before | After |
|---|---|---|
| Entities accepted / normalized | 117 / 117 | 117 / 117 |
| Non-asserted (negated, family, hypothetical) | 42 | **17** |
| Spans retained below threshold (0.35–0.50) | 0 (discarded) | **33** |
| Relations found | 3 | 3 |
| Relations with an unlinked endpoint | 2 | **0** |

Two results worth carrying forward:

**The `flat_ner` unification was expected to reduce relation yield and did the opposite.** Relation count held at 3 while endpoint linking went from 33% to 100%, because two models producing flat spans agree on boundaries and clear the overlap threshold. Consistency between the models proved worth more than the extra candidate endpoints nesting supplied — the reverse of the assumption. Recorded in `src/extraction.py` because the prediction was confidently wrong and only the measurement caught it.

**The original-form fallback fired zero times, correctly.** It triggers only when abbreviation expansion actually altered the text, and this note's unmapped entities (`lasix`, `Lasix`, `AP`, `Tbili1`) are not expansion artefacts — they fail because brand names and value-bearing lab spans have no standard concept. The mechanism targets the `NAD`/`TTP` error class, which this note happens not to exhibit in a way that reaches normalisation. Zero is the right answer here, not a null result; the fallback needs a note containing a mis-expanded abbreviation that *would otherwise* have mapped to demonstrate itself.

**33 retained sub-threshold spans is a substantial population** — 22% on top of the 117 accepted. That is the pool Stage 3 could recover from, and the basis for measuring whether the 0.50 extraction threshold is set correctly.
