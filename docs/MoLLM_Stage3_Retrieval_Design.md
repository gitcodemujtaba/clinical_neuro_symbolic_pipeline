# Stage 3 (MoLLM) — Grounding Retrieval Design

> **Model-choice note (2026-08-14):** this doc was written against the earlier two-model vLLM ensemble (BioMistral-7B-AWQ / OpenBioLLM-Llama3-8B-AWQ). The standing ensemble as of 2026-08-14 is qwen2.5:3b / llama3.2:3b / phi4-mini served via Ollama (`src/llm_client.py`) — see `docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md` §6 for why, and `docs/2026-08-14_Dead_Code_Audit.md` for what else changed alongside it. Design rationale below may still apply structurally; specific model names, base URLs, and context-window numbers do not.

**Status:** design, pending approval. Supersedes nothing; extends `Databases.md` §3 and `Implementation_Methodology.md` Stage 3.
**Date:** 2026-08-08.
**Scope:** how Stage 3 queries the knowledge bases to assemble grounding context for the BioMistral 7B / OpenBioLLM 8B ensemble, given the actual Stage 2 output and the actual contents of the curated guideline KG.

> **Ensemble membership changed 2026-08-09.** The proposal named MedGemma 4B as the first member. MedGemma is Gemma-3-based and **cannot run on the project's Tesla T4** under any dtype: vLLM refuses `float16` for `gemma3` outright (numerical instability), `bfloat16` requires compute capability ≥ 8.0 against the T4's 7.5, and `float32` needs ~16GB of weights against 15.36GB of VRAM. Quantisation does not help — the rejected quantity is the compute dtype, not the weight format — and the constraint applies to all of MedGemma 4B, 1.5-4B and 27B. **BioMistral 7B (AWQ)** replaces it. The substitution preserves the property the consensus gate depends on: OpenBioLLM fine-tunes Llama-3 and BioMistral fine-tunes Mistral-7B, so the members' errors are not correlated through a shared base model. An ensemble of two Llama-3 derivatives would agree for reasons unrelated to the evidence, silently weakening the disagreement→HITL rule that is the whole point of §6.

Every quantitative claim below was measured directly against `data/local_triplets_db2_v6_cleaned/` (76 files) and `data/evaluaiton-dataset/snomed-ct-entity-linking-challenge-1.2.0/train_annotations.csv` (75,491 annotations), not estimated. Measurement commands are reproducible from §2.

---

## 1. Why this document exists

`Implementation_Methodology.md` describes Stage 3 as: *"High-confidence inputs undergo a guideline-contradiction check, while low-confidence inputs prompt a deeper resolution attempt using provenance, the SNOMED hierarchy, and guideline-triplet evidence."* `Databases.md` describes the mechanism as a Memgraph `MATCH` retrieving "guideline triplets and neighboring context" plus a Neo4j `MATCH` for "FSNs and IS_A hierarchy."

Both are correct at the level of intent and both are silent on the question that actually determines whether Stage 3 works: **given an entity that Stage 2 produced, how do we find the guideline rules that apply to it?** The obvious answer — match on SNOMED code — turns out to be both insufficient (§2.1) and unsafe (§2.2) against the real curated corpus. This document specifies the retrieval technique in enough detail to implement, and records the evidence that forced each decision.

---

## 2. Empirical baseline — what the data actually looks like

### 2.0 Corpus shape (`data/local_triplets_db2_v6_cleaned/`, post-cleaning)

| Measure | Value |
|---|---|
| Files / source guideline documents | 76 / 10 |
| Concept nodes | 1,697 |
| Rule edges | 1,119 |
| Nodes with a real SNOMED code | 715 (42.1%) |
| Nodes with `snomed: "N/A"` | 974 (57.4%) |
| Nodes missing the `snomed` key entirely | 8 (0.5%) |
| **Distinct SNOMED codes across the whole corpus** | **447** |
| Nodes with zero rules in *or* out (within-file) | 471 (27.8%) |
| Max rules attached to one code (corpus-wide, post cross-file merge) | 62 (`14669001` AKI) |

Node `@type` distribution: Finding 834, Condition 284, Intervention 263, Acuity 135, Medication 115, Quantitative Threshold 63, Timeframe 1, none 2.

Rule `citation_type` distribution: `verbatim` 492, `pointer_unverifiable` 213, `paraphrase` 202, `paraphrase_with_recovered_excerpt` 132, **absent 80**. The 80 rules with no `citation_type` at all are rules that carry no citation field — they were not covered by the cleaning script's classifier and must be treated as non-citable, same as `pointer_unverifiable`.

### 2.1 Finding A — guideline evidence reaches 8.5% of entities. Hierarchy traversal supplies 42% of that.

**Fully measured 2026-08-09** via `scripts/measure_channel_b_coverage.py` over all 75,491 gold annotations, against the grounded corpus (`local_triplets_db2_v6_cleaned_grounded`: 1,314 nodes, 871 rules, 570 SNOMED codes). Measured over GOLD concept_ids, so this is an **upper bound on retrieval** that excludes extraction and normalisation error — the end-to-end figure will be lower and must be measured separately.

| Stage | Concepts | Annotations | Coverage |
|---|---|---|---|
| Channel A — direct code match | 116 | 6,333 | 8.39% |
| Channel B — hierarchy only (≤3 hops) | +1,002 | +13,532 | +17.93% |
| **Combined, RAW** | **1,118** | **19,865** | **26.31%** |
| Channel A, after name-agreement guard | 67 | 3,715 | 4.92% |
| Channel B, after name-agreement guard | 303 | 2,727 | 3.61% |
| **Combined, GUARDED — the figure to quote** | **370** | **6,442** | **8.53%** |

Nearest-guideline-ancestor hop distribution: 254 concepts at 1 hop, 343 at 2, 405 at 3 (still rising at the cap; `--hops 4` untested).

**Two conclusions, both load-bearing.**

1. **Channel B is not an optimisation — it supplies 3.61 of the 8.53 percentage points, 42% of all reachable coverage.** Without hierarchy traversal, guideline grounding would reach under 5% of entities. This justifies the whole `athena_concept_ancestor` design.
2. **~91.5% of entities have no guideline evidence at all.** Not a defect: a 10-guideline corpus covering AKI/sepsis/HF/COPD/ACS/CAP/triage was never going to speak to every entity in a discharge summary, and most (`family history`, `discharge instructions`, an incidental anatomical mention) legitimately have nothing a guideline says about them.

**Framing consequence for the dissertation.** Stage 3 must be described as *ontology-grounded validation for all entities* (Channel C, which runs for 100%) with *guideline-grounded contradiction checking for a ~8.5% clinically-concentrated subset*. That subset is concentrated in exactly the high-acuity conditions the guidelines cover, which is where contradiction detection carries clinical weight — but claiming broad guideline grounding would misrepresent the measurement.

Three further design consequences:

1. **"No guideline evidence found" is the common case and must be a first-class, well-handled path** — not an error, not an empty string dropped into a prompt. A model handed an empty evidence block and asked "does this contradict the guidelines?" will confabulate. The prompt must state explicitly that no guideline evidence was retrieved and constrain the permissible verdict accordingly (§6.4).
2. **SNOMED `IS_A` hierarchy traversal is load-bearing, not a nicety.** It is the only mechanism that can connect a specific extracted entity (`Stage 2 AKI`, `NSTEMI`) to a guideline rule stated at a more general level (`Acute Kidney Injury`, `Acute coronary syndrome`). Without it, Stage 3's guideline-grounding claim rests on a 10.8% base rate.
3. **The ungrounded-`:GuidelineFact` name channel contributes almost nothing on its own (0.78%)** and cannot justify a SapBERT pass over 982 node names per entity. Demoted from a primary channel to a constrained one (§5, Channel D).

### 2.2 Finding B — the SNOMED code is *not* a safe join key on its own. This is the most important finding in this document.

The guideline KG's codes come from MedCAT linking over guideline text. Of the 91 codes carrying more than one distinct node name, **43 (47%) attach clinically unrelated names to a single code.** Measured by max-pairwise name similarity < 0.45:

| Code | Names sharing it | What went wrong |
|---|---|---|
| `24484000` | `GOLD 3 (severe)`, `major bleeding`, `severe AKI` | Code is the SNOMED **qualifier** "Severe" — attached to three unrelated severe things |
| `272118002` | `acute NSTEMI`, `ST-segment elevation myocardial infarction` | **Clinically opposite** — NSTEMI vs STEMI drive different reperfusion decisions |
| `278061009` | `CURB-65 score`, `HEART Score`, `CAAT score` | Generic "score" concept absorbing three different instruments |
| `25876001` | `American College of Emergency Physicians`, `B-lines on lung ultrasound`, `Medical emergency` | Journal/organisation boilerplate coded as clinical content |
| `401303003` | `reperfusion therapy`, `STEMI` | Intervention conflated with the condition it treats |
| `255560000` | `IV iron replacement`, `procalcitonin level`, `PSI class IV` | Unrelated across three domains |

`clean_local_triplets.py` already guards the *within-file* merge with an `@type`-agreement check (108 nodes correctly left unmerged and flagged `same_snomed_type_mismatch_not_merged`). That guard is necessary but **not sufficient for retrieval**, because:

- It only ran within each file. Cross-file, `25876001` spans 14 nodes across 11 files, `56675007` spans 20 nodes across 10 files.
- `@type` agreement does not imply concept agreement. `24484000`'s three names (`GOLD 3 (severe)`, `major bleeding`, `severe AKI`) can all legitimately be typed `Finding` and still be three different clinical facts.

**Consequence:** a naive `MATCH (c:Concept {snomed_code: $code})` retrieval would, for any note mentioning an NSTEMI, happily return STEMI reperfusion rules as grounding evidence, with full apparent confidence and a verbatim citation. That is a *worse* failure than retrieving nothing — it is an authoritative-looking wrong answer, in exactly the scenario (drug/reperfusion decisions) where being wrong is most costly. Every code-based match must therefore carry a **name-agreement guard** (§5.1).

This also means the `MERGE ... ON snomed` cross-file ingestion step recommended in `Guideline_Triplets_KG_Review.md` §3.2 **must not be implemented as written.** Merging `24484000`'s three nodes would fuse major bleeding with severe AKI into one graph node and make the error permanent and untraceable. Revised ingestion rule in §4.2.

### 2.2b Finding B-bis — the binding constraint is now the KG's content, not the retrieval code

`scripts/diagnose_guard_suppression.py` decomposes *why* a direct-code match fails to yield usable evidence. Over the 116 gold concepts that match a guideline code directly:

| Outcome | Concepts | Annotations | Share of gold |
|---|---|---|---|
| ACCEPTED | 66 | 3,712 | 4.92% |
| **`node_has_no_rules`** | **41** | **2,451** | **3.25%** |
| `name_reject` | 7 | 155 | 0.21% |
| `unverified_code_assertion` | 2 | 15 | 0.02% |

**`node_has_no_rules` is nearly as large as everything that succeeds.** These are concepts the guideline corpus *names* but attaches no rule to — isolated nodes, the same population as the zero-degree nodes in `Guideline_Triplets_KG_Review.md`. The highest-frequency examples are not obscure: `Physical examination` (645 annotations), `Hypertension` (551), `Clinical evaluation` (345), `Cardiac dysfunction` (120), `GERD` (102), `Anemia` (99).

Two things follow. First, **no amount of retrieval tuning can recover this 3.25%** — it is a curation gap, and attaching rules to the high-frequency concepts already present in the KG is the single highest-yield improvement available, worth roughly +3pp of coverage for authoring effort on ~40 concepts. Second, the guard is now demonstrably well-calibrated: 0.21% residual rejection, and the surviving rejections are defensible on inspection (`Myocardial infarction` vs `Reinfarction`, `Acute myocardial infarction` vs `Recent or remote MI` — genuinely different clinical events).

Getting there required three corrections to the guard, each found by measurement rather than reasoning, and each recorded in `src/retrieval.py`:

| Fix | Guarded coverage |
|---|---|
| (initial: guard compared entity name to ancestor node name) | 4.02% |
| Channel B guards against the **ancestor's** name, not the entity's | 6.96% |
| Guard scores against the concept's **synonym set**, not just its FSN | 7.45% |
| **Acronym** relationship (`WBC` ← White **b**lood **c**ell) + shared-stem morphology | **8.53%** |

The first was a design error: guarding a generalisation against the specific entity's name rejects precisely what Channel B exists to produce. The second and third were assumptions about what the vocabulary contains — the synonym table holds `'Stroke'` and `'WBC count'` but not a bare `'COPD'`, so both the vocabulary lookup and the string-derived acronym rule were needed, and neither alone was sufficient.

### 2.3 Finding C — documentation/code discrepancy: boilerplate flagging never fired

`Implementation_Checklist.md` and `Guideline_Triplets_KG_Review.md` §6 both state boilerplate nodes were flagged `quality_flag: likely_boilerplate`. Verified against the cleaned corpus: **zero nodes and zero rules carry that flag**, and `boilerplate_flags` is absent from the cleaning report summary.

The code path exists and runs by default (`flag_boilerplate()`, `--no-flag-boilerplate` opt-out), so the patterns simply never matched. Cause: `BOILERPLATE_PATTERNS` targets the strings found in §3.7's review of the *grounded chunk text* (`Annals of Emergency Medicine`, `Key words/phrases for literature searches`, `Study Selection:`), but is applied to triplet **node names** and rule `citation`/`rationale` fields, where the boilerplate surfaces differently. `American College of Emergency Physicians` — a confirmed boilerplate node name under code `25876001` — is not in the pattern list at all.

Not a blocker for this design (§5.1's name-agreement guard and §5.4's exclusion list independently suppress these nodes), but the two docs currently overstate what was done and should be corrected.

### 2.4 Finding D — rule volume per concept is manageable but long-tailed

Corpus-wide rule counts per SNOMED code after conceptual merge: median 1, but AKI `14669001` = 62, ICU admission `309904001` = 48, septic shock `76571007` = 41, sepsis `91302008` = 39. Since the highest-degree concepts are precisely the ones most likely to appear in an ICU/ED discharge summary, **the worst case is also the common case** — a hard cap and a principled ranking are required, not optional (§5.5).

---

## 3. The Stage 2 → Stage 3 contract

Stage 3 consumes one **validation record** per entity. This is the complete input; Stage 3 performs no re-extraction.

```
ValidationRecord
  entity_id            <note_id>-e<hash(orig_start, orig_end, entity_label)>   # Stage 2a, new
  note_id
  original_text        surface form as written in the note
  expanded_text        post-abbreviation-expansion form
  gliner_label         Condition | Symptom | Medication | Procedure | Anatomy | Lab Test
  gliner_confidence
  orig_start/orig_end  offsets into the ORIGINAL note (the "time machine" output)
  local_context        sentence-bounded window around the entity, capped ~800 chars   # new

  # --- clinical assertion block (new; see Stage1_2_Completeness_Audit.md sev 1) ---
  assertion_status     PRESENT | ABSENT | POSSIBLE | CONDITIONAL
  experiencer          PATIENT | FAMILY | OTHER
  temporality          CURRENT | HISTORICAL
  assertion_cue        matched cue text + offset, or null
  # --- note structure (new; sev 2) ---
  section_name         e.g. History of Present Illness | Family History | Allergies
  sentence_id
  # --- Stage 1 expansion provenance (new; sev 3) ---
  expansion_applied    the expansion used for this span, or null
  expansion_ambiguous  true if the abbreviation had >1 known meaning
  candidate_expansions[]  all known meanings, when ambiguous

  candidates[]         1 candidate if unambiguous, else top-3                          # Stage 2b, new
      omop_concept_id, concept_name, domain_id, vocabulary_id,
      match_tier (1|2|3), similarity_score
  confidence_tier_in   HIGH | LOW
  relations[]          Stage 2a relations where this entity_id is head or tail
      relation_id, relation_label, other_endpoint_text, other_entity_id|null,
      relation_confidence, entity_link_status, overlap_ratio
```

**`confidence_tier_in` = LOW if any of:** `gliner_confidence < 0.60`; the Stage 2b ambiguity gate fired (>1 distinct `concept_id` at Tier 1/2; or Tier 3 top-1 below the 0.72 floor; or Tier 3 top1−top2 margin < 0.05); **or `expansion_ambiguous = true`**. Otherwise HIGH. These capture three distinct failure modes — *"is this even an entity"*, *"which concept is it"*, and *"was the abbreviation expanded correctly in the first place"* — and any one being weak warrants deeper resolution. The third is a particularly good demonstration of Objective 2: disambiguating `MS` → multiple sclerosis vs mitral stenosis vs morphine sulfate from local context and KG evidence is exactly the neuro-symbolic task the thesis claims.

**Relation endpoint linking uses character-offset overlap, not text matching.** Both GLiNER-BioMed and GLiNER-relex run on `expanded_text`, so their spans share a coordinate system; `head_entity_id`/`tail_entity_id` resolve by maximum overlap against `extracted_entities` (≥50% required, else `entity_link_status = unresolved`). This supersedes the text+label matching originally proposed — see `Stage1_2_Completeness_Audit.md` §4.

`local_context` is sentence-bounded and capped rather than note-scoped because OpenBioLLM 8B's context window is **8,192 tokens** (it fine-tunes base Meta-Llama-3-8B, not 3.1) while notes run 2,374–24,858 chars. A single long note is ~6,200 tokens on its own. Per-record calls with a bounded local window keep every prompt's size independent of note length. BioMistral inherits Mistral-7B-v0.1's 32K positions, but that headroom is irrelevant — both ensemble members must see identical input for their votes to be comparable, so OpenBioLLM's 8K is the binding budget. In practice the servers are launched with `--max-model-len 4096` so both models fit on one 15.36GB card, making **4,096** the operative limit; 8,192 is the ceiling if they are ever given separate GPUs.

---

## 4. KG1 — unified reference graph

Confirmed decision: collapse the Neo4j/Memgraph reference split into **one** graph (KG1). `Databases.md`'s "Unified Graph Design" paragraph already describes shared `:Concept` nodes; its Stage-to-DB matrix contradicts this by splitting SNOMED into Neo4j and guidelines into Memgraph. Splitting makes §5's core traversal — *walk IS_A ancestors and collect guideline rules attached to any of them* — impossible in one query. Naming completes the existing scheme: **KG1** reference ontology, **KG2** DuckDB lexical/vector store, **KG3** patient instances + provenance ledger (unchanged).

### 4.1 Node and edge types

```
(:Concept)          snomed_code (unique, indexed), fsn, preferred_term, semantic_tag, in_snomed_release
                    -- authoritative, from the SNOMED CT release import
(:GuidelineNode)    guideline_node_id, name, name_norm (indexed), node_type,
                    source_document, section_title, snomed_code_asserted (nullable),
                    code_link_status, quality_flag
                    -- one node per curated triplet node; NEVER merged across files
(:Concept)-[:IS_A]->(:Concept)
(:GuidelineNode)-[:ASSERTS_CODE]->(:Concept)      -- only when the name-agreement guard passes
(:GuidelineNode)-[:<PREDICATE>]->(:GuidelineNode) -- 49 canonicalized predicate types
```

### 4.2 Ingestion rules (these are the corrections Finding B forces)

1. **`:GuidelineNode` identity is `(source_file, @id)` — never the SNOMED code.** Do **not** `MERGE ON snomed`. §2.2 shows this would fuse clinically opposite concepts. `Guideline_Triplets_KG_Review.md` §3.2's recommendation is superseded here, and that document should be annotated accordingly.
2. **Grounding is an edge, not a property.** A guideline node's asserted code becomes an `ASSERTS_CODE` edge to the real `:Concept` **only if the name-agreement guard (§5.1) passes**. Otherwise `code_link_status` is set to `asserted_unverified` and no edge is created — the assertion is retained for audit but is unreachable by code-based traversal.
3. **Predicates stay distinct.** All 49 canonicalized predicates (`INDICATES` 465, `REQUIRES_INTERVENTION` 274, `TRIGGERS_SEVERITY` 130, `REQUIRES_MEDICATION` 70, …) become distinct edge types. `Databases.md`'s generic `GUIDELINE_RELATION` placeholder is dropped — collapsing them discards the decision semantics that make triplets more useful than text.
4. **Drop nothing, flag everything.** The 471 zero-degree nodes, the 108 `same_snomed_type_mismatch_not_merged` nodes, and boilerplate nodes are all ingested with flags. §5.4 excludes them at *retrieval* time, which is reversible; deletion is not.

---

## 5. The retrieval algorithm

For each `ValidationRecord`, and for each of its `candidates[]`, run four channels, pool the results, score, rank, cap. Every retrieved rule carries a `match_confidence ∈ [0,1]` and a `match_channel` — both surfaced in the prompt and persisted to provenance, so a reviewer can always see *why* a piece of evidence was considered relevant.

### 5.1 The name-agreement guard (applies to every code-based match)

Before any code match is trusted:

```
guard(entity_name, guideline_node_name) =
    max( token_set_ratio(entity_name, node_name),
         token_set_ratio(concept_fsn,  node_name) )     # normalized, stopword-stripped
```

- `≥ 0.75` → **agree**, `match_confidence` unpenalized
- `0.45 – 0.75` → **weak**, `match_confidence × 0.6`, rule tagged `name_agreement: weak`
- `< 0.45` → **reject**, rule not retrieved, logged as `suppressed_code_collision`

Comparing against the SNOMED **FSN** as well as the raw entity text matters: an entity extracted as `NSTEMI` and a guideline node named `acute NSTEMI` agree at FSN level even when surface forms differ, while `24484000`'s `major bleeding` vs `severe AKI` fail against both. This single guard is what prevents the §2.2 failure mode, and its rejection count is a metric worth reporting in the dissertation — it directly quantifies noise in the curated KG.

### 5.2 Channel A — direct code match (`match_confidence` 1.00, ×guard)

Candidate's SNOMED code → `:Concept` → attached `:GuidelineNode`s via `ASSERTS_CODE` → their rules (in *and* out edges; a rule targeting the entity's concept is as relevant as one originating from it). Expected reach ~10.8% of entities.

**Medication entities** normalize against RxNorm, which has no SNOMED code. Resolution path (your suggestion, adopted): query **KG2** `athena_concept_relationship` for a `Maps to` / `RxNorm – SNOMED eq` row from the RxNorm `concept_id` to a SNOMED-vocabulary `concept_id`, then use that code as the Channel A key. Falls back to Channel D if no crosswalk row exists. **This is unverified** — `code/data/athena_omop/` is still a `.gitkeep`, so whether the crosswalk is populated in your Athena download must be checked once `import_athena.py` runs (§8).

### 5.3 Channel B — hierarchy match (`match_confidence` = `0.9^hops`, ×guard)

The channel that carries the design. From the candidate's `:Concept`, walk `IS_A` **upward** up to 3 hops; at each ancestor, collect guideline rules as in Channel A.

```cypher
MATCH (c:Concept {snomed_code: $code})-[:IS_A*1..3]->(anc:Concept)
      <-[:ASSERTS_CODE]-(gn:GuidelineNode)
MATCH (gn)-[r]-(other:GuidelineNode)
RETURN anc.snomed_code, anc.fsn, length(path) AS hops, gn, type(r), r, other
```

Three constraints, each with a reason:

- **Upward only.** Downward (`Acute Kidney Injury` → its 40 descendant subtypes) retrieves rules about conditions the patient was never documented as having. Generalizing from what the note says is valid inference; specializing into what it doesn't say is not.
- **Hard 3-hop cap.** SNOMED's upper hierarchy converges on near-root concepts (`Clinical finding`, `Disorder of body system`) that subsume most of the corpus. Beyond ~3 hops, every entity matches every rule and `match_confidence` becomes meaningless.
- **Stop-node list.** Explicitly bar traversal through a small set of semantically empty ancestors (`404684003 Clinical finding`, `64572001 Disease`, `71388002 Procedure`, `123037004 Body structure`, `362981000 Qualifier value`). `Qualifier value` in particular is how the `24484000` "Severe" collision propagates, so barring it kills a whole class of false matches at the traversal level rather than relying solely on the §5.1 guard.

The `0.9^hops` decay (1 hop 0.90, 2 hops 0.81, 3 hops 0.73) is deliberately gentle: a 1-hop generalization is usually clinically sound, and over-penalizing it would suppress the channel that provides most of Stage 3's real coverage. Flagged as calibration-tunable (§8).

### 5.4 Retrieval-time exclusions

A rule is dropped, whatever channel found it, if any hold:

- Source or target node carries `quality_flag: likely_boilerplate` (**once §2.3 is fixed** — currently a no-op)
- Source or target node name matches the boilerplate pattern list evaluated at query time (interim mitigation while §2.3 is unfixed)
- Source or target node carries `same_snomed_type_mismatch_not_merged` **and** the match came through Channel A/B — these 108 nodes are exactly the known-unreliable code assertions
- The rule has no citation, or `citation_type ∈ {pointer_unverifiable}` **and** the rule would be the *only* evidence retrieved — an unverifiable pointer as sole grounding invites an uncheckable citation (§7). Retained when other, checkable rules accompany it.

### 5.5 Ranking and budget

Pool all channels, deduplicate by `(predicate, source_node, target_node)` keeping the highest `match_confidence`, then sort by:

1. `match_confidence` **descending** (primary — as you specified: how well the rule's node matches the entity dominates)
2. `citation_type` rank as tiebreaker only: `verbatim` > `paraphrase_with_recovered_excerpt` > `paraphrase` > `pointer_unverifiable`/none
3. Predicate–label affinity as final tiebreaker (a `REQUIRES_MEDICATION` rule ranks above a `IS_OUTCOME_OF` rule for a `Medication` entity)

**Cap: 5 rules per validation record.** With `rationale` + citation excerpt each rule costs ~80–150 tokens; 5 rules ≈ 400–750 tokens, which fits the §6.5 budget. Nodes like AKI (62 rules) would otherwise blow the window on their own.

### 5.6 Channel C — SNOMED context (always retrieved, not ranked)

Independent of guideline evidence, every record gets: candidate FSN, semantic tag, immediate `IS_A` parents (≤3), and — **only when `confidence_tier_in = LOW` and multiple candidates are being disambiguated** — each candidate's parent chain, since that is precisely the signal that distinguishes `Ed District` from `Erectile dysfunction`. This is the channel that serves the ~87.5% of entities with no guideline evidence: they still get real symbolic grounding, just ontological rather than guideline-based.

### 5.7 Channel D — semantic name match (constrained, LOW-tier only)

SapBERT cosine between entity text and `:GuidelineNode.name_norm`, floor 0.72, top-3, `match_confidence` = the cosine score. Runs **only** when `confidence_tier_in = LOW` **and** Channels A/B returned nothing — because §2.1 measured its standalone contribution at 0.78%, which does not justify a vector pass per entity as a default. It exists to reach the 982 ungrounded `:GuidelineFact` nodes that are structurally unreachable by code, and for the Medication-crosswalk-miss fallback.

Note the asymmetry that makes this channel weak: ungrounded node names average 5.3 words (`Absence of volume overload or alternative diagnosis`), grounded ones 2. Compound guideline phrasings do not embed close to short clinical spans — the same problem `backfill_guideline_grounding.py` addresses with GLiNER span pre-extraction, and the reason this channel is a fallback rather than a peer of A/B.

---

## 6. Prompt assembly

Fixed block order, budgeted, identical for both models:

1. **System** — role, the closed verdict set, output JSON schema, and the standing instruction that evidence may only be cited from the EVIDENCE block by `rule_id`.
2. **ENTITY** — `original_text`, `expanded_text`, `gliner_label`, `gliner_confidence`; **plus the assertion block stated as fact**: `assertion_status`, `experiencer`, `temporality`, and the matched `assertion_cue`. Where `expansion_ambiguous`, the alternatives are listed explicitly as a disambiguation task for the model.
3. **CONTEXT** — `section_name`, then the sentence-bounded `local_context` window with the entity marked inline.
4. **CANDIDATE(S)** — 1 (HIGH) or up to 3 (LOW), each with concept name, domain, vocabulary, match tier, similarity, FSN, parents.
5. **RELATIONS** — Stage 2a relations touching this entity, each annotated with its `entity_link_status`.
6. **EVIDENCE** — ≤5 ranked rules. Per rule: `rule_id`, predicate, source→target names, `rationale`, the citable text selected by `citation_type` (§7), plus **`match_channel` and `match_confidence` stated explicitly**, and `name_agreement: weak` where applicable. The model is told directly that a 3-hop hierarchy match is weaker evidence than a direct code match — surfacing retrieval provenance to the model rather than laundering everything into an undifferentiated evidence list.
7. **TASK** — mode-dependent (§6.4).

### 6.4 Two modes, two verdict sets

- **HIGH → contradiction check.** *"Does this concept assignment contradict the retrieved guideline evidence?"* Verdicts: `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE`.
- **LOW → deeper resolution.** *"Which candidate best matches the entity in context, or none?"* Verdicts: `RESOLVED_TO_CANDIDATE_N` / `NONE_CORRECT` / `INSUFFICIENT_EVIDENCE`.

**When the EVIDENCE block is empty** (~87.5% of entities), the block is rendered as an explicit `NO GUIDELINE EVIDENCE RETRIEVED` marker and the permitted verdict set narrows: HIGH-tier records may only return `SUPPORTED` (ontologically consistent per Channel C) or `INSUFFICIENT_EVIDENCE` — **`CONTRADICTED` is structurally unavailable**, because a contradiction with no retrieved guideline to contradict is definitionally a hallucination. This is the single most important guardrail in the prompt design, given how common the empty case is.

### 6.4b Assertion status gates the contradiction check symbolically, before the model sees it

Guideline rules describe what to do when a finding **is present**. Applying them to a negated, family-history, or hypothetical mention is a category error, and it is one that should be caught by rule rather than delegated to the LLM. Given the measured distribution (~21% of spans non-assertive — 14.2% negated alone), this is a high-frequency path, not an edge case:

- `assertion_status = ABSENT` or `experiencer ≠ PATIENT` → **skip guideline retrieval entirely** (Channels A/B/D), run Channel C only, and route to a reduced validation task: *"is this concept assignment correct for a negated / family-history mention?"* The contradiction check is not asked, because there is no asserted patient fact to contradict.
- `temporality = HISTORICAL` → guideline evidence **is** retrieved but the prompt states the finding is historical, and `CONTRADICTED` requires the model to justify why a past finding conflicts with current management.
- `assertion_status = POSSIBLE | CONDITIONAL` → retrieved normally, flagged as uncertain; a `CONTRADICTED` verdict on a merely suspected finding is downgraded to `INSUFFICIENT_EVIDENCE`.

This also protects KG3: `ABSENT` and `FAMILY` records must never be written back as `:PatientObservation` facts regardless of MoLLM verdict — an explicit Stage 4 rule, not a model decision.

### 6.5 Token budget (8,192 hard, OpenBioLLM)

| Block | Budget |
|---|---|
| System + schema | ~450 |
| Entity + assertion block + context window + section | ~450 |
| Candidates (≤3, with FSN/parents) | ~400 |
| Relations | ~150 |
| Evidence (≤5 rules) | ~750 |
| Task instructions | ~150 |
| **Input subtotal** | **~2,350** |
| Reserved for generation | ~800 |
| **Headroom** | **~5,100** |

Comfortable. The headroom exists deliberately — SNOMED FSNs and guideline rationales vary widely in length, and a blown context window silently truncates the EVIDENCE block, which would corrupt citation verification without raising an error. A hard pre-flight token count with evidence-block trimming (drop lowest-ranked rules first) is required in the implementation.

---

## 7. Citation verification

Prompt side, branching on the source rule's `citation_type`:

| `citation_type` | Presented as | Verification |
|---|---|---|
| `verbatim` | `citation` as quotable source text | **Strict** |
| `paraphrase_with_recovered_excerpt` | `citation_verbatim_excerpt` as quotable | **Strict** |
| `paraphrase` | `citation`, explicitly labelled *"paraphrase — not a direct quote"* | **Loose** |
| `pointer_unverifiable` / absent | not offered as citable (§5.4) | n/a |

- **Strict:** the model's `cited_evidence` must appear in the offered source text by longest-common-substring containment ≥ 0.8 — reusing the exact metric and threshold `clean_local_triplets.py` used to assign `citation_type` in the first place, so the same methodology builds the ground truth and audits the model against it.
- **Loose:** verify the cited `rule_id` exists in what was actually shown to the model — i.e. detect fabricated attribution, which is all that is checkable when the source was never a verbatim quote.

**`citation_verified: false` forces HITL routing**, bypassing the confidence math entirely. A model citing evidence it was never given is the precise hallucination this mechanism exists to catch; it should not be recoverable by a high confidence score.

---

## 8. Prerequisites, calibration parameters, and honest limitations

**Blocking prerequisites**

1. `import_athena.py` — Channel A's Medication path needs `athena_concept_relationship` populated, **and the RxNorm→SNOMED crosswalk's actual coverage measured**. If it is sparse, Medication entities fall to Channel D and that limitation must be stated, not assumed away.
2. SNOMED release import into KG1 — Channel B is inert without `IS_A` edges, and Channel B is what lifts coverage past 10.8%.
3. Stage 2 patch — `entity_id`, dedup-fanout, relation FKs (offset-overlap), `LIMIT 3` + ambiguity gate, `local_context` extraction.
3b. **Stage 1/2 completeness changes** — assertion status via medspacy, section segmentation, abbreviation ambiguity. Specified in `Stage1_2_Completeness_Audit.md` (severity 1–4 approved 2026-08-08). §6.4b's symbolic gating and the `ValidationRecord` assertion block both depend on these; without them Stage 3 validates negated findings as asserted patient facts on ~14% of entities. Adds `medspacy` to `requirements.txt`.
4. `init_kg1_guidelines.py` — ingest per §4.2 (**not** the superseded merge-by-code rule).
5. Fix or retract the §2.3 boilerplate-flagging claim in the two affected docs.
6. Human pass over the 108 `same_snomed_type_mismatch_not_merged` nodes.

**Parameters requiring calibration against the validation slice** (per `Evaluation_Criteria.md`, none are load-bearing until calibrated): GLiNER floor 0.60 · Tier-3 margin 0.05 · name-agreement bands 0.75/0.45 · hierarchy decay 0.9^hops · hop cap 3 · evidence cap 5 · Channel D floor 0.72 · relation overlap threshold 0.50 · the two `composite_confidence` routing thresholds.

**Limitations to state plainly in the dissertation**

- Guideline grounding reaches a minority of extracted entities by construction (10.8% direct, plus whatever Channel B adds — *measure this once the hierarchy is loaded; do not estimate it*). The honest framing is that Stage 3 provides **guideline-grounded** validation for a well-defined clinical subset and **ontology-grounded** validation for the remainder — which is still a meaningfully stronger claim than ungrounded LLM validation, and is falsifiable.
- The curated KG's SNOMED assertions carry a measured ~47% collision rate among multi-name codes. The §5.1 guard mitigates this at retrieval time; it does not repair the underlying data. Report the suppression count.
- ~~Channel B's real coverage lift is **unmeasured**~~ — **RESOLVED 2026-08-09.** Measured at +3.61pp guarded (42% of all reachable coverage), 26.31% raw before guarding. See §2.1. This was named as the largest single uncertainty in the design; it is now the best-characterised part of it.
- **Guideline coverage tops out at 8.53% of gold entities**, and ~91.5% receive ontology-only grounding. The dissertation must state this rather than imply broad guideline grounding. A further ~3.25% is blocked by guideline nodes that carry no rules — a curation gap, not a retrieval one (§2.2b).
- All coverage figures are measured over **gold concept_ids**, isolating retrieval from extraction/normalisation error. The end-to-end figure will be lower and is a separate measurement: run the pipeline across the 272 notes, then repeat `measure_channel_b_coverage.py` against `normalized_entities` instead of the gold CSV. Conflating the two would make a retrieval problem indistinguishable from a normalisation problem.
- Hop distribution is still rising at the 3-hop cap (254/343/405 concepts at 1/2/3 hops), so `--hops 4` may add coverage. Untested. The stop-code list is what makes deeper traversal defensible.
- Assertion detection is rule-based (medspacy ConText). Published NegEx/ConText negation performance on clinical text is strong but imperfect, and errors are systematic rather than random. Report assertion-detection accuracy on a manually-reviewed sample rather than assuming it, and note that the DrivenData gold set annotates spans irrespective of assertion — so span/concept F1 is **blind to assertion errors** and cannot be used to validate this component.

---

## 9. Module structure

```
src/mollm_ensemble.py     orchestration, routing, voting, citation verification
src/retrieval.py          KG1Retriever (channels A/B/C), VocabularyRetriever (KG2), ranking
src/llm_client.py         vLLM OpenAI-compatible client, logprob extraction, schema validation
```

`retrieval.py` is deliberately separable from `mollm_ensemble.py` so the retrieval channels can be unit-tested against fixtures — and, more importantly, so coverage can be measured over the whole gold set without any LLM calls. That measurement (Channel B's real lift) is the first experiment to run once KG1 is populated.
