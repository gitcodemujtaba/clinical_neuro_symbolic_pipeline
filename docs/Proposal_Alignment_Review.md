# Proposal Alignment Review
## Clinical Neuro-Symbolic Pipeline — SME Review (Active Learning / Grounded LLM NER)

Reviewer stance: subject-matter review against the COM748 proposal ("Towards Trustworthy Clinical NER: Knowledge Graph-Enhanced Local LLMs with Active Learning"), the four `docs/*.md` design docs, and the current `code/` implementation.

---

## 1. Overall Verdict

The **architecture is well specified and stays within the proposal's boundaries** — arguably it's a faithful, more rigorous elaboration of the 5 objectives (three KGs instead of one, explicit provenance schema, a real held-out benchmark). The gap is **implementation maturity, not scope**: of the 5 objectives, only Objective 1 (partially) and pieces of Objective 4 have working code. Objectives 2, 3, and 5 — the parts that make this a *neuro-symbolic active-learning* project rather than a NER-plus-normalization script — exist only as documentation right now.

| Objective | Proposal claim | Code status |
|---|---|---|
| 1. Local LLM + Ontology Integration | Deploy local LLM + domain KG | **Partial.** DuckDB lexical store + normalization work. Neo4j/Memgraph ingestion scripts (`init_memgraph_snomed.py`, `init_memgraph_guidelines.py`, `import_athena.py`) are 0-line stubs. No LLM client code exists yet. |
| 2. Neuro-Symbolic Prompting | KG-grounded context injection into LLM | **Not started.** `mollm_ensemble.py` (Stage 3) is empty. |
| 3. Active Learning / Feedback Loop | Uncertainty routing, HITL, KG write-back | **Not started.** All HITL Streamlit pages and `ui/app.py` are empty. No routing/write-back logic anywhere. |
| 4. Dataset Repurposing + KG Embeddings | Repurpose classification data → KG embeddings | **Partially re-scoped** (see §3.1) and **partially implemented** — `merge_embeddings.py` works but `build_concept_embeddings.py` (the step that generates them) is an empty stub. |
| 5. Evaluation & Cost-Benefit | F1/recall vs. baseline, annotation cost reduction | **Design only.** `Evaluation_Criteria.md` is thorough and good; `eval_suite.py`, `cal_eval.py`, `ablations.py` are all 0 lines. |

**Working end-to-end today:** Stage 1 (preprocessing/abbreviation expansion) → Stage 2a (GLiNER span extraction) → Stage 2b (OMOP normalization). That's roughly 40% of the documented 5-stage pipeline and none of the "neuro-symbolic" or "active learning" parts that the thesis title promises.

---

## 2. What's Solid

- **Offset provenance ("time machine") is genuinely well engineered.** `preprocessing.py` and `entity_extraction.py` correctly track original↔expanded character offsets through abbreviation expansion, which is a real, often-botched detail in clinical NLP pipelines. This directly supports the proposal's "white box"/traceability claim.
- **Evaluation design is rigorous and appropriately skeptical of itself.** `Evaluation_Criteria.md` specifies note-level bootstrap CIs, paired comparison, ECE calibration, and — notably — flags its own **baseline asymmetry risk** (Clinical-T5 may have seen the DrivenData notes during pretraining). That kind of self-flagged threat-to-validity is exactly what a methodology section should do, and it will read well in the dissertation.
- **Provenance schema (`Provenance_Schema.md`) is a good match** for the proposal's "deterministic traceability" and "white box explainability" ethical claims — the append-only, stage-by-stage field accumulation is the right design for an audit trail.
- **Three-database split (DuckDB / Neo4j / Memgraph)** is a sensible, if ambitious, elaboration of "a domain specific Knowledge Graph" — separating static reference ontology from dynamic patient-instance data is good practice (keeps PHI-adjacent data structurally isolated from the reusable reference graph, which also helps the privacy story).

---

## 3. Scope & Consistency Issues Worth Resolving

### 3.1 Objective 4 has quietly changed shape
The proposal's problem statement and Objective 4 center on **repurposing legacy *classification* datasets** (weak supervision → KG embeddings) because gold-standard NER data is assumed scarce. The actual evaluation design instead uses the **DrivenData SNOMED CT Entity Linking Challenge** — 272 MIMIC-IV-Note discharge summaries with *gold-standard span + concept annotations* from SNOMED International. That's a stronger dataset, but it's a different premise: you're no longer demonstrating "how to bootstrap NER from noisy repurposed classification labels," you're benchmarking against a clean, expert-annotated NER set. This isn't necessarily wrong, but the dissertation needs to explicitly justify the pivot — right now the docs and the proposal tell two different stories about what "dataset repurposing" means. Worth a paragraph in the methodology chapter reconciling this.

### 3.2 Clinical-T5's dual role — RESOLVED (2026-08-07)
Originally flagged: Clinical-T5 appeared both inside the pipeline (Stage 2a ReLEx, per the README) and as the external baseline (per `Implementation_Methodology.md`), which would have confounded the "matches/exceeds baseline" claim.

**Update:** confirmed removed from the pipeline. Clinical-T5 was pretrained on MIMIC-III/MIMIC-IV, so keeping it as an in-pipeline component created a data-contamination risk (it may have seen the actual evaluation notes during pretraining) — correctly identified as a reason to exclude it from the system under test. It is now used exclusively as an external baseline for the accuracy/hallucination comparison at the evaluation stage. This is the right call and should be stated explicitly in the dissertation methodology chapter (not just implied), since it strengthens rather than weakens the eventual claim: if the pipeline matches or exceeds a baseline that had a pretraining-exposure advantage on the same corpus, that's a *harder* bar cleared, not an easier one. The README (`code/README.md`) still describes Clinical-T5 as part of Stage 2a "Extraction & Relation Extraction" — worth updating so the README matches the current architecture.

One open question this resolution creates: with Clinical-T5 removed from the live pipeline, **relation extraction (ReLEx) currently has no owner** — see §3.5.

### 3.3 Stage 2a database access contradicts its own doc
`Implementation_Methodology.md` states Stage 2a (GLiNER-ReLEx) runs "directly on the expanded text **without database or graph access**." In practice, `entity_extraction.py` reads Stage-1 provenance from DuckDB and writes extracted entities back to DuckDB within the same function. Harmless in practice, but worth correcting the doc (or refactoring so extraction is pure and persistence is a separate caller) so the architecture description matches the code — this kind of doc/code drift compounds fast once Stages 3–5 land.

### 3.4 Documented thresholds aren't implemented
- `Implementation_Methodology.md` specifies a **0.72 global similarity threshold** gating Tier-3 SapBERT matches. `normalization.py` has no threshold check at all — it always returns the top-1 nearest concept regardless of similarity score. As written, "Tier 0 (Failed)" can never actually be reached except when the table has zero embedded rows.
- `Databases.md` describes **domain-filtering** the Tier-3 vector search (e.g., GLiNER `Medication` → OMOP `Drug` domain) to cut cross-domain mismatches. The current `tier3_query` has no `domain_id` filter.

Both are called out as "open items" in the docs, so the team is aware — but they're not cosmetic: without them, normalization accuracy numbers from any current test run will be optimistic/misleading versus what Stage 2b will do once these are added. Worth fixing before generating any evaluation numbers that go in the dissertation.

### 3.5 Relation extraction is not yet implemented at all
GLiNER-ReLEx and Clinical-T5's relation-triple extraction ("[Medication] -[:TREATED_WITH]-> [Condition]") are described in the README as core to Stage 2a, but `extraction.py` (the presumed Clinical-T5 wrapper) is empty, and `entity_extraction.py` only extracts spans — no relations, no relation confidence, no `head_entity_id`/`tail_entity_id` fields from `Provenance_Schema.md`. This matters because the KG (Neo4j/Memgraph) triple structure described throughout the docs depends on relations, not just spans.

### 3.6 Local-only compute claim vs. `ec2-user` paths
Every module hardcodes `PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"`, and `boot_infra.sh` docker-starts named containers — strongly implying development/inference is happening on an AWS EC2 instance. The proposal's Legal section is emphatic that the LLM ensemble runs "exclusively on local infrastructure or within a secure private network" specifically to avoid cross-border transfer and third-party API exposure risk, and the Risk Management section pre-authorizes cloud use only for *training/fine-tuning*, with inference "pulled back to local infrastructure." Two things to confirm before this becomes a dissertation liability:
1. Is inference (not just training) currently running on EC2, or only dev/testing with synthetic data?
2. If MIMIC-IV-Note (a PhysioNet credentialed-access dataset) is ever processed on that EC2 instance, PhysioNet's own cloud-hosting attestation requirements apply — worth confirming your DUA covers this specific cloud setup, since "local infrastructure" was the explicit ethical/legal justification in the proposal.

This may already be handled (e.g., a private VPC used only for de-identified/synthetic dev data) — flagging so it's an explicit, documented decision rather than an implicit one, given how central the "no cloud" claim is to the proposal's ethics section.

### 3.7 GLiNER model is general-purpose, not clinical-domain
`entity_extraction.py` loads `urchade/gliner_medium-v2.1`, a general-domain zero-shot NER model, not a biomedical/clinical-tuned variant. This is a reasonable placeholder for pipeline plumbing, but for the accuracy claims in Objective 5 to be credible against Clinical-T5, swapping to a clinical/biomedical GLiNER checkpoint (several now exist) before running benchmark numbers is worth prioritizing — otherwise the F1 comparison risks under-representing what the architecture can do.

### 3.8 DrivenData SNOMED CT Entity Linking Benchmark — scope and metric mismatches (checked 2026-08-07; corrected 2026-08-07 after inspecting the local dataset)
Pulled the live benchmark page (drivendata.org/benchmarks/310) to sanity-check `Evaluation_Criteria.md` against the actual benchmark spec, then re-checked directly against the dataset now in `evaluaiton-dataset/snomed-ct-entity-linking-challenge-1.2.0/`. Two of the three original points needed correction once real data was available — **this section supersedes what was originally written here.**

1. **CORRECTED — medications are partially present, not absent.** The live Benchmark page states substances/medications are out of scope. But this is the archived PhysioNet **v1.2.0 "Challenge"** release, not the same artifact as the live rolling **"Benchmark"** — they are related but distinct (see point 4 below), and the Challenge's actual scope is broader. A keyword scan of `train_annotations.csv` found 96+ drug-related spans with real `concept_id`s (`Aspirin`, `Warfarin`, `heparin drip`, `insulin regimen`, `oral antibiotics`, etc.). These are most likely SNOMED **Clinical Finding** concepts (e.g., "taking warfarin" as a finding/therapy status) rather than **Substance**-hierarchy drug-product concepts — which would actually reconcile with the live Benchmark's stated rule, just at one level of indirection. Action: once the SNOMED KG is loaded, check the FSN/domain of a few of these concept_ids (e.g. `722045009`, `281789004`) to confirm whether they're findings or substances, then decide whether `Medication` extractions should be scored against this dataset directly (if mapped to the finding/therapy concept) or need a separate ground-truth source (if the benchmark truly has no drug-substance-level labels). Don't assume either way without checking.
2. **The official metric is character-level Intersection-over-Union (macro and support-weighted), not precision/recall/F1.** `Evaluation_Criteria.md` specifies "(span, concept_id) pair" precision/recall/F1, which is a reasonable and arguably more informative metric for your own internal reporting, but it is not what the DrivenData leaderboard scores. If any claim compares your numbers to "other benchmark participants," it needs to run the actual `scoring.py` (linked from the benchmark's GitHub, `drivendataorg/snomed-ct-benchmark-runtime`) to get a like-for-like char-IoU score — your own F1 numbers and leaderboard IoU numbers are not directly comparable as-is.
3. **CORRECTED — test-set ground truth *is* present locally, for this dataset version.** Originally flagged as unavailable based on the live Benchmark page (whose hidden test set is only scored via code submission). The local `train_annotations.csv` actually carries an `annotation_type` column with `train` (51,295 annotations / 204 notes) and `test` (23,131 annotations / 68 notes) values, disjoint at the note level and matching the ~200/~70 split `Evaluation_Criteria.md` already assumed (204 + 68 = 272). So a genuine evaluation against the **original official challenge split** is possible directly from this file — no need to carve out your own split, and no need to caveat the dissertation the way this section originally suggested. There's also a third value, `proposed_ACCEPTED` (1,065 annotations / 233 notes) — supplementary post-challenge labels; decide explicitly whether to include these in scoring or hold them out for fidelity to the original split, and report it either way.
4. **Terminology clarification:** "Challenge" (this archived PhysioNet dataset, `physionet.org/content/snomed-ct-entity-challenge/1.2.0`, fully labeled) and "Benchmark" (the live, ongoing `drivendata.org/benchmarks/310`, hidden test set, pinned to the November 2025 SNOMED edition) are two different artifacts from the same underlying effort. Worth being precise about which one the dissertation is referencing — right now `Evaluation_Criteria.md` says "DrivenData SNOMED CT Entity Linking Challenge," which correctly names the dataset you actually have. If your Neo4j SNOMED ingestion uses a different SNOMED release than whatever version the Challenge's `concept_id`s were originally annotated against, double check concept IDs still line up — SNOMED concept IDs are generally stable across releases, but confirm rather than assume.
5. **Known data-loading gotcha:** `start`/`end` in `train_annotations.csv` are stored as floats (`"180.0"`), not ints — cast with `int(float(x))` before slicing note text.

None of this blocks using the dataset — it's a strong, real gold-standard source, the right choice, and (per the corrections above) actually more directly usable than originally assessed.

---

## 4. Gaps Against the Objectives (Priority Order)

Given the risk register in the proposal explicitly names "timeline slippage" as the top management risk and states the plan is to "prioritize the core active learning loop before secondary optimization features," the current state is a little inverted: preprocessing/normalization (secondary/plumbing) is polished, while the actual AL loop and neuro-symbolic grounding (the core, per your own risk mitigation plan) haven't started. Suggested build order to get back on the proposal's own stated priority:

1. **`mollm_ensemble.py` (Stage 3)** — this is Objective 2 in its entirety and the primary novelty claim of the thesis. Nothing else matters for the "trustworthy/grounded" story until this exists.
2. **KG ingestion scripts** (`init_memgraph_snomed.py`, `init_memgraph_guidelines.py`, `import_athena.py`, `build_concept_embeddings.py`) — Stage 3 can't query a graph that was never populated.
3. **HITL routing + write-back** (`ui/pages/2_HITL_Review_Queue.py`, Stage 4/5 logic) — this is Objective 3, and it's the mechanism that produces your "deflection rate" and "false deflection rate" metrics, which are your headline effort-reduction claims.
4. **`eval_suite.py` / `cal_eval.py` / `ablations.py`** — can't be built meaningfully until 1–3 exist, but the design in `Evaluation_Criteria.md` is ready to implement as soon as there's a pipeline to measure.
5. Fix the two normalization gaps (§3.4) before treating any Stage 2b output as ground truth for later stages.

---

## 5. Bottom Line

No evidence of scope creep away from the proposal — if anything the design docs are more rigorous than the proposal required (three-DB split, explicit provenance schema, self-flagged baseline confound). The risk is the opposite: **the implemented 40% is the least distinctive part of the pipeline** (NER + normalization is fairly standard engineering), while the parts that justify the thesis title — KG-grounded prompting, hallucination mitigation via citation verification, the active-learning cold-start solution, and the cost/effort evaluation — are currently documentation, not code. Given the proposal's own risk register names timeline slippage as the top risk and explicitly says to build the core AL loop first, recommend re-sequencing effort toward §4's priority order before polishing Stage 1/2 further.

---

## 6. KG 3 Write-Back: Triplets or KG Embeddings? (2026-08-07)

Confirmed plan: high-confidence outputs from GLiNER + SapBERT will be looped back to populate KG 3, which is intended to resolve Objective 4's "repurpose data into KG embedding pipelines" goal. Confirmed also that KG 2 (DuckDB) currently contains **only** static reference data — OMOP vocabulary and the abbreviation dictionary — no note-derived data has been written back anywhere yet. That's consistent with what the code shows (Memgraph ingestion scripts are still empty stubs), so this is a "here's the plan," not a "this already happened."

**Recommendation: triplets first, KGE as a derived layer on top — not instead of.** Concretely:

- **Store the pipeline's high-confidence extractions as actual graph triples/nodes in KG 3** (`:PatientObservation` –`INSTANCE_OF`→ `:Concept`, per `Databases.md`). This is not optional: it's the substrate that makes the provenance chain (`Provenance_Schema.md`), the HITL review, and the MoLLM Stage-3 "retrieve fact-grounded context" step possible at all. A dense embedding vector can't be shown to a human reviewer as "the evidence," and it can't be traced back to a source guideline sentence. If you skip triplets and go straight to embeddings, you lose the entire "white-box, deterministic, auditable" pitch that differentiates this from a black-box LLM — you'd just be trading one opaque representation for another.
- **Compute KG embeddings (CompGCN, per your own literature review, or TransE/RotatE as simpler baselines) as a periodic batch job over the accumulated triples**, not as the storage format itself. This is where the actual novel technical contribution lives — your background research section specifically cites fusing graph entity embeddings with pretrained LM output (BERT/SapBERT) to improve downstream NER, and that's exactly the mechanism Stage 5 describes ("KG 3 feeds back into GLiNER's prompt and search space"). Practical use for the embeddings once computed: re-ranking SapBERT's Tier-3 candidates by graph proximity, and giving the system a way to reason about entities that are graph-connected but textually dissimilar (the actual cold-start-mitigation mechanism, since a brand-new note-derived concept can still inherit context from its KG neighbors even with zero prior text-similarity signal).

**One risk to flag before you start writing back real extractions:** the "high confidence" gate that decides what enters KG 3 currently means "GLiNER's raw softmax ≥ 0.5" and "Tier-3 top-1 regardless of similarity" (see §7 below for concrete failures at that confidence bar). Seeding KG 3 from the pipeline's own unfiltered outputs before Stage 3 (MoLLM contradiction-check) and the 0.72 threshold gate (§3.4) exist risks a classic pseudo-labeling feedback loop: today's silent errors (e.g. `bioplar` → `Bourgvilain`) get written into KG 3 as if verified, then get retrieved as "grounding evidence" for future extractions, compounding rather than correcting the error. Recommend gating KG 3 write-back behind Stage 3 + the calibrated threshold, not behind Stage 2b confidence alone — write-back should be the last step in the pipeline, not something bootstrapped early to unblock Objective 4 on a shorter timeline.

---

## 7. Addendum — Empirical Evidence from Live Pipeline Runs (2026-08-07)

Two consecutive runs of `test_pipeline_e2e.py` on the same note (`10000032-DS-21`) surfaced concrete, reproducible evidence for the gaps predicted in §3.4.

**Correct behavior confirmed:** Tier-1 exact matches on well-formed clinical terms are reliable (`ascites`, `pain`, `PTSD`, `COPD` in run 2, `Pt`→Prothrombin Time). Character-offset reconciliation between expanded and original text is working as designed.

**Non-determinism — same input, different output across runs:**

| Entity | Run 1 | Run 2 |
|---|---|---|
| `ED` | `Ed District` (Tier 2) | `Erectile dysfunction` (Tier 1) |
| `ABD distension` | `Abdominal distension symptom` (Tier 3) | `Bisalbuminemia` (Tier 3) |
| `hiv` / `COPD` | abbreviation-form text | full expansion text |

Root cause (as of 2026-08-07): `normalize_entity()` in `normalization.py` had no `ORDER BY` tiebreaker on any tier — just `LIMIT 1`. When multiple OMOP rows share a lowercased `concept_name` (routine across a multi-vocabulary Athena dump) or Tier-3 cosine similarities are near-tied, DuckDB's row order under parallel/vectorized execution isn't stable, so `LIMIT 1` could return a different physical row on different runs. This directly undercuts the proposal's "deterministic, traceable, white-box" claim (Legal/Ethical section).

**Fixed 2026-08-08.** Every tier query (main `normalize_entity()` queries, `_tier_queries()`, `_lookup_tier12()`) now carries an explicit `ORDER BY concept_id ASC` (Tier 1/2) or `ORDER BY similarity DESC, concept_id ASC` (Tier 3) before `LIMIT`. Empirically re-confirmed 2026-08-11: after clearing stale `mollm_decisions` test rows, `test_stage3_live.py --store` was run twice back-to-back on the same note (`10000032-DS-21`); every entity — including `Worsening ABD distension and pain` and `Paracentesis`, which had earlier appeared to retrieve different rule counts across runs — returned identical `snomed_code` and identical retrieved rule IDs both times. That earlier discrepancy is now understood to have come from stale `is_test=TRUE` rows accumulating across sessions (no dedup or timestamp column on `mollm_decisions`, `mollm_call_id` is a random UUID), not from live non-determinism. See `docs/Stage3_Issue1_Rule_Backfill.md` for the full trace.

**Ungated Tier-3 fallbacks producing confidently wrong labels**, independent of the non-determinism above: `lasix` → `Laslades`, `spirnolactone` (note-text typo for spironolactone) → `SPIRILENE`, `bioplar` (typo for bipolar) → `Bourgvilain` (a nonsense concept). This is precisely the class of error Stage 3's contradiction-check is meant to intercept — but since Stage 3 doesn't exist yet, and Tier 3 has no minimum-similarity cutoff (§3.4), these currently flow through as if fully resolved with no signal they're weak. `lasix` failing Tier 1/2 despite being a standard RxNorm drug also suggests brand-name synonym coverage in `athena_concept_synonym` may be incomplete — worth checking independently of the threshold fix.

**Domain collision at Tier 2, not just Tier 3:** `ED` → `Ed District` is an exact-synonym match into a non-clinical concept. `Databases.md` scopes domain-filtering only to the Tier-3 fallback; this shows the same collision risk exists at Tier 2 and should be gated too.

**Minor:** near-duplicate entity text (`abd distension` / `ABD distension` / `abd  distension` with a double space) is deduplicated by exact string match in the e2e test script, so case/whitespace variants each trigger a separate, independent Tier-3 lookup for what is the same clinical fact — and, as shown above, can land on different concepts from each other. Canonicalizing whitespace/case before the dedup key would fix this. Also worth setting an explicit tokenizer `max_length` for GLiNER — the truncation warning in the run output means long discharge notes could silently get cut off with no error raised.

**Net effect for the dissertation:** these are good, concrete findings to include as a "known limitations, currently being addressed" subsection in the methodology chapter — they demonstrate exactly the kind of empirical rigor examiners look for, and all four are already understood/scoped (§3.4, this addendum) rather than surprises.
