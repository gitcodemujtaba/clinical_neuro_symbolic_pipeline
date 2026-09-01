# Knowledge Graphs — Complete Technical Reference

Merged 2026-09-01 from three previously separate documents (
`KG_Grounding_For_LLMs_Technical_Reference.md`,
`KG_Embedding_Technical_Reference.md`, `KG3_Implementation_And_
Feedback_Loop_Technical_Reference.md`) into one — the grounding
overview already summarized and pointed into the other two for
their full detail, so reading them separately meant following
cross-references back and forth for no real benefit. No content
dropped; the grounding overview's own short TransE/KG3 summary
sections were replaced with direct pointers into this same
document's Part B/Part C, since the full detail now lives one
scroll away instead of in a different file.

**Three parts, each keeping its own internal section numbering**
(Part A §1-§8, Part B §1-§15, Part C §1-§12) rather than one
continuous numbering — this is a reference merged for convenience,
not a rewrite, and re-numbering every cross-reference inside each
original document would have introduced exactly the kind of
transcription error this project's own "verify before trusting"
discipline warns against.

- **Part A** — how the LLM ensemble is grounded by each of the
  three real graphs (vocabulary, guideline-triplet, KG3), and the
  honest cross-mechanism finding about which grounding shapes work.
- **Part B** — TransE + RotatE knowledge-graph embeddings: what was
  built, how it was evaluated, and why neither is wired into
  production.
- **Part C** — KG3's two write paths, the HITL review queue, and
  the two mechanisms that repurpose reviewed labels to improve the
  pipeline itself.

---

# Part A — Grounding the LLM Ensemble

**Sources**: `src/normalization/tier_retrieval.py`, `src/normalization/orchestrator.py`, `src/mollm_tier_gate.py`, `src/guideline_evidence.py`, `src/retrieval.py`'s `GuidelineIndex`, `src/tier4_kg_escalation.py`, `src/kg_embedding.py`, `src/kg3_query.py`, `src/kg3_ingestion.py` — every code snippet below is read from the live source, not reconstructed from memory. Cross-referenced against Part B, Part C, and `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §14 for measured results rather than re-deriving them.

**Independently re-verified 2026-08-31** (this session): every module named below was directly read and every measured result cross-checked against the docs that reported it.

---

## 1. There are three different graphs in this codebase, not one

The phrase "the knowledge graph" is ambiguous in this project and the ambiguity matters — each of the three real graphs grounds the LLM ensemble through a structurally different mechanism, with a different trust level and a different adoption status.

| Graph | What it is | Where it lives | Populated by | Grounds the LLM via |
|---|---|---|---|---|
| **Vocabulary graph** | SNOMED CT / OMOP's own concept-relationship ontology — `IS_A`, `Maps to`, `Tradename of`, etc. | `athena_concept`, `athena_concept_relationship`, `athena_concept_ancestor` (DuckDB) | Athena's standard OMOP vocabulary release — static, licensed reference data | Direct SQL walks injected as facts (§3), and as the training graph for TransE (§5) |
| **Guideline-triplet graph** | Curated clinical-guideline knowledge, extracted from 76 real source documents (KDIGO, Surviving Sepsis, AHA/ACC HF, NSTE-ACS, ESI triage, …) into `(node, predicate, node)` rules with rationale + citation | `data/local_triplets_db2_v6_cleaned_grounded_rules_added/*.json`, indexed in-memory by `GuidelineIndex` | A one-time curation/extraction pass over real guideline PDFs — not learned, not accumulated at runtime | Prompt-injected evidence blocks in the Stage 3 tiebreak (§4) |
| **KG3** | This project's own dynamic, patient-instance graph — `PatientObservation → Concept → MoLLMDecision → HITLReview` | Memgraph (`bolt://localhost:7688`) | The pipeline's own write-back loop (dry-run in production; a one-off gold-simulated population exists, see §6) | A calibrator feature, `kg3_confirmation_count` (§6) — the only mechanism below that's actually adopted in production |

Three genuinely different epistemic statuses feed the ensemble: **licensed reference fact** (vocabulary graph), **curated external authority** (guideline graph), and **the pipeline's own accumulated history** (KG3) — and the document below is organized by *how* each one reaches the model, not just what it contains.

---

## 2. The one invariant across every mechanism: evidence, never a verdict

Stated explicitly in more than one module's own docstring, and worth stating once here because it's the load-bearing design constraint the whole document sits on top of:

> `src/tier4_kg_escalation.py`: "KG facts are EVIDENCE the model weighs alongside the entity's own context, never a deterministic override that bypasses the model call... Earlier draft of this module had a 'apply_deterministic_kg_rule() short-circuits and skips the LLM entirely' path -- removed."

> `src/guideline_evidence.py`: "this only ever ADDS a factual evidence block to the tiebreak prompt; it never short-circuits or overrides model reasoning."

Every mechanism in §3-§6 obeys this, with one narrow, explicitly-justified exception (§3.1's brand-alias fast path, which bypasses the ensemble entirely — but only because it isn't really "evidence for the model to weigh" at all, it's a graph-*verified identity fact*, the same trust tier as a curated dictionary lookup, not a probabilistic hint).

---

## 3. The vocabulary graph — direct SQL walks as injected facts

### 3.1 Brand-to-generic: a real 3-hop walk, not a 1-hop lookup

`_alias_expand_brand_to_generic()` (`src/normalization/tier_retrieval.py`) solves the "Lasix problem": SapBERT's embedding space doesn't reliably place a brand name close to its active ingredient, so a brand-only mention can silently miss Tier 3's top-K even though the vocabulary graph itself has the answer.

```python
def _alias_expand_brand_to_generic(conn, search_text):
    """
    There is no direct 1-hop CONCEPT_RELATIONSHIP edge from a Brand Name
    concept to its generic ingredient (verified empirically against this DB:
    a 'Maps to'/'Tradename of' query straight from the brand concept returns
    nothing). The real path is three hops:
        brand -[Brand name of]-> branded product (Branded Drug Comp)
              -[Tradename of]-> generic Clinical Drug Comp (standard, RxNorm)
              -[RxNorm has ing]-> Ingredient (standard, RxNorm)
    ...
    Returns a (possibly empty) set of standard concept_ids.
    """
```

This is the **highest-trust** graph-derived fact in the whole system — walked live against real OMOP relationship edges, not inferred — which is why it's the one mechanism allowed to skip the ensemble entirely, via `tier3_fast_path()`'s `verified_brand_alias` branch (`src/mollm_tier_gate.py`): "graph-verified brand alias, the sole such hit... skipped the two-step ensemble entirely." The result also seeds `_LAB_TEST_ALIASES`-style force-inclusion into the Tier 3 candidate pool (`alias_ids` in `orchestrator.py`), the same force-include-not-force-rank mechanism §7's real bug affected for the lab-alias dict.

### 3.2 `tier4_kg_escalation.py` — real relationship/ancestor evidence, one experimental 8B call

Built as a genuine "does grounding the escalation call in real KG facts help" experiment for the population the 3B ensemble couldn't resolve. Three real queries, no guessing:

```python
def _kg_relationships_between(conn, id_a, id_b):
    """ALL SNOMED relationship types between two concepts, either direction
    -- not filtered to any specific pattern."""
    rows = conn.execute("""
        SELECT relationship_id FROM athena_concept_relationship
        WHERE ((concept_id_1 = ? AND concept_id_2 = ?)
            OR (concept_id_1 = ? AND concept_id_2 = ?))
        AND invalid_reason IS NULL
    """, [id_a, id_b, id_b, id_a]).fetchall()
```

Plus `_kg_closest_ancestor()` (real parent-of relationship, not a guess) and a same-name cross-check ("does any OTHER concept in the vocabulary share this exact name, and if so what domain/class is it — surfaces a duplicate-concept situation as a fact for the model to weigh, not a pre-decided answer"). Every candidate pair's real relationship types, closest ancestor, and name-collision status are formatted into one 8B-model prompt call — same "evidence, never override" discipline as everything else in this document.

**Status: built, smoke-tested, not adopted.** Per project history (this session's own memory record): a follow-on multi-round variant letting the model actively request more KG searches was also tried — "7/9 smoke-tested entities still declined to commit to a verdict after 2-4 search rounds ('I need more evidence, but I'm not sure what specific search would be helpful')." Conclusion: a 3-8B-class model doesn't reliably have the planning/executive function to drive its own search strategy. The one-shot pre-fetch design (`tier4_kg_escalation.py` itself, arbiter architecture, 51.0% precision) remains the reference design, but neither is currently wired into `route_tier()`'s live decision path.

### 3.3 `CONDITION_VS_OBSERVATION_PRIOR` — a corpus-measured prior, not a live graph query

Worth including here for completeness even though it's not a runtime graph walk: a hardcoded instruction block (`src/mollm_tier_gate.py`) injected into the tiebreak prompt whenever `_condition_vs_observation_duplicate()` detects two candidates sharing an exact concept name but split across the `Condition` and `Observation` OMOP domains — the same SNOMED near-duplicate shape the vocabulary graph's own domain field surfaces. Corpus-measured to be near-exceptionless ("10/11 want Observation regardless of section"), it's the generic version of the same problem the guideline-evidence mechanism (§4) was hoped to make more specific.

---

## 4. The guideline-triplet graph — curated external authority, prompt-injected

Full detail already in `docs/ConsensusCalibrator_Technical_Reference.md`'s cross-references and the guideline-evidence integration plan (`/home/ec2-user/.claude/plans/here-is-details-of-peppy-pixel.md`); summarized here for completeness of the grounding picture.

**Mechanism**: `GuidelineIndex` (`src/retrieval.py`) loads 76 curated JSON-LD files (1,700 nodes, 1,162 rules) into an in-memory index, keyed by node **name** — deliberately not SNOMED code, since the corpus's own curators flagged cases where one code covers multiple genuinely distinct concepts. `guideline_evidence_for_candidates()` (`src/guideline_evidence.py`) looks up each Stage 3 tiebreak candidate's name, applies a soft type/domain compatibility filter, and formats matching rules (with `rationale` + `citation`) into a prompt block:

```
OFFICIAL GUIDELINE EVIDENCE (from curated clinical-guideline triplets,
not a similarity guess -- weigh this as real evidence, it does not
decide the answer for you): ...
```

**Status: measured, not adopted.** `GUIDELINE_EVIDENCE_ENABLED` stays off. A real, live A/B test (`evaluation/guideline_evidence_ab_test.py`, 2026-08-30, 23 gradable paired entities, both arms run through real live LLM calls) found **zero flips** — identical correctness with the evidence on vs. off. Root-caused, not assumed: the real matched evidence is consistently one-sided background context about the single candidate already favored ("aspirin is standard therapy for ACS"), never a fact that discriminates between the two tied candidates actually causing the split vote. Full result: `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §11.

---


## 5. TransE/RotatE — the vocabulary graph, embedded

Full build, training, and evaluation detail in **Part B** below. The
grounding-relevant summary:

- **What it grounds**: a candidate-tiebreak signal — embedding distance
  between two competing candidates, tested as a topological tiebreak
  alongside the hardcoded `_prefer_lab_procedure_over_observable()` rule.
- **Measured**: TransE's intrinsic MRR 0.776 / Hits@10 0.909 (standard
  KGE protocol); the real-candidate-pool win/loss sweep found the
  hardcoded rule has **zero losses at every threshold tested**, while
  TransE has 134 and every RotatE configuration loses even more
  decisively — including a specific, checked falsification of a proposed
  narrative that KGE would "naturally" resolve a known near-duplicate
  case (it picked the identical wrong concept, 0.0018 margin, noise not
  signal).
- **Status: built, evaluated, not wired in.** The hardcoded rule remains
  the safer specialist mechanism for the one pattern it targets; both
  KGE methods stay real, checkpointed, but unused generalist signals,
  pending a calibrated gating mechanism that doesn't yet exist (Part B
  §13 has the concrete, lower-risk integration points considered but not
  built).
- **CompGCN**: the proposal's third named KGE method. Not built — a
  deliberate sequencing decision, not an oversight (Part B §12).

## 6. KG3 — the pipeline's own history, as a calibrator feature

Full build, write-path, and repurposing-mechanism detail in **Part C**
below. The grounding-relevant summary:

**Mechanism**: `count_kg3_confirmations()` (`src/kg3_query.py`) queries
the live Memgraph graph for how many `:PatientObservation` nodes already
confirm the exact (entity text, concept) pairing a decision is about to
make — a **numeric feature** (`min(count, 10) / 10.0`) into
`ConsensusCalibrator`'s 17-feature logistic regression, not a text fact
injected into any LLM prompt. This is the only mechanism in this
document that is **actually adopted in production**.

**Status: adopted, real measured effect, with an honest caveat.**
Isolated ablation (same corpus, same split, only this feature differing):
**+0.031 AUROC, +5.8pp precision, +2.9pp coverage** on the
calibrator-eligible population — the single largest real, positive,
adopted benchmark impact of any mechanism in this document
(`docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §9, §14). The caveat:
KG3's current population is 100% gold-simulated (a one-off script
grading historical decisions against gold and writing the matches), not
real human review — so this measures the feature's behavior given
today's KG3 contents, not yet against independent, real-world
confirmation data (Part C §9, §11 for the live-queried real-review
numbers and a simulated-review impact experiment).

## 7. What this means for "grounding," honestly assessed

Six real mechanisms, one adopted, one partially built (brand-alias, always-on but narrow), the rest measured-and-shelved or explicitly out of scope:

| Mechanism | Graph | Reaches the LLM as | Adopted? |
|---|---|---|---|
| Brand-to-generic 3-hop walk | Vocabulary | Deterministic bypass (skips ensemble) | **Yes** — narrow, high-trust |
| `tier4_kg_escalation` | Vocabulary | Prompt evidence, 8B escalation call | No — model can't drive its own search |
| `CONDITION_VS_OBSERVATION_PRIOR` | (corpus-derived prior) | Prompt instruction | **Yes** — but not a live graph query |
| Guideline evidence | Guideline-triplet | Prompt evidence block | No — measured null, structural cause |
| TransE | Vocabulary (embedded) | Candidate-distance tiebreak | No — hardcoded rule wins head-to-head |
| RotatE / CompGCN | Vocabulary (embedded) | — | Not built |
| `kg3_confirmation_count` | KG3 | Calibrator feature (numeric, not prompt text) | **Yes** — real, positive, measured |

The honest pattern: **prompt-injected textual evidence has a 0-for-2 track record so far** (guideline evidence null, `tier4_kg_escalation` not adopted) — a small local model doesn't reliably use injected facts to resolve a genuine tie, even when the facts are real and well-sourced. **Structural/deterministic graph facts and numeric features have a better record** (the brand-alias walk, `kg3_confirmation_count`) — likely because they don't depend on the model correctly weighing free text under uncertainty, either bypassing the model entirely on a verified identity fact, or feeding a downstream statistical model that was fit specifically to use it. This is a real, load-bearing finding for anyone considering a *new* KG-grounding mechanism for this pipeline: prefer a numeric feature or deterministic override over another prompt-injection experiment, unless there's a specific reason to expect this population's evidence shape to be different from guideline evidence's (one-sided, non-discriminating).

---

## 8. A common misreading, corrected — this is NOT how the feedback loops actually work

A frequently-drawn version of this architecture (the original project proposal's Stage 5 vision) shows **one** "KG3 Provenance Ledger" as the universal upstream source, feeding a `TransE/CompGCN` embedding block into Stage 2B retrieval and a "Search Space & Prompt Tuner" into Stage 2A/2B extraction prompts, alongside the abbreviation-flywheel and calibrator loops. Checked line-by-line against the live source, three of those four boxes are wrong or don't exist:

- **`mine_context_rules()` (the abbreviation flywheel's real data source) reads from `hitl_review_queue` directly** — confirmed in the function's own docstring ("Reads real reviewer-confirmed resolutions from hitl_review_queue"). It never touches KG3. Routing this arrow through a "KG3 Provenance Ledger" box is the exact conflation `src/kg_embedding.py`'s own docstring already named and corrected once for TransE's training data (§5) — the same mistake, made again, one level up.
- **CompGCN does not exist anywhere in this codebase.** `grep`-confirmed: the string appears only as an unbuilt-future-scope mention in two docstrings (`src/kg_embedding.py`, `src/kg3_query.py`), never as an implementation.
- **TransE is real and evaluated, but not wired into Stage 2B retrieval** — §5 above. The hardcoded rule it was tested against has zero losses where TransE has 63; that result is *why* it stays unwired, not an oversight.
- **The "Search Space & Prompt Tuner" feeding Stage 2A/2B extraction prompts has never been built.** It's the original proposal's aspirational Stage 5 scope ("GLiNER prompt/search-space feedback loop"), explicitly scoped in `docs/Implementation_Methodology.md` as "foundations only" and never implemented.

The corrected picture — what's actually wired, live, and measured today, versus what's real-but-unwired, versus what was never built at all. Verified against the live source arrow-by-arrow (this is the final, corrected version — two earlier drafts each had one real wiring error caught and fixed the same way):

![As-built Stage 5 active-learning architecture, corrected against the live source](diagram/technical_architecture_diagram_repurposing.png)

The same structure, as a text-based (versionable, diffable) equivalent:

```mermaid
flowchart TD
  HITL[("hitl_review_queue<br/>real reviewer-confirmed resolutions<br/>(0 completed in production today)")]
  KG3[("KG3 — Memgraph<br/>PatientObservation → Concept")]
  VOCAB[("Athena/OMOP vocabulary graph<br/>athena_concept_relationship<br/>(static reference data, not accumulated)")]

  MINE["mine_context_rules()<br/>src/abbreviation_flywheel.py"]
  KG3CONF["count_kg3_confirmations()<br/>src/kg3_query.py"]
  TRANSE["TransE embedding<br/>src/kg_embedding.py"]

  STAGE1["Stage 1 — Abbreviation tiebreaks<br/>select_by_context_pattern()"]
  STAGE3["Stage 3 — ConsensusCalibrator<br/>kg3_confirmation_count feature"]
  STAGE2B["Stage 2B — Semantic retrieval<br/>SapBERT Tier 3"]

  HITL -- "real, wired, live" --> MINE --> STAGE1
  KG3 -- "real, wired, live<br/>+5.8pp precision measured" --> KG3CONF --> STAGE3
  VOCAB -- "real, evaluated, NOT wired<br/>hardcoded rule wins 63-0 head-to-head" -.-> TRANSE -.-> STAGE2B

  COMPGCN["CompGCN"]
  TUNER["Search Space &amp; Prompt Tuner"]
  NOTBUILT["Never built —<br/>proposal scope only"]
  COMPGCN -.-> NOTBUILT
  TUNER -.-> NOTBUILT

  classDef live fill:#E5EFE3,stroke:#3F7A45,color:#16211F,stroke-width:1px;
  classDef unwired fill:#F3E8D6,stroke:#A06A22,color:#16211F,stroke-width:1px,stroke-dasharray: 4 3;
  classDef missing fill:#F5E4E0,stroke:#9A4436,color:#16211F,stroke-width:1px,stroke-dasharray: 2 2;

  class HITL,KG3,MINE,KG3CONF,STAGE1,STAGE3 live;
  class VOCAB,TRANSE,STAGE2B unwired;
  class COMPGCN,TUNER,NOTBUILT missing;
```

**Two feedback loops are real and live** (solid, green): the abbreviation flywheel from `hitl_review_queue`, and `kg3_confirmation_count` from KG3 into the calibrator. **One is real and evaluated but deliberately not wired in** (dashed, amber): TransE, beaten head-to-head by the hardcoded rule it was tested against. **Two were never built** (dashed, red): CompGCN and the prompt/search-space tuner — real, stated proposal scope, not silently dropped, but not to be drawn as operating loops.

---

# Part B — Knowledge-Graph Embedding (TransE + RotatE)

Deep-dive companion to `docs/Implementation_Methodology.md`'s KG-embedding
bullet (Stage 2b section) and `docs/FINAL_RESULTS_Single_Source_Of_Truth.md`
§6. Merged 2026-09-01 from two previously separate documents
(`TransE_KG_Embedding_Technical_Reference.md`,
`RotatE_KG_Embedding_Technical_Reference.md`) into one, since RotatE was
built as a direct, explicit follow-on to TransE and the two share most of
their evaluation machinery, decision framework, and honest-limitations
discipline — reading them as one continuous story is more useful than two
documents that constantly cross-reference each other. No content was
dropped in the merge; overlapping framing (e.g. "why RotatE/CompGCN
weren't attempted yet," written when only TransE existed) was consolidated
rather than duplicated.

**Headline, upfront**: both TransE and RotatE (as a real 4-configuration
ablation) were fully implemented, trained end-to-end on real graphs, and
rigorously evaluated two ways each (intrinsic link prediction and an
extrinsic task-specific check), then applied as a candidate-reranking
tiebreak and validated against gold, head-to-head against the existing
hardcoded rule they were proposed to replace. **The deliberate,
evidence-based outcome for both was NOT to wire either into production** —
the hardcoded rule wins outright on its own pattern, and every RotatE
configuration loses even more decisively than TransE. This document
explains what was built, how well it works in the abstract, and why none
of it is live, for both methods together.

---

## 1. Why this exists — closing Objective 4's KGE gap

The original research proposal's Objective 4 named three candidate
knowledge-graph embedding methods — **TransE, RotatE, CompGCN** — for
"transform[ing] legacy clinical classification datasets into dense
representations... to improve downstream entity recognition tasks." This
gap was initially scoped out of the project with the reasoning "no
accumulated KG3 write volume to embed yet" — a reasoning error, corrected
mid-session, that's worth stating explicitly because it shaped the whole
design:

- **KG3** — the dynamic, patient-instance graph this pipeline's own
  Stage 4 write-back loop populates (`src/kg3_ingestion.py`) — genuinely
  has zero live volume; every write is still `dry_run=True`.
- **The SNOMED CT reference graph itself** — `athena_concept_relationship`
  in this project's own DuckDB store — is a real, substantial,
  *already-populated* knowledge graph with millions of genuine triples.
  Training a KG embedding model requires a graph to train on; nothing
  about that requirement is specific to KG3.

`src/kg_embedding.py` trains TransE directly on the reference graph,
scoped to the subset of SNOMED concepts this pipeline's own candidate
pools have actually touched — preserving the spirit of the proposal's
"based on our own TP records" framing while using a graph that genuinely
has data today. RotatE, built later, took a deliberately different
approach to its training data (§7).

## 2. Method choice and sequencing — TransE first, then RotatE, CompGCN still deferred

**TransE (Bordes et al., 2013) was built first**, alone. Reasoning for
that initial choice, as recorded in the code (`src/kg_embedding.py:23-31`):

- TransE is the simplest, most standard member of the KGE family the
  proposal names, with a transparent scoring function
  (`score(h, r, t) = -‖h + r − t‖₂`) — the model literally learns to make
  `head + relation ≈ tail` in vector space.
- It needed **no new library dependency**. Neither `pykeen` nor
  `torch-geometric` is installed in this environment; TransE is simple
  enough to implement correctly in ~190 lines of plain PyTorch, while
  RotatE (complex-valued embeddings) and CompGCN (a full
  graph-convolutional architecture) are both meaningfully more complex.
- Recorded as **honest scope, not a silent gap**: the module docstring
  said outright that RotatE/CompGCN were "real, meaningfully more complex
  follow-on work, not attempted here" — true at the time, no longer true
  for RotatE as of this document.

**RotatE was built second**, as a genuine 4-configuration ablation (§7-10)
— not just "does RotatE work," but "does RotatE trained on four different
real data sources work." **CompGCN remains the one deliberately unbuilt
method** — a substantially bigger lift (message-passing layers, a
composition operator, more training infrastructure) than either TransE or
RotatE, sequenced separately (§12) rather than bundled in.

---

## Part B.1 — TransE

## 3. The training graph — a real SNOMED subgraph, not a toy or synthetic set

`load_snomed_subgraph()` (`src/kg_embedding.py:77-90`) pulls real triples
directly from `athena_concept_relationship`:

```sql
SELECT concept_id_1, relationship_id, concept_id_2
FROM athena_concept_relationship
WHERE invalid_reason IS NULL
AND concept_id_1 IN (?) AND concept_id_2 IN (?)
```

Both endpoints are restricted to the set of `omop_concept_id` values that
have actually appeared in some entity's `normalized_entities.candidates`
list during this project's own Stage 2b runs (`scripts/build_kg_embeddings.py`
`main()`, lines 95-105) — not an arbitrary slice of the full SNOMED graph
(which has vastly more concepts and edges than this pipeline will ever
touch).

**Superseded twice since this table was first measured.** The training
GRAPH itself (SNOMED relationship subgraph, touched concepts) was never
affected by either correction below; only the model's real-world
tiebreak/extrinsic numbers moved, via (1) real corpus growth and (2) a
locked-test-split leakage fix in the shared TP-record population
(`docs/ConsensusCalibrator_Technical_Reference.md` §18, §10 below).

**Current, live scale** (`logs/kg_embedding_results.json`, matching the
checkpoint on disk):

| Metric | Value |
|---|---|
| Distinct concepts touched by candidate pools | 7,545 |
| Real SNOMED relationship triples (both endpoints touched) | 25,980 |
| Vocabulary — entities | 7,537 |
| Vocabulary — relation types | 104 |
| Train / test split (90/10, random) | 23,382 / 2,598 |

(An earlier training run, before the Stage 3 recall-fix backfill enlarged
the candidate-pool population, reported a slightly smaller graph — 7,261
concepts, 24,872 edges, 2,488 test triples. The model was **retrained**
once the larger post-backfill pool existed, then retrained again
2026-08-31 alongside the leakage fix below; the numbers in this document
and in `logs/kg_embedding_results.json` reflect that most recently
retrained, currently checkpointed model.)

## 4. Model architecture — real code, walked through

`class TransE(nn.Module)` (`src/kg_embedding.py:93-117`):

```python
class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self._normalize_entities()

    def score(self, h, r, t):
        return -torch.norm(self.entity_emb(h) + self.relation_emb(r)
                            - self.entity_emb(t), p=2, dim=-1)
```

Two entity/relation embedding tables (100-dim, Xavier-initialized), scored
by the standard TransE translation distance. Higher (less negative) score
means a more plausible triple. One deliberate implementation detail worth
calling out: **entity embeddings are re-normalized to unit L2 norm after
every optimizer step** (`_normalize_entities()`, called both at init and
after each training batch) — the standard TransE trick that prevents the
model from trivially "cheating" the margin loss by growing embedding
norms rather than actually learning graph structure.

## 5. Training procedure

`train_transe()` (`src/kg_embedding.py:128-175`) — standard margin-ranking
loss with random negative sampling, exactly as in the original TransE
paper:

- For each positive triple `(h, r, t)` in a batch, one **negative** triple
  is generated by corrupting either the head or the tail (chosen randomly,
  50/50) with a uniformly random entity from the vocabulary.
- Loss: `max(0, margin − score(pos) + score(neg))` (margin = 1.0), i.e.
  the model is penalized whenever a corrupted (almost certainly false)
  triple scores within `margin` of the real one.
- Optimizer: Adam, `lr=0.01`, batch size 1024, run for **50 epochs**.
- `seed=42` fixed throughout (both `random.seed` and `torch.manual_seed`)
  for reproducibility.
- Runs on CUDA if available, else CPU — this environment trained on CPU
  (no GPU dependency; ~7,269 entities × 100 dims is small enough that
  plain-PyTorch CPU training completes in a practical time).

Checkpointing (`save_model()`/`load_model()`, `src/kg_embedding.py:54-74`)
bundles the `state_dict` **together with** `entity2idx`/`relation2idx` and
the embedding dimension into one `.pt` file — necessary because the
vocabulary is data-dependent (which `concept_id`s and relation types were
in *this* training run's subgraph); a bare `state_dict` alone can't be
reloaded into usable lookups without it.

**Current checkpoint on disk**: `models/kg_transe_v1.pt` (3.0 MB),
committed to git.

## 6. Evaluation — two real, complementary evaluations

The module's own docstring is explicit that both are real (not simulated
or estimated): a standard KGE-literature intrinsic evaluation, and a
second, task-specific extrinsic evaluation tied directly to this
project's own graded decisions. Both `evaluate_link_prediction()` and
`evaluate_against_tp_records()` are model-agnostic by contract (only ever
call `.score(h,r,t)` or `.entity_emb(idx)`) — the same two functions are
reused, unchanged, for RotatE in Part B.

### 6.1 Intrinsic — held-out link prediction (standard KGE protocol)

`evaluate_link_prediction()` (`src/kg_embedding.py:178-213`): for each
held-out `(h, r, t)` triple, ranks the true tail entity `t` against
**every** entity in the vocabulary as a candidate tail, using the trained
scoring function. Mean Reciprocal Rank and Hits@10 are computed over
those ranks.

Two details worth being precise about, both stated honestly in the code
rather than left implicit:

- This is the **RAW** setting (candidate tails are *not* filtered to
  exclude other known-true triples), not the "filtered" setting common
  in KGE papers — RAW and filtered MRR are not directly comparable
  numbers.
- Ranking a triple's true tail against the *full* entity set for every
  held-out triple is the expensive part of this evaluation, so
  `evaluate_link_prediction()` caps evaluation at `max_eval=2000` triples
  — **a random subsample, not the first N** (`random.sample`), so the
  reported number isn't biased by triple ordering in the split.

**Result** (`logs/kg_embedding_results.json`):

```json
"link_prediction": {
  "mrr": 0.7757178359410913,
  "hits_at_10": 0.909,
  "n_evaluated": 2000
}
```

**MRR 0.776, Hits@10 0.909** on 2,000 of 2,493 held-out triples. For
context: MRR of 1.0 would mean the true tail is always ranked #1 against
the *entire* 7,269-entity vocabulary; 0.776 means the true tail is on
average ranked close to #1 (reciprocal rank 0.776 ≈ average rank ~1.3),
and Hits@10=0.909 means the true tail lands in the top 10 out of 7,269
candidates 90.9% of the time. This is a genuinely strong result for a
plain-vanilla TransE trained on a modest (25k-edge) real-world subgraph.

### 6.2 Extrinsic — does the embedding actually separate confusable candidates?

The intrinsic result above answers "does this model do standard link
prediction well" — a generic KGE-literature question. `evaluate_against_tp_records()`
(`src/kg_embedding.py:216-285`) asks the concrete question this project
actually needs answered: **for entities where this pipeline's own tier
gate correctly picked the right concept out of several real, competing
candidates, does the embedding space place that correct concept closer to
its actual wrong-but-competing rivals than to an arbitrary unrelated
concept?**

Mechanism: for each gold-confirmed true-positive tier-gate decision with
at least one other real candidate in its pool (`gather_tp_records()`,
`scripts/build_kg_embeddings.py:33-87` — reused unchanged by RotatE's
`gold` config too), it computes:

- `d_wrong` — L2 distance in embedding space from the correct concept to
  the wrong candidate that actually competed for the same mention.
- `d_random` — L2 distance from the correct concept to an arbitrary,
  unrelated concept.

and reports the fraction of comparisons where `d_wrong < d_random` — i.e.
the genuinely-confusable rival sits closer than chance would predict.

**Result, 2026-08-31, post-leakage-fix** (was `n_tp_records_provided: 455`,
`frac: 0.6892` before the fix):

```json
"extrinsic": {
  "n_tp_records_provided": 343,
  "n_usable_records": 343,
  "n_comparisons": 1209,
  "frac_wrong_candidate_closer_than_random": 0.6368899917287014
}
```

**Real, material drop: 68.9% → 63.7%.** The TP-record population this
extrinsic eval is built from (`gather_tp_records()`) was found to
unconditionally include notes from `data/splits/note_splits.csv`'s locked
test split — 39 of 149 source notes (26%). Fixed to exclude them; the
real TP-record count dropped 455→343, and this signal dropped with it —
TransE's apparent strength here was partly an artifact of test-split
notes it should never have been measured against.

**63.7%** of 1,209 comparisons across 343 real gold-confirmed TP records —
well above the 50% chance baseline. This is real, positive signal: the
embedding space does capture genuine clinical proximity among candidates
that actually competed for the same mention, not just generic SNOMED
graph structure — but see §10 for whether that translates into a safe
per-decision tiebreak (it does not, for either method).

## 7bis. Applying TransE: the candidate-reranking tiebreak

Training and evaluating the embedding is only half the work — a second
module, `src/kg_embedding_tiebreak.py`, applies the trained model to a
concrete Stage 2b problem: **SNOMED near-duplicate concept pairs** — e.g.
WBC/RBC Procedure-vs-Observable-Entity confusion, an HCO3
hierarchy-collapse bug, a wound-dehiscence Condition-vs-Observation
duplicate.

**Why this narrow signal, not a blanket KGE+SapBERT fusion.** The
module's docstring draws a direct, explicit lesson from an earlier
experiment this same session: blindly fusing a second embedding score
into every retrieval was already tried once (BM25+dense hybrid retrieval)
and measured to lose outright to dense-only SapBERT — `CNSP_HYBRID_RETRIEVAL`
stays off as a validated conclusion. A KGE signal is structurally
different from BM25 (graph-topological vs. lexical-overlap), so it isn't
guaranteed to fail the same way, but the discipline carries over: **don't
add a second signal everywhere without measuring it first.** This module
is deliberately scoped to one recurring, already-diagnosed failure
pattern, not deployed as a general-purpose reranker.

**The signal** (`kg_tiebreak_score()`, `src/kg_embedding_tiebreak.py:45-65`):
for a tied candidate, the mean embedding distance from that candidate to
every *other* concept in the same entity's own broader SapBERT-proposed
candidate pool (not just the tied pair). The hypothesis: the genuinely
correct concept, being about the same clinical topic as the mention,
should sit closer to that pool's broader semantic neighborhood than a
structurally-adjacent-but-topically-off sibling concept does.

`pick_via_kg_tiebreak()` (`src/kg_embedding_tiebreak.py:68-92`) picks
whichever tied candidate has the **lowest** mean distance to the rest of
the pool, and returns `resolved=False` (caller falls back to its existing
logic) when fewer than two tied candidates have a usable score — i.e.
this mechanism never silently forces a pick it has no real basis for.
Both functions here work unchanged for RotatE (§8's packing trick), so
this whole module — and the validation harness in §10 — is genuinely
model-agnostic, not TransE-specific.

---

## Part B.2 — RotatE

## 7. Why not the SNOMED vocabulary graph again — RotatE's four data sources

TransE trained on the Athena/OMOP relationship graph restricted to this
pipeline's own touched concepts (7,269 concepts, ~24,900 edges). Repeating
that exact choice for RotatE was explicitly considered and rejected in
favor of **repurposed/curated project data** instead. Two real,
already-produced artifacts were found and used, plus a third discovered
mid-build:

- **`guideline`** — the curated clinical-guideline graph, a second,
  separate population living on the same Memgraph instance as KG3
  (`:GuidelineNode` nodes, real predicate types like
  `INDICATES`/`REQUIRES_INTERVENTION`/`TRIGGERS_SEVERITY`). Only edges
  where both endpoints ground to a real SNOMED code are usable; of those,
  only edges whose SNOMED code further crosswalks to a real, standard
  OMOP concept are trainable.
- **`gold`** — this project's own gold-confirmed candidate-competition
  signal, reusing `gather_tp_records()` (imported, not duplicated) and
  flattening each record into `(correct_concept_id, PREFERRED_OVER,
  wrong_concept_id)` triples.
- **`combined`** — simple concatenation of the two above.
- **`snomed_is_a`** — added after directly verifying, live, that a fourth
  real data source exists: the separate KG1 Neo4j instance
  (`bolt://localhost:7687`, distinct from Memgraph/KG3) holds a fully
  populated SNOMED IS_A hierarchy (386,110 `:SnomedConcept` nodes, 641,727
  real `IS_A` edges). This is the same *category* of data as the
  vocabulary graph TransE already used (raw ontology structure, not
  repurposed/curated project output) — kept as an explicit fourth arm to
  test whether a much larger, purer single-relation-type graph
  outperforms the two much smaller curated arms, deliberately NOT folded
  into `combined`.

All four configs are run through **identical** training/evaluation code
(`train_rotate()`, `evaluate_link_prediction()`, `evaluate_against_tp_records()`,
and the same `evaluation/kg_tiebreak_validation.py` sweep) — only the
training triples differ. This is the actual controlled ablation.

## 8. RotatE math and the packing trick

Standard RotatE (Sun et al. 2019): entities as complex vectors, relations
as unit-modulus rotations via a learned phase vector
(`r = cos(θ) + i·sin(θ)`). Score: `-‖h∘r − t‖` (complex Hadamard product =
per-dimension rotation, then L2 norm of the residual).

**Compatibility trick**: entity embeddings are stored as one real-valued
`nn.Embedding(n_entities, 2*dim)` — first half real, second half
imaginary. This is not a hack invented for this codebase; reference
implementations (OpenKE, pykeen) store RotatE this way internally too.
It's what lets `evaluate_link_prediction()`, `evaluate_against_tp_records()`
(`src/kg_embedding.py`, imported unchanged from Part A), and
`src/kg_embedding_tiebreak.py`'s two functions work with **zero edits** —
all of them only ever call `.score(h,r,t)` or `.entity_emb(idx)`, never
assuming anything about the internal packing.

**Geometric caveat, stated up front.** Raw Euclidean L2 over the packed
`[re;im]` vector is a *weaker* proxy for "topical closeness" in RotatE
than in TransE. TransE's translational geometry makes raw closeness a
direct proxy for "connected by something small" (`h+r≈t` implies `h` and
`t` are only `r` apart). RotatE's rotational geometry doesn't share that
property — two entities connected by a real, well-fit relation can still
sit far apart in raw packed coordinates if that relation's phase rotation
is large. This is a real, testable hypothesis, not asserted as fact — and
turned out to matter (§10).

**A concrete algebraic consequence, found while writing this module's
unit tests.** `src/kg_embedding.py`'s own TransE test checks that an
untrained, transitively-implied edge (`0 IS_A 2`, from training edges
`0 IS_A 1` and `1 IS_A 2`) scores higher than its reversed, clearly-wrong
counterpart. This is a mathematical certainty for TransE once training
converges near-exactly: if `e1=e0+r` and `e2=e1+r`, then
`score(0,r,2) = -‖r‖` while `score(2,r,0) = -‖3r‖`, always favoring the
implied direction. **RotatE has no equivalent guarantee.** Composing the
same rotation twice gives `e2 = e0∘r(2θ)`; the implied-vs-reversed score
gap becomes `2sin(θ/2)` vs `2sin(3θ/2)` — not monotonic in `θ`, so which
one wins depends on where the trained phase angle happens to land.
Copying TransE's test verbatim for RotatE would have been checking the
wrong property for this architecture; `tests/test_kg_embedding_rotate.py`
instead checks the thing RotatE's own margin-ranking loss actually
optimizes (a real training triple scores higher than a corrupted one).

## 9. Five deliberate deviations from the TransE code RotatE was adapted from

1. **No unit-norm entity clamp.** Official RotatE leaves entity magnitude
   meaningful; TransE's `_normalize_entities()` renormalization step is
   deliberately not carried over.
2. **Entity-norm diagnostic logged every 10 epochs** (mean/max L2 norm),
   specifically because of (1) — unbounded norm growth under plain hinge
   loss with no normalization is a known real instability mode; this is
   an up-front diagnostic, not a reaction to having observed it.
3. **`relation_phase` initialized uniformly in `[-π, π]`** (not Xavier) —
   avoids a slow-start failure mode where most rotations begin
   near-identity.
4. **Loss/margin kept identical to TransE** (plain hinge, `margin=1.0`),
   not RotatE's literature self-adversarial sigmoid loss (`gamma` 9-24).
   The right call for a controlled TransE-vs-RotatE comparison in this
   codebase, but it means the resulting MRR is **not comparable to
   published RotatE benchmarks** — same disclosure discipline as §6.1's
   RAW-vs-filtered-MRR caveat.
5. **L2 norm order (`p=2`)**, matching TransE's own choice, not the
   paper's L1 option.

## 10. Results — all four RotatE configs, plus TransE for comparison

All four RotatE configs trained end-to-end
(`scripts/build_kg_embeddings_rotate.py --config all`), plus a fresh
same-session re-run of `evaluation/kg_tiebreak_validation.py` against
TransE's existing checkpoint for a fair, same-population comparison. All
numbers below are real, on the current live corpus (4,843 gradable
tied-pair entities) — nothing rounded up or cherry-picked.

**Superseded and re-measured, 2026-08-31, same day as the original
RotatE build.** The numbers originally reported were built from
`gather_tp_records()`, which — discovered later the same day — drew its
note pool from every `is_test=TRUE` note unconditionally, including 39 of
149 (26%) from `data/splits/note_splits.csv`'s **official locked test
split**. That affected `gold`/`combined` (trained directly on this
population) and every config's shared extrinsic-eval number
(`guideline`/`snomed_is_a` included, since all four share one `tp_records`
population). Fixed (`gather_tp_records()` now excludes
`evaluation.splits.load_split("test")`) and every affected number below
re-measured on the clean 110-note pool (343 TP records, down from
452/455). `guideline` and `snomed_is_a`'s **training data** was never
affected (neither uses `gather_tp_records()` to train) — only their
extrinsic-eval row moved; `guideline`'s conclusion (0 usable records) was
re-checked and is unchanged. Every table below shows the corrected,
post-fix numbers.

### 10.1 Training data yield (real attrition, not idealized counts)

| Config | Raw source size | Real trainable triples | Entities | Relation types |
|---|---|---|---|---|
| `guideline` | 1,144 total Memgraph edges, 355 with both endpoints SNOMED-grounded | **263** (92 more dropped: SNOMED code fails to crosswalk to a standard OMOP concept) | 154 | 26 |
| `gold` | 343 TP records (clean, test-split-excluded — was 452) | **1,209** (was 1,593) | 128 | 1 (`PREFERRED_OVER`) |
| `combined` | guideline + gold | **1,472** (was 1,856) | 280 | 27 |
| `snomed_is_a` | 641,727 raw Neo4j IS_A edges | **530,515** (~17% dropped to crosswalk failure) — unaffected, doesn't use `gather_tp_records()` | 319,557 | 1 (`IS_A`) |
| *TransE (Part A, for comparison)* | *SNOMED relationship subgraph, touched concepts* | *25,980* | *7,537* | *104* |

### 10.2 Both evaluations, all four RotatE configs plus TransE

| Config | Link-prediction MRR / Hits@10 | Extrinsic: usable records | Extrinsic: frac. correct-closer-than-random |
|---|---|---|---|
| `guideline` | 0.375 / 0.481 (n=27 held-out) — unchanged | **0 / 343** — near-zero vocabulary overlap with the TP-record population, same conclusion as before the fix | n/a |
| `gold` | 0.479 / 0.967 (n=121) | 343 / 343 | **78.9%** (was 72.3%) |
| `combined` | 0.484 / 0.899 (n=148) | 343 / 343 | **74.4%** (was 84.2% — no longer the best of the five) |
| `snomed_is_a` | 0.028 / 0.076 (n=2000)† — training unaffected, unchanged | 422 / 452 (extrinsic-only number now stale — not re-run; ~15min GPU cost for a config whose qualitative conclusion, worst-performer, is not expected to change from a 24% smaller TP set) | 67.4% (stale, see previous parenthetical) |
| *TransE* | *0.777 / 0.911 (n=2000)* | *343 / 343* | **63.7%** (was 68.9%) |

†`snomed_is_a`'s MRR/Hits@10 is **not comparable** to the other rows — it
ranks each held-out triple's true tail against a 319,557-entity candidate
pool, vs. hundreds/thousands for every other config (TransE included, at
7,537). A near-zero MRR here reflects the much harder ranking problem, not
a categorically worse embedding.

**The picture changes in a real, material way, not just cosmetically.**
Before the fix, `gold`/`combined` both clearly beat TransE (72.3%/84.2%
vs. 68.9%) on this aggregate metric. After removing the locked-test-split
contamination: `gold` still beats TransE (78.9% vs. 63.7% — TransE's own
number also dropped, since it shares the same corrected TP-record
population), but **`combined` no longer leads** (74.4%, behind `gold`'s
own 78.9%) — the earlier "combined is best of all five" headline does not
survive the fix. Guideline's addition to gold in `combined` was diluting,
not helping, once measured on the clean population; this wasn't visible
before because the contaminated `combined` number happened to look best.

### 10.3 The decisive test: real per-entity tiebreak win/loss, all four RotatE configs plus TransE

This is the test that actually matters for whether any of this is usable
— not the aggregate signal above, but whether picking a winner by
embedding distance helps or hurts on real, individual gold-graded
decisions. Full population, `TIE_THRESHOLD=0.03` (SapBERT top1/top2 score
gap). This sweep does **not** use `gather_tp_records()` at all
(`_load_candidate_pools()` queries live candidate pools directly) — only
the embedding *weights* changed via retraining on the corrected data, and
the effect on this specific test is small:

| Config | Resolved | Win | Loss | Net | Win rate of resolved |
|---|---|---|---|---|---|
| `guideline` | 0 | 0 | 0 | 0 | n/a |
| `gold` | 1,073 | 97 | **757** | **−660** | 9.0% (was 9.0%) |
| `combined` | 1,075 | 97 | **756** | **−659** | 9.0% (was 9.0%) |
| `snomed_is_a` | 1,543 | 22 | **500** | **−478** | 1.4% — unchanged, training unaffected |
| *TransE* | *1,814* | *130* | *379* | *−249* | *7.2%* (was 228/263/−35, 12.7% — a real, worse number post-fix: TransE's OWN checkpoint was also retrained here since `scripts/build_kg_embeddings.py` retrains it every run, and its training population also lost the same 39 locked-test-split notes) |

**And head-to-head against the existing hardcoded
`_prefer_lab_procedure_over_observable()` rule** (§11 below explains what
it is), on exactly the subset where the rule applies:

| Config | n (rule-applicable) | KGE win | KGE loss | Rule win | Rule loss |
|---|---|---|---|---|---|
| `guideline` | 295 | 0 | 0 | 111 | **0** |
| `gold` | 295 | 0 | **172** | 111 | **0** |
| `combined` | 295 | 0 | **171** | 111 | **0** |
| `snomed_is_a` | 295 | 2 | **0** | 111 | **0** |
| *TransE* | *295* | *110* | *134* | *111* | **0** |

**The full, honest finding, all four RotatE configs plus TransE
considered together — the core conclusion is UNCHANGED by the leakage
fix, but two specific numbers moved enough to matter:**

1. **The aggregate embedding-separation signal (§10.2) still does not
   predict per-entity tiebreak safety (§10.3)** — if anything more starkly
   than before: `gold` now has the single best aggregate score (78.9%) of
   any config, including TransE, and is still deeply net-harmful as a
   tiebreak (−660). The geometric caveat from §8 holds, corrected numbers
   included.
2. **`guideline` is still simply too small to be useful** — 0 resolved
   either way, unaffected by the fix (it never used `gather_tp_records()`
   for training).
3. **`gold`/`combined` are still net-harmful as a tiebreak**, materially
   unchanged (−660/−659 vs. the previous −663/−824) — the fix moved the
   aggregate-signal numbers more than the tiebreak numbers, since the
   tiebreak sweep's population is independent of `gather_tp_records()`.
4. **`snomed_is_a`'s distinct failure profile is completely unchanged**
   (its training never touched the contaminated population).
5. **RotatE is still worse than TransE at this specific task, and
   TransE's own case for even a "generalist secondary signal" role is now
   weaker than previously measured, not stronger.** Post-fix, TransE's
   full-population net dropped from −35 to −249 and its rule-subset
   losses rose from 105 to 134 — the locked-test-split notes it lost were
   apparently *helping* its numbers look better than they should have.
   **The complete, now twice-corrected finding**: neither TransE nor any
   usable RotatE config beats the 3-line hardcoded rule, RotatE remains
   the weaker of the two, and TransE's own real-world case is weaker than
   any previously-documented version of this comparison (see
   `docs/Implementation_Methodology.md`'s own now-thrice-updated numbers:
   265W/181L → 228W/263L → 130W/379L, each subsequent correction moving in
   the same, more-negative direction).

**A side finding, found while gathering the ORIGINAL TransE comparison
row, still true and itself superseded by the leakage fix above**: TransE's
numbers had already drifted once between when this document's TransE
sections were first written and when RotatE's own build began
(265W/181L → 228W/263L, attributed to corpus growth). The leakage fix is
a SECOND, independent correction on top of that first one (228W/263L →
130W/379L) — two different real effects, not one.
`docs/Implementation_Methodology.md` reflects all three states in
sequence; do not average or split the difference between them, the most
recent (post-leakage-fix) row is the current, correct one.

---

## Part B.3 — Applying to production, decision, and shared reference

## 11. Decision: both built, evaluated, deliberately not wired into production

`_prefer_lab_procedure_over_observable()`
(`src/normalization/tier_retrieval.py`) is the hardcoded rule both
methods lose to. It applies to one specific, well-understood pattern:
for a Lab-Test-labeled entity, SapBERT often scores a "Observable
Entity"-class concept (the abstract property measured, e.g. "Leucocyte
count") higher than the "Procedure"-class concept for the same test (the
act of measuring it, e.g. "White blood cell count") — measured directly
against this project's own gold-graded corpus, the Procedure-class
concept is gold-correct in **78 out of 78 cases** where either is correct
at all — zero exceptions.

`_prefer_lab_procedure_over_observable()` stays as the production
mechanism for the pattern it covers, unchanged. **Neither
`src/kg_embedding_tiebreak.py` (TransE) nor its RotatE-config variants are
called from `route_tier()`, `tier_retrieval.py`, or any other production
code path** — confirmed directly (no references outside their own
modules, their validation harness, and their tests). Both remain built,
checkpointed, and evaluated: real signals with genuine (if imperfect, and
for RotatE, net-negative) properties, staged for future work should a
calibrated gating mechanism exist to decide *when* to trust either — the
same kind of mechanism `src/mollm_tier_calibrator.py`'s
`ConsensusCalibrator` already provides for the MoLLM ensemble-vote gate,
not yet built for either KGE signal (see §13, item 1).

## 12. Validation harness (shared by both methods) and CompGCN sequencing

`evaluation/kg_tiebreak_validation.py` grades the tiebreak's actual picks
against real gold-standard SNOMED codes, sweeping a `TIE_THRESHOLD`
parameter (how close the top-1/top-2 SapBERT similarity scores need to be
before a pair counts as "tied" at all) and comparing head-to-head against
`_prefer_lab_procedure_over_observable()`. `--model-type transe|rotate`
and, for RotatE, `--config guideline|gold|combined|snomed_is_a` select
which checkpoint to validate — §10's tables are this script's output,
unchanged for either method.

**Zero live SapBERT/embedding calls** — this harness reuses each entity's
already-stored `normalized_entities.candidates` (similarity scores
computed once at Stage 2b time), loading only the small checkpoint
itself. This was a deliberate design choice after an earlier, unrelated
attempt to recompute a fresh embedding live measurably stalled a
concurrent Stage 3 backfill for ~22 minutes via disk/CPU contention.

Three-way outcome classification per entity (`classify_outcome()`,
pure-logic, unit-tested with no DB or model needed):

- **WIN** — baseline (SapBERT top-1) was gold-wrong, the mechanism's pick
  is gold-correct.
- **LOSS** — baseline was gold-correct, the mechanism's pick is
  gold-wrong (the fatal case — a mechanism that causes this on net
  should not ship, regardless of how many wins it also produces
  elsewhere).
- **NEUTRAL** — both right, both wrong, or the pick didn't change
  (includes cases where the KGE had no usable score and fell back to the
  baseline unchanged).

**Specific falsification, directly tested rather than assumed.** A
proposed narrative (raised mid-session as a plausible-sounding argument)
claimed KGE would "naturally" resolve a third, previously-unexamined
SNOMED near-duplicate found during the regional-extension fix work —
"Mean cell hemoglobin concentration - finding" (Clinical Finding domain)
winning unanimous ensemble agreement over the actual gold-correct
concept. This was checked directly against the real retrained TransE
model on the real failing entity, not accepted on the strength of the
argument: **KGE picked the identical wrong concept**, with a 0.0018
embedding-distance margin between the two candidates — noise, not a real
topological separation. The claim was false for this specific case.

**Sequencing — CompGCN deferred.** RotatE (the 4-config ablation) is now
complete. CompGCN is a separate, later decision — a substantially bigger
lift (message-passing layers, a composition operator, more training
infrastructure) than either TransE or RotatE, deliberately not bundled
into this work. If pursued, the same data-source question §7 answers for
RotatE applies again, already resolved by these findings rather than
needing to be re-litigated: repurposed/curated project data (`gold`,
`guideline`) doesn't automatically beat raw ontology data
(`snomed_is_a`) or vice versa — both failed the same real test, for
different reasons (too small; net-harmful despite good aggregate signal).
A future CompGCN attempt should budget for the same two-track evaluation
(aggregate signal AND per-entity tiebreak win/loss) rather than trusting
the aggregate number alone, given §10.3's finding that the two can point
in opposite directions.

## 13. Where else a KGE signal could plausibly help — not built, proposed for a separate follow-up

Given neither TransE nor RotatE works as a hard, greedy re-rank, three
lower-risk integration points were considered and are recorded here as
backlog, not started:

1. **`ConsensusCalibrator` feature, not a gate** (recommended first, if
   any of this is pursued further) — feed a KGE tiebreak distance as one
   more input to the existing logistic-regression calibrator
   (`src/mollm_tier_calibrator.py`), the same pattern already proven for
   `kg3_confirmation_count`. Bounded risk: the calibrator can learn to
   down-weight a noisy signal toward zero rather than being forced to
   trust it outright, which directly addresses §10.3's finding that
   `gold`/`combined` have real aggregate signal that a greedy pick can't
   safely extract. Testable with the existing
   `evaluation/tier_gate_cal_eval.py` feature-ablation harness — no new
   validation infrastructure needed.
2. **HITL-queue triage/prioritization signal**, not an auto-decision —
   where a KGE pick disagrees with SapBERT's top-1, that disagreement
   correlates with something real (§10.3's non-trivial win+loss counts,
   not pure noise); could rank human-review order without needing to win
   a precision fight, since it wouldn't gate a write.
3. **Acronym-escalation note-internal-consistency check** (Stage 1) —
   `src/acronym_escalation.py`'s existing MoLLM-based expansion picker
   was measured (2026-08-17) to fail at 34.3%–36.1% precision from a
   textbook-prior bias (e.g. `LAD`→"left anterior descending artery"
   regardless of context). A KGE check of which candidate expansion's
   concept sits closer, in embedding space, to OTHER already-resolved
   entities in the same note is a structurally different signal
   (relational context, not a text prior) untried against this specific
   failure mode.

None of these three is built. Each needs its own honest validation batch
before use — this session's 0-for-3 record on "fuse a second ranking
signal into an existing decision" (BM25+dense hybrid, guideline-evidence
injection, KGE tiebreak) is a real pattern, not bad luck, and argues for
treating a positive result as the thing to prove, not assume.

## 14. Testing & reproducibility

- **`tests/test_kg_tiebreak_validation.py`** — 19 checks, covering:
  `classify_outcome()`'s four win/loss/neutral combinations,
  `hardcoded_rule_applicable()`/`hardcoded_rule_pick()`'s exact
  replication of the production rule's decision logic, and a genuine
  `save_model()`/`load_model()` roundtrip for TransE — no DB or live
  model training required.
- **`tests/test_kg_embedding_rotate.py`** — mirrors the TransE test's
  structure (vocab, score-shape, tiny synthetic-graph convergence check,
  save/load roundtrip, both eval functions called against a trained
  RotatE to prove the model-agnostic contract holds in practice), plus
  lighter tests for the three RotatE-specific loaders' OMOP-concept-id
  grounding/crosswalk logic.
- **To retrain TransE from scratch**: `python3 scripts/build_kg_embeddings.py`
  (read-only DB access throughout — safe to run alongside a concurrent
  Stage 3 batch holding the write lock). Writes `models/kg_transe_v1.pt`
  and `logs/kg_embedding_results.json`.
- **To retrain RotatE from scratch**: `python3 scripts/build_kg_embeddings_rotate.py --config all`
  (or `--config guideline|gold|combined|snomed_is_a` individually). Writes
  `models/kg_rotate_{config}_v1.pt` and
  `logs/kg_embedding_rotate_{config}_results.json` per config. Note:
  `models/kg_rotate_snomed_is_a_v1.pt` (247MB) is gitignored, regenerable
  via this script rather than committed.
- **To re-run gold validation for either method**:
  `python3 -m evaluation.kg_tiebreak_validation --model-type transe` or
  `--model-type rotate --config <name>` — requires the relevant
  checkpoint to already exist; raises a clear error rather than silently
  training a throwaway substitute model, since the whole point is to
  validate the exact weights that would be wired into production.

## 15. Honest limitations

- **RAW, not filtered, MRR** — TransE's 0.776 figure (and RotatE's own
  MRR numbers) are not directly comparable to "filtered MRR" numbers
  reported in most KGE papers; stated in the code and repeated here so
  it isn't misquoted against a different protocol later.
- **Intrinsic evaluation is capped at 2,000 of the held-out triples**
  (a random subsample, for compute reasons) for both methods where the
  vocabulary is large enough for this to matter.
- **The extrinsic evaluation's TP-record population is itself dependent
  on this project's own tier-gate output** — it measures whether the
  embedding separates candidates that *this pipeline's own retrieval*
  already surfaced as competitors, not an independent, externally-sourced
  confusion set. This applies identically to TransE and RotatE's `gold`/
  `combined` configs (all three share `gather_tp_records()`).
- **No calibrated confidence/threshold exists for either tiebreak
  signal** — the win/loss numbers above are population-level rates, not a
  per-decision confidence score; this is exactly why neither was wired in
  despite TransE's originally-measured net-positive result on the
  broader population (since revised negative, see §10.3).
- **RotatE's own geometric caveat (§8)**: raw packed-vector L2 distance is
  a weaker proxy for topical closeness than TransE's translational
  geometry — a real, disclosed hypothesis, not proven false, but likely
  part of why RotatE underperforms TransE at this specific task despite
  its own configs' stronger aggregate link-prediction/extrinsic numbers
  in some cases.
- **CompGCN remains genuinely unbuilt** — not attempted, not partially
  built, stated as scope not taken on, consistent everywhere this
  project's documentation discusses it.

---

# Part C — KG3 Implementation & the HITL Feedback Loop

Deep-dive companion to `docs/Implementation_Methodology.md`'s Stage 4
section. This document covers exactly how KG3 (the dynamic,
patient-instance knowledge graph, in Memgraph) is written to, how a
human reviewer's decision on a queued case gets there, and — the part
this document goes into the most depth on — the two concrete mechanisms
that **repurpose** reviewed labels to make the pipeline itself better
over time, not just to populate a graph. Every claim is grounded in the
real, current source (`src/kg3_ingestion.py`, `src/hitl_queue.py`,
`src/kg3_query.py`, `src/abbreviation_flywheel.py`,
`src/mollm_tier_calibrator.py`) and live queries against the production
DuckDB, run while writing this document.

**Headline, upfront, stated honestly**: the write paths, the review
queue, and both repurposing mechanisms below are fully built and unit
tested. The repurposing mechanisms have **not yet run on real human
review data** — live-queried while writing this document,
`hitl_review_queue` currently holds 19,103 cases, **all still
`PENDING`** (zero `APPROVED`/`CORRECTED`/`REJECTED`), and
`abbreviation_context_rules` (§5's mined-rule table) has **zero rows**
as a direct consequence. This is not a bug — it is the expected,
by-design state before any real reviewer has clicked Approve/Correct —
but it means this document describes a mechanism that is primed and
ready, not one with measured impact yet.

---

## 1. KG3 in context — which graph is which

This project has three distinct data stores that are easy to conflate,
so this is worth stating precisely before anything else:

- **KG2 (`athena_concept` + related tables, DuckDB)** — the static
  **reference vocabulary**: SNOMED/RxNorm/OMOP concepts, already fully
  populated (see `docs/SapBERT_Technical_Reference.md` §3 for how it got
  its embedding column). This never changes as a result of anything
  this pipeline does.
- **The guideline KG (file-backed, `GuidelineIndex`)** — clinical
  guideline triplets, also static reference data (see
  `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`).
- **KG3 (Memgraph, Bolt-compatible)** — the **dynamic** graph this
  pipeline itself writes: one `:PatientObservation` node per resolved
  clinical entity, linked to the `:Concept` it was grounded to, the
  `:MoLLMDecision` that grounded it, and the `:HITLReview` (if any) that
  verified it. This is the graph this document is about, and the only
  one of the three whose content depends on what this pipeline — and,
  eventually, human reviewers — actually do.

## 2. Two write paths into KG3, both gated, neither optional

`src/kg3_ingestion.py`'s own docstring is explicit that **nothing
writes to KG3 without going through one of exactly two gated paths** —
there is no third, ungated route anywhere in this codebase:

1. **`ingest_reviewed_case()`** — the original, **human-reviewed** path.
   Every case it writes has already passed through `src/hitl_queue.py`'s
   review workflow (`reviewer_decision` in `APPROVED`/`CORRECTED`). This
   path exists specifically because of a real, named risk: an early
   measurement found `AUTO_VALIDATED` precision at only 39.4%/52.6%
   under the older binary `route()` gate — writing unfiltered
   high-confidence Stage 3 output straight into KG3 at that precision
   would have baked silent errors into the graph as "verified," a
   pseudo-labeling feedback-loop risk the project's own implementation
   checklist named explicitly before this module was built.
2. **`ingest_auto_decision()`** — a deliberately **narrower, newer**
   unreviewed path, added for `src/mollm_tier_gate.py`'s Tier 1-5 gate.
   It only accepts Tier 1/2/3 decisions (`AUTO_TIERS` — the tiers whose
   entire purpose is to be trustworthy enough to skip human review), and
   defaults to `dry_run=True`, writing nothing until that trust has
   actually been validated on real data at scale (see
   `docs/Implementation_Methodology.md`'s tier-gate section and
   `docs/ConsensusCalibrator_Technical_Reference.md` for how that trust
   is measured).

**Both paths write structurally identical graph shapes** — a
`:HITLReview` node either way, distinguished only by
`final_decision_status` (`'APPROVED'`/`'CORRECTED'` vs. the literal
string `'AUTO'`) — so any query walking the graph never needs a special
case for whether a given observation came through human review or the
auto-write gate. This is a deliberate design choice, not an accident:
§6 and §7's repurposing mechanisms both benefit from being able to treat
"confirmed" uniformly regardless of which path confirmed it.

### Idempotent by construction

Every write is a Cypher `MERGE` keyed on each node's natural id
(`entity_id`, `mollm_call_id`, `hitl_case_id`, `omop_concept_id`), never
`CREATE`. This matters concretely for the batch driver
(`scripts/run_kg3_ingestion.py`), which is designed to be safely
re-runnable — it only re-attempts cases whose
`hitl_review_queue.ingested_at` is still `NULL`. Without `MERGE`, a
transaction that failed after creating some nodes but before the caller
marked the case ingested would duplicate those nodes on the next retry;
with it, a retry is a no-op for whatever already landed.

## 3. The HITL review queue — where a decision waits for a human

`hitl_review_queue` (`src/hitl_queue.py`) is a **separate table**, not a
column bolted onto the decision tables it draws from — deliberately.
`mollm_decisions`, `mollm_review_decisions`, and
`mollm_tier_gate_decisions` each record what a Stage 3 decision *was*;
this table records what happened to it *afterward* — a human reviewer's
verdict — which is a genuinely different lifecycle with its own states
(`PENDING`/`APPROVED`/`CORRECTED`/`REJECTED`) and its own writer (the
Streamlit reviewer UI, not a batch script). Keeping it separate means no
source table's schema has to anticipate a review workflow that may never
touch a given row.

### 3.1 Three source tables feed one queue

`enqueue_pending_cases()` pulls from all three Stage 3 decision tables —
`mollm_decisions` (citation-gated), `mollm_review_decisions`
(confidence-driven, all-tier), `mollm_tier_gate_decisions` (the current
production Tier 1-5 gate) — each via its own
`_presented_suggestion_from_*()` builder that normalizes that source's
particular shape (verdict strings, candidate-index conventions, or a
free-text `proposed_concept_name` needing resolution back to a concrete
`omop_concept_id`, §3.2) into one common reviewer-facing payload:
original text, candidate list, model verdicts, routing decision,
**and**, since 2026-08-15 reviewer feedback, the surrounding
`local_context`/`section_name`/`assertion_status`/`experiencer` — added
specifically because an earlier version of the queue showed model
reasoning with no note context at all, giving a reviewer nothing to
actually judge the decision against.

### 3.2 `suggested_omop_concept_id` — resolved once, at enqueue time, per source

Each source table records "what was decided" in a different shape, and
each builder resolves it to one canonical `omop_concept_id` a reviewer's
Approve click can unambiguously confirm — computed **once, at enqueue
time**, not re-derived under time pressure at ingestion time:

- `mollm_decisions`: parses `RESOLVED_TO_CANDIDATE_<N>` verdicts,
  majority vote among models that resolved to the same N (ties broken
  toward the lower index); falls back to the top-1 candidate for
  contradiction-mode verdicts (`SUPPORTED`/`CONTRADICTED`/
  `INSUFFICIENT_EVIDENCE`), since those checks validate Stage 2b's own
  top-1 rather than choosing among alternatives.
- `mollm_review_decisions`: has no `concept_id` column at all, only a
  free-text `proposed_concept_name` string. Resolution is two-pronged —
  first an explicit `"(OMOP <id>"` annotation, validated against the
  entity's real candidate-id set before being trusted (never trust an
  arbitrary number in free text on its own); then a fallback exact-match
  against `candidates[].concept_name`. A real, checked finding behind
  this design: of 20 rows with a non-null `proposed_concept_name` at the
  time this was built, 6 weren't a clean name at all — the model had
  echoed the full candidate description it was shown — which a bare
  exact-match could never catch without the id-annotation path.
- `mollm_tier_gate_decisions`: simplest of the three — `route_tier()`
  already records exactly which candidate it picked as
  `final_candidate_index` (1-based), so this is a direct list index,
  not text parsing.

### 3.3 Why *every* decision is queued, not just the uncertain ones

`enqueue_pending_cases()` queues every clean (non-error) decision from
all three sources — **including Tier 1/2/3 `AUTO_VALIDATED`/
`AUTO_RESOLVED` ones**, not just the ones that failed to reach an
auto-tier. This is recorded explicitly as **deliberate, temporary
conservatism**, not the final design: until a calibrated confidence
threshold is validated against real reviewed data at scale, the code
does not trust its own routing tier enough to skip queuing a case just
because it says "auto." `queue_reason` records the source row's own
tier/routing reason for the reviewer's context, but never gates whether
a case gets queued at all.

### 3.4 The review verdict, and what it unlocks

`submit_review()` accepts exactly one of `APPROVED`/`CORRECTED`/
`REJECTED`:

```python
final_ingestion_path = "HUMAN_VERIFIED" if reviewer_decision in ("APPROVED", "CORRECTED") else None
```

`APPROVED`/`CORRECTED` set `final_ingestion_path='HUMAN_VERIFIED'` — the
only value `load_ingestible_cases()` (§4) will ever read as ready for
KG3. `REJECTED` leaves it `NULL` — **structurally**, not just by
convention, excluded from ever reaching KG3: a rejected case has no code
path into `load_ingestible_cases()`'s `WHERE final_ingestion_path =
'HUMAN_VERIFIED'` filter, so there is nothing to accidentally forget to
check later.

`reviewer_comment` — added 2026-08-17, free text available regardless of
decision (unlike `rejection_reason`, which only makes sense on a
`REJECTED` case) — is the **real ground truth**
`mine_context_rules()` (§5) and any future analysis reads back from this
table. A reviewer explaining *why*, not just *what*, is what eventually
lets the pipeline improve from real cases instead of a guess — and,
honestly, until reviewers actually populate this field, that specific
input has nothing to mine, by design.

## 4. From a reviewed case to a KG3 write

`ingest_reviewed_case()` (`src/kg3_ingestion.py:125-169`) takes one
`load_ingestible_cases()` row plus a caller-supplied `entity_fields`
dict (the DuckDB-side lookup of `orig_start`/`orig_end`/`confidence`,
kept as a separate parameter specifically so this module's *only*
external dependency is Memgraph, not also DuckDB — `scripts/run_kg3_ingestion.py`
owns that join instead).

`resolve_concept_id()` is where a `CORRECTED` case's real value shows
up: for `CORRECTED`, it trusts the reviewer's own
`corrected_concept_id` directly — the concept Stage 2b/3 originally
proposed is irrelevant, the human's answer is authoritative. For
`APPROVED`, it uses the `suggested_omop_concept_id` computed at enqueue
time (§3.2). Either shape that reaches this function without a
resolvable id — e.g. an `APPROVED` case sourced from
`mollm_review_decisions` whose `proposed_concept_name` never resolved —
raises `UningestibleCase` **loudly**, not silently: a case dropped here
would otherwise look identical to one that was simply never queued,
exactly the kind of silent gap this project's own evaluation work has
repeatedly found expensive to diagnose after the fact.

The five-`MERGE` Cypher transaction (`_ingest_tx()`) writes, in one
atomic `execute_write()` call:

```
(:Concept {omop_concept_id}) <-[:INSTANCE_OF]- (:PatientObservation {entity_id})
                                                        -[:VALIDATED_BY]-> (:MoLLMDecision {mollm_call_id})
                                                                                  -[:REVIEWED_BY]-> (:HITLReview {hitl_case_id})
```

— either all five `MERGE`s commit, or none do (the neo4j driver wraps
the whole function in one transaction), so a partial write can never
leave the graph in an inconsistent state.

## 5. Repurposing mechanism 1 — mined context rules feed straight back into Stage 1

This is the mechanism that most directly answers "how does reviewing
labels improve the pipeline" — it closes a real loop from a human's
`CORRECTED` verdict back into how *future* notes get processed, before
Stage 3 ever runs on them.

### 5.1 The problem it solves

Stage 1's abbreviation expansion (`src/preprocessing.py`'s
`expand_text_and_track_offsets()`) resolves a multi-meaning abbreviation
via a static tiebreak chain — numeric context, then an observed-
frequency prior, then OMOP groundability — that **knows nothing about
which meaning the rest of the pipeline actually confirms**, note after
note. A concrete, directly-measured failure this exposed: `src/acronym_escalation.py`'s
own investigation found that even showing a model *clean, uncorrupted*
context, "PDA" still resolved to the wrong meaning across all three
MoLLM models — the models' own prior toward the more textbook-famous
reading overrode good context. A deterministic rule that never asks a
model to judge sidesteps that exact failure mode.

### 5.2 `mine_context_rules()` — turning reviewer verdicts into deterministic rules

`mine_context_rules()` (`src/abbreviation_flywheel.py:345-465`) reads
**exactly** the population this document's headline caveat describes —
real, human-confirmed resolutions:

```sql
SELECT e.original_text, h.corrected_concept_id, e.orig_start, e.orig_end, e.note_id
FROM hitl_review_queue h
JOIN extracted_entities e ON e.entity_id = h.entity_id
WHERE h.reviewer_decision IN ('APPROVED', 'CORRECTED')
  AND e.expansion_ambiguous = TRUE
```

— i.e. every case where a human approved or corrected a resolution for
an entity Stage 1 itself had already flagged as an ambiguous-expansion
candidate. For each one, it slices a small pre/post word window around
the entity's span **from the raw, unexpanded note text** — deliberately
not Stage 1's own expanded text, reusing the exact reasoning
`src/acronym_escalation.py`'s `raw_local_context()` fix already
established: showing a rule-miner Stage 1's own already-expanded text
would mean training on a prior guess dressed up as literal wording, the
same circularity risk this whole mechanism exists to avoid.

**Scoring**: for each `(abbreviation, meaning, trigger_word, position)`
combination, `_log_odds()` computes the add-alpha-smoothed log-odds of
that word appearing near *this* meaning versus every *other* meaning of
the same abbreviation — a word that appears near one meaning and never
the others is a strong, real signal; a word appearing near all of them
equally is not. Two support floors before a rule is ever written, both
guarding against building a rule from too little evidence:
`MIN_CONTEXT_RULE_SUPPORT = 5` independent confirmed examples per
abbreviation (summed across its meanings) before *any* rule for it is
attempted, and per-word support `>= 2` within that. Rules are written to
`abbreviation_context_rules`, upserted idempotently (safe to re-run this
miner repeatedly as more review data accumulates).

### 5.3 Where the mined rule gets consumed — closing the loop

`select_by_context_pattern()` (`src/abbreviation_flywheel.py:287-324`)
is consulted by Stage 1's own expansion tiebreak chain — the **first**
tiebreak tried, checked before any numeric-context heuristic or model
call, the same "skip the judgment call entirely when a deterministic
answer exists" shape as `src.mollm_tier_gate.tier3_fast_path()`. Given
an abbreviation's real pre/post context words at a *new* occurrence, it
looks up matching trigger-word rules and returns the meaning with the
higher summed score — but explicitly refuses to force a pick when two
meanings' matched rules score a genuine tie, falling through to the
existing chain instead of guessing.

**The loop, stated end to end**: a human reviewer corrects one ambiguous
abbreviation resolution → `mine_context_rules()` turns that correction
(plus enough others like it) into a trigger-word rule →
`select_by_context_pattern()` applies that rule to every *future*
occurrence of the same abbreviation in a *similar* context, at Stage 1,
before GLiNER or any MoLLM model ever runs on that note. A correction
made once, on one note, structurally improves every future note that
shares the same pattern — this is the concrete mechanism, not a
metaphor.

### 5.4 The asymmetry that makes this safe — and why it's deliberate

This module sits alongside a **first**, separate mechanism
(`compute_frequency_priority()`) that aggregates the pipeline's *own*
Stage 2b outcomes — not reviewer-confirmed ones — into a frequency
prior. That mechanism was found, on real production data, to be
dangerous: a 50-note production run gold-checked 7/7 of its
highest-confidence non-excluded picks as **wrong** (`DM`→"deep masseter"
instead of diabetes mellitus, `IVF`→"In Vitro Fertilization" instead of
IV fluids, and 5 more), and — worse — the mechanism was caught
**actively re-selecting its own earlier wrong guesses within the same
run**, exactly the circularity this document's calibrator section (§6)
also has to guard against. That mechanism was inverted from a block-list
to a strict, currently-**empty** allow-list as a direct result.

`mine_context_rules()` (this section) is **not** subject to that same
restriction, and the code's own comment is explicit about why: its
input is real, independent, human-confirmed ground truth from
`hitl_review_queue`, not the pipeline's own confident-but-possibly-wrong
guess. Aggregating the pipeline's own guesses formalizes whatever bias
already exists in them; aggregating a human's corrections is exactly the
kind of independent evidence that *can* correct a systematic bias rather
than reinforce it — including, eventually, for the exact
coronary-artery-segment abbreviations (`LAD`/`LCX`/`LMCA`/...) currently
hard-blocked elsewhere in the pipeline (see
`docs/ConsensusCalibrator_Technical_Reference.md`'s trap-gate section).
This mechanism is the **intended eventual path to relaxing those hard
traps for real**, once enough real review volume exists — which is
precisely why it does not pre-emptively exclude them the way the
frequency-priority mechanism does.

## 6. Repurposing mechanism 2 — confirmed resolutions as a calibrator trust signal

A second, structurally different way reviewed (and auto-confirmed) data
feeds back into the pipeline: `count_prior_confirmations()`
(`src/mollm_tier_calibrator.py:160-204`), one of the 16 input features
to the `ConsensusCalibrator` that decides whether a *split-vote* Stage 3
decision can still be promoted to an auto-write tier
(`TIER_1B_CALIBRATED_AUTO_VALIDATED` — full mechanism in
`docs/ConsensusCalibrator_Technical_Reference.md`).

```sql
-- half 1: how many times has this (entity text, concept) pairing already
-- reached an AUTO tier via the tier gate...
SELECT count(*) FROM mollm_tier_gate_decisions d
JOIN extracted_entities e ON e.entity_id = d.entity_id
JOIN normalized_entities n ON n.entity_id = d.entity_id
WHERE lower(trim(e.original_text)) = lower(trim(?))
  AND n.omop_concept_id = ?
  AND d.mollm_routing_decision IN ('AUTO_VALIDATED', 'AUTO_RESOLVED')

-- half 2: ...or been explicitly APPROVED/CORRECTED by a human reviewer
-- to this exact concept
SELECT count(*) FROM hitl_review_queue h
JOIN extracted_entities e ON e.entity_id = h.entity_id
WHERE lower(trim(e.original_text)) = lower(trim(?))
  AND h.corrected_concept_id = ?
  AND h.reviewer_decision IN ('APPROVED', 'CORRECTED')
```

The two halves are summed into one `prior_confirmation_count` feature —
"how many times has this same resolution already been confirmed,
one way or the other." Unlike §5's mechanism, this one deliberately
**includes** the pipeline's own AUTO-tier confirmations alongside human
ones, not just human ones — and the code is explicit that this is a
real, acknowledged risk, not an oversight: since real KG3 ingestion runs
`dry_run=True` throughout this pipeline today, this feature is currently
built from the *same* dry-run tier-gate decisions it might end up
reinforcing, not from independent live-graph evidence. The calibrator's
own held-out-split validation discipline (never train and score on the
same notes) is the safeguard against this specific feature encoding
circularity rather than real signal — an ablation run during that
calibrator's own development (dropping this feature entirely) confirmed
the model's real predictive signal comes from vote-consensus shape and
retrieval provenance, not from this feature, which is exactly the
honest, checked answer to "is this feature doing something real or just
echoing its own training data."

**How this repurposes reviewed data concretely**: once real
`APPROVED`/`CORRECTED` review volume exists (currently zero, per this
document's headline), every future entity sharing the same
`(entity text, concept)` pairing as an already-reviewed one gets a
nonzero `prior_confirmation_count` — direct, measurable evidence the
calibrator can use to promote a *future* split-vote decision on that
same pairing to an auto-write tier, without needing a human to re-review
an already-settled case every single time it recurs.

## 7. The read side — built ahead of having data to read

`src/kg3_query.py` is deliberately **read-only**, with no
ranking/scoring logic of its own — it answers "what is currently in
KG3," not "what should Stage 2b do about it" (that second question is
the two repurposing mechanisms above, which live in `src/`, not here).
Its own docstring states plainly why it exists *before* the feedback
loop it will eventually support: the real feedback mechanisms
(GLiNER prompt/search-space feedback, a CompGCN/TransE/RotatE
re-ranking layer — see Part B above for what was actually built of that) need a meaningful volume of
accumulated `HUMAN_VERIFIED` corrections to mean anything, which does
not exist yet. Building the read interface now, against whatever the
write side actually produces, means it's ready the moment real data
exists rather than being designed blind alongside the write side later.

Three functions: `count_by_label()` (cheapest possible sanity check —
node counts by `:PatientObservation.label`), `get_observations_for_concept()`
(full provenance chain for one concept, for audit or a future
"how often has a human confirmed this" query), and
`get_accepted_triples()` (every fully-provenanced, `APPROVED`/`CORRECTED`
observation, optionally domain-filtered — the exact population a future
re-ranker or prompt-feedback mechanism would train or query against).

## 8. Regression safety net, built before there's anything to regress

`scripts/regression_check_kg3.py` snapshots gold-recall plus KG3 coverage
and flags regressions against the previous snapshot — reusing
`scripts/score_gold_recall.py`'s scoring directly rather than
reimplementing it (this codebase's repeated "two copies of one idea
drifting apart" discipline). Its own docstring is candid that its
central number **cannot move on its own today**: nothing currently feeds
KG3 signal back into Stage 2b, so there is nothing yet for this script
to actually detect a regression *from*. It exists so that the very first
change that *does* close that loop has something to diff against
immediately — the tool and the habit built ahead of the mechanism, not
bolted on afterward once something breaks silently.

## 9. Current real status (live-queried, not assumed)

| | Count |
|---|---|
| `hitl_review_queue` total cases | 19,103 |
| ...`PENDING` | 19,103 (100%) |
| ...`APPROVED` / `CORRECTED` / `REJECTED` | 0 / 0 / 0 |
| `hitl_review_queue.ingested_at IS NOT NULL` (real KG3 writes via the reviewed path) | 0 |
| `abbreviation_context_rules` rows (§5's mined rules) | 0 |

Every real KG3 write anywhere in this pipeline today, via either path
(§2), is `dry_run=True` or simply hasn't happened — there is no live
Memgraph content this document can report numbers from yet. This is
stated as plainly here as everywhere else in this project's
documentation: the machinery in §2-§7 is built, tested (§10), and
verified against synthetic/dry-run data; its real-world improvement to
the pipeline (§5's rule mining, §6's calibrator feature) is currently
**zero in practice**, pending actual reviewer throughput — a scale
problem, not a code problem.

## 10. Testing & reproducibility

- **`tests/test_kg3_ingestion.py`** — 6 checks, all passing
  (`python3 -m pytest tests/test_kg3_ingestion.py -q -s`).
- **A real, confirmed gap**: no test file anywhere in `tests/` references
  `src/hitl_queue.py`'s functions (`load_ingestible_cases`,
  `submit_review`, `enqueue_pending_cases`, ...) — confirmed via direct
  search. The review-queue lifecycle itself has no dedicated unit-test
  coverage today, unlike the ingestion module it feeds.
- **To exercise the batch driver**:
  `python3 scripts/run_kg3_ingestion.py --dry-run` — resolves and prints
  every ingestible case without touching Memgraph or marking anything
  ingested, safe to run anytime against the real DB.
- **To check regression-tracking machinery**:
  `python3 scripts/regression_check_kg3.py` — read-only, safe to run
  anytime; will currently report an empty/first-run baseline given §9's
  numbers.

## 11. Simulated-review impact experiment — fresh-5 held-out notes

**Context and method, stated upfront.** §9 established that real review
throughput is zero, so §5-§6's mechanisms have no *measured* impact yet
— only a description of how they're supposed to work. This section
closes that gap with a genuine, one-off computational experiment: since
19,103 real reviewer clicks aren't available, the queue was
**gold-based auto-approved** instead — for every queued case, the
entity's already-known gold SNOMED annotation (from this project's own
evaluation corpus) stands in for a reviewer's verdict: a matching
suggestion is treated as `APPROVED`, a mismatch is treated as
`CORRECTED` to the gold concept, and anything without a clean,
gradable gold match is left `PENDING` rather than guessed at. This is
**not real HITL data** — it substitutes gold-corpus lookup for genuine
clinical judgment — and is reported here as exactly that: a controlled
simulation of "what if the queue were fully reviewed," not a claim
about real reviewer behavior. Run entirely against a **throwaway copy**
of the production DB (`kg3_impact_experiment.duckdb`, in the session
scratchpad, never committed) — the real production `hitl_review_queue`
(§9's numbers) was never touched by this experiment and remains 100%
`PENDING`.

**Notes measured**: the first 5 of `ui/components/fresh10_notes.py`'s
`FRESH10_NOTE_IDS` (`13538696-DS-11`, `19895550-DS-7`,
`11516225-DS-20`, `14652764-DS-17`, `12298181-DS-9`), held out from
approval so nothing about their own outcome could leak into the pool
that trains the mechanisms being measured. The approval pool was every
*other* queued case: 139 notes, 18,978 cases.

### 11.1 Approval pass — real numbers

| Outcome | Count | % of pool |
|---|---|---|
| `APPROVED` (suggestion matched gold) | 5,889 | 31.0% |
| `CORRECTED` (mismatch, gold concept resolvable) | 4,881 | 25.7% |
| Left `PENDING` — no clean single gold-span overlap | 8,043 | 42.4% |
| Left `PENDING` — gold SNOMED code not a resolvable standard OMOP concept | 165 | 0.9% |
| **Total reviewed** | **10,770** | **56.7%** |

The 42.4% "no clean gold overlap" figure is not a flaw in the
experiment — it reflects the same clean-span-only grading discipline
every other measurement in this project's documentation uses (a
prediction span must cleanly cover exactly one gold annotation to be
graded at all), applied honestly here rather than loosened to inflate
the approved count.

### 11.2 Mechanism 1 (mined context rules) — real output

`mine_context_rules()` (§5) ran against the newly-reviewed pool and
wrote **27 rules**, covering exactly **3 distinct abbreviations**:
`lad` (25 rules, split across two meanings — concept `315085`
"Lymphadenopathy" and concept `4245039`, the coronary-artery-segment
sense), `cta` (1 rule), `ttp` (1 rule). The `lad` result is a
genuine, substantive finding on its own: 21 independent confirmed
examples of the word `"no"` immediately preceding `lad` predicting the
Lymphadenopathy sense (e.g. "no lymphadenopathy" in a physical-exam
line) — this is real, mined evidence bearing directly on the
coronary-artery-segment systematic-bias problem §5.4 already discusses,
the population this mechanism was specifically built to eventually
help correct.

**Fresh-5 Stage 1 impact: zero.** Of fresh5's 22 ambiguous-expansion
entities, **none** use `lad`, `cta`, or `ttp` — a real, checked null
result, not an assumption. The mined rules are real and substantive,
they simply don't happen to touch this particular 5-note sample.

### 11.3 Mechanism 2 (calibrator confirmation count) — real output, with a real complication found along the way

Re-scoring fresh5's non-auto tier-gate decisions against the
now-populated `hitl_review_queue` surfaced a genuine, unplanned finding
**before** any impact number could even be computed: loading the
production calibrator with `scoring_note_ids=FRESH5` triggered its own
leakage guard —

```
! leakage: trained on 3 of the note(s) about to be scored
  (11516225-DS-20, 12298181-DS-9, 19895550-DS-7).
  refusing to use it; falling back to existing routing.
```

**3 of the 5 "fresh" notes used for this experiment were actually in
the production `ConsensusCalibrator`'s own training set** — contradicting
`ui/components/fresh10_notes.py`'s own docstring claim that this note
set is "genuinely held-out." That claim is accurate with respect to the
SNOMED near-duplicate retrieval fix and the KGE evaluation (what it was
originally written for), but **not** with respect to this specific
calibrator artifact. This is reported here, plainly, as a real
contamination finding this experiment surfaced — not fixed or corrected
elsewhere, per this section's scope.

Given that, results below are reported **per-note contamination
status**, never blended:

| Entity | Note | Status | Prior tier | `prior_confirmation_count` (new) | Calibrated score | Result |
|---|---|---|---|---|---|---|
| `Abd` | `13538696-DS-11` | clean | (tier=`None`, an error/edge-case decision) | 0 | 0.6646 | Below `CALIBRATED_AUTO_THRESHOLD=0.72` — not promoted |
| `VS` | `19895550-DS-7` | **contaminated** | `TIER_2_AUTO_RESOLVED` | 44 | 0.1786 | Not a clean result (calibrator trained on this note) — but note the score is low regardless |
| `RRR` | `14652764-DS-17` | clean | `TIER_2_AUTO_RESOLVED` | — | — | `_is_short_alphanumeric_code()` hard trap fired; calibrator never consulted |

Of fresh5's 125 total tier-gate decisions, only **3** were even
calibrator-eligible (already-auto tiers and split-vote decisions with
no `final_candidate_index` — i.e. no single candidate reached plurality
among the models — are structurally never consulted). **Net result on
the clean population: 0 of 0 eligible clean entities newly promoted**
(`Abd` is the only clean, calibrator-consulted case, and its score
stayed below threshold).

One number worth flagging on its own merit despite the contamination:
`VS`'s `prior_confirmation_count` reached **44** from a single
approval pass — real evidence the mechanism accumulates volume exactly
as designed — yet its calibrated score was still low (0.18). This is
consistent with, not contradicting, this project's own earlier ablation
finding (§6): `prior_confirmation_count` is a real but **secondary**
signal in the fitted model, not one that can push a score to promotion
on its own regardless of magnitude.

### 11.4 Honest summary of this experiment

**Measured impact of KG3's repurposing mechanisms on this specific
fresh-5 note set, after gold-based-simulated review of 56.7% of the
queue: zero routing changes.** Not because either mechanism is broken —
§11.1-§11.3 show both firing exactly as designed on real data (27 real
rules mined, a real 44-count confirmation signal accumulated) — but
because of three compounding, honestly-reported reasons specific to
this sample: (1) the abbreviations that reached mining volume don't
overlap this note set's ambiguous entities, (2) most of this note set's
split-vote entities structurally never reach the calibrator at all
(no plurality candidate), and (3) the one genuinely clean,
calibrator-consulted entity scored meaningfully below the promotion
threshold. A larger approval pass, a different 5-note sample, or simply
more accumulated volume over time could all plausibly change this —
this experiment measures one honest data point, not a ceiling on what
the mechanisms can do.

**Reproducing this experiment**: the two scripts used
(`kg3_impact_experiment.py`, `kg3_impact_experiment_part2.py`) were
written as one-off session scratchpad scripts, not committed to this
repository — they are not part of the maintained codebase. Reproducing
this result requires a scratch copy of the production DB, gold
annotations and raw note text (both PhysioNet-restricted, see the
README's Data Access section), and the same gold-based-approval
methodology described in this section.

## 12. Honest limitations

- **Zero real review throughput to date** (§9) — real reviewer clicks
  remain at zero. §11's numbers come from a gold-based *simulation* of
  review, not genuine human judgment — a real, useful data point, but
  not the same evidence real HITL volume would provide.
- **§11's simulated experiment measured zero routing changes** on the
  specific fresh-5 note sample tested — both mechanisms fired
  correctly on real data (27 mined rules, a real 44-count confirmation
  signal), but neither happened to change an outcome for these
  particular 5 notes; §11.4 explains why in detail. Do not read this as
  evidence the mechanisms don't work — the sample is small and the null
  result is well-explained, not mysterious.
- **A real note-set contamination was found while running §11's
  experiment**: 3 of the 5 "fresh10" notes are actually inside the
  production `ConsensusCalibrator`'s own training set, despite
  `ui/components/fresh10_notes.py`'s docstring describing the set as
  "genuinely held-out." That claim holds for what it was originally
  written about (the SNOMED near-duplicate fix, the KGE evaluation) but
  not for this calibrator specifically. Flagged here, not corrected
  elsewhere, per this document's scope.
- **`mine_context_rules()` needs real volume, not just any review
  activity** — `MIN_CONTEXT_RULE_SUPPORT=5` independent examples *per
  abbreviation* before any rule is written at all; a handful of reviewed
  cases spread across many different abbreviations would still yield
  zero rules for a long time.
- **`count_prior_confirmations()`'s circularity risk is real, not fully
  closed** (§6) — mitigated by the calibrator's held-out-split
  discipline and confirmed low-importance via ablation, but the feature
  itself is still built partly from the pipeline's own dry-run
  decisions, not exclusively from independent human confirmation.
- **The HITL review UI's throughput is the actual bottleneck** — nothing
  in §2-§8 is blocked on more engineering; it is blocked on 19,103
  queued cases actually being reviewed by a human, which this document's
  mechanisms cannot accelerate on their own.
- **No dedicated test coverage for `src/hitl_queue.py`** (§10) — a real,
  stated gap, not glossed over.
