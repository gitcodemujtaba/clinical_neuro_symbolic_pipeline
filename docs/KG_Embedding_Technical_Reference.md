# Knowledge-Graph Embedding (TransE + RotatE) — Technical Reference

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

# Part A — TransE

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

# Part B — RotatE

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

# Part C — Applying to production, decision, and shared reference

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
