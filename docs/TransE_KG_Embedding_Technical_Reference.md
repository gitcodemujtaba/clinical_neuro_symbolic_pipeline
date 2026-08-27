# TransE Knowledge-Graph Embedding — Technical Reference

Deep-dive companion to `docs/Implementation_Methodology.md`'s KG-embedding
bullet (Stage 2b section) and `docs/FINAL_RESULTS_Single_Source_Of_Truth.md`
§6. This document covers *how* the TransE model was built, trained, and
evaluated, and exactly what was and wasn't achieved — grounded in the real
code (`src/kg_embedding.py`, `src/kg_embedding_tiebreak.py`,
`scripts/build_kg_embeddings.py`, `evaluation/kg_tiebreak_validation.py`)
and the real, current checkpoint on disk (`models/kg_transe_v1.pt`) and
results log (`logs/kg_embedding_results.json`).

**Headline, upfront**: TransE was fully implemented, trained end-to-end on
a real SNOMED subgraph, and rigorously evaluated two ways (intrinsic link
prediction and an extrinsic task-specific check). A second module then
applied it as a candidate-reranking tiebreak and validated that application
against gold data, head-to-head against the existing hardcoded rule it was
proposed to replace. **The deliberate, evidence-based outcome was NOT to
wire it into production** — the hardcoded rule it was compared against won
outright on its own pattern. This document explains both halves honestly:
what was built and how well it works in the abstract, and why it still
isn't live.

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

`src/kg_embedding.py` trains directly on the reference graph, scoped to
the subset of SNOMED concepts this pipeline's own candidate pools have
actually touched. This preserves the spirit of the proposal's "based on
our own TP records" framing — the graph's *scope* is defined by concepts
this project's own tier gate has actually resolved and graded — while
using a graph that genuinely has data today.

## 2. Method choice — TransE, not RotatE or CompGCN

Only **TransE** (Bordes et al., 2013) was implemented. This is stated
plainly in the module's own docstring
(`src/kg_embedding.py:23-31`) and repeated consistently across every
current doc that discusses it — nowhere in this project's active
documentation is RotatE or CompGCN claimed as built.

Reasoning for the choice, as recorded in the code:

- TransE is the simplest, most standard member of the KGE family the
  proposal names, with a transparent scoring function
  (`score(h, r, t) = -‖h + r − t‖₂`) — the model literally learns to make
  `head + relation ≈ tail` in vector space.
- It needed **no new library dependency**. Neither `pykeen` nor
  `torch-geometric` is installed in this environment; TransE is simple
  enough to implement correctly in ~190 lines of plain PyTorch
  (`src/kg_embedding.py`), while RotatE (complex-valued embeddings) and
  CompGCN (a full graph-convolutional architecture) are both
  meaningfully more complex and were not attempted.
- This is recorded as **honest scope, not a silent gap**: the module
  docstring says outright that RotatE/CompGCN are "real, meaningfully
  more complex follow-on work, not attempted here."

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

**Current, live scale** (`logs/kg_embedding_results.json`, matching the
checkpoint on disk):

| Metric | Value |
|---|---|
| Distinct concepts touched by candidate pools | 7,276 |
| Real SNOMED relationship triples (both endpoints touched) | 24,922 |
| Vocabulary — entities | 7,269 |
| Vocabulary — relation types | 104 |
| Train / test split (90/10, random) | 22,429 / 2,493 |

(An earlier training run, before the Stage 3 recall-fix backfill enlarged
the candidate-pool population, reported a slightly smaller graph — 7,261
concepts, 24,872 edges, 2,488 test triples. The model was **retrained**
once the larger post-backfill pool existed; the numbers in this document
and in `logs/kg_embedding_results.json` reflect that retrained, currently
checkpointed model, not the earlier run.)

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
after each training batch). This is the standard TransE trick that
prevents the model from trivially "cheating" the margin loss by growing
embedding norms rather than actually learning graph structure — without
it, the model can minimize loss by making everything far apart in an
absolute sense, without the *relative* geometry (`h + r ≈ t`) meaning
anything.

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

Checkpointing (`save_model()`/`load_model()`, `src/kg_embedding.py:54-74`,
added specifically so the model didn't need retraining for every
downstream validation run) bundles the `state_dict` **together with**
`entity2idx`/`relation2idx` and the embedding dimension into one `.pt`
file — necessary because the vocabulary is data-dependent (which
`concept_id`s and relation types were in *this* training run's subgraph);
a bare `state_dict` alone can't be reloaded into usable lookups without it.

**Current checkpoint on disk**: `models/kg_transe_v1.pt` (3.0 MB),
committed to git.

## 6. Evaluation — two real, complementary evaluations

The module's own docstring is explicit that both are real (not simulated
or estimated): a standard KGE-literature intrinsic evaluation, and a
second, task-specific extrinsic evaluation tied directly to this
project's own graded decisions.

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
  numbers, and the docstring flags this explicitly so the figure isn't
  misread against a filtered benchmark.
- Ranking a triple's true tail against the *full* entity set for every
  held-out triple is the expensive part of this evaluation, so
  `evaluate_link_prediction()` caps evaluation at `max_eval=2000` triples
  — **a random subsample of the 2,493 held-out test triples, not the
  first N** (`random.sample`, not slicing), so the reported number isn't
  biased by triple ordering in the split.

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
concept?** That's the literal signal a KGE-assisted re-ranker would need
to be useful.

Mechanism: for each gold-confirmed true-positive tier-gate decision with
at least one other real candidate in its pool (`gather_tp_records()`,
`scripts/build_kg_embeddings.py:33-87` — reuses this session's standard
clean-span gold-matching methodology, only counting records where the
tier gate's winning candidate was independently confirmed correct against
gold), it computes:

- `d_wrong` — L2 distance in embedding space from the correct concept to
  the wrong candidate that actually competed for the same mention.
- `d_random` — L2 distance from the correct concept to an arbitrary,
  unrelated concept.

and reports the fraction of comparisons where `d_wrong < d_random` — i.e.
the genuinely-confusable rival sits closer than chance would predict.

**Result** (`logs/kg_embedding_results.json`):

```json
"extrinsic": {
  "n_tp_records_provided": 455,
  "n_usable_records": 455,
  "n_comparisons": 1612,
  "frac_wrong_candidate_closer_than_random": 0.6892059553349876
}
```

**68.9%** of 1,612 comparisons across 455 real gold-confirmed TP records
— well above the 50% chance baseline. This is real, positive signal: the
embedding space does capture genuine clinical proximity among candidates
that actually competed for the same mention, not just generic SNOMED
graph structure.

(An earlier pre-backfill-retrain run reported 457 records / 1,623
comparisons / 70.2% — consistent order of magnitude, same conclusion; the
68.9%/455/1,612 figures above are the current, checkpointed model's
numbers and are the ones to cite going forward.)

## 7. Applying it: the candidate-reranking tiebreak

Training and evaluating the embedding is only half the work — the second
module, `src/kg_embedding_tiebreak.py`, applies the trained model to a
concrete Stage 2b problem: **SNOMED near-duplicate concept pairs**, the
same failure class this session's SNOMED regional-extension fix
(`docs/Implementation_Decisions_Log.md` §7-related work) kept surfacing —
WBC/RBC Procedure-vs-Observable-Entity confusion, an HCO3
hierarchy-collapse bug, and a wound-dehiscence Condition-vs-Observation
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
for a tied candidate, the mean TransE embedding distance from that
candidate to every *other* concept in the same entity's own broader
SapBERT-proposed candidate pool (not just the tied pair). The hypothesis:
the genuinely correct concept, being about the same clinical topic as the
mention, should sit closer to that pool's broader semantic neighborhood
than a structurally-adjacent-but-topically-off sibling concept does.

`pick_via_kg_tiebreak()` (`src/kg_embedding_tiebreak.py:68-92`) picks
whichever tied candidate has the **lowest** mean distance to the rest of
the pool, and returns `resolved=False` (caller falls back to its existing
logic) when fewer than two tied candidates have a usable score — i.e.
this mechanism never silently forces a pick it has no real basis for.

This deliberately stays *within* KGE's own embedding space (comparing
candidate-to-candidate distances to each other), rather than trying to
bridge KGE's graph-structural coordinate system and SapBERT's
text-semantic coordinate system directly — the two are not the same
space, and comparing raw distances across them without a learned mapping
would not be a principled comparison.

## 8. Validation against gold — the decisive test

`evaluation/kg_tiebreak_validation.py` grades the tiebreak's actual picks
against real gold-standard SNOMED codes, sweeping a `TIE_THRESHOLD`
parameter (how close the top-1/top-2 SapBERT similarity scores need to be
before a pair counts as "tied" at all) and comparing head-to-head against
the existing hardcoded `_prefer_lab_procedure_over_observable()` rule on
the specific pattern that rule already covers.

**Zero live SapBERT/embedding calls** — this harness reuses each
entity's already-stored `normalized_entities.candidates` (similarity
scores computed once at Stage 2b time), loading only the small TransE
checkpoint itself. This was a deliberate design choice after an earlier,
unrelated attempt to recompute a fresh embedding live measurably stalled
a concurrent Stage 3 backfill for ~22 minutes via disk/CPU contention —
this harness was built specifically to never repeat that mistake.

Three-way outcome classification per entity (`classify_outcome()`,
pure-logic, unit-tested with no DB or model needed):

- **WIN** — baseline (SapBERT top-1) was gold-wrong, the mechanism's pick
  is gold-correct.
- **LOSS** — baseline was gold-correct, the mechanism's pick is
  gold-wrong (the fatal case — a mechanism that causes this on net
  should not ship, regardless of how many wins it also produces
  elsewhere).
- **NEUTRAL** — both right, both wrong, or the pick didn't change
  (includes cases where KGE had no usable score and fell back to the
  baseline unchanged).

**Results, real gold-validated threshold sweep**:

| Threshold | Full population win/loss | KGE loss on hardcoded rule's own pattern |
|---|---|---|
| 0.01 | 12 win / 20 loss (net negative) | 0 / 0 (n=5) |
| 0.02 | 93 win / 104 loss (net negative) | 0 / 0 (n=93, tied) |
| 0.03 | 265 win / 181 loss (**net +84**) | **63 / 0** |
| 0.05 | 347 win / 200 loss | 63 / 0 |
| 0.08 | 347 win / 202 loss | 63 / 0 |

Two separate conclusions come out of this table, and they point in
opposite directions — both are reported honestly:

1. **On the hardcoded rule's own pattern** (Lab Test entities, Procedure
   vs. Observable-Entity/Qualifier-Value candidates — exactly what
   `_prefer_lab_procedure_over_observable()` was built for): the rule has
   **zero losses at every threshold tested**, up to n=380. KGE has **63
   losses** once the threshold widens past 0.02. On its own specialist
   pattern, the hardcoded rule is strictly safer.
2. **On the broader tied-pair population beyond the rule's scope**
   (n=1,828 at threshold 0.03, any entity label, any tied pair) — KGE
   shows a genuine **positive net** (265 win / 181 loss). Real value as a
   generalist secondary signal for patterns the hardcoded rule was never
   built to cover, but a ~9.9% net-loss rate is not risk-free enough to
   auto-write on its own without a calibrated gating mechanism that
   doesn't yet exist.

**Specific falsification, directly tested rather than assumed.** A
proposed narrative (raised mid-session as a plausible-sounding argument)
claimed KGE would "naturally" resolve a third, previously-unexamined
SNOMED near-duplicate found during the regional-extension fix work —
"Mean cell hemoglobin concentration - finding" (Clinical Finding domain)
winning unanimous ensemble agreement over the actual gold-correct
concept. This was checked directly against the real retrained model on
the real failing entity, not accepted on the strength of the argument:
**KGE picked the identical wrong concept**, with a 0.0018 embedding-distance
margin between the two candidates — noise, not a real topological
separation. The claim was false for this specific case.

## 9. Decision: built, evaluated, deliberately not wired into production

`_prefer_lab_procedure_over_observable()` (`src/normalization/tier_retrieval.py`)
stays as the production mechanism for the pattern it covers, unchanged.
The KGE tiebreak (`src/kg_embedding_tiebreak.py`) is **not called from
`route_tier()`, `tier_retrieval.py`, or any other production code path** —
confirmed directly (no references outside its own module, its
validation harness, and its tests). It remains built, checkpointed, and
evaluated: a real, positive-net-but-imperfect generalist signal, staged
for future work once a calibrated gating mechanism exists to decide
*when* to trust it — the same kind of mechanism `src/mollm_tier_calibrator.py`'s
`ConsensusCalibrator` already provides for the MoLLM ensemble-vote gate,
not yet built for this signal.

## 10. Testing & reproducibility

- **`tests/test_kg_tiebreak_validation.py`** — 19 checks, all passing
  (`python3 -m pytest tests/test_kg_tiebreak_validation.py -q -s`),
  covering: `classify_outcome()`'s four win/loss/neutral combinations,
  `hardcoded_rule_applicable()`/`hardcoded_rule_pick()`'s exact
  replication of the production rule's decision logic, and a genuine
  `save_model()`/`load_model()` roundtrip (train a tiny TransE, save,
  reload, confirm identical scores) — no DB or live model training
  required for any of these, deliberately, so they can run any time.
- **To retrain from scratch**: `python3 scripts/build_kg_embeddings.py`
  (read-only DB access throughout — safe to run alongside a concurrent
  Stage 3 batch holding the write lock). Writes
  `models/kg_transe_v1.pt` and `logs/kg_embedding_results.json`.
- **To re-run the gold validation**:
  `python3 -m evaluation.kg_tiebreak_validation [--thresholds 0.01,0.02,0.03,0.05,0.08]`
  — requires the checkpoint above to already exist; raises a clear error
  rather than silently training a throwaway substitute model, since the
  whole point is to validate the exact weights that would be wired into
  production.

## 11. Honest limitations

- **RAW, not filtered, MRR** — the 0.776 figure is not directly
  comparable to "filtered MRR" numbers reported in most KGE papers; this
  is stated in the code and repeated here so it isn't misquoted against
  a different protocol later.
- **Intrinsic evaluation is capped at 2,000 of 2,493 held-out triples**
  (a random subsample, for compute reasons) — not the full held-out set.
- **The extrinsic evaluation's TP-record population is itself dependent
  on this project's own tier-gate output** — it measures whether the
  embedding separates candidates that *this pipeline's own retrieval*
  already surfaced as competitors, not an independent, externally-sourced
  confusion set.
- **No calibrated confidence/threshold exists for the tiebreak signal** —
  the win/loss numbers above are population-level rates, not a
  per-decision confidence score; this is exactly why it wasn't wired in
  despite the net-positive result on the broader population.
- **RotatE and CompGCN remain genuinely unbuilt** — not attempted, not
  partially built, stated as scope not taken on, consistent everywhere
  this project's documentation discusses it.
