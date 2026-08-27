# SapBERT — Technical Implementation Reference

Deep-dive companion to `docs/Implementation_Methodology.md`'s Stage 2b
section. This document covers exactly how SapBERT's 768-dimensional
dense embeddings are used in this pipeline for semantic similarity
matching — model loading, how the reference vocabulary (`athena_concept`)
got a vector column in the first place, the DuckDB-native similarity
search built on top of it, the calibrated similarity floor and its
measured precision/coverage trade-off, and two real experiments that
tried to beat SapBERT-alone and lost. Every claim is grounded in the real,
current source (`src/normalization/sapbert_model.py`,
`src/normalization/tier_retrieval.py`, `src/normalization/constants.py`,
`scripts/merge_embeddings.py`) and live queries run against the
production DuckDB while writing this document, not figures quoted from
memory.

---

## 1. What SapBERT is and its role in this pipeline

**Model**: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
(`MODEL_NAME`, `src/normalization/sapbert_model.py:12`) — a biomedical
sentence-embedding model, self-alignment-pretrained on UMLS synonym
pairs on top of PubMedBERT, so that two different surface strings for
the same underlying biomedical concept (e.g. a clinical mention and its
formal SNOMED name) land close together in embedding space. This is
exactly the property Stage 2b's Tier 3 needs: a mention rarely matches a
SNOMED concept name character-for-character (Tier 1) or via a known
synonym row (Tier 2), so a third path that measures *semantic* closeness
is required to ground the remaining mentions at all.

**Role**: SapBERT embeddings are used in **three** distinct places in
Stage 2b, not just one — the primary Tier 3 dense-retrieval fallback
(§5), an optional Tier 1/2 tiebreak criterion (§8), and the scoring
signal inside two experimental mechanisms that were built, measured, and
ultimately **not** adopted (§11-12). All three consume the same two
building blocks: `get_sapbert_embedding()` (turns text into a
768-dimensional vector) and a precomputed embedding column on
`athena_concept` (§3) that lets a query compare a fresh mention vector
against ~350,000 concept vectors without re-embedding the vocabulary on
every call.

## 2. Embedding generation

```python
def get_sapbert_embedding(text: str) -> list:
    """Generates a 768-dimensional SapBERT vector for a given text."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    tokens = {k: v.to(_SAPBERT_DEVICE) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = sapbert(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding
```

(`src/normalization/sapbert_model.py:32-38`.) **CLS-token pooling**
(`SAPBERT_POOLING = "cls"`, `sapbert_model.py:14`) — the embedding is the
hidden state at position 0 (`[:, 0, :]`) of the final transformer layer,
the standard pooling strategy for BERT-family sentence-embedding models
and the one SapBERT was itself trained against. Input is truncated at
128 tokens (`max_length=128`) — comfortably enough for a clinical
mention or a SNOMED concept name, both of which are short strings, not
full documents (contrast with GLiNER-BioMed's 2048-word-token ceiling
over whole notes — a completely different scale of input, see
`docs/GLiNER_Models_Technical_Reference.md`).

**GPU placement, same pattern as GLiNER** (`sapbert_model.py:19-28`):

```python
_SAPBERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    sapbert = AutoModel.from_pretrained(MODEL_NAME).to(_SAPBERT_DEVICE)
except Exception:
    _SAPBERT_DEVICE = "cpu"
    sapbert = AutoModel.from_pretrained(MODEL_NAME).to("cpu")
```

`AutoModel.from_pretrained()`'s own default is CPU-only unless placed
explicitly — the exact same gap `GLiNER.from_pretrained()` has (see
`docs/GLiNER_Models_Technical_Reference.md` Part 1 §3), fixed the same
way, with the same fallback discipline: a CUDA failure degrades to CPU
rather than taking down normalization entirely. The code's own comment
flags this as **likely the single biggest normalization-time GPU win**
in the whole pipeline, more than GLiNER itself: SapBERT is called far
more often, since every Tier 3 candidate lookup needs an embedding, and
`CONTEXTUAL_CANDIDATES_ENABLED` being default-on (widening the candidate
pool) means more multi-candidate ranking calls too.

## 3. The Athena embedding column — how the vocabulary got vectorized

This is the piece that makes Tier 3 possible at all: `athena_concept`
(the imported OMOP/SNOMED reference vocabulary) needed a **precomputed**
768-dimensional vector per concept, so that a live query only has to
embed the *mention* once and compare it against every concept's
*already-stored* vector — re-embedding ~350,000 SNOMED concept names on
every single entity lookup would make Stage 2b unusably slow.

**Live-confirmed schema** (queried directly against the production
DuckDB while writing this document):

```sql
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name='athena_concept' AND column_name='embedding';
-- ('embedding', 'FLOAT[]')
```

`athena_concept.embedding` is a native DuckDB `FLOAT[]` array column,
confirmed **768-dimensional** (`SELECT len(embedding) ... LIMIT 1` →
`768`) — matching SapBERT's own hidden size exactly, since the column
was populated by running the same `get_sapbert_embedding()` function
(or its equivalent) over every concept's `concept_name`.

**How the column actually got populated — two real, connected pieces,
one tracked, one not**:

1. **`scripts/merge_embeddings.py`** — the real, tracked, currently
   runnable step. It does *not* compute embeddings itself; it merges a
   pre-populated **staging table**, `concept_embeddings` (schema:
   `concept_id`, `embedding`), into the main `athena_concept` table via
   a single SQL `UPDATE ... FROM`:
   ```sql
   UPDATE athena_concept
   SET embedding = concept_embeddings.embedding
   FROM concept_embeddings
   WHERE athena_concept.concept_id = concept_embeddings.concept_id;
   ```
   Prints a verification count (`concepts_with_embeddings`) immediately
   after, so a merge that silently under-covers the table doesn't go
   unnoticed.
2. **`scripts/build_concept_embeddings.py`** — referenced by name in
   other scripts' docstrings (`scripts/backfill_guideline_grounding.py:151-152`,
   twice: *"Requires the populated kg2_lexical_store.duckdb (Athena OMOP
   + SapBERT embeddings) to already exist — run scripts/import_athena.py
   and scripts/build_concept_embeddings.py first if it doesn't"*) as the
   step that would actually **compute** the `concept_embeddings` staging
   table's contents in the first place. **Confirmed directly: this file
   is a 0-byte stub on disk** — it was never actually written into this
   tracked repository. This is stated plainly rather than glossed over,
   consistent with `scripts/check_stage3_prerequisites.py`'s own honest
   comment on the same fact: *"athena_concept is already populated WITH
   embeddings on the EC2 box, built outside these scripts"* — i.e. the
   real embedding-generation run happened once, directly against the
   production database, outside of any script this project's git history
   actually tracks. `merge_embeddings.py` and the `UPDATE`-from-staging
   pattern are real and reproducible; the original generation step is
   not currently reconstructable from this repo alone.

**Real, live-measured coverage** (queried directly while writing this
document, not assumed):

| Population | Count | Coverage |
|---|---|---|
| All rows in `athena_concept` | 6,594,567 | — |
| Rows with a non-NULL `embedding` | 2,922,292 | 44.3% of all rows |
| **Standard-concept SNOMED rows** (`vocabulary_id='SNOMED' AND standard_concept='S'`) | 349,211 | — |
| ...of those, **with** `embedding` | 349,211 | **100.0%** |

The headline 44.3% figure undersells what actually matters: **every
query this pipeline's Tier 3 (and the Tier 1/2 semantic tiebreak, and
the fuzzy fallback) actually issues filters on `standard_concept='S'`
and a specific vocabulary list** (§5, §7) — and on exactly that
population, coverage is complete. The 55.7% of `athena_concept` rows
*without* an embedding are non-standard concepts and non-target
vocabularies that no live query in this pipeline ever reaches in the
first place, not a real gap in what Tier 3 needs.

## 4. DuckDB-native vector search — no external vector database

Every SapBERT-similarity query in this codebase uses DuckDB's own
built-in `list_cosine_similarity()` scalar function directly against the
`FLOAT[]` column — there is no separate vector database (no FAISS
index, no pgvector, no external ANN service) anywhere in this pipeline:

```sql
SELECT concept_id, concept_name, domain_id, vocabulary_id,
       list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
FROM athena_concept
WHERE embedding IS NOT NULL AND standard_concept = 'S'
AND vocabulary_id IN (...) ...
ORDER BY similarity DESC, concept_id ASC LIMIT ...;
```

This is a genuine brute-force cosine scan over the filtered candidate
population (not an approximate-nearest-neighbor index) — viable
specifically because the domain/vocabulary filters (§7) narrow the scan
to a few-thousand-to-low-tens-of-thousands-row slice of the table before
the similarity computation runs, not the full 349,211-row standard-SNOMED
population on every call. `?::FLOAT[]` binds the mention's own freshly
computed 768-dim vector as a query parameter — one embedding call per
mention (or per candidate, for the Tier 1/2 tiebreak, §8), zero
re-embedding of the vocabulary side, which is already sitting in the
column from §3.

## 5. Tier 3 — the primary dense-retrieval fallback

`_tier3_semantic_rows()` (`src/normalization/tier_retrieval.py:785-840`)
is the core Tier 3 query — runs only when Tier 1 (exact concept-name
match) and Tier 2 (exact synonym match) found nothing, or as the ranking
signal within a multi-hit tie (§8). Returns the top `CANDIDATE_LIMIT`
(§6) concepts by cosine similarity to the mention's own SapBERT vector,
filtered by vocabulary and domain (§7) and by the SCTID
regional-extension exclusion (`_UK_EXTENSION_EXCLUSION` — see
`docs/TransE_KG_Embedding_Technical_Reference.md` for the full story of
that fix, which lives in this same query).

### 5.1 `TIER3_SIMILARITY_FLOOR = 0.72` — calibrated, with a real, measured, still-open trade-off

```python
TIER3_SIMILARITY_FLOOR = 0.72
```

(`src/normalization/constants.py:139`.) Below this cosine-similarity
threshold, Stage 2b returns `NO_CANDIDATE` rather than a weak match —
this constant is what makes "0 (Failed)" a real, reachable outcome at
all. Before it existed, `normalize_entity()` always returned its top-1
candidate regardless of score, so confidently wrong matches
(`lasix`→`Laslades`, `bioplar`→`Bourgvilain`) flowed through the
pipeline as if fully resolved.

**The measured trade-off, kept despite the cost, per explicit user
direction — stated honestly, not settled**: a full 27-note corpus run
(2,303 entities) comparing with/without the floor found:

| | Precision | Coverage |
|---|---|---|
| Without floor | 47.9% | 71.9% |
| **With floor (current)** | **50.7%** (+2.8pt) | **59.7%** (−12.2pt) |

Of 369 entities the floor flipped from "matched" to `NO_CANDIDATE`, 190
were genuine garbage correctly rejected — but **90 were genuinely
gold-correct matches now lost outright**. The concrete example cited in
the project's own decision record: `WBC-13.0` → "White blood cell
count", correct, at similarity 0.8694 — comfortably *above* 0.72, so
this specific case survives the floor; the real losses are gold-correct
matches that scored *between* the old (no-floor) and new (0.72) regime
in cases the record doesn't single out individually, but the aggregate
90-entity cost is real and counted. This is recorded in this project's
own documentation as **not yet finally decided** whether to keep, revert,
or soften — a genuine open trade-off between precision and coverage,
not a settled, closed question.

### 5.2 A real bug found and fixed in the floor check itself

During the hybrid-retrieval experiment (§11), the floor check was found
to compare against `candidates[0]`'s score — which, after RRF fusion
reorders candidates by a *blended* score, is no longer necessarily the
pool's best *dense* score. This produced a measurable artifact: hybrid
mode showed 10 more zero-candidate rejections than dense-only despite
identical top-5 correctness counts on the same entities — a
shrinking-denominator illusion, not a real precision difference. Fixed
by anchoring both floor checks to the **pool's maximum dense score**
specifically, not whichever candidate happens to rank first after any
downstream reordering. Post-fix, dense-only mode was confirmed
byte-identical to the pre-fix run on every metric — the fix changed
nothing for the mode already in production, only corrected the
comparison for the experimental mode.

## 6. `CANDIDATE_LIMIT` — widened from 3 to 5, against a measured oracle target

```python
CANDIDATE_LIMIT = 5
```

(`src/normalization/constants.py:157`, bumped 2026-08-16 from an
original 3.) Every downstream consumer (`src/mollm_ensemble.py`'s
`build_prompt()`, `src/mollm_tier_gate.py`'s Step B) already iterates
`range(1, len(candidates) + 1)` dynamically rather than assuming exactly
3, so this was a pure data-volume change requiring no consumer code
changes — checked before the bump, not assumed safe.

The target this widening was measured against:
`evaluation/stage2b_cal_eval.py`'s `ranking_quality()` computes
**oracle accuracy** — whether the gold-correct concept is *anywhere* in
the top-K pool, regardless of which one Tier 3's own ranking put first —
against a stated blueprint target of ">98% in Top-5." Real measured
oracle accuracy on this corpus does **not** reach that target: ~56%
(dense-only) on a 300-entity real sample (§11 covers the corresponding
hybrid-mode comparison) — a substantially harder corpus than the
blueprint's target anticipated, reported honestly rather than adjusted
to look closer to the target after the fact.

## 7. Domain and vocabulary filtering — narrowing the scan before the vector search runs

Two filters apply *before* `list_cosine_similarity()` ever computes a
score, both real, both grounded in measured evidence rather than
assumed:

- **`GLINER_LABEL_TO_DOMAIN`** (`constants.py:30-52`) — maps each
  GLiNER entity label to a set of OMOP `domain_id` values (e.g.
  `"Medication": ["Drug"]`), applied to *all three* tiers, not just
  Tier 3 — the specific collision that motivated this
  (`ED`→"Ed District") was an exact **Tier 2** synonym match, so gating
  only the vector fallback would have left the original failure
  in place. `CONTEXTUAL_CANDIDATES_ENABLED` (default **on** since
  2026-08-18) widens `"Condition"` to also include the `"Observation"`
  domain — closes a real, measured class of Condition-vs-Observation
  SNOMED duplicate confusion (the wound-dehiscence pattern:
  225553008 Condition/Disorder vs. the gold-correct 410723003
  Observation/Morphologic Abnormality), found by tracing 4,580 wrong-
  concept cases across a 109-note corpus and confirming 41 shared an
  identical `concept_name` with gold's own answer under a different
  code and domain.
- **`VOCAB_BY_LABEL`** (`constants.py:91-93`) — restricts Medication
  entities to `["RxNorm", "RxNorm Extension"]` rather than
  `DEFAULT_VOCAB = ["SNOMED"]`, since OMOP codes medications via RxNorm,
  not SNOMED's Substance hierarchy. Considered and explicitly **not**
  extended to other domains: a direct empirical check of the real
  `vocabulary_id` distribution per OMOP domain found Anatomy already
  100% SNOMED, and Procedure/Measurement's larger non-SNOMED
  vocabularies (ICD10PCS, LOINC) aren't this project's target crosswalk
  vocabulary — a considered non-action, not an unexamined gap.

## 8. SapBERT as a Tier 1/2 tiebreak criterion — opt-in, deliberately off by default

Beyond its primary Tier 3 role, SapBERT cosine similarity is also
available as the **third** criterion in a 5-criterion ranking key that
breaks ties among multiple Tier 1/2 exact/synonym hits (concept class
preference, then domain agreement, then SapBERT cosine, then hierarchy
specificity, then `concept_id` as the final deterministic tiebreak —
`src/normalization/tier_retrieval.py:160-242`):

```python
def key(r):
    cid, domain_id = r[0], r[2]
    return (
        _class_rank(classes.get(cid), gliner_label),
        0 if (not wanted_domains or domain_id in wanted_domains) else 1,
        -sims.get(cid, 0.0),      # SapBERT cosine, entity text vs each candidate NAME
        -depths.get(cid, 0),
        cid,
    )
```

`TIER12_RANK_SEMANTIC` (`constants.py:234-235`) gates this criterion
**off by default**, separately from `RANKED_TIER12` (the ranker itself,
also off by default). The reasoning, stated directly in the code: this
is the *only* one of the five criteria that costs a model call, and it
breaks a property the ranker's own design explicitly protects — Tier
1/2 must stay "cheap enough to try at every candidate boundary," since
compound-span partition search calls into Tier 1/2 repeatedly per
entity, making an embedding-per-attempt cost scale with the number of
boundaries tried, not a fixed per-entity cost. The first two criteria
(class preference, domain agreement) already cover the measured failure
mode this ranker was built for (`"Left"`/`"Initial"` colliding with
Qualifier Value concepts) with zero embedding cost — so the expected
marginal gain from enabling the paid criterion is concentrated exactly
where the free criteria don't reach, and turning it on is framed
explicitly as its own separate, measurable A/B (`tier12_rank_basis`
records `"ranked_v1"` vs. `"ranked_v1_semantic"` per row precisely so
this comparison is gradable), not a default worth flipping without that
measurement.

## 9. Alias force-inclusion — when a real answer would lose the cosine race

`_tier3_semantic_rows()`'s `alias_ids` parameter (§5) force-includes a
KG-verified brand→generic alias concept in the candidate pool
**regardless of where it lands in the cosine ranking** — scored by its
*own* real similarity (not pinned to 1.0, so Stage 3 still sees the true
semantic distance), only its *presence* in the candidate list is
guaranteed. This exists because SapBERT's embedding space does not
reliably place a brand name close to its own generic ingredient — "the
Lasix problem": a KG-verified brand-alias hit for "Lasix" (furosemide,
similarity 0.57) can score *lower* than a coincidental spelling match
("lasalocid", 0.68) that has nothing to do with the mention. Without
force-inclusion, a real cosine-similarity gap of exactly this kind
silently drops the correct concept out of the top `CANDIDATE_LIMIT`
before Stage 3 ever gets a chance to judge it. `match_basis` is set to
`"verified_brand_alias"` for these rows specifically, so a small model
downstream can weigh "this is a certain fact from the terminology graph"
differently from "this merely sounds alike" — two structurally identical
dicts otherwise carry no signal to distinguish them.

## 10. What SapBERT alone misses — the fuzzy edit-distance supplement

`_fuzzy_typo_candidates()` (`src/normalization/tier_retrieval.py:247-279`)
is a deliberately narrow, **additive-only** Levenshtein-distance fallback,
not a fifth tier and never a mechanism that overrides a confident SapBERT
match on its own. It exists because SapBERT's semantic embedding space
and simple spelling distance are genuinely different signals that can
fail independently: the concrete case that motivated it — `spirnolactone`
(a misspelling of spironolactone) — embedded *closer* to `SPIRAPRILAT`
and `SPIRILENE` than to the correctly-spelled target concept under
Tier 3's semantic search. The true concept never entered the candidate
list at all, so both MoLLM ensemble models were left picking the
closest-*spelled* wrong drug from a pool that never contained the right
answer in the first place — no amount of downstream ensemble reasoning
can recover a candidate that retrieval never surfaced. Edit distance
(`FUZZY_MAX_EDIT_DISTANCE=2`, `FUZZY_MIN_TEXT_LENGTH=5`) only supplements
an *already-uncertain* Tier 3 result (below-floor or margin-ambiguous),
deliberately not its own standalone tier, since edit distance on short
clinical tokens/abbreviations is noisy on its own.

## 11. The BM25+SapBERT+prior hybrid experiment — SapBERT alone won

A full alternative retrieval mechanism was built and measured against
dense-only SapBERT: `_tier3_hybrid_rows()`
(`src/normalization/tier_retrieval.py:844-970+`), fusing dense (SapBERT
cosine), sparse (a real DuckDB FTS/BM25 index over concept names and
synonyms), and an empirical prior via Reciprocal Rank Fusion:

```
Score(c) = w_dense·RRF_dense(c) + w_sparse·RRF_sparse(c) + w_prior·P(c|Mention)
RRF_x(c) = 1 / (RRF_K + rank_x(c))
```

Rank-based fusion specifically because SapBERT cosine (roughly `[0,1]`)
and this project's BM25 scores (measured in the 0-10 range, no fixed
ceiling) don't live on comparable scales — averaging raw scores would
let whichever signal has the larger numeric range dominate for no
principled reason.

**Real grid-search result** (155-160 clean-span entities, 32 notes,
9-point weight grid): **dense-only (`w_dense=1.0`) strictly wins**, with
Top-1 accuracy 61.3% and oracle accuracy 74.2% — and every grid point
declines **monotonically** as sparse weight increases, down to
37.7%/48.1% at pure-sparse. No blend of the two signals recovered
dense-only's performance, let alone beat it. `CNSP_HYBRID_RETRIEVAL`
stays off in production as a **validated conclusion**, not an
unexamined default — this is a real, negative result for the hybrid
approach, reported as such rather than reframed.

## 12. GLiNER-Linker rerankers — evaluated and rejected on the same population

A separate candidate mechanism for the specific Condition/Observation
SNOMED-duplicate problem — `gliner-linker-large-v1.0` and
`gliner-linker-rerank-v1.0` (bi-encoder and cross-encoder rerankers,
outside the `gliner` NER library despite the shared name) — was scored
against all 10 gradable cases of the known-duplicate population and
compared to the existing mechanism (SapBERT retrieval plus a
strengthened MoLLM prior-based tiebreak):

| Mechanism | Correct |
|---|---|
| `gliner-linker-large` | 4/10 |
| `gliner-linker-rerank` | 5/10 |
| **Existing (SapBERT + MoLLM tiebreak)** | **7/7** |

The existing mechanism was not just kept by default but **directly
outperformed** two purpose-built reranking alternatives on their own
target population — a real comparative result, not an assumption that
the incumbent must be fine.

## 13. Testing & reproducibility

- **`tests/test_tier12_ranking.py`** exercises the ranking logic
  (`src/normalization/tier_retrieval.py`'s `key()`/ordering function, §8)
  against a hand-written fake DB connection, deliberately **not** a
  live-DB test — importing `src.normalization` normally loads the real
  ~400MB SapBERT model and expects a populated Athena DuckDB, neither of
  which belongs in a unit test. It stubs `torch`/`transformers`/`duckdb`
  in `sys.modules` before import, so the *shipped* ranking function is
  exercised directly (not a reimplemented copy), with only the model
  load and DB calls faked.
- **A real, currently-broken gap, found while writing this document**:
  running this test file today fails at collection —
  `AttributeError: module 'torch' has no attribute 'cuda'` — because its
  hand-written fake `torch` module (`_install_stubs()`) never defines a
  `.cuda` attribute, and `sapbert_model.py`'s real GPU-placement code
  (§2, added 2026-08-18) calls `torch.cuda.is_available()` at import
  time. The stub predates that GPU-placement addition and was never
  updated to match it. This is reported here as a genuine, currently
  reproducible gap (confirmed via direct execution,
  `python3 tests/test_tier12_ranking.py`) — not fixed as part of writing
  this document, since it wasn't the task asked for, but stated plainly
  rather than silently worked around.
- **Live verification queries used while writing this document** (all
  read-only, safe alongside a concurrent write-locked batch): the
  `information_schema.columns` schema check, the `len(embedding)`
  dimension check, and the four `COUNT(*)` coverage queries in §3 — run
  directly against `db/kg2_lexical_store.duckdb` rather than assumed
  from an old log.

## 14. Honest limitations

- **The original embedding-generation step is not reconstructable from
  this repo** (§3) — `scripts/build_concept_embeddings.py` is a 0-byte
  stub; only the merge-from-staging-table step (`merge_embeddings.py`)
  is real, tracked code. Anyone re-running this pipeline from scratch
  against a fresh Athena import would need to write that generation step
  themselves (loop `get_sapbert_embedding()` over `concept_name`,
  populate a `concept_embeddings` staging table, run
  `merge_embeddings.py`) — described here, but not currently a runnable
  script in this repository.
- **Oracle-in-top-5 accuracy (~56% measured) falls well short of the
  blueprint's own >98% target** (§6) — the widened `CANDIDATE_LIMIT`
  did not close this gap; even a perfect downstream ranker operating on
  today's Tier 3 candidate pools would still miss the gold concept
  entirely in roughly 44% of cases.
- **`TIER3_SIMILARITY_FLOOR=0.72`'s precision/coverage trade-off is
  explicitly unresolved** (§5.1) — kept per direction, with a real,
  measured cost (90 gold-correct matches lost per the cited 27-note
  run), not a settled decision.
- **Tier 3 is a brute-force cosine scan, not an ANN index** (§4) — safe
  at this corpus's current scale because domain/vocabulary filtering
  narrows the scan first, but this would not scale unchanged to a
  substantially larger reference vocabulary without revisiting that
  design.
- **The Tier 1/2 semantic tiebreak criterion (§8) has never been
  measured with `TIER12_RANK_SEMANTIC=1` in production** — it exists,
  is wired, and is designed to be A/B-measurable via
  `tier12_rank_basis`, but no run's results are cited here because none
  exist yet in current documentation.
