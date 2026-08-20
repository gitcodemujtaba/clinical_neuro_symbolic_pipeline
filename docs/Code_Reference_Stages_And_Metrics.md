# Code Reference — Stage-Wise Core Logic & Metric Formulas

Companion to `docs/Implementation_Methodology.md` (architecture narrative)
and `docs/2026-08-20_Session_Results_And_Status.md` (results + decision
log). This document pulls the actual, real, currently-shipping code for
each pipeline stage's core mechanism, plus every metric formula used to
grade the pipeline — for direct inclusion in the paper's methodology
section. Every snippet below is copied verbatim from the source file
named above it, not reconstructed from memory.

---

## Stage 1 — Preprocessing / Abbreviation Expansion

`src/preprocessing.py :: expand_text_and_track_offsets()`

Core idea: an abbreviation with multiple possible meanings is resolved
through an ordered tiebreak chain — a mined context rule, then numeric
context, then an observed-frequency prior (allow-listed abbreviations
only), then OMOP groundability — and left **unexpanded** rather than
guessed at if none of them resolve it.

```python
def expand_text_and_track_offsets(text: str, abbrev_dict: dict, conn=None):
    """Expands known abbreviations, tracking original<->expanded offsets.

    abbrev_dict maps abbreviation -> LIST of meanings. When an abbreviation
    has more than one known meaning, up to four tiebreaks are tried in
    order, each only consulted when every one before it declined to pick
    (returned None):
      (1) a mined context-pattern rule from real reviewer-confirmed data
          (src.abbreviation_flywheel.select_by_context_pattern())
      (2) numeric context around the token
          (_numeric_context_kind / _select_by_numeric_context)
      (3) the pipeline's own observed-frequency prior, excluding
          abbreviations with known systematic bias
          (src.abbreviation_flywheel.compute_frequency_priority())
      (4) OMOP groundability (_select_by_groundability) -- BUT ONLY if the
          abbreviation is on VERIFIED_ALLOW_LIST (real gold-checked data
          measured alphabetical_default correct only 20.1% of the time and
          omop_groundability only 53.1%, neither safe as a silent default)

    If none of (1)-(3) resolve it AND the abbreviation isn't allow-listed,
    the token is left UNEXPANDED (selection_basis
    "unvetted_ambiguous_unexpanded") rather than guessed at.
    """
```

---

## Stage 2a — Extraction (GLiNER-BioMed)

`src/entity_extraction.py :: extract_and_store_entities()`

Core idea: a cheap word-count pre-check (GLiNER's own tokenizer, no model
inference) decides chunked-vs-single BEFORE the expensive model call, so
long notes never silently truncate.

```python
all_words = list(model.data_processor.words_splitter(expanded_text))
if len(all_words) > CHUNK_WORD_BUDGET:
    raw_entities, possibly_truncated, gliner_input_token_count = (
        _extract_entities_chunked(expanded_text, sentences, floor))
else:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_entities = model.predict_entities(
            expanded_text, CLINICAL_LABELS, threshold=floor, flat_ner=FLAT_NER
        )
    possibly_truncated = False
    gliner_input_token_count = None
    for w in caught:
        m = re.search(r"Sentence of length (\d+) has been truncated to (\d+)", str(w.message))
        if m:
            possibly_truncated = True
            gliner_input_token_count = int(m.group(1))
            break

# Assertion detection over the SAME text/coordinate system GLiNER produced
# the spans in -- no offset mapping happens inside src/assertion.py.
spans = [(e["start"], e["end"], e["label"]) for e in raw_entities]
assertions = annotate_assertions(expanded_text, spans)
```

Chunking (`_build_chunks()`, `_extract_entities_chunked()`): 128-token
overlap, sentence-boundary-snapped, `CHUNK_WORD_BUDGET=1800` (margin under
GLiNER's real ceiling `model.config.max_len=2048`). Measured directly
against the corpus's longest note (24,858 chars): single-call extraction
found 124 entities (truncated); chunked found 399 — recovering 282 real
clinical entities.

---

## Stage 2b — Normalization / Grounding

### Tiered candidate retrieval

`src/normalization/tier_retrieval.py :: _tier_queries()` — Tier 1 (exact
concept-name match), Tier 2 (exact synonym match), Tier 3 (SapBERT dense
semantic similarity, gated at `TIER3_SIMILARITY_FLOOR`):

```python
rows = conn.sql(f"""
    SELECT concept_id, concept_name, domain_id, vocabulary_id
    FROM athena_concept
    WHERE lower(concept_name) = ? AND standard_concept = 'S'
    AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause} {_UK_EXTENSION_EXCLUSION}
    ORDER BY concept_id ASC LIMIT {CANDIDATE_LIMIT};
""", params=[search_text, *vocabs, *(domains or [])]).fetchall()
```

### SNOMED regional-extension exclusion (2026-08-20 fix)

The core discovery: `vocabulary_id` cannot discriminate International
Release from regional extensions (only one distinct value exists in the
whole table); `concept_class_id` alone is unreliable (23,842 extension
concepts are themselves `'Procedure'` class). The real signal is the
SCTID's own namespace-identifier block:

```python
# 2026-08-20: regional-extension SCTID exclusion. OMOP bundles every SNOMED
# national extension (UK, in this dump) into the same vocabulary_id='SNOMED'
# string as the International Release -- confirmed empirically, only one
# distinct vocabulary_id value exists in athena_concept.
#
# The real, robust signal is the SCTID itself: extension concepts carry a
# 7-digit namespace-identifier block (UK's is reserved as "1000000") between
# the item-identifier and the 2-digit partition+check-digit suffix, which
# International Release concepts never have. Verified against the live DB:
# 98,487 concepts (9% of the whole SNOMED table) match this pattern, and
# ZERO of our 4,522 distinct gold-standard SNOMED codes are among them.
_UK_EXTENSION_EXCLUSION = "AND concept_code NOT LIKE '%1000000___'"
```

### Procedure-vs-near-duplicate class preference

`src/normalization/tier_retrieval.py :: _prefer_lab_procedure_over_observable()`
— rank-only penalty (never changes the displayed similarity_score) for a
Lab-Test entity where a Procedure-class candidate competes against an
Observable-Entity or Qualifier-Value-class near-duplicate describing the
same test:

```python
_LAB_PROCEDURE_CLASS_BONUS = 0.1
_LAB_PROCEDURE_PENALIZED_CLASSES = {"Observable Entity", "Qualifier Value"}

def _prefer_lab_procedure_over_observable(conn, cands, gliner_label):
    if gliner_label != "Lab Test" or len(cands) < 2:
        return cands
    ids = [c["omop_concept_id"] for c in cands]
    classes = dict(conn.sql(
        f"SELECT concept_id, concept_class_id FROM athena_concept "
        f"WHERE concept_id IN ({','.join('?'*len(ids))})", params=ids).fetchall())
    has_observable = any(classes.get(c["omop_concept_id"]) in _LAB_PROCEDURE_PENALIZED_CLASSES for c in cands)
    has_procedure = any(classes.get(c["omop_concept_id"]) == "Procedure" for c in cands)
    if not (has_observable and has_procedure):
        return cands

    def _sort_key(c):
        penalty = _LAB_PROCEDURE_CLASS_BONUS \
            if classes.get(c["omop_concept_id"]) in _LAB_PROCEDURE_PENALIZED_CLASSES else 0.0
        return c["similarity_score"] - penalty

    return sorted(cands, key=_sort_key, reverse=True)
```

### KG-embedding (TransE) topological tiebreak

`src/kg_embedding.py :: TransE` — standard TransE (Bordes et al. 2013),
`score(h, r, t) = -||h + r - t||₂`:

```python
class TransE(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int = 100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self._normalize_entities()  # unit L2 norm after every optimizer step

    def score(self, h, r, t):
        return -torch.norm(self.entity_emb(h) + self.relation_emb(r) - self.entity_emb(t), p=2, dim=-1)
```

`src/kg_embedding_tiebreak.py :: kg_tiebreak_score()` — the tiebreak
signal: mean embedding distance from a tied candidate to the REST of that
entity's own SapBERT-proposed candidate pool (not a GLiNER-label centroid,
not a neighboring-entity anchor):

```python
def kg_tiebreak_score(model, entity2idx, candidate_id, pool_concept_ids):
    if candidate_id not in entity2idx:
        return None
    others = [c for c in pool_concept_ids if c != candidate_id and c in entity2idx]
    if not others:
        return None
    with torch.no_grad():
        c_emb = model.entity_emb(torch.tensor(entity2idx[candidate_id]))
        other_emb = model.entity_emb(torch.tensor([entity2idx[o] for o in others]))
        distances = torch.norm(other_emb - c_emb.unsqueeze(0), p=2, dim=1)
        return distances.mean().item()

def pick_via_kg_tiebreak(model, entity2idx, tied_concept_ids, full_pool_concept_ids):
    rest_of_pool = [c for c in full_pool_concept_ids if c not in tied_concept_ids]
    scores = {cid: kg_tiebreak_score(model, entity2idx, cid, rest_of_pool) for cid in tied_concept_ids}
    usable = {cid: s for cid, s in scores.items() if s is not None}
    if len(usable) < 2:
        return {"winner": None, "scores": scores, "resolved": False}
    return {"winner": min(usable, key=usable.get), "scores": scores, "resolved": True}
```

**Not wired into production** — evaluated (§ Metrics below) and found
strictly less safe than the hardcoded rule on the rule's own pattern.

---

## Stage 3 — MoLLM Tier Gate

`src/mollm_tier_gate.py` — three local Ollama models (qwen2.5:3b,
llama3.2:3b, phi4-mini) independently vote; `route_tier()` combines them:

```python
TIER_1_AUTO_VALIDATED = "TIER_1_AUTO_VALIDATED"
TIER_2_AUTO_RESOLVED = "TIER_2_AUTO_RESOLVED"
TIER_3_AUTO_VALIDATED = "TIER_3_AUTO_VALIDATED"
TIER_4_ENSEMBLE_SPLIT = "TIER_4_ENSEMBLE_SPLIT"
TIER_5_TRUE_AMBIGUITY = "TIER_5_TRUE_AMBIGUITY"

AUTO_TIERS = {TIER_1_AUTO_VALIDATED, TIER_3_AUTO_VALIDATED,
             TIER_1B_CALIBRATED_AUTO_VALIDATED}
```

Core voting/routing logic (`route_tier()`):

```python
verdicts = [m["verdict"] for m in usable]
vote_counts = collections.Counter(verdicts)
top_verdict, top_count = vote_counts.most_common(1)[0]

confs = [m["logprob_confidence"] for m in usable
         if m.get("logprob_confidence") is not None and m["verdict"] == top_verdict]
composite_confidence = round(sum(confs) / len(confs), 6) if confs else None

unanimous = len(usable) == 3 and top_count == 3

if unanimous and top_verdict == "SUPPORTED_1":
    # Hard safety gate checked BEFORE granting AUTO status -- unanimity
    # does not protect against a known-fragile candidate list.
    trapped, trap_reason = _fragile_shorthand_trap(entity, 1, entity.get("candidates") or [])
    if trapped:
        return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED", ...}
    if composite_confidence is not None and composite_confidence < TIER1_CONFIDENCE_FLOOR:
        return {"tier": None, "mollm_routing_decision": "HITL_REQUIRED",
               "queue_reason": "below_confidence_threshold", ...}
    return {"tier": TIER_1_AUTO_VALIDATED, "mollm_routing_decision": "AUTO_VALIDATED",
           "final_candidate_index": 1, "composite_confidence": composite_confidence, ...}
```

`TIER_2_AUTO_RESOLVED` (3/3 unanimous re-rank to the same candidate N≠1)
is **deliberately excluded from `AUTO_TIERS`** — 100% of Tier 2 decisions
are on `is_ambiguous=True` retrieval, meaning unanimous agreement more
likely reflects shared model bias than independent verification.

### Exhaustive-candidate-eval tiebreak-eligible detection

`evaluation/exhaustive_candidate_eval_impact.py` — the population
`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` (`src/mollm_tier_gate.py`) creates,
detected directly from a stored decision's `eval_trail`:

```python
def is_tiebreak_eligible(models: list) -> bool:
    for m in models or []:
        trail = m.get("eval_trail") or []
        n_accepts = sum(1 for t in trail if t.get("match"))
        if n_accepts >= 2:
            return True
    return False
```

---

## Stage 4 — Routing & Ingestion

`src/kg3_ingestion.py` — `ingest_auto_decision()` imports `AUTO_TIERS`
directly from `mollm_tier_gate` (a real bug of the two silently drifting
apart was found and fixed once already, hence this discipline):

```python
from src.mollm_tier_gate import AUTO_TIERS

if decision.get("tier") in AUTO_TIERS and memgraph_driver is not None:
    write_result = ingest_auto_decision(memgraph_driver, decision, entity_fields, dry_run=True)
```

Everything else enqueues into `hitl_review_queue` for human review. **All
writes remain `dry_run=True`** — no code path writes to KG3 unreviewed and
live today.

---

## Metrics & Formulas

### 1. Span overlap primitive

`scripts/score_gold_recall.py :: overlaps()` — "any character overlap,"
not exact-span match, used by every recall/precision computation below:

```python
def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start
```

### 2. Span recall & linked recall (Stage 1/2 completeness)

`scripts/score_gold_recall.py :: score()`:

```
span_recall   = span_covered   / n_gold   (a gold span has ANY overlapping prediction)
linked_recall = linked_correct / n_gold   (that overlapping prediction's concept == gold's concept)
```

```python
for g in gold:
    overlapping = [p for p in preds if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
    if not overlapping:
        continue  # missed span
    span_covered += 1
    hit = next((p for p in overlapping if p["snomed_code"] == g["concept_id"]), None)
    if hit:
        linked_correct += 1

per_note["span_recall"] = span_covered / n_gold if n_gold else 0.0
per_note["linked_recall"] = linked_correct / n_gold if n_gold else 0.0
```

### 3. AUTO-tier precision (the paper's headline metric)

Not a single named function — computed per grading script as: of decisions
that landed in `AUTO_TIERS` (auto-written, no human review) AND have a
clean single gold-span overlap, what fraction picked gold's exact SNOMED
concept:

```python
picked = candidates[final_candidate_index - 1]
snomed = vocab.snomed_code_for_concept(picked["omop_concept_id"])
correct = snomed is not None and str(snomed) == str(gold_concept_id)

if tier in AUTO_TIERS:
    auto_n += 1
    if correct:
        auto_correct += 1

auto_tier_precision = auto_correct / auto_n   # e.g. 43/56 = 76.8% (fresh-10-note validation)
```

"Clean span" gate applied before grading any entity (used identically
across every script in this codebase):

```python
overlapping = [g for g in gold if overlaps(pred_start, pred_end, g["start"], g["end"])]
if len(overlapping) != 1:
    continue  # skip: no gold match, or ambiguous multi-overlap
g0 = overlapping[0]
if (pred_end - pred_start) < (g0["end"] - g0["start"]):
    continue  # skip: our span is narrower than gold's -- not a clean match
```

### 4. Set IoU (decision-level, per stage)

`evaluation/iou_metrics.py :: set_iou()` — the generalization of
Jaccard/IoU to any binary correct/incorrect decision set, not just
bounding boxes:

```python
def set_iou(tp, fp, fn):
    denom = tp + fp + fn
    return round(tp / denom, 4) if denom else None
```

TP/FP/FN are defined per stage: span-is-real (2a), concept-is-right (2b),
AUTO-tier-decision-is-right (3) — never compared directly across stages.

### 5. Benchmark character IoU (DrivenData SNOMED-CT challenge definition)

`evaluation/iou_metrics.py :: char_iou_by_class()` / `benchmark_char_iou()`
— confirmed against the benchmark's own metric section: "class" = SNOMED
concept ID; a predicted span's characters only count toward a class if its
OWN resolved concept matches exactly (relationships between concepts are
not scored):

```
IoU_class = |chars(pred) ∩ chars(gold)| / |chars(pred) ∪ chars(gold)|
macro_char_iou    = mean(IoU_class for class in classes)               # unweighted
weighted_char_iou = sum(IoU_class * gold_support) / sum(gold_support)   # weighted by gold span count
```

```python
def char_iou_by_class(pred_spans, gold_spans):
    # pred_spans / gold_spans: list of (label, note_id, start, end)
    classes = set(l for l, _, _, _ in pred_spans) | set(l for l, _, _, _ in gold_spans)
    per_class = {}
    for cls in classes:
        pred_chars = set((nid, p) for l, nid, s, e in pred_spans if l == cls for p in range(s, e))
        gold_chars = set()
        gold_support = 0
        for l, nid, s, e in gold_spans:
            if l == cls:
                gold_chars.update((nid, p) for p in range(s, e))
                gold_support += 1
        union = pred_chars | gold_chars
        inter = pred_chars & gold_chars
        iou = len(inter) / len(union) if union else None
        per_class[cls] = {"iou": iou, "gold_support": gold_support}

    valid = [(c, v) for c, v in per_class.items() if v["iou"] is not None]
    macro = sum(v["iou"] for _, v in valid) / len(valid) if valid else None
    total_support = sum(v["gold_support"] for _, v in valid)
    weighted = (sum(v["iou"] * v["gold_support"] for _, v in valid) / total_support
                if total_support else None)
    return macro, weighted, per_class
```

Character positions keyed by `(note_id, offset)`, not raw offset alone —
prevents cross-note collision when scoring several notes at once.

### 6. Expected Calibration Error (ECE)

`evaluation/metrics.py :: compute_ece_report()`:

```
ECE = Σ over bins of (bin_size / N) * |accuracy − mean_confidence|
```
0 = perfectly calibrated; higher = confidence is systematically
misleading.

```python
n = len(clean)
ece = 0.0
mce = 0.0
for b, (lo, hi) in zip(bins, edges):
    if not b:
        continue
    mean_conf = sum(c for c, _ in b) / len(b)
    acc = sum(1 for _, ok in b if ok) / len(b)
    gap = abs(acc - mean_conf)
    ece += (len(b) / n) * gap
    mce = max(mce, gap)
```

Brier score (a **proper** scoring rule, reported alongside ECE since ECE
alone can be gamed by a model that always predicts the base rate):
`brier = mean((confidence - outcome)²)`.

### 7. KGE intrinsic evaluation — MRR & Hits@k

`src/kg_embedding.py :: evaluate_link_prediction()` — standard KGE
literature protocol, RAW setting (not filtered):

```
MRR = mean(1 / rank_of_true_tail)   over held-out triples
Hits@k = fraction of held-out triples where rank_of_true_tail <= k
```

```python
for h, r, t in sample:
    h_idx, r_idx, t_idx = entity2idx[h], relation2idx[r], entity2idx[t]
    scores = model.score(h_idx.expand(len(all_entities)), r_idx.expand(len(all_entities)), all_entities)
    rank = (scores > scores[t_idx]).sum().item() + 1
    reciprocal_ranks.append(1.0 / rank)
    if rank <= k:
        hits += 1

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
hits_at_k = hits / len(reciprocal_ranks)
```

Measured (post-backfill retrain, 455 TP records): **MRR = 0.776, Hits@10 = 0.909**.

### 8. KGE extrinsic evaluation (task-specific, not standard KGE literature)

`src/kg_embedding.py :: evaluate_against_tp_records()` — does the
embedding space separate a real competing-wrong candidate from a random
unrelated concept more than chance:

```python
d_wrong = distance(correct_concept, wrong_candidate)   # a real competitor for the same mention
d_random = distance(correct_concept, random_unrelated_concept)
# A useful embedding should show d_wrong < d_random more often than not
frac_wrong_closer_than_random = count(d_wrong < d_random) / n_comparisons
```

### 9. KGE tiebreak win/loss/neutral classification

`evaluation/kg_tiebreak_validation.py :: classify_outcome()` — the metric
used to compare KGE against the hardcoded rule:

```python
def classify_outcome(baseline_correct, new_correct):
    if baseline_correct and not new_correct:
        return "loss"      # the fatal case -- mechanism broke a correct answer
    if not baseline_correct and new_correct:
        return "win"       # mechanism fixed a wrong answer
    return "neutral"       # both right, both wrong, or no change
```

Swept over `TIE_THRESHOLD` (the SapBERT top1/top2 score delta that defines
"tied"): 0.01 through 0.08. Head-to-head against the hardcoded rule
restricted to the subset where both mechanisms apply
(`hardcoded_rule_applicable()`):

```python
def hardcoded_rule_applicable(entity_label, top1_class, top2_class):
    if entity_label != "Lab Test":
        return False
    classes = {top1_class, top2_class}
    return "Procedure" in classes and bool(classes & {"Observable Entity", "Qualifier Value"})
```

---

## Summary Table — Metric → Formula → Where Computed

| Metric | Formula | Module |
|---|---|---|
| Span recall | `span_covered / n_gold` | `scripts/score_gold_recall.py` |
| Linked recall | `linked_correct / n_gold` | `scripts/score_gold_recall.py` |
| AUTO-tier precision | `auto_correct / auto_n` (clean-span, gold-crosswalked) | grading scripts (this codebase-wide convention) |
| Set IoU | `TP / (TP + FP + FN)` | `evaluation/iou_metrics.py` |
| Benchmark char IoU | `|chars(pred) ∩ chars(gold)| / |chars(pred) ∪ chars(gold)|` per SNOMED concept class | `evaluation/iou_metrics.py` |
| ECE | `Σ (bin_size/N) · |accuracy − mean_confidence|` | `evaluation/metrics.py` |
| Brier score | `mean((confidence − outcome)²)` | `evaluation/metrics.py` |
| KGE MRR | `mean(1/rank_of_true_tail)` | `src/kg_embedding.py` |
| KGE Hits@k | `fraction(rank ≤ k)` | `src/kg_embedding.py` |
| KGE extrinsic | `count(d_wrong < d_random) / n_comparisons` | `src/kg_embedding.py` |
| KGE tiebreak win/loss | baseline-wrong→right = win; baseline-right→wrong = loss | `evaluation/kg_tiebreak_validation.py` |
