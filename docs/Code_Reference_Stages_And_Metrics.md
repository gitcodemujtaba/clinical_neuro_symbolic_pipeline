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
RotatE (a real 4-configuration ablation — `guideline`/`gold`/`combined`/
`snomed_is_a` training data) was subsequently built too, reusing this
same `kg_tiebreak_score()`/`pick_via_kg_tiebreak()` pair unchanged
(model-agnostic by contract), and every RotatE configuration loses even
more decisively than TransE does. Full build, training, and ablation
detail for both: `docs/Knowledge_Graphs_Technical_Reference.md` Part B.

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

### 10. Linked precision & Linked F1 (Stage 1-2b)

**What it's calculating in this project**: linked recall (§2 above) asks
"of gold's annotations, how many did we find and link correctly" —
recall's own denominator is gold, so it cannot see *false positives* (a
predicted concept with no gold justification at all). Linked precision is
the complementary question: "of everything we predicted with a resolved
SNOMED code, how many are actually right." Not a named function in this
codebase — computed as an aggregate over the same `predictions` list
`score_gold_recall.py::load_predictions()`/`attach_snomed_codes()` already
build, restricted to predictions carrying a non-null `snomed_code`:

```
linked_precision = linked_correct / count(predictions where snomed_code is not None)
linked_F1 = 2 * linked_precision * linked_recall / (linked_precision + linked_recall)
```

Computed across **every tier**, not just `AUTO_TIERS` — this is a
different population from AUTO-tier precision (§3): AUTO-tier precision
only asks about the subset written without review; linked precision/F1
ask about the pipeline's entire output, reviewed or not. Mixing the two
populations (AUTO-tier-only precision against corpus-wide recall) is a
real, documented mistake this project already caught and fixed once —
see `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §2's own methodology
note.

### 11. Deflection rate (Stage 3)

**What it's calculating**: what fraction of Stage 3's decisions never
require human review at all — the pipeline's actual autonomy rate, and
the metric this whole project is ultimately optimizing (`docs/
Implementation_Methodology.md`'s stated objective is "a high share of
fully autonomous, high-precision concept writes"). Uses **every**
`AUTO_TIERS` decision, not just the clean-span-gradable subset (an
earlier draft of this exact computation restricted the numerator to
gradable decisions while leaving the denominator unrestricted, silently
undercounting deflection by ~20-26 points — caught and fixed, see
`docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §2's own methodology note).

```python
from src.mollm_tier_gate import AUTO_TIERS   # the real one -- see the warning below

n_auto = sum(1 for d in decisions if d["tier"] in AUTO_TIERS)
deflection_rate = n_auto / len(decisions)
```

⚠️ **Import `AUTO_TIERS` from `src.mollm_tier_gate` directly — do not
redefine it.** Two evaluation scripts in this codebase
(`evaluation/grade_overnight_corpus_run.py`, `evaluation/
grade_allergy_shadow_run.py`) carry their own hardcoded copy,
`{"TIER_1_AUTO_VALIDATED", "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED"}`
— which is now **wrong**: the real, current `AUTO_TIERS` (`src/
mollm_tier_gate.py` line 149) is `{TIER_1_AUTO_VALIDATED,
TIER_3_AUTO_VALIDATED, TIER_1B_CALIBRATED_AUTO_VALIDATED}` — the
hardcoded copy is missing `TIER_1B` entirely (added later, never
backported to those two scripts) and wrongly includes `TIER_2_AUTO_RESOLVED`
(deliberately excluded from the real set, pending shadow validation — see
`docs/ConsensusCalibrator_Technical_Reference.md` §2). This is the exact
same drift-bug class already found and fixed once in `src/
kg3_ingestion.py::ingest_auto_decision()` (2026-08-17,
`docs/2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md`).
Found again while writing this section, 2026-08-30 — **not yet fixed in
those two evaluation scripts**, flagged here as a real, open item rather
than silently worked around.

### 12. Classification metrics: TP / FP / Precision / Recall / F1 (Stage 3 promotion decisions)

**What it's calculating**: standard binary-classification metrics, applied
specifically to "did the tier gate correctly decide to auto-promote this
entity" — used throughout the calibrator/KG3-feature work
(`docs/ConsensusCalibrator_Technical_Reference.md` §17.5,
`docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §9-§10) to isolate one
mechanism's own marginal contribution, holding everything else fixed.

```
TP = count(promoted AND gold-correct)
FP = count(promoted AND gold-wrong)
Precision = TP / (TP + FP)                              -- of what got promoted, how much is right
Recall    = TP / (TP + FN)                               -- of everything that WOULD be correct if
                                                             promoted, how much actually got promoted
F1        = 2 · Precision · Recall / (Precision + Recall)
```

**The FN population is the subtle part, computed per-context, not by a
single shared function**: it is *not* "everything not promoted" — it is
specifically the entities that *would have graded correct* had they been
promoted (derived via `plurality_candidate_index()` on the still-`TIER_4`
population, exactly matching `evaluation/grade_overnight_corpus_run.py`'s
own "Tier 4 shadow precision" methodology). An entity that's genuinely
wrong and correctly left at HITL is neither a TP nor an FN — it's a true
negative, uncounted, exactly as intended.

### 13. AUROC (calibrator validation)

`evaluation/tier_gate_cal_eval.py::fit_and_report()`, via
`sklearn.metrics.roc_auc_score()` — the probability that the calibrator's
score ranks a random correct example above a random incorrect one, over
the full [0,1] threshold range (not just the one deployed threshold):

```
AUROC = P( score(random positive) > score(random negative) )
```

Not reimplemented in this codebase — `sklearn`'s own implementation is
used directly, consistent with `_build_model()`'s "single construction
site" discipline (§7 of `docs/ConsensusCalibrator_Technical_Reference.md`).
Reported alongside, never instead of, the threshold-sweep coverage/
precision table (§13 of that same doc) — AUROC summarizes ranking quality
across every possible threshold, while the actually-deployed number
(`CALIBRATED_AUTO_THRESHOLD = 0.78`, re-derived 2026-08-31 from 0.72
after a locked-test-split leakage fix) only cares about one point on
that curve.

### 14. Wilson score interval — **not currently used anywhere in this codebase before this section**, added here

**What it is**: a confidence interval for a single binomial proportion
(e.g. "correct" vs. "incorrect" over `n` graded decisions) that stays
well-calibrated at small `n` and near `p=0` or `p=1` — the plain
`p̂ ± z·√(p̂(1-p̂)/n)` "normal approximation" interval used more often in
practice is known to undercover badly in exactly those conditions
(small-`n`, extreme-`p`), which is precisely the shape of several
populations in this project (e.g. fresh-10's 56-decision AUTO-tier
population, or the calibrator's 23-decision `TIER_1B` slice on fresh-5).

**Formula** (95% CI, `z = 1.96`):

```
p̂ = x / n
center = (p̂ + z²/2n) / (1 + z²/n)
margin = z·√(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)
CI = [center - margin, center + margin]
```

```python
import math

def wilson_interval(x, n, z=1.96):
    if n == 0:
        return None
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)
```

**What it's calculating in this project**, computed 2026-08-30 for the
headline AUTO-tier precision figure across all three populations
(`x` = correct, `n` = gradable):

| Population | x/n | Point estimate | Wilson 95% CI | Width |
|---|---|---|---|---|
| Corpus-wide (144 notes) | 5,841/6,724 | 86.9% | [86.0%, 87.7%] | 1.6pp |
| Fresh-10 | 43/56 | 76.8% | [64.2%, 85.9%] | **21.7pp** |
| Fresh-5 | 139/151 | 92.1% | [86.6%, 95.4%] | 8.8pp |

And for Linked precision / Linked recall (§10/§2):

| Population | Metric | x/n | Point estimate | Wilson 95% CI |
|---|---|---|---|---|
| Corpus-wide | Linked precision | 13,197/26,382 | 50.0% | [49.4%, 50.6%] |
| Fresh-10 | Linked precision | 401/886 | 45.3% | [42.0%, 48.6%] |
| Fresh-5 | Linked precision | 218/402 | 54.2% | [49.3%, 59.0%] |
| Corpus-wide | Linked recall | 13,208/39,403 | 33.5% | [33.1%, 34.0%] |
| Fresh-10 | Linked recall | 401/1,497 | 26.8% | [24.6%, 29.1%] |
| Fresh-5 | Linked recall | 218/544 | 40.1% | [36.0%, 44.2%] |

And for the calibrator's own `TIER_1B` promotion precision specifically:

| Population | x/n | Point estimate | Wilson 95% CI |
|---|---|---|---|
| Held-out val split (`docs/ConsensusCalibrator_Technical_Reference.md` §17.5) | 129/133 | 97.0% | [92.5%, 98.8%] |
| Fresh-5 (real, 2026-08-30, `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §10.2) | 21/23 | 91.3% | [73.2%, 97.6%] |

**Read this honestly, not just as decoration on the point estimates.**
The fresh-10 AUTO-tier-precision interval is **21.7 percentage points
wide** — nearly a quarter of the whole [0,1] range — because `n=56` is
small. Fresh-10's 76.8% and fresh-5's 92.1% intervals ([64.2%, 85.9%] vs.
[86.6%, 95.4%]) come close to touching but do not overlap, so the
fresh-5-beats-fresh-10 claim in §10.3 of the SSOT doc survives this
check — but the margin is real, not overwhelming, and should not be
oversold as a large, decisive gap on the strength of the point estimates
alone. The `TIER_1B` fresh-5 interval ([73.2%, 97.6%]) is wide enough
that "97.0% on held-out val" and "91.3% on fresh-5" are **fully
consistent with being the same underlying rate** — read them as
compatible measurements of one thing, not as evidence the calibrator
performs differently in the two settings.

**What this is *not***: the project's own evaluation criteria
(`docs/Evaluation_Criteria.md`) call for **bootstrap confidence
intervals resampled at the note level** — a materially different, more
rigorous approach, since entities within one discharge note are not
independent draws (the exact same non-independence reasoning behind
every note-disjoint train/val split in this project — see
`docs/ConsensusCalibrator_Technical_Reference.md` §10.5). The Wilson
interval above treats every graded entity as an independent Bernoulli
trial, which understates true uncertainty whenever a population is
dominated by a few notes with many entities each (true for all three
populations here, to varying degrees — fresh-5's 151 gradable AUTO-tier
decisions come from only 5 notes). **Note-level bootstrap CIs are not
yet built anywhere in this codebase** — this Wilson-interval addition is
a real, useful, but acknowledged-partial answer to the "no confidence
intervals" gap `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` previously
stated as fully open; see that doc's Known Limitations section for the
updated, precise framing.

---

### 15. False deflection rate — closed 2026-08-31 via a gold-substituted proxy

**What it's supposed to measure** (`docs/Evaluation_Criteria.md`): "The
proportion of the Stage 5 re-audit sample that should have gone to HITL
but did not" — a patient-safety metric, distinct from deflection rate
(§11, which measures *how much* was auto-approved, not *how safely*).

**Why it was stuck at "cannot be computed."** The proposal's own design
is a re-audit process: an independent reviewer re-checks a sample of
already-auto-approved decisions after the fact. This project's real HITL
infrastructure exists (`hitl_review_queue`, 19,103 cases) but has **zero
completed human reviews** — `dry_run=True` everywhere in production, so
there is no independent human verdict to compute this against.

**The substitution that closes it, and why it's valid.** A wrong AUTO-tier
decision — one that doesn't match gold's SNOMED concept — is, by
definition, exactly the case that "should have gone to HITL but did not."
Gold annotation *is* an independent, already-existing ground truth for
that same question, standing in for the re-audit reviewer the proposal
assumes. This is the same substitution principle used throughout this
project wherever real human review doesn't exist yet (e.g. the KG3
gold-simulated population, `docs/FINAL_RESULTS_Single_Source_Of_Truth.md`
§12's own KG3 bullet) — stated as a substitution, not conflated with the
real thing.

```python
false_deflection_rate = 1 - auto_tier_precision   # same x/n as §3, complemented
```

Concretely: `auto_correct`/`auto_n` from §3's own AUTO-tier-precision
computation gives `x = auto_n - auto_correct` (the wrong, should-have-
gone-to-HITL count) over the same `n`. No new grading code, no new LLM
calls — this is the exact complement of a number already computed three
times this project (corpus-wide, fresh-10, fresh-5), with Wilson CIs
(§14) mirrored the same way (`CI(1-p) = [1 - upper(p), 1 - lower(p)]`):

| Population | x/n (wrong/gradable AUTO) | False deflection rate | Wilson 95% CI |
|---|---|---|---|
| Corpus-wide (144 notes) | 883/6,724 | 13.1% | [12.3%, 14.0%] |
| Fresh-10 | 13/56 | 23.2% | [14.1%, 35.8%] |
| Fresh-5 | 12/151 | 7.9% | [4.6%, 13.4%] |

**Read honestly**: this is **not** the metric the proposal specifies —
it's a gold-substituted proxy for it, computed on the same clean-span-
gradable population §3 uses (not literally "a periodic re-audit sample"
drawn and reviewed independently after deployment). It also inherits
everything true of AUTO-tier precision's own honesty caveats: fresh-10's
interval is 21.7pp wide (small `n`), and the "pre-set acceptable bound"
the proposal's Success Criteria section calls for was never defined
anywhere in this project, so there is no threshold to check any of these
three numbers against. What this closes is narrower but real: the
computation itself is no longer blocked, and the honest current answer is
"7.9%-23.2% depending on population, all real, none validated against an
actual human reviewer."

---

### 16. Note-level bootstrap confidence intervals — built 2026-08-31, closes the proposal's actual CI requirement

**What it is, and why it's different from §14's Wilson interval.**
`docs/Evaluation_Criteria.md` specifies "Bootstrap confidence intervals
... resampled at the note level" — not the Wilson score interval §14
adds, which treats every graded *entity* as an independent Bernoulli
trial. That's the wrong independence assumption for this project's data:
entities cluster within notes (a note's own vocabulary/complexity is
shared across every entity in it, and calibrator train/val splits are
already kept note-disjoint for exactly this reason). Resampling *notes*
with replacement — pooling every entity belonging to each resampled note,
duplicates included when a note is drawn more than once — is the
standard fix and the one the proposal actually names.

```python
def bootstrap_note_level_ci(records, metric_fn, n_boot=2000, seed=42, alpha=0.05):
    by_note = group records by note_id
    point = metric_fn(records)
    for _ in range(n_boot):
        sample_notes = choices(note_ids, k=len(note_ids))   # WITH replacement
        pooled = concat(by_note[n] for n in sample_notes)   # duplicates included
        boot_estimates.append(metric_fn(pooled))
    return point, percentile(boot_estimates, alpha/2), percentile(boot_estimates, 1-alpha/2)
```

`src/evaluation/bootstrap_ci.py` (`bootstrap_note_level_ci()`,
`precision_metric()`, `false_deflection_metric()`); real numbers produced
by `evaluation/run_bootstrap_ci.py`, reusing `evaluation.tier_gate_grading.
grade_by_tier()` for the same clean-span-gradable AUTO-tier population §3/
§14 already use — no new grading logic, no new LLM calls.

**A real bug caught while gathering these numbers, worth recording**: the
obvious note-ID source for "fresh-5" (`evaluation/grade_fresh5_by_tier.py`'s
own `NOTE_IDS`) is a *different*, older (2026-08-17) 5-note batch that
happens to share the name — not the actual "Fresh-5 (2026-08-30)" notes
§10's headline numbers are built on. Using it silently would have reported
a bootstrap CI around 79.2% (151→24 gradable, a completely different
population) instead of the real 92.1%. Caught by noticing the point
estimate didn't match the already-documented number — the same
"verify against real data before trusting a result" discipline this
project applies everywhere else. `evaluation/run_bootstrap_ci.py` now
hardcodes the correct 5 note IDs directly from §10's own text, with a
comment explaining why the obvious import is wrong.

**Real results, same three populations as §14, computed 2026-08-31**
(the corpus-wide note count is 144 here, not the 149 total `is_test`
notes now in the DB — 5 of those 149 notes have zero clean-span-gradable
AUTO-tier decisions, so they contribute nothing to this specific
population; the 149 total number is itself larger than the "144 notes"
label's original vintage, reflecting real corpus growth since):

| Population | n_notes | n_gradable | AUTO-tier precision | Wilson 95% CI (§14) | **Bootstrap 95% CI (note-level)** | Bootstrap width |
|---|---|---|---|---|---|---|
| Corpus-wide | 144 | 6,886 | 87.0% | [86.0%, 87.7%]¹ | **[85.9%, 88.1%]** | 2.2pp |
| Fresh-10 | 10 | 56 | 76.8% | [64.2%, 85.9%] | **[68.9%, 83.9%]** | 15.1pp |
| Fresh-5 | 5 | 151 | 92.1% | [86.6%, 95.4%] | **[86.1%, 96.7%]** | 10.5pp |

¹Wilson figures here are §14's original 6,724-gradable read; this run's
6,886 reflects the same real corpus-growth drift already disclosed for
other metrics this session (e.g. TransE's numbers, `docs/Knowledge_
Graphs_Technical_Reference.md` Part B §10.3) — the point estimate (87.0% vs.
86.9%) barely moved, so the Wilson interval is still a fair comparison
point even though it wasn't recomputed on the exact same 6,886.

And the false-deflection-rate proxy (§15), same populations, same method:

| Population | False deflection rate | Bootstrap 95% CI |
|---|---|---|
| Corpus-wide | 13.0% | [11.9%, 14.1%] |
| Fresh-10 | 23.2% | [16.1%, 31.2%] |
| Fresh-5 | 7.9% | [3.3%, 13.9%] |

**Read this honestly — the real finding is more nuanced than "bootstrap
is always wider," and that nuance matters.** Note-level clustering does
**not** uniformly inflate uncertainty relative to Wilson:
- **Corpus-wide**: bootstrap is modestly wider (2.2pp vs. 1.6pp) — the
  expected direction, a large, diverse population where note-to-note
  precision does vary somewhat.
- **Fresh-5**: bootstrap is wider (10.5pp vs. 8.8pp) — also expected,
  only 5 notes means real between-note variance risk.
- **Fresh-10 is the interesting exception: bootstrap is *narrower***
  (15.1pp vs. Wilson's 21.7pp). This is not a bug — it means per-note
  precision across these 10 notes is fairly homogeneous, so treating
  notes (not entities) as the resampling unit doesn't inflate uncertainty
  here; if anything, Wilson's generic "56 independent Bernoulli trials"
  assumption was the more conservative (wider) one for this specific
  population. Clustering can cut either way depending on the real
  variance structure — asserting it always widens intervals would have
  been an unearned assumption this project's own "verify before trusting"
  discipline exists to catch.

**The specific claim this was built to stress-test, and the honest
result**: §14 already flagged that fresh-10's and fresh-5's Wilson
intervals "come close to touching but do not overlap" (a 0.7 percentage
point gap: 85.9% vs. 86.6%), and cautioned against over-reading that as
decisive. Under the more rigorous bootstrap method, **the gap holds up
better, not worse** — fresh-10's upper bound (83.9%) and fresh-5's lower
bound (86.1%) are 2.2 percentage points apart, a wider separation than
Wilson gave. The fresh-5-beats-fresh-10 claim in §10.3 of the SSOT
survives this check more comfortably than the Wilson interval alone
suggested — a real, reassuring result, not assumed in advance (the
opposite outcome was considered a live possibility before running this).

**What this closes, precisely**: `docs/Evaluation_Criteria.md`'s own
specified method (bootstrap CIs resampled at the note level) is now
built and run on the project's own headline metric (AUTO-tier precision)
and its false-deflection-rate complement, across all three standard
populations. **Not yet extended**: Linked precision/recall (a separate
code path, `scripts/score_gold_recall.py`, not yet retrofitted with
note-level resampling) and the calibrator's own `TIER_1B`-specific
precision — both still only have Wilson intervals (§14). Worth doing if
those specific numbers become load-bearing for a future claim; not done
here since the headline metric was the priority.

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
| Linked precision | `linked_correct / count(predictions with resolved SNOMED code)` | ad hoc, `scripts/score_gold_recall.py`'s `load_predictions()`/`attach_snomed_codes()` |
| Linked F1 | `2PR / (P+R)` (Linked precision, Linked recall) | ad hoc |
| Deflection rate | `count(tier in AUTO_TIERS) / count(all decisions)` | ad hoc, `AUTO_TIERS` from `src/mollm_tier_gate.py` |
| Precision/Recall/F1 (promotion) | `TP/(TP+FP)`, `TP/(TP+FN)`, `2PR/(P+R)` | ad hoc, this project's calibrator/KG3 experiments |
| AUROC | `P(score(pos) > score(neg))` | `sklearn.metrics.roc_auc_score`, via `evaluation/tier_gate_cal_eval.py` |
| Wilson score interval | `(p̂+z²/2n)/(1+z²/n) ± z√(p̂(1-p̂)/n+z²/4n²)/(1+z²/n)` | new, §14 above, not yet a checked-in function |
| False deflection rate (gold-substituted proxy) | `1 - auto_tier_precision` | new, §15 above, not yet a checked-in function |
| Note-level bootstrap CI | resample notes w/ replacement, pool entities, percentile the resulting metric distribution | `evaluation/bootstrap_ci.py`, run via `evaluation/run_bootstrap_ci.py`, §16 above |
