"""src/kg_embedding_tiebreak.py — 2026-08-20: a targeted use of the trained
TransE model (src.kg_embedding) as a Stage 2b tiebreak signal, specifically
for the SNOMED near-duplicate-concept pattern src.normalization.
tier_retrieval._collapse_hierarchy_duplicates() already detects (two
candidates connected by a direct "Is a"/"Subsumes" edge).

WHY THIS SIGNAL, NOT A BLANKET KGE+SAPBERT FUSION. Blindly fusing a second
embedding score into every retrieval was already tried once this session
(BM25+dense hybrid retrieval) and measured to lose outright to dense-only
-- CNSP_HYBRID_RETRIEVAL stays off as a validated conclusion, not an
oversight. A KGE signal is structurally different from BM25 (lexical
overlap) -- it's not guaranteed to fail the same way -- but the lesson
holds: don't add a second signal everywhere without measuring it. This
module is scoped narrowly to the one recurring failure mode this session
found repeatedly (WBC/RBC Procedure-vs-Observable-Entity, the HCO3
hierarchy-collapse bug, the wound-dehiscence Condition-vs-Observation
duplicate) -- cases where two candidates read similarly enough that
SapBERT can't reliably separate them (or actively prefers the wrong one),
but sit in genuinely different positions in the SNOMED graph.

THE SIGNAL: average KGE-embedding distance from a candidate to every
OTHER concept in the SAME entity's broader candidate pool (the pool
SapBERT independently proposed for this mention, not the tied pair
alone). Hypothesis: the genuinely correct concept, being about the same
clinical topic as the mention, should sit closer to that broader
semantic neighborhood than a structurally-adjacent-but-topically-off
sibling does. This deliberately stays WITHIN KGE's own embedding space
(comparing candidate-to-candidate distances) rather than attempting to
bridge KGE's graph-structural space and SapBERT's text-semantic space
directly -- the two are not the same coordinate system, and comparing
raw distances across them without a learned mapping would not be a
principled comparison.

NOT WIRED INTO route_tier() OR tier_retrieval.py. Standalone, tested
module pending its own validation batch against real gold data (the
same discipline as every other new tiebreak mechanism this session:
build, smoke-test, validate on real data, only then wire in) -- held
specifically until the Stage 3 recall-fix backfill completes and the KG
embedding model itself is retrained on the larger, post-backfill TP
record pool (src.kg_embedding, scripts/build_kg_embeddings.py).
"""
import torch


def kg_tiebreak_score(model, entity2idx: dict, candidate_id: int, pool_concept_ids: list) -> float:
    """Mean L2 distance in TransE embedding space from `candidate_id` to
    every OTHER concept_id in `pool_concept_ids` (candidate_id itself and
    any id missing from the trained vocabulary are excluded). Lower means
    more consistent with the broader candidate pool. Returns None if
    candidate_id isn't in the vocabulary, or no other pool member is
    either (nothing to compare against).
    """
    if candidate_id not in entity2idx:
        return None
    others = [c for c in pool_concept_ids if c != candidate_id and c in entity2idx]
    if not others:
        return None

    device = next(model.parameters()).device
    with torch.no_grad():
        c_emb = model.entity_emb(torch.tensor(entity2idx[candidate_id], device=device))
        other_idx = torch.tensor([entity2idx[o] for o in others], device=device)
        other_emb = model.entity_emb(other_idx)
        distances = torch.norm(other_emb - c_emb.unsqueeze(0), p=2, dim=1)
        return distances.mean().item()


def pick_via_kg_tiebreak(model, entity2idx: dict, tied_concept_ids: list,
                         full_pool_concept_ids: list) -> dict:
    """For a set of hierarchy-tied candidates (2, in practice, matching
    _collapse_hierarchy_duplicates()'s own connected-component shape),
    picks whichever has the LOWEST mean KGE distance to the rest of the
    entity's own candidate pool -- i.e. whichever is most consistent with
    the broader semantic neighborhood SapBERT independently proposed for
    this mention, not just consistent with its own tied sibling.

    Returns {"winner": concept_id|None, "scores": {concept_id: float|None},
    "resolved": bool} -- resolved=False when fewer than 2 tied candidates
    have a usable score (nothing to discriminate between; caller should
    fall back to its own existing tiebreak, e.g. raw similarity).
    """
    rest_of_pool = [c for c in full_pool_concept_ids if c not in tied_concept_ids]
    scores = {}
    for cid in tied_concept_ids:
        scores[cid] = kg_tiebreak_score(model, entity2idx, cid, rest_of_pool)

    usable = {cid: s for cid, s in scores.items() if s is not None}
    if len(usable) < 2:
        return {"winner": None, "scores": scores, "resolved": False}

    winner = min(usable, key=usable.get)
    return {"winner": winner, "scores": scores, "resolved": True}
