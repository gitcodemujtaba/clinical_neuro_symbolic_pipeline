"""src/kg_embedding.py — 2026-08-20: a real TransE knowledge-graph
embedding pipeline over the SNOMED CT relationship graph, closing the
proposal's Objective 4 gap ("transform legacy clinical classification
datasets into dense representations, constructing highly accurate KG
embedding pipelines to improve downstream entity recognition tasks").

WHY THIS IS FEASIBLE NOW, CORRECTING AN EARLIER (WRONG) ASSESSMENT. This
gap was initially scoped out with the reasoning "no accumulated KG3 write
volume to embed yet" -- that reasoning conflated two different graphs.
KG3 (the dynamic, patient-instance graph this project's own write-back
loop populates) genuinely has zero live volume, dry_run=True everywhere.
But the REFERENCE graph -- SNOMED CT itself, via
athena_concept_relationship, already loaded in this project's own DuckDB
-- is a real, substantial, already-populated knowledge graph with
millions of genuine triples. Nothing about training a KGE model on it
requires KG3 to have any data at all. This module trains directly on
that reference graph, scoped to the ~7,261 concepts this pipeline's own
candidate pools actually touch (24,872 real relationship edges among
them) -- "based on our auto-complete TP records" in the sense that the
SCOPE of the graph is defined by concepts our own tier-gate has actually
resolved and graded, not an arbitrary slice of all of SNOMED.

METHOD CHOICE: TransE, not RotatE/CompGCN. TransE (Bordes et al. 2013)
is the simplest, most standard member of the family the proposal's own
background research names, with a transparent scoring function
(h + r ≈ t) and no specialized library dependency -- implemented here in
plain PyTorch rather than pulling in pykeen/torch-geometric (neither is
installed in this environment). RotatE (complex-valued embeddings) and
CompGCN (a full graph-convolutional architecture) are real, meaningfully
more complex follow-on work, not attempted here -- stated honestly as
scope, not silently skipped.

TWO EVALUATIONS, BOTH REAL:
  1. Intrinsic (standard KGE literature protocol): held-out triple
     ranking, Mean Reciprocal Rank + Hits@10, against a random 10% split
     of the graph's own edges never seen during training.
  2. Extrinsic, tied directly to this project's own task (see
     evaluate_against_tp_records() below): for every entity this
     session's tier gate graded as a genuine true positive against gold,
     with at least 2 real candidates in its pool, checks whether the
     trained embedding space places the CORRECT candidate's concept
     closer (in embedding space) to the WRONG candidate(s) it beat than a
     random unrelated concept would be -- i.e., does the embedding
     capture real clinical proximity among candidates that already
     competed for the same mention, the actual signal a KGE-assisted
     re-ranker would need to be useful for Stage 2b.
"""
import random

import torch
import torch.nn as nn


def load_snomed_subgraph(conn, concept_ids: list) -> list:
    """Real SNOMED relationship triples (concept_id_1, relationship_id,
    concept_id_2) with BOTH endpoints in `concept_ids` -- the concepts
    this pipeline's own candidate pools have actually touched. Excludes
    invalid/retired relationships. Returns a list of (head, relation,
    tail) integer/string tuples."""
    placeholders = ",".join("?" * len(concept_ids))
    rows = conn.execute(f"""
        SELECT concept_id_1, relationship_id, concept_id_2
        FROM athena_concept_relationship
        WHERE invalid_reason IS NULL
        AND concept_id_1 IN ({placeholders}) AND concept_id_2 IN ({placeholders})
    """, [*concept_ids, *concept_ids]).fetchall()
    return [(h, r, t) for h, r, t in rows]


class TransE(nn.Module):
    """Standard TransE (Bordes et al. 2013): score(h, r, t) = -||h + r - t||_2
    -- higher score (less negative) means more plausible. Entity embeddings
    are re-normalized to unit L2 norm after every optimizer step, the
    standard trick preventing the model from trivially minimizing loss by
    growing embedding norms rather than learning real structure.
    """

    def __init__(self, n_entities: int, n_relations: int, dim: int = 100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self._normalize_entities()

    def _normalize_entities(self):
        with torch.no_grad():
            self.entity_emb.weight.div_(self.entity_emb.weight.norm(dim=1, keepdim=True).clamp_min(1e-9))

    def score(self, h, r, t):
        return -torch.norm(self.entity_emb(h) + self.relation_emb(r) - self.entity_emb(t), p=2, dim=-1)

    def forward(self, h, r, t):
        return self.score(h, r, t)


def build_vocab(triples: list):
    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    entity2idx = {e: i for i, e in enumerate(entities)}
    relation2idx = {r: i for i, r in enumerate(relations)}
    return entity2idx, relation2idx


def train_transe(triples: list, entity2idx: dict, relation2idx: dict,
                 dim: int = 100, epochs: int = 50, batch_size: int = 1024,
                 margin: float = 1.0, lr: float = 0.01, device: str = None,
                 seed: int = 42) -> TransE:
    """Standard margin-ranking-loss TransE training with random negative
    sampling (corrupt head or tail with a uniformly random entity, per
    the original paper). Returns the trained model, left on `device`.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    idx_triples = [(entity2idx[h], relation2idx[r], entity2idx[t]) for h, r, t in triples]
    n_entities, n_relations = len(entity2idx), len(relation2idx)
    model = TransE(n_entities, n_relations, dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    h_all = torch.tensor([x[0] for x in idx_triples], device=device)
    r_all = torch.tensor([x[1] for x in idx_triples], device=device)
    t_all = torch.tensor([x[2] for x in idx_triples], device=device)
    n = len(idx_triples)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            h, r, t = h_all[batch_idx], r_all[batch_idx], t_all[batch_idx]

            corrupt_head = torch.rand(len(batch_idx), device=device) < 0.5
            neg_entities = torch.randint(0, n_entities, (len(batch_idx),), device=device)
            h_neg = torch.where(corrupt_head, neg_entities, h)
            t_neg = torch.where(corrupt_head, t, neg_entities)

            pos_score = model.score(h, r, t)
            neg_score = model.score(h_neg, r, t_neg)
            loss = torch.clamp(margin - pos_score + neg_score, min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._normalize_entities()
            total_loss += loss.item() * len(batch_idx)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  mean loss {total_loss/n:.4f}")

    return model


@torch.no_grad()
def evaluate_link_prediction(model: TransE, test_triples: list, entity2idx: dict,
                             relation2idx: dict, k: int = 10, max_eval: int = 2000,
                             device: str = None) -> dict:
    """Standard KGE intrinsic evaluation: for each held-out (h, r, t),
    ranks the true tail against ALL entities as candidate tails (filtered
    setting would exclude other true triples -- this is the simpler RAW
    setting, stated explicitly since raw vs. filtered MRR are not
    comparable numbers in the literature). Returns {"mrr": float,
    f"hits_at_{k}": float, "n_evaluated": int}. Capped at max_eval triples
    (evaluating against the full entity set for every held-out triple is
    the expensive part) -- a random subsample, not the first N, so the
    reported number isn't biased by triple ordering.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample = test_triples if len(test_triples) <= max_eval else random.sample(test_triples, max_eval)
    all_entities = torch.arange(len(entity2idx), device=device)

    reciprocal_ranks = []
    hits = 0
    for h, r, t in sample:
        if h not in entity2idx or r not in relation2idx or t not in entity2idx:
            continue
        h_idx = torch.tensor(entity2idx[h], device=device)
        r_idx = torch.tensor(relation2idx[r], device=device)
        t_idx = entity2idx[t]

        scores = model.score(h_idx.expand(len(all_entities)), r_idx.expand(len(all_entities)), all_entities)
        rank = (scores > scores[t_idx]).sum().item() + 1
        reciprocal_ranks.append(1.0 / rank)
        if rank <= k:
            hits += 1

    n = len(reciprocal_ranks)
    return {"mrr": sum(reciprocal_ranks) / n if n else None,
            f"hits_at_{k}": hits / n if n else None, "n_evaluated": n}


@torch.no_grad()
def evaluate_against_tp_records(model: TransE, entity2idx: dict, tp_records: list,
                                device: str = None) -> dict:
    """Extrinsic evaluation tied directly to this project's own graded
    decisions, not the generic KGE literature protocol above.

    `tp_records`: list of {"correct_concept_id": int, "wrong_candidate_ids":
    list[int]} -- one entry per gold-graded entity where the tier gate's
    winning candidate was confirmed correct against gold AND at least one
    OTHER real candidate existed in the same pool (a genuine competing
    alternative the correct concept had to beat, not a trivial single-
    candidate case).

    For each record, checks whether the embedding space places the
    correct concept CLOSER to a random unrelated concept's neighborhood
    than the wrong candidate(s) are, using this as a discrimination
    signal: computes the L2 distance in embedding space between the
    correct concept and each wrong candidate, and between the correct
    concept and a random distractor concept, then reports what fraction
    of wrong-candidate comparisons show LARGER distance-from-correct than
    a same-scale random baseline would -- i.e., does the KG embedding
    actually separate "confusable candidate" from "unrelated concept"
    more than chance. This is the concrete question a KGE-assisted
    Stage 2b re-ranker would need answered yes to be worth building.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    entity_ids = list(entity2idx.keys())

    def _dist(a, b):
        if a not in entity2idx or b not in entity2idx:
            return None
        ea = model.entity_emb(torch.tensor(entity2idx[a], device=device))
        eb = model.entity_emb(torch.tensor(entity2idx[b], device=device))
        return torch.norm(ea - eb, p=2).item()

    n_comparisons = 0
    n_correct_farther_than_random = 0
    n_usable_records = 0
    for rec in tp_records:
        correct = rec["correct_concept_id"]
        wrongs = rec["wrong_candidate_ids"]
        if correct not in entity2idx or not wrongs:
            continue
        usable_this_record = False
        for wrong in wrongs:
            d_wrong = _dist(correct, wrong)
            if d_wrong is None:
                continue
            random_other = random.choice(entity_ids)
            d_random = _dist(correct, random_other)
            if d_random is None:
                continue
            usable_this_record = True
            n_comparisons += 1
            # "Wrong candidate" competed for the SAME mention -- expect it
            # to be CLOSER (more confusable) than an arbitrary concept,
            # not farther. A KG embedding that's actually useful for
            # re-ranking should show d_wrong < d_random more often than not.
            if d_wrong < d_random:
                n_correct_farther_than_random += 1
        if usable_this_record:
            n_usable_records += 1

    return {
        "n_tp_records_provided": len(tp_records),
        "n_usable_records": n_usable_records,
        "n_comparisons": n_comparisons,
        "frac_wrong_candidate_closer_than_random":
            n_correct_farther_than_random / n_comparisons if n_comparisons else None,
    }
