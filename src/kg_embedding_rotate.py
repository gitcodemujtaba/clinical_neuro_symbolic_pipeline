"""src/kg_embedding_rotate.py -- 2026-08-31: RotatE (Sun et al. 2019), the
second of the proposal's two named-but-unbuilt KGE methods (TransE is
built and evaluated in src.kg_embedding; CompGCN remains deliberately
deferred, see docs/Knowledge_Graphs_Technical_Reference.md S5).

WHY NOT THE SNOMED VOCABULARY GRAPH AGAIN. TransE already trained on the
Athena/OMOP relationship graph restricted to this pipeline's own touched
concepts. Repeating that choice for RotatE was explicitly rejected in
favor of REPURPOSED/CURATED data -- this module trains on real, already-
produced project artifacts instead: a curated clinical-guideline graph,
this project's own gold-confirmed candidate-competition signal, their
union, and (added after discovering it's live) the full SNOMED IS_A
hierarchy from the separate KG1 Neo4j instance. Four independent training
configurations, run as a genuine ablation -- see
scripts/build_kg_embeddings_rotate.py and the technical reference doc for
the real, honestly-reported per-config results.

THE COMPATIBILITY TRICK: entity embeddings are stored as ONE real-valued
nn.Embedding(n_entities, 2*dim) -- first half the real component, second
half the imaginary component -- exactly how reference implementations
(OpenKE, pykeen) store RotatE internally too, not a hack invented for this
codebase. This is what lets evaluate_link_prediction() and
evaluate_against_tp_records() (src.kg_embedding, imported unchanged below)
and src.kg_embedding_tiebreak's two functions work with RotatE with ZERO
edits -- they only ever call .score(h, r, t) or .entity_emb(idx), and
never assume anything about what's packed inside that embedding vector.

ONE GEOMETRIC CAVEAT, STATED UP FRONT RATHER THAN DISCOVERED LATER: raw
Euclidean L2 distance over the packed [re; im] vector is a WEAKER proxy
for "topical closeness" in RotatE than it is in TransE. TransE's
translational geometry makes raw closeness a direct proxy for "connected
by something small" (h + r ~= t implies h and t are only 'r' apart).
RotatE's rotational geometry doesn't have that property: two entities
connected by a real, well-fit relation can still sit far apart in raw
packed coordinates if that relation's phase rotation is large. This is a
genuine, testable hypothesis -- not asserted as fact -- and a legitimate
reason RotatE's extrinsic/tiebreak numbers might land closer to chance
than TransE's, which would itself be a complete, honestly-reportable
finding, not a bug in this implementation.

FOUR DELIBERATE DEVIATIONS FROM THE TRANSE CODE THIS WAS ADAPTED FROM,
each RotatE-specific and each worth stating explicitly:
  1. NO unit-norm entity clamp (TransE's _normalize_entities()). Official
     RotatE deliberately leaves entity magnitude meaningful -- clamping it
     the way TransE does would be silently reintroducing TransE's own
     geometry assumption into a model that doesn't share it.
  2. Mean/max entity-embedding L2 norm is logged every 10 epochs (see
     train_rotate()'s print) specifically BECAUSE of (1) -- unbounded norm
     growth under plain hinge loss with no normalization is a known real
     instability mode for embedding models, and this is the up-front
     diagnostic against it, not a reaction to having observed it.
  3. relation_phase is initialized uniformly in [-pi, pi] (not Xavier) --
     Xavier-initializing a phase would start most rotations near-identity
     and slow early learning; the paper's own uniform-phase-init choice is
     kept.
  4. Loss/margin are kept IDENTICAL to TransE (plain hinge, margin=1.0),
     not RotatE's literature self-adversarial sigmoid loss (gamma
     9-24) -- the right call for a controlled TransE-vs-RotatE
     comparison in this codebase, but it means the resulting MRR is NOT
     comparable to published RotatE benchmarks. Said explicitly here and
     again in the results doc, matching this project's existing RAW-vs-
     filtered-MRR disclosure discipline (src.kg_embedding's own
     evaluate_link_prediction() docstring).
  5. L2 norm order (p=2), stated as a deliberate one-line choice matching
     TransE's p=2, not the paper's own L1-distance option.
"""
import random

import torch
import torch.nn as nn

# Re-exported so scripts/tests can import everything KGE-related from one
# place per config, without needing to know which module a given generic
# function actually lives in.
from src.kg_embedding import (  # noqa: F401
    build_vocab, evaluate_against_tp_records, evaluate_link_prediction)


def save_model(model: "RotatE", entity2idx: dict, relation2idx: dict, path: str, dim: int = None):
    """Mirrors src.kg_embedding.save_model(). `dim` here is the PER-
    COMPONENT dimension (real or imaginary half), not the raw embedding
    width -- load_model() reconstructs RotatE(dim=dim), whose __init__
    itself doubles it for the real embedding table."""
    dim = dim if dim is not None else model.dim
    torch.save({
        "state_dict": model.state_dict(),
        "entity2idx": entity2idx,
        "relation2idx": relation2idx,
        "dim": dim,
    }, path)


def load_model(path: str, device: str = "cpu"):
    """Inverse of save_model(). Returns (model, entity2idx, relation2idx)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = RotatE(len(ckpt["entity2idx"]), len(ckpt["relation2idx"]), dim=ckpt["dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["entity2idx"], ckpt["relation2idx"]


class RotatE(nn.Module):
    """score(h, r, t) = -|| h ∘ r_rot - t ||_2, where h ∘ r_rot is the
    complex Hadamard product (per-dimension 2D rotation of h by the unit-
    modulus rotation r_rot = cos(theta) + i*sin(theta)). Entities are
    stored as one real nn.Embedding(n_entities, 2*dim): [:dim] real part,
    [dim:] imaginary part -- see module docstring for why."""

    def __init__(self, n_entities: int, n_relations: int, dim: int = 100):
        super().__init__()
        self.dim = dim
        self.entity_emb = nn.Embedding(n_entities, 2 * dim)
        self.relation_phase = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.uniform_(self.relation_phase.weight, a=-torch.pi, b=torch.pi)
        # Deliberately NO _normalize_entities() call here -- see module
        # docstring, deviation (1).

    def score(self, h, r, t):
        h_emb, t_emb = self.entity_emb(h), self.entity_emb(t)
        h_re, h_im = h_emb[..., :self.dim], h_emb[..., self.dim:]
        t_re, t_im = t_emb[..., :self.dim], t_emb[..., self.dim:]

        theta = self.relation_phase(r)
        r_re, r_im = torch.cos(theta), torch.sin(theta)

        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        diff = torch.cat([rot_re - t_re, rot_im - t_im], dim=-1)
        return -torch.norm(diff, p=2, dim=-1)

    def forward(self, h, r, t):
        return self.score(h, r, t)

    def entity_norm_stats(self):
        """Mean/max L2 norm across all entity embeddings -- the diagnostic
        deviation (2) calls for. No-grad by convention of the caller."""
        with torch.no_grad():
            norms = self.entity_emb.weight.norm(dim=1)
            return norms.mean().item(), norms.max().item()


def train_rotate(triples: list, entity2idx: dict, relation2idx: dict,
                 dim: int = 100, epochs: int = 50, batch_size: int = 1024,
                 margin: float = 1.0, lr: float = 0.01, device: str = None,
                 seed: int = 42) -> RotatE:
    """Structurally mirrors src.kg_embedding.train_transe() (same margin-
    ranking loss, same random negative sampling scheme, same seed/epoch/
    batch conventions) so the two are a controlled comparison of the
    embedding geometry specifically -- see the four deviations in the
    module docstring for the parts that are deliberately NOT copied."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    idx_triples = [(entity2idx[h], relation2idx[r], entity2idx[t]) for h, r, t in triples]
    n_entities, n_relations = len(entity2idx), len(relation2idx)
    model = RotatE(n_entities, n_relations, dim=dim).to(device)
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
            total_loss += loss.item() * len(batch_idx)

        if epoch % 10 == 0 or epoch == epochs - 1:
            mean_norm, max_norm = model.entity_norm_stats()
            print(f"  epoch {epoch+1}/{epochs}  mean loss {total_loss/n:.4f}  "
                 f"entity norm mean/max {mean_norm:.3f}/{max_norm:.3f}")

    return model


# ==========================================================================
# Three (now four) OMOP-concept-id-keyed data loaders. All three configs
# below key entities by OMOP concept_id specifically -- NOT raw guideline-
# node UUIDs, NOT raw SNOMED code strings -- so evaluate_against_tp_records()
# and src.kg_embedding_tiebreak's functions work unchanged, since Stage 2b's
# own candidate pools are already OMOP-concept-id-keyed.
# ==========================================================================

def load_guideline_subgraph(driver, conn) -> list:
    """Real curated guideline-triplet edges from the same Memgraph instance
    KG3 lives on (:GuidelineNode -[REL]-> :GuidelineNode, 52+ real
    predicate types corpus-wide). Only edges where BOTH endpoints ground to
    a real, valid OMOP concept are usable -- confirmed live: 355 of 1,144
    total edges, spanning 31 distinct relation types among the grounded
    subset. Reuses resolve_snomed_cui() (scripts.backfill_guideline_
    grounding, imported not duplicated) for the SNOMED-code -> OMOP
    crosswalk -- the exact same grounding logic already validated (and
    found to have real known failure modes on SIMILARITY-based matching,
    which is why this function only trusts nodes that ALREADY carry a
    real `snomed` property, never attempts its own fuzzy grounding).
    """
    from scripts.backfill_guideline_grounding import resolve_snomed_cui

    with driver.session() as session:
        rows = session.run("""
            MATCH (a:GuidelineNode)-[rel]->(b:GuidelineNode)
            WHERE a.snomed IS NOT NULL AND a.snomed <> 'N/A'
              AND b.snomed IS NOT NULL AND b.snomed <> 'N/A'
            RETURN a.snomed AS h, type(rel) AS r, b.snomed AS t
        """).data()

    xwalk_cache = {}

    def _to_concept_id(snomed_code):
        if snomed_code not in xwalk_cache:
            hit = resolve_snomed_cui(conn, str(snomed_code))
            xwalk_cache[snomed_code] = hit[0] if hit else None
        return xwalk_cache[snomed_code]

    triples = []
    for row in rows:
        h_id, t_id = _to_concept_id(row["h"]), _to_concept_id(row["t"])
        if h_id and t_id:
            triples.append((h_id, row["r"], t_id))
    return triples


def load_gold_competition_triples(conn) -> list:
    """Wraps scripts.build_kg_embeddings.gather_tp_records() (imported, not
    duplicated) -- real gold-grounded true-positive records, flattened into
    (correct_concept_id, "PREFERRED_OVER", wrong_concept_id) triples. Real,
    measured yield: 1,593 triples from 452 TP records, 38 distinct correct
    concepts. The single relation type is deliberate: this IS the one
    relation this task needs -- "concept A beat concept B for a real
    mention, confirmed by gold" -- not an ontology-breadth stand-in.
    """
    from scripts.build_kg_embeddings import gather_tp_records

    records = gather_tp_records(conn)
    triples = []
    for rec in records:
        correct = rec["correct_concept_id"]
        for wrong in rec["wrong_candidate_ids"]:
            triples.append((correct, "PREFERRED_OVER", wrong))
    return triples


def load_combined_subgraph(driver, conn) -> list:
    """Simple concatenation of the guideline and gold-competition triples.
    PREFERRED_OVER cannot collide with any real guideline predicate name
    (checked: guideline predicates are all-caps clinical-action verbs like
    INDICATES/REQUIRES_INTERVENTION/TRIGGERS_SEVERITY, never this exact
    string), so no de-duplication is needed beyond what build_vocab()
    already does generically for entity/relation vocabularies."""
    return load_guideline_subgraph(driver, conn) + load_gold_competition_triples(conn)


def load_snomed_is_a_subgraph(neo4j_driver, conn, chunk_size: int = 5000) -> list:
    """Real SNOMED IS_A hierarchy edges from the SEPARATE KG1 Neo4j
    instance (bolt://localhost:7687, distinct from Memgraph/KG3) --
    (:SnomedConcept {id})-[:IS_A]->(:SnomedConcept {id}), 641,727 edges /
    386,110 nodes confirmed live. Added as a fourth config after directly
    verifying this graph is real and fully populated (unlike Neo4jHierarchy's
    own "for once the graph is populated" docstring caveat elsewhere in this
    codebase, written before this was checked).

    Same category of data as the vocabulary graph TransE already trained on
    (raw SNOMED structure, not repurposed/curated pipeline output) -- kept
    as an explicit fourth arm specifically to show whether a much larger,
    purer single-relation-type ontology graph outperforms the two much
    smaller curated/repurposed graphs, not folded into "combined" (which
    stays guideline+gold only, the two genuinely repurposed sources).

    Crosswalks SNOMED id -> OMOP concept_id via ONE batched DuckDB query
    per chunk (not resolve_snomed_cui()'s per-code query pattern -- at
    641K edges / up to ~772K distinct codes that would be far too slow).
    Prefers a standard SNOMED concept; drops any code with no standard
    OMOP mapping. Edges where either endpoint fails to crosswalk are
    dropped, same discipline as load_guideline_subgraph().
    """
    with neo4j_driver.session() as session:
        rows = session.run("""
            MATCH (a:SnomedConcept)-[:IS_A]->(b:SnomedConcept)
            RETURN a.id AS h, b.id AS t
        """).data()

    codes = sorted({row["h"] for row in rows} | {row["t"] for row in rows})
    code2concept = {}
    for start in range(0, len(codes), chunk_size):
        chunk = codes[start:start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        hits = conn.execute(f"""
            SELECT concept_code, concept_id FROM athena_concept
            WHERE vocabulary_id = 'SNOMED' AND standard_concept = 'S'
              AND concept_code IN ({placeholders})
        """, chunk).fetchall()
        for code, concept_id in hits:
            code2concept.setdefault(code, concept_id)

    triples = []
    for row in rows:
        h_id, t_id = code2concept.get(row["h"]), code2concept.get(row["t"])
        if h_id and t_id:
            triples.append((h_id, "IS_A", t_id))
    return triples
