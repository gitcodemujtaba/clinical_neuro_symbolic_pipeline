"""
scripts/measure_relation_coverage.py — Phase 0 for Channel E (relation-aware
retrieval, docs/MoLLM_Redesign_Proposal.md S10).

Two things must be measured before Channel E's predicate mapping or coverage
claims can be trusted, per the same discipline
docs/MoLLM_Stage3_Retrieval_Design.md used for every other threshold in this
system -- measured, not assumed:

  1. THE FULL GUIDELINE PREDICATE LIST. RELEX_LABEL_TO_PREDICATES
     (src/retrieval.py) currently maps only 2 of GLiNER-relex's 5 relation
     labels ("treated with", "indicates"), using only the 4 predicate names
     that happened to be quoted in passing in earlier docs
     (REQUIRES_MEDICATION, REQUIRES_INTERVENTION, INDICATES,
     TRIGGERS_SEVERITY). "causes", "located in" and "measured by" are
     deliberately left unmapped (empty set) because no one has looked at the
     other ~45 of the corpus's ~49 canonicalized predicates to see which, if
     any, are plausible matches. This script dumps the full predicate
     frequency list so that mapping can be completed by inspection rather
     than guessed.

  2. CHANNEL E'S PLAUSIBLE REACH. Channel E only fires for an entity with a
     `linked` GLiNER-relex relation whose partner ALSO normalized to a
     SNOMED-anchored concept (src/retrieval.py's channel_e_relation()
     requires both ends coded, same as Channel A). Nobody has measured what
     fraction of entities that actually describes. If it's a small
     percentage, Channel E is a real but minor addition; if large, it's worth
     prioritizing verification of the predicate mapping above.

Run (read-only, safe to run anytime, does not touch normalized_entities):
  python3 scripts/measure_relation_coverage.py
  python3 scripts/measure_relation_coverage.py --note-ids 10000032-DS-21,...
"""

import argparse
import collections
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

TRIPLETS_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
]


def dump_predicate_frequencies(index) -> None:
    print("--- 1. FULL GUIDELINE PREDICATE FREQUENCY LIST ---")
    counts = collections.Counter()
    for rules in index.rules_by_source.values():
        for r in rules:
            counts[r["predicate"]] += 1
    print(f"  {len(counts)} distinct predicate(s), {sum(counts.values())} rule(s) total\n")
    for predicate, n in counts.most_common():
        print(f"    {n:>5}  {predicate}")
    print()
    print("  Cross-check against src/retrieval.py's RELEX_LABEL_TO_PREDICATES:")
    from src.retrieval import RELEX_LABEL_TO_PREDICATES
    mapped = set()
    for preds in RELEX_LABEL_TO_PREDICATES.values():
        mapped |= preds
    unmapped_confirmed = set(counts) - mapped
    print(f"    currently mapped: {sorted(mapped)}")
    print(f"    real predicates NOT yet mapped to any relex label "
          f"({len(unmapped_confirmed)}):")
    for p in sorted(unmapped_confirmed, key=lambda p: -counts[p]):
        print(f"      {counts[p]:>5}  {p}")
    stale = mapped - set(counts)
    if stale:
        print(f"    WARNING -- mapped predicate name(s) that do NOT appear in "
              f"the live corpus at all (typo or renamed?): {sorted(stale)}")


def measure_channel_e_reach(conn, note_ids) -> None:
    print("\n--- 2. CHANNEL E PLAUSIBLE REACH (linked relation + both ends SNOMED-anchored) ---")
    from src.retrieval import VocabularyRetriever

    vocab = VocabularyRetriever(conn)
    total_entities = 0
    total_relations = 0
    linked_relations = 0
    both_ends_anchored = 0

    for note_id in note_ids:
        entities = conn.sql(
            "SELECT entity_id FROM extracted_entities WHERE note_id = ?",
            params=[note_id],
        ).fetchall()
        total_entities += len(entities)

        rels = conn.sql("""
            SELECT head_entity_id, tail_entity_id, head_link_status, tail_link_status
            FROM extracted_relations WHERE note_id = ?
        """, params=[note_id]).fetchall()
        total_relations += len(rels)

        for head_id, tail_id, head_status, tail_status in rels:
            if head_status != "linked" or tail_status != "linked":
                continue
            linked_relations += 1

            head_snap = vocab.entity_snapshot(head_id)
            tail_snap = vocab.entity_snapshot(tail_id)
            head_code = (vocab.snomed_code_for_concept(head_snap["primary_omop_concept_id"])
                         if head_snap.get("primary_omop_concept_id") else None)
            tail_code = (vocab.snomed_code_for_concept(tail_snap["primary_omop_concept_id"])
                         if tail_snap.get("primary_omop_concept_id") else None)
            if head_code and tail_code:
                both_ends_anchored += 1

    print(f"  notes measured:                    {len(note_ids)}")
    print(f"  entities (denominator, for scale):  {total_entities}")
    print(f"  relations extracted:                {total_relations}")
    print(f"  relations with BOTH endpoints linked: {linked_relations}"
          + (f" ({100*linked_relations/total_relations:.1f}% of relations)"
             if total_relations else ""))
    print(f"  ... of those, BOTH ends SNOMED-anchored (Channel E's actual reach): "
          f"{both_ends_anchored}"
          + (f" ({100*both_ends_anchored/linked_relations:.1f}% of linked relations)"
             if linked_relations else ""))
    print()
    print("  This is Channel E's plausible reach BEFORE the predicate mapping is even")
    print("  checked -- a relation clearing this bar still needs its relation_label to")
    print("  map to a real predicate (see section 1 above) to actually retrieve anything.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "rows in extracted_relations.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    from src.retrieval import GuidelineIndex

    triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
    if not triplets:
        print(f"No guideline corpus found. Tried: {TRIPLETS_CANDIDATES}")
        return 1
    index = GuidelineIndex(triplets)
    print(f"guideline KG: {index.stats['nodes']} nodes, {index.stats['rules']} rules, "
          f"{index.stats['files']} files\n")

    dump_predicate_frequencies(index)

    conn = duckdb.connect(args.db, read_only=True)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            note_ids = [r[0] for r in conn.sql(
                "SELECT DISTINCT note_id FROM extracted_relations"
            ).fetchall()]
        if not note_ids:
            print("\nNo rows in extracted_relations -- run Stage 2a relation "
                  "extraction first.")
            return 1
        measure_channel_e_reach(conn, note_ids)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
