"""scripts/check_pluralization_gap.py -- quick, targeted check: is a plural
surface form (e.g. "fevers") failing Tier 1 exact-match against a singular
OMOP concept name ("Fever"), while a mass-noun control ("sinusitis", no
natural plural) matches cleanly? Answers whether the LOW-tier volume for
common symptom words includes a pluralization gap worth building a
candidate-fallback for, or whether Tier 2's synonym table already absorbs it.

NOT a blind fix -- this only looks. Read-only. Needs the DB free (a
read-write connection elsewhere, e.g. the live Stage 3 batch, will block even
read_only=True; DuckDB's file lock is exclusive against a read-write holder).

Run on EC2:
    cd ~/clinical_neuro_symbolic_pipeline/code
    source ~/.venv/bin/activate
    python3 scripts/check_pluralization_gap.py
"""
import os
import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# (surface form actually seen in tonight's batch, plausible singular/base form)
TERMS = [
    ("fevers", "fever"),
    ("sinusitis", "sinusitis"),  # control: no natural plural, should already work if the mechanism is fine
]


def check_concept_exact(conn, text):
    """Tier 1 shape: does this exact lowercased string exist as a standard concept_name?"""
    rows = conn.execute("""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        ORDER BY concept_id ASC LIMIT 5
    """, [text.lower()]).fetchall()
    return rows


def check_synonym_exact(conn, text):
    """Tier 2 shape: does this exact lowercased string exist as a synonym of a standard concept?"""
    rows = conn.execute("""
        SELECT c.concept_id, c.concept_name, s.concept_synonym_name, c.domain_id, c.vocabulary_id
        FROM athena_concept_synonym s
        JOIN athena_concept c ON s.concept_id = c.concept_id
        WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
        ORDER BY c.concept_id ASC LIMIT 5
    """, [text.lower()]).fetchall()
    return rows


def check_actual_run_result(conn, text):
    """What match_tier did this EXACT surface form actually get in the live/test
    normalized_entities rows already on disk (any is_test value -- tonight's
    live batch and any earlier test runs both count)?"""
    rows = conn.execute("""
        SELECT note_id, original_text, expanded_text, match_tier, omop_concept_name,
               omop_vocab, is_test
        FROM normalized_entities
        WHERE lower(original_text) = ? OR lower(expanded_text) = ?
        ORDER BY note_id LIMIT 10
    """, [text.lower(), text.lower()]).fetchall()
    return rows


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    print(f"Connected read-only to {DB_PATH}\n")

    for surface, base in TERMS:
        print("=" * 78)
        print(f"### {surface!r} (base/singular form checked: {base!r})")

        print(f"\n  Tier-1 exact concept_name match for {surface!r}:")
        hits = check_concept_exact(conn, surface)
        print(f"    {hits if hits else '(none)'}")

        if base != surface:
            print(f"\n  Tier-1 exact concept_name match for base form {base!r}:")
            hits_base = check_concept_exact(conn, base)
            print(f"    {hits_base if hits_base else '(none)'}")

        print(f"\n  Tier-2 exact synonym match for {surface!r}:")
        syn_hits = check_synonym_exact(conn, surface)
        print(f"    {syn_hits if syn_hits else '(none)'}")

        print(f"\n  Actual match_tier already recorded for {surface!r} in this run:")
        actual = check_actual_run_result(conn, surface)
        if actual:
            for row in actual:
                print(f"    note={row[0]} orig={row[1]!r} expanded={row[2]!r} "
                      f"tier={row[3]} concept={row[4]!r} vocab={row[5]} is_test={row[6]}")
        else:
            print("    (no normalized_entities row found for this exact text yet)")
        print()

    print("=" * 78)
    print("Read: if the plural form has NO Tier-1 concept_name hit but the "
          "singular form DOES, and no Tier-2 synonym covers the plural either, "
          "that's the gap confirmed -- worth adding singular-form as a "
          "strip_lab_value_suffix()-style fallback CANDIDATE (never a blind "
          "rewrite), gated the same ontology-arbitrates-not-guesses way as "
          "every other fix tonight. If Tier-2 already covers it, or the "
          "recorded match_tier is already 1/2, there's nothing to fix here.")


if __name__ == "__main__":
    main()
