"""
tests/test_compound_span_salt_suffix.py -- src/normalization/compound_span.py's
find_compound_split(), against a real (throwaway, in-memory) DuckDB connection.

Reproduces the exact real-data shape that caught the 2026-08-18 bug:
"Metoprolol Succinate" (and "Metoprolol Tartrate") were being split into two
separate entities ("Metoprolol" + "Succinate"), each independently normalized
and auto-written to KG3 as unrelated facts. GLiNER itself extracts the phrase
correctly as ONE span (confirmed via direct model re-test this session) --
the split happened because "metoprolol succinate" as a bare two-word string
has no exact/synonym Tier 1/2 hit in this RxNorm dump (only combined with
dose/form), so the whole-phrase guard doesn't fire, and "succinate" alone
independently resolves as a real chemical/ingredient concept, passing the
"every part must resolve" bar even though a drug and its salt form are never
two separate clinical facts.

Run: python3 -m pytest tests/test_compound_span_salt_suffix.py -v
"""
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalization.compound_span import find_compound_split  # noqa: E402


def _build_conn():
    conn = duckdb.connect(":memory:")
    conn.sql("""
        CREATE TABLE athena_concept (
            concept_id BIGINT, concept_name VARCHAR,
            domain_id VARCHAR, vocabulary_id VARCHAR, standard_concept VARCHAR
        );
    """)
    conn.sql("""
        CREATE TABLE athena_concept_synonym (
            concept_id BIGINT, concept_synonym_name VARCHAR, language_concept_id BIGINT
        );
    """)
    rows = [
        # (concept_id, concept_name, domain_id, vocabulary_id, standard_concept)
        # NOTE: no row for the bare two-word phrase "metoprolol succinate" or
        # "metoprolol tartrate" -- matching the real gap (RxNorm only has
        # these combined with dose/form), which is WHY the whole-phrase
        # guard doesn't block the split before this fix.
        (1, "metoprolol", "Drug", "RxNorm", "S"),
        (2, "succinate", "Drug", "RxNorm", "S"),
        (3, "tartrate", "Drug", "RxNorm", "S"),
        # Genuine compound-split case (unrelated to the bug) must still work.
        (4, "gunshot wound", "Condition", "SNOMED", "S"),
        (5, "abdomen", "Spec Anatomic Site", "SNOMED", "S"),
        # SNOMED-vocab duplicates of the same two words, so the label-scoping
        # check below isn't confounded by Condition's SNOMED-only vocab
        # restriction (VOCAB_BY_LABEL has no "Condition" entry -> falls back
        # to DEFAULT_VOCAB = ["SNOMED"], which wouldn't see the RxNorm rows
        # above at all regardless of the salt-suffix guard).
        (6, "metoprolol", "Condition", "SNOMED", "S"),
        (7, "succinate", "Condition", "SNOMED", "S"),
    ]
    for r in rows:
        conn.execute("INSERT INTO athena_concept VALUES (?, ?, ?, ?, ?)", r)
    return conn


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    conn = _build_conn()

    check("'Metoprolol Succinate' no longer splits (Medication label)",
          find_compound_split(conn, "Metoprolol Succinate", "Medication") is None)
    check("'Metoprolol Tartrate' no longer splits (Medication label)",
          find_compound_split(conn, "Metoprolol Tartrate", "Medication") is None)
    check("lowercase 'metoprolol succinate' no longer splits",
          find_compound_split(conn, "metoprolol succinate", "Medication") is None)

    # The guard is scoped to gliner_label == "Medication" only -- same raw
    # text under a different (hypothetical) label must NOT be silently
    # exempted, proving this is a real label-scoped guard, not an accidental
    # blanket string-match that would also suppress genuine splits elsewhere.
    result = find_compound_split(conn, "metoprolol succinate", "Condition")
    check("the guard does not fire for a non-Medication label (still splits)",
          result is not None and len(result["parts"]) == 2)

    # Regression: a genuine compound split (unrelated drug/salt pattern)
    # must be completely unaffected.
    result = find_compound_split(conn, "gunshot wound to abdomen", "Condition")
    check("genuine compound split ('gunshot wound to abdomen') still works",
          result is not None and [p["text"] for p in result["parts"]] == ["gunshot wound", "abdomen"])

    conn.close()

    print(f"compound-span salt-suffix tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_compound_span_salt_suffix():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
