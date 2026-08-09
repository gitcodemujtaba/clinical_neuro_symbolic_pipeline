"""
scripts/check_stage3_prerequisites.py

Read-only diagnostic. Answers, against the real EC2 DuckDB, the questions
docs/MoLLM_Stage3_Retrieval_Design.md S8 lists as unverified prerequisites --
without running a single LLM call.

WHY THIS EXISTS RATHER THAN "just run import_athena.py":
scripts/import_athena.py is a 0-byte stub, so there is nothing to run. But the
pipeline runs recorded in docs/Proposal_Alignment_Review.md S7 show working
Tier 1, Tier 2 AND Tier 3 (SapBERT vector) matches, which means athena_concept
is already populated WITH embeddings on the EC2 box, built outside these
scripts. So the open question is not "is the vocabulary loaded" but
specifically "which TABLES were loaded" -- a vocabulary import can easily have
covered CONCEPT.csv and CONCEPT_SYNONYM.csv while skipping
CONCEPT_ANCESTOR.csv and CONCEPT_RELATIONSHIP.csv, and Stage 3 depends on
exactly those two:

  * athena_concept_ancestor  -> Channel B (hierarchy). This is the channel that
    lifts guideline coverage above the measured 10.80% direct-code baseline.
    Without it, ~87.5% of entities get no guideline evidence at all.
  * athena_concept_relationship -> the RxNorm->SNOMED crosswalk, without which
    NO Medication entity can reach guideline evidence, since Stage 2b maps
    medications to RxNorm and the guideline KG is keyed on SNOMED.

Run on the EC2 host:  python3 scripts/check_stage3_prerequisites.py
"""

import os
import sys
import json
import glob
import time

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
TRIPLETS_DIR = os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned")

# Known-good SNOMED codes drawn from the guideline corpus, used as live probes
# rather than synthetic ones -- a hierarchy that resolves for a made-up code
# proves nothing about the codes retrieval will actually ask for.
PROBE_CODES = {
    "14669001": "Acute Kidney Injury (62 guideline rules — highest in corpus)",
    "91302008": "Sepsis",
    "13645005": "COPD",
    "233604007": "Pneumonia",
    "56675007": "Acute heart failure",
}

REQUIRED = {
    "athena_concept": "Stage 2b tiers 1-3; Channel C concept context",
    "athena_concept_synonym": "Stage 2b tier 2",
    "athena_concept_ancestor": "Channel B (hierarchy) AND Channel C parents",
    "athena_concept_relationship": "RxNorm->SNOMED crosswalk for Medication entities",
}


def _ok(b):
    return "OK " if b else "XX "


def _timed(conn, label, sql, params=None, warn_secs=10.0):
    """Runs a query and reports how long it took.

    Timings are printed rather than discarded because the point of this script
    is partly to find out which queries are affordable at this data scale --
    athena_concept_ancestor has 78.4M rows, and whether Channel B needs a
    narrowed materialised view is a question about measured latency, not about
    row count in the abstract.
    """
    t0 = time.time()
    try:
        rows = conn.sql(sql, params=params or []).fetchall()
    except Exception as exc:
        print(f"    ! {label} FAILED: {exc}")
        return None
    dt = time.time() - t0
    print(f"    [{label}: {dt:.1f}s{'  <-- SLOW' if dt > warn_secs else ''}]")
    return rows


def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        print("   Run this on the EC2 host, or point DB_PATH elsewhere.")
        return 1

    # Line-buffered so progress appears as each check completes. Some of these
    # queries touch tens of millions of rows; without this, a slow step looks
    # indistinguishable from a hang.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    conn = duckdb.connect(DB_PATH, read_only=True)
    verdict = {"blocking": [], "warnings": []}

    print("=" * 78)
    print("STAGE 3 PREREQUISITE CHECK")
    print("=" * 78)

    tables = {t[0] for t in conn.sql("SHOW TABLES").fetchall()}
    print("\n--- 1. REQUIRED TABLES ---")
    for name, why in REQUIRED.items():
        present = name in tables
        n = None
        if present:
            try:
                n = conn.sql(f"SELECT count(*) FROM {name}").fetchone()[0]
            except Exception:
                n = "?"
        print(f"  {_ok(present)} {name:<30} {'' if not present else f'{n:>12,} rows'}   [{why}]")
        if not present:
            verdict["blocking"].append(f"{name} missing — {why}")
        elif n == 0:
            verdict["blocking"].append(f"{name} exists but is EMPTY — {why}")

    # ---------------------------------------------------------------- concepts
    if "athena_concept" in tables:
        print("\n--- 2. athena_concept COVERAGE ---")
        try:
            total = conn.sql("SELECT count(*) FROM athena_concept").fetchone()[0]

            # SAMPLED, not exact, and deliberately so. `count(embedding)` over
            # the full table forces DuckDB to scan a 768-dimensional FLOAT[]
            # column across every row -- roughly 20 GB of floats on a 6.6M-row
            # vocabulary -- which hangs for minutes and tells us nothing we
            # could not learn from a sample. All we need to know here is
            # "are embeddings present, and roughly how widespread", so a
            # 100k-row sample answers it in seconds.
            print(f"  concepts: {total:,}")

            # MEASURED OVER THE POPULATION TIER 3 ACTUALLY SEARCHES, not the
            # whole table. An earlier version of this check reported raw
            # table-wide coverage and raised a false alarm: it showed "~44%
            # embedded" and warned that Tier 3 recall was degraded, when in
            # fact 100% of STANDARD SNOMED concepts in every queried domain
            # were embedded and the missing majority was non-standard concepts
            # plus vocabularies (NDC, SPL, ICD10CM) this pipeline never
            # queries. src/normalization.py filters on standard_concept='S'
            # and a fixed vocabulary list, so anything outside that is
            # irrelevant to recall -- and reporting it as a deficit invites
            # exactly the wrong remediation (re-running a 6.6M-row embedding
            # job that was already correctly scoped).
            sample_n = min(150_000, total)
            rows = conn.sql(f"""
                SELECT vocabulary_id, domain_id, count(*) AS n, count(embedding) AS e
                FROM (
                    SELECT vocabulary_id, domain_id, embedding
                    FROM athena_concept
                    WHERE standard_concept = 'S'
                      AND vocabulary_id IN ('SNOMED','RxNorm','RxNorm Extension')
                    USING SAMPLE {sample_n} ROWS
                )
                WHERE domain_id IN ('Condition','Observation','Drug','Procedure',
                                    'Spec Anatomic Site','Measurement')
                GROUP BY 1,2 HAVING count(*) > 50 ORDER BY 3 DESC LIMIT 12
            """).fetchall()

            print("  embedding coverage among STANDARD concepts in queried "
                  "vocabularies/domains:")
            worst = []
            for v, d, n, e in rows:
                pct = e / max(1, n) * 100
                print(f"    {v:<18} {d:<20} {e:>6}/{n:<6} = {pct:5.1f}%")
                if pct < 90:
                    worst.append(f"{v}/{d} at {pct:.0f}%")

            if rows and all(e == 0 for _, _, _, e in rows):
                verdict["blocking"].append(
                    "no embeddings on standard concepts — Stage 2b Tier 3 cannot fire")
            elif worst:
                verdict["warnings"].append(
                    "incomplete embedding coverage on searched concepts: "
                    + "; ".join(worst)
                    + ". Tier 3 cannot return a concept that has no vector, so these "
                      "look like unmappable entities rather than missing embeddings.")
            # Projects only vocabulary_id, so the embedding column is never
            # touched -- DuckDB is columnar, which is exactly why the query
            # above was the expensive one and this is not.
            print("  vocabularies present:")
            for v, c in conn.sql("""
                SELECT vocabulary_id, count(*) FROM athena_concept
                GROUP BY 1 ORDER BY 2 DESC LIMIT 8
            """).fetchall():
                print(f"    {v:<20} {c:>12,}")
        except Exception as exc:
            print(f"  ! could not profile: {exc}")

    # -------------------------------------------------------------- hierarchy
    print("\n--- 3. CHANNEL B (hierarchy) — the decisive check ---")
    print("    (joins against athena_concept_ancestor; may take a few seconds per probe)")
    if "athena_concept_ancestor" not in tables:
        print("  XX  athena_concept_ancestor absent: Channel B CANNOT RUN.")
        print("      Guideline coverage stays at the measured 10.80% direct-code baseline.")
    else:
        for code, label in PROBE_CODES.items():
            try:
                rows = conn.sql("""
                    SELECT count(DISTINCT anc.concept_code)
                    FROM athena_concept d
                    JOIN athena_concept_ancestor a ON a.descendant_concept_id = d.concept_id
                    JOIN athena_concept anc ON anc.concept_id = a.ancestor_concept_id
                    WHERE d.concept_code = ? AND d.vocabulary_id = 'SNOMED'
                      AND anc.vocabulary_id = 'SNOMED'
                      AND a.min_levels_of_separation BETWEEN 1 AND 3
                """, params=[code]).fetchone()[0]
            except Exception as exc:
                rows = f"error: {exc}"
            print(f"  {_ok(isinstance(rows, int) and rows > 0)} {code:<12} "
                  f"{str(rows):>5} ancestors within 3 hops   ({label})")

    # ------------------------------------------------------------- crosswalk
    print("\n--- 4. RxNorm -> SNOMED CROSSWALK (Medication entities) ---")
    if "athena_concept_relationship" not in tables:
        print("  XX  athena_concept_relationship absent: no Medication entity can reach")
        print("      guideline evidence; all fall back to the text-only path.")
    else:
        try:
            rows = conn.sql("""
                SELECT r.relationship_id, count(*) AS n
                FROM athena_concept_relationship r
                JOIN athena_concept c1 ON c1.concept_id = r.concept_id_1
                JOIN athena_concept c2 ON c2.concept_id = r.concept_id_2
                WHERE c1.vocabulary_id LIKE 'RxNorm%' AND c2.vocabulary_id = 'SNOMED'
                GROUP BY 1 ORDER BY 2 DESC LIMIT 6
            """).fetchall()
            if rows:
                for rel, n in rows:
                    print(f"  OK  {rel:<28} {n:>10,} RxNorm->SNOMED links")
            else:
                print("  XX  zero RxNorm->SNOMED links found.")
                verdict["warnings"].append(
                    "no RxNorm->SNOMED crosswalk: Medication entities will be text-only. "
                    "State this as a limitation rather than assuming coverage.")
        except Exception as exc:
            print(f"  ! query failed: {exc}")

    # -------------------------------------- guideline codes vs the vocabulary
    print("\n--- 5. GUIDELINE CODES RESOLVABLE IN THE VOCABULARY ---")
    codes = set()
    for path in glob.glob(os.path.join(TRIPLETS_DIR, "*.json")):
        try:
            for node in json.load(open(path, encoding="utf-8")).get("@graph", []):
                c = node.get("snomed")
                if c and c != "N/A":
                    codes.add(str(c))
        except Exception:
            continue

    if not codes:
        print(f"  ! no guideline codes found under {TRIPLETS_DIR}")
    elif "athena_concept" in tables:
        try:
            found = conn.sql(f"""
                SELECT count(DISTINCT concept_code) FROM athena_concept
                WHERE vocabulary_id = 'SNOMED' AND concept_code IN
                ({','.join(['?'] * len(codes))})
            """, params=list(codes)).fetchone()[0]
            print(f"  {found}/{len(codes)} guideline SNOMED codes exist in athena_concept "
                  f"({found / max(1, len(codes)) * 100:.1f}%)")
            if found < len(codes) * 0.8:
                verdict["warnings"].append(
                    f"only {found}/{len(codes)} guideline codes resolve — the guideline KG may "
                    "have been coded against a different SNOMED release than the one loaded "
                    "(Proposal_Alignment_Review.md S3.8.4 flagged exactly this risk).")
        except Exception as exc:
            print(f"  ! query failed: {exc}")

    # ------------------------------------------------- schema & Channel B sizing
    print("\n--- 6. SCHEMA + CHANNEL B WORKING-SET SIZING ---")
    print("    Establishes whether Channel B needs a narrowed materialised view,")
    print("    or whether the full 78M-row closure is already affordable.")

    for t in ("athena_concept", "athena_concept_ancestor", "athena_concept_relationship"):
        if t not in tables:
            continue
        print(f"\n  schema: {t}")
        try:
            for row in conn.sql(f"DESCRIBE {t}").fetchall():
                print(f"    {row[0]:<28} {row[1]}")
        except Exception as exc:
            print(f"    ! DESCRIBE failed: {exc}")

    if "athena_concept_ancestor" in tables:
        print("\n  ancestor closure — how much of it Channel B actually needs:")

        # Single-column filter, no join: cheap on a columnar store even at 78M
        # rows, and it isolates the hop-depth cut from the vocabulary cut so we
        # can see which one does the work.
        rows = _timed(conn, "hop-depth filter only", """
            SELECT count(*) FROM athena_concept_ancestor
            WHERE min_levels_of_separation BETWEEN 1 AND 3
        """)
        if rows:
            n3 = rows[0][0]
            print(f"    within 3 hops (any vocabulary): {n3:,}")

        rows = _timed(conn, "hop-depth distribution", """
            SELECT min_levels_of_separation, count(*)
            FROM athena_concept_ancestor
            WHERE min_levels_of_separation <= 5
            GROUP BY 1 ORDER BY 1
        """)
        if rows:
            for lvl, n in rows:
                print(f"      {lvl} hop(s): {n:>14,}")

        # The real Channel B working set: SNOMED->SNOMED, within 3 hops. This
        # is the expensive one (two joins into a 6.6M-row table), and its
        # timing is what decides the materialised-view question.
        rows = _timed(conn, "SNOMED-only within 3 hops (two joins)", """
            SELECT count(*)
            FROM athena_concept_ancestor a
            JOIN athena_concept d   ON d.concept_id  = a.descendant_concept_id
            JOIN athena_concept anc ON anc.concept_id = a.ancestor_concept_id
            WHERE a.min_levels_of_separation BETWEEN 1 AND 3
              AND d.vocabulary_id = 'SNOMED' AND anc.vocabulary_id = 'SNOMED'
        """, warn_secs=20.0)
        if rows:
            print(f"    >>> Channel B working set: {rows[0][0]:,} rows")
            print("        If this is small and the query was fast, no view is needed.")
            print("        If slow, create it once:")
            print("        CREATE TABLE snomed_ancestor_3hop AS <the SELECT above, "
                  "returning d.concept_code, anc.concept_code, hops>;")

    if "athena_concept" in tables:
        print("\n  domain distribution (drives Stage 2b's GLINER_LABEL_TO_DOMAIN filter):")
        rows = _timed(conn, "domain counts for SNOMED+RxNorm", """
            SELECT vocabulary_id, domain_id, count(*)
            FROM athena_concept
            WHERE vocabulary_id IN ('SNOMED','RxNorm','RxNorm Extension')
              AND domain_id IN ('Condition','Observation','Drug','Procedure',
                                'Spec Anatomic Site','Measurement')
            GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15
        """)
        for v, d, n in (rows or []):
            print(f"    {v:<18} {d:<20} {n:>10,}")

    # ----------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    if verdict["blocking"]:
        print("BLOCKING — Stage 3 cannot run as designed:")
        for b in verdict["blocking"]:
            print(f"  • {b}")
    else:
        print("No blocking issues found.")
    if verdict["warnings"]:
        print("\nDEGRADED — Stage 3 runs, but with reduced grounding:")
        for w in verdict["warnings"]:
            print(f"  • {w}")
    print("=" * 78)

    conn.close()
    return 1 if verdict["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main())
