"""src/normalization/vocab_release.py — Athena vocabulary release lookup (split from src/normalization.py, 2026-08-14)."""

_VOCAB_RELEASE = None




def get_athena_vocabulary_release(conn) -> str:
    """Identifies which Athena vocabulary release produced a mapping.

    docs/Provenance_Schema.md Stage 2b specifies this field and it was never
    written. It is not bookkeeping: SNOMED concept IDs are stable across
    releases but concept NAMES, standard_concept flags and the ancestor closure
    are not, so a mapping produced against one release is not guaranteed
    reproducible against another. Without the stamp there is no way to tell,
    after the fact, whether two runs disagreed because the code changed or
    because the vocabulary did -- which is exactly the ambiguity that made the
    Stage 2b non-determinism bug hard to characterise.

    Prefers a real `vocabulary` table if the Athena dump included one. Falls
    back to a CONTENT SIGNATURE (row count + latest valid_start_date over
    SNOMED) rather than a load timestamp: a timestamp records when the file was
    read, not what was in it, and would make two different vocabularies look
    identical if loaded at the same moment.
    """
    global _VOCAB_RELEASE
    if _VOCAB_RELEASE is not None:
        return _VOCAB_RELEASE

    release = "unknown"
    try:
        tables = {t[0].lower() for t in conn.sql("SHOW TABLES").fetchall()}
        for candidate in ("athena_vocabulary", "vocabulary"):
            if candidate in tables:
                row = conn.sql(
                    f"SELECT vocabulary_version FROM {candidate} "
                    "WHERE vocabulary_id = 'SNOMED' LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    _VOCAB_RELEASE = str(row[0])
                    return _VOCAB_RELEASE
    except Exception:
        pass

    try:
        n, latest = conn.sql("""
            SELECT count(*), max(valid_start_date)
            FROM athena_concept WHERE vocabulary_id = 'SNOMED'
        """).fetchone()
        release = f"signature:snomed_n={n},latest_valid_start={latest}"
    except Exception:
        pass

    _VOCAB_RELEASE = release
    return release



