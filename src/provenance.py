"""
src/provenance.py — run/version/candidate-list provenance for every table this
pipeline writes decisions to.

WHY THIS EXISTS. docs/2026-08-13_Calibration_Diagnostics_And_Fixes.md S6 flags
a confound that cannot be resolved after the fact:

    "mollm_decisions has accumulated across multiple Stage 1/2 reruns with no
     timestamp column to distinguish which decisions were made against which
     version of the candidate list -- some INTRODUCED_ERROR cases could in
     principle be grading artifacts from candidate-index drift rather than
     genuine model failures."

That report's NET VALUE = -17 finding (MoLLM's resolution ensemble measured as
net-harmful) carries this caveat precisely because the table cannot answer
"which candidate list was this decision made against". No amount of later
analysis recovers the answer; it has to be written down at decision time.

WHY candidates_hash AND NOT JUST created_at. A timestamp only orders rows -- it
cannot tell you whether the candidate list a decision was made against is the
SAME list you are now grading it against. Two runs minutes apart can straddle a
Stage 2b change; two runs weeks apart can be identical. candidates_hash answers
the question directly: an evaluation joins on it and drops (or separately
reports) any decision whose hash no longer matches the current normalization
output. That is the column that actually kills the drift confound. See
evaluation/cal_eval.py for the consuming side.

ORDER-SENSITIVITY IS DELIBERATE. The hash covers the candidate list AS
PRESENTED, in order, because presentation order is itself a variable under test
-- S6's INTRODUCED_ERROR cluster (AST-956/AST-909/AST-16/CK(CPK)) is consistent
with position bias, and scripts/run_stage3_batch.py's --shuffle-candidates flag
exists to test exactly that. A hash that sorted the ids first would make the
shuffle experiment invisible in the data it produces.

DEGRADES, NEVER RAISES. Every function here returns a usable value even when
git is absent, the repo is not a checkout, or the candidate list is malformed.
Provenance capture must never be the reason a batch run dies -- a row with
code_version="unknown" is strictly better than no row at all.
"""

import datetime
import hashlib
import json
import os
import subprocess
import uuid

# The four columns every decision-bearing table gets. Kept in one place so the
# three call sites (mollm_decisions, normalized_entities, mollm_review_decisions)
# cannot drift apart -- the same "two copies of one idea drifting" failure class
# as the decoding_mode_mismatch bug in the 2026-08-13 report S3.3.
PROVENANCE_COLUMNS = [
    ("created_at", "TIMESTAMP"),
    ("run_id", "VARCHAR"),
    ("code_version", "VARCHAR"),
    ("candidates_hash", "VARCHAR"),
]

_RUN_ID = None
_CODE_VERSION = None


def provenance_alter_statements(table: str) -> list:
    """Additive ALTER TABLE ... ADD COLUMN IF NOT EXISTS for every provenance
    column, for `table`.

    Additive-only by design, matching the migration pattern already used by
    src/mollm_ensemble.py's store_decision() and src/extraction.py's
    extracted_relations table: a CREATE TABLE IF NOT EXISTS is a no-op against
    an EC2 table that already exists, so new columns have to arrive this way or
    they never arrive at all. Existing rows get NULL, which is the honest value
    -- their provenance genuinely was not recorded and must not be invented.
    """
    return [
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sqltype};"
        for name, sqltype in PROVENANCE_COLUMNS
    ]


def apply_provenance_migration(conn, table: str):
    """Runs provenance_alter_statements() against `conn`, tolerating a missing
    table (the caller's own CREATE TABLE IF NOT EXISTS runs first in every
    current call site, but a caller that migrates before creating should get a
    no-op, not a crash).
    """
    for stmt in provenance_alter_statements(table):
        try:
            conn.sql(stmt)
        except Exception:
            # Table absent, or a DuckDB build without ADD COLUMN IF NOT EXISTS.
            # Either way this must not take down the write path.
            pass


def get_run_id() -> str:
    """A stable id for THIS process's run, so every row a single batch writes
    shares one value and can be selected as a unit.

    Read from CNSP_RUN_ID when set, so a shell script driving several python
    processes (scripts/run_stage3_batch.py invoked per note-chunk, say) can
    group them into one logical run. Otherwise a fresh uuid4 per process,
    generated once and memoised -- NOT per call, or rows from the same run
    would not group.
    """
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = os.environ.get("CNSP_RUN_ID") or f"run_{uuid.uuid4().hex[:16]}"
    return _RUN_ID


def get_code_version() -> str:
    """Short git rev of the working tree, suffixed "-dirty" when there are
    uncommitted changes.

    The dirty flag matters more than the rev here: this project's own history
    shows fixes being measured in the same session they are written, so a bare
    commit hash would frequently describe code that is not what actually ran.
    "unknown" when git is unavailable or this is not a checkout -- callers must
    treat code_version as advisory provenance, never as a join key.
    """
    global _CODE_VERSION
    if _CODE_VERSION is not None:
        return _CODE_VERSION
    env = os.environ.get("CNSP_CODE_VERSION")
    if env:
        _CODE_VERSION = env
        return _CODE_VERSION
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rev = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if rev.returncode != 0:
            _CODE_VERSION = "unknown"
            return _CODE_VERSION
        version = rev.stdout.strip()
        status = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0 and status.stdout.strip():
            version += "-dirty"
        _CODE_VERSION = version or "unknown"
    except Exception:
        _CODE_VERSION = "unknown"
    return _CODE_VERSION


def candidates_hash(candidates) -> str:
    """Stable 16-hex-char digest of a candidate list AS PRESENTED.

    Covers, per candidate and IN ORDER: omop_concept_id, match_tier,
    similarity_score (rounded to 4dp, matching the precision normalization.py
    itself stores). Concept NAME is deliberately excluded -- a vocabulary
    refresh that renames a concept without changing its id does not change what
    decision was available to be made, and including the name would produce
    spurious drift signals on every Athena reload.

    similarity_score IS included: a Tier 3 list with the same three concept ids
    at meaningfully different similarities is a different decision problem, and
    rounding to 4dp keeps float jitter from producing false drift.

    Returns None for an empty/None list rather than the hash of nothing, so
    "no candidates" (contradiction-mode records, non_asserted_check records)
    is distinguishable from "candidates I could not read". Never raises: an
    unhashable/malformed list falls back to a digest of its repr, which is
    still stable enough to detect change even if it is not interpretable.
    """
    if not candidates:
        return None
    try:
        payload = []
        for c in candidates:
            if not isinstance(c, dict):
                payload.append(["__nondict__", repr(c)])
                continue
            score = c.get("similarity_score")
            payload.append([
                c.get("omop_concept_id", c.get("concept_id")),
                c.get("match_tier"),
                round(float(score), 4) if score is not None else None,
            ])
        blob = json.dumps(payload, sort_keys=False, default=str)
    except Exception:
        blob = repr(candidates)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def provenance_values(candidates=None, precomputed_hash=None) -> dict:
    """The four values to write, as a dict keyed by column name. Callers thread
    this into their INSERT rather than calling the four functions individually,
    so a table that forgets one column fails visibly at the parameter-count
    check rather than silently writing NULL.

    precomputed_hash wins over `candidates` when supplied. This exists because
    the candidate list is in scope at DECISION time (validate_record) but not
    at WRITE time (store_decision), and threading the full list through the
    artifact purely to hash it later would bloat every stored row with a copy
    of data the artifact already references. Hash where the data lives; carry
    the 16 characters.
    """
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "run_id": get_run_id(),
        "code_version": get_code_version(),
        "candidates_hash": precomputed_hash if precomputed_hash is not None
                           else candidates_hash(candidates),
    }


def provenance_params(candidates=None, precomputed_hash=None) -> list:
    """provenance_values() flattened into PROVENANCE_COLUMNS order, for direct
    splatting into an INSERT's params list. Order is guaranteed by
    PROVENANCE_COLUMNS, not by dict insertion order.
    """
    vals = provenance_values(candidates, precomputed_hash)
    return [vals[name] for name, _ in PROVENANCE_COLUMNS]


def provenance_column_sql() -> str:
    """Comma-joined column names for an INSERT column list."""
    return ", ".join(name for name, _ in PROVENANCE_COLUMNS)


def provenance_placeholders() -> str:
    """Matching comma-joined `?` placeholders."""
    return ", ".join("?" for _ in PROVENANCE_COLUMNS)
