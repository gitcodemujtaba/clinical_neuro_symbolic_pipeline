"""
src/batch_status.py -- lightweight, DB-independent progress reporting for
long-running batch scripts that hold DuckDB's exclusive write lock for
their entire run (scripts/test_pipeline_e2e.py, scripts/run_stage3_tier_gate.py).

WHY THIS EXISTS. DuckDB's single-writer model means NOTHING else -- not
even a read-only connection -- can open the database while one of these
scripts is running (confirmed repeatedly this session: a read-only
connect() from another process raises duckdb.IOException while a batch job
holds the write lock). A UI (ui/pages/1_🚀_Pipeline_Runner.py) or any other
monitoring tool therefore cannot ask the DATABASE how a running job is
doing. This module is the alternative: a small, best-effort JSON file
written to disk (NOT the DB) that any reader can poll without needing DB
access at all.

SAFE BY DEFAULT: every write is wrapped in a broad try/except -- a status-
file write failure (disk full, permissions, whatever) must never interrupt
or slow down the actual batch job it's reporting on. This is best-effort
telemetry, not a source of truth -- mollm_tier_gate_decisions/
extracted_entities remain the real source of truth once the DB is
readable again.
"""
import json
import os
import time

DEFAULT_STATUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", ".batch_status.json")


def write_status(job: str, *, started_at: float, notes_done: int, notes_total: int,
                 entities_done: int, current_note_id: str = None, errors: int = 0,
                 path: str = None) -> None:
    """Overwrites the status file with the current progress snapshot.
    Cheap enough to call once per note from within a batch script's own
    loop (NOT once per entity -- this stays lightweight by design).

    ETA is a simple linear extrapolation from notes_done/notes_total and
    elapsed wall time, deliberately NOT entity-level: this session's own
    real runs showed note-to-note entity counts vary too much (33 to 227+
    entities per note) for a per-entity rate to extrapolate any better than
    a per-note one, and per-note is simpler to reason about.

    Written atomically (write to a .tmp path, then os.replace() -- atomic
    on POSIX) so a concurrent reader never sees a half-written/corrupt
    JSON file, regardless of when it happens to read.
    """
    path = path or DEFAULT_STATUS_PATH
    now = time.time()
    elapsed = now - started_at
    eta_seconds = None
    if 0 < notes_done < notes_total:
        eta_seconds = round(elapsed * (notes_total - notes_done) / notes_done)
    payload = {
        "job": job,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": now,
        "current_note_id": current_note_id,
        "notes_done": notes_done,
        "notes_total": notes_total,
        "entities_done": entities_done,
        "errors": errors,
        "elapsed_seconds": round(elapsed),
        "eta_seconds": eta_seconds,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        pass


def clear_status(job: str, path: str = None) -> None:
    """Called on batch completion (success OR failure -- callers should use
    a try/finally) so a stale status file doesn't claim a job is still
    running after it's actually finished.

    Only removes the file if it still belongs to THIS process (matching
    both job name and pid): scripts/run_stage3_tier_gate.py chains after
    scripts/test_pipeline_e2e.py in the same shell (two separate Python
    processes, same status path) -- without this check, stage 1-2b's own
    cleanup at the end of ITS run could delete the status file stage 3 has
    already started writing, right as the UI is about to read it.
    """
    path = path or DEFAULT_STATUS_PATH
    try:
        if os.path.exists(path):
            with open(path) as f:
                current = json.load(f)
            if current.get("job") == job and current.get("pid") == os.getpid():
                os.remove(path)
    except Exception:
        pass


def format_duration(seconds) -> str:
    """'1h 05m', '12m 03s', '45s' -- shared so both UI pages (and any future
    consumer) format elapsed/ETA identically rather than each inventing
    their own rounding. None/negative -> '—', not '0s' or a crash: an
    unknown duration should look unknown, not falsely precise.
    """
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


JOB_LABELS = {
    "stage1_2b": "Stage 1 → 2b (extraction, expansion, normalization)",
    "stage3_tier_gate": "Stage 3 (two-step CoT ensemble + tier gate)",
}


def read_status(path: str = None):
    """Returns the current status dict, or None if no job is running (file
    absent/unreadable/corrupt). Does NOT guess at process liveness from
    updated_at -- a killed job leaves a stale file behind with no other
    signal, and second-guessing that here would just be a different way to
    be wrong; a caller that cares can compare updated_at against wall time
    itself and decide what "stale" means for its own purpose.
    """
    path = path or DEFAULT_STATUS_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
