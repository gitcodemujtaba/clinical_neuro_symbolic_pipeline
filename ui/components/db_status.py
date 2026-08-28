"""ui/components/db_status.py — shared "the DB is locked" status display.

Both ui/pages/1_🚀_Pipeline_Runner.py (read-only) and
ui/pages/2_🩺_HITL_Review_Queue.py (read-write) can hit DuckDB's single-
writer lock while a batch job (scripts/test_pipeline_e2e.py,
scripts/run_stage3_tier_gate.py) is running -- confirmed repeatedly this
session, not a hypothetical. Written once here rather than duplicated in
both pages, since the two error paths need to say the exact same thing.
"""
import time

import streamlit as st

from src.batch_status import JOB_LABELS, format_duration, read_status


def render_mixed_connection_status(exc):
    """Call this, then st.stop() (it calls st.stop() itself), when a DuckDB
    connect() call raised duckdb.ConnectionException -- "Can't open a
    connection to same database file with a different configuration than
    existing connections". Distinct cause from render_locked_db_status()'s
    IOException case: this isn't a batch job holding the write lock, it's
    ANOTHER PAGE in this same Streamlit server process (e.g. 🩺 HITL Review
    Queue's read-write connection) still open with different settings than
    the one this page is trying to open. Confirmed live, 2026-08-28: every
    page in this app now explicitly closes its connection at every exit
    point specifically to prevent this, but a genuine race between two
    browser tabs rerunning at the same moment can still hit it, so this
    stays as a friendly fallback rather than assuming the fix makes it
    impossible.
    """
    st.error(
        "**Could not open the database — a connection with different "
        "settings is already open elsewhere in this app.** This usually "
        "means another tab (often 🩺 HITL Review Queue, which needs "
        "read-write access) is mid-render right now. Reload this page — "
        "if it recurs, close other tabs of this app and try again."
    )
    st.caption(f"Underlying error: {exc}")
    st.stop()


def render_locked_db_status(exc):
    """Call this, then st.stop() (it calls st.stop() itself, but a caller
    should treat this as terminal either way) when a DuckDB connect() call
    raised duckdb.IOException. Reads src.batch_status's on-disk progress
    file -- written by the batch script itself, independent of the DB lock
    it's reporting on, so it's readable even while the DB is unreachable.
    Degrades to a plain message if no status file exists (the lock could be
    held by something outside this project's own batch scripts) rather than
    claiming to know more than it does.
    """
    st.error(
        "**Could not open the database.** It's held open by a background "
        "batch job — DuckDB's single-writer model means even a read-only "
        "connection can't open while one runs. This is expected, not a bug."
    )
    status = read_status()
    if not status:
        st.caption("No progress file found for the job holding the lock — try again shortly.")
        st.caption(f"Underlying error: {exc}")
        st.stop()

    job_label = JOB_LABELS.get(status["job"], status["job"])
    st.info(f"**Currently running:** {job_label}")

    notes_done, notes_total = status.get("notes_done", 0), status.get("notes_total", 0)
    if notes_total:
        st.progress(min(1.0, notes_done / notes_total),
                    text=f"Note {notes_done}/{notes_total} complete")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current note", status.get("current_note_id") or "—")
    c2.metric("Entities processed", status.get("entities_done", 0))
    c3.metric("Elapsed", format_duration(status.get("elapsed_seconds")))
    c4.metric("Estimated remaining", format_duration(status.get("eta_seconds")))
    if status.get("errors"):
        st.caption(f"⚠️ {status['errors']} error(s) so far this run.")

    age = time.time() - status.get("updated_at", time.time())
    if age > 300:
        st.warning(
            f"⚠️ This status hasn't updated in {format_duration(age)} — the "
            f"job may have stalled or been killed without cleaning up. If "
            f"it's genuinely gone, the DB should already be free (a stale "
            f"status file doesn't hold the actual lock)."
        )
    st.stop()
