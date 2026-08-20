"""src/db_utils.py — shared DuckDB connection helpers for batch scripts.

connect_with_retry() exists for one reason: DuckDB is single-writer-
exclusive, and a connection held open for an entire multi-minute batch
(the pattern every batch script in this repo used until 2026-08-18) locks
out every other connection -- including Streamlit's read-only ones -- for
the connection's WHOLE LIFETIME, not just the moments it's actually
querying. Confirmed directly: a single note's Stage 3 run could hold the
lock for 10-45+ minutes even though the actual DB reads/writes inside that
window are each well under a second -- the rest of the time is spent on
GLiNER/SapBERT/LLM inference that never touches the connection at all.

The fix is cycling connections (open right before a DB operation, close
right after) rather than holding one open for a whole batch -- but a
short-lived connection can now legitimately collide with something else's
brief connection (another batch script's own cycled connection, a
Streamlit page render). connect_with_retry() waits out that collision
instead of crashing, since the other side is expected to also be brief.
"""
import time

import duckdb


def connect_with_retry(db_path: str, read_only: bool = False,
                       max_wait_seconds: float = 300, retry_delay_seconds: float = 2):
    """duckdb.connect(), retrying on a lock-conflict IOException until
    max_wait_seconds elapses. Any other error (a real bug, a missing file,
    ...) raises immediately, unretried -- only the specific "Could not set
    lock" signature is treated as transient."""
    deadline = time.time() + max_wait_seconds
    while True:
        try:
            return duckdb.connect(db_path, read_only=read_only)
        except duckdb.IOException as exc:
            if "Could not set lock" not in str(exc) or time.time() >= deadline:
                raise
            time.sleep(retry_delay_seconds)
