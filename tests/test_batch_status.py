"""
tests/test_batch_status.py -- src/batch_status.py's DB-independent progress
file (write_status()/read_status()/clear_status()). Uses a real temp file
on disk (this module's whole point is disk I/O, not DB access), no DB or
heavy deps needed.

Run: python3 -m pytest tests/test_batch_status.py -v
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_status import clear_status, format_duration, read_status, write_status  # noqa: E402


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    check("no status file -> read_status() returns None",
          read_status(path="/tmp/definitely-does-not-exist-cnsp-status.json") is None)

    tmp_path = os.path.join(tempfile.mkdtemp(), "status.json")
    started = time.time() - 120  # pretend the job has been running 2 minutes

    write_status("stage3_tier_gate", started_at=started, notes_done=5, notes_total=20,
                entities_done=300, current_note_id="10043750-DS-6", errors=0, path=tmp_path)
    status = read_status(path=tmp_path)
    check("write then read round-trips the job name", status["job"] == "stage3_tier_gate")
    check("...and current_note_id", status["current_note_id"] == "10043750-DS-6")
    check("...and notes_done/notes_total", status["notes_done"] == 5 and status["notes_total"] == 20)
    check("...and entities_done", status["entities_done"] == 300)
    check("elapsed_seconds is computed and roughly matches the fake started_at",
          115 <= status["elapsed_seconds"] <= 125)
    check("eta_seconds is a positive linear extrapolation (15 notes left, 5 done in ~120s)",
          status["eta_seconds"] is not None and status["eta_seconds"] > 0)
    # elapsed=120s for 5/20 notes -> 15 remaining * (120/5) = 360s expected
    check("eta_seconds is in the right ballpark (~360s)",
          300 <= status["eta_seconds"] <= 420)

    # notes_done == notes_total -> no meaningful "remaining" to extrapolate
    write_status("stage3_tier_gate", started_at=started, notes_done=20, notes_total=20,
                entities_done=1200, path=tmp_path)
    status2 = read_status(path=tmp_path)
    check("eta_seconds is None once notes_done == notes_total (nothing left to extrapolate)",
          status2["eta_seconds"] is None)

    # A second job's status overwrites the file (same path, later write wins)
    write_status("stage1_2b", started_at=time.time(), notes_done=1, notes_total=20,
                entities_done=50, path=tmp_path)
    status3 = read_status(path=tmp_path)
    check("a later write_status() call overwrites the previous job's status",
          status3["job"] == "stage1_2b")

    # clear_status() only removes a file that still belongs to this pid+job
    write_status("stage3_tier_gate", started_at=started, notes_done=1, notes_total=1,
                entities_done=1, path=tmp_path)
    clear_status("stage1_2b", path=tmp_path)  # wrong job name -- must NOT clear
    check("clear_status() with the WRONG job name does not remove the file",
          read_status(path=tmp_path) is not None)
    clear_status("stage3_tier_gate", path=tmp_path)  # right job, right pid (this process)
    check("clear_status() with the matching job+pid removes the file",
          read_status(path=tmp_path) is None)

    check("write_status() never raises even with an unwritable directory",
          write_status("x", started_at=time.time(), notes_done=1, notes_total=2,
                      entities_done=1, path="/root/definitely-not-writable/status.json") is None)
    check("clear_status() never raises on a missing file",
          clear_status("x", path="/tmp/definitely-does-not-exist-cnsp-status.json") is None)

    check("format_duration(None) -> '—'", format_duration(None) == "—")
    check("format_duration(negative) -> '—'", format_duration(-5) == "—")
    check("format_duration(45) -> '45s'", format_duration(45) == "45s")
    check("format_duration(125) -> '2m 05s'", format_duration(125) == "2m 05s")
    check("format_duration(3725) -> '1h 02m'", format_duration(3725) == "1h 02m")

    print(f"batch-status tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_batch_status():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
