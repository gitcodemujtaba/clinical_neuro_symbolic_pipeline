"""
evaluation/splits.py — enforces docs/Evaluation_Criteria.md's train/val/test
split at the point where it can actually be violated: the argument parser of
every evaluation script.

THE PROBLEM THIS SOLVES. Before 2026-08-13, every script in evaluation/ and
scripts/ defaulted `--note-ids` to "every note_id currently present in the
database". That default is the leakage. It is not that anyone chose to evaluate
on locked notes -- it is that the safe choice required remembering to type
something, and the unsafe choice was what you got by pressing enter.

THE FIX IS THE DEFAULT, NOT THE FLAG. add_split_args() gives every script
`--split`, defaulting to `val`. Getting test data now requires typing
`--split test --unlock-test`, and prints a banner when you do. Nobody
accidentally reports a number from the locked set.

WHY --unlock-test IS SEPARATE FROM --split test. docs/Evaluation_Criteria.md
locks ~70 notes for T0/T1/T2 benchmark comparison against Clinical-T5. Every
time the test set is looked at, a little of its value is spent: a threshold
nudged after seeing test performance is a threshold fit to the test set. One
flag would make that a typo away. Two flags, one of them awkward, makes it a
decision.

CONTAMINATION IS RE-CHECKED AT RUN TIME. scripts/make_splits.py can hold
already-processed notes out of the test bucket only if it was run on a machine
with the database. assert_no_contamination() re-checks on whatever machine is
actually running the evaluation, so a split generated without the DB still gets
caught before a number is reported from it.
"""

import os
import sys

PROJECT_DIR = os.environ.get(
    "CNSP_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

SPLIT_FILE = os.path.join(PROJECT_DIR, "data", "splits", "note_splits.csv")
VALID_SPLITS = ("train", "val", "test", "all")


class SplitUnavailable(RuntimeError):
    """Raised when the split file is missing. Deliberately NOT swallowed into
    a fallback of "score everything": the whole point of this module is that
    the permissive behavior must never be what happens by accident.
    """


def load_split_file(path=SPLIT_FILE):
    """{note_id: split}. Skips '#' header lines written by make_splits.py."""
    if not os.path.exists(path):
        raise SplitUnavailable(
            f"no split file at {path}.\n"
            "docs/Evaluation_Criteria.md requires a locked test set and a "
            "separate validation slice.\nGenerate it once with:\n"
            "    python3 scripts/make_splits.py --exclude-processed\n"
            "Then re-run this script. (Use --split all to deliberately ignore "
            "the split;\nany number produced that way is a development "
            "measurement, not a reportable one.)")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("note_id,"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    if not out:
        raise SplitUnavailable(f"split file {path} contains no rows")
    return out


def split_file_digest(path=SPLIT_FILE):
    """SHA256 of the split file, printed by every script that uses it so a
    reported number is traceable to the exact split that produced it."""
    import hashlib
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_split(name, path=SPLIT_FILE):
    """The set of note_ids in `name`. 'all' returns every note in the file."""
    if name not in VALID_SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {VALID_SPLITS}")
    mapping = load_split_file(path)
    if name == "all":
        return set(mapping)
    return {nid for nid, s in mapping.items() if s == name}


def add_split_args(ap, default="val"):
    """Adds --split / --unlock-test / --split-file to an ArgumentParser.

    Call this INSTEAD OF defining --note-ids as a free-for-all. Scripts that
    already have --note-ids keep it: resolve_note_ids() treats an explicit
    --note-ids as a deliberate override and says so out loud, which is the
    right behavior for one-off debugging, as long as it is never silent.
    """
    ap.add_argument("--split", default=default, choices=VALID_SPLITS,
                    help=f"which note split to evaluate on (default: {default}). "
                         f"'test' additionally requires --unlock-test.")
    ap.add_argument("--unlock-test", action="store_true",
                    help="required alongside --split test. The locked benchmark "
                         "set spends some of its value every time it is read.")
    ap.add_argument("--split-file", default=SPLIT_FILE,
                    help="path to data/splits/note_splits.csv")
    return ap


def resolve_note_ids(args, conn=None, table="extracted_entities", where=None):
    """The note_ids a script should actually evaluate, honouring --split,
    --note-ids and what is present in the database.

    Returns (note_ids, provenance_dict). The provenance dict is meant to be
    printed AND written into any --out JSON, so a saved report always records
    which split produced it -- reports outlive the terminal session that
    produced them.

    Intersects the split with what the database actually has, because a split
    is a design and the database is a fact: asking for 70 test notes when 12
    have been processed should evaluate 12 and SAY 12, not silently return
    the design's number.
    """
    requested = getattr(args, "split", "val")

    if getattr(args, "note_ids", None):
        ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        print("!! --note-ids given explicitly: the split is being BYPASSED. "
              "Any number\n   from this run is a development measurement, not "
              "a reportable one.")
        return ids, {"split": "explicit_note_ids", "n_requested": len(ids),
                     "split_file_sha256": split_file_digest(
                         getattr(args, "split_file", SPLIT_FILE))}

    if requested == "test" and not getattr(args, "unlock_test", False):
        raise SystemExit(
            "\n" + "=" * 72 + "\n"
            "--split test requires --unlock-test.\n\n"
            "The ~70-note test set is locked for the final T0/T1/T2 benchmark\n"
            "comparison against Clinical-T5 (docs/Evaluation_Criteria.md).\n"
            "Every look at it spends some of its value: a threshold adjusted\n"
            "after seeing test performance is a threshold fitted to the test\n"
            "set, and the comparison stops being a fair one.\n\n"
            "Use --split val for anything you intend to iterate on.\n"
            + "=" * 72)

    split_path = getattr(args, "split_file", SPLIT_FILE)
    wanted = load_split(requested, split_path)

    available = None
    if conn is not None:
        # `where` is a caller-supplied literal (e.g. "is_test = TRUE AND
        # mode = 'resolution'"), never user input -- these scripts are run by
        # the project author against a local read-only DuckDB file, and the
        # alternative (a parameterised predicate builder) would add machinery
        # for a threat that does not exist here. Kept as a named argument
        # rather than concatenated into `table` by callers so it is at least
        # obvious where the SQL comes from.
        sql = f"SELECT DISTINCT note_id FROM {table}"
        if where:
            sql += f" WHERE {where}"
        try:
            rows = conn.execute(sql).fetchall()
            available = {r[0] for r in rows}
        except Exception:
            available = None

    if available is not None:
        ids = sorted(wanted & available)
        n_missing = len(wanted) - len(ids)
    else:
        ids = sorted(wanted)
        n_missing = 0

    if requested == "test":
        print("\n" + "!" * 72)
        print("!! LOCKED TEST SET UNLOCKED. Record this run. Do not tune "
              "anything on\n!! what you are about to see.")
        print("!" * 72 + "\n")

    prov = {
        "split": requested,
        "split_file_sha256": split_file_digest(split_path),
        "n_in_split": len(wanted),
        "n_evaluated": len(ids),
        "n_in_split_not_yet_processed": n_missing,
    }
    print(f"split: {requested} "
          f"({len(ids)} of {len(wanted)} notes present in {table})")
    if n_missing:
        print(f"       {n_missing} note(s) in this split have not been "
              f"processed yet; numbers below cover only the {len(ids)} that have.")
    return ids, prov


def assert_no_contamination(conn, split_path=SPLIT_FILE, table="extracted_entities",
                            fatal=False):
    """Warns (or raises, with fatal=True) when notes in the LOCKED test split
    have already been processed through the pipeline.

    Why this is a real check and not paranoia: the 2026-08-13 report S4.3
    records that the ECE pool was "opportunistically grown" with no split in
    existence, so some already-processed notes almost certainly fall in
    whatever test set is drawn afterwards. Those notes informed the fixes that
    were then measured -- e.g. the `fx` groundability heuristic was derived by
    inspecting them. Knowing WHICH notes are affected is what lets you either
    regenerate the split around them (make_splits.py --exclude-processed) or
    report the contamination honestly.
    """
    try:
        test_ids = load_split("test", split_path)
    except SplitUnavailable:
        return None
    try:
        rows = conn.execute(f"SELECT DISTINCT note_id FROM {table}").fetchall()
    except Exception:
        return None
    processed = {r[0] for r in rows}
    overlap = sorted(test_ids & processed)
    if overlap:
        msg = (f"CONTAMINATION: {len(overlap)} of {len(test_ids)} LOCKED test "
               f"notes have already\nbeen processed through the pipeline: "
               f"{', '.join(overlap[:8])}"
               f"{' ...' if len(overlap) > 8 else ''}\n"
               "Regenerate the split with:\n"
               "    python3 scripts/make_splits.py --exclude-processed --force\n"
               "or report the overlap alongside any test-set number.")
        if fatal:
            raise SystemExit(msg)
        print("\n!! " + msg.replace("\n", "\n!! ") + "\n")
    return overlap


# ==========================================================================
# Stub tests -- run: python3 evaluation/splits.py
# ==========================================================================

def _selftest():
    import argparse
    import tempfile

    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "splits.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# header comment\n"
                     "# seed=1\n"
                     "note_id,split,n_gold_annotations\n"
                     "n1,train,10\nn2,val,20\nn3,test,30\nn4,test,40\n")

        check("load val", load_split("val", p) == {"n2"})
        check("load test", load_split("test", p) == {"n3", "n4"})
        check("load all", load_split("all", p) == {"n1", "n2", "n3", "n4"})
        check("comments skipped", "# header comment" not in load_split_file(p))
        check("digest is 64 hex", len(split_file_digest(p)) == 64)

        try:
            load_split("nonsense", p)
            check("bad split name raises", False)
        except ValueError:
            check("bad split name raises", True)

        try:
            load_split("val", os.path.join(d, "missing.csv"))
            check("missing file raises SplitUnavailable", False)
        except SplitUnavailable:
            check("missing file raises SplitUnavailable", True)

        ap = argparse.ArgumentParser()
        ap.add_argument("--note-ids", default=None)
        add_split_args(ap)

        # Default must be val, never all.
        args = ap.parse_args([f"--split-file={p}"])
        check("default split is val", args.split == "val")
        ids, prov = resolve_note_ids(args)
        check("resolves val", ids == ["n2"] and prov["split"] == "val")
        check("provenance carries digest", len(prov["split_file_sha256"]) == 64)

        # test without unlock must refuse.
        args = ap.parse_args(["--split=test", f"--split-file={p}"])
        try:
            resolve_note_ids(args)
            check("test without unlock refuses", False)
        except SystemExit:
            check("test without unlock refuses", True)

        args = ap.parse_args(["--split=test", "--unlock-test", f"--split-file={p}"])
        ids, prov = resolve_note_ids(args)
        check("test with unlock works", sorted(ids) == ["n3", "n4"])

        # Explicit --note-ids bypasses, and says so.
        args = ap.parse_args(["--note-ids=nX,nY", f"--split-file={p}"])
        ids, prov = resolve_note_ids(args)
        check("explicit note-ids bypass", ids == ["nX", "nY"]
              and prov["split"] == "explicit_note_ids")

        # DB intersection: split wants 2 test notes, DB has 1.
        class FakeConn:
            def execute(self, q):
                class R:
                    def fetchall(self_):
                        return [("n3",), ("n9",)]
                return R()

        args = ap.parse_args(["--split=test", "--unlock-test", f"--split-file={p}"])
        ids, prov = resolve_note_ids(args, conn=FakeConn())
        check("intersects with db", ids == ["n3"] and prov["n_evaluated"] == 1
              and prov["n_in_split_not_yet_processed"] == 1)

        overlap = assert_no_contamination(FakeConn(), p)
        check("contamination detected", overlap == ["n3"])

    print(f"splits.py self-test: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
