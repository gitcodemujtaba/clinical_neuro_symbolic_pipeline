"""
scripts/make_splits.py — writes data/splits/note_splits.csv, the file that
turns docs/Evaluation_Criteria.md's train/validation/test design from a
convention into an enforceable artifact.

WHY THIS EXISTS. docs/Evaluation_Criteria.md specifies a three-way split of the
272-note DrivenData corpus: ~70 notes locked for the final T0/T1/T2 benchmark
against Clinical-T5, and a validation slice used solely to calibrate MoLLM's
thresholds. As of 2026-08-13 NO FILE ANYWHERE IN THE REPOSITORY ENUMERATED
WHICH NOTES BELONG TO WHICH BUCKET (report S4.3). Every evaluation script
defaulted --note-ids to "every note_id currently in the database", leaving the
burden of respecting the split on whoever happened to run the script. The
report's own conclusion:

    "today's ECE numbers are computed over an ad hoc, opportunistically-grown
     note pool that likely overlaps with notes used to derive/validate today's
     own fixes -- a leakage risk the proposal's split design specifically
     exists to prevent."

A split that exists only in a document is not a control. This script produces
the artifact; evaluation/splits.py enforces it.

DETERMINISM. Fixed seed, sorted input, stratified assignment -- rerunning this
script on the same corpus reproduces the same file byte for byte. The script
prints the output's SHA256 so the exact split used for a reported number can be
recorded alongside it. It REFUSES to overwrite an existing file without
--force, because silently regenerating a split mid-project is indistinguishable
from cherry-picking one.

STRATIFICATION BY ANNOTATION DENSITY. Gold annotation counts per note range
from 40 to 675 (median 269) -- a 17x spread. An unstratified random 70-note
draw can easily land a test set that is systematically denser or sparser than
the validation set, which would make the T0/T1/T2 comparison partly a
measurement of note difficulty rather than of the pipeline. Notes are therefore
sorted by annotation count, cut into quintiles, and sampled proportionally
within each.

CONTAMINATION. Notes already processed through the pipeline have, in effect,
been used for development -- the 2026-08-13 fixes were derived by looking at
them. --exclude-processed (with --db) keeps those notes OUT of the test bucket,
so the locked set stays genuinely unseen. Without a DB connection this script
cannot know which notes those are, so it records `contamination_checked=False`
in the header and evaluation/splits.py's assert_no_contamination() re-checks at
run time on the machine that does have the database.

Run:
  python3 scripts/make_splits.py --exclude-processed --db db/kg2_lexical_store.duckdb
  python3 scripts/make_splits.py --dry-run
"""

import argparse
import collections
import csv
import hashlib
import os
import sys

PROJECT_DIR = os.environ.get(
    "CNSP_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, PROJECT_DIR)

DEFAULT_OUT = os.path.join(PROJECT_DIR, "data", "splits", "note_splits.csv")

ANNOTATION_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "snomed-ct-entity-linking-challenge-1.2.0",
                 "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "evaluation-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "evaluaiton-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
]

# docs/Evaluation_Criteria.md: "Approximately 70 notes are locked away and used
# exclusively for final benchmark evaluation at T0, T1, and T2." The validation
# size is not specified there beyond "a slice of the approximately 200 training
# notes"; 60 is chosen to give the calibrator a materially larger gradable
# sample than the n=140 it was fit on (report S5.4) while leaving ~142 notes for
# development, which is where the active-learning stream draws from.
N_TEST = 70
N_VAL = 60
SEED = 20260813
N_STRATA = 5


def load_annotation_counts(path):
    counts = collections.Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts[row["note_id"]] += 1
    return counts


def _first_existing(paths, what):
    for p in paths:
        if os.path.exists(p):
            return p
    raise SystemExit(f"could not find {what}; looked in:\n  " + "\n  ".join(paths))


def stratified_split(counts, n_test, n_val, seed, exclude_from_test=frozenset()):
    """Assigns every note to exactly one of test/val/train.

    Notes are ordered by (annotation_count, note_id) -- the note_id secondary
    key makes ties deterministic, which a bare count sort would not be -- then
    cut into N_STRATA equal strata. Within each stratum the order is shuffled
    with the fixed seed and the first slice goes to test, the next to val, the
    rest to train, proportionally to the requested totals.

    exclude_from_test notes are held out of the test draw entirely and fall
    through to val/train. If that leaves too few eligible notes to fill the
    test bucket, the shortfall is reported rather than silently accepted --
    a 55-note "70-note locked test set" is a different experiment and the
    caller must know.
    """
    import random

    ordered = sorted(counts, key=lambda nid: (counts[nid], nid))
    n = len(ordered)
    strata = []
    per = n // N_STRATA
    for i in range(N_STRATA):
        lo = i * per
        hi = n if i == N_STRATA - 1 else (i + 1) * per
        strata.append(ordered[lo:hi])

    rng = random.Random(seed)
    assignment = {}
    test_short = 0
    for s in strata:
        s = list(s)
        rng.shuffle(s)
        want_test = round(n_test * len(s) / n)
        want_val = round(n_val * len(s) / n)

        eligible_test = [nid for nid in s if nid not in exclude_from_test]
        chosen_test = eligible_test[:want_test]
        test_short += max(0, want_test - len(chosen_test))
        for nid in chosen_test:
            assignment[nid] = "test"

        rest = [nid for nid in s if nid not in assignment]
        for nid in rest[:want_val]:
            assignment[nid] = "val"
        for nid in rest[want_val:]:
            assignment[nid] = "train"

    return assignment, test_short


def processed_note_ids(db_path):
    """note_ids the pipeline has already produced extracted_entities for.
    Read-only. Returns an empty set (with a printed warning, never an
    exception) when the DB is absent -- this script must remain runnable on a
    machine that has the corpus but not the database.
    """
    try:
        import duckdb
    except ImportError:
        print("  ! duckdb not importable; cannot check contamination")
        return set()
    if not os.path.exists(db_path):
        print(f"  ! no database at {db_path}; cannot check contamination")
        return set()
    try:
        conn = duckdb.connect(db_path, read_only=True)
        rows = conn.execute("SELECT DISTINCT note_id FROM extracted_entities").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception as exc:
        print(f"  ! could not read note_ids from {db_path}: {exc}")
        return set()


def write_splits(path, assignment, counts, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Header comment lines start with '#'; evaluation/splits.py skips them.
    # They exist so the file is self-describing when someone opens it in six
    # months without this script to hand.
    lines = [
        "# data/splits/note_splits.csv -- generated by scripts/make_splits.py",
        "# DO NOT EDIT BY HAND. Regenerate with --force and record the new SHA256.",
        f"# seed={meta['seed']} n_test={meta['n_test']} n_val={meta['n_val']} "
        f"strata={meta['n_strata']}",
        f"# contamination_checked={meta['contamination_checked']} "
        f"n_excluded_from_test={meta['n_excluded_from_test']} "
        f"test_shortfall={meta['test_shortfall']}",
        "# split values: test (LOCKED -- final benchmark only) | val "
        "(threshold calibration) | train (development, active learning)",
        "note_id,split,n_gold_annotations",
    ]
    for nid in sorted(assignment):
        lines.append(f"{nid},{assignment[nid]},{counts[nid]}")
    body = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=None,
                    help="path to train_annotations.csv")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--db", default=os.path.join(PROJECT_DIR, "db",
                                                 "kg2_lexical_store.duckdb"))
    ap.add_argument("--exclude-processed", action="store_true",
                    help="keep already-processed notes out of the test bucket")
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the summary without writing anything")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force and not args.dry_run:
        raise SystemExit(
            f"{args.out} already exists.\n"
            "Refusing to overwrite: regenerating a split mid-project is\n"
            "indistinguishable from cherry-picking one. Pass --force only if\n"
            "you intend to invalidate every number measured against the old\n"
            "split, and record the new SHA256 wherever the old one was cited.")

    ann_path = args.annotations or _first_existing(ANNOTATION_CANDIDATES,
                                                   "train_annotations.csv")
    counts = load_annotation_counts(ann_path)
    print(f"annotations: {ann_path}")
    print(f"notes:       {len(counts)}")

    excluded = set()
    if args.exclude_processed:
        excluded = processed_note_ids(args.db) & set(counts)
        print(f"processed:   {len(excluded)} note(s) held out of the test bucket")

    assignment, shortfall = stratified_split(
        counts, args.n_test, args.n_val, args.seed, exclude_from_test=excluded)

    tally = collections.Counter(assignment.values())
    print("\nsplit sizes:")
    for name in ("train", "val", "test"):
        ids = [n for n, s in assignment.items() if s == name]
        dens = sorted(counts[n] for n in ids)
        med = dens[len(dens) // 2] if dens else 0
        print(f"  {name:<6} n={tally[name]:<4} median gold annotations={med}")
    if shortfall:
        print(f"\n  !! test bucket is {shortfall} note(s) short of the requested "
              f"{args.n_test}\n     (too many notes already processed to fill it "
              f"with unseen ones).\n     Report the ACTUAL test size, not the "
              f"intended one.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    meta = {"seed": args.seed, "n_test": args.n_test, "n_val": args.n_val,
            "n_strata": N_STRATA, "contamination_checked": bool(args.exclude_processed),
            "n_excluded_from_test": len(excluded), "test_shortfall": shortfall}
    digest = write_splits(args.out, assignment, counts, meta)
    print(f"\nwrote {args.out}")
    print(f"SHA256 {digest}")
    print("\nRecord that SHA256 alongside any number you report from this split.")
    if not args.exclude_processed:
        print("NOTE: contamination NOT checked (no --exclude-processed). "
              "evaluation/splits.py\n      will re-check at run time against the "
              "live database and warn if the\n      locked test set has already "
              "been processed.")


if __name__ == "__main__":
    main()
