"""ui/components/note_batches.py -- 2026-08-31: every real, named note
population this project has run and reported on, in one place, so the
Evaluation Metrics page can offer them as named batches instead of just
the single hardcoded FRESH10_NOTE_IDS restriction it had before.

WHY A SEPARATE FILE FROM fresh10_notes.py. That module is kept as-is
(still imported directly by name elsewhere) -- this one is the superset,
re-exporting FRESH10_NOTE_IDS alongside the two real fresh-5 batches this
session added, plus an "all processed notes" pseudo-batch. One place to
look up "what note populations does this project actually have," rather
than reconstructing the list from memory or old commit messages each time
a new comparison is needed.

Every list below is the REAL note_id set used for the run it names, not
approximated -- copied verbatim from the scripts that ran them
(scripts/run_fresh5_final_validation.py, scripts/
run_fresh5_gazetteer_validation.py), not re-derived.
"""

from ui.components.fresh10_notes import FRESH10_NOTE_IDS

# The real "Fresh-5 (2026-08-30)" batch -- docs/FINAL_RESULTS_Single_
# Source_Of_Truth.md S10's own headline numbers are built on this exact
# list. NOT evaluation/grade_fresh5_by_tier.py's NOTE_IDS -- that's a
# different, older (2026-08-17) 5-note batch that happens to share the
# name (see docs/Code_Reference_Stages_And_Metrics.md S16 for the mix-up
# this caused once already).
FRESH5_ORIGINAL_NOTE_IDS = [
    "13397956-DS-5", "17739994-DS-31", "16410990-DS-12",
    "16795604-DS-17", "17309807-DS-20",
]

# The real "5 notes (repurposed data)" batch, 2026-08-31 -- 5 short,
# genuinely never-before-processed notes, run with
# CNSP_GLINER_GAZETTEER_FALLBACK=1 (src.gliner_gazetteer_fallback) to
# validate the new GLiNER miss-recovery mechanism end-to-end.
# scripts/run_fresh5_gazetteer_validation.py.
FRESH5_GAZETTEER_NOTE_IDS = [
    "15285988-DS-7", "15906604-DS-2", "14809657-DS-15",
    "19015466-DS-9", "19884924-DS-14",
]

# Batch name -> note_id list, or None meaning "every processed, non-stale
# note currently in the DB" (resolved live, not a fixed list here).
# Ordered deliberately: broadest/most-recent-relevant first.
NOTE_BATCHES = {
    "All processed notes (corpus-wide)": None,
    "Fresh-10 (2026-08-20)": FRESH10_NOTE_IDS,
    "Fresh-5, original (2026-08-30)": FRESH5_ORIGINAL_NOTE_IDS,
    "Fresh-5, gazetteer batch (2026-08-31)": FRESH5_GAZETTEER_NOTE_IDS,
}
