"""ui/components/fresh10_notes.py -- 2026-08-20: the 10 notes covered by
this session's two fresh-note final-validation runs (genuinely held-out
w.r.t. the SNOMED near-duplicate retrieval fix and KGE evaluation), from
the official locked test split (data/splits/note_splits.csv). Used to
scope the Streamlit UI to exactly this validated population rather than
the full, mixed-vintage corpus.
"""

FRESH10_NOTE_IDS = [
    # batch 1
    "13538696-DS-11", "19895550-DS-7", "11516225-DS-20",
    "14652764-DS-17", "12298181-DS-9",
    # batch 2 (smallest-by-size remaining test-split notes)
    "15706386-DS-9", "15975714-DS-10", "14766716-DS-22",
    "10043750-DS-6", "10371195-DS-9",
]
