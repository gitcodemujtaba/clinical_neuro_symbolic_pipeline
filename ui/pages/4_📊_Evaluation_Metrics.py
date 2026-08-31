"""ui/pages/4_📊_Evaluation_Metrics.py — real, live metrics for whichever
notes are actually selected. Every number here is computed fresh from the
DB for the current selection -- nothing is pre-baked or simulated, matching
this project's own standing discipline (numbers get quoted from an actual
run, not assumed). read-only: this page never writes.

FOUR TABS, FOUR DIFFERENT QUESTIONS (deliberately not merged into one
"score" -- conflating them is exactly what this project's own grading
scripts have consistently avoided all session):
  Tier distribution — of what WE processed, where did it land (Stage 3).
  Precision          — of what WE decided, how often were we RIGHT (vs gold).
  Recall/completeness — of what GOLD has, how much did we even FIND at all
                        (Stage 1/2, a different question precision can't
                        answer -- see the "are we comparing via completeness"
                        discussion this session).
  Calibrator status   — is the ConsensusCalibrator model actually loaded/
                        fitted, and what was it trained on.
"""
import os
import sys

import duckdb
import streamlit as st

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
sys.path.insert(0, PROJECT_DIR)

from ui.components.db_status import (  # noqa: E402
    render_locked_db_status, render_mixed_connection_status)
from ui.components.note_batches import NOTE_BATCHES  # noqa: E402

DB_PATH = os.environ.get("CNSP_DB_PATH", os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))

st.set_page_config(page_title="Evaluation Metrics", page_icon="📊", layout="wide")
st.title("📊 Evaluation Metrics")


# 2026-08-18: deliberately NOT @st.cache_resource -- a cached connection
# stays open for the entire Streamlit server process lifetime, which holds
# DuckDB's single-writer lock (readers exclude a writer too, confirmed
# empirically) and blocks any background pipeline batch job from ever
# writing again once this page has been visited once. A fresh, uncached
# connection here is released (via Python refcounting) once superseded by
# the next script rerun, so the lock is only held for the duration of one
# page render/interaction, not indefinitely.
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


try:
    conn = get_conn()
except duckdb.IOException as exc:
    render_locked_db_status(exc)
except duckdb.ConnectionException as exc:
    render_mixed_connection_status(exc)


def _stop():
    """st.stop(), but closing `conn` first -- see
    ui/pages/1_🚀_Pipeline_Runner.py's identical helper for why an explicit
    close (not just letting `conn` go out of scope) is needed."""
    conn.close()
    st.stop()


def resolve_batch_note_ids(conn, batch_note_ids):
    """Every processed, non-stale note_id, optionally restricted to
    `batch_note_ids` (None means no restriction -- the full corpus).
    Same is_stale=FALSE / is_test=TRUE discipline this page has always
    used, just generalized past the old single hardcoded FRESH10 list.
    """
    if batch_note_ids is not None:
        note_ph = ",".join("?" * len(batch_note_ids))
        restrict_clause = f"AND e.note_id IN ({note_ph})"
        params = list(batch_note_ids)
    else:
        restrict_clause = ""
        params = []
    return [r[0] for r in conn.execute(f"""
        SELECT DISTINCT e.note_id FROM extracted_entities e
        WHERE e.is_test = TRUE {restrict_clause} AND e.note_id IN (
            SELECT DISTINCT note_id FROM normalized_entities
            WHERE is_test = TRUE AND is_stale = FALSE
        ) ORDER BY e.note_id
    """, params).fetchall()]


with st.sidebar:
    st.header("Selection")
    # 2026-08-31: was hardcoded to the single FRESH10_NOTE_IDS restriction
    # -- generalized to every named population this project has actually
    # run (ui.components.note_batches.NOTE_BATCHES), so a "10 notes" vs.
    # "5 notes" vs. "5 notes" question doesn't need re-typing note_ids by
    # hand. See the new "Batch comparison" tab for all of them side by
    # side at once, rather than one at a time.
    batch_name = st.selectbox("Batch", list(NOTE_BATCHES.keys()))
    all_note_ids = resolve_batch_note_ids(conn, NOTE_BATCHES[batch_name])
    if not all_note_ids:
        st.warning(f"No notes in the '{batch_name}' batch are ready (is_stale = FALSE) yet.")
        _stop()
    note_ids = st.multiselect("Notes to include", all_note_ids, default=all_note_ids)
    if not note_ids:
        st.info("Select at least one note.")
        _stop()

# A real gap found while building this: batches that were interrupted and
# resumed later (e.g. the fresh-5 gazetteer batch, capped at 30/note on
# 2026-08-31 then completed on a later date) have one huge consecutive
# gap where processing was genuinely PAUSED, not slow -- confirmed live,
# a ~20.5 HOUR gap in exactly this batch's own real data, which would
# otherwise dominate the mean and make "max entity time" meaningless.
# This project has independently measured real per-decision processing
# time before (median 17.1s / mean 18.6s, docs/FINAL_RESULTS_Single_
# Source_Of_Truth.md's guideline-evidence A/B test cost estimate, a much
# larger and cleaner sample) -- no real single decision has ever been
# observed anywhere near this threshold, so 10 minutes is a generous
# ceiling that only ever excludes a genuine inter-run pause, never real
# (if slow) processing.
MAX_PLAUSIBLE_ENTITY_GAP_SECONDS = 600


def compute_timing_stats(conn, note_ids):
    """Real Stage 3 processing-time stats, computed from
    mollm_tier_gate_decisions.created_at -- NOT gold-dependent (unlike
    compute_overall_metrics()), so this still returns real numbers even
    for a batch/selection with no gold coverage.

    Per-entity time: the gap between each decision's created_at and the
    PREVIOUS decision's, across the whole chronological stream for these
    notes (matches how every Stage 3 runner in this codebase actually
    processes entities -- one live LLM-ensemble call at a time,
    sequentially, not in parallel) -- mean/min/max of those gaps, EXCLUDING
    any gap over MAX_PLAUSIBLE_ENTITY_GAP_SECONDS (a real pause between
    runs, not real processing time -- see that constant's own comment).

    Per-note time: NOT simply (last - first) within a note -- that has
    the exact same pause-contamination problem if a note's own decisions
    span an interruption. Instead, the SUM of that note's own internal
    consecutive gaps that are individually under the same plausibility
    ceiling -- i.e. real accumulated processing time for that note, robust
    to a pause landing in the middle of it. mean/min/max ACROSS notes.

    Returns a dict of None values (not zeros) when fewer than 2 decisions
    exist to form a gap -- "no timing data" is a different, real state
    from "processing was instantaneous." Also reports how many gaps were
    excluded as implausible pauses, so this exclusion is never silent.
    """
    import collections

    note_ph = ",".join("?" * len(note_ids))
    rows = conn.execute(f"""
        SELECT note_id, created_at FROM mollm_tier_gate_decisions
        WHERE note_id IN ({note_ph}) AND created_at IS NOT NULL
        ORDER BY created_at
    """, note_ids).fetchall()

    empty = {"entity_mean_s": None, "entity_min_s": None, "entity_max_s": None,
            "note_mean_s": None, "note_min_s": None, "note_max_s": None,
            "n_entity_gaps": 0, "n_notes_timed": 0, "n_gaps_excluded_as_pause": 0}
    if len(rows) < 2:
        return empty

    # Per-entity: consecutive gaps across the whole chronological stream.
    # A negative/zero gap (two decisions with the same or out-of-order
    # timestamp -- e.g. two notes processed by concurrent runs interleaved
    # in this same note selection) is excluded as impossible, not counted
    # as real processing time; a gap over the plausibility ceiling is
    # excluded as a real pause, tracked separately so the exclusion is
    # visible, not silent.
    gaps = []
    n_excluded = 0
    for i in range(1, len(rows)):
        delta = (rows[i][1] - rows[i - 1][1]).total_seconds()
        if delta <= 0:
            continue
        if delta > MAX_PLAUSIBLE_ENTITY_GAP_SECONDS:
            n_excluded += 1
            continue
        gaps.append(delta)

    # Per-note: sum of that note's OWN internal plausible gaps (not
    # max-min, which would still be contaminated by a pause landing
    # inside a single note's own timestamp range).
    by_note = collections.defaultdict(list)
    for nid, ts in rows:
        by_note[nid].append(ts)
    note_spans = []
    for ts_list in by_note.values():
        if len(ts_list) < 2:
            continue
        ts_list = sorted(ts_list)
        real_span = sum(
            d for d in (
                (ts_list[i] - ts_list[i - 1]).total_seconds() for i in range(1, len(ts_list))
            ) if 0 < d <= MAX_PLAUSIBLE_ENTITY_GAP_SECONDS
        )
        if real_span > 0:
            note_spans.append(real_span)

    if not gaps and not note_spans:
        return {**empty, "n_gaps_excluded_as_pause": n_excluded}
    return {
        "entity_mean_s": sum(gaps) / len(gaps) if gaps else None,
        "entity_min_s": min(gaps) if gaps else None,
        "entity_max_s": max(gaps) if gaps else None,
        "note_mean_s": sum(note_spans) / len(note_spans) if note_spans else None,
        "note_min_s": min(note_spans) if note_spans else None,
        "note_max_s": max(note_spans) if note_spans else None,
        "n_entity_gaps": len(gaps),
        "n_notes_timed": len(note_spans),
        "n_gaps_excluded_as_pause": n_excluded,
    }


def compute_overall_metrics(conn, note_ids):
    """The exact grading logic the Overall tab has always run, factored out
    so the new Batch comparison tab can call it once per named batch
    instead of duplicating it. Returns None if there's no gold for
    `note_ids` (caller decides how to render that), else a dict of every
    number the Overall tab displays, plus the raw counts each is built
    from (n_pred_correct/n_pred_with_concept/auto_n/auto_correct/etc.) so
    a caller can build its own help text without recomputing anything.
    """
    from evaluation.tier_gate_grading import grade_by_tier
    from evaluation import iou_metrics
    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from scripts.score_gold_recall import (
        attach_snomed_codes, load_gold, load_predictions, overlaps, score,
    )
    from src.mollm_tier_gate import AUTO_TIERS
    from src.retrieval import VocabularyRetriever
    import collections as _collections

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    if not gold_rows:
        return None

    gold_by_note = _collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)
    vocab = VocabularyRetriever(conn)

    predictions = load_predictions(conn, note_ids)
    attach_snomed_codes(conn, predictions)
    recall_report = score(gold_rows, predictions)["combined"]

    n_pred_with_concept = sum(1 for p in predictions if p.get("snomed_code"))
    n_pred_correct = sum(
        1 for p in predictions if p.get("snomed_code") and any(
            overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])
            and g["concept_id"] == p["snomed_code"]
            for g in gold_by_note.get(p["note_id"], [])))
    linked_precision = n_pred_correct / n_pred_with_concept if n_pred_with_concept else None
    linked_recall = recall_report["linked_recall"]
    linked_f1 = (2 * linked_precision * linked_recall / (linked_precision + linked_recall)
                if linked_precision and linked_recall else None)

    note_ph = ",".join("?" * len(note_ids))
    all_tier_rows = conn.execute(
        f"SELECT tier, COUNT(*) FROM mollm_tier_gate_decisions "
        f"WHERE note_id IN ({note_ph}) GROUP BY tier", note_ids).fetchall()
    total_decisions = sum(n for _, n in all_tier_rows)
    auto_all = sum(n for t, n in all_tier_rows if t in AUTO_TIERS)
    deflection_rate = auto_all / total_decisions if total_decisions else None

    tier_report = grade_by_tier(conn, note_ids)
    auto_n = sum(r["clean"]["n"] for t, r in tier_report.items() if t in AUTO_TIERS)
    auto_correct = sum(r["clean"]["n_correct"] for t, r in tier_report.items() if t in AUTO_TIERS)
    auto_precision = auto_correct / auto_n if auto_n else None

    bench = iou_metrics.benchmark_char_iou(conn, note_ids, gold_by_note, vocab)

    n_eligible = conn.execute(f"""
        SELECT count(*) FROM extracted_entities e
        WHERE e.note_id IN ({note_ph}) AND e.is_test = TRUE
        AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
        AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
        AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
    """, note_ids).fetchone()[0]

    return {
        "gold_annotations": recall_report["gold_annotations"],
        "span_recall": recall_report["span_recall"],
        "linked_recall": linked_recall,
        "linked_precision": linked_precision,
        "linked_f1": linked_f1,
        "n_pred_correct": n_pred_correct,
        "n_pred_with_concept": n_pred_with_concept,
        "total_decisions": total_decisions,
        "auto_all": auto_all,
        "deflection_rate": deflection_rate,
        "auto_n": auto_n,
        "auto_correct": auto_correct,
        "auto_precision": auto_precision,
        "macro_char_iou": bench["macro_char_iou"],
        "weighted_char_iou": bench["weighted_char_iou"],
        "n_eligible": n_eligible,
        "n_notes": len(note_ids),
    }


@st.cache_data(persist="disk", show_spinner=False)
def _compute_overall_metrics_cached(_conn, note_ids_key: tuple, cache_version: int):
    """Disk-persisted wrapper around compute_overall_metrics(), for the
    FIXED-membership named batches only (Fresh-10, Fresh-5 original,
    Fresh-5 gazetteer -- finished, historical validation runs that don't
    change) -- NOT for "All processed notes", which genuinely grows as
    the corpus does and must always be graded fresh (see the batch loop
    below, which only calls this for batches with a non-None note list).

    `_conn` (underscore prefix): Streamlit's own convention for "don't
    hash this argument, but the function still needs it" -- a DuckDB
    connection object isn't meaningfully hashable/stable across reruns
    anyway. `note_ids_key`: the resolved note_ids as a SORTED TUPLE (the
    real cache key, alongside `cache_version`) -- persisted to disk
    (~/.streamlit/cache/), survives an app restart, so this genuinely
    avoids recomputing a finished batch's grading every single time this
    tab is opened, not just within one server session.

    `cache_version`: bumped by the "Force recalculate" button below to
    invalidate a specific batch's cache entry on demand -- these batches
    are believed stable, but if one is ever deliberately re-run/re-graded
    (e.g. the gazetteer batch gets a code fix and is re-processed), a
    reviewer needs a way to get a fresh number without waiting for
    something else to expire the cache.
    """
    return compute_overall_metrics(_conn, list(note_ids_key))


@st.cache_data(persist="disk", show_spinner=False)
def _compute_timing_stats_cached(_conn, note_ids_key: tuple, cache_version: int):
    """Same caching rationale/contract as _compute_overall_metrics_cached()
    above, for compute_timing_stats() instead -- a separate cache entry
    since timing doesn't depend on gold and the two are computed from
    different queries, but shares the same cache_version counter so one
    "Force recalculate" click refreshes both together."""
    return compute_timing_stats(_conn, list(note_ids_key))


tab_overall, tab_batches, tab_tiers, tab_precision, tab_recall, tab_calibration, tab_calibrator = st.tabs(
    ["Overall", "Batch comparison", "Tier distribution", "Precision vs. gold",
     "Recall / completeness", "ECE & IoU (per stage)", "Calibrator status"])

# ==========================================================================
# TAB 0 — Overall: one consolidated, cross-stage summary. The other tabs
# each answer ONE question for ONE stage (deliberately, see this file's
# own module docstring on why they're not merged) -- this tab exists
# because a single "how good is the pipeline, right now, on this
# selection" glance was still missing. Combines existing grading functions
# rather than re-deriving any of them.
# ==========================================================================
with tab_overall:
    st.caption("One consolidated view across all stages for the current note selection. "
              "Each number still comes from its own stage's own definition of 'correct' "
              "(see the per-stage tabs) -- this just puts them side by side.")
    if st.button("Run overall summary (grades every stage against gold)"):
        with st.spinner("Grading every stage against gold..."):
            m = compute_overall_metrics(conn, note_ids)

        if m is None:
            st.warning("No gold annotations found for the selected note(s).")
        else:
            st.markdown("#### Completeness (Stage 1/2 — of everything gold has, how much did we find?)")
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Gold annotations", m["gold_annotations"])
            oc2.metric("Span recall", f"{m['span_recall']*100:.1f}%")
            oc3.metric("Linked recall", f"{m['linked_recall']*100:.1f}%")

            st.markdown("#### Linked concept-level Precision / Recall / F1 (same population, all tiers)")
            of1, of2, of3 = st.columns(3)
            of1.metric("Linked precision", f"{m['linked_precision']*100:.1f}%" if m["linked_precision"] is not None else "n/a",
                      help=f"{m['n_pred_correct']}/{m['n_pred_with_concept']} of our own resolved links (any tier) match gold")
            of2.metric("Linked recall", f"{m['linked_recall']*100:.1f}%")
            of3.metric("Linked F1", f"{m['linked_f1']*100:.1f}%" if m["linked_f1"] is not None else "n/a")

            # 2026-08-28: made explicit after a real, confirmed-live finding
            # -- scripts/run_fresh5_final_validation.py caps Stage 3 at the
            # first 25 (by orig_start) Stage-2b-eligible entities PER NOTE,
            # regardless of note size ("still a real, gradable held-out
            # sample" -- its own comment). total_decisions below is real
            # and correctly computed, but a reader who doesn't know about
            # the cap could easily mistake it for full note coverage --
            # this note prevents that, using the real extracted-entity
            # count as the comparison, not a hardcoded "25".
            if m["n_eligible"] > m["total_decisions"]:
                st.caption(f"⚠️ **{m['total_decisions']} of {m['n_eligible']}** Stage-2b-eligible entities "
                          f"in the selected note(s) actually reached Stage 3 — a capped validation "
                          f"run (e.g. scripts/run_fresh5_final_validation.py, ~25-30/note) rather "
                          f"than full coverage. The numbers below are accurate for that capped "
                          f"population, not the whole note.")

            st.markdown("#### Stage 3 gate — deflection rate & AUTO-tier precision")
            oc4, oc5, oc6 = st.columns(3)
            oc4.metric("Total Stage 3 decisions", m["total_decisions"])
            oc5.metric("Deflection rate", f"{m['auto_all']}/{m['total_decisions']}",
                      f"{m['deflection_rate']*100:.1f}%" if m["deflection_rate"] is not None else "—",
                      help="Fraction of ALL Stage 3 decisions that auto-wrote with no human "
                           "review (every AUTO_TIERS decision, not just the gradable subset).")
            oc6.metric("AUTO-tier precision", f"{m['auto_correct']}/{m['auto_n']}" if m["auto_n"] else "n/a",
                      f"{m['auto_precision']*100:.1f}%" if m["auto_precision"] is not None else None,
                      help="Of AUTO-written decisions we can grade against gold (clean single "
                           "span overlap), how often did we pick gold's exact SNOMED concept?")

            st.markdown("#### Benchmark metric (Stage 2b — DrivenData's own char-level IoU)")
            oc7, oc8 = st.columns(2)
            oc7.metric("Macro char IoU", m["macro_char_iou"])
            oc8.metric("Support-weighted char IoU", m["weighted_char_iou"])
    else:
        st.info("Click the button above to grade the current note selection across every stage.")

# ==========================================================================
# TAB (new) — Batch comparison: the SAME compute_overall_metrics() grading,
# run once per NAMED batch (ui.components.note_batches.NOTE_BATCHES), side
# by side in one table -- "check the metric overall AND based on batches
# (10 notes, then the two 5-note batches)" in one place, rather than
# re-selecting one batch at a time in the sidebar. Deliberately IGNORES the
# sidebar's current note_ids/multiselect -- always grades each batch's own
# FULL resolved population, matching every manual SSOT comparison table
# this project has built by hand this session.
# ==========================================================================
with tab_batches:
    st.caption("Every named note population this project has actually run, graded and shown "
              "side by side -- the same comparison this session has built by hand repeatedly "
              "(docs/FINAL_RESULTS_Single_Source_Of_Truth.md's own tables), now live. Each batch "
              "is graded on its OWN full resolved population, independent of the sidebar's "
              "current note selection above.")
    st.caption("**Caching**: the fixed-membership batches (Fresh-10, Fresh-5 original, Fresh-5 "
              "gazetteer) are finished, historical validation runs that don't change -- their "
              "results are cached to disk (survive an app restart, not just this session) so "
              "this tab doesn't re-grade them every time it's opened. **'All processed notes'** "
              "genuinely grows as the corpus does and is always graded fresh, never cached.")

    if "batch_cache_version" not in st.session_state:
        st.session_state.batch_cache_version = 0
    bc1, bc2 = st.columns([3, 1])
    run_clicked = bc1.button("Run batch comparison (grades every named batch against gold)")
    if bc2.button("🔄 Force recalculate (ignore cache)",
                  help="Bumps the cache key for the fixed-membership batches, forcing them to "
                       "re-grade from scratch -- use this if one of them was deliberately "
                       "re-run/re-processed and the cached numbers are now stale."):
        st.session_state.batch_cache_version += 1
        run_clicked = True

    def _fmt_s(seconds):
        if seconds is None:
            return "n/a"
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"

    if run_clicked:
        rows = []
        for name, batch_ids in NOTE_BATCHES.items():
            resolved = resolve_batch_note_ids(conn, batch_ids)
            if not resolved:
                rows.append({"Batch": name, "Notes": 0})
                continue
            with st.spinner(f"Grading '{name}' ({len(resolved)} notes)..."):
                if batch_ids is None:
                    # "All processed notes" -- genuinely grows over time,
                    # never cached.
                    m = compute_overall_metrics(conn, resolved)
                    t = compute_timing_stats(conn, resolved)
                else:
                    # Fixed-membership batch -- cached to disk, keyed on
                    # the exact resolved note_ids plus the force-recalculate
                    # counter.
                    key = tuple(sorted(resolved))
                    m = _compute_overall_metrics_cached(conn, key, st.session_state.batch_cache_version)
                    t = _compute_timing_stats_cached(conn, key, st.session_state.batch_cache_version)

            # Timing is NOT gold-dependent -- shown even when m is None
            # (no gold coverage for this batch), unlike every grading
            # column below it.
            timing_cols = {
                "Entity time (mean/min/max)":
                    f"{_fmt_s(t['entity_mean_s'])} / {_fmt_s(t['entity_min_s'])} / {_fmt_s(t['entity_max_s'])}"
                    if t["entity_mean_s"] is not None else "n/a",
                "Note time (mean/min/max)":
                    f"{_fmt_s(t['note_mean_s'])} / {_fmt_s(t['note_min_s'])} / {_fmt_s(t['note_max_s'])}"
                    if t["note_mean_s"] is not None else "n/a",
                "Pauses excluded": t["n_gaps_excluded_as_pause"],
            }

            if m is None:
                rows.append({"Batch": name, "Notes": len(resolved), "Gold annotations": 0, **timing_cols})
                continue
            rows.append({
                "Batch": name,
                "Notes": len(resolved),
                "Gold annotations": m["gold_annotations"],
                "Span recall": f"{m['span_recall']*100:.1f}%",
                "Linked recall": f"{m['linked_recall']*100:.1f}%",
                "Linked precision": f"{m['linked_precision']*100:.1f}%" if m["linked_precision"] is not None else "n/a",
                "Linked F1": f"{m['linked_f1']*100:.1f}%" if m["linked_f1"] is not None else "n/a",
                "Deflection rate": f"{m['deflection_rate']*100:.1f}%" if m["deflection_rate"] is not None else "n/a",
                "AUTO-tier precision": f"{m['auto_precision']*100:.1f}%" if m["auto_precision"] is not None else "n/a",
                "Macro char IoU": m["macro_char_iou"],
                **timing_cols,
            })
        st.table(rows)
        st.caption("Every grading column above uses the same functions as the other tabs on this "
                  "page -- not a separate methodology. 'All processed notes' includes every batch "
                  "below it, so it is not an independent data point -- it is the corpus those "
                  "batches are drawn from.")
        st.caption(f"**Timing columns**: real Stage 3 processing time, from mollm_tier_gate_decisions"
                  f".created_at. 'Entity time' = the gap between each decision and the one before "
                  f"it, across the whole chronological stream for that batch (mean/min/max). "
                  f"'Note time' = each note's own ACCUMULATED real processing time (sum of its "
                  f"internal consecutive gaps, mean/min/max across notes in the batch) -- NOT "
                  f"simply last-minus-first, which would be thrown off by a pause landing inside "
                  f"one note. 'Pauses excluded' = gaps over "
                  f"{MAX_PLAUSIBLE_ENTITY_GAP_SECONDS/60:.0f} minutes, treated as the batch being "
                  f"interrupted and resumed later (this project's own real measured per-decision "
                  f"time has never been anywhere near that high), not as slow processing -- a real "
                  f"one was found and excluded while building this (the gazetteer batch's own "
                  f"~20.5-hour gap between its capped run and its later completion). Both timing "
                  f"metrics need >=2 plausible decisions to compute; 'n/a' means too few.")

# ==========================================================================
# TAB 1 — tier distribution (of what we processed, where did it land)
# ==========================================================================
with tab_tiers:
    st.caption("Of every Stage 3 decision for the selected notes, which tier did it land in? "
              "This is coverage of OUR OWN output, not completeness against gold — see the "
              "Recall tab for that.")
    note_ph = ",".join("?" * len(note_ids))
    tier_rows = conn.execute(f"""
        SELECT tier, mollm_routing_decision, count(*) FROM mollm_tier_gate_decisions
        WHERE note_id IN ({note_ph}) GROUP BY tier, mollm_routing_decision
    """, note_ids).fetchall()

    if not tier_rows:
        st.info("No Stage 3 tier-gate decisions found for the selected notes.")
    else:
        from src.mollm_tier_gate import AUTO_TIERS
        total = sum(r[2] for r in tier_rows)
        auto_n = sum(r[2] for r in tier_rows if r[0] in AUTO_TIERS)

        # Same real, confirmed-live cap as the Overall tab's identical note --
        # see that comment for the full explanation.
        n_eligible = conn.execute(f"""
            SELECT count(*) FROM extracted_entities e
            WHERE e.note_id IN ({note_ph}) AND e.is_test = TRUE
            AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
            AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
            AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
        """, note_ids).fetchone()[0]
        if n_eligible > total:
            st.caption(f"⚠️ This is {total} of {n_eligible} Stage-2b-eligible entities in the "
                      f"selected note(s) — the validation run caps Stage 3 at ~25/note, so this "
                      f"distribution is over that capped sample, not full note coverage.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total decisions", total)
        c2.metric("AUTO coverage", f"{auto_n}/{total}",
                  f"{auto_n/total*100:.1f}%" if total else "—")
        c3.metric("Distinct tiers seen", len({r[0] for r in tier_rows}))

        tier_counts = {}
        for tier, _routing, n in tier_rows:
            tier_counts[tier or "None"] = tier_counts.get(tier or "None", 0) + n
        st.bar_chart(tier_counts)

        st.markdown("**Breakdown:**")
        for tier, n in sorted(tier_counts.items(), key=lambda kv: -kv[1]):
            badge = "✅ AUTO" if tier in AUTO_TIERS else "🧑‍⚕️ HITL"
            st.text(f"{badge}  {tier:<38s} {n:>5d}  ({n/total*100:5.1f}%)")

# ==========================================================================
# TAB 2 — precision vs. gold, per tier
# ==========================================================================
with tab_precision:
    st.caption("Of decisions that already have a corresponding gold annotation, how often did "
              "we pick the SAME SNOMED concept as gold? TIER_4_ENSEMBLE_SPLIT is graded on its "
              "PLURALITY candidate even though it's routed to HITL — this is the \"shadow "
              "precision\" reading that shows how much of that population the calibrator has a "
              "real shot at.")
    if st.button("Run precision grading (queries gold annotations + SNOMED crosswalk)"):
        from evaluation.tier_gate_grading import grade_by_tier
        with st.spinner("Grading against gold..."):
            report = grade_by_tier(conn, note_ids)
        if not report:
            st.info("No gradable decisions for the selected notes.")
        else:
            for tier, r in report.items():
                clean = r["clean"]
                with st.container(border=True):
                    st.markdown(f"**{tier}** — {r['n_decisions']} decision(s)")
                    if clean["n"] == 0:
                        st.caption(f"0 clean-span gradable (skipped: {r['raw']['skipped']})")
                        continue
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Gradable (clean-span)", clean["n"])
                    cc2.metric("Correct", clean["n_correct"])
                    cc3.metric("Precision", f"{clean['precision']*100:.1f}%")

# ==========================================================================
# TAB 3 — recall / completeness against gold (Stage 1/2)
# ==========================================================================
with tab_recall:
    st.caption("Of everything GOLD annotated for these notes, how much did we even FIND "
              "(span recall, Stage 1) and how much did we find AND link to the correct concept "
              "(linked recall, Stage 1+2)? This is the completeness check — precision above "
              "can't reveal a systematic miss, since it only ever looks at entities we DID "
              "produce.")
    if st.button("Run recall grading (scripts/score_gold_recall.py)"):
        from scripts.score_gold_recall import (
            GOLD_CANDIDATES, _first_existing, attach_snomed_codes, load_gold, load_predictions,
            score,
        )
        with st.spinner("Loading gold annotations and predictions..."):
            gold_path = _first_existing(GOLD_CANDIDATES, "gold")
            gold_rows = load_gold(gold_path, note_ids)
            if not gold_rows:
                st.warning(f"No gold annotations found for the selected note(s) in "
                          f"{os.path.basename(gold_path)} — they may not be part of the "
                          f"annotated gold set.")
            else:
                predictions = load_predictions(conn, note_ids)
                attach_snomed_codes(conn, predictions)
                report = score(gold_rows, predictions)
                c = report["combined"]
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Gold annotations", c["gold_annotations"])
                rc2.metric("Span recall", f"{c['span_recall']*100:.1f}%",
                          help="Stage 1 (GLiNER) — fraction of gold spans with ANY overlapping prediction")
                rc3.metric("Linked recall", f"{c['linked_recall']*100:.1f}%",
                          help="Stage 1+2 — fraction of gold spans linked to the CORRECT SNOMED concept")
                rc4.metric("Compound spans", report["compound_spans"]["count"])

                ab = report.get("ambiguous_abbreviation_breakdown") or {}
                if ab:
                    st.markdown("**Ambiguous-abbreviation accuracy by winning tiebreak** "
                               "(abbreviation flywheel):")
                    for basis, v in ab.items():
                        st.text(f"  {basis:<32s} {v['correct']:>3d}/{v['total']:<3d} "
                               f"= {v['accuracy']*100:5.1f}%")

                if report["wrong_concept_examples"]:
                    with st.expander(f"Wrong-concept examples ({len(report['wrong_concept_examples'])} shown)"):
                        for e in report["wrong_concept_examples"]:
                            st.text(f"[{e['note_id']}] gold {e['gold_span']!r} ({e['gold_concept_id']}) "
                                   f"-> {e['predicted_concept']!r} ({e['predicted_snomed_code']})")
                if report["missed_span_examples"]:
                    with st.expander(f"Missed-span examples ({len(report['missed_span_examples'])} shown)"):
                        for g in report["missed_span_examples"]:
                            st.text(f"[{g['note_id']}] {g['span']!r} ({g['concept_id']}) "
                                   f"at [{g['start']}:{g['end']}]")

# ==========================================================================
# TAB 4 — per-stage ECE (calibration) + IoU (overlap), computed live
# ==========================================================================
with tab_calibration:
    st.caption("ECE: does confidence X actually mean 'correct' X% of the time, per stage "
              "(evaluation/stage_calibration.py). IoU: TP/(TP+FP+FN) at the decision level for "
              "each stage, plus the DrivenData SNOMED-CT benchmark's own character-level IoU "
              "definition, shown under Stage 2b since that's the first stage with a resolved "
              "concept (class = SNOMED concept ID; can't be scored before that) "
              "(https://www.drivendata.org/benchmarks/310/benchmark-snomed-ct/page/983/). Three "
              "different 'correct's per stage -- see evaluation/stage_calibration.py's module "
              "docstring -- don't compare ECE values across stages directly, only each stage's "
              "curve against its own threshold.")
    if st.button("Run ECE + IoU analysis"):
        import collections as _collections

        from evaluation import iou_metrics
        from evaluation.stage_calibration import (
            AUTO_VALIDATE_THRESHOLD, EXTRACTION_THRESHOLD, TIER3_SIMILARITY_FLOOR,
            grade_stage2a, grade_stage2b_candidates, load_stage2a_rows, load_stage2b_candidates,
            stage3_tier_gate_confidences,
        )
        from evaluation.metrics import compute_ece_report
        from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
        from scripts.score_gold_recall import load_gold
        from src.retrieval import VocabularyRetriever

        with st.spinner("Loading gold + computing per-stage ECE/IoU..."):
            gold_path = _first_existing(GOLD_CANDIDATES, "gold")
            gold_rows = load_gold(gold_path, note_ids)
            gold_by_note = _collections.defaultdict(list)
            for g in gold_rows:
                gold_by_note[g["note_id"]].append(g)

            if not gold_rows:
                st.warning("No gold annotations found for the selected note(s).")
            else:
                vocab = VocabularyRetriever(conn)

                def _show_ece(title, graded, threshold):
                    if not graded:
                        st.caption(f"**{title}**: no gradable data")
                        return
                    report = compute_ece_report(graded)
                    ec1, ec2 = st.columns(2)
                    ec1.metric(f"{title} — ECE", report["ece"])
                    ec2.metric("n", report["n"])
                    with st.expander(f"{title} — reliability table"):
                        for row in report["table"]:
                            if row["n"] == 0:
                                continue
                            st.text(f"  {row['bin']:<16} n={row['n']:>4}  "
                                   f"mean_conf={row['mean_confidence']:.3f}  acc={row['accuracy']:.3f}")

                st.markdown("### Stage 2a — extraction (span is real)")
                s2a_rows = load_stage2a_rows(conn, note_ids)
                s2a_graded = grade_stage2a(s2a_rows, gold_by_note)
                _show_ece("Stage 2a", s2a_graded, EXTRACTION_THRESHOLD)
                s2a_iou = iou_metrics.stage2a_iou(conn, note_ids, gold_by_note)
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Set IoU", s2a_iou["set_iou"])
                ic2.metric("Span-only char IoU", s2a_iou["span_only_char_iou"],
                          help="Concept-blind diagnostic (single aggregate class) -- extraction has "
                               "no resolved concept yet, so this is NOT the DrivenData benchmark "
                               "metric. See Stage 2b below for that.")
                ic3.metric("TP / FP / FN", f"{s2a_iou['tp']} / {s2a_iou['fp']} / {s2a_iou['fn']}")

                st.markdown("### Stage 2b — normalization / linking (concept is right)")
                s2b_candidates = load_stage2b_candidates(conn, note_ids)
                s2b_graded, _ = grade_stage2b_candidates(s2b_candidates, "3 (Semantic)", gold_by_note, vocab)
                _show_ece("Stage 2b (Tier 3 SapBERT)", s2b_graded, TIER3_SIMILARITY_FLOOR)
                s2b_iou = iou_metrics.stage2b_iou(conn, note_ids, gold_by_note, vocab)
                jc1, jc2 = st.columns(2)
                jc1.metric("Set IoU", s2b_iou["set_iou"])
                jc2.metric("TP / FP / FN", f"{s2b_iou['tp']} / {s2b_iou['fp']} / {s2b_iou['fn']}")

                bench = iou_metrics.benchmark_char_iou(conn, note_ids, gold_by_note, vocab)
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Benchmark macro char IoU", bench["macro_char_iou"],
                          help="DrivenData SNOMED-CT benchmark's own definition, confirmed directly "
                               "against the metric section 2026-08-20: class = SNOMED concept ID, a "
                               "predicted span's characters only count toward a class if its own "
                               "resolved concept matches exactly (https://www.drivendata.org/"
                               "benchmarks/310/benchmark-snomed-ct/page/983/). Uses Stage 2b's top "
                               "candidate as the answer for every span, no HITL deferral -- a real "
                               "benchmark submission wouldn't have one either.")
                bc2.metric("Benchmark support-weighted char IoU", bench["weighted_char_iou"])
                bc3.metric("Concept classes scored", bench["n_classes"])

                st.markdown("### Stage 3 — tier gate (AUTO-tier decision is right)")
                s3_graded, s3_n = stage3_tier_gate_confidences(conn, note_ids, gold_by_note, vocab)
                _show_ece("Stage 3 (AUTO tiers)", s3_graded, AUTO_VALIDATE_THRESHOLD)
                s3_iou = iou_metrics.stage3_iou(conn, note_ids, gold_by_note, vocab)
                kc1, kc2, kc3 = st.columns(3)
                kc1.metric("Set IoU", s3_iou["set_iou"])
                kc2.metric("AUTO coverage", f"{s3_iou['n_auto']}/{s3_iou['n_decisions']}")
                kc3.metric("TP / FP / FN", f"{s3_iou['tp']} / {s3_iou['fp']} / {s3_iou['fn']}")

# ==========================================================================
# TAB 5 — calibrator status (static .pkl metadata, no live fit/scoring here)
# ==========================================================================
with tab_calibrator:
    st.caption("Static metadata from the saved model file — this tab never fits or scores "
              "anything live (that needs the training pipeline, not a page load).")
    from src.mollm_tier_calibrator import DEFAULT_MODEL_PATH, ConsensusCalibrator
    from src.mollm_tier_gate import CALIBRATED_AUTO_THRESHOLD

    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH)
    if calibrator.model is None:
        st.warning(f"No fitted model at `{DEFAULT_MODEL_PATH}` (or it failed to load — "
                  f"a load failure always degrades to untrained, never raises).")
    else:
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Training examples", calibrator.n_training_examples)
        cc2.metric("Training notes", len(calibrator.training_note_ids))
        cc3.metric("Feature set version", calibrator.feature_set_version)
        cc4.metric("Auto-promote threshold", CALIBRATED_AUTO_THRESHOLD)
        st.caption(f"code_version: `{calibrator.code_version}`  |  "
                  f"training_split: `{calibrator.training_split}`")

        overlap = calibrator.trained_on_any_of(note_ids)
        if overlap:
            st.error(f"⚠️ **Leakage warning**: {len(overlap)} of the currently-selected note(s) "
                    f"were in this model's OWN training set — any precision number for them is "
                    f"not a fair read of the calibrator's generalization: {overlap[:10]}")
        else:
            st.success("✅ None of the currently-selected notes were in this model's training set "
                     "— a precision number for TIER_1B on this selection is a genuine "
                     "out-of-sample read.")
