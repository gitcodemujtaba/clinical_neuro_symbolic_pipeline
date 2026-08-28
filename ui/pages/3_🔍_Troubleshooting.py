"""ui/pages/3_🔍_Troubleshooting.py — step-by-step input/output walkthrough
for ONE note, built to answer a specific question this session kept hitting
manually via ad-hoc scripts: "why did we miss this entity / pick this wrong
meaning?" Every step shows its INPUT and its OUTPUT, plus a gold overlay, so
the root cause is visible without re-deriving it by hand each time.

LAYOUT (2026-08-17, redesigned after "quite confusing, make whole note
visible on each stage"): a persistent left column holds the full raw note
text, highlighted with every finding at once (ambiguous tokens, below-
threshold near-misses, missed/wrong-concept gold spans once computed) --
it never disappears while paging through the right column's per-stage
tabs, unlike the original stacked-headers layout where the note text was
one collapsed expander at the top, scrolled away from everything else.

Defaults to the smallest already-processed note (by entity count) so the
walkthrough stays fast to read end to end -- override in the sidebar for
any other note. read_only=True, never writes, same locked-DB handling as
every other page.

Reuses scripts/score_gold_recall.py's load_gold()/overlaps() and
src.abbreviation_flywheel.VERIFIED_ALLOW_LIST directly rather than
re-deriving either -- this page is a viewer onto real pipeline state, not a
second implementation of it.
"""
import csv
import html
import json
import os
import sys

import duckdb
import streamlit as st

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
sys.path.insert(0, PROJECT_DIR)

from scripts.score_gold_recall import best_tier, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402
from ui.components.db_status import render_locked_db_status  # noqa: E402
from ui.components.fresh10_notes import FRESH10_NOTE_IDS  # noqa: E402

# Colors used by the note highlighter (render_highlighted_note below):
# ours=green, abbreviation=blue (both pre-existing), plus the two new ones
# for the gold-vs-ours span diff -- GOLD_COLOR for gold's own annotation,
# OVERLAP_COLOR for the exact character range where our span and a gold
# span agree. Kept as module constants so the legend text and the span
# builders below can't drift apart.
GOLD_COLOR = "#e6c229"
OVERLAP_COLOR = "#9e9e9e"

DB_PATH = os.environ.get("CNSP_DB_PATH", os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))

# Same fallback this whole session used: data/raw_notes/{discharge,gold_notes}.csv
# only exist in the sibling (non-_reorder) worktree, not this one -- this path
# is the one that's actually present here and covers the full 272-note pool.
RAW_TEXT_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "raw_notes", "gold_notes.csv"),
    os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv"),
    os.path.join(PROJECT_DIR, "data", "snomed-ct-entity-linking-challenge-1.2.0", "train_notes.csv"),
]

st.set_page_config(page_title="Troubleshooting", page_icon="🔍", layout="wide")
st.title("🔍 Troubleshooting — Step-by-Step Input/Output")
st.caption(
    "Pick or type any processed note_id in the sidebar, then walk through "
    "Stage 1 → Stage 2a → gold comparison → Stage 2b → Stage 3 in the tabs "
    "below. The full note stays visible on the left at every stage. Findings "
    "are highlighted directly in the text as they're computed -- colors are "
    "explained in the legend above it."
)


# 2026-08-18: deliberately NOT @st.cache_resource -- see
# ui/pages/4_📊_Evaluation_Metrics.py's identical comment. A cached
# connection stays open for the server's whole lifetime and blocks any
# background batch job's writes for as long as this page has ever been
# visited.
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


try:
    conn = get_conn()
except duckdb.IOException as exc:
    render_locked_db_status(exc)


def _stop():
    """st.stop(), but closing `conn` first -- see
    ui/pages/1_🚀_Pipeline_Runner.py's identical helper for why an explicit
    close (not just letting `conn` go out of scope) is needed."""
    conn.close()
    st.stop()


@st.cache_data
def load_raw_text(note_id: str):
    for path in RAW_TEXT_CANDIDATES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("note_id") == note_id:
                    return row.get("text") or row.get("note_text")
    return None


def render_highlighted_note(text: str, spans: list, height: str = "78vh"):
    """spans: list of (start, end, priority, color, tooltip_fragment).

    A character can legitimately belong to several categories at once (an
    abbreviation's own span sits INSIDE a larger GLiNER entity span more
    often than not, and either can also be a missed/wrong-concept gold
    span) -- losing that by simple overlap-overwrite would hide exactly the
    kind of cross-stage information this page exists to show. Every
    applicable tag for a character is kept; the highest-`priority` one
    decides the background color, but the tooltip joins ALL of them so
    nothing is silently dropped.
    """
    n = len(text)
    tags = [[] for _ in range(n)]  # list of (priority, color, tooltip) per char
    for start, end, priority, color, tooltip in spans:
        start = max(0, start)
        end = min(n, end)
        tag = (priority, color, tooltip)
        for i in range(start, end):
            tags[i].append(tag)

    def _signature(i):
        # what determines whether consecutive characters can merge into one
        # <mark> run: same winning color AND same full tooltip set.
        if not tags[i]:
            return None
        ordered = tuple(sorted(set(tags[i]), key=lambda t: -t[0]))
        return ordered

    parts = []
    i = 0
    while i < n:
        sig = _signature(i)
        if sig is None:
            j = i
            while j < n and _signature(j) is None:
                j += 1
            parts.append(html.escape(text[i:j]))
            i = j
        else:
            j = i
            while j < n and _signature(j) == sig:
                j += 1
            color = sig[0][1]
            tip = " | ".join(t[2] for t in sig if t[2])
            safe_tip = html.escape(tip)
            parts.append(f'<mark style="background-color:{color}" title="{safe_tip}">'
                        f'{html.escape(text[i:j])}</mark>')
            i = j
    body = "".join(parts).replace("\n", "<br>")
    st.markdown(
        f'<div style="height:{height}; overflow-y:auto; border:1px solid #444; '
        f'border-radius:6px; padding:12px; font-family:monospace; font-size:0.85rem; '
        f'white-space:pre-wrap; line-height:1.5;">{body}</div>',
        unsafe_allow_html=True,
    )


def _split_overlap_spans(pred_start, pred_end, pred_tooltip,
                          gold_start, gold_end, gold_tooltip, match: bool) -> list:
    """One (prediction span, gold span) pair known to overlap -- slices the
    two into up to 3 character ranges rather than one flat highlight, so a
    start/end mismatch between our span and gold's is visible directly in
    the text: the SHARED range (both agree these characters belong to this
    entity) in OVERLAP_COLOR, and whatever each span extends beyond that
    shared range in its own color (green for ours, GOLD_COLOR for gold's).
    When the two spans are character-identical, only the overlap piece is
    non-empty and this collapses to a single grey highlight. Priorities are
    set above every other span type in this page (max elsewhere is 10) so
    this diff always wins the displayed color when it applies.
    """
    verdict = "SNOMED MATCH" if match else "SNOMED MISMATCH"
    ov_start, ov_end = max(pred_start, gold_start), min(pred_end, gold_end)
    out = []
    if ov_start < ov_end:
        out.append((ov_start, ov_end, 25, OVERLAP_COLOR,
                    f"{verdict} (overlap): {pred_tooltip} || GOLD: {gold_tooltip}"))
    if pred_start < ov_start:
        out.append((pred_start, ov_start, 24, "#a5d6a7", f"OURS ONLY (outside gold span): {pred_tooltip}"))
    if pred_end > ov_end:
        out.append((ov_end, pred_end, 24, "#a5d6a7", f"OURS ONLY (outside gold span): {pred_tooltip}"))
    if gold_start < ov_start:
        out.append((gold_start, ov_start, 23, GOLD_COLOR, f"GOLD ONLY (outside our span): {gold_tooltip}"))
    if gold_end > ov_end:
        out.append((ov_end, gold_end, 23, GOLD_COLOR, f"GOLD ONLY (outside our span): {gold_tooltip}"))
    return out


@st.cache_data
def load_gold_note_ids() -> set:
    """Distinct note_ids present in the gold annotations CSV -- read once
    (just the note_id column, not full annotation rows) and cached, so
    the sidebar can filter the browsable list down to notes a gold
    comparison is actually possible for. Returns an empty set (not an
    error) if the gold CSV isn't reachable here -- same degrade-gracefully
    discipline as load_raw_text() above; a caller must treat an empty
    set as 'gold unavailable', not 'zero notes have gold'.
    """
    from evaluation.cal_eval import GOLD_CANDIDATES
    for path in GOLD_CANDIDATES:
        if not os.path.exists(path):
            continue
        ids = set()
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                nid = row.get("note_id")
                if nid:
                    ids.add(nid)
        return ids
    return set()


with st.sidebar:
    st.header("Note selection")
    # 2026-08-28: widened from a fixed FRESH10_NOTE_IDS-only dropdown to
    # the full processed-note pool, per direct request -- "select OR
    # insert a note id" needs both a browsable list (any of the 144
    # notes this box has actually run Stage 1-3 on, not just the 10
    # held-out validation ones) and a free-text override for a note_id
    # typed/pasted directly. extracted_entities itself carries no
    # is_stale/provenance columns (only normalized_entities and
    # mollm_tier_gate_decisions do) -- joined here rather than filtered
    # directly. is_stale=FALSE means "processed by current code" -- see
    # scripts/mark_notes_stale.py for the migration.
    #
    # Further narrowed (same session, follow-up request) to only notes
    # that actually HAVE gold annotations -- tab 3's gold comparison is
    # meaningless (and was silently a no-op) on a note gold says nothing
    # about. Applied to the BROWSABLE list only, not the free-text
    # override below -- that path is a deliberate "I know what I'm
    # doing" escape hatch and stays available for a gold-less note, just
    # with an explicit warning instead of being blocked outright.
    gold_note_ids = load_gold_note_ids()
    if not gold_note_ids:
        st.warning("Gold annotations CSV not reachable here — showing all "
                  "processed notes, unfiltered by gold availability.")

    fresh10_only = st.checkbox("Fresh-10 (validated) notes only", value=False)
    scope_ph = FRESH10_NOTE_IDS if fresh10_only else None
    scope_clause = f"AND note_id IN ({','.join('?' * len(scope_ph))})" if scope_ph else ""
    note_rows = conn.execute(f"""
        SELECT note_id, count(*) AS n FROM extracted_entities
        WHERE is_test = TRUE {scope_clause} AND note_id IN (
            SELECT DISTINCT note_id FROM normalized_entities
            WHERE is_test = TRUE AND is_stale = FALSE
        ) GROUP BY note_id ORDER BY n ASC
    """, scope_ph or []).fetchall()
    if gold_note_ids:
        note_rows = [r for r in note_rows if r[0] in gold_note_ids]
    if not note_rows:
        st.warning("No processed, non-stale notes with gold annotations match this filter yet.")
        _stop()
    note_ids_sorted = [r[0] for r in note_rows]
    counts_by_note = dict(note_rows)
    is_fresh10 = {n: (n in FRESH10_NOTE_IDS) for n in note_ids_sorted}

    picked = st.selectbox(
        f"Browse ({len(note_ids_sorted)} processed notes — type to filter)",
        note_ids_sorted, index=0,
        format_func=lambda n: f"{n}  ({counts_by_note[n]} entities)"
                              f"{'  ⭐ fresh-10' if is_fresh10[n] else ''}")
    typed = st.text_input(
        "...or type/paste any note_id directly",
        placeholder="e.g. 10000032-DS-21", help=(
            "Overrides the dropdown above when non-empty. Works for any "
            "note this box has run Stage 1-3 on, not just the browsable "
            "list — useful for a note_id you already know (e.g. from a "
            "log or another page) without scrolling to find it above."))

    if typed.strip():
        note_id = typed.strip()
        # Looked up directly, unfiltered by the "Fresh-10 only" checkbox
        # above -- counts_by_note may be scoped to just those 10, and a
        # typed note_id outside that scope but otherwise fully processed
        # must not be misreported as unprocessed just because the
        # checkbox happens to be on.
        if note_id not in counts_by_note:
            direct = conn.execute("""
                SELECT count(*) FROM extracted_entities
                WHERE is_test = TRUE AND note_id = ? AND note_id IN (
                    SELECT DISTINCT note_id FROM normalized_entities
                    WHERE is_test = TRUE AND is_stale = FALSE
                )
            """, [note_id]).fetchone()[0]
            if direct:
                counts_by_note[note_id] = direct
                is_fresh10[note_id] = note_id in FRESH10_NOTE_IDS
        if note_id not in counts_by_note:
            # Distinguish "exists but not yet Stage-2b-processed / stale"
            # from "not a real note_id at all in this corpus" -- both are
            # real, different situations a typed id can land in, and the
            # message should say which one this is rather than one flat
            # "not found".
            any_rows = conn.execute(
                "SELECT count(*) FROM extracted_entities WHERE note_id = ?", [note_id]
            ).fetchone()[0]
            if any_rows:
                st.warning(
                    f"`{note_id}` has {any_rows} extracted entities but hasn't "
                    f"completed Stage 2b (or is marked stale) — this page needs "
                    f"a fully-processed note. Re-run it via the 🚀 Pipeline "
                    f"Runner page first.")
            else:
                st.warning(
                    f"`{note_id}` has no extracted entities in this database at "
                    f"all — either it hasn't been run through the pipeline yet "
                    f"(use the 🚀 Pipeline Runner page), or it isn't a note_id "
                    f"this corpus has under `is_test=TRUE`.")
            _stop()
        if gold_note_ids and note_id not in gold_note_ids:
            st.warning(f"`{note_id}` has no gold annotations — tab 3's gold "
                      f"comparison will find nothing for it. Stages 1/2a/2b/3 "
                      f"still work normally.")
        st.caption(f"Typed selection: {counts_by_note[note_id]} extracted entities"
                  f"{'  ⭐ fresh-10 validated' if is_fresh10[note_id] else ''}")
    else:
        note_id = picked
        st.caption(f"Selected: {counts_by_note[note_id]} extracted entities"
                  f"{'  ⭐ fresh-10 validated' if is_fresh10[note_id] else ''}")

raw_text = load_raw_text(note_id)

if "gold_report_by_note" not in st.session_state:
    st.session_state.gold_report_by_note = {}

# --------------------------------------------------------------------------
# Data pulled once, shared by both the highlighter and the tabs below --
# avoids querying the same tables twice for the same information.
# --------------------------------------------------------------------------
prov_row = conn.execute(
    "SELECT provenance FROM note_expansions WHERE note_id = ?", [note_id]).fetchone()
expansions_log = []
if prov_row and prov_row[0]:
    # provenance is a JSON OBJECT ({note_id, abbreviation_dict_version,
    # expansions_count, ambiguous_expansions_count, expansions: [...]}), not
    # a bare list -- the per-token records are under "expansions". Caught
    # live building this page (AttributeError: 'str' object has no
    # attribute 'get'), same shape of mistake made once already this
    # session investigating this table by hand.
    prov = json.loads(prov_row[0]) if isinstance(prov_row[0], str) else prov_row[0]
    expansions_log = prov.get("expansions", [])
ambiguous_entries = [e for e in expansions_log if e.get("ambiguous")]

entity_rows = conn.execute("""
    SELECT original_text, entity_label, confidence, orig_start, orig_end, below_threshold
    FROM extracted_entities WHERE note_id = ? ORDER BY orig_start
""", [note_id]).fetchall()
accepted = [r for r in entity_rows if not r[5]]
below = [r for r in entity_rows if r[5]]

norm_rows = conn.execute("""
    SELECT e.original_text, n.match_tier, n.omop_concept_name, n.similarity_score,
           n.is_ambiguous, e.orig_start, e.orig_end, n.omop_concept_id
    FROM extracted_entities e JOIN normalized_entities n ON n.entity_id = e.entity_id
    WHERE e.note_id = ? ORDER BY e.orig_start
""", [note_id]).fetchall()
# Keyed by (orig_start, orig_end) so the note highlighter can attach
# Stage 2b/SapBERT results to the exact same span GLiNER found -- an
# entity_id join would work too, but offsets are what the highlighter
# already indexes everything else by.
norm_by_span = {(r[5], r[6]): r for r in norm_rows}

# SNOMED codes for the tooltip -- same crosswalk scripts/score_gold_recall.py's
# attach_snomed_codes() uses, just called directly (per-concept, cached) since
# this page needs it attached to a dict keyed by span, not a predictions list.
_vocab = VocabularyRetriever(conn)


@st.cache_data
def _snomed_for_concept(concept_id):
    if concept_id is None:
        return None
    return _vocab.snomed_code_for_concept(concept_id)

tier_rows = conn.execute("""
    SELECT e.original_text, d.tier, d.routing_basis, d.composite_confidence,
           d.models, d.queue_reason, d.final_candidate_index, n.candidates
    FROM extracted_entities e
    JOIN mollm_tier_gate_decisions d ON d.entity_id = e.entity_id
    JOIN normalized_entities n ON n.entity_id = e.entity_id
    WHERE e.note_id = ? ORDER BY e.orig_start
""", [note_id]).fetchall()


def _chosen_candidate_name(final_candidate_index, models_json, candidates_json):
    """The concept name Stage 3 actually landed on, whether that came from
    a clean final_candidate_index (unanimous tiers) or has to be derived
    from the plurality vote (TIER_4_ENSEMBLE_SPLIT -- the same
    plurality_candidate_index() logic evaluation/tier_gate_grading.py uses
    for shadow-precision grading, not re-derived a second way here).

    Returns (label, note) where note explains WHY there's no Stage-3-chosen
    candidate when that's the case -- distinguishing THREE situations that
    used to collapse into one misleading message (caught live on
    'erythema': the first version said "ensemble never evaluated it" while
    the SAME record showed 3 real per-model verdicts, a direct
    self-contradiction):

    (a) Stage 2b genuinely found nothing to link to.
    (b) The ensemble never ran at all -- `models` is a genuinely empty
        array, a true pre-ensemble bypass (queue_reason like
        unresolved_acronym/no_candidates/standalone_qualifier_span;
        confirmed live: 1,105 corpus-wide TIER_5 decisions have empty
        `models`). Caught on 'ITP': Stage 2b resolved it to 'Immune
        thrombocytopenia' at Tier 1 Exact, similarity=1.0, but Stage 3
        never got to see it.
    (c) The ensemble DID run -- `models` has real per-model verdicts --
        but the PLURALITY verdict itself was NONE_CORRECT, so there's no
        candidate to point to even though 3 real model calls happened.
        This is a genuine ensemble rejection, not a bypass, and needs a
        different message: caught on 'erythema' (2/3 NONE_CORRECT, 1/3
        SUPPORTED_1, queue_reason=ensemble_split) -- Stage 2b's Tier 1
        exact match existed, the majority of the ensemble looked at it
        and said it was wrong.
    """
    import json as _json
    candidates = candidates_json
    if isinstance(candidates, str):
        try:
            candidates = _json.loads(candidates)
        except Exception:
            candidates = None

    models = models_json
    if isinstance(models, str):
        try:
            models = _json.loads(models)
        except Exception:
            models = None
    ensemble_ran = bool(models)

    if not candidates:
        return None, "Stage 2b found no candidates at all"

    idx = final_candidate_index
    if idx is None:
        from evaluation.tier_gate_grading import plurality_candidate_index
        idx, _top_verdict, _votes = plurality_candidate_index(models_json)
    if idx is not None and 0 < idx <= len(candidates):
        return candidates[idx - 1].get("concept_name"), None

    top = candidates[0]
    top_desc = f"{top.get('match_tier', '?')}, similarity={top.get('similarity_score')}"
    if ensemble_ran:
        return (top.get("concept_name"),
                f"Stage 2b found this ({top_desc}) but the ensemble's PLURALITY "
                f"verdict was NONE_CORRECT -- a genuine rejection after real "
                f"model evaluation, not a bypass")
    return (top.get("concept_name"),
            f"Stage 2b found this ({top_desc}) but Stage 3's ensemble "
            f"never evaluated it at all -- bypassed by a pre-ensemble hard rule "
            f"before any model call happened")

AUTO_TIER_NAMES = {"TIER_1_AUTO_VALIDATED", "TIER_1B_CALIBRATED_AUTO_VALIDATED",
                   "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED"}


def _vote_summary(models_json):
    """'2/3 SUPPORTED_1, 1/3 NONE_CORRECT' style breakdown from the raw
    per-model verdict array -- this IS the consensus, shown directly rather
    than only as the single composite_confidence number (which is only
    ever populated for calibrator-consulted TIER_1B decisions)."""
    import collections
    import json as _json
    models = models_json
    if isinstance(models, str):
        try:
            models = _json.loads(models)
        except Exception:
            return "—"
    if not models:
        return "—"
    verdicts = [m.get("verdict") for m in models if m.get("verdict")]
    if not verdicts:
        return "—"
    counts = collections.Counter(verdicts)
    n = len(verdicts)
    return ", ".join(f"{c}/{n} {v}" for v, c in counts.most_common())

gold_report = st.session_state.gold_report_by_note.get(note_id)

# --------------------------------------------------------------------------
# Layout: note text (left, persistent) + step tabs (right)
# --------------------------------------------------------------------------
left_col, right_col = st.columns([2, 3])

with left_col:
    st.subheader("The note")
    legend = (
        "🟩 GLiNER entity (+ SapBERT/Stage 2b match) &nbsp; "
        "🟪 GLiNER found it, below acceptance threshold &nbsp; "
        "🟧 ambiguous abbreviation &nbsp; 🟦 abbreviation (single meaning) &nbsp; "
    )
    if gold_report:
        legend += (
            "🟨 gold annotation &nbsp; 🟥 gold entity we missed entirely &nbsp; "
            "⬛ overlap: our span and gold's span agree on these exact characters "
            "&nbsp; <small>(where our span and gold's don't share the same start/end, "
            "the non-overlapping part of each keeps its own color -- green for ours, "
            "gold/yellow for theirs)</small>"
        )
    else:
        legend += "<i>(run Step 3 to also see gold spans here)</i>"
    st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)
    st.caption("Hover any highlight for the full detail — colors can and do stack "
              "(an abbreviation is often inside a larger GLiNER entity span).")

    if raw_text is None:
        st.warning("Raw note text not found in any known CSV location.")
    else:
        spans = []  # (start, end, priority, color, tooltip)

        # Abbreviation expansions -- ALL of them, not just ambiguous ones.
        # Lowest priority: this is Stage 1 input-side info, the entity/
        # normalization highlights on top of it matter more visually.
        for e in expansions_log:
            if e.get("ambiguous"):
                basis = e.get("selection_basis")
                spans.append((e["orig_start"], e["orig_end"], 2, "#ffcc80",
                              f"AMBIGUOUS ABBREV: {e['abbrev']} -> {e['expansion']} (basis: {basis})"))
            else:
                spans.append((e["orig_start"], e["orig_end"], 1, "#90caf9",
                              f"ABBREV: {e['abbrev']} -> {e['expansion']}"))

        # GLiNER entities -- accepted ones carry their Stage 2b/SapBERT
        # match (concept name + tier + similarity) in the SAME tooltip,
        # since it's the same physical span, not a separate highlight.
        for text, label, conf, start, end, is_below in entity_rows:
            if is_below:
                spans.append((start, end, 3, "#ce93d8",
                              f"GLiNER (below threshold): [{label}] confidence={conf:.3f}"))
                continue
            norm = norm_by_span.get((start, end))
            tip = f"GLiNER: {text!r} [{label}] confidence={conf:.3f}"
            if norm:
                _, tier, concept, sim, is_amb, _, _, concept_id = norm
                sim_str = f"{sim:.3f}" if sim is not None else "—"
                snomed = _snomed_for_concept(concept_id)
                tip += (f"  |  SapBERT/Stage2b: tier={tier or '—'} sim={sim_str} "
                       f"-> {concept or 'Unmapped'} (SNOMED {snomed or '—'})")
            spans.append((start, end, 5, "#a5d6a7", tip))

        if gold_report:
            # Missed: gold annotated this, we found nothing overlapping it
            # at all -- no prediction span exists to diff against, so this
            # is a plain gold-colored highlight, not a split.
            for g in gold_report["missed"]:
                spans.append((g["start"], g["end"], 10, "#ef5350",
                              f"MISSED: gold={g['span']!r} ({g['concept_id']})"))
            # Correct and wrong-concept both have a real overlapping
            # prediction -- diff our span against gold's span in both
            # cases (concept correctness is conveyed in the tooltip, not
            # as a separate hue, per the offset-mismatch highlighting spec).
            for e in gold_report["correct"]:
                spans.extend(_split_overlap_spans(
                    e["pred_start"], e["pred_end"], f"{e['predicted_text']!r} -> {e['predicted_concept']!r}",
                    e["start"], e["end"], f"gold={e['span']!r} ({e['concept_id']})", match=True))
            for e in gold_report["wrong_concept"]:
                spans.extend(_split_overlap_spans(
                    e["pred_start"], e["pred_end"],
                    f"{e['predicted_text']!r} -> {e['predicted_concept']!r} ({e['predicted_snomed_code']})",
                    e["start"], e["end"], f"gold wanted {e['gold_concept_id']} ({e['gold_span']!r})",
                    match=False))
            # Extra: a pipeline entity with no gold overlap at all keeps its
            # plain green from the GLiNER-entity loop above -- this only
            # adds a lower-priority tooltip note, it does not change color.
            for p in gold_report["extra"]:
                spans.append((p["orig_start"], p["orig_end"], 4, "#a5d6a7",
                              "PIPELINE EXTRA (no gold overlap here)"))
        render_highlighted_note(raw_text, spans)

with right_col:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Abbreviations", "2. GLiNER extraction", "3. Gold comparison",
        "4. Stage 2b", "5. Stage 3",
    ])

    with tab1:
        st.caption("INPUT: the raw token as GLiNER would otherwise see it. OUTPUT: "
                  "what Stage 1 substituted before extraction, and which tiebreak "
                  "decided it. Highlighted 🟧 in the note.")
        if not expansions_log:
            st.info("No expansion provenance recorded for this note.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Tokens matched in dictionary", len(expansions_log))
            c2.metric("Ambiguous", len(ambiguous_entries))
            c3.metric("Non-ambiguous", len(expansions_log) - len(ambiguous_entries))

            if ambiguous_entries:
                from src.abbreviation_flywheel import VERIFIED_ALLOW_LIST  # noqa: E402
                for e in ambiguous_entries:
                    basis = e.get("selection_basis", "?")
                    allow_listed = e["abbrev"].strip().lower() in VERIFIED_ALLOW_LIST
                    badge = {
                        "unvetted_ambiguous_unexpanded": "🛑 left unexpanded (not on allow-list)",
                        "context_pattern_rule": "✅ deterministic rule (real reviewer-confirmed data)",
                        "observed_frequency_priority": "📊 pipeline's own frequency prior (allow-listed)",
                        "omop_groundability": "🔎 groundability (allow-listed)",
                    }.get(basis, f"🔤 {basis}")
                    with st.container(border=True):
                        cc1, cc2 = st.columns(2)
                        cc1.markdown(f"**INPUT**: `{e['abbrev']}`")
                        cc2.markdown(f"**OUTPUT**: `{e['expansion']}`")
                        st.caption(f"{badge}  |  allow-listed: {'yes' if allow_listed else 'no'}  |  "
                                  f"candidates: {e.get('candidate_expansions')}")

    with tab2:
        st.caption("INPUT: the expanded note text. OUTPUT: every span GLiNER "
                  "proposed, including ones below the 0.5 acceptance threshold "
                  "that get silently excluded downstream. Below-threshold ones "
                  "highlighted 🟪 in the note.")
        c1, c2 = st.columns(2)
        c1.metric("Accepted (confidence ≥ 0.5)", len(accepted))
        c2.metric("Below threshold (0.35–0.5)", len(below))
        with st.expander(f"Below-threshold entities ({len(below)})", expanded=bool(below)):
            if not below:
                st.caption("None for this note.")
            for text, label, conf, start, end, _ in sorted(below, key=lambda r: -r[2]):
                st.text(f"  {text!r:<40s} [{label}]  confidence={conf:.3f}  span=[{start}:{end}]")
        with st.expander(f"Accepted entities ({len(accepted)})"):
            for text, label, conf, start, end, _ in accepted:
                st.text(f"  {text!r:<40s} [{label}]  confidence={conf:.3f}  span=[{start}:{end}]")

    with tab3:
        st.caption("The FULL picture: every gold annotation classified as correct/wrong/missed, "
                  "plus every pipeline entity gold has nothing to say about at all. "
                  "🟨 wrong-concept, 🟥 missed, ⬜ extra highlight in the note once this runs "
                  "(correct matches are already 🟩 in tab 2's highlighting).")
        if st.button("Run gold comparison (may take a moment)"):
            from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
            from scripts.score_gold_recall import (
                attach_snomed_codes, load_gold, load_predictions, score)

            with st.spinner("Scoring against gold..."):
                gold_path = _first_existing(GOLD_CANDIDATES, "gold")
                gold_rows = load_gold(gold_path, [note_id])
                if not gold_rows:
                    st.warning("No gold annotations found for this note.")
                else:
                    predictions = load_predictions(conn, [note_id])
                    attach_snomed_codes(conn, predictions)
                    report = score(gold_rows, predictions)

                    below_spans = [(r[3], r[4], r[2]) for r in below]

                    # Full, uncapped classification of every gold annotation --
                    # score()'s own wrong_concept_examples/missed_span_examples
                    # cap at 15 for its own printed report, which silently drops
                    # entries on any note with more misses than that. Re-derived
                    # here directly from the same overlap+snomed_code match logic
                    # so this view is complete, not a truncated sample.
                    correct, wrong, missed = [], [], []
                    for g in gold_rows:
                        overlapping = [p for p in predictions
                                      if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
                        if not overlapping:
                            missed.append(g)
                            continue
                        hit = next((p for p in overlapping if p["snomed_code"] == g["concept_id"]), None)
                        if hit:
                            # orig_start/orig_end kept alongside gold's own
                            # start/end (both present under the same **g
                            # spread) so the left-column highlighter can
                            # diff the two spans, not just flag a match.
                            correct.append({**g, "predicted_text": hit["original_text"],
                                           "predicted_concept": hit["omop_concept_name"],
                                           "match_tier": hit["match_tier"],
                                           "pred_start": hit["orig_start"], "pred_end": hit["orig_end"]})
                        else:
                            best = best_tier(overlapping)
                            wrong.append({**g, "gold_span": g["span"], "gold_concept_id": g["concept_id"],
                                         "predicted_text": best["original_text"],
                                         "predicted_concept": best["omop_concept_name"],
                                         "predicted_snomed_code": best["snomed_code"],
                                         "match_tier": best["match_tier"],
                                         "start": g["start"], "end": g["end"],
                                         "pred_start": best["orig_start"], "pred_end": best["orig_end"]})

                    # The other direction: pipeline entities gold says NOTHING
                    # about (no overlapping gold span at all) -- not covered by
                    # recall at all, since recall only ever looks FROM gold
                    # outward. Needed for a genuinely "total" comparison, not
                    # just a miss/wrong-concept breakdown.
                    gold_spans_list = [(g["start"], g["end"]) for g in gold_rows]
                    extra = [p for p in predictions
                            if not any(overlaps(p["orig_start"], p["orig_end"], s, e)
                                      for s, e in gold_spans_list)]

                    st.session_state.gold_report_by_note[note_id] = {
                        "combined": report["combined"],
                        "correct": correct,
                        "missed": missed,
                        "wrong_concept": wrong,
                        "extra": extra,
                        "predicted_total": len(predictions),
                        "n_recoverable": sum(
                            1 for g in missed
                            if any(overlaps(b[0], b[1], g["start"], g["end"]) for b in below_spans)),
                    }
                    st.rerun()
        elif gold_report:
            c = gold_report["combined"]
            n_gold = c["gold_annotations"]
            n_correct = len(gold_report["correct"])
            n_wrong = len(gold_report["wrong_concept"])
            n_missed = len(gold_report["missed"])
            n_extra = len(gold_report["extra"])

            st.markdown(f"#### Note `{note_id}`: {n_gold} gold entities, "
                        f"{gold_report['predicted_total']} pipeline entities")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("✅ Correct", n_correct, f"{n_correct/n_gold*100:.0f}% of gold" if n_gold else None)
            rc2.metric("🟨 Wrong concept", n_wrong, f"{n_wrong/n_gold*100:.0f}% of gold" if n_gold else None)
            rc3.metric("🟥 Missed entirely", n_missed, f"{n_missed/n_gold*100:.0f}% of gold" if n_gold else None)
            rc4.metric("⬜ Pipeline extra", n_extra, "no gold overlap")
            st.caption(f"Every gold entity lands in exactly one of correct/wrong/missed "
                      f"({n_correct} + {n_wrong} + {n_missed} = {n_gold}). Separately, "
                      f"{n_extra} pipeline entities have no gold counterpart at all -- this "
                      f"corpus's gold annotation is itself partial, so 'extra' is not "
                      f"automatically a false positive, just unverifiable against gold. "
                      f"(A compound span -- one predicted concept overlapping 2+ gold "
                      f"spans, e.g. 'gunshot wound to abdomen' -- can satisfy at most one "
                      f"of them, so it still lands as correct-for-one, wrong-for-the-rest "
                      f"here, not a separate bucket.)")

            below_spans = [(r[3], r[4], r[2]) for r in below]
            with st.expander(f"✅ Correctly linked ({n_correct})"):
                if not gold_report["correct"]:
                    st.caption("None for this note.")
                for e in gold_report["correct"]:
                    st.text(f"  {e['span']!r:<35s} ({e['concept_id']}) -> "
                           f"{e['predicted_concept']!r}  tier={e['match_tier']}")

            with st.expander(f"🟨 Wrong-concept ({n_wrong})"):
                if not gold_report["wrong_concept"]:
                    st.caption("None for this note.")
                for e in gold_report["wrong_concept"]:
                    st.text(f"  gold {e['gold_span']!r} ({e['gold_concept_id']}) -> "
                           f"predicted {e['predicted_concept']!r} ({e['predicted_snomed_code']})  "
                           f"tier={e['match_tier']}")

            with st.expander(f"🟥 Missed spans ({n_missed}) — with root cause", expanded=True):
                if not gold_report["missed"]:
                    st.caption("None for this note.")
                for g in gold_report["missed"]:
                    hit = [b for b in below_spans if overlaps(b[0], b[1], g["start"], g["end"])]
                    reason = (f"🟡 GLiNER proposed it at confidence {hit[0][2]:.3f} (below threshold)"
                             if hit else "🔴 never proposed by GLiNER at any confidence")
                    st.markdown(f"**{g['span']!r}** ({g['concept_id']}) — {reason}")
                if gold_report["missed"]:
                    st.info(f"{gold_report['n_recoverable']}/{len(gold_report['missed'])} recoverable "
                           f"just by accepting below-threshold candidates.")

            with st.expander(f"⬜ Pipeline extra, no gold overlap ({n_extra})"):
                st.caption("Entities the pipeline confidently extracted and linked, that gold "
                          "doesn't annotate at all here -- either genuine over-extraction, or "
                          "simply outside this corpus's (partial) annotation coverage.")
                if not gold_report["extra"]:
                    st.caption("None for this note.")
                for p in gold_report["extra"]:
                    st.text(f"  {p['original_text']!r:<35s} [{p['entity_label']}] -> "
                           f"{p['omop_concept_name'] or 'Unmapped'}  "
                           f"span=[{p['orig_start']}:{p['orig_end']}]")
        else:
            st.caption("Not run yet for this note.")

    with tab4:
        st.caption("Stage 2b: candidate retrieval + match tier for every entity.")
        with st.expander(f"All normalized entities ({len(norm_rows)})", expanded=True):
            for text, tier, concept, sim, is_amb, _start, _end, _concept_id in norm_rows:
                amb_flag = " 🔀" if is_amb else ""
                sim_str = f"{sim:.3f}" if sim is not None else "—"
                st.text(f"  {text!r:<35s} tier={tier or '—':<14s} sim={sim_str:<6s} -> {concept}{amb_flag}")

    with tab5:
        st.caption("Stage 3 tier-gate routing, grouped by tier — AUTO tiers write to "
                  "KG3 directly, HITL tiers queue for human review.")
        st.caption(
            "⚠️ **composite_confidence is NOT the calibrator's decisive score** — "
            "it's the raw average of the models' own logprob_confidence for "
            "whichever verdict won the plurality vote, stored on every decision "
            "regardless of outcome (src/mollm_tier_gate.py:681). The actual "
            "gating number for TIER_1B (`calibrated_score >= 0.72`) is only "
            "persisted into `routing_basis` text when it CLEARS the threshold — "
            "for a rejected TIER_4 entity, the calibrator's real score is "
            "computed but never stored anywhere, a real pipeline gap, not a "
            "UI omission.")
        if not tier_rows:
            st.info("This note hasn't been through Stage 3 yet (Stage 1→2b only).")
        else:
            by_tier = {}
            for text, tier, basis, conf, models, queue_reason, final_idx, candidates in tier_rows:
                by_tier.setdefault(tier or "None", []).append(
                    (text, basis, conf, models, queue_reason, final_idx, candidates))

            tier_counts = {t: len(rows) for t, rows in by_tier.items()}
            st.bar_chart(tier_counts)

            n_auto = sum(len(rows) for t, rows in by_tier.items() if t in AUTO_TIER_NAMES)
            c1, c2 = st.columns(2)
            c1.metric("AUTO (writes to KG3)", n_auto)
            c2.metric("HITL (queued for review)", len(tier_rows) - n_auto)

            # AUTO tiers first, then HITL tiers, each a distinct expander so
            # "which entities are in HITL" is a direct look, not a scroll
            # through one flat list.
            ordered_tiers = (
                [t for t in by_tier if t in AUTO_TIER_NAMES]
                + [t for t in by_tier if t not in AUTO_TIER_NAMES])
            for tier in ordered_tiers:
                rows = by_tier[tier]
                badge = "✅ AUTO" if tier in AUTO_TIER_NAMES else "🧑‍⚕️ HITL"
                with st.expander(f"{badge}  {tier}  ({len(rows)})", expanded=tier in AUTO_TIER_NAMES):
                    for text, basis, conf, models, queue_reason, final_idx, candidates in rows:
                        conf_str = f"raw model-vote avg={conf:.3f}" if conf is not None else _vote_summary(models)
                        concept, bypass_note = _chosen_candidate_name(final_idx, models, candidates)
                        label = f"-> {concept}" if concept else "-> (nothing found at Stage 2b either)"
                        st.markdown(f"**{text!r}** {label}")
                        if bypass_note:
                            st.caption(f"⚠️ {bypass_note}")
                        st.caption(conf_str)
                        if queue_reason:
                            st.caption(f"queue_reason: {queue_reason}")
                        if basis:
                            st.caption(f"{basis}")

conn.close()
