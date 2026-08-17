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

from scripts.score_gold_recall import overlaps  # noqa: E402
from ui.components.db_status import render_locked_db_status  # noqa: E402

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
    "The full note stays visible on the left at every stage. Findings are "
    "highlighted directly in the text as they're computed -- colors are "
    "explained in the legend above it."
)


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


try:
    conn = get_conn()
except duckdb.IOException as exc:
    render_locked_db_status(exc)


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


with st.sidebar:
    st.header("Note selection")
    note_rows = conn.execute("""
        SELECT note_id, count(*) AS n FROM extracted_entities
        WHERE is_test = TRUE GROUP BY note_id ORDER BY n ASC
    """).fetchall()
    if not note_rows:
        st.warning("No processed notes found (is_test=TRUE).")
        st.stop()
    note_ids_sorted = [r[0] for r in note_rows]
    counts_by_note = dict(note_rows)
    note_id = st.selectbox(
        "Note (defaults to smallest)", note_ids_sorted, index=0,
        format_func=lambda n: f"{n}  ({counts_by_note[n]} entities)")
    st.caption(f"Selected: {counts_by_note[note_id]} extracted entities")

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
           n.is_ambiguous, e.orig_start, e.orig_end
    FROM extracted_entities e JOIN normalized_entities n ON n.entity_id = e.entity_id
    WHERE e.note_id = ? ORDER BY e.orig_start
""", [note_id]).fetchall()
# Keyed by (orig_start, orig_end) so the note highlighter can attach
# Stage 2b/SapBERT results to the exact same span GLiNER found -- an
# entity_id join would work too, but offsets are what the highlighter
# already indexes everything else by.
norm_by_span = {(r[5], r[6]): r for r in norm_rows}

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
        legend += "🟥 gold entity we missed entirely &nbsp; 🟨 found, but wrong SNOMED concept"
    else:
        legend += "<i>(run Step 3 to also see missed/wrong-concept gold spans here)</i>"
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
                _, tier, concept, sim, is_amb, _, _ = norm
                sim_str = f"{sim:.3f}" if sim is not None else "—"
                tip += (f"  |  SapBERT/Stage2b: tier={tier or '—'} sim={sim_str} "
                       f"-> {concept or 'Unmapped'}")
            spans.append((start, end, 5, "#a5d6a7", tip))

        if gold_report:
            for g in gold_report["missed"]:
                spans.append((g["start"], g["end"], 10, "#ef5350",
                              f"MISSED: gold={g['span']!r} ({g['concept_id']})"))
            for e in gold_report["wrong_concept"]:
                spans.append((e["start"], e["end"], 9, "#fff176",
                              f"WRONG CONCEPT: gold wanted {e['gold_concept_id']}, "
                              f"we predicted {e['predicted_concept']!r}"))
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
        st.caption("Missed spans highlight 🟥, wrong-concept spans highlight 🟨 "
                  "in the note once this runs.")
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

                    all_spans = [(r[3], r[4]) for r in entity_rows]
                    below_spans = [(r[3], r[4], r[2]) for r in below]
                    missed = [g for g in gold_rows
                             if not any(overlaps(s, e, g["start"], g["end"]) for s, e in all_spans)]
                    # wrong_concept_examples doesn't carry offsets by default;
                    # re-derive them from gold_rows by matching span+concept_id
                    # so the note highlighter can place them precisely.
                    wrong_with_offsets = []
                    for e in report["wrong_concept_examples"]:
                        match = next((g for g in gold_rows
                                     if g["span"] == e["gold_span"]
                                     and g["concept_id"] == e["gold_concept_id"]), None)
                        if match:
                            wrong_with_offsets.append({**e, "start": match["start"], "end": match["end"]})

                    st.session_state.gold_report_by_note[note_id] = {
                        "combined": report["combined"],
                        "missed": missed,
                        "wrong_concept": wrong_with_offsets,
                        "n_recoverable": sum(
                            1 for g in missed
                            if any(overlaps(b[0], b[1], g["start"], g["end"]) for b in below_spans)),
                    }
                    st.rerun()
        elif gold_report:
            c = gold_report["combined"]
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Gold annotations", c["gold_annotations"])
            rc2.metric("Span recall", f"{c['span_recall']*100:.1f}%")
            rc3.metric("Linked recall", f"{c['linked_recall']*100:.1f}%")

            below_spans = [(r[3], r[4], r[2]) for r in below]
            with st.expander(f"Missed spans ({len(gold_report['missed'])}) — with root cause",
                            expanded=True):
                for g in gold_report["missed"]:
                    hit = [b for b in below_spans if overlaps(b[0], b[1], g["start"], g["end"])]
                    reason = (f"🟡 GLiNER proposed it at confidence {hit[0][2]:.3f} (below threshold)"
                             if hit else "🔴 never proposed by GLiNER at any confidence")
                    st.markdown(f"**{g['span']!r}** ({g['concept_id']}) — {reason}")
                if gold_report["missed"]:
                    st.info(f"{gold_report['n_recoverable']}/{len(gold_report['missed'])} recoverable "
                           f"just by accepting below-threshold candidates.")
            with st.expander(f"Wrong-concept examples ({len(gold_report['wrong_concept'])})"):
                for e in gold_report["wrong_concept"]:
                    st.text(f"  gold {e['gold_span']!r} ({e['gold_concept_id']}) -> "
                           f"predicted {e['predicted_concept']!r} ({e['predicted_snomed_code']})")
        else:
            st.caption("Not run yet for this note.")

    with tab4:
        st.caption("Stage 2b: candidate retrieval + match tier for every entity.")
        with st.expander(f"All normalized entities ({len(norm_rows)})", expanded=True):
            for text, tier, concept, sim, is_amb, _start, _end in norm_rows:
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
