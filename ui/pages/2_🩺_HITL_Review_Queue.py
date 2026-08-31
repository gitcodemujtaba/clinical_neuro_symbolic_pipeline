"""ui/pages/2_🩺_HITL_Review_Queue.py — Stage 4 reviewer interface.

First real Streamlit page in this repo (ui/ was entirely empty stubs before
2026-08-14's Stage 4 build) -- establishes rather than follows a convention.

WHY EVERY CASE SHOWS "reviewed here because" TEXT EVEN FOR AUTO_VALIDATED
ONES. Every mollm_decisions/mollm_review_decisions row gets queued right
now regardless of its own routing tier (see src/hitl_queue.py's module
docstring for why: AUTO_VALIDATED precision measured 39.4% on 2026-08-14,
below what docs/Implementation_Checklist.md says is safe to auto-write to
KG3 unreviewed). A reviewer seeing an AUTO_VALIDATED case should understand
it is here because of that blanket policy, not because Stage 3 itself
flagged anything wrong with it -- the queue_reason field alone doesn't say
that, so the page adds it explicitly.

2026-08-20 REBUILD: full-note side-by-side view + per-model provenance.
Previously this page only showed `local_context` (one sentence-bounded
window, the same text the MoLLM prompt itself sees) -- enough to judge the
model's own reasoning, but not enough for a reviewer who needs the
surrounding paragraph/section to catch something the model's narrow window
couldn't see. Now shows the FULL raw note (same load_raw_text()/highlight
convention as ui/pages/3_🔍_Troubleshooting.py, kept independent rather
than cross-imported from another page module) with the entity highlighted
by its real orig_start/orig_end offset, alongside the full per-model
eval_trail (not just the final verdict+reasoning) so a reviewer can see
HOW each of the 3 models got there, not just what they concluded.
"""
import csv
import html
import os
import time

import duckdb
import streamlit as st

import sys
PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
sys.path.insert(0, PROJECT_DIR)

from src.hitl_queue import (  # noqa: E402
    ensure_hitl_queue_table, enqueue_pending_cases, load_hitl_queue, submit_review,
)
from ui.components.concept_search import format_concept_option, search_concepts  # noqa: E402
from ui.components.db_status import (  # noqa: E402
    render_locked_db_status, render_mixed_connection_status)
from ui.components.fresh10_notes import FRESH10_NOTE_IDS  # noqa: E402
from ui.components.hitl_context_aids import (  # noqa: E402
    ENTITY_LABEL_OPTIONS, agreement_summary, known_risk_flags)

# CNSP_DB_PATH override matches the OLLAMA_HOST/NEO4J_URI/MEMGRAPH_URI
# env-var pattern already used elsewhere in this codebase -- lets this page
# be pointed at a throwaway test DB (e.g. via streamlit.testing.v1.AppTest)
# without touching the production file, which is the only production write
# path this page has.
DB_PATH = os.environ.get("CNSP_DB_PATH", os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))

# Same fallback list as ui/pages/3_🔍_Troubleshooting.py -- data/raw_notes/
# {discharge,gold_notes}.csv only exist in the sibling (non-_reorder)
# worktree, not this one; this covers both layouts.
RAW_TEXT_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "raw_notes", "gold_notes.csv"),
    os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv"),
    os.path.join(PROJECT_DIR, "data", "snomed-ct-entity-linking-challenge-1.2.0", "train_notes.csv"),
]

st.set_page_config(page_title="HITL Review Queue", page_icon="🩺", layout="wide")
st.title("🩺 HITL Review Queue")


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


def render_note_with_highlight(raw_text: str, start, end, height: str = "70vh"):
    """One-span highlighter -- deliberately simpler than Troubleshooting's
    multi-span/multi-color version (this page only ever needs to show ONE
    entity's location, not a full cross-stage diff overlay)."""
    if start is None or end is None or not (0 <= start < end <= len(raw_text)):
        st.caption("⚠️ span offset unavailable for this case (queued before "
                  "2026-08-20) -- showing local context only, see below.")
        return
    before = html.escape(raw_text[:start])
    span_text = html.escape(raw_text[start:end])
    after = html.escape(raw_text[end:])
    body = (f"{before}<mark style='background:#ff9d3d;padding:1px 2px;"
           f"border-radius:2px;'>{span_text}</mark>{after}")
    st.markdown(
        f"<div style='white-space:pre-wrap; font-family:monospace; font-size:0.85em; "
        f"max-height:{height}; overflow-y:auto; border:1px solid rgba(128,128,128,0.3); "
        f"border-radius:4px; padding:10px;'>{body}</div>",
        unsafe_allow_html=True,
    )


# 2026-08-18: deliberately NOT @st.cache_resource -- a cached connection
# here is a WRITE connection held open for the server's entire lifetime,
# which would exclude EVERY other connection (reader or writer, this
# project's own or a background batch job's) for as long as the Streamlit
# server keeps running, not just while this page is actively being used.
# A fresh, uncached connection is released once superseded by the next
# script rerun, so the exclusive lock is only held for one page
# render/interaction at a time.
def get_conn():
    # read_only=False: this page writes reviewer decisions. DuckDB allows
    # many readers OR one writer -- if a batch job (scripts/run_stage3_batch.py,
    # src/mollm_review.py) holds the DB open for writing, this connect()
    # will fail or block until that job releases it. That is correct,
    # expected behavior, not a bug to work around.
    return duckdb.connect(DB_PATH, read_only=False)


try:
    conn = get_conn()
except duckdb.IOException as exc:
    render_locked_db_status(exc)
except duckdb.ConnectionException as exc:
    render_mixed_connection_status(exc)


# 2026-08-28: a real, confirmed-live bug -- this page never explicitly
# closed `conn` at ANY of its exit points (the one st.stop() below, or
# any of the three st.rerun() calls further down after a review
# submission or Previous/Next navigation), despite this exact page's own
# get_conn() docstring already flagging the general risk. A lingering
# read-write connection from here left open into a LATER script rerun
# (or a later page navigation within the same Streamlit server process)
# collided with a fresh read-only connect() elsewhere, surfacing as
# `duckdb.ConnectionException: Can't open a connection to same database
# file with a different configuration than existing connections` on
# ui/pages/3_🔍_Troubleshooting.py -- reproduced and root-caused live.
# _stop()/_rerun() below close the connection first, every time,
# mirroring the _stop() helper every other page in this app already
# uses for exactly this reason.
def _stop():
    conn.close()
    st.stop()


def _rerun():
    conn.close()
    st.rerun()


ensure_hitl_queue_table(conn)

with st.sidebar:
    st.header("Filters")
    if st.button("🔄 Pull new decisions into queue"):
        n = enqueue_pending_cases(conn, is_test=True)
        st.success(f"Queued {n} new case(s).")

    note_filter = st.text_input("note_id contains", value="")
    source_filter = st.selectbox(
        "source", ["(all)", "mollm_decisions", "mollm_review_decisions"], index=0
    )
    status_filter = st.selectbox(
        "status", ["PENDING", "APPROVED", "CORRECTED", "REJECTED", "(all)"], index=0
    )

queue = load_hitl_queue(
    conn,
    status=None if status_filter == "(all)" else status_filter,
    source_table=None if source_filter == "(all)" else source_filter,
)
# Scoped to the 10 fresh, held-out validation notes only (see
# ui/components/fresh10_notes.py) -- same convention as pages 1/3/4, so the
# demo consistently reflects the one validated population, not the full
# mixed-vintage corpus.
queue = [c for c in queue if c["note_id"] in FRESH10_NOTE_IDS]
if note_filter:
    queue = [c for c in queue if note_filter in (c["note_id"] or "")]

pending_count = len([c for c in queue if c["reviewer_decision"] == "PENDING"])
st.caption(f"{len(queue)} case(s) match filters — {pending_count} pending review.")

if not queue:
    st.info("No cases match the current filters. Click **Pull new decisions into queue** in the sidebar.")
    _stop()

# One case at a time, tracked by index in session_state so Approve/Correct/
# Reject can advance the reviewer to the next case without a page reload
# losing their place.
if "hitl_index" not in st.session_state:
    st.session_state.hitl_index = 0
if "hitl_case_started_at" not in st.session_state:
    st.session_state.hitl_case_started_at = time.time()

idx = min(st.session_state.hitl_index, len(queue) - 1)
case = queue[idx]
suggestion = case["presented_suggestion"] or {}

st.progress((idx + 1) / len(queue), text=f"Case {idx + 1} of {len(queue)}")

col_note, col_provenance, col_review = st.columns([2.2, 2.0, 1.3])

# ==========================================================================
# LEFT — full note, side by side, entity highlighted in its real position
# ==========================================================================
with col_note:
    st.subheader(suggestion.get("original_text") or "(no text)")
    st.caption(f"entity_id: {case['entity_id']}  |  note_id: {case['note_id']}  |  source: {case['source_table']}")

    meta_bits = [b for b in [
        f"section: {suggestion.get('section_name')}" if suggestion.get("section_name") else None,
        f"assertion: {suggestion.get('assertion_status')}" if suggestion.get("assertion_status") else None,
        f"experiencer: {suggestion.get('experiencer')}" if suggestion.get("experiencer") else None,
    ] if b]
    if meta_bits:
        st.caption(" | ".join(meta_bits))

    raw_text = load_raw_text(case["note_id"])
    if raw_text:
        st.markdown("**Full note** (entity highlighted):")
        render_note_with_highlight(raw_text, suggestion.get("orig_start"), suggestion.get("orig_end"))
    else:
        st.caption("⚠️ raw note text not found in data/raw_notes/ — showing local sentence context only.")

    # Kept as a fallback/supplement even when the full note IS shown -- this
    # is the EXACT window the MoLLM prompt itself saw (build_local_context()),
    # useful to compare directly against what a model reasoned over.
    local_context = suggestion.get("local_context")
    if local_context:
        entity_text = suggestion.get("original_text") or ""
        highlighted = local_context
        if entity_text and entity_text in local_context:
            highlighted = local_context.replace(entity_text, f"**:orange[{entity_text}]**", 1)
        with st.expander("Local context (exactly what the MoLLM prompt saw)", expanded=not raw_text):
            st.markdown(f"> {highlighted}")

    if case["reviewer_decision"] != "PENDING":
        st.info(
            f"Already reviewed: **{case['reviewer_decision']}**"
            + (f" — {case['rejection_reason']}" if case.get("rejection_reason") else "")
        )
        if case.get("reviewer_comment"):
            st.caption(f"💬 {case['reviewer_comment']}")
        if case.get("corrected_orig_start") is not None or case.get("corrected_entity_label"):
            st.caption(
                "✏️ Span/label corrected: "
                + (f"span → [{case.get('corrected_orig_start')}:{case.get('corrected_orig_end')}]  "
                  if case.get("corrected_orig_start") is not None else "")
                + (f"label → {case.get('corrected_entity_label')}"
                  if case.get("corrected_entity_label") else "")
            )
    else:
        st.warning(
            "Queued for human review regardless of this decision's own confidence tier "
            "(current policy: every decision is reviewed before KG3 write-back)."
        )

# ==========================================================================
# MIDDLE — provenance: routing, candidates, and EVERY model's full trail
# ==========================================================================
with col_provenance:
    st.markdown("### Provenance")
    routing = suggestion.get("routing_decision")
    tier = suggestion.get("confidence_tier_in")
    conf = suggestion.get("composite_confidence")
    st.markdown(f"**Routing decision:** `{routing}`  **Tier:** `{tier}`  **Confidence:** `{conf}`")
    if case["queue_reason"]:
        st.markdown(f"**Queue reason:** `{case['queue_reason']}`")

    # 2026-08-31: agreement-at-a-glance -- previously a reviewer had to
    # expand each of the (up to 3) per-model boxes below just to tell a
    # clean unanimous case from a genuine split. This is the same
    # plurality logic route_tier() itself uses (reused, not re-derived).
    models = suggestion.get("models") or []
    if models:
        agreement = agreement_summary(models)
        (st.success if agreement["unanimous"] else st.warning)(agreement["label"])

    # 2026-08-31: surfaces the SAME known-fragile patterns the production
    # gate itself already checks for (coronary-segment trap, short-
    # alphanumeric-code trap) plus a real domain-conflict signal this
    # project measured repeatedly -- tells a reviewer WHERE to look
    # harder, before they've read anything else on the page.
    candidates = suggestion.get("candidates") or []
    risk_flags = known_risk_flags(
        suggestion.get("original_text"), candidates, suggestion.get("suggested_omop_concept_id"))
    for flag in risk_flags:
        st.warning(flag)

    routing_basis = suggestion.get("routing_basis")
    if routing_basis:
        st.markdown("**How the pipeline reached this conclusion:**")
        st.info(routing_basis)

    if candidates:
        st.markdown("**Candidates (Stage 2b retrieval):**")
        # 2026-08-31: two real context aids per candidate, both reusing
        # already-built project functions rather than new logic --
        # (a) prior_confirmation_count: how many times THIS EXACT (mention
        # text, concept) pairing already cleared an auto-tier or a human
        # review before (src.mollm_tier_calibrator.count_prior_
        # confirmations() -- the same DuckDB-sourced signal the calibrator
        # itself trains on, just never surfaced to a human reviewer
        # before); (b) domain/parent context (src.retrieval.
        # VocabularyRetriever.concept_context() -- Channel C's own
        # ontology-grounding lookup, reused here for a reviewer instead of
        # a prompt). One extra query pair per candidate -- kept in an
        # expander, not inline, so it doesn't visually clutter the
        # already-dense candidate list; only fetched when opened.
        from src.mollm_tier_calibrator import count_prior_confirmations
        from src.retrieval import VocabularyRetriever
        vocab = VocabularyRetriever(conn)
        entity_text = suggestion.get("original_text")
        for i, c in enumerate(candidates, 1):
            st.text(
                f"[{i}] {c.get('concept_name')}  "
                f"(OMOP {c.get('omop_concept_id')}, domain {c.get('domain_id')}, "
                f"tier {c.get('match_tier')}, score {c.get('similarity_score')}, "
                f"basis {c.get('match_basis')})"
            )
            with st.expander(f"  ↳ context for [{i}]", expanded=False):
                cid_i = c.get("omop_concept_id")
                prior_count = count_prior_confirmations(conn, entity_text, cid_i)
                st.caption(f"Prior confirmations of this exact (mention, concept) pairing: "
                          f"**{prior_count}**" + (" — never seen before" if prior_count == 0 else ""))
                ctx = vocab.concept_context(cid_i)
                if ctx:
                    st.caption(f"FSN: {ctx.get('fsn')}  |  {ctx.get('domain_id')}/"
                              f"{ctx.get('concept_class_id')} ({ctx.get('vocabulary_id')})")
                    parents = ctx.get("parents") or []
                    if parents:
                        st.caption("Parent concept(s): " + ", ".join(p["name"] for p in parents))
    proposed_name = suggestion.get("proposed_concept_name")
    if proposed_name:
        st.markdown(f"**Proposed concept:** {proposed_name}")

    # 2026-08-20: full per-model trail, not just the final verdict. Each
    # model's eval_trail is Step B's own sequential candidate-by-candidate
    # record (src.mollm_tier_gate._evaluate_one_model()) -- clinical_meaning
    # (Step A, no candidate list visible) then one accept/reject judgment
    # per candidate in order, plus a tiebreak entry if EXHAUSTIVE_CANDIDATE_
    # EVAL_ENABLED triggered one. Showing this (not just "model X -> verdict
    # Y") is what actually lets a reviewer catch a model that reasoned
    # correctly then contradicted its own reasoning in the final verdict --
    # a real failure mode this project found and documented this session.
    models = suggestion.get("models") or []
    if models:
        st.markdown(f"**MoLLM decisions ({len(models)} model(s)):**")
        for m in models:
            verdict = m.get("verdict") or m.get("assessment")
            meaning = m.get("clinical_meaning")
            reasoning = m.get("reasoning") or m.get("reasoning_text") or ""
            conf_m = m.get("logprob_confidence")
            with st.expander(f"{m.get('model', 'model')} → {verdict}  (conf {conf_m})"):
                if meaning:
                    st.markdown(f"**Step A — clinical meaning (no candidates visible):** {meaning}")
                trail = m.get("eval_trail") or []
                if trail:
                    st.markdown("**Step B — per-candidate evaluation, in order:**")
                    for t in trail:
                        if t.get("tiebreak"):
                            st.text(f"  [tiebreak] considered {t.get('candidates_considered')} "
                                   f"-> chose [{t.get('chosen_index')}]: {t.get('reasoning', '')}")
                            continue
                        if "error" in t:
                            st.text(f"  [{t.get('candidate_index')}] ERROR: {t['error']}")
                            continue
                        mark = "✅" if t.get("match") else "❌"
                        st.text(f"  {mark} [{t.get('candidate_index')}] {t.get('concept_name')}")
                        if t.get("reasoning"):
                            st.caption(f"     {t['reasoning']}")
                else:
                    st.write(reasoning)

with col_review:
    st.markdown("### Review")
    # 2026-08-31: no longer an st.form(). The live concept-search box below
    # is a plain st.text_input -- INSIDE an st.form, widget interactions
    # (including every keystroke of a text_input) don't trigger a script
    # rerun until the form's own submit button is pressed, so search
    # results would never appear while the reviewer is actually typing.
    # Every widget here gets an explicit case-id-scoped `key=` instead of
    # relying on the form's own key to reset state per case (verified live
    # in tests/test_hitl_review_queue_page.py: navigating between
    # cases with a stale key would otherwise leak a decision/selection from
    # the PREVIOUS case).
    cid = case["hitl_case_id"]
    decision = st.radio("Decision", ["APPROVED", "CORRECTED", "REJECTED"], key=f"decision_{cid}")
    corrected_id = None
    if decision == "CORRECTED":
        options = {f"[{i}] {c.get('concept_name')}": c.get("omop_concept_id")
                  for i, c in enumerate(candidates, 1)}
        choice = st.selectbox("Correct concept (from Stage 2b's own retrieved candidates)",
                              list(options.keys()) or ["(none available)"], key=f"cand_choice_{cid}")
        corrected_id = options.get(choice)

        # 2026-08-31: the real gap this closes -- previously the ONLY way to
        # correct to a concept Stage 2b's own retrieval never surfaced at
        # all (a genuine retrieval miss, not just a mis-ranking) was to
        # already know its exact numeric OMOP concept_id and type it in
        # blind. A reviewer almost never does. This searches athena_concept
        # by name live, the same population Stage 2b itself searches, so
        # "the retrieval missed it" cases become actually fixable, not just
        # flaggable.
        st.markdown("**...or search for a different concept:**")
        search_query = st.text_input(
            "Search by name (e.g. 'chest pain')", key=f"concept_search_{cid}")
        if search_query:
            search_results = search_concepts(conn, search_query)
            if not search_results:
                st.caption("No matches (need 3+ characters, or nothing found).")
            else:
                search_options = {format_concept_option(c): c["concept_id"] for c in search_results}
                search_choice = st.selectbox(
                    f"{len(search_results)} match(es)", list(search_options.keys()),
                    key=f"concept_search_choice_{cid}")
                corrected_id = search_options[search_choice]

        manual_id = st.text_input("...or enter an OMOP concept_id directly", key=f"manual_id_{cid}")
        if manual_id.strip():
            try:
                corrected_id = int(manual_id.strip())
            except ValueError:
                st.error("concept_id must be an integer")
    rejection_reason = None
    if decision == "REJECTED":
        rejection_reason = st.text_area("Rejection reason", key=f"rejection_reason_{cid}")

    # 2026-08-31: span/label correction -- a genuinely different axis from
    # corrected_concept_id above. GLiNER's own span boundary or entity
    # label can be wrong independent of whether the right CONCEPT was
    # ultimately picked (e.g. a compound span that should have been split,
    # or "Symptom" when it should have been "Condition"). Available
    # regardless of decision (APPROVED/CORRECTED alike) -- a case can be
    # concept-correct but still have a span/label worth fixing. Collapsed
    # by default so the common "everything's fine, just click a button"
    # case isn't cluttered by it.
    orig_start = suggestion.get("orig_start")
    orig_end = suggestion.get("orig_end")
    entity_label = suggestion.get("entity_label")
    corrected_orig_start = corrected_orig_end = corrected_entity_label = None
    with st.expander("✏️ Adjust span or entity label (optional)", expanded=False):
        if orig_start is None or orig_end is None:
            st.caption("No span offset on this case (queued before 2026-08-20) — nothing to adjust.")
        else:
            sc1, sc2 = st.columns(2)
            new_start = sc1.number_input("Span start", min_value=0, value=int(orig_start),
                                         key=f"span_start_{cid}")
            new_end = sc2.number_input("Span end", min_value=0, value=int(orig_end),
                                       key=f"span_end_{cid}")
            if new_start != orig_start or new_end != orig_end:
                if new_start < new_end and raw_text and 0 <= new_start < new_end <= len(raw_text):
                    st.caption("Preview of the adjusted span:")
                    render_note_with_highlight(raw_text, int(new_start), int(new_end), height="15vh")
                    corrected_orig_start, corrected_orig_end = int(new_start), int(new_end)
                else:
                    st.error("Adjusted span must have start < end and fit within the note.")

        if entity_label:
            label_options = ([entity_label] if entity_label not in ENTITY_LABEL_OPTIONS
                            else []) + ENTITY_LABEL_OPTIONS
            new_label = st.selectbox("Entity label", label_options,
                                     index=label_options.index(entity_label),
                                     key=f"entity_label_{cid}")
            if new_label != entity_label:
                corrected_entity_label = new_label

    # 2026-08-17: independent of rejection_reason -- available on every
    # decision, not just REJECTED. This is the real ground truth
    # src.abbreviation_flywheel.mine_context_rules() (and any future
    # pipeline-improvement analysis) reads back from hitl_review_queue;
    # a reviewer explaining WHY, not just WHAT, is what actually
    # accumulates into something the pipeline can learn from later.
    comment = st.text_area(
        "Comments (optional) — notes for future pipeline improvement",
        placeholder="e.g. 'context clearly says X, a rule for this "
                    "abbreviation would have caught it' or 'candidate "
                    "#2 was closer but still not quite right'",
        key=f"comment_{cid}",
    )

    if st.button("Submit", key=f"submit_{cid}"):
        duration = time.time() - st.session_state.hitl_case_started_at
        submit_review(
            conn, case["hitl_case_id"], decision,
            corrected_concept_id=corrected_id,
            rejection_reason=rejection_reason,
            review_duration=round(duration, 1),
            reviewer_comment=comment.strip() or None,
            corrected_orig_start=corrected_orig_start,
            corrected_orig_end=corrected_orig_end,
            corrected_entity_label=corrected_entity_label,
        )
        st.session_state.hitl_index = min(idx + 1, len(queue) - 1)
        st.session_state.hitl_case_started_at = time.time()
        _rerun()

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("← Previous", disabled=idx == 0):
            st.session_state.hitl_index = idx - 1
            st.session_state.hitl_case_started_at = time.time()
            _rerun()
    with nav2:
        if st.button("Next →", disabled=idx >= len(queue) - 1):
            st.session_state.hitl_index = idx + 1
            st.session_state.hitl_case_started_at = time.time()
            _rerun()

# Natural fall-through (no button clicked this run, no stop/rerun
# triggered) -- same explicit-close discipline as every exit point
# above, not just the ones that terminate the script early.
conn.close()
