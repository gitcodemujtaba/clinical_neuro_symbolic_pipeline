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
"""
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

# CNSP_DB_PATH override matches the OLLAMA_HOST/NEO4J_URI/MEMGRAPH_URI
# env-var pattern already used elsewhere in this codebase -- lets this page
# be pointed at a throwaway test DB (e.g. via streamlit.testing.v1.AppTest)
# without touching the production file, which is the only production write
# path this page has.
DB_PATH = os.environ.get("CNSP_DB_PATH", os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))

st.set_page_config(page_title="HITL Review Queue", page_icon="🩺", layout="wide")
st.title("🩺 HITL Review Queue")


@st.cache_resource
def get_conn():
    # read_only=False: this page writes reviewer decisions. DuckDB allows
    # many readers OR one writer -- if a batch job (scripts/run_stage3_batch.py,
    # src/mollm_review.py) holds the DB open for writing, this connect()
    # will fail or block until that job releases it. That is correct,
    # expected behavior, not a bug to work around.
    return duckdb.connect(DB_PATH, read_only=False)


conn = get_conn()
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
if note_filter:
    queue = [c for c in queue if note_filter in (c["note_id"] or "")]

pending_count = len([c for c in queue if c["reviewer_decision"] == "PENDING"])
st.caption(f"{len(queue)} case(s) match filters — {pending_count} pending review.")

if not queue:
    st.info("No cases match the current filters. Click **Pull new decisions into queue** in the sidebar.")
    st.stop()

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

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(suggestion.get("original_text") or "(no text)")
    st.caption(f"entity_id: {case['entity_id']}  |  note_id: {case['note_id']}  |  source: {case['source_table']}")

    # 2026-08-15 (reviewer feedback): the queue was showing only model
    # verdicts/reasoning with no surrounding note text at all -- a reviewer
    # cannot actually judge "MD" -> muscular dystrophy vs. Doctor of
    # Medicine, say, without seeing the sentence it came from. local_context
    # is the same sentence-bounded window Stage 3's own prompt shows the
    # models (src/entity_extraction.py's build_local_context()), so the
    # reviewer sees exactly what the models saw, not less.
    local_context = suggestion.get("local_context")
    if local_context:
        entity_text = suggestion.get("original_text") or ""
        highlighted = local_context
        if entity_text and entity_text in local_context:
            highlighted = local_context.replace(entity_text, f"**:orange[{entity_text}]**", 1)
        meta_bits = [b for b in [
            f"section: {suggestion.get('section_name')}" if suggestion.get("section_name") else None,
            f"assertion: {suggestion.get('assertion_status')}" if suggestion.get("assertion_status") else None,
            f"experiencer: {suggestion.get('experiencer')}" if suggestion.get("experiencer") else None,
        ] if b]
        if meta_bits:
            st.caption(" | ".join(meta_bits))
        st.markdown(f"> {highlighted}")
    else:
        st.caption("⚠️ no note context available for this case")

    if case["reviewer_decision"] != "PENDING":
        st.info(
            f"Already reviewed: **{case['reviewer_decision']}**"
            + (f" — {case['rejection_reason']}" if case.get("rejection_reason") else "")
        )
        if case.get("reviewer_comment"):
            st.caption(f"💬 {case['reviewer_comment']}")
    else:
        st.warning(
            "Queued for human review regardless of this decision's own confidence tier "
            "(current policy: every decision is reviewed before KG3 write-back)."
        )

    routing = suggestion.get("routing_decision")
    tier = suggestion.get("confidence_tier_in")
    conf = suggestion.get("composite_confidence")
    st.markdown(f"**Routing decision:** `{routing}`  **Tier:** `{tier}`  **Confidence:** `{conf}`")
    if case["queue_reason"]:
        st.markdown(f"**Queue reason:** `{case['queue_reason']}`")

    # 2026-08-17: route_tier()'s own plain-language explanation -- "how did
    # the pipeline reach this conclusion", the thing a reviewer previously
    # had to reconstruct by reading raw model verdicts. Only present for
    # mollm_tier_gate_decisions-sourced cases (the current production gate);
    # absent for the two older sources, which is expected, not a bug.
    routing_basis = suggestion.get("routing_basis")
    if routing_basis:
        st.markdown("**How the pipeline reached this conclusion:**")
        st.info(routing_basis)

    candidates = suggestion.get("candidates") or []
    if candidates:
        st.markdown("**Candidates:**")
        for i, c in enumerate(candidates, 1):
            st.text(
                f"[{i}] {c.get('concept_name')}  "
                f"(OMOP {c.get('omop_concept_id')}, tier {c.get('match_tier')}, "
                f"score {c.get('similarity_score')}, basis {c.get('match_basis')})"
            )
    proposed_name = suggestion.get("proposed_concept_name")
    if proposed_name:
        st.markdown(f"**Proposed concept:** {proposed_name}")

    models = suggestion.get("models") or []
    if models:
        st.markdown("**Model verdicts:**")
        for m in models:
            verdict = m.get("verdict") or m.get("assessment")
            reasoning = m.get("reasoning") or m.get("reasoning_text") or ""
            with st.expander(f"{m.get('model', 'model')} → {verdict}"):
                st.write(reasoning)

with col2:
    st.markdown("### Review")
    with st.form(key=f"review_form_{case['hitl_case_id']}"):
        decision = st.radio("Decision", ["APPROVED", "CORRECTED", "REJECTED"])
        corrected_id = None
        if decision == "CORRECTED":
            options = {f"[{i}] {c.get('concept_name')}": c.get("omop_concept_id")
                      for i, c in enumerate(candidates, 1)}
            choice = st.selectbox("Correct concept", list(options.keys()) or ["(none available)"])
            corrected_id = options.get(choice)
            manual_id = st.text_input("...or enter an OMOP concept_id directly")
            if manual_id.strip():
                try:
                    corrected_id = int(manual_id.strip())
                except ValueError:
                    st.error("concept_id must be an integer")
        rejection_reason = None
        if decision == "REJECTED":
            rejection_reason = st.text_area("Rejection reason")

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
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            duration = time.time() - st.session_state.hitl_case_started_at
            submit_review(
                conn, case["hitl_case_id"], decision,
                corrected_concept_id=corrected_id,
                rejection_reason=rejection_reason,
                review_duration=round(duration, 1),
                reviewer_comment=comment.strip() or None,
            )
            st.session_state.hitl_index = min(idx + 1, len(queue) - 1)
            st.session_state.hitl_case_started_at = time.time()
            st.rerun()

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("← Previous", disabled=idx == 0):
            st.session_state.hitl_index = idx - 1
            st.session_state.hitl_case_started_at = time.time()
            st.rerun()
    with nav2:
        if st.button("Next →", disabled=idx >= len(queue) - 1):
            st.session_state.hitl_index = idx + 1
            st.session_state.hitl_case_started_at = time.time()
            st.rerun()
