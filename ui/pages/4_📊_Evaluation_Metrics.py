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

from ui.components.db_status import render_locked_db_status  # noqa: E402

DB_PATH = os.environ.get("CNSP_DB_PATH", os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))

st.set_page_config(page_title="Evaluation Metrics", page_icon="📊", layout="wide")
st.title("📊 Evaluation Metrics")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


try:
    conn = get_conn()
except duckdb.IOException as exc:
    render_locked_db_status(exc)

with st.sidebar:
    st.header("Selection")
    all_note_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE ORDER BY note_id"
    ).fetchall()]
    if not all_note_ids:
        st.warning("No processed notes found (is_test=TRUE).")
        st.stop()
    note_ids = st.multiselect("Notes to include", all_note_ids, default=all_note_ids)
    if not note_ids:
        st.info("Select at least one note.")
        st.stop()

tab_tiers, tab_precision, tab_recall, tab_calibrator = st.tabs(
    ["Tier distribution", "Precision vs. gold", "Recall / completeness", "Calibrator status"])

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
# TAB 4 — calibrator status (static .pkl metadata, no live fit/scoring here)
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
