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
from ui.components.fresh10_notes import FRESH10_NOTE_IDS  # noqa: E402

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


def _stop():
    """st.stop(), but closing `conn` first -- see
    ui/pages/1_🚀_Pipeline_Runner.py's identical helper for why an explicit
    close (not just letting `conn` go out of scope) is needed."""
    conn.close()
    st.stop()


with st.sidebar:
    st.header("Selection")
    # extracted_entities itself carries no is_stale/provenance columns
    # (only normalized_entities and mollm_tier_gate_decisions do) -- joined
    # here rather than filtered directly. is_stale=FALSE means "processed by
    # current code" -- see the 2026-08-18 migration that added this column
    # (docs in scripts/mark_notes_stale.py) rather than the earlier,
    # fragile hand-maintained-timestamp approach this replaced.
    note_ph = ",".join("?" * len(FRESH10_NOTE_IDS))
    all_note_ids = [r[0] for r in conn.execute(f"""
        SELECT DISTINCT e.note_id FROM extracted_entities e
        WHERE e.is_test = TRUE AND e.note_id IN ({note_ph}) AND e.note_id IN (
            SELECT DISTINCT note_id FROM normalized_entities
            WHERE is_test = TRUE AND is_stale = FALSE
        ) ORDER BY e.note_id
    """, FRESH10_NOTE_IDS).fetchall()]
    if not all_note_ids:
        st.warning("None of the 10 fresh-validation notes are ready (is_stale = FALSE) yet. "
                  "See scripts/run_fresh5_final_validation.py.")
        _stop()
    note_ids = st.multiselect("Notes to include", all_note_ids, default=all_note_ids)
    if not note_ids:
        st.info("Select at least one note.")
        _stop()

tab_overall, tab_tiers, tab_precision, tab_recall, tab_calibration, tab_calibrator = st.tabs(
    ["Overall", "Tier distribution", "Precision vs. gold", "Recall / completeness",
     "ECE & IoU (per stage)", "Calibrator status"])

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
        from evaluation.tier_gate_grading import grade_by_tier
        from evaluation import iou_metrics
        from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
        from scripts.score_gold_recall import (
            attach_snomed_codes, load_gold, load_predictions, overlaps, score,
        )
        from src.mollm_tier_gate import AUTO_TIERS
        from src.retrieval import VocabularyRetriever
        import collections as _collections

        with st.spinner("Grading every stage against gold..."):
            gold_path = _first_existing(GOLD_CANDIDATES, "gold")
            gold_rows = load_gold(gold_path, note_ids)
            if not gold_rows:
                st.warning("No gold annotations found for the selected note(s).")
            else:
                gold_by_note = _collections.defaultdict(list)
                for g in gold_rows:
                    gold_by_note[g["note_id"]].append(g)
                vocab = VocabularyRetriever(conn)

                # Stage 1/2 completeness: span recall / linked recall
                predictions = load_predictions(conn, note_ids)
                attach_snomed_codes(conn, predictions)
                recall_report = score(gold_rows, predictions)["combined"]

                # Linked PRECISION -- same population as linked recall (every
                # resolved prediction, all tiers), NOT AUTO-tier-only. Mixing
                # AUTO-tier precision with full-population recall would score
                # two different populations and make F1 meaningless -- see
                # docs/2026-08-20_Session_Results_And_Status.md §15.
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

                # Stage 3: full tier distribution (ALL tiers, not grade_by_tier's
                # AUTO_TIERS+TIER_4-only default) -- needed for a correct
                # deflection rate. A real bug (gradable-restricted AUTO count
                # divided by an unrestricted/tier-limited total) was caught and
                # fixed in this exact computation -- see §15's "Methodology" note.
                note_ph = ",".join("?" * len(note_ids))
                all_tier_rows = conn.execute(
                    f"SELECT tier, COUNT(*) FROM mollm_tier_gate_decisions "
                    f"WHERE note_id IN ({note_ph}) GROUP BY tier", note_ids).fetchall()
                total_decisions = sum(n for _, n in all_tier_rows)
                auto_all = sum(n for t, n in all_tier_rows if t in AUTO_TIERS)
                deflection_rate = auto_all / total_decisions if total_decisions else None

                # AUTO-tier PRECISION still needs the gradable (clean-span)
                # restriction -- that part was never the bug, precision can only
                # be checked on decisions we can actually grade against gold.
                tier_report = grade_by_tier(conn, note_ids)
                auto_n = sum(r["clean"]["n"] for t, r in tier_report.items() if t in AUTO_TIERS)
                auto_correct = sum(r["clean"]["n_correct"] for t, r in tier_report.items() if t in AUTO_TIERS)

                # Stage 2b benchmark char IoU (DrivenData definition)
                bench = iou_metrics.benchmark_char_iou(conn, note_ids, gold_by_note, vocab)

            if gold_rows:
                st.markdown("#### Completeness (Stage 1/2 — of everything gold has, how much did we find?)")
                oc1, oc2, oc3 = st.columns(3)
                oc1.metric("Gold annotations", recall_report["gold_annotations"])
                oc2.metric("Span recall", f"{recall_report['span_recall']*100:.1f}%")
                oc3.metric("Linked recall", f"{recall_report['linked_recall']*100:.1f}%")

                st.markdown("#### Linked concept-level Precision / Recall / F1 (same population, all tiers)")
                of1, of2, of3 = st.columns(3)
                of1.metric("Linked precision", f"{linked_precision*100:.1f}%" if linked_precision is not None else "n/a",
                          help=f"{n_pred_correct}/{n_pred_with_concept} of our own resolved links (any tier) match gold")
                of2.metric("Linked recall", f"{linked_recall*100:.1f}%")
                of3.metric("Linked F1", f"{linked_f1*100:.1f}%" if linked_f1 is not None else "n/a")

                st.markdown("#### Stage 3 gate — deflection rate & AUTO-tier precision")
                oc4, oc5, oc6 = st.columns(3)
                oc4.metric("Total Stage 3 decisions", total_decisions)
                oc5.metric("Deflection rate", f"{auto_all}/{total_decisions}",
                          f"{deflection_rate*100:.1f}%" if deflection_rate is not None else "—",
                          help="Fraction of ALL Stage 3 decisions that auto-wrote with no human "
                               "review (every AUTO_TIERS decision, not just the gradable subset).")
                oc6.metric("AUTO-tier precision", f"{auto_correct}/{auto_n}" if auto_n else "n/a",
                          f"{auto_correct/auto_n*100:.1f}%" if auto_n else None,
                          help="Of AUTO-written decisions we can grade against gold (clean single "
                               "span overlap), how often did we pick gold's exact SNOMED concept?")

                st.markdown("#### Benchmark metric (Stage 2b — DrivenData's own char-level IoU)")
                oc7, oc8 = st.columns(2)
                oc7.metric("Macro char IoU", bench["macro_char_iou"])
                oc8.metric("Support-weighted char IoU", bench["weighted_char_iou"])
    else:
        st.info("Click the button above to grade the current note selection across every stage.")

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
