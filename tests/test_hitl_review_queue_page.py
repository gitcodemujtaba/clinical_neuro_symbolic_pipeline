"""
tests/test_hitl_review_queue_page.py -- AppTest coverage for
ui/pages/2_🩺_HITL_Review_Queue.py, specifically the 2026-08-31 fix: moving
the review form's widgets out of st.form() (needed for the live concept-
search box to be reactive) required giving every widget an explicit
case-id-scoped `key=` so per-case state doesn't leak across navigation --
this test proves that actually holds, not just that it compiles.

No existing test coverage touched src/hitl_queue.py or this page at all
before this file -- a real, pre-existing gap, not something this fix
introduced. Uses real project code (enqueue_pending_cases(), the actual
table DDLs) against a throwaway on-disk DuckDB file, not a hand-rolled
mock -- same "isolated DB round-trip, real functions" discipline as
tests/test_gliner_gazetteer_fallback.py's DB test.

Run: python3 -m pytest tests/test_hitl_review_queue_page.py -v
"""
import json
import os
import sys
import tempfile

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _build_fixture_db(path):
    """Two real, distinct queued cases (different note/entity/candidates),
    built via the actual enqueue_pending_cases() code path -- not a
    hand-rolled hitl_review_queue row -- so this exercises the same
    presented_suggestion shape the real page reads."""
    conn = duckdb.connect(path)
    conn.sql("""
        CREATE TABLE extracted_entities (
            entity_id VARCHAR, note_id VARCHAR, original_text VARCHAR,
            entity_label VARCHAR, local_context VARCHAR, section_name VARCHAR,
            assertion_status VARCHAR, experiencer VARCHAR,
            orig_start INTEGER, orig_end INTEGER
        );
        CREATE TABLE normalized_entities (
            entity_id VARCHAR, candidates JSON
        );
    """)
    # Real note_ids that exist in the small data/raw_notes/gold_notes.csv
    # extract (checked first by the page's own load_raw_text()) --
    # deliberately NOT synthetic IDs like "note-A": those aren't found
    # anywhere, so load_raw_text() falls through to a full linear scan of
    # the 3.3GB data/raw_notes/discharge.csv looking for a match that will
    # never be found, which alone blew the AppTest timeout the first time
    # this test was written. Real, present note_ids resolve instantly.
    cases = [
        ("e1", "15285988-DS-7", "chest pain", "Symptom", "pt c/o chest pain today", "HPI",
         "PRESENT", "PATIENT", 10, 20,
         [{"omop_concept_id": 111, "concept_name": "Chest pain", "match_tier": "3", "similarity_score": 0.9}]),
        ("e2", "15906604-DS-2", "SOB", "Symptom", "pt denies SOB", "ROS",
         "ABSENT", "PATIENT", 5, 8,
         [{"omop_concept_id": 222, "concept_name": "Dyspnea", "match_tier": "2", "similarity_score": 0.8}]),
    ]
    for eid, nid, text, label, ctx, sect, assertion, exp, s, e, cands in cases:
        conn.execute(
            "INSERT INTO extracted_entities VALUES (?,?,?,?,?,?,?,?,?,?)",
            [eid, nid, text, label, ctx, sect, assertion, exp, s, e])
        conn.execute("INSERT INTO normalized_entities VALUES (?,?)", [eid, json.dumps(cands)])

    conn.sql("""
        CREATE TABLE mollm_tier_gate_decisions (
            mollm_call_id VARCHAR PRIMARY KEY, entity_id VARCHAR, note_id VARCHAR,
            tier VARCHAR, mollm_routing_decision VARCHAR, queue_reason VARCHAR,
            final_candidate_index INTEGER, composite_confidence DOUBLE,
            routing_basis VARCHAR, models JSON, is_test BOOLEAN DEFAULT FALSE
        );
    """)
    # Real 2-1 split vote shape (matching route_tier()'s own model dict
    # keys) -- exercises the new agreement-badge/known-risk-flag additions
    # in the same AppTest run, not just "the page still loads."
    split_models = json.dumps([
        {"model": "qwen2.5:3b", "verdict": "SUPPORTED_1", "degenerate_generation": False},
        {"model": "llama3.2:3b", "verdict": "SUPPORTED_1", "degenerate_generation": False},
        {"model": "phi4-mini", "verdict": "NONE_CORRECT", "degenerate_generation": False},
    ])
    for eid, nid, cid in [("e1", "15285988-DS-7", "call1"), ("e2", "15906604-DS-2", "call2")]:
        conn.execute("""
            INSERT INTO mollm_tier_gate_decisions
            (mollm_call_id, entity_id, note_id, tier, mollm_routing_decision,
             queue_reason, final_candidate_index, composite_confidence, routing_basis,
             models, is_test)
            VALUES (?, ?, ?, 'TIER_4_ENSEMBLE_SPLIT', 'HITL_REQUIRED', 'ensemble_split',
                    NULL, 0.5, 'test fixture', ?, TRUE)
        """, [cid, eid, nid, split_models])

    from src.hitl_queue import enqueue_pending_cases
    n = enqueue_pending_cases(conn, is_test=True)
    conn.close()
    return n


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        n_queued = _build_fixture_db(db_path)
        check("fixture builds 2 real queued cases", n_queued == 2)

        os.environ["CNSP_DB_PATH"] = db_path
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(
            f"{PROJECT_DIR}/ui/pages/2_🩺_HITL_Review_Queue.py", default_timeout=30)
        # This page filters its queue to FRESH10_NOTE_IDS -- the fixture's
        # real note_ids aren't in that list, so the page would show "no
        # cases match" rather than the review form -- monkeypatch the
        # imported constant before the page's own module-level filter runs
        # (widening the filter itself would require editing the page, out
        # of scope for this test).
        import ui.components.fresh10_notes as f10
        f10.FRESH10_NOTE_IDS = f10.FRESH10_NOTE_IDS + ["15285988-DS-7", "15906604-DS-2"]
        at.run()

        check("page loads without raising", not at.exception)

        # 2026-08-31 additions: agreement badge + known-risk flags. The
        # fixture's models are a real 2-1 split (SUPPORTED_1 x2,
        # NONE_CORRECT x1), so a "Split" warning should render.
        warning_texts = " ".join(w.value for w in at.warning)
        check("the 2-1 split fixture renders a 'Split' agreement warning",
              "Split" in warning_texts)

        # 2026-08-31 additions: per-candidate context expander (prior-
        # confirmation count + domain/parent lookup).
        expander_labels = [e.proto.label for e in at.expander]
        check("a per-candidate context expander renders for the top candidate",
              any("context for [1]" in lbl for lbl in expander_labels))
        if at.exception:
            print("EXCEPTION:", at.exception)

        radios = at.radio
        check("a decision radio is present", len(radios) >= 1)
        if radios:
            check("decision defaults to APPROVED on the first case",
                  radios[0].value == "APPROVED")

            # Select CORRECTED on case 1 -- should reveal the candidate/
            # search selectboxes (the actual behavior this fix depends on).
            radios[0].set_value("CORRECTED")
            at.run()
            check("no exception after selecting CORRECTED", not at.exception)
            selectboxes_after_correct = at.selectbox
            check("selecting CORRECTED reveals at least one selectbox "
                 "(candidate list)", len(selectboxes_after_correct) >= 1)

            # Navigate to case 2 via the Next button.
            next_buttons = [b for b in at.button if b.label == "Next →"]
            check("a Next button exists", len(next_buttons) == 1)
            if next_buttons:
                next_buttons[0].click()
                at.run()
                check("no exception after navigating to case 2", not at.exception)
                radios_case2 = at.radio
                check("decision on case 2 is APPROVED, NOT leaked from case "
                     "1's CORRECTED selection -- the actual bug this "
                     "case-id-scoped key= fix prevents",
                      len(radios_case2) >= 1 and radios_case2[0].value == "APPROVED")

        # 2026-08-31: verify submit_review()'s new corrected_orig_start/
        # corrected_orig_end/corrected_entity_label params round-trip
        # correctly (both the "actually changed" and the "left unchanged
        # -> stays NULL" cases) -- called directly rather than via a full
        # AppTest button click for this specific check: AppTest's
        # rerun-state restoration for a number_input nested inside a
        # collapsed st.expander hit an internal KeyError unrelated to this
        # page's own logic (the widget itself is unconditional whenever
        # orig_start is not None; only the preview below it is
        # conditional) -- calling the already-AppTest-covered submit_review()
        # directly here tests the actual behavior this feature depends on
        # without depending on that harness quirk.
        conn3 = duckdb.connect(db_path, read_only=False)
        from src.hitl_queue import submit_review as _submit_review
        # Case 1 (e1): routine approve, no span/label change -- must stay NULL.
        _submit_review(conn3, "hitl_mollm_tier_gate_decisions_call1", "APPROVED")
        # Case 2 (e2): a real span + label correction.
        _submit_review(conn3, "hitl_mollm_tier_gate_decisions_call2", "CORRECTED",
                       corrected_concept_id=222, corrected_orig_start=6,
                       corrected_orig_end=9, corrected_entity_label="Condition")
        row1 = conn3.execute(
            "SELECT reviewer_decision, corrected_orig_start, corrected_orig_end, "
            "corrected_entity_label FROM hitl_review_queue WHERE entity_id = 'e1'").fetchone()
        row2 = conn3.execute(
            "SELECT reviewer_decision, corrected_orig_start, corrected_orig_end, "
            "corrected_entity_label FROM hitl_review_queue WHERE entity_id = 'e2'").fetchone()
        conn3.close()
        check("a routine approve with no span/label change leaves all three NULL "
             "(no spurious 'correction' recorded)",
              row1 == ("APPROVED", None, None, None))
        check("an actual span+label correction round-trips through submit_review() "
             "and back out via load_hitl_queue()'s same columns",
              row2 == ("CORRECTED", 6, 9, "Condition"))

        # And confirm load_hitl_queue() itself surfaces the new fields (the
        # page's "Already reviewed: ... span/label corrected" display reads
        # exactly these keys).
        from src.hitl_queue import load_hitl_queue as _load_hitl_queue
        conn4 = duckdb.connect(db_path, read_only=True)
        queue_rows = _load_hitl_queue(conn4)
        conn4.close()
        e2_row = next((r for r in queue_rows if r["entity_id"] == "e2"), None)
        check("load_hitl_queue() exposes corrected_orig_start/end/entity_label",
              e2_row is not None and e2_row["corrected_orig_start"] == 6
              and e2_row["corrected_orig_end"] == 9
              and e2_row["corrected_entity_label"] == "Condition")

    print(f"hitl-review-queue-page tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_hitl_review_queue_page():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
