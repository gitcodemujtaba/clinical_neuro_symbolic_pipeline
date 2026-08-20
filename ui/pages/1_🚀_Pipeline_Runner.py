"""ui/pages/1_🚀_Pipeline_Runner.py — pipeline demonstration / entity tracer.

Picks an already-PROCESSED note (is_test=TRUE in extracted_entities) rather
than triggering a fresh live run: a real run needs GLiNER-BioMed + SapBERT +
3 concurrent Ollama models loaded, costs real wall-clock time per entity
(minutes, not seconds -- confirmed repeatedly this session), and would
contend with any batch job already using those same models/the DB write
lock. Replaying an already-stored trace is instant, needs only a read-only
connection, and shows the exact same stage-by-stage decisions a live run
would have made -- nothing here is simulated or approximated.

read_only=True deliberately: this page never writes. If a batch job
(scripts/test_pipeline_e2e.py, scripts/run_stage3_tier_gate.py) holds the DB
open for writing, DuckDB's single-writer model means even a read-only
connect() here will fail -- caught explicitly below with a page-level
message rather than a raw traceback, since "the DB is busy right now" is a
normal, expected state during this project's own overnight/batch runs, not
an error a user of this page should have to interpret from a stack trace.
"""
import json
import os
import sys

import duckdb
import streamlit as st

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
sys.path.insert(0, PROJECT_DIR)

from ui.components.db_status import render_locked_db_status  # noqa: E402

DB_PATH = os.environ.get("CNSP_DB_PATH", os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))

st.set_page_config(page_title="Pipeline Runner", page_icon="🚀", layout="wide")
st.title("🚀 Pipeline Runner — Entity Tracer")
st.caption(
    "Pick an already-processed note and a number of entities to track. Each "
    "one is shown stage by stage -- Stage 1 extraction, Stage 2a abbreviation "
    "expansion, Stage 2b normalization, Stage 3 tier gate -- ending in its "
    "real AUTO_VALIDATED or HITL_REQUIRED outcome."
)

AUTO_TIERS = {"TIER_1_AUTO_VALIDATED", "TIER_1B_CALIBRATED_AUTO_VALIDATED",
             "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED"}


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
    """st.stop(), but closing `conn` first -- explicit close rather than
    relying on GC timing, since a lingering read-only connection (even one
    Python has already dropped its own reference to) has been confirmed to
    block ui/pages/2_🩺_HITL_Review_Queue.py's write connection when both
    are touched within the same Streamlit server process (streamlit.testing
    AppTest reproduced this directly -- del+gc.collect() alone was not
    enough without an explicit .close()). Every st.stop() call site in this
    file goes through this instead of calling st.stop() directly."""
    conn.close()
    st.stop()


def _json_field(v):
    if v is None:
        return None
    return json.loads(v) if isinstance(v, str) else v


def _tier_badge(tier: str, routing_decision: str):
    if tier is None:
        st.warning("⏳ Not yet reached Stage 3 (no tier-gate decision recorded for this entity).")
    elif tier in AUTO_TIERS:
        st.success(f"✅ **{tier}** — {routing_decision} (auto, no human review required for KG3 write)")
    else:
        st.error(f"🧑‍⚕️ **{tier or 'HITL'}** — {routing_decision or 'HITL_REQUIRED'} (queued for human review)")


def render_entity_trace(entity: dict, norm: dict, decision: dict):
    original_text = entity["original_text"] or "(no text)"
    with st.container(border=True):
        st.subheader(f"{original_text}  ·  {entity['entity_label']}")
        st.caption(f"entity_id: {entity['entity_id']}")

        # ------------------------------------------------------------
        # Stage 1 — extraction
        # ------------------------------------------------------------
        st.markdown("#### 1️⃣ Stage 1 — Extraction (GLiNER-BioMed)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GLiNER confidence", f"{entity['confidence']:.3f}" if entity["confidence"] is not None else "—")
        c2.metric("Assertion", entity["assertion_status"] or "—")
        c3.metric("Experiencer", entity["experiencer"] or "—")
        c4.metric("Section", (entity["section_name"] or "—")[:24])
        if entity["local_context"]:
            highlighted = entity["local_context"]
            if original_text and original_text in highlighted:
                highlighted = highlighted.replace(original_text, f"**:orange[{original_text}]**", 1)
            st.markdown(f"> {highlighted}")

        # ------------------------------------------------------------
        # Stage 2a — abbreviation expansion
        # ------------------------------------------------------------
        st.markdown("#### 2️⃣ Stage 2a — Abbreviation Expansion")
        if entity["expanded_text"] and entity["expanded_text"] != original_text:
            st.markdown(f"**{original_text!r}** → **{entity['expanded_text']!r}**")
            if entity["expansion_ambiguous"]:
                candidates_exp = _json_field(entity["candidate_expansions"]) or []
                st.warning(
                    f"⚠️ Ambiguous abbreviation — {len(candidates_exp)} possible meaning(s): "
                    + ", ".join(candidates_exp)
                )
                st.caption(f"Tiebreak used: `{entity['selection_basis'] or 'alphabetical_default'}`")
            else:
                st.caption("Unambiguous — single dictionary meaning.")
        else:
            st.caption("No abbreviation expansion applied — text unchanged from Stage 1.")

        # ------------------------------------------------------------
        # Stage 2b — normalization
        # ------------------------------------------------------------
        st.markdown("#### 3️⃣ Stage 2b — Normalization (OMOP/SNOMED)")
        if not norm:
            st.warning("⏳ Not yet normalized (no Stage 2b record for this entity).")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Match tier", norm.get("match_tier") or "—")
            m2.metric("Top similarity score", norm.get("similarity_score"))
            if norm.get("is_ambiguous"):
                st.warning(f"⚠️ Retrieval ambiguous — {norm.get('ambiguity_reason') or 'no reason recorded'}")
            if norm.get("domain_conflict"):
                st.warning("⚠️ Domain conflict flagged between the GLiNER label and the resolved concept's OMOP domain.")
            candidates = _json_field(norm.get("candidates")) or []
            if candidates:
                st.markdown("**Ranked candidates** (this is where a 2nd/3rd possibility, if any, comes from):")
                for i, c in enumerate(candidates, 1):
                    marker = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"#{i}"))
                    st.text(
                        f"{marker} {c.get('concept_name')}  "
                        f"(OMOP {c.get('omop_concept_id')}, {c.get('domain_id')}, "
                        f"score {c.get('similarity_score')}, basis {c.get('match_basis')})"
                    )

        # ------------------------------------------------------------
        # Stage 3 — tier gate
        # ------------------------------------------------------------
        st.markdown("#### 4️⃣ Stage 3 — Tier Gate (two-step CoT ensemble + calibrator)")
        if not decision:
            st.warning("⏳ Not yet reached Stage 3 (no mollm_tier_gate_decisions row for this entity).")
        else:
            tier = decision.get("tier")
            routing = decision.get("mollm_routing_decision")
            if decision.get("routing_basis"):
                st.info(f"**How the pipeline reached this conclusion:** {decision['routing_basis']}")
            models = _json_field(decision.get("models")) or []
            if models:
                st.markdown("**Per-model verdicts:**")
                vcols = st.columns(len(models))
                for col, m in zip(vcols, models):
                    with col:
                        verdict = m.get("verdict", "—")
                        icon = "✅" if verdict == "SUPPORTED_1" else (
                            "🔁" if str(verdict).startswith("RE_RANK") else "❌")
                        st.markdown(f"**{m.get('model', '?')}**")
                        st.markdown(f"{icon} `{verdict}`")
                        if m.get("reasoning"):
                            with st.expander("reasoning"):
                                st.write(m["reasoning"])
            _tier_badge(tier, routing)


with st.sidebar:
    st.header("Selection")
    # extracted_entities itself carries no is_stale/provenance columns
    # (only normalized_entities and mollm_tier_gate_decisions do) -- joined
    # here rather than filtered directly. is_stale=FALSE means "processed
    # by current code" -- see scripts/mark_notes_stale.py for the migration.
    note_ids = [r[0] for r in conn.execute("""
        SELECT DISTINCT e.note_id FROM extracted_entities e
        WHERE e.is_test = TRUE AND e.note_id IN (
            SELECT DISTINCT note_id FROM normalized_entities
            WHERE is_test = TRUE AND is_stale = FALSE
        ) ORDER BY e.note_id
    """).fetchall()]
    if not note_ids:
        st.warning("No fresh (is_stale = FALSE) notes found -- older pre-fix runs are "
                  "deliberately hidden. Run the pipeline on some notes first "
                  "(scripts/test_pipeline_e2e.py), then reload this page.")
        _stop()
    note_id = st.selectbox("Note", note_ids)
    n_entities = st.number_input("Number of entities to track", min_value=1, max_value=50, value=1, step=1)
    search = st.text_input("Filter: entity text contains (optional)")

entity_rows = conn.execute("""
    SELECT entity_id, original_text, expanded_text, entity_label, confidence,
           orig_start, orig_end, assertion_status, experiencer, section_name,
           local_context, expansion_ambiguous, candidate_expansions, selection_basis
    FROM extracted_entities
    WHERE note_id = ? AND is_test = TRUE
    ORDER BY orig_start
""", [note_id]).fetchall()
entity_cols = ["entity_id", "original_text", "expanded_text", "entity_label", "confidence",
              "orig_start", "orig_end", "assertion_status", "experiencer", "section_name",
              "local_context", "expansion_ambiguous", "candidate_expansions", "selection_basis"]
entities = [dict(zip(entity_cols, r)) for r in entity_rows]

if search:
    entities = [e for e in entities if search.lower() in (e["original_text"] or "").lower()]

st.caption(f"**{note_id}** — {len(entities)} entit{'y' if len(entities) == 1 else 'ies'} available "
          f"(after filter), tracking the first {min(n_entities, len(entities))}.")

tracked = entities[:n_entities]
if not tracked:
    st.info("No entities match this filter in this note. Adjust the filter or pick a different note.")
    _stop()

entity_ids = [e["entity_id"] for e in tracked]
placeholders = ",".join("?" * len(entity_ids))

norm_rows = conn.execute(f"""
    SELECT entity_id, match_tier, similarity_score, is_ambiguous, ambiguity_reason,
           domain_conflict, candidates, normalized_from
    FROM normalized_entities WHERE entity_id IN ({placeholders})
""", entity_ids).fetchall()
norm_cols = ["entity_id", "match_tier", "similarity_score", "is_ambiguous", "ambiguity_reason",
            "domain_conflict", "candidates", "normalized_from"]
norm_by_entity = {r[0]: dict(zip(norm_cols, r)) for r in norm_rows}

decision_rows = conn.execute(f"""
    SELECT entity_id, tier, mollm_routing_decision, routing_basis, models, composite_confidence
    FROM mollm_tier_gate_decisions WHERE entity_id IN ({placeholders})
""", entity_ids).fetchall()
decision_cols = ["entity_id", "tier", "mollm_routing_decision", "routing_basis", "models",
                 "composite_confidence"]
decision_by_entity = {r[0]: dict(zip(decision_cols, r)) for r in decision_rows}

for e in tracked:
    render_entity_trace(e, norm_by_entity.get(e["entity_id"]), decision_by_entity.get(e["entity_id"]))

conn.close()
