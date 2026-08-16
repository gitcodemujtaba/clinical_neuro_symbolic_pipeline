"""
src/hitl_queue.py — Stage 4 (Routing & Ingestion) HITL review queue.

WHY THIS IS A SEPARATE TABLE, NOT A COLUMN ON mollm_decisions/mollm_review_decisions.
Those two tables are each owned by their own module (src/mollm_ensemble.py's
Objective 2, src/mollm_review.py's Objective 3) and record what a decision
WAS. This table records what happened to it AFTERWARD -- a human reviewer's
verdict on it -- which is a different lifecycle with its own states
(PENDING/APPROVED/CORRECTED/REJECTED) and its own writer (the Streamlit
reviewer UI, not a batch script). Keeping it separate means neither source
table's schema has to anticipate a review workflow that may or may not ever
touch a given row, and a row can be re-queued or re-reviewed without ever
touching the original decision artifact.

WHY EVERY DECISION IS QUEUED, NOT JUST HITL_REQUIRED/AL_HITL_REQUIRED ONES.
2026-08-14: AUTO_VALIDATED precision measured at 39.4% on the freshly
re-validated 27-note corpus -- see docs/Implementation_Checklist.md's own
warning against writing unfiltered high-confidence Stage 3 output straight
into KG3 ("risks baking silent errors into the graph as 'verified'... a
pseudo-labeling feedback-loop risk"). Until a calibrated confidence threshold
(src/mollm_calibrator.py) is fit and validated against real reviewed data,
EVERY decision from EITHER source table is queued for human review
regardless of its own routing tier -- queue_reason records the SOURCE row's
own tier/reason for a reviewer's context, but does not gate queuing itself.
This is a deliberate, temporary conservatism, not the final design.

WHY THREE SOURCE TABLES FEED ONE QUEUE.
src/mollm_review.py's own docstring: this module "produces the CANDIDATE
rows a future Stage 4 job would consume." mollm_decisions (Objective 2,
citation-gated), mollm_review_decisions (Objective 3, confidence-driven,
all-tier), and mollm_tier_gate_decisions (2026-08-16, Pass 4's two-step CoT
+ Tier 1-5 gate, src/mollm_tier_gate.py) are independent judgments over
often-overlapping entities; a human reviewer benefits from seeing all that
exist, tagged by source_table so the UI/analysis can tell them apart.
"""
import json
import re

from src.provenance import (
    provenance_alter_statements,
    provenance_column_sql,
    provenance_params,
    provenance_placeholders,
)


def ensure_hitl_queue_table(conn):
    """Creates/migrates hitl_review_queue. Idempotent, matches the
    CREATE TABLE IF NOT EXISTS + additive ALTER pattern every other
    decision-bearing table in this codebase uses (see
    src/mollm_ensemble.py's store_decision(), src/mollm_review.py's
    _ensure_review_table()) so a table that already exists on a deployed box
    picks up new columns without a manual migration step.
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS hitl_review_queue (
        hitl_case_id VARCHAR PRIMARY KEY,
        source_table VARCHAR,
        source_call_id VARCHAR,
        entity_id VARCHAR,
        note_id VARCHAR,
        queue_reason VARCHAR,
        presented_suggestion JSON,
        reviewer_decision VARCHAR DEFAULT 'PENDING',
        corrected_concept_id BIGINT,
        rejection_reason VARCHAR,
        review_duration DOUBLE,
        final_ingestion_path VARCHAR,
        is_test BOOLEAN DEFAULT FALSE
    );
    """)
    # Additive migration (2026-08-14, Step 4): tracks whether a
    # HUMAN_VERIFIED case has already been written to KG3, so
    # scripts/run_kg3_ingestion.py can be re-run safely -- only rows with
    # final_ingestion_path='HUMAN_VERIFIED' AND ingested_at IS NULL are
    # candidates for the next batch, same "additive ALTER, never destructive"
    # pattern as every other migration in this codebase.
    conn.sql("ALTER TABLE hitl_review_queue ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP;")
    for stmt in provenance_alter_statements("hitl_review_queue"):
        conn.sql(stmt)


def load_ingestible_cases(conn) -> list:
    """Cases ready for Step 4's KG3 write-back: reviewed APPROVED/CORRECTED
    (final_ingestion_path='HUMAN_VERIFIED') and not yet ingested. Excludes
    REJECTED cases structurally, not just by convention -- REJECTED rows
    never get final_ingestion_path set (see submit_review()), so they can
    never appear here.
    """
    rows = conn.execute("""
        SELECT hitl_case_id, source_table, source_call_id, entity_id, note_id,
               presented_suggestion, reviewer_decision, corrected_concept_id
        FROM hitl_review_queue
        WHERE final_ingestion_path = 'HUMAN_VERIFIED' AND ingested_at IS NULL
        ORDER BY created_at ASC
    """).fetchall()
    out = []
    for r in rows:
        suggestion = r[5]
        if isinstance(suggestion, str):
            suggestion = json.loads(suggestion)
        out.append({
            "hitl_case_id": r[0], "source_table": r[1], "source_call_id": r[2],
            "entity_id": r[3], "note_id": r[4], "presented_suggestion": suggestion,
            "reviewer_decision": r[6], "corrected_concept_id": r[7],
        })
    return out


def mark_ingested(conn, hitl_case_id: str):
    """Stamps ingested_at so load_ingestible_cases() won't return this case
    again. Called once per case AFTER its Cypher transaction commits
    successfully -- see scripts/run_kg3_ingestion.py.
    """
    conn.execute(
        "UPDATE hitl_review_queue SET ingested_at = CURRENT_TIMESTAMP WHERE hitl_case_id = ?",
        [hitl_case_id],
    )


_RESOLVED_RE = re.compile(r"^RESOLVED_TO_CANDIDATE_(\d+)$")


def _suggested_omop_concept_id(models: list, candidates: list):
    """The concept_id an APPROVE click confirms, computed once at enqueue
    time so "Approve" has an unambiguous, explicit meaning rather than being
    re-derived from raw model verdicts at ingestion time (or worse, at
    ingestion time under time pressure to ship Step 4).

    Two mollm_ensemble.py verdict shapes to handle, per its own route()/
    evaluation/cal_eval.py's RESOLVED_RE pattern:
      - "resolution" mode: verdicts are RESOLVED_TO_CANDIDATE_<N> (or
        NONE_CORRECT). Majority vote among models that resolved to a
        specific N wins; N maps to candidates[N-1].
      - "contradiction" mode: verdicts are SUPPORTED/CONTRADICTED/
        INSUFFICIENT_EVIDENCE -- there is no candidate array being chosen
        among, Stage 3 is checking Stage 2b's OWN top-1 pick
        (candidates[0], by this codebase's standing convention) against
        guideline evidence. SUPPORTED/no clear signal defaults to
        candidates[0]; there is nothing better to suggest without a human.

    Returns None when neither pattern yields a usable candidate (e.g. every
    model said NONE_CORRECT) -- see ingest_reviewed_case()'s docstring for
    why an APPROVED case with no resolvable id here is a loud error, not a
    silent skip.
    """
    if not candidates:
        return None
    votes = []
    for m in models or []:
        verdict = m.get("verdict") or ""
        match = _RESOLVED_RE.match(verdict)
        if match:
            votes.append(int(match.group(1)))
    if votes:
        # majority index (ties broken by lowest index, matching this
        # codebase's existing "prefer the earlier/higher-ranked candidate"
        # convention elsewhere, e.g. _rank_tier12_candidates())
        idx = max(set(votes), key=lambda v: (votes.count(v), -v))
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1].get("omop_concept_id")
        return None
    # contradiction mode (or no parseable verdicts at all): Stage 2b's own
    # top-1 is what was actually being validated.
    non_resolution = all(
        (m.get("verdict") or "") in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
        for m in (models or []) if m.get("verdict")
    )
    if non_resolution and models:
        return candidates[0].get("omop_concept_id")
    return None


def _presented_suggestion_from_decision(row: dict) -> dict:
    """Builds the reviewer-facing suggestion payload from a mollm_decisions
    row plus its joined entity text/candidates/context. The candidate list
    and model verdicts alone are not enough for a reviewer to judge a
    decision -- 2026-08-15 reviewer feedback: the queue was showing model
    reasoning with no surrounding note text at all, so `local_context`
    (the sentence-bounded window Stage 2a builds specifically to carry
    negation/qualifier cues -- see src/entity_extraction.py's
    build_local_context()) plus `section_name`/`assertion_status`/
    `experiencer` are included too. The full `prompt` VARCHAR already stored
    on the source row remains available separately for a re-audit that
    needs even more than this (the exact evidence block, retrieval trace).
    """
    candidates = row.get("candidates") or []
    models = row.get("models") or []
    return {
        "source": "mollm_decisions",
        "original_text": row.get("original_text"),
        "entity_label": row.get("entity_label"),
        "candidates": candidates,
        "routing_decision": row.get("mollm_routing_decision"),
        "confidence_tier_in": row.get("confidence_tier_in"),
        "composite_confidence": row.get("composite_confidence"),
        "models": models,
        "suggested_omop_concept_id": _suggested_omop_concept_id(models, candidates),
        "local_context": row.get("local_context"),
        "section_name": row.get("section_name"),
        "assertion_status": row.get("assertion_status"),
        "experiencer": row.get("experiencer"),
    }


_OMOP_ID_IN_TEXT_RE = re.compile(r"\(OMOP\s+(\d+)")


def _suggested_id_from_proposed_name(proposed_concept_name, candidates):
    """Resolves mollm_review_decisions' own `proposed_concept_name` (a plain
    string -- that table has no concept_id column of its own) to a concrete
    omop_concept_id, against Stage 2b's candidate list (normalized_entities.
    candidates, the same list mollm_decisions-sourced cases already use).

    2026-08-15, second round of reviewer feedback: only 20/2305 rows have
    a non-null proposed_concept_name at all (this field is a CORRECTION --
    an assessment='CORRECT' row genuinely has nothing to propose, since the
    existing Stage 2b top-1 candidate already is the answer; that's
    expected sparsity, not a bug). Of those 20, checked every one directly:
    6 aren't a clean concept name at all -- the model echoed the full
    candidate description it was shown, e.g. "Partial thromboplastin time
    finding (OMOP 4187646, SNOMED/Measurement)" -- which a bare exact-match
    against candidates[].concept_name can never match. Two-pronged fix:

    1. PRIMARY: extract an explicit "(OMOP <id>" annotation and validate it
       against the actual candidate id set before trusting it -- never
       trust an arbitrary number in free text on its own (a hallucinated or
       copy-pasted-from-elsewhere id must not silently pass through).
    2. FALLBACK: exact-match (case-insensitive, whitespace-trimmed) against
       candidates[].concept_name, tried against both the raw string and the
       string with a trailing " (...)" annotation stripped -- handles both
       clean names ("ECR muscle") and non-OMOP-tagged suffixed ones
       ("Inferior vena cava (Anatomical Structure)"). Deliberately no
       fuzzy/partial matching beyond this -- that would risk silently
       resolving to the WRONG candidate among near-duplicate names, the
       same risk _prefer_lab_procedure_over_observable() and
       _collapse_hierarchy_duplicates() exist elsewhere to guard against.

    Returns None when nothing resolves -- the caller (ingest_reviewed_case())
    already treats None as a loud error on APPROVED, not a silent skip.
    """
    if not proposed_concept_name or not candidates:
        return None

    candidate_ids = {c.get("omop_concept_id") for c in candidates}
    id_match = _OMOP_ID_IN_TEXT_RE.search(proposed_concept_name)
    if id_match:
        candidate_id = int(id_match.group(1))
        if candidate_id in candidate_ids:
            return candidate_id

    for text in (proposed_concept_name, proposed_concept_name.split(" (")[0]):
        target = text.strip().lower()
        for c in candidates:
            name = (c.get("concept_name") or "").strip().lower()
            if name == target:
                return c.get("omop_concept_id")
    return None


def _presented_suggestion_from_review(row: dict) -> dict:
    """Same shape as _presented_suggestion_from_decision(), sourced from
    mollm_review_decisions instead, so the UI can render either
    interchangeably regardless of which pipeline produced the case.

    suggested_omop_concept_id is only non-None when proposed_concept_name
    exactly matches one of the joined candidates -- see
    _suggested_id_from_proposed_name(). Genuinely unresolvable (no exact
    match, or proposed_concept_name empty/None) still leaves it None; an
    APPROVED case in that state has no id ingest_reviewed_case() can write
    without a reviewer supplying one via CORRECTED instead -- see that
    function's docstring for why this is a loud error, not a silent guess.
    """
    candidates = row.get("candidates") or []
    proposed_name = row.get("proposed_concept_name")
    return {
        "source": "mollm_review_decisions",
        "original_text": row.get("original_text"),
        "entity_label": row.get("proposed_entity_label"),
        "proposed_concept_name": proposed_name,
        "candidates": candidates,
        "routing_decision": row.get("al_routing_decision"),
        "confidence_tier_in": row.get("confidence_tier_in"),
        "composite_confidence": row.get("composite_confidence"),
        "assessment": row.get("assessment"),
        "models": row.get("models"),
        "suggested_omop_concept_id": _suggested_id_from_proposed_name(proposed_name, candidates),
        "local_context": row.get("local_context"),
        "section_name": row.get("section_name"),
        "assertion_status": row.get("assertion_status"),
        "experiencer": row.get("experiencer"),
    }


def _presented_suggestion_from_tier_gate_decision(row: dict) -> dict:
    """Same shape as _presented_suggestion_from_decision()/
    _presented_suggestion_from_review(), sourced from
    mollm_tier_gate_decisions (src/mollm_tier_gate.py's route_tier(), Pass 4
    two-step CoT + Tier 1-5 gate) instead.

    suggested_omop_concept_id resolution is simpler here than the other two
    sources' string-matching/parsing: route_tier() already records exactly
    which candidate it picked as `final_candidate_index` (1-based, or None
    for a Tier 4/5 decision with no chosen candidate), so this is a direct
    list index rather than a name/id extracted from free text.
    """
    candidates = row.get("candidates") or []
    idx = row.get("final_candidate_index")
    suggested_id = None
    if idx and 1 <= idx <= len(candidates):
        suggested_id = candidates[idx - 1].get("omop_concept_id")
    return {
        "source": "mollm_tier_gate_decisions",
        "original_text": row.get("original_text"),
        "entity_label": row.get("entity_label"),
        "candidates": candidates,
        "routing_decision": row.get("mollm_routing_decision"),
        "tier": row.get("tier"),
        "composite_confidence": row.get("composite_confidence"),
        "models": row.get("models"),
        "suggested_omop_concept_id": suggested_id,
        "local_context": row.get("local_context"),
        "section_name": row.get("section_name"),
        "assertion_status": row.get("assertion_status"),
        "experiencer": row.get("experiencer"),
    }


def enqueue_pending_cases(conn, is_test: bool = True) -> int:
    """Inserts one PENDING hitl_review_queue row for every mollm_decisions
    and mollm_review_decisions row (error IS NULL, matching
    already_processed_entity_ids()'s own "only a clean decision counts"
    rule) that isn't already queued. Idempotent -- ON CONFLICT DO NOTHING on
    hitl_case_id, and hitl_case_id is deterministic (source_table +
    source_call_id), so re-running this after either batch job produces more
    decisions only adds the new ones.

    Returns the number of rows actually inserted.
    """
    ensure_hitl_queue_table(conn)

    decision_rows = conn.execute("""
        SELECT d.mollm_call_id, d.entity_id, d.note_id, d.queue_reason,
               d.mollm_routing_decision, d.confidence_tier_in, d.composite_confidence,
               d.models, e.original_text, e.entity_label, n.candidates,
               e.local_context, e.section_name, e.assertion_status, e.experiencer
        FROM mollm_decisions d
        LEFT JOIN extracted_entities e ON e.entity_id = d.entity_id
        LEFT JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.is_test = ? AND d.error IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM hitl_review_queue q
              WHERE q.source_table = 'mollm_decisions' AND q.source_call_id = d.mollm_call_id
          )
    """, [is_test]).fetchall()
    decision_cols = ["mollm_call_id", "entity_id", "note_id", "queue_reason",
                     "mollm_routing_decision", "confidence_tier_in", "composite_confidence",
                     "models", "original_text", "entity_label", "candidates",
                     "local_context", "section_name", "assertion_status", "experiencer"]

    # LEFT JOIN extracted_entities here too -- mollm_review_decisions (Objective
    # 3, src/mollm_review.py) stores its own original_text but has NO context/
    # section/assertion columns of its own; those only exist on the entity's
    # actual extracted_entities row (2026-08-15, reviewer feedback: the queue
    # was showing model verdicts with no surrounding note text, offering no
    # way to actually judge them).
    # LEFT JOIN normalized_entities here too -- mollm_review_decisions itself
    # has no candidates column (only a proposed_concept_name STRING), but
    # Stage 2b's candidate list is the same regardless of which Stage 3
    # pipeline is reviewing the entity. 2026-08-15 reviewer feedback: the
    # queue wasn't showing which OMOP/SNOMED code the entity was mapped to
    # at all for this source -- the concept_id existed only buried inside a
    # model's free-text reasoning.
    review_rows = conn.execute("""
        SELECT r.review_call_id, r.entity_id, r.note_id, r.queue_reason,
               r.al_routing_decision, r.confidence_tier_in, r.composite_confidence,
               r.models, r.original_text, r.proposed_entity_label, r.proposed_concept_name,
               r.assessment,
               e.local_context, e.section_name, e.assertion_status, e.experiencer,
               n.candidates
        FROM mollm_review_decisions r
        LEFT JOIN extracted_entities e ON e.entity_id = r.entity_id
        LEFT JOIN normalized_entities n ON n.entity_id = r.entity_id
        WHERE r.is_test = ? AND r.error IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM hitl_review_queue q
              WHERE q.source_table = 'mollm_review_decisions' AND q.source_call_id = r.review_call_id
          )
    """, [is_test]).fetchall()
    review_cols = ["review_call_id", "entity_id", "note_id", "queue_reason",
                  "al_routing_decision", "confidence_tier_in", "composite_confidence",
                  "models", "original_text", "proposed_entity_label", "proposed_concept_name",
                  "assessment", "local_context", "section_name", "assertion_status", "experiencer",
                  "candidates"]

    def _json_field(v):
        if v is None:
            return None
        return json.loads(v) if isinstance(v, str) else v

    inserted = 0
    for raw in decision_rows:
        row = dict(zip(decision_cols, raw))
        row["models"] = _json_field(row["models"])
        row["candidates"] = _json_field(row["candidates"])
        suggestion = _presented_suggestion_from_decision(row)
        conn.execute("""
            INSERT INTO hitl_review_queue
            (hitl_case_id, source_table, source_call_id, entity_id, note_id,
             queue_reason, presented_suggestion, reviewer_decision, is_test,
             {provenance_cols})
            VALUES (?, 'mollm_decisions', ?, ?, ?, ?, ?, 'PENDING', ?, {provenance_ph})
            ON CONFLICT (hitl_case_id) DO NOTHING;
        """.format(provenance_cols=provenance_column_sql(), provenance_ph=provenance_placeholders()),
        [f"hitl_mollm_decisions_{row['mollm_call_id']}", row["mollm_call_id"],
         row["entity_id"], row["note_id"], row["queue_reason"],
         json.dumps(suggestion, default=str), is_test] + provenance_params())
        inserted += 1

    for raw in review_rows:
        row = dict(zip(review_cols, raw))
        row["models"] = _json_field(row["models"])
        row["candidates"] = _json_field(row["candidates"])
        suggestion = _presented_suggestion_from_review(row)
        conn.execute("""
            INSERT INTO hitl_review_queue
            (hitl_case_id, source_table, source_call_id, entity_id, note_id,
             queue_reason, presented_suggestion, reviewer_decision, is_test,
             {provenance_cols})
            VALUES (?, 'mollm_review_decisions', ?, ?, ?, ?, ?, 'PENDING', ?, {provenance_ph})
            ON CONFLICT (hitl_case_id) DO NOTHING;
        """.format(provenance_cols=provenance_column_sql(), provenance_ph=provenance_placeholders()),
        [f"hitl_mollm_review_decisions_{row['review_call_id']}", row["review_call_id"],
         row["entity_id"], row["note_id"], row["queue_reason"],
         json.dumps(suggestion, default=str), is_test] + provenance_params())
        inserted += 1

    # mollm_tier_gate_decisions (2026-08-16, production deploy of
    # src/mollm_tier_gate.py's Tier 1-5 gate). Table may not exist yet in a
    # DB that has never had store_tier_decision() called against it --
    # created here too (same DDL that function uses) rather than letting
    # this query fail on a fresh DB.
    conn.sql("""
    CREATE TABLE IF NOT EXISTS mollm_tier_gate_decisions (
        mollm_call_id VARCHAR PRIMARY KEY,
        entity_id VARCHAR,
        note_id VARCHAR,
        tier VARCHAR,
        mollm_routing_decision VARCHAR,
        queue_reason VARCHAR,
        final_candidate_index INTEGER,
        composite_confidence DOUBLE,
        routing_basis VARCHAR,
        models JSON,
        is_test BOOLEAN DEFAULT FALSE
    );
    """)
    # EVERY tier-gate decision is queued too, same "deliberate, temporary
    # conservatism" this module's docstring already establishes for the
    # other two sources -- including Tier 1/2/3 (AUTO_VALIDATED/
    # AUTO_RESOLVED) ones. src.kg3_ingestion.ingest_auto_decision() exists
    # and is exercised (dry_run=True) for those tiers by the production
    # runner, but nothing here skips queuing them for human review just
    # because their tier is "auto" -- that gate is a future, deliberate
    # change once precision is validated at real scale, not a side effect
    # of wiring the new gate in.
    tier_gate_rows = conn.execute("""
        SELECT g.mollm_call_id, g.entity_id, g.note_id, g.queue_reason,
               g.mollm_routing_decision, g.tier, g.composite_confidence,
               g.final_candidate_index, g.models, e.original_text, e.entity_label,
               n.candidates, e.local_context, e.section_name, e.assertion_status,
               e.experiencer
        FROM mollm_tier_gate_decisions g
        LEFT JOIN extracted_entities e ON e.entity_id = g.entity_id
        LEFT JOIN normalized_entities n ON n.entity_id = g.entity_id
        WHERE g.is_test = ?
          AND NOT EXISTS (
              SELECT 1 FROM hitl_review_queue q
              WHERE q.source_table = 'mollm_tier_gate_decisions'
                AND q.source_call_id = g.mollm_call_id
          )
    """, [is_test]).fetchall()
    tier_gate_cols = ["mollm_call_id", "entity_id", "note_id", "queue_reason",
                      "mollm_routing_decision", "tier", "composite_confidence",
                      "final_candidate_index", "models", "original_text", "entity_label",
                      "candidates", "local_context", "section_name", "assertion_status",
                      "experiencer"]

    for raw in tier_gate_rows:
        row = dict(zip(tier_gate_cols, raw))
        row["models"] = _json_field(row["models"])
        row["candidates"] = _json_field(row["candidates"])
        suggestion = _presented_suggestion_from_tier_gate_decision(row)
        conn.execute("""
            INSERT INTO hitl_review_queue
            (hitl_case_id, source_table, source_call_id, entity_id, note_id,
             queue_reason, presented_suggestion, reviewer_decision, is_test,
             {provenance_cols})
            VALUES (?, 'mollm_tier_gate_decisions', ?, ?, ?, ?, ?, 'PENDING', ?, {provenance_ph})
            ON CONFLICT (hitl_case_id) DO NOTHING;
        """.format(provenance_cols=provenance_column_sql(), provenance_ph=provenance_placeholders()),
        [f"hitl_mollm_tier_gate_decisions_{row['mollm_call_id']}", row["mollm_call_id"],
         row["entity_id"], row["note_id"], row["queue_reason"],
         json.dumps(suggestion, default=str), is_test] + provenance_params())
        inserted += 1

    return inserted


def load_hitl_queue(conn, status: str = None, note_id: str = None,
                    source_table: str = None) -> list:
    """Reads the queue back as a list of dicts, newest-queued first. `status`
    filters on reviewer_decision (e.g. 'PENDING'); None returns every case
    regardless of review state. This is the read side the UI (Step 3) and
    ingestion driver (Step 4) both use -- neither talks to
    mollm_decisions/mollm_review_decisions directly once a case is queued.
    """
    where = ["is_test = TRUE"]
    params = []
    if status:
        where.append("reviewer_decision = ?")
        params.append(status)
    if note_id:
        where.append("note_id = ?")
        params.append(note_id)
    if source_table:
        where.append("source_table = ?")
        params.append(source_table)

    rows = conn.execute(f"""
        SELECT hitl_case_id, source_table, source_call_id, entity_id, note_id,
               queue_reason, presented_suggestion, reviewer_decision,
               corrected_concept_id, rejection_reason, review_duration,
               final_ingestion_path, created_at
        FROM hitl_review_queue
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
    """, params).fetchall()

    out = []
    for r in rows:
        suggestion = r[6]
        if isinstance(suggestion, str):
            suggestion = json.loads(suggestion)
        out.append({
            "hitl_case_id": r[0], "source_table": r[1], "source_call_id": r[2],
            "entity_id": r[3], "note_id": r[4], "queue_reason": r[5],
            "presented_suggestion": suggestion, "reviewer_decision": r[7],
            "corrected_concept_id": r[8], "rejection_reason": r[9],
            "review_duration": r[10], "final_ingestion_path": r[11],
            "created_at": r[12],
        })
    return out


def submit_review(conn, hitl_case_id: str, reviewer_decision: str,
                  corrected_concept_id: int = None, rejection_reason: str = None,
                  review_duration: float = None):
    """Records a human reviewer's verdict on one queued case.

    reviewer_decision must be one of 'APPROVED' / 'CORRECTED' / 'REJECTED'
    (matching docs/Databases.md's :HITLReview final decision status enum,
    minus PENDING which is the pre-review default, not a submittable value).
    final_ingestion_path is set to 'HUMAN_VERIFIED' on APPROVED/CORRECTED --
    the only two outcomes Step 4's KG3 write-back (src/kg3_ingestion.py)
    will ever read as ingestible. REJECTED leaves final_ingestion_path NULL:
    a rejected case is not written to KG3 at all, by design.
    """
    if reviewer_decision not in ("APPROVED", "CORRECTED", "REJECTED"):
        raise ValueError(f"reviewer_decision must be APPROVED/CORRECTED/REJECTED, got {reviewer_decision!r}")
    final_ingestion_path = "HUMAN_VERIFIED" if reviewer_decision in ("APPROVED", "CORRECTED") else None
    conn.execute("""
        UPDATE hitl_review_queue
        SET reviewer_decision = ?, corrected_concept_id = ?, rejection_reason = ?,
            review_duration = ?, final_ingestion_path = ?
        WHERE hitl_case_id = ?
    """, [reviewer_decision, corrected_concept_id, rejection_reason,
          review_duration, final_ingestion_path, hitl_case_id])
