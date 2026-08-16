"""
src/acronym_escalation.py -- Pass 1: MoLLM acronym escalation for
already-flagged ambiguous abbreviations.

TARGETS entities with expansion_ambiguous=TRUE (src/preprocessing.py's
expand_text_and_track_offsets() already computed and stored their
candidate_expansions list -- this module resolves ambiguity, it does not
discover it). See docs/2026-08-16_Shadow_Run_Precision_At_Scale.md and the
project plan's Phase 4 section for the full design rationale: local Ollama
(not an external API), domain classification via
src/preprocessing.py's _omop_domain_for_meaning() rather than an LLM guess,
and interception in src/normalization/orchestrator.py's
process_and_normalize_entities() (reads `mollm_resolved_expansion` off each
entity dict, the same upstream-attaches-a-field pattern already used for
assertion_status/is_allergy_context).

BUILD-ORDER STEP 1 (current state): resolve_ambiguous_acronyms() is a
HARDCODED MOCK, not yet backed by a real MoLLM call or the acronym_priors
cache -- this file exists to prove the orchestrator.py interception wiring
end-to-end before either of those exist. MOCK_RESOLUTIONS is a smoke-test
fixture only, never a production data source.
"""

# entity_id -> {"expansion": str, "omop_domain": str|None}
#
# Real case from this session's investigation
# (docs/2026-08-16_Shadow_Run_Precision_At_Scale.md): note 11134545-DS-21's
# "PDA" (candidate_expansions: "patent ductus arteriosus" / "posterior
# descending artery") currently resolves to "patent ductus arteriosus"
# (alphabetically first) even though its own local_context -- "3-vessel
# coronary artery disease ___ left anterior descending artery 60%, midLAD
# 100%, patent ductus arteriosus 80% diffusely diseased, ___ right coronary
# artery 100% on cath" -- is unambiguously listing THREE coronary arteries
# (LAD, PDA, RCA) by stenosis percentage, not a congenital heart defect.
# Chosen deliberately over a simpler case because it also exercises the
# domain-correction path (not just the search-text one): GLiNER's own
# entity_label for this span is "Condition" (shaped by the wrong "patent
# ductus arteriosus" text it was extracted against), but the CORRECT
# meaning is an Anatomy/vessel-structure concept -- domain_id='Spec Anatomic
# Site', not 'Condition'. Verified directly against the live DB:
# normalize_entity("posterior descending artery", gliner_label="Condition",
# domain_override=["Spec Anatomic Site"]) finds "Structure of posterior
# descending coronary artery" at 0.9453 similarity. The domain here is
# hand-specified for this mock rather than sourced from
# src.preprocessing._omop_domain_for_meaning() (build-order step 2's job) --
# that function's own Tier-1/2-exact-only lookup returns None for this
# exact meaning string (its wording doesn't exact-match any concept name),
# a real limitation to keep in mind once step 2 replaces this mock.
MOCK_RESOLUTIONS = {
    "11134545-DS-21-e70e4701cd0": {
        "expansion": "posterior descending artery",
        "omop_domain": "Spec Anatomic Site",
    },
}


def resolve_ambiguous_acronyms(entities: list, raw_text: str, note_id: str, conn,
                               client=None) -> dict:
    """Returns {entity_id: {"expansion": str, "omop_domain": str|None,
    "source": "mock"}} for every entity in `entities` that is
    expansion_ambiguous=TRUE and has a MOCK_RESOLUTIONS entry. An entity
    with no entry is simply absent from the returned dict -- it falls
    through to today's Stage 1 alphabetical-default expansion unchanged,
    same as any entity this phase doesn't (yet) touch.

    Signature matches what build-order step 4 will need
    (src/clinical_pipeline.py's run_pipeline(), between
    split_compound_entities() and process_and_normalize_entities()) even
    though raw_text/conn/client are unused by the mock -- callers don't
    need to change when the mock is replaced with the real implementation.
    """
    resolved = {}
    for ent in entities:
        if not ent.get("expansion_ambiguous"):
            continue
        entity_id = ent.get("entity_id")
        mock = MOCK_RESOLUTIONS.get(entity_id)
        if mock is None:
            continue
        resolved[entity_id] = {
            "expansion": mock["expansion"],
            "omop_domain": mock.get("omop_domain"),
            "source": "mock",
        }
    return resolved
