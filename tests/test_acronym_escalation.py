"""
tests/test_acronym_escalation.py -- src/acronym_escalation.py's
build-order step 2 (real single-model escalation, replacing step 1's mock).

Pure-logic tests only, no live Ollama server / DB required: a FakeClient
stands in for src.llm_client.LLMClient, and _omop_domain_for_meaning is
stubbed rather than imported for real. AST-extraction technique (same as
tests/test_tier_gate.py/test_allergy_domain_tiebreak.py) specifically to
avoid importing src.acronym_escalation directly, which pulls in
src.preprocessing -- and THAT loads a real spaCy model (en_core_sci_sm) at
import time, ~3.6s, confirmed empirically. LLMUnavailable/parse_json_response
ARE imported directly (src.llm_client itself is lightweight, ~0.3s, no heavy
NLP/ML deps at import time).

Run: python3 -m pytest tests/test_acronym_escalation.py -v
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMUnavailable, parse_json_response  # noqa: E402

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _load_pure_functions(module_filename: str, wanted: set, extra_globals: dict = None) -> dict:
    """See tests/test_offset_mapping.py's identical helper."""
    path = os.path.join(SRC_DIR, module_filename)
    tree = ast.parse(open(path, encoding="utf-8").read())

    def _is_literal_assign(node):
        if not isinstance(node, ast.Assign):
            return False
        try:
            ast.literal_eval(node.value)
            return True
        except (ValueError, SyntaxError, TypeError):
            return False

    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted) or _is_literal_assign(n)
    ]
    ns = dict(extra_globals or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), f"<{module_filename}>", "exec"), ns)
    return ns


_domain_lookups = []  # records every call, so tests can assert conn-gating behavior


def _stub_omop_domain_for_meaning(conn, meaning):
    _domain_lookups.append((conn, meaning))
    return {"emergency department": "Meas Value",
            "posterior descending artery": "Spec Anatomic Site"}.get(meaning)


AE = _load_pure_functions(
    "acronym_escalation.py",
    {"build_escalation_prompt", "build_escalation_schema", "raw_local_context",
     "escalate_one_entity", "resolve_ambiguous_acronyms", "clinical_context_for",
     "lookup_acronym_prior", "upsert_acronym_prior"},
    extra_globals={
        "LLMUnavailable": LLMUnavailable,
        "parse_json_response": parse_json_response,
        "_omop_domain_for_meaning": _stub_omop_domain_for_meaning,
        # build_clients isn't exercised by these tests (client is always
        # passed explicitly), but resolve_ambiguous_acronyms() references
        # the name at call time in the own_client branch -- stub it so the
        # exec'd module doesn't NameError merely from existing.
        "build_clients": lambda: {},
    },
)

build_escalation_prompt = AE["build_escalation_prompt"]
build_escalation_schema = AE["build_escalation_schema"]
raw_local_context = AE["raw_local_context"]
escalate_one_entity = AE["escalate_one_entity"]
resolve_ambiguous_acronyms = AE["resolve_ambiguous_acronyms"]
clinical_context_for = AE["clinical_context_for"]
lookup_acronym_prior = AE["lookup_acronym_prior"]
upsert_acronym_prior = AE["upsert_acronym_prior"]


class FakeConn:
    """Stands in for a DuckDB connection, backing acronym_priors with a
    plain in-memory dict keyed (abbreviation, clinical_context, expansion)
    -- just enough to exercise lookup_acronym_prior()/upsert_acronym_prior()'s
    own logic (highest-hit_count-wins lookup, upsert-increments-hit_count)
    without a real DB."""

    def __init__(self):
        self.rows = {}  # (abbreviation, clinical_context, expansion) -> {"omop_domain":..., "hit_count":...}
        self.ddl_calls = 0

    def sql(self, _query):
        self.ddl_calls += 1  # just proves the DDL guard fires; no real schema to create

    def execute(self, query, params):
        q = " ".join(query.split())
        if q.startswith("SELECT expansion, omop_domain FROM acronym_priors"):
            abbreviation, clinical_context = params
            candidates = [
                (key[2], v["omop_domain"], v["hit_count"])
                for key, v in self.rows.items()
                if key[0] == abbreviation and key[1] == clinical_context
            ]
            if not candidates:
                return self
            candidates.sort(key=lambda t: -t[2])
            self._fetchone_result = (candidates[0][0], candidates[0][1])
            return self
        if q.startswith("INSERT INTO acronym_priors"):
            abbreviation, clinical_context, expansion, omop_domain = params
            key = (abbreviation, clinical_context, expansion)
            if key in self.rows:
                self.rows[key]["hit_count"] += 1
            else:
                self.rows[key] = {"omop_domain": omop_domain, "hit_count": 1}
            self._fetchone_result = None
            return self
        raise AssertionError(f"FakeConn got an unexpected query: {q!r}")

    def fetchone(self):
        result = getattr(self, "_fetchone_result", None)
        self._fetchone_result = None
        return result


class FakeClient:
    """Stands in for src.llm_client.LLMClient. `response_text` is the raw
    JSON string the model "replied" with; `raises` overrides it with an
    LLMUnavailable to simulate an outage."""

    def __init__(self, response_text=None, raises=None):
        self.response_text = response_text
        self.raises = raises
        self.calls = []

    def complete(self, system_prompt, user_prompt, schema=None, max_tokens=None):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt,
                           "schema": schema})
        if self.raises:
            raise self.raises
        return {"text": self.response_text}


# A raw note text containing the literal (unexpanded) abbreviation "ED",
# with orig_start/orig_end pointing at its exact span -- mirrors the real
# extracted_entities shape (original_text/orig_start/orig_end always refer
# to the RAW note, never the Stage-1-expanded one).
_RAW_NOTE = "presented from outside hospital ED with worsening abdomen distension over past week"
_ED_START = _RAW_NOTE.index("ED")
_ED_END = _ED_START + 2


def _entity(entity_id="e1", original_text="ED", candidate_expansions=None,
           expansion_ambiguous=True, orig_start=_ED_START, orig_end=_ED_END,
           **overrides):
    base = {
        "entity_id": entity_id, "original_text": original_text,
        "expansion_ambiguous": expansion_ambiguous,
        "candidate_expansions": candidate_expansions or [
            "Ectodermal Dysplasia", "eating disorder", "emergency department",
            "erectile dysfunction"],
        "orig_start": orig_start, "orig_end": orig_end,
        # Deliberately a WRONG/stale value, to prove escalate_one_entity no
        # longer reads this field at all -- it must build its own window
        # from raw_text/orig_start/orig_end instead. See raw_local_context()'s
        # own docstring for why the stored field is untrustworthy here.
        "local_context": "STALE STAGE-1-EXPANDED TEXT, MUST NOT BE USED",
        "section_name": "Brief Hospital Course",
    }
    base.update(overrides)
    return base


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # build_escalation_prompt / build_escalation_schema
    # ======================================================================
    prompt = build_escalation_prompt(
        "RAW NOTE TEXT HERE", "ED", ["eating disorder", "emergency department"],
        local_context="presented to ___", section_name="HPI")
    check("prompt includes the raw note text", "RAW NOTE TEXT HERE" in prompt)
    check("prompt includes the abbreviation", "'ED'" in prompt)
    check("prompt lists every candidate expansion",
          "eating disorder" in prompt and "emergency department" in prompt)
    check("prompt includes the section", "HPI" in prompt)

    schema = build_escalation_schema(["eating disorder", "emergency department"])
    check("schema constrains chosen_expansion to an enum of the given candidates",
          schema["properties"]["chosen_expansion"]["enum"] ==
          ["eating disorder", "emergency department"])
    check("schema requires chosen_expansion and reasoning",
          set(schema["required"]) == {"chosen_expansion", "reasoning"})

    # ======================================================================
    # raw_local_context -- 2026-08-16 fix: must slice the RAW note around
    # orig_start/orig_end, never read the stored (Stage-1-expanded,
    # untrustworthy for an ambiguous entity) local_context column.
    # ======================================================================
    window = raw_local_context(_RAW_NOTE, _ED_START, _ED_END, window=10)
    check("raw_local_context slices around the given offsets",
          window == _RAW_NOTE[_ED_START - 10:_ED_END + 10])
    check("raw_local_context contains the literal abbreviation, not an expansion",
          "ED" in window and "Ectodermal" not in window)

    check("raw_local_context clamps to the start of the string",
          raw_local_context("short", 0, 2, window=100) == "short")
    check("raw_local_context returns None when raw_text is missing",
          raw_local_context(None, 0, 2) is None)
    check("raw_local_context returns None when offsets are missing",
          raw_local_context(_RAW_NOTE, None, None) is None)

    # ======================================================================
    # escalate_one_entity
    # ======================================================================
    ent = _entity()
    client = FakeClient(response_text=
                        '{"chosen_expansion": "emergency department", "reasoning": "context"}')
    result = escalate_one_entity(client, _RAW_NOTE, ent)
    check("valid guided response resolves to the chosen expansion",
          result is not None and result["chosen_expansion"] == "emergency department")
    check("the schema actually sent constrains to this entity's own candidates",
          client.calls[0]["schema"]["properties"]["chosen_expansion"]["enum"] ==
          ent["candidate_expansions"])
    check("the prompt sent uses the RAW-text-derived window, not the stale stored field",
          "STALE STAGE-1-EXPANDED TEXT" not in client.calls[0]["user_prompt"]
          and _RAW_NOTE[max(0, _ED_START - 50):_ED_START] in client.calls[0]["user_prompt"])

    # Model somehow returns something outside the enum (guided decoding
    # should prevent this, but never trust blindly -- same discipline every
    # other guided-decoding consumer in this codebase applies).
    client_bad = FakeClient(response_text=
                            '{"chosen_expansion": "not a real candidate", "reasoning": "x"}')
    check("a response outside the candidate list is rejected, not trusted",
          escalate_one_entity(client_bad, _RAW_NOTE, ent) is None)

    # Model/transport failure.
    client_down = FakeClient(raises=LLMUnavailable("simulated outage"))
    check("LLMUnavailable during escalation returns None, does not raise",
          escalate_one_entity(client_down, _RAW_NOTE, ent) is None)

    # Unparseable JSON.
    client_garbage = FakeClient(response_text="not json at all")
    check("unparseable response returns None, does not raise",
          escalate_one_entity(client_garbage, _RAW_NOTE, ent) is None)

    # Fewer than 2 candidates -- nothing to disambiguate, no call made at all.
    single_candidate_ent = _entity(candidate_expansions=["only one meaning"])
    client_unused = FakeClient(response_text="should never be called")
    check("fewer than 2 candidate_expansions -> no model call, returns None",
          escalate_one_entity(client_unused, "raw note", single_candidate_ent) is None
          and client_unused.calls == [])

    # ======================================================================
    # clinical_context_for / lookup_acronym_prior / upsert_acronym_prior
    # ======================================================================
    check("clinical_context_for uses section_name when present",
          clinical_context_for(_entity(section_name="Allergies")) == "Allergies")
    check("clinical_context_for falls back to 'General' when section_name is absent",
          clinical_context_for(_entity(section_name=None)) == "General")

    fake_conn = FakeConn()
    check("lookup_acronym_prior is a clean miss against an empty cache",
          lookup_acronym_prior(fake_conn, "ED", "HPI", ["eating disorder", "emergency department"])
          is None)

    upsert_acronym_prior(fake_conn, "ED", "HPI", "emergency department", "Meas Value")
    hit = lookup_acronym_prior(fake_conn, "ED", "HPI",
                               ["eating disorder", "emergency department"])
    check("a single upsert is immediately visible to lookup",
          hit == {"expansion": "emergency department", "omop_domain": "Meas Value",
                  "source": "cache"})

    check("lookup rejects a cached expansion no longer in the CURRENT candidate list "
          "(e.g. a dictionary edit removed it)",
          lookup_acronym_prior(fake_conn, "ED", "HPI", ["eating disorder"]) is None)

    # Highest hit_count wins when multiple expansions were ever confirmed
    # for the same (abbreviation, clinical_context).
    upsert_acronym_prior(fake_conn, "MS", "Neuro", "multiple sclerosis", "Condition")
    upsert_acronym_prior(fake_conn, "MS", "Neuro", "mental status", "Observation")
    upsert_acronym_prior(fake_conn, "MS", "Neuro", "mental status", "Observation")
    hit = lookup_acronym_prior(fake_conn, "MS", "Neuro", ["multiple sclerosis", "mental status"])
    check("the expansion with the higher hit_count wins the lookup",
          hit["expansion"] == "mental status")

    # conn=None -- both functions must no-op cleanly, never raise.
    check("lookup_acronym_prior(conn=None) is a clean no-op",
          lookup_acronym_prior(None, "ED", "HPI", ["emergency department"]) is None)
    upsert_acronym_prior(None, "ED", "HPI", "emergency department", "Meas Value")  # must not raise

    # ======================================================================
    # resolve_ambiguous_acronyms -- end-to-end over a small batch
    # ======================================================================
    _domain_lookups.clear()
    entities = [
        _entity(entity_id="e1", original_text="ED"),
        _entity(entity_id="e2", original_text="PDA",
               candidate_expansions=["patent ductus arteriosus", "posterior descending artery"]),
        _entity(entity_id="e3", original_text="not_ambiguous", expansion_ambiguous=False),
    ]
    client = FakeClient(response_text=
                        '{"chosen_expansion": "emergency department", "reasoning": "x"}')
    # Same FakeClient returns the SAME text for every call in this test --
    # fine for e1 (valid candidate), but for e2 "emergency department" is
    # NOT among e2's own candidates, so e2 should be rejected and absent.
    conn_for_batch = FakeConn()
    resolved = resolve_ambiguous_acronyms(entities, "raw note", "note1", conn=conn_for_batch,
                                          client=client)
    check("e1 (valid response) resolves", "e1" in resolved)
    check("e1's resolution includes the domain classification (via conn)",
          resolved["e1"]["omop_domain"] == "Meas Value" and resolved["e1"]["source"] == "mollm")
    check("e2 (response not in e2's own candidate list) is rejected, absent from result",
          "e2" not in resolved)
    check("e3 (not ambiguous) never even reaches escalation",
          "e3" not in resolved)
    check("domain lookup was passed the real conn value through",
          any(c is conn_for_batch for c, _ in _domain_lookups))

    # A cache hit must skip the model call entirely -- both for cost
    # ("don't spam the LLM for common acronyms", the original spec's own
    # framing) and so a batch that hits the cache for every entity never
    # even needs a working Ollama connection.
    cache_conn = FakeConn()
    upsert_acronym_prior(cache_conn, "ED", "Brief Hospital Course", "emergency department",
                         "Meas Value")
    client_must_not_be_called = FakeClient(response_text="SHOULD NEVER BE CALLED")
    resolved_cached = resolve_ambiguous_acronyms(
        [_entity(entity_id="e1", original_text="ED")], "raw note", "note1",
        conn=cache_conn, client=client_must_not_be_called)
    check("cache hit resolves without ever calling the model",
          resolved_cached["e1"]["source"] == "cache"
          and resolved_cached["e1"]["expansion"] == "emergency department"
          and client_must_not_be_called.calls == [])

    # conn=None -- omop_domain must be None, not attempt a lookup that would
    # crash on a None connection.
    _domain_lookups.clear()
    resolved_no_conn = resolve_ambiguous_acronyms(
        [_entity(entity_id="e1")], "raw note", "note1", conn=None,
        client=FakeClient(response_text=
                          '{"chosen_expansion": "emergency department", "reasoning": "x"}'))
    check("conn=None -> omop_domain is None, no lookup attempted",
          resolved_no_conn["e1"]["omop_domain"] is None and _domain_lookups == [])

    print(f"acronym-escalation tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_acronym_escalation():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
