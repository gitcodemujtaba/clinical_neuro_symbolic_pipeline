"""scripts/test_hyphen_preprocessing_hypothesis.py -- EMPIRICAL TEST, not a
pipeline change.

WHY THIS EXISTS. Two competing claims were made in conversation about
replacing "-" with " " in raw note text BEFORE GLiNER extraction:
  (a) it would let GLiNER emit a clean "GLUCOSE" span instead of
      "GLUCOSE-131", helping Tier 1 exact-match at normalization time.
  (b) it would also corrupt hyphens that are NOT lab name-value delimiters
      (COVID-19, CK-MB, darunavir-cobicistat, T2-L1) since preprocessing runs
      on raw text before any entity/label exists to gate the substitution.
Neither claim was tested against the real model -- this script runs both
sides of the hypothesis through the ACTUAL GLiNER-BioMed model and the ACTUAL
normalize_entity() Tier 1/2/3 lookup, side by side, on ORIGINAL vs.
HYPHEN-REPLACED text, so the comparison is measured rather than argued.

Run on EC2 (needs the real model + DB):
    cd ~/clinical_neuro_symbolic_pipeline/code
    source ~/.venv/bin/activate
    python3 scripts/test_hyphen_preprocessing_hypothesis.py

This does not write to any table and does not modify src/entity_extraction.py
or src/normalization.py -- it imports the same model name, threshold, label
set, and normalize_entity() function real Stage 2a/2b use, so the comparison
is apples-to-apples with production behavior, not a simplified stand-in.
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Same model, threshold, labels, flat_ner as src/entity_extraction.py -- see
# that module's GLINER_MODEL_NAME / EXTRACTION_THRESHOLD / CLINICAL_LABELS /
# FLAT_NER for why these specific values.
GLINER_MODEL_NAME = "Ihor/gliner-biomed-large-v1.0"
EXTRACTION_THRESHOLD = 0.35  # SUBTHRESHOLD_FLOOR, matches production's floor
CLINICAL_LABELS = ["Condition", "Symptom", "Medication", "Procedure", "Anatomy", "Lab Test"]
FLAT_NER = True

# ---------------------------------------------------------------------------
# TEST SENTENCES. Two groups:
#   GROUP A -- the case the hypothesis is meant to help (lab panel NAME-VALUE
#       lines). Sentence 1 is quoted verbatim from a real MIMIC note, cited in
#       assertion.py's is_structured_result() docstring. Sentence 2 mirrors
#       tonight's live batch run (repeated GLUCOSE fingerstick readings).
#   GROUP B -- counter-examples: real hyphenated entities seen in TONIGHT'S
#       own Stage 3 batch run, plus the COVID-19 case assertion.py's own
#       docstring already names as the reason its NAME-VALUE signal is
#       label-gated rather than applied blindly.
# ---------------------------------------------------------------------------
TEST_SENTENCES = {
    "A1_real_BMP_panel_line": (
        "___ 10:25PM   GLUCOSE-109* UREA N-25* CREAT-0.3* SODIUM-138 "
        "POTASSIUM-3.4 CHLORIDE-105 TOTAL carbon dioxide-27 ANION GAP-9"
    ),
    "A2_fingerstick_series": (
        "Fingerstick glucose readings recorded overnight: GLUCOSE-131 "
        "GLUCOSE-137 GLUCOSE-122 GLUCOSE-119 GLUCOSE-121."
    ),
    "B1_ck_mb_two_different_hyphens": (
        "Cardiac enzymes were sent: CK-MB-12 and Troponin-0.02, both within "
        "normal limits."
    ),
    "B2_combination_drug_name": (
        "Patient was continued on darunavir-cobicistat and dolutegravir for "
        "HIV management."
    ),
    "B3_covid_negation": (
        "Patient denies COVID-19 and denies any recent sick contacts."
    ),
    "B4_vertebral_range": (
        "MRI of the spine revealed disc herniation at T2-L1 and L4-5 without "
        "cord compression."
    ),
}


def run_gliner_both_ways(model, label, text):
    """Runs GLiNER on `text` as-is, and again with every '-' replaced by ' ',
    returns (original_entities, replaced_entities, replaced_text)."""
    orig_entities = model.predict_entities(
        text, CLINICAL_LABELS, threshold=EXTRACTION_THRESHOLD, flat_ner=FLAT_NER
    )
    replaced_text = text.replace("-", " ")
    replaced_entities = model.predict_entities(
        replaced_text, CLINICAL_LABELS, threshold=EXTRACTION_THRESHOLD, flat_ner=FLAT_NER
    )
    return orig_entities, replaced_entities, replaced_text


def fmt_entities(entities):
    if not entities:
        return "    (none extracted)"
    lines = []
    for e in sorted(entities, key=lambda x: x["start"]):
        lines.append(
            f"    [{e['start']:>4}:{e['end']:>4}] {e['label']:<10} "
            f"{e['text']!r:<35} score={e['score']:.3f}"
        )
    return "\n".join(lines)


def try_normalize(conn, normalize_entity, entities, label_filter="Lab Test"):
    """For each Lab-Test-labeled entity, runs the REAL normalize_entity() and
    reports match_tier -- this is the part that actually determines whether
    Stage 2b would treat the span as resolved."""
    out = []
    for e in entities:
        if e["label"] != label_filter:
            continue
        mapping = normalize_entity(e["text"], conn, gliner_label=e["label"])
        out.append((e["text"], mapping["match_tier"], mapping["concept_name"]))
    return out


def main():
    print("Loading GLiNER-BioMed model (same checkpoint entity_extraction.py uses)...")
    from gliner import GLiNER
    model = GLiNER.from_pretrained(GLINER_MODEL_NAME)

    have_db = True
    try:
        import duckdb
        from src.normalization import normalize_entity
        db_path = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
        conn = duckdb.connect(db_path, read_only=True)
        print(f"Connected read-only to {db_path}\n")
    except Exception as exc:
        have_db = False
        print(f"(DB/normalize_entity unavailable -- {exc}; extraction-only comparison)\n")

    print("=" * 90)
    for name, text in TEST_SENTENCES.items():
        print(f"\n### {name}")
        print(f"ORIGINAL:  {text!r}")
        orig_e, repl_e, replaced_text = run_gliner_both_ways(model, name, text)
        print(f"REPLACED:  {replaced_text!r}")

        print("\n  -- GLiNER spans on ORIGINAL text --")
        print(fmt_entities(orig_e))
        print("\n  -- GLiNER spans on HYPHEN-REPLACED text --")
        print(fmt_entities(repl_e))

        if have_db:
            orig_norm = try_normalize(conn, normalize_entity, orig_e)
            repl_norm = try_normalize(conn, normalize_entity, repl_e)
            if orig_norm or repl_norm:
                print("\n  -- Lab Test normalization (match_tier) ORIGINAL vs REPLACED --")
                print(f"    ORIGINAL: {orig_norm}")
                print(f"    REPLACED: {repl_norm}")
        print("-" * 90)

    print("\nDone. Compare span boundaries/labels and match_tier columns above for "
          "A1/A2 (does replacement help) vs B1-B4 (does replacement damage entities "
          "that were fine as hyphenated text).")


if __name__ == "__main__":
    main()
