import os
import json
import duckdb
import spacy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    nlp = spacy.load("en_core_sci_sm")
except Exception:
    import en_core_sci_sm
    nlp = en_core_sci_sm.load()

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

def load_abbreviations_dict(conn) -> dict:
    tables = [t[0] for t in conn.sql("SHOW TABLES").fetchall()]
    if "kg2a_abbreviations" in tables:
        rows = conn.sql("SELECT abbreviation, meaning FROM kg2a_abbreviations").fetchall()
        return {abbr.lower(): meaning for abbr, meaning in rows if abbr}
    return {}

def expand_text_and_track_offsets(text: str, abbrev_dict: dict):
    doc = nlp(text)
    expanded_text = text
    expansions_log = []
    offset_shift = 0

    for token in doc:
        # FIXED: Skip basic English stopwords, punctuation, and spaces
        if token.is_stop or token.is_punct or token.is_space:
            continue

        token_lower = token.text.lower()
        if token_lower in abbrev_dict:
            expansion = abbrev_dict[token_lower]
            orig_start = token.idx
            orig_end = token.idx + len(token.text)
            
            exp_start = orig_start + offset_shift
            exp_end = exp_start + len(expansion)

            expansions_log.append({
                "abbrev": token.text,
                "expansion": expansion,
                "orig_start": orig_start,
                "orig_end": orig_end,
                "exp_start": exp_start,
                "exp_end": exp_end
            })

            expanded_text = (
                expanded_text[:exp_start] + 
                expansion + 
                expanded_text[exp_start + len(token.text):]
            )

            offset_shift += (len(expansion) - len(token.text))

    return expanded_text, expansions_log

def process_and_store_note(note_id: str, raw_text: str, conn):
    abbrev_dict = load_abbreviations_dict(conn)
    expanded_text, expansions_log = expand_text_and_track_offsets(raw_text, abbrev_dict)

    provenance_data = {
        "note_id": note_id,
        "expansions_count": len(expansions_log),
        "expansions": expansions_log
    }

    conn.sql("""
    CREATE TABLE IF NOT EXISTS note_expansions (
        note_id VARCHAR PRIMARY KEY,
        provenance JSON
    );
    """)

    conn.sql("""
    INSERT INTO note_expansions (note_id, provenance)
    VALUES (?, ?)
    ON CONFLICT (note_id) DO UPDATE SET provenance = EXCLUDED.provenance;
    """, params=[note_id, json.dumps(provenance_data)])

    return expanded_text, provenance_data