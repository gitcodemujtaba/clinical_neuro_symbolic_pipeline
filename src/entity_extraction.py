import os
import json
import duckdb
from gliner import GLiNER
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# Load GLiNER model (using the standard medium model for balance of speed and accuracy)
print("Loading GLiNER model... (this may take a moment on the first run)")
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Define the clinical domains we want to extract
CLINICAL_LABELS = [
    "Condition", 
    "Symptom", 
    "Medication", 
    "Procedure", 
    "Anatomy", 
    "Lab Test"
]

def map_offsets_to_original(exp_start_idx, exp_end_idx, expansions):
    """
    The 'Time Machine': Maps character offsets from the expanded text back 
    to the exact character offsets in the original raw clinical note.
    """
    orig_start = exp_start_idx
    orig_end = exp_end_idx
    
    for exp in expansions:
        shift = (exp["exp_end"] - exp["exp_start"]) - (exp["orig_end"] - exp["orig_start"])
        
        # Adjust start index
        if exp["exp_end"] <= exp_start_idx:
            orig_start -= shift
        elif exp["exp_start"] <= exp_start_idx < exp["exp_end"]:
            orig_start = exp["orig_start"]
            
        # Adjust end index
        if exp["exp_end"] <= exp_end_idx:
            orig_end -= shift
        elif exp["exp_start"] < exp_end_idx <= exp["exp_end"]:
            orig_end = exp["orig_end"]
            
    return orig_start, orig_end

def extract_and_store_entities(note_id: str, expanded_text: str, raw_text: str, conn):
    """Runs GLiNER on expanded text, reconciles offsets, and stores in DuckDB."""
    
    # 1. Fetch provenance log from Stage 1
    prov_row = conn.sql(f"SELECT provenance FROM note_expansions WHERE note_id = '{note_id}'").fetchone()
    if not prov_row:
        print(f"⚠️ No provenance found for {note_id}. Did it pass through Stage 1?")
        return []
    
    provenance = json.loads(prov_row[0])
    expansions = provenance.get("expansions", [])
    
    # 2. Run Zero-Shot Extraction
    # threshold=0.5 limits low-confidence guesses
    raw_entities = model.predict_entities(expanded_text, CLINICAL_LABELS, threshold=0.5)
    
    # 3. Create the extracted entities table if it doesn't exist
    conn.sql("""
    CREATE TABLE IF NOT EXISTS extracted_entities (
        note_id VARCHAR,
        entity_label VARCHAR,
        expanded_text VARCHAR,
        original_text VARCHAR,
        confidence FLOAT,
        orig_start INT,
        orig_end INT,
        exp_start INT,
        exp_end INT,
        UNIQUE(note_id, orig_start, orig_end, entity_label)
    );
    """)
    
    processed_entities = []
    
    # 4. Reconcile offsets and store
    for ent in raw_entities:
        exp_start = ent["start"]
        exp_end = ent["end"]
        label = ent["label"]
        confidence = ent["score"]
        exp_text = ent["text"]
        
        # Apply the time machine math
        orig_start, orig_end = map_offsets_to_original(exp_start, exp_end, expansions)
        
        # Extract exactly what the doctor originally wrote based on the reconciled offsets
        orig_text = raw_text[orig_start:orig_end]
        
        processed_entities.append((
            note_id, label, exp_text, orig_text, confidence, 
            orig_start, orig_end, exp_start, exp_end
        ))
    
    # Batch insert into DuckDB
    if processed_entities:
        conn.executemany("""
        INSERT INTO extracted_entities 
        (note_id, entity_label, expanded_text, original_text, confidence, orig_start, orig_end, exp_start, exp_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING;
        """, processed_entities)
        
    return processed_entities