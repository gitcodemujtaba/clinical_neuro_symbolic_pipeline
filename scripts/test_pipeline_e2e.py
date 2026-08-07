import os
import sys
import duckdb
import csv
import json

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
sys.path.append(PROJECT_DIR)

from src.preprocessing import process_and_store_note
from src.entity_extraction import extract_and_store_entities
from src.normalization import process_and_normalize_entities

DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv")

def run_e2e():
    print("=" * 80)
    print("🚀 RUNNING END-TO-END PIPELINE (STAGE 1 -> STAGE 2a -> STAGE 2b)")
    print("=" * 80)

    conn = duckdb.connect(DB_PATH)
    
    try:
        # Get just the first note from the dataset
        with open(NOTES_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            first_row = next(reader)
            
        note_id = first_row.get('note_id', 'test_note')
        raw_text = first_row.get('text', '')

        print(f"\n[1/3] Preprocessing (scispaCy + Dictionary) on {note_id}...")
        expanded_text, provenance = process_and_store_note(note_id, raw_text, conn)

        print("[2/3] Extracting Entities (GLiNER)...")
        # Grabbing just a slice of text to speed up this test run (first 1000 chars)
        entities = extract_and_store_entities(note_id, expanded_text[:1000], raw_text, conn)
        
        # Deduplicate identical entity text for a cleaner test output
        unique_entities = {e[3]: e for e in entities}.values()

        print(f"[3/3] Normalizing {len(unique_entities)} Entities (OMOP + SapBERT)...")
        normalized = process_and_normalize_entities(list(unique_entities), conn)

        print("\n" + "=" * 80)
        print("📊 FINAL NORMALIZED ENTITIES")
        print("=" * 80)
        
        print(f"{'DOCTOR WROTE':<20} | {'GLiNER LABEL':<12} | {'OMOP MAPPING (CONCEPT NAME)':<30} | {'TIER':<15}")
        print("-" * 80)
        
        for n in normalized:
            orig = n['original_text'].replace('\n', ' ')[:18]
            label = n['gliner_label']
            omop = n['omop_concept_name'][:28] if n['omop_concept_name'] else "None"
            tier = n['match_tier']
            print(f"{orig:<20} | {label:<12} | {omop:<30} | {tier:<15}")
            
    finally:
        conn.close()

if __name__ == "__main__":
    run_e2e()