import os
import sys
import duckdb
import csv

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
sys.path.append(PROJECT_DIR)

from src.preprocessing import process_and_store_note
from src.entity_extraction import extract_and_store_entities

DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv")

def run_test():
    print("=" * 60)
    print("🧠 TESTING STAGE 2a: GLiNER NER + PROVENANCE RECONCILIATION")
    print("=" * 60)

    conn = duckdb.connect(DB_PATH)
    
    try:
        # Get just the first note from the dataset
        with open(NOTES_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            first_row = next(reader)
            
        note_id = first_row.get('note_id', 'test_note')
        raw_text = first_row.get('text', '')

        print(f"1. Running Stage 1 (Preprocessing) on Note {note_id}...")
        # is_test=True: flags rows in note_expansions/extracted_entities so
        # this smoke test's output can be purged before production.
        expanded_text, provenance = process_and_store_note(note_id, raw_text, conn, is_test=True)

        print("2. Running Stage 2a (GLiNER Extraction)...")
        entities = extract_and_store_entities(note_id, expanded_text, raw_text, conn, is_test=True)

        print("\n" + "=" * 60)
        print(f"✅ Extracted {len(entities)} clinical entities.")
        print("=" * 60)
        
        # Display the first 10 entities to verify the "Time Machine"
        print(f"{'LABEL':<12} | {'WHAT AI SAW (Expanded)':<30} | {'WHAT DOCTOR WROTE (Original)':<30}")
        print("-" * 75)
        
        # 2026-08-08: extract_and_store_entities() now returns dicts rather
        # than positional tuples (see src/entity_extraction.py's docstring),
        # so these were e[5]/e[1]/e[2]/e[3] before.
        entities.sort(key=lambda x: x["orig_start"])

        for e in entities[:15]:
            label = e["entity_label"]
            exp_text = e["expanded_text"].replace('\n', ' ')
            orig_text = e["original_text"].replace('\n', ' ')
            assertion = e["assertion_status"]
            if e["experiencer"] != "PATIENT":
                assertion += f"/{e['experiencer']}"
            if e["temporality"] != "CURRENT":
                assertion += f"/{e['temporality']}"
            print(f"{label:<12} | {exp_text:<30} | {orig_text:<30} | {assertion}")
            
    finally:
        conn.close()

if __name__ == "__main__":
    run_test()