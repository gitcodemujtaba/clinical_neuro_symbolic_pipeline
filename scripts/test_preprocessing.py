import os
import csv
import json
import sys
import duckdb

# Ensure Python can find the 'src' module
PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
sys.path.append(PROJECT_DIR)

from src.preprocessing import process_and_store_note

DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv")

def test_pipeline_on_mimic():
    print("=" * 60)
    print("🚀 TESTING STAGE 1 PREPROCESSING ON MIMIC-IV NOTES")
    print("=" * 60)

    if not os.path.exists(NOTES_PATH):
        print(f"❌ Error: Could not find {NOTES_PATH}")
        return

    # Connect to DuckDB
    conn = duckdb.connect(DB_PATH)
    
    try:
        # Open the massive CSV file in streaming mode
        with open(NOTES_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            count = 0
            for row in reader:
                if count >= 5:
                    break
                
                # MIMIC-IV typically uses 'note_id' and 'text' headers
                note_id = row.get('note_id', f"unknown_note_{count}")
                raw_text = row.get('text', '')
                
                if not raw_text:
                    continue
                
                print(f"\n--- Processing Note {count + 1}: {note_id} ---")
                
                # Execute Stage 1 preprocessing
                expanded_text, provenance = process_and_store_note(note_id, raw_text, conn)
                
                # Find how many expansions occurred
                exp_count = provenance.get("expansions_count", 0)
                print(f"✅ Found and expanded {exp_count} abbreviations.")
                
                if exp_count > 0:
                    # Print a sample of the logged expansions
                    sample_logs = provenance['expansions'][:3]
                    print("Sample of Logged Expansions (Character Offsets):")
                    print(json.dumps(sample_logs, indent=2))
                
                count += 1
        
        # Verify database storage was successful
        print("\n" + "=" * 60)
        print("🗄️ VERIFYING DUCKDB STORAGE (note_expansions table)")
        print("=" * 60)
        
        verify_query = """
        SELECT 
            note_id, 
            json_extract(provenance, '$.expansions_count') AS expansion_count 
        FROM note_expansions 
        LIMIT 5
        """
        conn.sql(verify_query).show()

    except Exception as e:
        print(f"❌ Error during processing: {e}")
    finally:
        conn.close()
        print("\nConnection closed. Test complete.")

if __name__ == "__main__":
    test_pipeline_on_mimic()