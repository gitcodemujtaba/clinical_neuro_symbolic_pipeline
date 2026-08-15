import os
import duckdb
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

def merge_embeddings():
    print("=" * 60)
    print("🔄 MERGING VECTORS INTO athena_concept")
    print("=" * 60)
    
    start_time = time.time()
    
    # We must connect without read_only to allow the UPDATE
    conn = duckdb.connect(DB_PATH)
    
    try:
        print("Executing SQL UPDATE (This may take a few minutes)...")
        
        # SQL to update the main table with vectors from the staging table
        update_sql = """
        UPDATE athena_concept
        SET embedding = concept_embeddings.embedding
        FROM concept_embeddings
        WHERE athena_concept.concept_id = concept_embeddings.concept_id;
        """
        
        conn.sql(update_sql)
        
        # Verify the update worked
        result = conn.sql("""
        SELECT COUNT(*) AS concepts_with_embeddings 
        FROM athena_concept 
        WHERE embedding IS NOT NULL
        """).fetchone()[0]
        
        print(f"\n✅ MERGE COMPLETE: {result:,} concepts now have embeddings.")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
    finally:
        conn.close()
        
    print(f"Time elapsed: {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    merge_embeddings()