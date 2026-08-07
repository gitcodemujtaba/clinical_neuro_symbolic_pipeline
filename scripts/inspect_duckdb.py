import os
import duckdb

# Define the absolute path to your database
PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

def inspect_db():
    print("=" * 60)
    print(f"🔍 INSPECTING DUCKDB: {DB_PATH}")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"❌ ERROR: Database file not found at {DB_PATH}")
        return

    # Connect in read-only mode to prevent accidental locks or overwrites
    conn = duckdb.connect(DB_PATH, read_only=True)

    try:
        # 1. List all tables
        print("\n--- 1. TABLES IN DATABASE ---")
        conn.sql("SHOW TABLES").show()

        # 2. Show the schema for athena_concept
        print("\n--- 2. SCHEMA: athena_concept ---")
        conn.sql("DESCRIBE athena_concept").show()

        # 3. Sample Data & Embedding Verification
        print("\n--- 3. SAMPLE DATA (Top 5 Concepts) ---")
        sample_query = """
        SELECT 
            concept_id, 
            concept_name, 
            domain_id, 
            vocabulary_id,
            CASE WHEN embedding IS NOT NULL THEN '✅ Vector Present' ELSE '❌ No Vector' END AS embedding_status
        FROM athena_concept 
        LIMIT 5
        """
        conn.sql(sample_query).show()

        # 4. Summary Statistics for athena_concept
        print("\n--- 4. SUMMARY STATISTICS: athena_concept ---")
        conn.sql("SELECT COUNT(*) AS total_concepts FROM athena_concept").show()
        
        conn.sql("""
        SELECT COUNT(*) AS concepts_with_embeddings 
        FROM athena_concept 
        WHERE embedding IS NOT NULL
        """).show()

        # 5. Inspect concept_embeddings table
        print("\n" + "=" * 60)
        print("🔍 INSPECTING 'concept_embeddings' TABLE")
        print("=" * 60)
        
        print("\n--- SCHEMA: concept_embeddings ---")
        conn.sql("DESCRIBE concept_embeddings").show()
        
        print("\n--- TOTAL VECTORS IN concept_embeddings ---")
        conn.sql("SELECT COUNT(*) AS total_staged_vectors FROM concept_embeddings").show()
        
        print("\n--- SAMPLE DATA: concept_embeddings ---")
        conn.sql("SELECT * FROM concept_embeddings LIMIT 3").show()

    except Exception as e:
        print(f"❌ Query Error: {e}")
    finally:
        conn.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    inspect_db()