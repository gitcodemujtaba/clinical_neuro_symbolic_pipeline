import os
from neo4j import GraphDatabase

# Database Configurations pulled from environment variables with fallbacks
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "secure_password_here"))

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7688")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

def profile_db(name, uri, auth):
    print("=" * 50)
    print(f" Profiling {name}")
    print("=" * 50)
    
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        with driver.session() as session:
            # 1. Total Nodes & Labels
            print("--- Node Counts by Label ---")
            nodes_query = """
            MATCH (n) 
            RETURN coalesce(labels(n)[0], 'Unlabeled') AS Label, count(n) AS Count 
            ORDER BY Count DESC
            """
            for record in session.run(nodes_query):
                print(f" - {record['Label']}: {record['Count']:,}")
            
            # 2. Total Relationships
            print("\n--- Relationship Counts by Type ---")
            rels_query = """
            MATCH ()-[r]->() 
            RETURN type(r) AS Type, count(r) AS Count 
            ORDER BY Count DESC
            """
            for record in session.run(rels_query):
                print(f" - {record['Type']}: {record['Count']:,}")
            
            # 3. Database-Specific Sanity Checks
            if "Neo4j" in name:
                print("\n--- Sanity Check: SNOMED FSN Sample ---")
                fsn_query = """
                MATCH (c:SnomedConcept) 
                WHERE c.fullySpecifiedName IS NOT NULL 
                RETURN c.id AS ID, c.fullySpecifiedName AS FSN 
                LIMIT 3
                """
                for record in session.run(fsn_query):
                    print(f" - {record['ID']}: {record['FSN']}")
            
            elif "Memgraph" in name:
                print("\n--- Sanity Check: SNOMED Cross-Link Sample ---")
                link_query = """
                MATCH (n) 
                WHERE n.snomedCode IS NOT NULL 
                RETURN coalesce(labels(n)[0], 'Node') AS Label, n.name AS Name, n.snomedCode AS SnomedCode 
                LIMIT 3
                """
                for record in session.run(link_query):
                    print(f" - [{record['Label']}] {record['Name']} -> SNOMED: {record['SnomedCode']}")
                    
        driver.close()
    except Exception as e:
        print(f"Failed to connect or query {name}: {e}")
    print("\n")

if __name__ == "__main__":
    profile_db("DB 1: SNOMED CT Lexicon (Neo4j)", NEO4J_URI, NEO4J_AUTH)
    profile_db("DB 2: Clinical Rules Engine & Provenance (Memgraph)", MEMGRAPH_URI, MEMGRAPH_AUTH)