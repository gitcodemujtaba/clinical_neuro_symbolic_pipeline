#!/bin/bash
# ==============================================================================
# KG-AL Pipeline: Post-Reboot Startup & Infrastructure Verification
# ==============================================================================

set -e # Exit immediately on error

PROJECT_DIR="/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_DIR="$PROJECT_DIR/db"

echo "======================================================================"
echo "          WAKING UP NEURO-SYMBOLIC KG-AL PIPELINE SERVICES            "
echo "======================================================================"

echo "1. Verifying Project Directory Structure..."
if [ -d "$PROJECT_DIR" ]; then
    echo "   ✅ Found master project directory: $PROJECT_DIR"
else
    echo "   ❌ ERROR: Master directory missing: $PROJECT_DIR"
    exit 1
fi

echo "2. Ensuring Docker daemon is active..."
sudo systemctl enable docker --now

echo "3. Starting Graph Database Containers (Neo4j Lexicon & Memgraph Ledger)..."
# Replace container names below if your docker containers have specific names
sudo docker start neo4j_snomed memgraph_kg3_ledger || true

echo "4. Verifying KG 2 Lexical Store (DuckDB)..."
if [ -f "$DB_DIR/kg2_lexical_store.duckdb" ]; then
    echo "   ✅ DuckDB instance online: $DB_DIR/kg2_lexical_store.duckdb"
else
    echo "   ⚠️ DuckDB missing! You may need to run: python3 scripts/import_athena.py"
fi

echo "5. Allowing Graph Databases 5 seconds to bind ports..."
sleep 5

echo "6. Checking Graph Database Connectivity..."
python3 scripts/profile_databases.py

echo "7. Checking vLLM Model Serving Endpoints..."
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "   ✅ MedGemma vLLM Service (Port 8000): ONLINE"
else
    echo "   ❌ MedGemma vLLM Service (Port 8000): OFFLINE"
fi

if curl -s http://localhost:8001/v1/models > /dev/null; then
    echo "   ✅ OpenBioLLM vLLM Service (Port 8001): ONLINE"
else
    echo "   ❌ OpenBioLLM vLLM Service (Port 8001): OFFLINE"
fi

echo "======================================================================"
echo "STARTUP SEQUENCE COMPLETE."
echo "======================================================================"