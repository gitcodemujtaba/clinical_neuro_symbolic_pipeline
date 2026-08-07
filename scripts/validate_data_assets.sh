#!/bin/bash
# ==============================================================================
# Data Asset Validator for Clinical Neuro-Symbolic Pipeline
# ==============================================================================

DATA_DIR="/home/ec2-user/clinical_neuro_symbolic_pipeline/data"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo " 🔍 VALIDATING DATA DIRECTORY STRUCTURE & FILES"
echo "======================================================================"

# 1. Check Raw Notes (MIMIC-IV)
echo -e "\n${YELLOW}1. Checking Raw Notes (data/raw_notes/)...${NC}"
for file in "discharge.csv" "discharge_detail.csv" "radiology.csv" "radiology_detail.csv"; do
    if [ -f "$DATA_DIR/raw_notes/$file" ]; then
        echo -e "${GREEN}  ✅ Found: $file${NC}"
    else
        echo -e "${RED}  ❌ Missing: $file${NC}"
    fi
done

# 2. Check OMOP Vocabularies
echo -e "\n${YELLOW}2. Checking OMOP Vocabularies (data/athena_omop/)...${NC}"
for vocab in "CONCEPT" "CONCEPT_SYNONYM" "CONCEPT_RELATIONSHIP" "CONCEPT_ANCESTOR"; do
    # Check for either .csv or .tsv extensions
    if ls "$DATA_DIR/athena_omop/"$vocab.* >/dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Found: $vocab${NC}"
    else
        echo -e "${RED}  ❌ Missing: $vocab (.csv or .tsv)${NC}"
    fi
done

# 3. Check Medical Abbreviations
echo -e "\n${YELLOW}3. Checking Medical Abbreviations (data/medical_abbreviations/)...${NC}"
abbrev_count=$(ls -1 "$DATA_DIR/medical_abbreviations/"*.csv 2>/dev/null | wc -l)
if [ "$abbrev_count" -gt 0 ]; then
    echo -e "${GREEN}  ✅ Found $abbrev_count CSV files (Expected ~17).${NC}"
else
    echo -e "${RED}  ❌ Missing: No CSV files found in medical_abbreviations/.${NC}"
fi

# 4. Check SNOMED CT Graph Files
echo -e "\n${YELLOW}4. Checking SNOMED CT Ontology (data/snomed_kg/)...${NC}"
if ls "$DATA_DIR/snomed_kg/"*Concept_Snapshot*.txt >/dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Found: Concept Snapshot file${NC}"
else
    echo -e "${RED}  ❌ Missing: SNOMED Concept Snapshot (*Concept_Snapshot*.txt)${NC}"
fi

if ls "$DATA_DIR/snomed_kg/"*Relationship_Snapshot*.txt >/dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Found: Relationship Snapshot file${NC}"
else
    echo -e "${RED}  ❌ Missing: SNOMED Relationship Snapshot (*Relationship_Snapshot*.txt)${NC}"
fi

# 5. Check Guidelines
echo -e "\n${YELLOW}5. Checking Clinical Guidelines (data/guidelines/)...${NC}"
guideline_count=$(ls -1 "$DATA_DIR/guidelines/"*.csv "$DATA_DIR/guidelines/"*.json 2>/dev/null | wc -l)
if [ "$guideline_count" -gt 0 ]; then
    echo -e "${GREEN}  ✅ Found guideline files (.csv or .json).${NC}"
else
    echo -e "${YELLOW}  ⚠️ Warning: No guideline files found. (Optional for initial DuckDB tests, required for Memgraph Stage 3)${NC}"
fi

echo -e "\n======================================================================"
echo "VALIDATION COMPLETE."
echo "======================================================================"