import os
import duckdb
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# Load SapBERT for Tier 3 Semantic Grounding
print("Loading SapBERT model for vector normalization... (this may take a moment)")
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sapbert = AutoModel.from_pretrained(MODEL_NAME)

def get_sapbert_embedding(text: str) -> list:
    """Generates a 768-dimensional SapBERT vector for a given text."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = sapbert(**tokens)
        # Use the [CLS] token representation as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding

def normalize_entity(entity_text: str, conn) -> dict:
    """
    Attempts to map an entity to a standard OMOP Concept ID using a 3-Tier approach.
    Returns the mapping dictionary containing the tier used, concept details, and similarity score.
    """
    search_text = entity_text.lower().strip()

    # ==========================================
    # TIER 1: Exact Lexical Match
    # ==========================================
    tier1_query = """
    SELECT concept_id, concept_name, domain_id, vocabulary_id 
    FROM athena_concept 
    WHERE lower(concept_name) = ? 
    AND standard_concept = 'S' 
    LIMIT 1;
    """
    result = conn.sql(tier1_query, params=[search_text]).fetchone()
    if result:
        return {
            "match_tier": "1 (Exact)", 
            "concept_id": result[0], 
            "concept_name": result[1], 
            "domain_id": result[2], 
            "vocab": result[3], 
            "score": 1.0000
        }

    # ==========================================
    # TIER 2: Synonym Lexical Match
    # ==========================================
    tier2_query = """
    SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id 
    FROM athena_concept_synonym s
    JOIN athena_concept c ON s.concept_id = c.concept_id
    WHERE lower(s.concept_synonym_name) = ? 
    AND c.standard_concept = 'S' 
    LIMIT 1;
    """
    result = conn.sql(tier2_query, params=[search_text]).fetchone()
    if result:
        return {
            "match_tier": "2 (Synonym)", 
            "concept_id": result[0], 
            "concept_name": result[1], 
            "domain_id": result[2], 
            "vocab": result[3], 
            "score": 1.0000
        }

    # ==========================================
    # TIER 3: Semantic Vector Match (SapBERT)
    # ==========================================
    vector = get_sapbert_embedding(entity_text)
    
    tier3_query = """
    SELECT concept_id, concept_name, domain_id, vocabulary_id, 
           list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
    FROM athena_concept 
    WHERE embedding IS NOT NULL 
    AND standard_concept = 'S'
    ORDER BY similarity DESC 
    LIMIT 1;
    """
    result = conn.sql(tier3_query, params=[vector]).fetchone()
    
    if result:
        return {
            "match_tier": "3 (Semantic)", 
            "concept_id": result[0], 
            "concept_name": result[1], 
            "domain_id": result[2], 
            "vocab": result[3], 
            "score": round(result[4], 4)
        }

    # Fallback (Should rarely hit if SapBERT vectors are loaded)
    return {"match_tier": "0 (Failed)", "concept_id": None, "concept_name": "Unmapped", "domain_id": None, "vocab": None, "score": 0.0}

def process_and_normalize_entities(extracted_entities: list, conn) -> list:
    """Takes a list of GLiNER extracted entities and normalizes them."""
    normalized_results = []
    for ent in extracted_entities:
        orig_text = ent[3] 
        expanded_text = ent[2]
        
        # FIXED: We now search the OMOP vocabulary using the expanded_text
        # instead of the shorthand abbreviation, which prevents collision errors.
        mapping = normalize_entity(expanded_text, conn)
        
        # Merge extraction metadata with mapping metadata
        normalized_results.append({
            "original_text": orig_text,
            "expanded_text": expanded_text,
            "gliner_label": ent[1],
            "gliner_confidence": round(ent[4], 4),
            "omop_concept_id": mapping["concept_id"],
            "omop_concept_name": mapping["concept_name"],
            "omop_domain": mapping["domain_id"],
            "omop_vocab": mapping["vocab"],
            "match_tier": mapping["match_tier"],
            "similarity_score": mapping["score"]
        })
    return normalized_results