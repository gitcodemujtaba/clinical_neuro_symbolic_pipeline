"""src/normalization/sapbert_model.py — SapBERT model loading + embedding/cosine helpers (split from src/normalization.py, 2026-08-14)."""
import math
import warnings
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

print("Loading SapBERT model for vector normalization... (this may take a moment)")

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


SAPBERT_POOLING = "cls"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2026-08-18: same GPU-placement gap as GLiNER (src/entity_extraction.py's
# own comment has the full rationale) -- AutoModel.from_pretrained() loads to
# CPU by default. SapBERT is called far more often than GLiNER itself (every
# Tier 3 candidate/entity needs an embedding, and CONTEXTUAL_CANDIDATES_ENABLED
# being default-on now means more multi-candidate ranking calls too), so this
# is likely the single biggest normalization-time win. Falls back to CPU on
# any failure rather than crashing normalization entirely.
_SAPBERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    sapbert = AutoModel.from_pretrained(MODEL_NAME).to(_SAPBERT_DEVICE)
except Exception:
    _SAPBERT_DEVICE = "cpu"
    sapbert = AutoModel.from_pretrained(MODEL_NAME).to("cpu")



def get_sapbert_embedding(text: str) -> list:
    """Generates a 768-dimensional SapBERT vector for a given text."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    tokens = {k: v.to(_SAPBERT_DEVICE) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = sapbert(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding




def _cosine(a, b):
    """Plain cosine similarity between two equal-length vectors. Returns 0.0
    on a zero vector rather than raising -- a degenerate embedding must cost a
    candidate its ranking bonus, not take down normalization."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)



