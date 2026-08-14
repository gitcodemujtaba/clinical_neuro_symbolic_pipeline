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


sapbert = AutoModel.from_pretrained(MODEL_NAME)



def get_sapbert_embedding(text: str) -> list:
    """Generates a 768-dimensional SapBERT vector for a given text."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
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



