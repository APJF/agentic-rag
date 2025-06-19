# src/core/embedding.py
from sentence_transformers import SentenceTransformer
from src.config import settings

print(f"Loading embedding model for RAG core: {settings.EMBEDDING_MODEL_NAME}")
_embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
print("Embedding model for RAG core loaded.")

def get_embedding_model():
    return _embedding_model

def encode_text(text: str) -> list[float]:
    return _embedding_model.encode(text).tolist()