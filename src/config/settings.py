# src/config/settings.py
import os
from dotenv import load_dotenv

# Tải các biến từ file .env ở thư mục gốc của dự án
load_dotenv()


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agentic_rag_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))

RAG_CONTENT_CHUNK_TABLE = os.getenv("RAG_CONTENT_CHUNK_TABLE", "content_chunks")
RAG_VOCAB_TABLE = os.getenv("RAG_VOCAB_TABLE", "vocabulary_items")
RAG_KANJI_TABLE = os.getenv("RAG_KANJI_TABLE", "kanji_items")


R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

CMS_API_BASE_URL = os.getenv("CMS_API_BASE_URL", "http://localhost:8080/api")