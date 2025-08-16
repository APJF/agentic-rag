# src/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agentic_rag_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_ORG = os.getenv("OPENAI_ORG")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
RAG_CONTENT_CHUNK_TABLE = os.getenv("RAG_CONTENT_CHUNK_TABLE", "content_chunks")
RAG_SAMPLE_DATA_DIR = os.getenv("RAG_SAMPLE_DATA_DIR")
CMS_API_BASE_URL = os.getenv("CMS_API_BASE_URL", "http://localhost:8080/api")