import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# LLM and Embedding Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")

# Table Names
CHUNK_TABLE_NAME = os.getenv("CHUNK_TABLE_NAME", "pdf_document_chunks")
VOCAB_TABLE_NAME_FOR_LESSON_SUGGESTION = os.getenv("VOCAB_TABLE_NAME", "japanese_english_knowledge") # Hoặc tên mới từ pdf_vocab_loader.py
