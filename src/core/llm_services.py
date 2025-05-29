# src/core/llm_services.py
from langchain_openai import ChatOpenAI
from src.config import settings

llm = None
if settings.OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.DEFAULT_LLM_MODEL
        )
        print(f"LLM ({settings.DEFAULT_LLM_MODEL}) initialized.")
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        print("Please ensure you have the correct version of langchain-openai and valid API key/model name.")
        print("Common parameter names are 'api_key' and 'model'.")
else:
    print("Warning: OPENAI_API_KEY not found. LLM functionalities will be disabled.")

def get_llm():
    if not llm:
        print("LLM was not successfully initialized. Please check previous errors or OPENAI_API_KEY.")
        raise ValueError("LLM not initialized.")
    return llm