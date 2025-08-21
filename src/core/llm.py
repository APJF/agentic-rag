# src/core/llm.py
from langchain_openai import ChatOpenAI
from src.config import settings

llm = None
if settings.OPENAI_API_KEY:
    try:
        client_kwargs = {
            "api_key": settings.OPENAI_API_KEY,
            "model": settings.DEFAULT_LLM_MODEL,
        }
        # Hỗ trợ endpoint OpenAI-compatible nếu có
        if settings.OPENAI_BASE_URL:
            client_kwargs["base_url"] = settings.OPENAI_BASE_URL
        if settings.OPENAI_ORG:
            client_kwargs["organization"] = settings.OPENAI_ORG

        llm = ChatOpenAI(**client_kwargs)
        print(f"LLM ({settings.DEFAULT_LLM_MODEL}) initialized.")
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        print("Please ensure your OPENAI_* envs and model name are valid.")
else:
    print("Warning: OPENAI_API_KEY not found. LLM functionalities will be disabled.")

def get_llm():
    if not llm:
        # Không ném lỗi để tránh làm app crash khi import router/agent lúc startup
        # Trả về None để các nơi gọi có thể tự degrade hoặc xử lý phù hợp
        print("LLM was not successfully initialized. Please check previous errors or OPENAI_API_KEY.")
        return None
    return llm