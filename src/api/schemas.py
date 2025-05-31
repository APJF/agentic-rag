# src/api/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None # Tùy chọn: Nếu bạn muốn theo dõi người dùng
    # Thêm các trường khác nếu cần, ví dụ: session_id

class ChatResponse(BaseModel):
    answer: str
    # Bạn có thể thêm các trường khác vào response nếu muốn, ví dụ:
    # sources: Optional[List[Dict[str, any]]] = None
    # error_message: Optional[str] = None