# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class StatelessChatRequest(BaseModel):
    user_input: str

class StatelessChatResponse(BaseModel):
    ai_response: str

# Bạn có thể giữ lại các schema cũ để dùng cho API có trạng thái sau này
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    session_id: str
    ai_response: str

class SessionInfo(BaseModel):
    id: int
    session_name: str
    updated_at: datetime

# Schema cho response trả về danh sách các phiên
class SessionListResponse(BaseModel):
    user_id: str
    sessions: List[SessionInfo]

# Schema cho một tin nhắn duy nhất trong lịch sử
class Message(BaseModel):
    type: Literal['human', 'ai']
    content: str

# Schema cho response trả về toàn bộ lịch sử của một phiên
class HistoryResponse(BaseModel):
    session_id: int
    messages: List[Message]