# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

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

class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., description="ID của người dùng đang tạo phiên mới.")
    session_name: str = Field(..., min_length=1, description="Tên do người dùng đặt cho phiên trò chuyện mới.")

class SessionRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1, description="Tên mới cho phiên trò chuyện.")

class ChatEditRequest(BaseModel):
    session_id: int = Field(..., description="ID của phiên trò chuyện cần sửa.")
    corrected_input: str = Field(..., description="Nội dung tin nhắn mới đã được người dùng sửa lại.")

class RewindResponse(BaseModel):
    session_id: int
    message: str
    rewound_messages_count: int