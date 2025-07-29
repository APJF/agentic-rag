# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# --- Schemas cho Quản lý Phiên (Sessions) ---

class SessionInfo(BaseModel):
    id: int
    session_name: str
    updated_at: datetime

class SessionListResponse(BaseModel):
    user_id: str
    sessions: List[SessionInfo]

class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., description="ID của người dùng đang tạo phiên.")
    session_name: str = Field(..., description="Tên do người dùng đặt cho phiên.")
    session_type: Optional[str] = Field('GENERAL', description="Loại phiên: GENERAL, PLANNER, STUDY, EXAM_REVIEW.")
    context: Optional[Dict[str, Any]] = Field(None, description="Ngữ cảnh ban đầu cho phiên.")

class SessionRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1, description="Tên mới cho phiên.")

# --- Schemas cho Tương tác Chat (Chat & Planner) ---

class ChatRequest(BaseModel):
    session_id: int = Field(..., description="ID của phiên trò chuyện đang diễn ra.")
    user_input: str = Field(..., description="Nội dung tin nhắn mới của người dùng.")

class ChatResponse(BaseModel):
    session_id: str
    ai_response: str

class ChatEditRequest(BaseModel):
    session_id: int = Field(..., description="ID của phiên cần sửa.")
    corrected_input: str = Field(..., description="Nội dung tin nhắn đã được sửa lại.")

class SessionInfo(BaseModel):
    id: int
    session_name: str
    updated_at: datetime
# --- Schemas cho Lịch sử Chat ---

class Message(BaseModel):
    type: Literal['human', 'ai']
    content: str

class HistoryResponse(BaseModel):
    session_id: int
    messages: List[Message]

# === THÊM CÁC SCHEMA MỚI CHO CHỨC NĂNG CHẤM BÀI ===

class ExamGradeRequest(BaseModel):
    exam_result_id: int = Field(..., description="ID của bài làm (exam_result) cần được chấm và nhận xét.")

class ExamGradeResponse(BaseModel):
    exam_result_id: int
    overall_score: float
    ai_feedback: str #

class ChatInitiateRequest(BaseModel):
    user_id: str = Field(..., description="ID của người dùng.")
    session_type: str = Field(..., description="Loại phiên: GENERAL, PLANNER, STUDY, EXAM_REVIEW.")
    first_message: str = Field(..., description="Nội dung tin nhắn đầu tiên của người dùng.")
    context: Optional[Dict[str, Any]] = Field(None, description="Ngữ cảnh ban đầu cho phiên.")

class ChatInitiateResponse(BaseModel):
    session_id: int
    session_name: str
    ai_first_response: str