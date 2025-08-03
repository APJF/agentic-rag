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
    level: Optional[str] = None  # Thêm trường level cho speaking

class ChatResponse(BaseModel):
    session_id: int
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
    user_id: str
    exam_id: str
    answers: Dict[str, str]  # {question_id: user_answer}

class ExamGradeResponse(BaseModel):
    exam_result_id: str
    score: int
    advice: str
    session_id: int

class ExamResultDetailResponse(BaseModel):
    score: int
    advice: str
    details: List[Dict]
    chat_history: List[Dict]

class ChatInitiateRequest(BaseModel):
    user_id: str = Field(..., description="ID của người dùng.")
    session_type: str = Field(..., description="Loại phiên: GENERAL, PLANNER, STUDY, EXAM_REVIEW.")
    first_message: str = Field(..., description="Nội dung tin nhắn đầu tiên của người dùng.")
    context: Optional[Dict[str, Any]] = Field(None, description="Ngữ cảnh ban đầu cho phiên.")

class ChatInitiateResponse(BaseModel):
    session_id: int
    session_name: str
    ai_first_response: str

class ExamAdviceRequest(BaseModel):
    user_id: str
    exam_id: str
    exam_result_id: str
    score: int
    total: int
    details: List[Dict]  # [{question_id, is_correct, user_answer, correct_answer, ...}]

class ExamAdviceResponse(BaseModel):
    advice: Dict  # {"strengths": [...], "weaknesses": [...], "suggestions": [...]}

class ExamReviewChatRequest(BaseModel):
    user_id: str
    exam_id: str
    exam_result_id: str
    question_id: Optional[str]
    user_input: str
    # Thêm các trường khác: details, đáp án, v.v. nếu cần

class EssayGradeRequest(BaseModel):
    user_id: str
    course_id: str
    unit_id: str
    essay_text: str
    learned_courses: List[str]
    learned_grammar: List[str]
    learned_vocab: List[str]

class EssayGradeResponse(BaseModel):
    total_score: float
    content_score: float
    grammar_score: float
    vocab_score: float
    coherence_score: float
    advice: Dict  # {"strengths": [...], "weaknesses": [...], "suggestions": [...], "warnings": [...]}

class EssayReviewChatRequest(BaseModel):
    user_id: str
    essay_result_id: str
    essay_text: str
    content_score: float
    grammar_score: float
    vocab_score: float
    coherence_score: float
    advice: Dict
    user_input: str

class SpeakingChatRequest(BaseModel):
    user_id: str
    session_id: Optional[int]
    user_input: str
    topic: Optional[str]  # Chủ đề do user chọn
    level: Optional[str]
    target: Optional[str]
    hobby: Optional[str]