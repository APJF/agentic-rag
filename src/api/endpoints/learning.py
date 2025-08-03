# src/api/endpoints/learning.py

from fastapi import APIRouter, Body, HTTPException, status
from ..schemas import ChatRequest, ChatResponse

# Import các thành phần cần thiết
from ...features.learning.agent import initialize_learning_agent
from ...core.session_manager import load_session_data, add_new_messages
from langchain_core.messages import HumanMessage, AIMessage

# Khởi tạo LearningAgent một lần duy nhất
learning_agent_executor = initialize_learning_agent()

router = APIRouter()

# Đã chuyển endpoint /chat sang dispatcher. Nếu có hàm đặc thù, giữ lại, còn lại comment hoặc xóa endpoint /chat.
