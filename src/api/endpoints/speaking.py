from fastapi import APIRouter, Body, HTTPException
from ..schemas import SpeakingChatRequest, ChatResponse
from ...features.speaking.agent import initialize_speaking_agent
from ...core.session_manager import add_new_messages
from ...core.database import execute_sql_query
from langchain_core.messages import HumanMessage, AIMessage

speaking_agent_executor = initialize_speaking_agent()
router = APIRouter()

# Đã chuyển endpoint /chat sang dispatcher. Nếu có hàm đặc thù, giữ lại, còn lại comment hoặc xóa endpoint /chat.