# src/api/endpoints/planner.py

from fastapi import APIRouter, Body, HTTPException
from ..schemas import ChatRequest, ChatResponse

from ...features.planner.agent import initialize_planning_agent
from ...core.session_manager import load_session_data, add_new_messages
from langchain_core.messages import HumanMessage, AIMessage

planner_agent_executor = initialize_planning_agent()
router = APIRouter()

# Đã chuyển endpoint /chat sang dispatcher. Nếu có hàm đặc thù, giữ lại, còn lại comment hoặc xóa endpoint /chat.
