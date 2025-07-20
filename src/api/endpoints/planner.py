# src/api/endpoints/planner.py

from fastapi import APIRouter, Body
from ..schemas import ChatRequest, ChatResponse  # Tái sử dụng schema chat

# Import các thành phần cần thiết
from ...features.planner.agent import initialize_planning_agent
from ...core.session_manager import load_chat_history, add_new_messages
from langchain_core.messages import HumanMessage, AIMessage

# Khởi tạo PlannerAgent một lần duy nhất
planner_agent_executor = initialize_planning_agent()

router = APIRouter()


@router.post("/invoke", response_model=ChatResponse)
async def invoke_planner_agent(request: ChatRequest = Body(...)):
    """
    Endpoint chuyên dụng để tương tác với PlannerAgent.
    Nó quản lý một cuộc hội thoại riêng để tạo lộ trình học.
    """
    session_id = request.session_id
    user_input = request.user_input

    print(f"--- Planner API nhận input: '{user_input}' cho session '{session_id}' ---")

    chat_history = load_chat_history(session_id)

    user_id = "temp_user"
    input_data = {
        "user_id": user_id,
        "input": user_input,
        "chat_history": chat_history
    }
    result = planner_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: PlannerAgent không có output.")

    human_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [human_msg, ai_msg])

    return ChatResponse(
        session_id=session_id,
        ai_response=ai_response_text
    )