# src/api/endpoints/planner.py

from fastapi import APIRouter, Body, HTTPException
from ..schemas import ChatRequest, ChatResponse

from ...features.planner.agent import initialize_planning_agent
from ...core.session_manager import load_session_data, add_new_messages
from langchain_core.messages import HumanMessage, AIMessage

planner_agent_executor = initialize_planning_agent()
router = APIRouter()

@router.post("/invoke", response_model=ChatResponse)
async def invoke_planner_agent(request: ChatRequest = Body(...)):
    """
    Endpoint chuyên dụng để tương tác với PlannerAgent.
    """
    session_data = load_session_data(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên ID {request.session_id} không tồn tại.")

    print(f"--- Planner API nhận input: '{request.user_input}' cho session '{request.session_id}' ---")

    input_data = {
        "user_id": session_data["user_id"],
        "input": request.user_input,
        "chat_history": session_data["history"]
    }
    result = planner_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: PlannerAgent không có output.")

    human_msg = HumanMessage(content=request.user_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(request.session_id, [human_msg, ai_msg])

    return ChatResponse(
        session_id=str(request.session_id),
        ai_response=ai_response_text
    )
