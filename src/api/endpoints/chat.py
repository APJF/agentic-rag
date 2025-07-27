# src/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, status, Body
from ..schemas import ChatRequest, ChatResponse, ChatEditRequest
from ...features.qna.agent import initialize_qna_agent
from ...core.session_manager import load_session_data, add_new_messages, rewind_last_turn
from langchain_core.messages import HumanMessage, AIMessage

qna_agent_executor = initialize_qna_agent()
router = APIRouter()


@router.post("/invoke", response_model=ChatResponse)
async def invoke_assistant(request: ChatRequest):
    session_data = load_session_data(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên ID {request.session_id} không tồn tại.")

    input_data = {
        "user_id": session_data["user_id"],
        "input": request.user_input,
        "chat_history": session_data["history"]
    }
    result = qna_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: Agent không có output.")

    human_msg = HumanMessage(content=request.user_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(request.session_id, [human_msg, ai_msg])

    return ChatResponse(session_id=str(request.session_id), ai_response=ai_response_text)


@router.post("/edit_and_resubmit", response_model=ChatResponse)
async def edit_and_resubmit_message(request: ChatEditRequest = Body(...)):
    success = rewind_last_turn(request.session_id)
    if not success:
        raise HTTPException(status_code=404,
                            detail=f"Không tìm thấy phiên {request.session_id} hoặc không đủ tin nhắn để sửa.")

    session_data = load_session_data(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên ID {request.session_id} không tồn tại.")

    input_data = {
        "user_id": session_data["user_id"],
        "input": request.corrected_input,
        "chat_history": session_data["history"]
    }
    result = qna_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: Agent không có output.")

    human_msg = HumanMessage(content=request.corrected_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(request.session_id, [human_msg, ai_msg])

    return ChatResponse(session_id=str(request.session_id), ai_response=ai_response_text)
