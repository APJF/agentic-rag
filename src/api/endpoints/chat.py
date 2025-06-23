# src/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, status, Body
from ..schemas import ChatRequest, ChatResponse, StatelessChatRequest, StatelessChatResponse, ChatEditRequest
from ...main_orchestrator import run_orchestrator
from ...core.session_manager import load_chat_history, add_new_messages, rewind_last_turn
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

@router.post("/invoke/stateless", response_model=StatelessChatResponse)
async def invoke_stateless_assistant(request: StatelessChatRequest):
    """Endpoint tạm thời, không lưu trữ lịch sử."""
    print(f"Stateless API received input: '{request.user_input}'")
    ai_response_text = run_orchestrator(request.user_input, chat_history=[])
    return StatelessChatResponse(ai_response=ai_response_text)

@router.post("/invoke", response_model=ChatResponse)
async def invoke_assistant(request: ChatRequest):
    """Endpoint chính để chat có trạng thái, sử dụng session_id."""
    session_id = request.session_id
    user_input = request.user_input

    chat_history = load_chat_history(session_id)
    ai_response_text = run_orchestrator(user_input, chat_history)

    human_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=ai_response_text)

    add_new_messages(session_id, [human_msg, ai_msg])

    return ChatResponse(session_id=session_id, ai_response=ai_response_text)


@router.post("/edit_and_resubmit", response_model=ChatResponse)
async def edit_and_resubmit_message(request: ChatEditRequest = Body(...)):
    """
    Endpoint chuyên dụng cho việc sửa tin nhắn.
    Nó sẽ "tua lại" lượt nói cuối cùng và xử lý tin nhắn đã sửa như một lượt nói mới.
    """
    session_id = request.session_id
    corrected_input = request.corrected_input
    print(f"API: Nhận yêu cầu sửa tin nhắn cho session_id: {session_id}")

    success = rewind_last_turn(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy phiên {session_id} hoặc không có đủ tin nhắn để sửa."
        )

    chat_history = load_chat_history(session_id)
    ai_response_text = run_orchestrator(corrected_input, chat_history)
    human_msg = HumanMessage(content=corrected_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [human_msg, ai_msg])

    return ChatResponse(
        session_id=str(session_id),
        ai_response=ai_response_text
    )