# src/api/endpoints/chat.py

from fastapi import APIRouter
# Sử dụng schema mới cho endpoint tạm thời
from ..schemas import StatelessChatRequest, StatelessChatResponse

# Import hàm orchestrator cốt lõi
from ...main_orchestrator import run_orchestrator

# Tạo một router cho các endpoint
router = APIRouter()

@router.post("/invoke/stateless", response_model=StatelessChatResponse)
async def invoke_stateless_assistant(request: StatelessChatRequest):
    """
    Endpoint tạm thời, không lưu trữ lịch sử chat.
    Nhận vào một câu hỏi duy nhất và trả về một câu trả lời.
    Mỗi lần gọi là một cuộc hội thoại mới.
    """
    user_input = request.user_input

    print(f"Stateless API received input: '{user_input}'")

    ai_response_text = run_orchestrator(user_input, chat_history=[])

    return StatelessChatResponse(ai_response=ai_response_text)

# Bạn có thể giữ lại endpoint cũ ở đây để phát triển song song
# @router.post("/invoke", ...)