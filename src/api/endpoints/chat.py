# src/api/endpoints/chat.py

from fastapi import APIRouter, HTTPException, status, Body
from ..schemas import ChatRequest, ChatResponse,ChatEditRequest

# === THAY ĐỔI IMPORT: SỬ DỤNG QNA AGENT MỚI THAY VÌ ORCHESTRATOR ===
from ...features.qna.agent import initialize_qna_agent
from ...core.session_manager import load_chat_history, add_new_messages, rewind_last_turn
from langchain_core.messages import HumanMessage, AIMessage

# Khởi tạo QnAAgent một lần duy nhất khi server khởi động để tối ưu hiệu suất
qna_agent_executor = initialize_qna_agent()

router = APIRouter()


# --- Endpoint chat có trạng thái (đã cập nhật) ---
@router.post("/invoke", response_model=ChatResponse)
async def invoke_assistant(request: ChatRequest):
    """
    Endpoint chính để chat có trạng thái. Mọi yêu cầu (QnA, Quiz, Translate)
    đều được xử lý bởi QnAAgent.
    """
    session_id = int(request.session_id)
    user_input = request.user_input

    # Giả định user_id được lấy từ hệ thống xác thực.
    # Tạm thời, chúng ta có thể dùng một ID giả hoặc trích xuất từ session_id nếu có quy ước.
    user_id = "test_user_from_api"

    chat_history = load_chat_history(session_id)

    # Chuẩn bị dữ liệu và gọi thẳng đến QnAAgent
    input_data = {
        "user_id": user_id,
        "input": user_input,
        "chat_history": chat_history
    }
    result = qna_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: Agent không có output.")

    # Lưu lại lịch sử trò chuyện
    human_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [human_msg, ai_msg])

    return ChatResponse(session_id=str(session_id), ai_response=ai_response_text)


# --- Endpoint sửa tin nhắn (đã cập nhật) ---
@router.post("/edit_and_resubmit", response_model=ChatResponse)
async def edit_and_resubmit_message(request: ChatEditRequest = Body(...)):
    """
    Endpoint sửa tin nhắn, cũng sử dụng QnAAgent để xử lý lại yêu cầu.
    """
    session_id = request.session_id
    corrected_input = request.corrected_input
    print(f"API: Nhận yêu cầu sửa tin nhắn cho session_id: {session_id}")

    # Bước 1: Tua lại lượt nói cuối cùng trong database
    success = rewind_last_turn(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy phiên {session_id} hoặc không có đủ tin nhắn để sửa."
        )

    # Bước 2: Tải lại lịch sử đã được rút gọn và gọi QnAAgent
    user_id = "test_user_from_api"  # Lấy user_id tương tự
    chat_history = load_chat_history(session_id)

    input_data = {
        "user_id": user_id,
        "input": corrected_input,
        "chat_history": chat_history
    }
    result = qna_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: Agent không có output.")

    # Bước 3: Lưu lại lượt nói mới (đã sửa)
    human_msg = HumanMessage(content=corrected_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [human_msg, ai_msg])

    return ChatResponse(
        session_id=str(session_id),
        ai_response=ai_response_text
    )

