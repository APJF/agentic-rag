# src/api/endpoints/sessions.py

from fastapi import APIRouter, HTTPException, Path
from typing import List

# Import các schema vừa tạo
from ..schemas import SessionListResponse, HistoryResponse, Message

# Import các hàm xử lý logic từ session_manager
from ...core.session_manager import list_sessions_for_user, load_chat_history
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()


@router.get("/user/{user_id}", response_model=SessionListResponse)
async def get_user_sessions(user_id: str = Path(..., description="ID của người dùng cần lấy danh sách phiên")):
    """
    Endpoint để lấy danh sách tóm tắt tất cả các phiên trò chuyện của một người dùng.
    """
    print(f"API: Nhận yêu cầu lấy danh sách phiên cho user_id: {user_id}")
    # Gọi hàm logic đã có để truy vấn database
    user_sessions = list_sessions_for_user(user_id)

    # Chuyển đổi kết quả từ database thành định dạng response của API
    # Pydantic sẽ tự động chuyển đổi datetime thành chuỗi ISO 8601
    return SessionListResponse(user_id=user_id, sessions=user_sessions)


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: int = Path(..., description="ID của phiên cần lấy lịch sử chat")):
    """
    Endpoint để lấy toàn bộ lịch sử tin nhắn của một phiên trò chuyện cụ thể.
    """
    print(f"API: Nhận yêu cầu lấy lịch sử cho session_id: {session_id}")
    # Tải lịch sử chat (dạng đối tượng LangChain)
    history_messages = load_chat_history(session_id)

    if not history_messages and session_id > 0:
        # Có thể session_id không tồn tại, trả về lỗi 404
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên trò chuyện.")

    # Chuyển đổi danh sách đối tượng LangChain thành danh sách đối tượng Pydantic 'Message'
    formatted_messages: List[Message] = []
    for msg in history_messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(Message(type='human', content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted_messages.append(Message(type='ai', content=msg.content))

    return HistoryResponse(session_id=session_id, messages=formatted_messages)