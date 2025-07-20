# src/api/endpoints/sessions.py

from fastapi import APIRouter, HTTPException, Path, Body, status, Response
from typing import List

# Import các schema cần thiết
from ..schemas import SessionListResponse, HistoryResponse, Message, SessionCreateRequest, SessionInfo, SessionRenameRequest, RewindResponse

# Import các hàm xử lý logic từ session_manager
from ...core.session_manager import (
    list_sessions_for_user,
    load_chat_history,
    create_new_session,
    get_or_create_user,
    delete_session,
    rename_session,
    rewind_last_turn
)
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

@router.post("/", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
async def create_new_chat_session(request: SessionCreateRequest = Body(...)):
    """
    Endpoint để tạo một phiên trò chuyện (chat session) mới cho người dùng.
    """
    print(f"API: Nhận yêu cầu tạo phiên mới cho user '{request.user_id}' với tên '{request.session_name}'")

    get_or_create_user(request.user_id)

    new_session_id = create_new_session(request.user_id, request.session_name)

    if new_session_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể tạo phiên mới trong database."
        )

    from datetime import datetime
    return SessionInfo(
        id=new_session_id,
        session_name=request.session_name,
        updated_at=datetime.utcnow()
    )

@router.get("/user/{user_id}", response_model=SessionListResponse)
async def get_user_sessions(user_id: str = Path(..., description="ID của người dùng cần lấy danh sách phiên")):
    """
    Endpoint để lấy danh sách tóm tắt tất cả các phiên trò chuyện của một người dùng.
    """
    print(f"API: Nhận yêu cầu lấy danh sách phiên cho user_id: {user_id}")
    user_sessions = list_sessions_for_user(user_id)
    return SessionListResponse(user_id=user_id, sessions=user_sessions)

@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: int = Path(..., description="ID của phiên cần lấy lịch sử chat")):
    """
    Endpoint để lấy toàn bộ lịch sử tin nhắn của một phiên trò chuyện cụ thể.
    """
    # ... (giữ nguyên logic của hàm này)
    print(f"API: Nhận yêu cầu lấy lịch sử cho session_id: {session_id}")
    history_messages = load_chat_history(session_id)
    if not history_messages:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên trò chuyện.")
    formatted_messages: List[Message] = []
    for msg in history_messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(Message(type='human', content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted_messages.append(Message(type='ai', content=msg.content))

    return HistoryResponse(session_id=session_id, messages=formatted_messages)

@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(session_id: int = Path(..., description="ID của phiên cần xóa")):
    """
    Endpoint để xóa một phiên trò chuyện và tất cả tin nhắn liên quan.
    """
    print(f"API: Nhận yêu cầu xóa session_id: {session_id}")
    success = delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy phiên trò chuyện với ID {session_id} để xóa."
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.put("/{session_id}/rename", response_model=SessionInfo)
async def rename_chat_session(
        session_id: int = Path(..., description="ID của phiên cần đổi tên"),
        request: SessionRenameRequest = Body(...)
):
    """
    Endpoint để cập nhật lại tên của một phiên trò chuyện.
    """
    print(f"API: Nhận yêu cầu đổi tên session_id: {session_id} thành '{request.new_name}'")
    success = rename_session(session_id, request.new_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy phiên trò chuyện với ID {session_id} để đổi tên."
        )

    from datetime import datetime
    return SessionInfo(
        id=session_id,
        session_name=request.new_name,
        updated_at=datetime.utcnow()
    )
