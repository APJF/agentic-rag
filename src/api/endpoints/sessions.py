# src/api/endpoints/sessions.py

from fastapi import APIRouter, HTTPException, Path, Body, status, Response, Request
from typing import List, Optional
from datetime import datetime, timezone

from ..schemas import SessionListResponse, HistoryResponse, Message, SessionCreateRequest, SessionInfo, \
    SessionRenameRequest
from ...core.session_manager import (
    list_sessions_for_user,
    load_chat_history,
    create_new_session,
    get_or_create_user,
    delete_session,
    rename_session, find_session
)
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()


@router.post("/", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
async def create_new_chat_session(request: SessionCreateRequest = Body(...)):
    """
    Endpoint để tạo một phiên trò chuyện mới, có hỗ trợ loại và ngữ cảnh.
    """
    print(f"API: Nhận yêu cầu tạo phiên '{request.session_type}' cho user '{request.user_id}'")

    get_or_create_user(request.user_id)

    new_session_id = create_new_session(
        user_id=request.user_id,
        session_name=request.session_name,
        session_type=request.session_type,
        context=request.context
    )

    if new_session_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể tạo phiên mới trong database."
        )

    return SessionInfo(
        id=new_session_id,
        session_name=request.session_name,
        updated_at=datetime.now(timezone.utc)
    )


@router.get("/user/{user_id}", response_model=SessionListResponse)
async def get_user_sessions(user_id: str = Path(..., description="ID của người dùng")):
    user_sessions = list_sessions_for_user(user_id)
    return SessionListResponse(user_id=user_id, sessions=user_sessions)


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: int = Path(..., description="ID của phiên")):
    history_messages = load_chat_history(session_id)
    if not history_messages:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy lịch sử cho phiên ID {session_id}.")

    formatted_messages: List[Message] = []
    for msg in history_messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(Message(type='human', content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted_messages.append(Message(type='ai', content=msg.content))

    return HistoryResponse(session_id=session_id, messages=formatted_messages)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(session_id: int = Path(..., description="ID của phiên cần xóa")):
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phiên ID {session_id} để xóa.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/{session_id}/rename", response_model=SessionInfo)
async def rename_chat_session(session_id: int = Path(..., description="ID của phiên"),
                              request: SessionRenameRequest = Body(...)):
    success = rename_session(session_id, request.new_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phiên ID {session_id} để đổi tên.")
    return SessionInfo(id=session_id, session_name=request.new_name, updated_at=datetime.now(timezone.utc))


@router.get("/find", response_model=Optional[SessionInfo])
async def find_existing_session(request: Request):
    """
    Tìm kiếm một phiên làm việc đã tồn tại dựa trên các tiêu chí.
    Dùng để kiểm tra trước khi tạo một phiên mới.
    Ví dụ: GET /sessions/find?user_id=nunulac&session_type=PLANNER
    Ví dụ: GET /sessions/find?user_id=nunulac&session_type=STUDY&context_material_id=M101
    """
    # Lấy các tham số từ query string của URL
    params = request.query_params
    user_id = params.get("user_id")
    session_type = params.get("session_type")

    if not user_id or not session_type:
        raise HTTPException(status_code=400, detail="user_id và session_type là bắt buộc.")

    context = {k.replace('context_', ''): v for k, v in params.items() if k.startswith('context_')}

    print(f"API: Nhận yêu cầu tìm phiên: user='{user_id}', type='{session_type}', context={context}")

    session_info = find_session(user_id, session_type, context if context else None)

    if session_info:
        return session_info
    else:
        return None