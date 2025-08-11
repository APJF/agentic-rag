# src/api/endpoints/sessions.py

from fastapi import APIRouter, HTTPException, Path, Body, status, Response, Request
from typing import List, Optional
from datetime import datetime, timezone

from ..schemas import SessionListResponse, HistoryResponse, Message, SessionCreateRequest, SessionInfo, SessionRenameRequest, ChatInitiateRequest, ChatInitiateResponse
from ...core.session_manager import (
    list_sessions_for_user,
    load_chat_history,
    create_new_session,
    get_user,
    delete_session,
    rename_session, find_session, add_new_messages
)
from langchain_core.messages import HumanMessage, AIMessage
from ...features.qna.agent import initialize_qna_agent
from ...features.planner.agent import initialize_planning_agent
from ...features.learning.agent import initialize_learning_agent
from ...features.reviewer.agent import initialize_reviewer_agent
from ...features.speaking.agent import initialize_speaking_agent
from langchain_openai import ChatOpenAI
import re

router = APIRouter()


@router.post("/create", response_model=ChatInitiateResponse)
async def create_session(request: ChatInitiateRequest = Body(...)):
    get_user(int(request.user_id))

    def deep_has_key(obj: dict, keys: List[str]) -> bool:
        if not isinstance(obj, dict):
            return False
        for k, v in obj.items():
            if k in keys:
                return True
            if isinstance(v, dict) and deep_has_key(v, keys):
                return True
        return False

    # 1) Xác định intent ưu tiên theo request.session_type nếu có
    intent = (request.session_type or "").lower() if request.session_type else None

    # 2) Nếu chưa có, cố gắng nhận diện theo context (dò sâu)
    if not intent and request.context:
        if deep_has_key(request.context, ["exam_id", "exam_result_id", "exam", "exam_result"]):
            intent = "reviewer"
        elif deep_has_key(request.context, ["course_id", "material_id", "lesson_id", "material"]):
            intent = "learning"

    # 3) Nếu vẫn chưa có, dùng rules từ câu hỏi đầu
    if not intent:
        def detect_intent_custom(user_input: str, context: dict = None) -> str:
            # Rule: Nếu có từ khóa lộ trình
            roadmap_keywords = [
                "lộ trình", "roadmap", "kế hoạch học", "plan học", "học gì", "nên học", "học như thế nào",
                "học jlpt", "thi jlpt", "học n3", "học n2", "học n1", "làm sao để thi", "chuẩn bị thi"
            ]
            user_input_lower = user_input.lower()
            if any(kw in user_input_lower for kw in roadmap_keywords):
                return "planner"
            # Nếu không match gì
            return "qna"
        intent = detect_intent_custom(request.first_message, request.context)

    # 4) Sinh tên session tự động
    async def generate_session_name(first_message: str, intent: str) -> str:
        from ...core.llm import get_llm
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        if intent == "speaking":
            return f"Luyện nói: {first_message[:20]}"
        elif intent == "planner":
            return "Tư vấn Lộ trình học"
        elif intent == "reviewer":
            return "Chữa bài kiểm tra"
        elif intent == "learning":
            return "Học tập cùng AI"
        llm_instance = get_llm()
        if llm_instance:
            prompt = ChatPromptTemplate.from_template("Hãy tóm tắt câu sau thành một tiêu đề không quá 5 từ: '{text}'")
            chain = prompt | llm_instance | StrOutputParser()
            try:
                return await chain.ainvoke({"text": first_message})
            except Exception:
                pass
        return f"Chat: {first_message[:20]}"

    session_name = await generate_session_name(request.first_message, intent)

    # 5) Tạo session
    session_id = create_new_session(
        user_id=int(request.user_id),
        session_name=session_name,
        session_type=intent,
        context=request.context
    )
    if not session_id:
        raise HTTPException(status_code=500, detail="Không thể tạo phiên mới.")

    # 6) Lưu tin nhắn đầu tiên
    human_msg = HumanMessage(content=request.first_message)
    add_new_messages(session_id, [human_msg])

    # 7) Gọi agent phù hợp với context đính kèm
    agent_map = {
        "qna": initialize_qna_agent(),
        "planner": initialize_planning_agent(),
        "learning": initialize_learning_agent(),
        "reviewer": initialize_reviewer_agent(),
        "speaking": initialize_speaking_agent(),
    }
    agent = agent_map.get(intent)
    input_data = {
        "user_id": int(request.user_id),
        "input": request.first_message,
        "chat_history": [human_msg],
        "context": request.context or {}
    }

    # Đảm bảo Planner tools nhận đúng user_id
    if intent == "planner":
        from ...features.planner.tools import set_session_user_id
        set_session_user_id(int(request.user_id))

    result = agent.invoke(input_data)
    ai_response_text = result.get('output', "Xin hãy cung cấp thêm thông tin.")

    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [ai_msg])

    return ChatInitiateResponse(
        session_id=session_id,
        session_name=session_name,
        ai_first_response=ai_response_text
    )


@router.get("/user/{user_id}", response_model=SessionListResponse)
async def get_user_sessions(user_id: str = Path(..., description="ID của người dùng")):
    user_sessions = list_sessions_for_user(int(user_id))
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

    session_info = find_session(int(user_id), session_type, context if context else None)

    if session_info:
        return session_info
    else:
        return None