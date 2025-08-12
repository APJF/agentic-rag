# src/api/endpoints/sessions.py

from fastapi import APIRouter, HTTPException, Path, Body, status, Response, Request, Query
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

# CREATE: POST /api/sessions
@router.post("/", response_model=ChatInitiateResponse)
async def create_session(request: ChatInitiateRequest = Body(...)):
    # ... giữ nguyên logic đã viết trong hàm create_session trước đó ...
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

    intent = (request.session_type or "").lower() if request.session_type else None
    if not intent and request.context:
        if deep_has_key(request.context, ["exam_id", "exam_result_id", "exam", "exam_result"]):
            intent = "reviewer"
        elif deep_has_key(request.context, ["course_id", "material_id", "lesson_id", "material"]):
            intent = "learning"
    if not intent:
        def detect_intent_custom(user_input: str, context: dict = None) -> str:
            roadmap_keywords = [
                "lộ trình", "roadmap", "kế hoạch học", "plan học", "học gì", "nên học", "học như thế nào",
                "học jlpt", "thi jlpt", "học n3", "học n2", "học n1", "làm sao để thi", "chuẩn bị thi"
            ]
            user_input_lower = user_input.lower()
            if any(kw in user_input_lower for kw in roadmap_keywords):
                return "planner"
            return "qna"
        intent = detect_intent_custom(request.first_message, request.context)

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
            prompt = ChatPromptTemplate.from_template("Hãy tóm tắt câu sau thành một tiêu đề không quá 10 từ: '{text}'")
            chain = prompt | llm_instance | StrOutputParser()
            try:
                return await chain.ainvoke({"text": first_message})
            except Exception:
                pass
        return f"Chat: {first_message[:20]}"

    session_name = await generate_session_name(request.first_message, intent)

    session_id = create_new_session(
        user_id=int(request.user_id),
        session_name=session_name,
        session_type=intent,
        context=request.context
    )
    if not session_id:
        raise HTTPException(status_code=500, detail="Không thể tạo phiên mới.")

    human_msg = HumanMessage(content=request.first_message)
    add_new_messages(session_id, [human_msg])

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

# LIST: GET /api/sessions?user_id=...
@router.get("/", response_model=SessionListResponse)
async def list_sessions(user_id: str = Query(..., description="ID người dùng")):
    sessions = list_sessions_for_user(int(user_id))
    return SessionListResponse(user_id=user_id, sessions=sessions)

# DETAIL: GET /api/sessions/{id}
@router.get("/{session_id}", response_model=HistoryResponse)
async def get_session_detail(session_id: int = Path(...)):
    from ...core.database import get_db_connection
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Không thể kết nối DB.")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, content, "order"
                FROM message
                WHERE session_id = %s
                ORDER BY "order" ASC;
                """,
                (session_id,)
            )
            rows = cur.fetchall()
            if rows is None:
                raise HTTPException(status_code=404, detail="Không tìm thấy session.")
            messages = []
            for rid, mtype, content, order in rows:
                messages.append({
                    "id": rid,
                    "order": order,
                    "type": 'human' if mtype == 'human' else 'ai',
                    "content": content
                })
    finally:
        if conn:
            conn.close()
    return HistoryResponse(session_id=session_id, messages=messages)

# UPDATE: PUT /api/sessions/{id} (toàn phần) và PATCH /api/sessions/{id} (một phần)
from pydantic import BaseModel
class SessionUpdateRequest(BaseModel):
    session_name: Optional[str] = None
    context: Optional[dict] = None

@router.put("/{session_id}", response_model=SessionInfo)
async def put_session(session_id: int, request: SessionUpdateRequest = Body(...)):
    updated = False
    if request.session_name:
        updated = rename_session(session_id, request.session_name)
    # TODO: cập nhật context nếu cần (cần hàm update_context trong session_manager)
    if not updated and not request.context:
        raise HTTPException(status_code=400, detail="Không có trường nào để cập nhật.")
    return SessionInfo(id=session_id, session_name=request.session_name or "", updated_at=datetime.now(timezone.utc))

@router.patch("/{session_id}", response_model=SessionInfo)
async def patch_session(session_id: int, request: SessionUpdateRequest = Body(...)):
    return await put_session(session_id, request)