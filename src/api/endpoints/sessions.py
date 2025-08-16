# src/api/endpoints/sessions.py

from fastapi import APIRouter, HTTPException, Path, Body, status, Response, Request, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from ..schemas import SessionListResponse, HistoryResponse, Message, SessionCreateRequest, SessionInfo, SessionRenameRequest, ChatInitiateRequest, ChatInitiateResponse
from ...core.session_manager import (
    list_sessions_for_user,
    load_chat_history,
    create_new_session,
    get_user,
    delete_session,
    rename_session,
    find_session,
    add_new_messages,
    get_session_info,
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

    # Seed context for planner: derive info from first_message and user profile
    merged_context: Dict[str, Any] = dict(request.context or {})
    if intent == "planner":
        def _derive_planner_context_from_text(text: str) -> Dict[str, Any]:
            t = (text or "").lower()
            ctx: Dict[str, Any] = {}
            # current level inference
            if any(k in t for k in ["mới học", "moi hoc", "chưa biết gì", "chua biet gi", "newbie", "bắt đầu", "bat dau"]):
                ctx["current_level"] = "N5_L"
            # learning goal / target level
            import re as _re
            m_target = _re.search(r"\b(n5|n4|n3|n2|n1)\b", t)
            if m_target and ("học" in t or "thi" in t or "jlpt" in t):
                target = m_target.group(1).upper()
                ctx["learning_goal"] = f"JLPT {target}"
                ctx["target_level"] = target
            # deadline "trong năm nay"
            if "trong năm nay" in t:
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc) + timedelta(hours=7)
                deadline = now.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
                ctx["deadline_info"] = deadline.isoformat()
            return ctx

        # merge user profile (level/hobby/target) if exists
        try:
            from ...core.database import get_db_connection
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT level, hobby, target FROM "users" WHERE id = %s;', (int(request.user_id),))
                    row = cur.fetchone()
                    if row:
                        level, hobby, target = row
                        if level and not merged_context.get("current_level"):
                            # Chuẩn hóa N5-L -> N5_L khi nạp từ hồ sơ
                            import re as _re
                            lvl_norm = _re.sub(r"\b(N[1-5])[-_]([HML])\b", r"\1_\2", str(level).upper())
                            merged_context["current_level"] = lvl_norm
                        if hobby and not merged_context.get("hobby"):
                            merged_context["hobby"] = hobby
                        if target and not merged_context.get("learning_goal"):
                            merged_context["learning_goal"] = target
        except Exception:
            pass

        derived = _derive_planner_context_from_text(request.first_message)
        merged_context.update({k: v for k, v in derived.items() if v is not None})

    session_id = create_new_session(
        user_id=int(request.user_id),
        session_name=session_name,
        session_type=intent,
        context=merged_context
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
    sessions_raw = list_sessions_for_user(int(user_id))
    # Chuẩn hóa khóa 'name' -> 'session_name' để khớp schema và convert UTC->UTC+7
    sessions_norm = []
    for s in sessions_raw:
        created_at = s.get("created_at")
        updated_at = s.get("updated_at")
        try:
            if created_at:
                created_at = created_at + timedelta(hours=7)
            if updated_at:
                updated_at = updated_at + timedelta(hours=7)
        except Exception:
            pass
        sessions_norm.append({
            "id": s.get("id"),
            "session_name": s.get("session_name") or s.get("name") or "",
            "type": s.get("type"),
            "created_at": created_at,
            "updated_at": updated_at,
        })
    return SessionListResponse(user_id=user_id, sessions=sessions_norm)

# DETAIL: GET /api/sessions/{id}
@router.get("/{session_id}", response_model=HistoryResponse)
async def get_session_detail(session_id: int = Path(...)):
    from ...core.database import get_db_connection
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Không thể kết nối DB.")
    try:
        with conn.cursor() as cur:
            # Detect time column in message table
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name='message' AND table_schema='public';
                """
            )
            cols = {r[0] for r in cur.fetchall()}
            time_col = 'timestamp' if 'timestamp' in cols else ('created_at' if 'created_at' in cols else None)

            if time_col:
                cur.execute(
                    f"""
                    SELECT id, type, content, "order", {time_col}
                    FROM message
                    WHERE session_id = %s
                    ORDER BY "order" ASC;
                    """,
                    (session_id,)
                )
            else:
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
            if 'ts' not in locals():
                # populate messages based on whether time_col exists
                if time_col:
                    for rid, mtype, content, order, tsv in rows:
                        ts_utc7 = (tsv + timedelta(hours=7)) if tsv else None
                        messages.append({
                            "id": rid,
                            "order": order,
                            "type": 'human' if mtype == 'human' else 'ai',
                            "content": content,
                            "created_at": ts_utc7,
                        })
                else:
                    now_utc7 = datetime.now(timezone.utc) + timedelta(hours=7)
                    for rid, mtype, content, order in rows:
                        messages.append({
                            "id": rid,
                            "order": order,
                            "type": 'human' if mtype == 'human' else 'ai',
                            "content": content,
                            "created_at": now_utc7,
                        })
    finally:
        if conn:
            conn.close()
    return HistoryResponse(session_id=session_id, messages=messages)

# UPDATE: PUT /api/sessions/{id} (toàn phần) và PATCH /api/sessions/{id} (một phần)
from pydantic import BaseModel
class SessionUpdateRequest(BaseModel):
    name: Optional[str] = None
    context: Optional[dict] = None

@router.put("/{session_id}", response_model=SessionInfo)
async def put_session(session_id: int, request: SessionUpdateRequest = Body(...)):
    updated = False
    if request.name:
        updated = rename_session(session_id, request.name)
    # TODO: cập nhật context nếu cần (cần hàm update_context trong session_manager)
    if not updated and not request.context:
        raise HTTPException(status_code=400, detail="Không có trường nào để cập nhật.")
    info = get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên.")
    # convert UTC -> UTC+7
    created_utc7 = (info["created_at"] + timedelta(hours=7)) if info.get("created_at") else None
    updated_utc7 = (info["updated_at"] + timedelta(hours=7)) if info.get("updated_at") else None
    return SessionInfo(id=info["id"], session_name=info["name"], type=info.get("type"), created_at=created_utc7, updated_at=updated_utc7)

@router.patch("/{session_id}", response_model=SessionInfo)
async def patch_session(session_id: int, request: SessionUpdateRequest = Body(...)):
    return await put_session(session_id, request)

# DELETE: DELETE /api/sessions/{id}
@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session_endpoint(session_id: int = Path(...)):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Phiên không tồn tại hoặc đã bị xóa.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)