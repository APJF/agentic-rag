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
    """
    Endpoint tạo session mới, bắt buộc có tin nhắn đầu tiên, tự động sinh tên session, lưu tin nhắn đầu tiên và phản hồi AI.
    """
    get_user(int(request.user_id))

    # 1. Xác định intent bằng rule + LLM
    def detect_intent_custom(user_input: str, context: dict = None) -> str:
        # Ưu tiên context nếu có session_type
        if context and context.get("session_type"):
            return str(context["session_type"]).lower()
        # Ưu tiên context nếu có
        if context:
            if context.get("course_id"):
                return "learning"
            if context.get("exam_id") or context.get("exam_result_id") or context.get("question_id"):
                return "reviewer"
        # Rule: Nếu có mã môn học (ví dụ: M101, N3-01, ...), gán learning
        if re.search(r'\b(M\d{3}|N\d-\d{2}|course_id\s*:\s*\w+)\b', user_input, re.IGNORECASE):
            return "learning"
        # Rule: Nếu có từ khóa chữa/sửa/dịch câu + gắn với bài kiểm tra cụ thể
        if re.search(r'(chữa|sửa|giải thích|dịch).*câu.*(bài kiểm tra|exam|exam_id|exam_result_id|question_id)', user_input, re.IGNORECASE):
            return "reviewer"
        # Rule: Nếu chỉ có từ khóa dịch/sửa/chữa câu (không gắn bài kiểm tra), để qna
        if re.search(r'(chữa|sửa|giải thích|dịch).*câu', user_input, re.IGNORECASE):
            return "qna"
        # Rule: Nếu có từ khóa lộ trình, roadmap, học gì, thi JLPT... thì planner
        roadmap_keywords = [
            "lộ trình", "roadmap", "kế hoạch học", "plan học", "học gì", "nên học", "học như thế nào",
            "học jlpt", "thi jlpt", "học n3", "học n2", "học n1", "làm sao để thi", "chuẩn bị thi"
        ]
        user_input_lower = user_input.lower()
        if any(kw in user_input_lower for kw in roadmap_keywords):
            return "planner"
        # Nếu không match gì, fallback sang LLM hoặc qna
        return "qna"

    intent = detect_intent_custom(request.first_message, request.context)

    # 2. Sinh tên session tự động (có thể dùng LLM hoặc rule đơn giản)
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

        # Mặc định: dùng LLM tóm tắt tin nhắn thành tiêu đề ≤5 từ
        llm_instance = get_llm()
        if llm_instance:
            prompt = ChatPromptTemplate.from_template("Hãy tóm tắt câu sau thành một tiêu đề không quá 5 từ: '{text}'")
            chain = prompt | llm_instance | StrOutputParser()
            try:
                return await chain.ainvoke({"text": first_message})
            except Exception:
                pass  # fallback bên dưới
        return f"Chat: {first_message[:20]}"

    session_name = await generate_session_name(request.first_message, intent)

    # 3. Tạo session mới trong DB
    session_id = create_new_session(
        user_id=int(request.user_id),
        session_name=session_name,
        session_type=intent,
        context=request.context
    )
    if not session_id:
        raise HTTPException(status_code=500, detail="Không thể tạo phiên mới.")

    # 4. Lưu tin nhắn đầu tiên của người dùng
    human_msg = HumanMessage(content=request.first_message)
    add_new_messages(session_id, [human_msg])

    # 5. Gọi agent phù hợp để lấy phản hồi AI đầu tiên
    from ...features.planner.tools import set_session_user_id

    # Đảm bảo Planner tools nhận đúng user_id
    if intent == "planner":
        set_session_user_id(int(request.user_id))

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
        "chat_history": [human_msg]
    }
    result = agent.invoke(input_data)
    ai_response_text = result.get('output', "Xin hãy cung cấp thêm thông tin.")

    # 6. Lưu tin nhắn trả lời của AI
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [ai_msg])

    # 7. Trả về session_id, session_name, ai_response
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