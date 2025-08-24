from fastapi import APIRouter, Body, HTTPException
from ..schemas import ChatRequest, ChatResponse, ChatEditRequest, ChatMultiResponse, ChatMultiResult
from ...features.qna.agent import initialize_qna_agent
from ...features.planner.agent import initialize_planning_agent
from ...features.learning.agent import initialize_learning_agent
from ...features.reviewer.agent import initialize_reviewer_agent
from ...features.speaking.agent import initialize_speaking_agent
from ...features.planner.tools import set_session_user_id
from ...core.session_manager import add_new_messages, load_session_data, rewind_last_turn
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import re
from typing import Optional

router = APIRouter()

DEFAULT_QNA_LINK = "/chatbot"

qna_agent_executor = initialize_qna_agent()
planner_agent_executor = initialize_planning_agent()
learning_agent_executor = initialize_learning_agent()
reviewer_agent_executor = initialize_reviewer_agent()
speaking_agent_executor = initialize_speaking_agent()

def split_user_requests(user_input: str) -> list:
    parts = re.split(r'[\n.;!?]+', user_input)
    return [p.strip() for p in parts if p.strip()]

def detect_intent_llm_multi(user_input: str) -> list:
    llm = ChatOpenAI(temperature=0, model="gpt-4.1")
    prompt = (
        "Bạn là hệ thống phân tích yêu cầu người dùng cho chatbot đa năng. "
        "Hãy đọc đoạn sau, tách thành từng yêu cầu nhỏ (nếu có), với mỗi yêu cầu hãy trả về JSON gồm: "
        "{intent, ai_response, redirect_link}. "
        "Nếu yêu cầu là tạo lộ trình, không biết học gì, nên học gì hay những bên tương tự thì, hãy trả về intent là planner, redirect_link là /roadmap. "
        "Các intent hợp lệ: qna, planner, speaking, reviewer, learning. "
        f"Nếu intent là planner thì redirect_link là /roadmap, qna là {DEFAULT_QNA_LINK}, speaking là /speaking, reviewer là /review, learning là /learning. "
        f"Nếu không xác định được intent thì để là qna. "
        "Đầu ra là 1 mảng JSON.\n"
        f"Đầu vào: {user_input}"
    )
    result = llm.invoke(prompt)
    import json
    try:
        parsed = json.loads(result.content)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [{"intent": "qna", "ai_response": result.content, "redirect_link": DEFAULT_QNA_LINK}]

ROADMAP_KEYWORDS = [
    "lộ trình", "roadmap", "kế hoạch học", "plan học", "học gì", "nên học", "học như thế nào",
    "học jlpt", "thi jlpt", "học n3", "học n2", "học n1", "làm sao để thi", "chuẩn bị thi"
]

def detect_intent_with_keywords(user_input: str) -> Optional[str]:
    user_input_lower = user_input.lower()
    if any(kw in user_input_lower for kw in ROADMAP_KEYWORDS):
        return "planner"
    return None

def get_intent_and_redirect(req: str, session_type: Optional[str]) -> tuple[str, str, str]:
    """Xác định intent, redirect_link và ai_response mặc định."""
    intent = detect_intent_with_keywords(req)

    if intent == "planner":
        redirect_link = "/roadmap"
        ai_response = "Bạn đang hỏi về lộ trình học. Hệ thống sẽ chuyển sang chế độ lập kế hoạch học tập."
    else:
        detected = detect_intent_llm_multi(req)
        d = detected[0] if detected else {
            "intent": "qna",
            "ai_response": "Xin hãy cung cấp thêm thông tin.",
            "redirect_link": DEFAULT_QNA_LINK
        }
        intent = d.get("intent", "qna")
        ai_response = d.get("ai_response", "Xin hãy cung cấp thêm thông tin.")
        redirect_link = d.get("redirect_link", None)

    if session_type:
        session_intent = session_type.lower()
        intent = session_intent
        default_links = {
            "planner": "/roadmap",
            "reviewer": "/review",
            "learning": "/learning",
            "speaking": "/speaking",
            "qna": DEFAULT_QNA_LINK,
        }
        redirect_link = default_links.get(session_intent)

    if not intent:
        intent = "qna"

    return intent, redirect_link, ai_response

@router.post("/chat", response_model=ChatMultiResponse, include_in_schema=False)
async def chat_dispatcher(request: ChatRequest = Body(...)):
    session_id = request.session_id
    user_input = request.user_input

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id là bắt buộc.")

    session_data = load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên ID {session_id} không tồn tại.")

    user_id = session_data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=500, detail=f"Không tìm thấy user_id trong dữ liệu phiên {session_id}.")

    session_type = session_data.get("type")
    chat_history = session_data.get("history", [])
    results = []

    for req in split_user_requests(user_input):
        intent, redirect_link, ai_response = get_intent_and_redirect(req, session_type)

        agent_map = {
            "qna": qna_agent_executor,
            "planner": planner_agent_executor,
            "learning": learning_agent_executor,
            "reviewer": reviewer_agent_executor,
            "speaking": speaking_agent_executor,
        }
        agent = agent_map.get(intent)
        if not agent:
            results.append(ChatMultiResult(intent=intent, ai_response="Xin lỗi, tôi chưa hỗ trợ chức năng này.", redirect_link=redirect_link))
            continue

        input_data = {
            "user_id": user_id,
            "input": req,
            "chat_history": chat_history,
            "context": session_data.get("context", {})
        }
        if intent == "planner":
            set_session_user_id(user_id)
        ai_result = agent.invoke(input_data)
        ai_response_final = ai_result.get("output", ai_response)
        human_msg = HumanMessage(content=req)
        ai_msg = AIMessage(content=ai_response_final)
        add_new_messages(session_id, [human_msg, ai_msg])
        chat_history.extend([human_msg, ai_msg])

        results.append(ChatMultiResult(intent=intent, ai_response=ai_response_final, redirect_link=redirect_link))

    return ChatMultiResponse(session_id=session_id, results=results)

@router.post("/chat/edit_and_resubmit", response_model=ChatResponse, include_in_schema=False)
async def edit_and_resubmit_message(request: ChatEditRequest = Body(...)):
    success = rewind_last_turn(request.session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phiên {request.session_id} hoặc không đủ tin nhắn để sửa.")
    session_data = load_session_data(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên ID {request.session_id} không tồn tại.")
    session_type = session_data.get("type", "qna").lower()
    input_data = {
        "user_id": session_data["user_id"],
        "input": request.corrected_input,
        "chat_history": session_data["history"]
    }
    agent_map = {
        "qna": initialize_qna_agent(),
        "planner": initialize_planning_agent(),
        "learning": initialize_learning_agent(),
        "reviewer": initialize_reviewer_agent(),
        "speaking": initialize_speaking_agent(),
    }
    agent = agent_map.get(session_type, agent_map["qna"])
    result = agent.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: Agent không có output.")
    human_msg = HumanMessage(content=request.corrected_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(request.session_id, [human_msg, ai_msg])
    return ChatResponse(session_id=request.session_id, ai_response=ai_response_text)
