from fastapi import APIRouter, Body, HTTPException
from ..schemas import ChatRequest, ChatResponse, ChatEditRequest
from ...features.qna.agent import initialize_qna_agent
from ...features.planner.agent import initialize_planning_agent
from ...features.learning.agent import initialize_learning_agent
from ...features.reviewer.agent import initialize_reviewer_agent
from ...features.speaking.agent import initialize_speaking_agent
from ...core.session_manager import create_new_session, add_new_messages, load_session_data, rewind_last_turn
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

router = APIRouter()

# Constants
VALID_INTENTS = ["qna", "planner", "speaking", "reviewer", "learning"]
DEFAULT_INTENT = "qna"
ERROR_NO_OUTPUT = "Lỗi: Agent không có output."
ERROR_SESSION_NOT_FOUND = "Phiên ID {session_id} không tồn tại."
ERROR_UNSUPPORTED_FEATURE = "Xin lỗi, tôi chưa hỗ trợ chức năng này."
DEFAULT_AI_RESPONSE = "Xin hãy cung cấp thêm thông tin."

qna_agent_executor = initialize_qna_agent()
planner_agent_executor = initialize_planning_agent()
learning_agent_executor = initialize_learning_agent()
reviewer_agent_executor = initialize_reviewer_agent()
speaking_agent_executor = initialize_speaking_agent()

# Agent mapping
AGENT_MAP = {
    "qna": qna_agent_executor,
    "planner": planner_agent_executor,
    "learning": learning_agent_executor,
    "reviewer": reviewer_agent_executor,
    "speaking": speaking_agent_executor,
}


def detect_intent_llm(user_input: str) -> str:
    """Detect user intent using LLM."""
    prompt = (
        "Bạn là hệ thống phân loại ý định cho chatbot đa năng. "
        "Hãy đọc câu hỏi sau và trả về 1 trong các intent sau: "
        "qna, planner, speaking, reviewer, learning.\n"
        "Câu hỏi: '{user_input}'\n"
        "Chỉ trả về đúng 1 từ khóa intent."
    )
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    result = llm.invoke(prompt.format(user_input=user_input))
    intent = result.content.strip().lower()
    return intent if intent in VALID_INTENTS else DEFAULT_INTENT


def _handle_redirect_request(request: ChatRequest) -> tuple[int, str, str]:
    """Handle redirect to different agent."""
    intent = request.redirect_to
    session_id = create_new_session(request.user_id, f"Session {intent}", session_type=intent)
    user_input = getattr(request, "original_question", request.user_input)
    return session_id, user_input, intent


def _check_intent_mismatch(user_input: str, session_type: str, session_id: int) -> dict | None:
    """Check if detected intent differs from current session type."""
    detected_intent = detect_intent_llm(user_input)
    if detected_intent and session_type and detected_intent != session_type.lower():
        return {
            "session_id": session_id,
            "ai_response": f"Bạn đang hỏi về chủ đề '{detected_intent}'. Hệ thống sẽ chuyển sang chế độ phù hợp.",
            "redirect_to": detected_intent,
            "original_question": user_input
        }
    return None


def _handle_existing_session(request: ChatRequest) -> tuple[int, str, str]:
    """Handle request for existing session."""
    session_id = request.session_id
    user_input = request.user_input

    session_data = load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=ERROR_SESSION_NOT_FOUND.format(session_id=session_id))

    session_type = session_data.get("type")

    # Check for intent mismatch
    if session_type:
        mismatch_response = _check_intent_mismatch(user_input, session_type, session_id)
        if mismatch_response:
            raise HTTPException(status_code=200, detail=mismatch_response)
        intent = session_type.lower()
    else:
        intent = detect_intent_llm(user_input)

    return session_id, user_input, intent


def _handle_new_session(request: ChatRequest) -> tuple[int, str, str]:
    """Handle request for new session."""
    user_input = request.user_input
    intent = detect_intent_llm(user_input)
    session_id = create_new_session(request.user_id, f"Session {intent}", session_type=intent)
    return session_id, user_input, intent


@router.post("/chat", response_model=ChatResponse)
async def chat_dispatcher(request: ChatRequest = Body(...)):
    """Main chat dispatcher with reduced cognitive complexity."""
    try:
        # Handle different request types
        if hasattr(request, "redirect_to") and getattr(request, "redirect_to", None):
            session_id, user_input, intent = _handle_redirect_request(request)
        elif request.session_id:
            session_id, user_input, intent = _handle_existing_session(request)
        else:
            session_id, user_input, intent = _handle_new_session(request)

        # Get appropriate agent
        agent = AGENT_MAP.get(intent)
        if not agent:
            return ChatResponse(session_id=session_id, ai_response=ERROR_UNSUPPORTED_FEATURE)

        # Process request
        context = request.model_dump()
        context["session_id"] = session_id
        ai_result = agent.invoke({"context": context})
        ai_response = ai_result.get("output", DEFAULT_AI_RESPONSE)

        # Save conversation
        add_new_messages(session_id, [
            HumanMessage(content=user_input),
            AIMessage(content=ai_response)
        ])

        return ChatResponse(session_id=session_id, ai_response=ai_response)

    except HTTPException as e:
        if e.status_code == 200:  # Intent mismatch response
            return e.detail
        raise


@router.post("/chat/edit_and_resubmit", response_model=ChatResponse)
async def edit_and_resubmit_message(request: ChatEditRequest = Body(...)):
    """Edit and resubmit the last message."""
    success = rewind_last_turn(request.session_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy phiên {request.session_id} hoặc không đủ tin nhắn để sửa."
        )

    session_data = load_session_data(request.session_id)
    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=ERROR_SESSION_NOT_FOUND.format(session_id=request.session_id)
        )

    session_type = session_data.get("type", DEFAULT_INTENT).lower()
    input_data = {
        "user_id": session_data["user_id"],
        "input": request.corrected_input,
        "chat_history": session_data["history"]
    }

    # Create fresh agent map for edit_and_resubmit
    agent_map = {
        "qna": initialize_qna_agent(),
        "planner": initialize_planning_agent(),
        "learning": initialize_learning_agent(),
        "reviewer": initialize_reviewer_agent(),
        "speaking": initialize_speaking_agent(),
    }

    agent = agent_map.get(session_type, agent_map[DEFAULT_INTENT])
    result = agent.invoke(input_data)
    ai_response_text = result.get('output', ERROR_NO_OUTPUT)

    human_msg = HumanMessage(content=request.corrected_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(request.session_id, [human_msg, ai_msg])

    return ChatResponse(session_id=request.session_id, ai_response=ai_response_text)