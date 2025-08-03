from fastapi import APIRouter, Body, HTTPException
from ..schemas import ChatRequest, ChatResponse, ChatEditRequest
from ...features.qna.agent import initialize_qna_agent
from ...features.planner.agent import initialize_planning_agent
from ...features.learning.agent import initialize_learning_agent
from ...features.reviewer.agent import initialize_reviewer_agent
from ...features.speaking.agent import initialize_speaking_agent
from ...core.session_manager import create_new_session, add_new_messages, load_session_data, rewind_last_turn
from ...core.database import execute_sql_query
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

router = APIRouter()

qna_agent_executor = initialize_qna_agent()
planner_agent_executor = initialize_planning_agent()
learning_agent_executor = initialize_learning_agent()
reviewer_agent_executor = initialize_reviewer_agent()
speaking_agent_executor = initialize_speaking_agent()

# Hàm detect intent bằng LLM thực tế (OpenAI, có thể thay model khác)
def detect_intent_llm(user_input: str) -> str:
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
    if intent not in ["qna", "planner", "speaking", "reviewer", "learning"]:
        return "qna"
    return intent

@router.post("/chat", response_model=ChatResponse)
async def chat_dispatcher(request: ChatRequest = Body(...)):
    session_id = request.session_id
    session_type = None
    # Nếu frontend gửi kèm redirect_to (chuyển agent), tạo session mới với intent mới và gửi lại câu hỏi gốc
    if hasattr(request, "redirect_to") and getattr(request, "redirect_to", None):
        intent = request.redirect_to
        session_id = create_new_session(request.user_id, f"Session {intent}", session_type=intent)
        user_input = getattr(request, "original_question", request.user_input)
    else:
        user_input = request.user_input
        if session_id:
            session_data = load_session_data(session_id)
            if not session_data:
                raise HTTPException(status_code=404, detail=f"Phiên ID {session_id} không tồn tại.")
            session_type = session_data.get("type", None)
            detected_intent = detect_intent_llm(user_input)
            if detected_intent and session_type and detected_intent != session_type.lower():
                return {
                    "session_id": session_id,
                    "ai_response": f"Bạn đang hỏi về chủ đề '{detected_intent}'. Hệ thống sẽ chuyển sang chế độ phù hợp.",
                    "redirect_to": detected_intent,
                    "original_question": user_input
                }
            intent = session_type.lower() if session_type else detected_intent
        else:
            intent = detect_intent_llm(user_input)
            session_id = create_new_session(request.user_id, f"Session {intent}", session_type=intent)
    agent_map = {
        "qna": qna_agent_executor,
        "planner": planner_agent_executor,
        "learning": learning_agent_executor,
        "reviewer": reviewer_agent_executor,
        "speaking": speaking_agent_executor,
    }
    agent = agent_map.get(intent)
    if not agent:
        return ChatResponse(session_id=session_id, ai_response="Xin lỗi, tôi chưa hỗ trợ chức năng này.")
    context = request.dict()
    context["session_id"] = session_id
    ai_result = agent.invoke({"context": context})
    ai_response = ai_result.get("output", "Xin hãy cung cấp thêm thông tin.")
    add_new_messages(session_id, [
        HumanMessage(content=user_input),
        AIMessage(content=ai_response)
    ])
    return ChatResponse(session_id=session_id, ai_response=ai_response)

@router.post("/chat/edit_and_resubmit", response_model=ChatResponse)
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