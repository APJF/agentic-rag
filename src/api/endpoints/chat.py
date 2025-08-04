# src/api/endpoints/chat.py

from fastapi import APIRouter, Body, HTTPException
from ..schemas import ChatRequest, ChatResponse, ChatEditRequest, ChatInitiateRequest, ChatInitiateResponse
from ...features.qna.agent import initialize_qna_agent
from ...features.planner.agent import initialize_planning_agent # Cần để điều phối
from ...core.session_manager import load_session_data, add_new_messages, rewind_last_turn, create_new_session
from ...core.llm import get_llm # Cần để tự đặt tên session
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Định nghĩa hằng số cho string literals
ERROR_NO_AGENT_OUTPUT = "Lỗi: Agent không có output."
DEFAULT_SESSION_NAME = "Cuộc trò chuyện mới"
SESSION_NOT_FOUND_ERROR = "Phiên ID {session_id} không tồn tại."
SESSION_CREATE_ERROR = "Không thể tạo phiên mới."

qna_agent_executor = initialize_qna_agent()
planner_agent_executor = initialize_planning_agent()
llm_instance = get_llm()

router = APIRouter()


async def generate_session_name(first_message: str) -> str:
    """Dùng AI để tóm tắt tin nhắn đầu tiên thành một tiêu đề ngắn."""
    if not llm_instance:
        return DEFAULT_SESSION_NAME
    prompt = ChatPromptTemplate.from_template("Hãy tóm tắt câu sau thành một tiêu đề không quá 5 từ: '{text}'")
    chain = prompt | llm_instance | StrOutputParser()
    return await chain.ainvoke({"text": first_message})


def _validate_session_exists(session_id: int):
    """Kiểm tra session có tồn tại không, nếu không sẽ raise HTTPException."""
    session_data = load_session_data(session_id)
    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=SESSION_NOT_FOUND_ERROR.format(session_id=session_id)
        )
    return session_data


def _process_agent_request(user_id: str, user_input: str, chat_history: list, session_type: str = "GENERAL") -> str:
    """Xử lý request với agent và trả về response text."""
    input_data = {
        "user_id": user_id,
        "input": user_input,
        "chat_history": chat_history
    }

    if session_type.upper() == "PLANNER":
        result = planner_agent_executor.invoke(input_data)
    else:  # Mặc định xử lý bằng QnA Agent
        result = qna_agent_executor.invoke(input_data)

    return result.get('output', ERROR_NO_AGENT_OUTPUT)


def _save_conversation_turn(session_id: int, user_input: str, ai_response: str):
    """Lưu một lượt hội thoại (user message + AI response) vào session."""
    human_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=ai_response)
    add_new_messages(session_id, [human_msg, ai_msg])


@router.post("/initiate_and_invoke", response_model=ChatInitiateResponse)
async def initiate_and_invoke(request: ChatInitiateRequest = Body(...)):
    """
    Endpoint "tất cả trong một": Tạo session mới, lưu tin nhắn đầu tiên,
    xử lý và trả về câu trả lời đầu tiên.
    """
    session_type = request.session_type.upper()

    # 1. Tự động tạo tên session
    if session_type == "GENERAL":
        session_name = await generate_session_name(request.first_message)
    elif session_type == "PLANNER":
        session_name = "Tư vấn Lộ trình học"
    else:
        # (Thêm logic đặt tên cho STUDY và EXAM_REVIEW ở đây)
        session_name = f"Phiên {session_type}"

    # 2. Tạo session mới trong DB
    session_id = create_new_session(
        user_id=request.user_id,
        session_name=session_name,
        session_type=session_type,
        context=request.context
    )
    if not session_id:
        raise HTTPException(status_code=500, detail=SESSION_CREATE_ERROR)

    # 3. Lưu tin nhắn đầu tiên của người dùng
    human_msg = HumanMessage(content=request.first_message)
    add_new_messages(session_id, [human_msg])

    # 4. Điều phối và xử lý tin nhắn đầu tiên
    chat_history = [human_msg]  # Lịch sử ban đầu chỉ có 1 tin nhắn
    ai_response_text = _process_agent_request(
        user_id=request.user_id,
        user_input=request.first_message,
        chat_history=chat_history,
        session_type=session_type
    )

    # 5. Lưu tin nhắn trả lời của AI
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [ai_msg])

    return ChatInitiateResponse(
        session_id=session_id,
        session_name=session_name,
        ai_first_response=ai_response_text
    )


@router.post("/invoke", response_model=ChatResponse)
async def invoke_assistant(request: ChatRequest):
    """Xử lý tin nhắn tiếp theo trong session đã có."""
    session_data = _validate_session_exists(request.session_id)

    ai_response_text = _process_agent_request(
        user_id=session_data["user_id"],
        user_input=request.user_input,
        chat_history=session_data["history"]
    )

    _save_conversation_turn(request.session_id, request.user_input, ai_response_text)

    return ChatResponse(session_id=request.session_id, ai_response=ai_response_text)


@router.post("/edit_and_resubmit", response_model=ChatResponse)
async def edit_and_resubmit_message(request: ChatEditRequest = Body(...)):
    """Sửa và gửi lại tin nhắn cuối cùng."""
    success = rewind_last_turn(request.session_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy phiên {request.session_id} hoặc không đủ tin nhắn để sửa."
        )

    session_data = _validate_session_exists(request.session_id)

    ai_response_text = _process_agent_request(
        user_id=session_data["user_id"],
        user_input=request.corrected_input,
        chat_history=session_data["history"]
    )

    _save_conversation_turn(request.session_id, request.corrected_input, ai_response_text)

    return ChatResponse(session_id=request.session_id, ai_response=ai_response_text)
