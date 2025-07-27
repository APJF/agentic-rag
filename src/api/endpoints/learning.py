# src/api/endpoints/learning.py

from fastapi import APIRouter, Body, HTTPException, status
from ..schemas import ChatRequest, ChatResponse

# Import các thành phần cần thiết
from ...features.learning.agent import initialize_learning_agent
from ...core.session_manager import load_session_data, add_new_messages
from langchain_core.messages import HumanMessage, AIMessage

# Khởi tạo LearningAgent một lần duy nhất
learning_agent_executor = initialize_learning_agent()

router = APIRouter()

@router.post("/invoke", response_model=ChatResponse)
async def invoke_learning_agent(request: ChatRequest = Body(...)):
    """
    Endpoint chuyên dụng để tương tác với LearningAgent trong ngữ cảnh một bài học (Material).
    """
    session_id = request.session_id
    user_input = request.user_input

    # Tải toàn bộ dữ liệu của phiên, bao gồm cả context
    session_data = load_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên ID {session_id} không tồn tại.")

    # Kiểm tra xem có đúng là phiên STUDY không
    if session_data.get("type") != "STUDY":
        raise HTTPException(status_code=400, detail="Endpoint này chỉ dành cho các phiên học tập (STUDY).")

    print(f"--- Learning API nhận input: '{user_input}' cho session '{session_id}' ---")

    # Chuẩn bị dữ liệu và gọi LearningAgent
    input_data = {
        "user_id": session_data["user_id"],
        "input": user_input,
        "chat_history": session_data["history"],
        # Truyền context (chứa material_id) vào cho agent
        "context": session_data["context"]
    }
    result = learning_agent_executor.invoke(input_data)
    ai_response_text = result.get('output', "Lỗi: LearningAgent không có output.")

    # Cập nhật và lưu lại lịch sử
    human_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=ai_response_text)
    add_new_messages(session_id, [human_msg, ai_msg])

    return ChatResponse(
        session_id=session_id,
        ai_response=ai_response_text
    )
