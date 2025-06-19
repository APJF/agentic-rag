# src/api/endpoints/chat.py
from fastapi import APIRouter, HTTPException, Body
from src.api.schemas import ChatRequest, ChatResponse
from src.core.rag_services import answer_question

router = APIRouter()

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Nhận câu hỏi và trả về câu trả lời từ RAG",
    description="Gửi một câu hỏi dạng text và nhận lại câu trả lời đã được xử lý bởi hệ thống RAG."
)
async def handle_chat_request(request_body: ChatRequest = Body(...)):
    """
    Xử lý yêu cầu chat:
    - Nhận **query** từ người dùng.
    - (Tùy chọn) Nhận **user_id**.
    - Gọi hàm RAG để lấy câu trả lời.
    - Trả về câu trả lời.
    """
    if not request_body.query or not request_body.query.strip():
        raise HTTPException(status_code=400, detail="Trường 'query' không được để trống.")

    try:
        # Gọi hàm RAG chính của bạn (đã được module hóa trong rag_services.py)
        response_text = answer_question(request_body.query)

        # Nếu bạn muốn xử lý user_id hoặc các thông tin khác, hãy thêm logic ở đây

        return ChatResponse(answer=response_text)
    except ValueError as ve: # Ví dụ bắt lỗi cụ thể từ RAG service
        print(f"ValueError during RAG processing: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Nên log lỗi này ra để debug
        print(f"Lỗi không xác định trong quá trình xử lý chat: {e}")
        # import traceback
        # traceback.print_exc() # Để debug chi tiết hơn
        raise HTTPException(status_code=500, detail="Đã có lỗi xảy ra trong quá trình xử lý yêu cầu của bạn.")

# Bạn có thể thêm các endpoints khác liên quan đến chat ở đây nếu cần