# src/api/main.py
from fastapi import FastAPI
from src.api.endpoints import chat as chat_router # Import router từ file chat.py
# Nếu có các router khác, bạn cũng import chúng ở đây
# from src.api.endpoints import document_management as doc_router
from src.config import settings # Để tiện truy cập các cài đặt nếu cần

app = FastAPI(
    title="API cho Hệ thống Agentic RAG Học Tiếng Nhật",
    description="Cung cấp các endpoint để tương tác với Agentic RAG, bao gồm chức năng chat, tra cứu, và các gợi ý học tập.",
    version="0.1.0",
    # docs_url="/documentation", # Tùy chỉnh đường dẫn Swagger UI
    # redoc_url="/redocumentation", # Tùy chỉnh đường dẫn ReDoc
)

# Include router cho chat
# Tiền tố "/api/v1" giúp quản lý phiên bản API
app.include_router(chat_router.router, prefix="/api/v1", tags=["Chat & Q&A"])

# (Tùy chọn) Include các router khác
# app.include_router(doc_router.router, prefix="/api/v1/documents", tags=["Document Management"])

@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Chào mừng đến với API của Hệ thống Agentic RAG Học Tiếng Nhật!",
        "documentation_swagger": "/docs",
        "documentation_redoc": "/redoc"
    }

# (Tùy chọn) Khởi tạo các service/resource cần thiết khi ứng dụng bắt đầu
# @app.on_event("startup")
# async def startup_event():
#     print("API bắt đầu chạy. Khởi tạo resources...")
#     # Ví dụ: tải model, kết nối DB nếu chưa được quản lý ở module khác
#     # from src.core.llm_services import get_llm # Đảm bảo LLM được tải
#     # get_llm()
#     # from src.core.embedding_manager import get_embedding_model
#     # get_embedding_model()
#     print("Resources đã sẵn sàng.")

# @app.on_event("shutdown")
# async def shutdown_event():
#     print("API đang tắt. Dọn dẹp resources...")
#     # Ví dụ: đóng kết nối DB