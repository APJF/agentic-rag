# src/api/main.py

from fastapi import FastAPI
from .endpoints import chat,sessions # Import router từ file chat.py

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Trợ lý ảo Tiếng Nhật API",
    description="API cho phép tương tác với hệ thống agent đa chức năng.",
    version="1.0.0"
)

# Gắn router của chat vào ứng dụng chính với tiền tố /chat
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "API của Trợ lý ảo Tiếng Nhật đã sẵn sàng!"}