# src/api/main.py

from fastapi import FastAPI
# === BƯỚC 1: IMPORT CORSMiddleware ===
from fastapi.middleware.cors import CORSMiddleware

# Giả sử bạn có các router này
from .endpoints import chat, sessions, planner, learning, reviewer, speaking, dispatcher

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Trợ lý ảo Tiếng Nhật API",
    description="API cho phép tương tác với hệ thống agent đa chức năng.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(sessions.router, prefix="/sessions", tags=["Session Management"])
app.include_router(planner.router, prefix="/planner", tags=["Learning Path Planner"])
app.include_router(dispatcher.router, prefix="", tags=["Dispatcher"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "API của Trợ lý ảo Tiếng Nhật đã sẵn sàng!"}