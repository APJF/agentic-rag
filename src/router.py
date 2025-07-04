# src/router.py

from langchain_core.runnables import Runnable, RunnableLambda
from typing import Dict, Any

# Import các thành phần của hệ thống
from src.features.planner.agent import initialize_planning_agent
from src.features.quizzer.agent import initialize_quiz_agent
from src.features.corrector.service import correct_japanese_text
from src.features.qna.service import answer_question
# Import hàm quản lý context mà chúng ta đã tạo
from src.core.context_manager import load_task_context

# --- 1. KHỞI TẠO CÁC AGENT/SERVICE CHUYÊN MÔN ---
print("Initializing specialist agents for the router...")
planner_agent = initialize_planning_agent()
quiz_agent = initialize_quiz_agent()
print("All specialist agents initialized.")


# --- 2. BỘ NÃO ĐIỀU PHỐI MỚI (CÓ TRÍ NHỚ) ---
def route(info: Dict[str, Any]) -> Runnable:
    """
    Hàm định tuyến chính, có khả năng kiểm tra các nhiệm vụ đang dang dở.
    """
    user_input = info.get("input", "").lower()
    session_id = info.get("session_id")

    # === ƯU TIÊN SỐ 1: KIỂM TRA NHIỆM VỤ ĐANG DANG DỞ ===
    if session_id:
        # Kiểm tra xem có đang tư vấn lộ trình dở không
        ongoing_plan_context = load_task_context(session_id, "PLANNER")
        if ongoing_plan_context:
            print("[Router] Phát hiện context PLANNER đang dang dở. Ưu tiên tiếp tục.")
            return planner_agent

        # (Trong tương lai, bạn có thể thêm kiểm tra cho QUIZ ở đây)
        # ongoing_quiz_context = load_task_context(session_id, "QUIZ")
        # if ongoing_quiz_context:
        #     return quiz_agent

    # === ƯU TIÊN SỐ 2: KIỂM TRA TỪ KHÓA ĐỂ BẮT ĐẦU NHIỆM VỤ MỚI ===
    if "lộ trình" in user_input or "kế hoạch" in user_input or "tư vấn" in user_input:
        print("[Router] Phát hiện ý định bắt đầu PLANNER mới.")
        return planner_agent
    if "quiz" in user_input or "câu hỏi" in user_input or "bài tập" in user_input:
        print("[Router] Phát hiện ý định bắt đầu QUIZ mới.")
        return quiz_agent
    if "sửa lỗi" in user_input or "sửa câu" in user_input:
         # Giả sử bạn có một corrector_chain
        print("[Router] Phát hiện ý định CORRECTOR.")
        return RunnableLambda(lambda x: correct_japanese_text(x["input"], "N4"))


    # === LỰA CHỌN MẶC ĐỊNH: CHAT THÔNG THƯỜNG ===
    print("[Router] Không có ý định rõ ràng, chuyển đến CHAT.")
    return RunnableLambda(lambda x: answer_question(x["input"]))


# --- 3. AGENTIC ROUTER CHAIN HOÀN CHỈNH ---
# Sử dụng RunnableLambda để bọc logic định tuyến phức tạp của chúng ta
agentic_router_chain = RunnableLambda(route)