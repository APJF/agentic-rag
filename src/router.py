    # src/router.py
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from typing import Dict, Any, List

# Import các thành phần của hệ thống
# Giả định các file này đã được di chuyển vào cấu trúc features/
from src.features.planner.agent import initialize_planning_agent
from src.features.quizzer.agent import initialize_quiz_agent
from src.features.corrector.service import correct_japanese_text
from src.features.qna.service import answer_question
from src.core.llm import get_llm
from src.core.session_manager import format_history_for_prompt

# Khoi tao Agent chuyen mon
print("Initializing specialist agents for the router...")
planner_agent = initialize_planning_agent()
quiz_agent = initialize_quiz_agent()
print("All specialist agents initialized for the router.")


# CHAIN phân loai
classification_llm = get_llm()
if not classification_llm:
    raise ValueError("LLM chưa được cấu tạo. Check settings.")

router_prompt_text = """Bạn là một tổng đài viên thông minh, nhiệm vụ của bạn là định tuyến yêu cầu của người dùng đến đúng phòng ban dựa trên cả LỊCH SỬ TRÒ CHUYỆN và YÊU CẦU MỚI NHẤT.
    Hãy chọn MỘT trong các phòng ban sau: PLANNER, QUIZ, CORRECTOR, CHAT.
    
    QUY TẮC ĐỊNH TUYẾN:
    - Nếu trong lịch sử trò chuyện cho thấy đang có một phiên tư vấn lộ trình dở dang (ví dụ: trợ lý vừa hỏi về level, mục tiêu...), hãy tiếp tục định tuyến đến PLANNER.
    - Chọn PLANNER nếu người dùng muốn tạo lộ trình, kế hoạch học tập.
    - Chọn QUIZ nếu người dùng muốn làm bài kiểm tra, câu hỏi trắc nghiệm.
    - Chọn CORRECTOR nếu người dùng yêu cầu sửa lỗi câu văn.
    - Với tất cả các trường hợp còn lại, chọn CHAT.
    
    Chỉ trả về MỘT từ duy nhất là tên phòng ban.
    
    LỊCH SỬ TRÒ CHUYỆN:
    {chat_history}
    ---
    YÊU CẦU MỚI NHẤT CỦA NGƯỜI DÙNG:
    {input}
    ---
    PHÒNG BAN ĐƯỢC CHỌN:"""

classification_prompt = ChatPromptTemplate.from_template(router_prompt_text)

classification_chain = (
        {"input": itemgetter("input"), "chat_history": lambda x: format_history_for_prompt(x.get("chat_history", []))}
        | classification_prompt
        | classification_llm
        | StrOutputParser()
)


    # --- 3. BỌC CÁC SERVICE ĐƠN GIẢN THÀNH RUNNABLE ---
def run_corrector_service(input_dict: Dict[str, Any]) -> str:
    user_level = "N4" # Tạm thời hardcode
    return correct_japanese_text(input_dict["input"], user_level)

def run_qna_service(input_dict: Dict[str, Any]) -> str:
    return answer_question(input_dict["input"])

corrector_chain = RunnableLambda(run_corrector_service)
qna_chain = RunnableLambda(run_qna_service)


    # --- 4. TẠO BỘ ĐỊNH TUYẾN (RUNNABLEBRANCH) ---
    # Đây là nơi logic if/elif/else được định nghĩa
    # Nó nhận vào một dictionary có chứa 'topic' và 'input'
agentic_router_branch = RunnableBranch(
        # Mỗi điều kiện là một tuple: (hàm kiểm tra, chain/agent sẽ chạy nếu đúng)
        (lambda x: "PLANNER" in x["topic"].upper(), planner_agent),
        (lambda x: "QUIZ" in x["topic"].upper(), quiz_agent),
        (lambda x: "CORRECTOR" in x["topic"].upper(), corrector_chain),
        qna_chain  # Nhánh mặc định nếu không có điều kiện nào đúng
)


    # --- 5. TẠO AGENTIC ROUTER CHAIN HOÀN CHỈNH ---
    # Ghép nối tất cả các bước lại với nhau bằng LCEL (LangChain Expression Language)
agentic_router_chain = (
        # Bước 1: Gán kết quả phân loại từ `classification_chain` vào một key mới là 'topic'.
        # `RunnablePassthrough.assign` sẽ giữ lại các key ban đầu ('input', 'chat_history')
        # và thêm vào key 'topic'.
        RunnablePassthrough.assign(
            topic=classification_chain
        )
        # Bước 2: Kết quả của bước 1 (một dictionary chứa input, chat_history, và topic)
        # được đưa vào bộ định tuyến `agentic_router_branch`.
        | agentic_router_branch
)
