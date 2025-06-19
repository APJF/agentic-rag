# src/main_orchestrator.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from src.features.planner.agent import initialize_planning_agent
from src.features.quizzer.agent import initialize_quiz_agent
from src.features.corrector.service import correct_japanese_text
from src.features.qna.service import answer_question
from src.core.llm import get_llm

print("Initializing specialist agents...")
planner_agent_executor = initialize_planning_agent()
quiz_agent_executor = initialize_quiz_agent()
print("All specialist agents initialized.")

def format_history_for_prompt(chat_history: list) -> str:
    """Chuyển đổi danh sách các đối tượng Message thành một chuỗi đơn giản."""
    if not chat_history:
        return "Không có lịch sử trò chuyện."

    formatted_lines = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_lines.append(f"Người dùng: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_lines.append(f"Trợ lý: {message.content}")
    return "\n".join(formatted_lines)


def create_orchestrator_router():
    """Tạo ra router chain có khả năng đọc lịch sử chat."""
    llm = get_llm()
    if not llm:
        raise ValueError("LLM not configured.")

    router_prompt_text = """
    Bạn là một tổng đài viên thông minh, nhiệm vụ của bạn là định tuyến yêu cầu của người dùng đến đúng phòng ban dựa trên cả LỊCH SỬ TRÒ CHUYỆN và YÊU CẦU MỚI NHẤT.
    Hãy chọn MỘT trong các phòng ban sau: PLANNER, QUIZ, CORRECTOR, CHAT.

    QUY TẮC ĐỊNH TUYẾN:
    - Nếu trong lịch sử trò chuyện cho thấy đang có một phiên tư vấn lộ trình dở dang (ví dụ: trợ lý vừa hỏi về level, mục tiêu...), hãy tiếp tục định tuyến đến PLANNER, kể cả khi yêu cầu mới nhất của người dùng chỉ là một câu trả lời ngắn như "N4" hay "anime".
    - Chỉ chuyển sang một phòng ban mới khi người dùng rõ ràng bắt đầu một chủ đề mới không liên quan.
    - Chọn PLANNER nếu người dùng muốn tạo lộ trình, kế hoạch học tập.
    - Chọn QUIZ nếu người dùng muốn làm bài kiểm tra, câu hỏi trắc nghiệm.
    - Chọn CORRECTOR nếu người dùng yêu cầu sửa lỗi câu văn.
    - Với các trường hợp còn lại, chọn CHAT.

    Chỉ trả về MỘT từ duy nhất là tên phòng ban.

    ---
    LỊCH SỬ TRÒ CHUYỆN:
    {chat_history}
    ---
    YÊU CẦU MỚI NHẤT CỦA NGƯỜI DÙNG:
    {input}
    ---
    PHÒNG BAN ĐƯỢC CHỌN:
    """
    prompt = ChatPromptTemplate.from_template(router_prompt_text)
    return prompt | llm | StrOutputParser()

router_chain = create_orchestrator_router()

def run_orchestrator(user_input: str, chat_history: list):
    """
    Hàm điều phối chính, giờ đây sẽ sử dụng cả lịch sử chat để định tuyến.
    """
    print(f"--- Orchestrator nhận input: '{user_input}' ---")

    history_for_router = format_history_for_prompt(chat_history)
    intent = router_chain.invoke({
        "input": user_input,
        "chat_history": history_for_router
    }).strip().upper()
    print(f"--- Router quyết định chuyển tới: {intent} ---")

    if intent == "PLANNER":
        return planner_agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        }).get("output")
    elif intent == "QUIZ":
        return quiz_agent_executor.invoke({"input": user_input}).get("output")
    elif intent == "CORRECTOR":
        user_level = "N4"
        return correct_japanese_text(user_input, user_level)
    else:
        return answer_question(user_input)
