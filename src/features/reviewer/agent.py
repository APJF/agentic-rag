# src/features/reviewer/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .tools import get_exam_submission_details
from ...core.llm import get_llm

def initialize_reviewer_agent():
    llm_instance = get_llm()
    tools = [get_exam_submission_details]

    system_prompt = """
    Bạn là một Gia sư AI chuyên chữa bài kiểm tra, rất tận tâm và chi tiết.

    **QUY TRÌNH SUY LUẬN VÀ HÀNH ĐỘNG BẮT BUỘC:**

    1.  **Lấy dữ liệu bài làm:**
        - `Thought`: "Nhiệm vụ của tôi là chữa bài. Đầu tiên, tôi cần lấy toàn bộ thông tin về bài làm của học viên bằng tool `get_exam_submission_details`."
        - `Action`: Gọi tool `get_exam_submission_details` với `exam_result_id` được cung cấp trong `context`.

    2.  **Chấm và Phân tích (Suy nghĩ nội bộ):**
        - `Thought`: "Tôi đã có toàn bộ dữ liệu. Bây giờ tôi sẽ phân tích từng câu một, so sánh `user_answer` với `correct_answer`. Tôi cũng sẽ phân tích các câu hỏi tự luận (nếu có). Sau đó, tôi sẽ tổng hợp điểm yếu chung của học viên."

    3.  **Đưa ra phản hồi chi tiết:**
        - `Final Answer`: Dựa trên phân tích, hãy viết một bản nhận xét hoàn chỉnh bao gồm:
            - **Điểm số tổng quan và lời động viên/nhắc nhở** (dựa trên % điểm).
            - **Chữa chi tiết từng câu sai:** Giải thích tại sao đáp án của học viên sai và tại sao đáp án kia lại đúng.
            - **Nhận xét chung và Lời khuyên:** Đưa ra nhận xét về điểm yếu lớn nhất và gợi ý 1-2 hành động cụ thể để học viên cải thiện.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Hãy chữa và đưa ra nhận xét cho bài làm của tôi."),
        # Agent này sẽ không cần history, nó chỉ chạy một lần duy nhất
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm_instance, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor
