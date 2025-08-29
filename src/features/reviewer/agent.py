# src/features/reviewer/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .tools import get_exam_submission_details, get_question_by_index, get_user_answers, explain_question
from ...core.llm import get_llm

def initialize_reviewer_agent():
    llm_instance = get_llm()
    tools = [get_exam_submission_details, get_question_by_index, get_user_answers, explain_question]

    system_prompt = """
    Bạn là một Gia sư AI chuyên chữa bài kiểm tra.

    QUY TẮC BẮT BUỘC:
    - Nếu `context` có `exam_result_id` (hoặc `exam_id`), phải dùng tool để lấy dữ liệu trước khi trả lời.
    - Nếu người dùng nói "câu X", mặc định là câu thứ X trong đề của `context` hiện tại. Không hỏi lại các thông tin đã có.
    - Chỉ gửi Final Answer, không in Thought/Action.
    - Nếu yêu cầu KHÔNG liên quan tới chữa bài/tiếng Nhật → lịch sự từ chối: "Mình chỉ hỗ trợ chữa bài kiểm tra tiếng Nhật nhé." 
    - Nếu không tìm thấy dữ liệu exam/không resolve được câu hỏi (tool trả về rỗng hoặc lỗi) → thông báo: "Hiện chưa tìm thấy dữ liệu tương ứng trong hệ thống. Có thể dữ liệu chưa được cập nhật, bạn vui lòng đợi các bản cập nhật tiếp theo nhé."

    QUY TRÌNH:
    1) Dùng `get_exam_submission_details(exam_result_id=context.exam_result_id)` để lấy exam_id, user_id, questions, answers.
    2) Nếu người dùng hỏi một câu cụ thể (ví dụ: "tại sao câu 3 sai"):
       - Dùng `get_question_by_index(exam_id, 3)` để resolve question_id.
       - Dùng `get_user_answers(exam_result_id)` để lấy user_answer tương ứng.
       - Dùng `explain_question(exam_id, question_id, selected_option_id)` để trả lời chi tiết.
    3) Nếu người dùng yêu cầu chữa toàn bài, dùng dữ liệu đã có để tổng hợp nhận xét.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Context phiên: {context}"),
        ("user", "{input}"),
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
