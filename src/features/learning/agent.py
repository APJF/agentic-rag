# src/features/learning/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from .tools import (
    get_material_context,
    get_material_question_by_index,
    explain_material_question,
    get_material_chunks,
    search_material_chunks,
    get_listening_script,
)
from ...core.llm import get_llm


def initialize_learning_agent():
    llm_instance = get_llm()
    tools = [
        get_material_context,
        get_material_chunks,
        search_material_chunks,
        get_material_question_by_index,
        explain_material_question,
        get_listening_script,
    ]

    system_prompt = """
    Bạn là Trợ lý Học tập theo tài liệu (material).

    QUY TẮC BẮT BUỘC:
    - Nếu `context.material_id` tồn tại (chuỗi RAG), PHẢI dùng tool để lấy ngữ cảnh và chunks theo material_id từ DB.
    - Nếu người dùng nói "câu X" thì mặc định là câu X trong material hiện tại, không hỏi lại.
    - Chỉ gửi Final Answer; không in Thought/Action.

    QUY TRÌNH:
    1) `get_material_context(material_id=context.material_id)` để lấy tổng quan.
    2) Với câu hỏi mở: dùng `search_material_chunks(material_id, query)` để lấy các đoạn liên quan rồi trả lời ngắn gọn.
    3) Nếu người dùng hỏi một câu cụ thể: `get_material_question_by_index(material_id, index)` rồi `explain_material_question(material_id, question_id)`.
    
    LƯU Ý VỀ PHẠM VI:
    - Nếu yêu cầu KHÔNG liên quan đến học tiếng Nhật hoặc KHÔNG liên quan tới tài liệu hiện tại → lịch sự từ chối: "Mình chỉ hỗ trợ nội dung học tiếng Nhật và theo tài liệu được cung cấp nhé." 
    - Nếu không tìm thấy ngữ cảnh/chunk phù hợp trong RAG (ví dụ: không có keyword người dùng nêu) → trả lời ngắn gọn: "Hiện chưa tìm thấy nội dung đó trong tài liệu. Có thể dữ liệu chưa được cập nhật, bạn vui lòng đợi hoặc cung cấp từ khóa khác/material_id khác giúp mình nhé."

    FALLBACK:
    4) Nếu kết quả RAG rỗng nhưng skill_type gợi ý là LISTENING hoặc tài liệu là listening: GỌI `get_listening_script(material_id)` để lấy transcript/bản dịch làm ngữ cảnh trả lời (tóm tắt nội dung, trích điểm ngữ pháp xuất hiện nếu có).
    5) Nếu vẫn không có dữ liệu → không suy đoán. Áp dụng thông điệp từ chối/đợi cập nhật.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Context phiên: {context}"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm_instance, tools, prompt)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor
