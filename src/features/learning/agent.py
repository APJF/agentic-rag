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
