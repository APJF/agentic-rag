# src/features/learning/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from .tools import get_material_context, get_material_question_by_index, explain_material_question, fetch_material_file, extract_text_from_material
from ...core.llm import get_llm


def initialize_learning_agent():
    llm_instance = get_llm()
    tools = [
        get_material_context,
        fetch_material_file,
        extract_text_from_material,
        get_material_question_by_index,
        explain_material_question,
    ]

    system_prompt = """
    Bạn là Trợ lý Học tập theo tài liệu (material).

    QUY TẮC BẮT BUỘC:
    - Nếu `context.material_id` tồn tại, PHẢI dùng tool để lấy ngữ cảnh tài liệu trước khi trả lời.
    - Tải file tài liệu từ MinIO (fetch_material_file) và trích xuất văn bản (extract_text_from_material) nếu cần để làm nền ngữ cảnh trả lời.
    - Nếu người dùng nói "câu X" thì mặc định là câu X trong material hiện tại, không hỏi lại.
    - Chỉ gửi Final Answer; không in Thought/Action.

    QUY TRÌNH:
    1) `get_material_context(material_id=context.material_id)` để lấy metadata + danh sách câu hỏi.
    2) `fetch_material_file(material_id)` và `extract_text_from_material(local_path)` để lấy text nền (nếu cần giải thích từ nội dung tài liệu).
    3) Nếu người dùng hỏi một câu cụ thể: `get_material_question_by_index(material_id, index)` rồi `explain_material_question(material_id, question_id)`.
    4) Nếu không rõ câu cụ thể: tóm tắt/giải thích dựa trên text đã trích.
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
