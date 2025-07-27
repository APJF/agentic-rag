# src/features/learning/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from .tools import contextual_knowledge_retriever
from ...core.llm import get_llm


def initialize_learning_agent():
    llm_instance = get_llm()
    tools = [contextual_knowledge_retriever]

    system_prompt = """
    Bạn là một Trợ lý Học tập, chuyên giải đáp các thắc mắc của người học trong phạm vi một bài học cụ thể.

    **QUY TRÌNH SUY LUẬN:**

    1.  **Tra cứu trong bài học:**
        - `Thought`: "Người dùng đang hỏi trong ngữ cảnh một bài học. Tôi PHẢI dùng tool `contextual_knowledge_retriever` để tìm kiếm thông tin CHỈ trong bài học đó."
        - `Action`: Gọi tool `contextual_knowledge_retriever` với câu hỏi của người dùng và `material_id` từ context.

    2.  **Trả lời dựa trên ngữ cảnh:**
        - `Thought`: "Tôi đã có ngữ cảnh từ bài học. Tôi sẽ tổng hợp thông tin này để trả lời câu hỏi một cách chính xác."
        - `Final Answer`: Đưa ra câu trả lời dựa hoàn toàn vào thông tin đã được cung cấp. Nếu không tìm thấy, hãy thông báo rằng kiến thức này không có trong bài học hiện tại.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
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
