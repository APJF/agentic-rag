# src/features/learning_path/agent.py

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Import bộ công cụ chuyên dụng mới
from .tools import (
    list_user_learning_paths,
    get_user_profile,
    save_new_learning_path,
    find_relevant_courses,
    get_course_structure,
    calculate_time_constraints
)
from ...core.llm import get_llm

def initialize_learning_path_agent():
    """
    Khởi tạo Agent chuyên về CRUD (Tạo, Đọc, Sửa, Xóa) lộ trình học.
    """
    llm_instance = get_llm()
    tools = [
        list_user_learning_paths,
        get_user_profile,
        save_new_learning_path,
        find_relevant_courses,
        get_course_structure,
        calculate_time_constraints
    ]
    system_prompt = """
    
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "User ID của tôi là {user_id}. Yêu cầu của tôi là: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm_instance, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )
    return agent_executor