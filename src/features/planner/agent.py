
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from src.core.agent_tools import create_full_learning_path
from src.core.llm import get_llm


def initialize_planning_agent():
    """
    Khởi tạo Agent tư vấn sử dụng kiến trúc OpenAI Tools Agent hiện đại,
    được thiết kế để vừa trò chuyện vừa sử dụng công cụ một cách linh hoạt.
    """
    llm_instance = get_llm()
    if not llm_instance:
        print("Lỗi: LLM chưa được khởi tạo. Vui lòng kiểm tra cấu hình.")
        return None

    tools = [create_full_learning_path]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         Bạn là một trợ lý tư vấn lộ trình học tiếng Nhật thân thiện và chuyên nghiệp.
         Vai trò của bạn là trò chuyện với người dùng để thu thập 4 thông tin sau:
         1. Trình độ hiện tại (current_level)
         2. Mục tiêu học tập (learning_goal)
         3. Sở thích cá nhân (personal_interests)
         4. Thời gian học mỗi tuần (weekly_study_time)

         QUY TẮC QUAN TRỌNG:
         - Hãy hỏi TỪNG CÂU MỘT. Chờ người dùng trả lời xong rồi mới hỏi câu tiếp theo.
         - Đừng bao giờ hỏi gộp nhiều câu vào một lúc.
         - Sau khi, và CHỈ SAU KHI, bạn đã có đủ cả 4 thông tin trên, hãy gọi công cụ `create_full_learning_path` để tạo kế hoạch.
         - Cuối cùng, hãy trình bày kế hoạch mà công cụ trả về cho người dùng.
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm_instance, tools, prompt)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    return agent_executor