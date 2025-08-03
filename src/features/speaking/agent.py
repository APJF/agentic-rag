from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from .tools import get_speaking_topic, analyze_speaking_answer

# Prompt template cho speaking agent
SPEAKING_SYSTEM_PROMPT = (
    "Bạn là trợ lý luyện nói tiếng Nhật. Hãy trò chuyện tự nhiên, phù hợp với trình độ (level) và mục tiêu (target) của người dùng. "
    "Nếu user chọn topic, hãy tập trung vào chủ đề đó. Nếu không, hãy gợi ý chủ đề dựa trên sở thích (hobby), mục tiêu hoặc trình độ. "
    "Luôn khuyến khích user nói nhiều, sửa lỗi nhẹ nhàng và đưa ra phản hồi tích cực."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SPEAKING_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    MessagesPlaceholder("agent_scratchpad"),
    ("human", "{user_input}")
])

def initialize_speaking_agent():
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    tools = [get_speaking_topic, analyze_speaking_answer]
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=False)
    return executor