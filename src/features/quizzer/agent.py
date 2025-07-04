
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.vector_store_interface import retrieve_relevant_documents_from_db
from src.core.llm import get_llm
from src.config import settings

class ContextSearchInput(BaseModel):
    level: str = Field(description="Cấp độ JLPT cho nội dung câu hỏi, ví dụ: 'N4'.")
    topic: str = Field(default=None, description="Chủ đề tùy chọn cho nội dung câu hỏi, ví dụ: 'anime'.")

@tool(args_schema=ContextSearchInput)
def context_retriever_tool(level: str, topic: str = None) -> str:
    """
    Truy xuất một đoạn văn bản (chunk) liên quan từ cơ sở dữ liệu để làm ngữ cảnh
    cho việc tạo ra một câu hỏi trắc nghiệm.
    """
    print(f"--- Tool tìm ngữ cảnh được gọi với Level: {level}, Topic: {topic} ---")
    query = f"Một điểm kiến thức hoặc từ vựng tiếng Nhật cấp độ {level}"
    if topic:
        query += f" về chủ đề {topic}"

    filters = {"level": level}
    if topic:
        filters["topic"] = topic

    docs = retrieve_relevant_documents_from_db(
        query,
        top_k=1,
        filters=filters,
        table_name=settings.RAG_CONTENT_CHUNK_TABLE
    )

    if not docs:
        return "Không tìm thấy ngữ cảnh phù hợp trong cơ sở dữ liệu để tạo câu hỏi."

    return docs[0]['text']

def initialize_quiz_agent():
    """
    Khởi tạo Agent chuyên trách việc tạo câu hỏi trắc nghiệm.
    """
    llm = get_llm()
    if not llm:
        return None

    tools = [context_retriever_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         Bạn là một giáo viên tiếng Nhật nhiều kinh nghiệm, chuyên tạo ra các câu hỏi trắc nghiệm hay và phù hợp với trình độ người học.
        
         QUY TRÌNH LÀM VIỆC:
         1. Dựa trên yêu cầu của người dùng về cấp độ (level) và chủ đề (topic), hãy sử dụng công cụ `context_retriever_tool` để lấy một đoạn văn bản làm "gợi ý" hoặc "nguồn cảm hứng".
         2. Sau khi nhận được kết quả (Observation) từ tool, hãy phân tích các từ khóa và ngữ pháp trong đó.
         3. DỰA TRÊN NGUỒN CẢM HỨNG ĐÓ, hãy sáng tạo ra MỘT câu hỏi trắc nghiệm hoàn toàn mới để kiểm tra một kỹ năng tiếng Nhật quan trọng (ví dụ: chọn đúng trợ từ, chia đúng thể động từ, chọn từ vựng đúng ngữ cảnh).
         4. **KHÔNG** được tạo câu hỏi chỉ hỏi về định nghĩa của một từ có sẵn trong ngữ cảnh. Câu hỏi phải yêu cầu người học suy luận và áp dụng kiến thức.

         ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
         Hỏi: [Nội dung câu hỏi bằng tiếng Nhật, có thể kèm giải thích tiếng Việt nếu cần]
         A. [Lựa chọn A]
         B. [Lựa chọn B]
         C. [Lựa chọn C]
         D. [Lựa chọn D]

         Đáp án: [Chữ cái của đáp án đúng]
         Giải thích: [Giải thích chi tiết tại sao đáp án đó đúng và các lựa chọn khác sai]
         """),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor