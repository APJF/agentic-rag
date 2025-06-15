# src/core/quiz_generator.py
import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.llm_services import get_llm
from src.core.vector_store_interface import retrieve_relevant_documents_from_db

# --- Khởi tạo ---
llm_instance = get_llm()

QUIZ_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một AI chuyên tạo câu hỏi trắc nghiệm tiếng Nhật.
Dựa vào đoạn văn bản ngữ cảnh được cung cấp dưới đây, hãy tạo ra MỘT câu hỏi trắc nghiệm để kiểm tra kiến thức của người học.
Yêu cầu cho câu hỏi:
1.  Định dạng phải là:
    Hỏi: [Nội dung câu hỏi bằng tiếng Việt]
    A. [Lựa chọn A]
    B. [Lựa chọn B]
    C. [Lựa chọn C]
    D. [Lựa chọn D]
    Đáp án: [Chữ cái của đáp án đúng]
    Giải thích: [Giải thích ngắn gọn bằng tiếng Việt tại sao đó là đáp án đúng]
2.  Câu hỏi phải liên quan trực tiếp đến nội dung trong ngữ cảnh.

--- NGỮ CẢNH ---
{context}
--- HẾT NGỮ CẢNH ---

Bắt đầu tạo câu hỏi:
""")

quiz_chain = QUIZ_GENERATION_PROMPT | llm_instance | StrOutputParser()

def generate_quiz_question(level: str, topic: str = None) -> str:
    """
    Tạo một câu hỏi trắc nghiệm dựa trên level và topic (tùy chọn).
    """
    if not llm_instance:
        return "Lỗi: Dịch vụ LLM chưa được cấu hình."

    try:
        random_query = f"Một từ vựng hoặc điểm ngữ pháp tiếng Nhật cấp độ {level}"
        if topic:
            random_query += f" về chủ đề {topic}"

        # Xây dựng bộ lọc
        filters = {"level": level}
        if topic:
            filters["topic"] = topic

        relevant_docs = retrieve_relevant_documents_from_db(
            random_query,
            top_k=5,
            filters=filters  # Sử dụng bộ lọc mới
        )

        if not relevant_docs:
            msg = f"Không tìm thấy đủ dữ liệu cho level {level}"
            if topic:
                msg += f" và chủ đề {topic}"
            msg += " để tạo câu hỏi."
            return msg

    except Exception as e:
        print(f"Lỗi khi tạo câu hỏi: {e}")
        return "Xin lỗi, đã có lỗi xảy ra trong quá trình tạo câu hỏi."