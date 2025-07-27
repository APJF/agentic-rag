# src/features/learning/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from ...core.database import execute_sql_query
from ...core.embedding import get_embedding_model

embedding_model = get_embedding_model()


class ContextualSearchInput(BaseModel):
    query: str = Field(description="Câu hỏi của người dùng.")
    material_id: str = Field(description="ID của tài liệu học tập đang xem để giới hạn phạm vi tìm kiếm.")


@tool(args_schema=ContextualSearchInput)
def contextual_knowledge_retriever(query: str, material_id: str) -> str:
    """
    Công cụ RAG theo ngữ cảnh. Chỉ tìm kiếm kiến thức trong phạm vi một
    tài liệu (Material) cụ thể.
    """
    print(f"--- Tool Learning: Đang tra cứu '{query}' trong Material ID '{material_id}' ---")
    if not embedding_model:
        return "Lỗi: Model embedding chưa được khởi tạo."

    query_embedding = embedding_model.encode(query).tolist()

    # Câu lệnh SQL có bộ lọc cứng theo material_id (hoặc unit_id tùy thiết kế)
    query_sql = """
                SELECT chunk_text
                FROM "content_chunks"
                WHERE material_id = %s -- Lọc cứng theo ngữ cảnh
                ORDER BY embedding <=> %s
                    LIMIT 3; \
                """
    params = (material_id, str(query_embedding))
    results = execute_sql_query(query_sql, params)

    if not results:
        return "Không tìm thấy thông tin liên quan trong bài học này."

    formatted_context = "Dưới đây là các thông tin liên quan được tìm thấy trong bài học:\n\n"
    for i, doc in enumerate(results):
        formatted_context += f"--- Trích đoạn {i + 1} ---\n{doc.get('chunk_text')}\n\n"

    return formatted_context
