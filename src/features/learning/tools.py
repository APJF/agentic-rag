# src/features/learning/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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


class MaterialContextInput(BaseModel):
    material_id: int


@tool(args_schema=MaterialContextInput)
def get_material_context(material_id: int) -> Dict[str, Any]:
    """Lấy metadata tài liệu và danh sách câu hỏi/bài tập (id, index, tiêu đề ngắn)."""
    meta = execute_sql_query('SELECT id, title, description FROM material WHERE id = %s;', (material_id,))
    if not meta:
        return {"error": "Không tìm thấy material."}
    questions = execute_sql_query(
        'SELECT id AS question_id, question_index, short_title FROM material_question WHERE material_id = %s ORDER BY question_index ASC;',
        (material_id,)
    )
    return {"success": True, "material": meta[0], "questions": questions}


class MaterialQuestionByIndexInput(BaseModel):
    material_id: int
    index: int = Field(..., description="Số thứ tự câu trong material (1-based)")


@tool(args_schema=MaterialQuestionByIndexInput)
def get_material_question_by_index(material_id: int, index: int) -> Dict[str, Any]:
    """Resolve câu hỏi theo index trong tài liệu, trả question_id và tiêu đề ngắn."""
    rows = execute_sql_query(
        'SELECT id AS question_id, question_index, short_title FROM material_question WHERE material_id = %s AND question_index = %s;',
        (material_id, index)
    )
    if not rows:
        return {"error": "Không tìm thấy câu hỏi theo index."}
    return {"success": True, "question_id": rows[0]["question_id"], "index": rows[0]["question_index"], "short_title": rows[0].get("short_title")}


class ExplainMaterialQuestionInput(BaseModel):
    material_id: int
    question_id: int


@tool(args_schema=ExplainMaterialQuestionInput)
def explain_material_question(material_id: int, question_id: int) -> Dict[str, Any]:
    """Giải thích câu hỏi trong tài liệu dựa trên nội dung/metadata liên quan."""
    q = execute_sql_query(
        'SELECT question, explanation FROM material_question WHERE id = %s AND material_id = %s;',
        (question_id, material_id)
    )
    if not q:
        return {"error": "Không tìm thấy câu hỏi trong tài liệu."}
    return {"success": True, "question": q[0]["question"], "explanation": q[0]["explanation"]}
