# src/features/qna/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Import các thành phần cốt lõi
from ...core.database import execute_sql_query
from ...core.embedding import get_embedding_model

embedding_model = get_embedding_model()


# --- Tool 1: Lấy hồ sơ người dùng ---
@tool
def get_user_profile_tool(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Sử dụng tool này để lấy thông tin hồ sơ (level, hobby, target) đã được lưu
    của một người dùng cụ thể từ database.
    """
    print(f"--- Tool: Đang lấy hồ sơ của user '{user_id}' ---")
    query = 'SELECT level, hobby, target FROM "User" WHERE id = %s;'
    results = execute_sql_query(query, (user_id,))

    if not results:
        print(f"--- Tool: Không tìm thấy hồ sơ cho user '{user_id}'.")
        return None

    profile = results[0]
    return {k: v for k, v in profile.items() if v is not None}


class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc chủ đề cần tra cứu.")
    course_id: str = Field(default=None, description="Lọc theo một mã môn học cụ thể, ví dụ: 'JPD113'.")
    level: str = Field(default=None, description="Lọc theo cấp độ JLPT, ví dụ: 'N3'.")
    skill_type: str = Field(default=None, description="Lọc theo loại kỹ năng, ví dụ: 'VOCABULARY'.")


@tool(args_schema=KnowledgeSearchInput)
def knowledge_retriever_tool(query: str, course_id: str = None, level: str = None, skill_type: str = None) -> str:
    """
    Truy xuất các mẩu kiến thức (chunks) liên quan nhất từ database.
    Có thể lọc theo mã môn, cấp độ, hoặc kỹ năng.
    """
    print(f"--- Tool RAG: Đang tra cứu cho query '{query}' với các bộ lọc: course_id={course_id}, level={level} ---")
    if not embedding_model:
        return "Lỗi: Model embedding chưa được khởi tạo."

    query_embedding = embedding_model.encode(query).tolist()

    # Xây dựng câu lệnh SQL linh hoạt
    base_query = 'SELECT chunk_text, course_id FROM "content_chunks"'
    where_clauses = []
    params = []

    # Thêm các điều kiện lọc vào mệnh đề WHERE
    if course_id:
        where_clauses.append('"course_id" = %s')
        params.append(course_id)
    if level:
        where_clauses.append("LOWER(level) = LOWER(%s)")
        params.append(level)
    if skill_type:
        where_clauses.append("LOWER(skill_type) = LOWER(%s)")
        params.append(skill_type)

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)

    base_query += " ORDER BY embedding <=> %s LIMIT 3;"
    params.append(str(query_embedding))

    results = execute_sql_query(base_query, tuple(params))

    if not results:
        return "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

    # Định dạng kết quả để AI dễ đọc
    formatted_context = "Dưới đây là các thông tin liên quan được tìm thấy:\n\n"
    for i, doc in enumerate(results):
        formatted_context += f"--- Trích đoạn {i + 1} (Từ Môn học ID: {doc.get('course_id')}) ---\n"
        formatted_context += f"{doc.get('chunk_text')}\n\n"

    return formatted_context


# --- Tool 3: Tra cứu thông tin khóa học ---
@tool
def get_course_context_tool(course_id: str) -> str:
    """
    Lấy thông tin chi tiết (tên, mô tả) về một khóa học dựa trên ID.
    """
    print(f"--- Tool Lookup: Đang tìm thông tin cho Course ID '{course_id}' ---")
    query = 'SELECT title, description FROM "Course" WHERE id = %s;'
    results = execute_sql_query(query, (course_id,))

    if not results:
        return f"Không tìm thấy thông tin cho khóa học có ID {course_id}."

    course = results[0]
    return f"Môn học '{course.get('title')}' (ID: {course_id})."