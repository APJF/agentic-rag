import re
from langchain.tools import tool
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.core.vector_store_interface import get_db_connection
from ...core.database import execute_sql_query

# =====================
# SCHEMA INPUTS
# =====================
class CreateLearningPathInput(BaseModel):
    user_id: str = Field(...)
    title: str = Field(...)
    description: str = Field(...)
    target_level: str = Field(...)
    primary_goal: str = Field(...)
    focus_skill: str = Field(...)
    course_ids: List[str] = Field(...)

class UpdateLearningPathInput(BaseModel):
    path_id: int = Field(...)
    user_id: str = Field(...)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    target_level: Optional[str] = Field(None)
    primary_goal: Optional[str] = Field(None)
    focus_skill: Optional[str] = Field(None)

class AddCoursesInput(BaseModel):
    path_id: int = Field(...)
    user_id: str = Field(...)
    course_ids: List[str] = Field(...)

class ReorderCoursesInput(BaseModel):
    path_id: int = Field(...)
    user_id: str = Field(...)
    ordered_course_ids: List[str] = Field(...)

# =====================
# TOOL: Lấy danh sách lộ trình học
# =====================
@tool
def list_learning_paths(user_id: str) -> dict:
    """
    Lấy danh sách lộ trình học (active và archived) của user.
    """
    try:
        query = 'SELECT id, title, status, last_updated_at FROM learning_path WHERE user_id = %s ORDER BY last_updated_at DESC;'
        paths = execute_sql_query(query, (user_id,))
        return {"success": True, "data": paths}
    except Exception as e:
        return {"error": str(e)}

# =====================
# TOOL: Lấy chi tiết lộ trình học
# =====================
@tool
def get_learning_path_details(path_id: int, user_id: str) -> dict:
    """
    Lấy chi tiết lộ trình học (chỉ khi user sở hữu).
    """
    try:
        query = 'SELECT * FROM learning_path WHERE id = %s AND user_id = %s;'
        path = execute_sql_query(query, (path_id, user_id))
        if not path:
            return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền truy cập."}
        # Lấy danh sách khóa học
        courses_query = '''
            SELECT C.id, C.title, CLP.course_order_number
            FROM course C
            JOIN course_learning_path CLP ON C.id = CLP.course_id
            WHERE CLP.learning_path_id = %s
            ORDER BY CLP.course_order_number ASC;
        '''
        courses = execute_sql_query(courses_query, (path_id,))
        path[0]['courses'] = courses
        return {"success": True, "data": path[0]}
    except Exception as e:
        return {"error": str(e)}

# =====================
# TOOL: Tạo lộ trình học mới
# =====================
@tool(args_schema=CreateLearningPathInput)
def create_learning_path(user_id: str, title: str, description: str, target_level: str, primary_goal: str, focus_skill: str, course_ids: List[str]) -> dict:
    """
    Tạo lộ trình mới, tự động archive lộ trình active cũ của user.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            # Archive lộ trình active cũ
            cur.execute('UPDATE learning_path SET status = %s WHERE user_id = %s AND status = %s;', ('ARCHIVED', user_id, 'ACTIVE'))
            # Tạo lộ trình mới
            cur.execute(
                '''INSERT INTO learning_path (user_id, title, description, target_level, primary_goal, focus_skill, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;''',
                (user_id, title, description, target_level, primary_goal, focus_skill, 'ACTIVE')
            )
            path_id = cur.fetchone()[0]
            # Thêm khóa học
            if course_ids:
                values = [(course_id, path_id, idx+1) for idx, course_id in enumerate(course_ids)]
                execute_values(cur, 'INSERT INTO course_learning_path (course_id, learning_path_id, course_order_number) VALUES %s;', values)
            conn.commit()
            return {"success": True, "path_id": path_id}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

# =====================
# TOOL: Cập nhật lộ trình học
# =====================
@tool(args_schema=UpdateLearningPathInput)
def update_learning_path(path_id: int, user_id: str, title: Optional[str] = None, description: Optional[str] = None, target_level: Optional[str] = None, primary_goal: Optional[str] = None, focus_skill: Optional[str] = None) -> dict:
    """
    Cập nhật thông tin lộ trình học (chỉ khi active và user sở hữu).
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            # Kiểm tra quyền sở hữu và trạng thái
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền cập nhật."}
            if row[0] != 'ACTIVE':
                return {"error": "Chỉ có thể cập nhật lộ trình đang hoạt động."}
            # Xây dựng câu lệnh update
            fields = []
            params = []
            if title: fields.append('title = %s'); params.append(title)
            if description: fields.append('description = %s'); params.append(description)
            if target_level: fields.append('target_level = %s'); params.append(target_level)
            if primary_goal: fields.append('primary_goal = %s'); params.append(primary_goal)
            if focus_skill: fields.append('focus_skill = %s'); params.append(focus_skill)
            if not fields:
                return {"error": "Không có trường nào để cập nhật."}
            params.append(path_id)
            cur.execute(f'UPDATE learning_path SET {", ".join(fields)}, last_updated_at = NOW() WHERE id = %s;', tuple(params))
            conn.commit()
            return {"success": True}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

# =====================
# TOOL: Archive (soft-delete) lộ trình học
# =====================
@tool
def archive_learning_path(path_id: int, user_id: str) -> dict:
    """
    Chuyển lộ trình sang trạng thái ARCHIVED (chỉ khi active và user sở hữu).
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền."}
            if row[0] != 'ACTIVE':
                return {"error": "Chỉ có thể archive lộ trình đang hoạt động."}
            cur.execute('UPDATE learning_path SET status = %s, last_updated_at = NOW() WHERE id = %s;', ('ARCHIVED', path_id))
            conn.commit()
            return {"success": True}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

# =====================
# TOOL: Thêm khóa học vào lộ trình
# =====================
@tool(args_schema=AddCoursesInput)
def add_courses_to_learning_path(path_id: int, user_id: str, course_ids: List[str]) -> dict:
    """
    Thêm một hoặc nhiều khóa học mới vào cuối lộ trình (chỉ khi active và user sở hữu).
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền."}
            if row[0] != 'ACTIVE':
                return {"error": "Chỉ có thể thêm khóa học vào lộ trình đang hoạt động."}
            # Lấy order number lớn nhất hiện có
            cur.execute('SELECT COALESCE(MAX(course_order_number), 0) FROM course_learning_path WHERE learning_path_id = %s;', (path_id,))
            last_order = cur.fetchone()[0]
            # Chèn các khóa học mới
            values = [(course_id, path_id, last_order + i + 1) for i, course_id in enumerate(course_ids)]
            execute_values(cur, 'INSERT INTO course_learning_path (course_id, learning_path_id, course_order_number) VALUES %s;', values)
            cur.execute('UPDATE learning_path SET last_updated_at = NOW() WHERE id = %s;', (path_id,))
            conn.commit()
            return {"success": True}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

# =====================
# TOOL: Đổi thứ tự khóa học trong lộ trình
# =====================
@tool(args_schema=ReorderCoursesInput)
def reorder_courses_in_learning_path(path_id: int, user_id: str, ordered_course_ids: List[str]) -> dict:
    """
    Cập nhật lại toàn bộ thứ tự các khóa học trong lộ trình (chỉ khi active và user sở hữu).
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền."}
            if row[0] != 'ACTIVE':
                return {"error": "Chỉ có thể sắp xếp lại khóa học trong lộ trình đang hoạt động."}
            for idx, course_id in enumerate(ordered_course_ids):
                cur.execute('UPDATE course_learning_path SET course_order_number = %s WHERE learning_path_id = %s AND course_id = %s;', (idx+1, path_id, course_id))
            cur.execute('UPDATE learning_path SET last_updated_at = NOW() WHERE id = %s;', (path_id,))
            conn.commit()
            return {"success": True}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()