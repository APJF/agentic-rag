import re
from langchain.tools import tool
from typing import Optional, Union
from math import ceil
from datetime import datetime, timezone, timedelta

_SESSION_USER_ID: Optional[int] = None

def set_session_user_id(real_user_id: Union[str, int]):
    """Gọi hàm này ở layer HTTP (endpoint) trước khi invoke Agent để đảm bảo tool nào cũng dùng đúng user_id."""
    global _SESSION_USER_ID
    _SESSION_USER_ID = _parse_user_id(real_user_id)


def _parse_user_id(user_id: Union[str, int, None]) -> Optional[int]:
    """Chuyển user_id sang int nếu có thể. Trả về None nếu không hợp lệ."""
    if user_id is None:
        return None
    try:
        return int(user_id)
    except (TypeError, ValueError):
        digits = re.findall(r"\d+", str(user_id))
        if digits:
            try:
                return int(digits[0])
            except ValueError:
                return None
    return None

from typing import List, Optional, Union
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field
from src.core.vector_store_interface import get_db_connection
from ...core.database import execute_sql_query
from src.data_processing.manifest_loader import courses_by_level, course_sequence_between

def _parse_jlpt_level_from_exam_id(exam_id: Union[str, int, None]) -> Optional[str]:
    if not exam_id:
        return None
    value = str(exam_id).upper()
    for lv in ["N5","N4","N3","N2","N1"]:
        if lv in value:
            return lv
    return None

def _jlpt_level_down(level_main: str) -> str:
    order = ["N5","N4","N3","N2","N1"]
    try:
        idx = order.index(level_main)
    except ValueError:
        return level_main
    return order[max(0, idx-1)]

def _compute_level_from_score(level_main: str, score_percent: float) -> str:
    if score_percent is None:
        return f"{level_main}-M"
    if score_percent >= 80:
        tier = "H"
    elif score_percent >= 60:
        tier = "M"
    elif score_percent >= 40:
        tier = "L"
    else:
        level_main = _jlpt_level_down(level_main)
        tier = "H"
    return f"{level_main}-{tier}"

class CreateLearningPathInput(BaseModel):
    user_id: Union[str, int] = Field(...)
    title: str = Field(...)
    description: str = Field(...)
    target_level: str = Field(...)
    primary_goal: str = Field(...)
    focus_skill: str = Field(...)
    course_ids: List[str] = Field(...)

class UpdateLearningPathInput(BaseModel):
    path_id: int = Field(...)
    user_id: Union[str, int] = Field(...)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    target_level: Optional[str] = Field(None)
    primary_goal: Optional[str] = Field(None)
    focus_skill: Optional[str] = Field(None)

class AddCoursesInput(BaseModel):
    path_id: int = Field(...)
    user_id: Union[str, int] = Field(...)
    course_ids: List[str] = Field(...)

class ReorderCoursesInput(BaseModel):
    path_id: int = Field(...)
    user_id: Union[str, int] = Field(...)
    ordered_course_ids: List[str] = Field(...)

@tool
def list_learning_paths(user_id: Union[str, int]) -> dict:
    """
    Lấy danh sách lộ trình học (active và archived) của user.
    """
    try:
        user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
        if user_id_int is None:
            return {"success": True, "data": []}
        query = 'SELECT id, title, status, last_updated_at FROM learning_path WHERE user_id = %s ORDER BY last_updated_at DESC;'
        paths = execute_sql_query(query, (user_id_int,))
        return {"success": True, "data": paths}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_learning_path_details(path_id: int, user_id: Union[str, int]) -> dict:
    """
    Lấy chi tiết lộ trình học (chỉ khi user sở hữu).
    """
    try:
        user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
        if user_id_int is None:
            return {"error": "user_id không hợp lệ."}
        query = 'SELECT * FROM learning_path WHERE id = %s AND user_id = %s;'
        path = execute_sql_query(query, (path_id, user_id_int))
        if not path:
            return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền truy cập."}
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
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            if not course_ids:
                return {"error": "Danh sách khóa học trống. Hãy gọi tool lấy sequence khóa học trước khi tạo lộ trình."}
            total_hours = _sum_course_duration(cur, course_ids)
            cur.execute('UPDATE learning_path SET status = %s WHERE user_id = %s AND status = %s;', ('PENDING', user_id_int, 'STUDYING'))
            cur.execute(
                '''INSERT INTO learning_path (user_id, title, description, target_level, primary_goal, focus_skill, status, created_at, last_updated_at, duration)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s) RETURNING id;''',
                (user_id_int, title, description, target_level, primary_goal, focus_skill, 'STUDYING', total_hours)
            )
            path_id = cur.fetchone()[0]
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
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền cập nhật."}
            if row[0] != 'STUDYING':
                return {"error": "Chỉ có thể cập nhật lộ trình đang hoạt động (STUDYING)."}
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
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền."}
            if row[0] != 'STUDYING':
                return {"error": "Chỉ có thể chuyển trạng thái lộ trình đang hoạt động (STUDYING)."}
            cur.execute('UPDATE learning_path SET status = %s, last_updated_at = NOW() WHERE id = %s;', ('PENDING', path_id))
            conn.commit()
            return {"success": True}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

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
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
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
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
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

@tool
def find_relevant_courses(target_level: str, focus_skill: str, learning_goal: str) -> dict:
    """
    Tìm các khóa học phù hợp trong bảng `course`.

    Quy tắc lọc đơn giản:
    1. Khớp trường `level` = target_level (nếu cột này tồn tại).
    2. Ưu tiên tiêu đề hoặc mô tả chứa focus_skill (ILIKE).
    3. Nếu không có cột level, hãy so khớp target_level trong title.
    Trả về danh sách dict(id, title). Nếu không tìm thấy, trả về error.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT id, title
                    FROM course
                    WHERE (level = %s OR title ILIKE %s)
                      AND (title ILIKE %s OR description ILIKE %s)
                    LIMIT 20;
                    """,
                    (target_level, f"%{target_level}%", f"%{focus_skill}%", f"%{focus_skill}%")
                )
            except Exception:
                cur.execute(
                    """
                    SELECT id, title FROM course
                    WHERE title ILIKE %s
                    LIMIT 20;
                    """,
                    (f"%{target_level}%",)
                )
            rows = cur.fetchall()
            if not rows:
                return {"error": "Không tìm thấy khóa học phù hợp."}
            courses = [{"id": r[0], "title": r[1]} for r in rows]
            return {"success": True, "courses": courses}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

@tool
def calculate_time_constraints(deadline_info: str) -> dict:
    """
    [TẠM THỜI] Phân tích deadline (chuỗi) và ước lượng tổng số giờ học còn lại.
    """
    return {"success": True, "hours_left": 150}

@tool
def get_now_utc7() -> dict:
    """Trả về thời gian hiện tại theo múi giờ UTC+7 ở định dạng ISO 8601."""
    now = datetime.now(timezone.utc) + timedelta(hours=7)
    return {"success": True, "now": now.isoformat()}

@tool
def weeks_until_deadline_utc7(deadline_iso: str) -> dict:
    """Tính số tuần còn lại tới deadline theo UTC+7. Nhận chuỗi thời gian ISO 8601; nếu thiếu timezone sẽ giả định UTC+7."""
    try:
        now = datetime.now(timezone.utc) + timedelta(hours=7)
        deadline = datetime.fromisoformat(deadline_iso)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone(timedelta(hours=7)))
        delta_days = (deadline - now).days
        return {"success": True, "weeks": max(0, delta_days // 7)}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_user_level(user_id: Union[str, int]) -> dict:
    """Trả về level hiện tại của user từ bảng users.level (ví dụ 'N4-M')."""
    try:
        user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
        if user_id_int is None:
            return {"error": "user_id không hợp lệ"}
        rows = execute_sql_query('SELECT level FROM "users" WHERE id = %s;', (user_id_int,))
        level_val = rows[0]["level"] if rows else None
        return {"success": True, "level": level_val}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_courses_for_level(level: str) -> dict:
    """Trả về list course_id cho một level (N5, N4, N3...)."""
    courses = courses_by_level(level.upper())
    if not courses:
        return {"error": f"Không tìm thấy khóa học cho level {level}"}
    return {"success": True, "courses": courses}

@tool
def get_course_sequence_between_levels(start_level: str, end_level: str) -> dict:
    """Ghép các course từ start_level tới end_level inclusive."""
    seq = course_sequence_between(start_level.upper(), end_level.upper())
    if not seq:
        return {"error": "Không tìm thấy sequence khóa học phù hợp."}
    return {"success": True, "courses": seq}

@tool
def get_course_sequence_for_improvement(current_level: str) -> dict:
    """Cho level hiện tại (ví dụ 'N5-M'), trả danh sách course_id để đạt +2 level (N3-M)."""
    try:
        main_level = current_level.upper().split('-')[0]
        order = ["N5", "N4", "N3", "N2", "N1"]
        if main_level not in order:
            return {"error": "Level không hợp lệ"}
        idx = order.index(main_level)
        target_idx = min(idx + 2, len(order)-1)
        seq = course_sequence_between(order[idx], order[target_idx])
        return {"success": True, "target_main": order[target_idx], "courses": seq}
    except Exception as e:
        return {"error": str(e)}

@tool
def update_user_level(user_id: Union[str, int], new_level: str) -> dict:
    """Cập nhật cột level ở bảng users."""
    try:
        user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
        if user_id_int is None:
            return {"error": "user_id không hợp lệ"}
        conn = get_db_connection()
        if not conn:
            return {"error": "Không kết nối DB"}
        with conn.cursor() as cur:
            cur.execute('UPDATE "users" SET level = %s WHERE id = %s;', (new_level, user_id_int))
            conn.commit()
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

def _sum_course_duration(cur, course_ids: List[str]) -> float:
    if not course_ids:
        return 0.0
    sql = 'SELECT SUM(duration) FROM course WHERE id = ANY(%s);'
    cur.execute(sql, (course_ids,))
    s = cur.fetchone()[0]
    return float(s or 0.0)

@tool
def calculate_path_duration(course_ids: List[str], hours_per_week: int = 10) -> dict:
    """Trả về tổng giờ học và số tuần ước tính.
    Nếu bảng course chưa có duration, hàm trả 0.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không kết nối DB"}
    try:
        with conn.cursor() as cur:
            total = _sum_course_duration(cur, course_ids)
        weeks = ceil(total / hours_per_week) if hours_per_week else None
        return {"success": True, "total_hours": total, "estimated_weeks": weeks}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

@tool
def get_latest_exam_result_for_exam(user_id: Union[str, int], exam_id: Union[str, int]) -> dict:
    """Trả về lần làm bài gần nhất của user cho exam_id (ưu tiên submitted trước, sau đó started_at mới nhất)."""
    user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
    if user_id_int is None:
        return {"error": "user_id không hợp lệ"}
    rows = execute_sql_query(
        """
        SELECT id, score, status, exam_id, started_at, submitted_at
        FROM exam_result
        WHERE user_id = %(user_id)s AND exam_id = %(exam_id)s
        ORDER BY (submitted_at IS NULL), submitted_at DESC, started_at DESC
        LIMIT 1;
        """,
        {"user_id": user_id_int, "exam_id": exam_id}
    )
    if not rows:
        return {"error": "Chưa có lần làm bài nào."}
    return {"success": True, "data": rows[0]}

class ConfirmLevelInput(BaseModel):
    user_id: Union[str, int]
    exam_id: Union[str, int]

@tool(args_schema=ConfirmLevelInput)
def confirm_and_update_level(user_id: Union[str, int], exam_id: Union[str, int]) -> dict:
    """Lấy lần làm bài mới nhất cho exam_id theo user, suy ra tier theo điểm, cập nhật users.level và trả về thông tin."""
    user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
    if user_id_int is None:
        return {"error": "user_id không hợp lệ"}

    exam_id_num = None
    exam_title = None
    try:
        exam_id_num = int(str(exam_id))
    except Exception:
        exam_title = str(exam_id)

    rows = execute_sql_query(
        """
        SELECT er.id, er.score, er.status, er.exam_id, er.started_at, er.submitted_at, e.title AS exam_title
        FROM exam_result er
        JOIN exam e ON e.id = er.exam_id
        WHERE er.user_id = %(user_id)s
          AND (
                (%(exam_id_num)s IS NOT NULL AND er.exam_id = %(exam_id_num)s)
             OR (%(exam_title)s IS NOT NULL AND e.title = %(exam_title)s)
          )
        ORDER BY (er.submitted_at IS NULL), er.submitted_at DESC, er.started_at DESC
        LIMIT 1;
        """,
        {"user_id": user_id_int, "exam_id_num": exam_id_num, "exam_title": exam_title}
    )
    if not rows:
        return {"error": "Chưa có lần làm bài nào cho exam này."}
    attempt = rows[0]
    level_main = _parse_jlpt_level_from_exam_id(attempt.get("exam_title") or attempt.get("exam_id"))
    if not level_main:
        return {"error": "Không xác định được level từ exam_id."}
    score = attempt.get("score")
    try:
        score_percent = float(score)
    except Exception:
        score_percent = None
    computed_level = _compute_level_from_score(level_main, score_percent)
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Không kết nối DB"}
        with conn.cursor() as cur:
            cur.execute('UPDATE "users" SET level = %s WHERE id = %s;', (computed_level, user_id_int))
            conn.commit()
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

    return {"success": True, "computed_level": computed_level, "score": score_percent, "exam_level": level_main, "exam_result_id": attempt.get("id")}
