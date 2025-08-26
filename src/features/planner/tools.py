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
        return f"{level_main}_M"
    if score_percent >= 80:
        tier = "H"
    elif score_percent >= 60:
        tier = "M"
    elif score_percent >= 40:
        tier = "L"
    else:
        level_main = _jlpt_level_down(level_main)
        tier = "H"
    return f"{level_main}_{tier}"

# === Level helpers ===
_LEVEL_ORDER = ["N5", "N4", "N3", "N2", "N1"]
_DIGIT_TO_LEVEL = {
    "1": "N5",
    "2": "N4",
    "3": "N3",
    "4": "N2",
    "5": "N1",
}

def _level_from_course_id(course_id: str) -> Optional[str]:
    """Suy ra level từ mã môn. Ví dụ: JPD113 -> N5, JPD216 -> N4, JPD316 -> N3.
    Quy tắc: chữ số đầu tiên trong cụm số sau prefix JPD: 1->N5, 2->N4, 3->N3, 4->N2, 5->N1.
    """
    if not course_id:
        return None
    try:
        import re as _re
        m = _re.search(r"[A-Za-z]+(\d{3,})", str(course_id))
        if not m:
            return None
        first_digit = m.group(1)[0]
        return _DIGIT_TO_LEVEL.get(first_digit)
    except Exception:
        return None

def _level_index(level: Optional[str]) -> int:
    try:
        return _LEVEL_ORDER.index(level) if level in _LEVEL_ORDER else len(_LEVEL_ORDER)
    except Exception:
        return len(_LEVEL_ORDER)

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
def set_primary_learning_path(path_id: int, user_id: Union[str, int]) -> dict:
    """
    Đặt lộ trình có `path_id` làm lộ trình chính (STUDYING) cho user.
    - Chuyển các lộ trình đang STUDYING/ACTIVE khác về PENDING.
    - Chỉ thực hiện nếu lộ trình thuộc về user.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            # Verify ownership
            cur.execute('SELECT 1 FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
            if not cur.fetchone():
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền."}
            # Demote current studying to pending
            cur.execute('UPDATE learning_path SET status = %s WHERE user_id = %s AND status = %s;', (
                'PENDING', user_id_int, 'STUDYING'
            ))
            # Promote selected path
            cur.execute('UPDATE learning_path SET status = %s, last_updated_at = NOW() WHERE id = %s AND user_id = %s;', (
                'STUDYING', path_id, user_id_int
            ))
            conn.commit()
            return {"success": True}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

@tool
def promote_latest_pending_learning_path(user_id: Union[str, int]) -> dict:
    """
    Đặt lộ trình PENDING mới nhất của user làm STUDYING.
    Tự động hạ lộ trình STUDYING hiện tại (nếu có) xuống PENDING.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            cur.execute('SELECT id FROM learning_path WHERE user_id = %s AND status = %s ORDER BY last_updated_at DESC LIMIT 1;', (user_id_int, 'PENDING'))
            row = cur.fetchone()
            if not row:
                return {"error": "Không có lộ trình PENDING để đặt làm chính."}
            path_id = row[0]
            # demote studying
            cur.execute('UPDATE learning_path SET status = %s WHERE user_id = %s AND status = %s;', ('PENDING', user_id_int, 'STUDYING'))
            # promote selected
            cur.execute('UPDATE learning_path SET status = %s, last_updated_at = NOW() WHERE id = %s AND user_id = %s;', ('STUDYING', path_id, user_id_int))
            conn.commit()
            return {"success": True, "path_id": path_id}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

@tool
def set_primary_learning_path_by_title(title: str, user_id: Union[str, int]) -> dict:
    """
    Đặt lộ trình có tiêu đề khớp chính xác làm STUDYING cho user.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            cur.execute('SELECT id FROM learning_path WHERE user_id = %s AND title = %s LIMIT 1;', (user_id_int, title))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình theo tiêu đề."}
            path_id = row[0]
            cur.execute('UPDATE learning_path SET status = %s WHERE user_id = %s AND status = %s;', ('PENDING', user_id_int, 'STUDYING'))
            cur.execute('UPDATE learning_path SET status = %s, last_updated_at = NOW() WHERE id = %s AND user_id = %s;', ('STUDYING', path_id, user_id_int))
            conn.commit()
            return {"success": True, "path_id": path_id}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()
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
            # Chỉ giữ các course tồn tại trong DB, nhưng PHẢI giữ NGUYÊN THỨ TỰ gốc của course_ids
            cur.execute('SELECT id FROM course WHERE id = ANY(%s);', (course_ids,))
            _rows = [r[0] for r in cur.fetchall()]
            existing_set = set(_rows)
            ordered_ids = [cid for cid in course_ids if cid in existing_set]
            skipped_ids = [cid for cid in course_ids if cid not in existing_set]
            if not ordered_ids:
                return {"error": "Không có mã môn hợp lệ trong cơ sở dữ liệu.", "skipped_courses": skipped_ids}
            # Re-validate ordering by level using manifest order as primary key
            try:
                # Build manifest position map for deterministic in-level ordering
                manifest_pos: dict = {}
                for lv in ["N5","N4","N3","N2","N1"]:
                    seq = courses_by_level(lv) or []
                    for idx, cid in enumerate(seq):
                        # Smaller index means earlier in level
                        if cid not in manifest_pos:
                            manifest_pos[cid] = idx
                input_index = {cid: i for i, cid in enumerate(ordered_ids)}
                def _sort_key(cid: str):
                    return (
                        _level_index(_level_from_course_id(cid)),
                        manifest_pos.get(cid, 10_000),
                        input_index.get(cid, 10_000)
                    )
                ordered_ids = sorted(ordered_ids, key=_sort_key)
            except Exception:
                pass
            total_hours = _sum_course_duration(cur, ordered_ids)

            # Build availability warning if target_level > max available in DB
            try:
                target_main = (_level_from_course_id(f"JPD5xx") and target_level) or target_level  # keep original target parse
                # compute max level present in ordered_ids
                levels_present = [_level_from_course_id(cid) for cid in ordered_ids]
                levels_present = [lv for lv in levels_present if lv in _LEVEL_ORDER]
                max_level_present = None
                if levels_present:
                    max_level_present = max(levels_present, key=lambda lv: _level_index(lv))
                warn_msg = None
                if max_level_present and target_level and _level_index(max_level_present) < _level_index((_level_from_course_id(target_level) or target_level).split('_')[0]):
                    warn_msg = (
                        f"Cảnh báo: Dữ liệu hiện tại chỉ có tới {max_level_present}. Lộ trình sẽ được tạo đến hết {max_level_present}. "
                        "Khi hệ thống cập nhật thêm môn ở level cao hơn, bạn có thể mở rộng lộ trình."
                    )
            except Exception:
                warn_msg = None
            cur.execute('UPDATE learning_path SET status = %s WHERE user_id = %s AND status = %s;', ('PENDING', user_id_int, 'STUDYING'))
            cur.execute(
                '''INSERT INTO learning_path (user_id, title, description, target_level, primary_goal, focus_skill, status, created_at, last_updated_at, duration)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s) RETURNING id;''',
                (user_id_int, title, description, target_level, primary_goal, focus_skill, 'STUDYING', total_hours)
            )
            path_id = cur.fetchone()[0]
            if ordered_ids:
                values = [(course_id, path_id, idx+1) for idx, course_id in enumerate(ordered_ids)]
                execute_values(cur, 'INSERT INTO course_learning_path (course_id, learning_path_id, course_order_number) VALUES %s;', values)
            conn.commit()
            result = {"success": True, "path_id": path_id, "used_course_ids": ordered_ids, "skipped_courses": skipped_ids}
            if warn_msg:
                result["warning"] = warn_msg
            return result
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

@tool(args_schema=UpdateLearningPathInput)
def update_learning_path(path_id: int, user_id: str, title: Optional[str] = None, description: Optional[str] = None, target_level: Optional[str] = None, primary_goal: Optional[str] = None, focus_skill: Optional[str] = None) -> dict:
    """
    Cập nhật thông tin lộ trình học (user sở hữu).

    Quy tắc:
    - Nếu lộ trình đang hoạt động (STUDYING): cho phép cập nhật các trường: title, description, target_level, primary_goal, focus_skill.
    - Nếu lộ trình ở trạng thái PENDING: CHỈ cho phép đổi tên (title) và mô tả (description). Các trường khác sẽ bị từ chối.
    - Không hỗ trợ cập nhật khi trạng thái khác (ví dụ ARCHIVED).
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
            status = row[0]
            fields = []
            params = []
            if status == 'STUDYING':
                if title: fields.append('title = %s'); params.append(title)
                if description: fields.append('description = %s'); params.append(description)
                if target_level: fields.append('target_level = %s'); params.append(target_level)
                if primary_goal: fields.append('primary_goal = %s'); params.append(primary_goal)
                if focus_skill: fields.append('focus_skill = %s'); params.append(focus_skill)
            elif status == 'PENDING':
                # Chỉ cho phép đổi title/description khi PENDING
                if target_level or primary_goal or focus_skill:
                    return {"error": "Lộ trình PENDING chỉ cho phép đổi tên và mô tả."}
                if title: fields.append('title = %s'); params.append(title)
                if description: fields.append('description = %s'); params.append(description)
            else:
                return {"error": "Không hỗ trợ cập nhật cho trạng thái hiện tại của lộ trình."}

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
def delete_learning_path(path_id: int, user_id: Union[str, int]) -> dict:
    """
    Xóa vĩnh viễn một lộ trình học thuộc về user khi trạng thái là PENDING hoặc ARCHIVED.
    - Không cho phép xóa lộ trình đang hoạt động (STUDYING). Hãy chuyển sang PENDING trước nếu cần.
    - Tự động xóa các bản ghi trong `course_learning_path` liên quan.
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
            status = row[0]
            if status == 'STUDYING':
                return {"error": "Không thể xóa lộ trình đang hoạt động. Hãy chuyển về PENDING trước."}
            # Xóa liên kết khóa học trước
            cur.execute('DELETE FROM course_learning_path WHERE learning_path_id = %s;', (path_id,))
            # Xóa lộ trình
            cur.execute('DELETE FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
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
            if row[0] not in ('STUDYING',):
                return {"error": "Chỉ có thể thêm khóa học vào lộ trình đang hoạt động."}
            # Loại bỏ mã đã có và đảm bảo thứ tự level hợp lý khi chèn
            cur.execute('SELECT course_id FROM course_learning_path WHERE learning_path_id = %s ORDER BY course_order_number ASC;', (path_id,))
            existing_ids = [r[0] for r in cur.fetchall()]
            filtered = [cid for cid in course_ids if cid not in set(existing_ids)]
            try:
                manifest_pos: dict = {}
                for lv in ["N5","N4","N3","N2","N1"]:
                    seq = courses_by_level(lv) or []
                    for idx, cid in enumerate(seq):
                        if cid not in manifest_pos:
                            manifest_pos[cid] = idx
                input_index = {cid: i for i, cid in enumerate(course_ids)}
                def _sort_key(cid: str):
                    return (
                        _level_index(_level_from_course_id(cid)),
                        manifest_pos.get(cid, 10_000),
                        input_index.get(cid, 10_000)
                    )
                filtered = sorted(filtered, key=_sort_key)
            except Exception:
                pass
            # Lấy order number lớn nhất hiện có
            cur.execute('SELECT COALESCE(MAX(course_order_number), 0) FROM course_learning_path WHERE learning_path_id = %s;', (path_id,))
            last_order = cur.fetchone()[0]
            # Chèn các khóa học mới theo thứ tự hợp lệ
            values = [(course_id, path_id, last_order + i + 1) for i, course_id in enumerate(filtered)]
            execute_values(cur, 'INSERT INTO course_learning_path (course_id, learning_path_id, course_order_number) VALUES %s;', values)
            cur.execute('UPDATE learning_path SET last_updated_at = NOW() WHERE id = %s;', (path_id,))
            conn.commit()
            return {"success": True, "added_courses": filtered}
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
            if row[0] not in ('STUDYING',):
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
def suggest_next_course_for_level(path_id: int, user_id: Union[str, int], level: str) -> dict:
    """
    Gợi ý môn tiếp theo cho một level (ví dụ 'N4') chưa có trong lộ trình `path_id`.
    Dựa trên manifest `courses_by_level(level)` rồi loại bỏ các môn đã có.
    """
    try:
        user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
        if user_id_int is None:
            return {"error": "user_id không hợp lệ."}
        # Lấy danh sách course hiện có trong lộ trình
        existing = execute_sql_query(
            'SELECT course_id FROM course_learning_path WHERE learning_path_id = %s ORDER BY course_order_number ASC;',
            (path_id,)
        ) or []
        existing_ids = [row.get('course_id') or row.get('course_id') for row in existing]
        candidates = courses_by_level(level.upper()) or []
        for cid in candidates:
            if cid not in existing_ids:
                return {"success": True, "course_id": cid}
        return {"error": "Không còn môn mới cho level này trong manifest."}
    except Exception as e:
        return {"error": str(e)}

@tool
def add_next_course_for_level(path_id: int, user_id: Union[str, int], level: str) -> dict:
    """
    Thêm môn kế tiếp (theo manifest) ở level chỉ định vào cuối lộ trình `path_id` nếu chưa có.
    Chỉ hoạt động khi lộ trình đang STUDYING.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không thể kết nối database."}
    try:
        with conn.cursor() as cur:
            user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
            if user_id_int is None:
                return {"error": "user_id không hợp lệ."}
            # verify ownership and status
            cur.execute('SELECT status FROM learning_path WHERE id = %s AND user_id = %s;', (path_id, user_id_int))
            row = cur.fetchone()
            if not row:
                return {"error": "Không tìm thấy lộ trình hoặc bạn không có quyền."}
            if row[0] != 'STUDYING':
                return {"error": "Chỉ có thể thêm khi lộ trình đang hoạt động (STUDYING)."}
            # existing ids
            cur.execute('SELECT course_id FROM course_learning_path WHERE learning_path_id = %s ORDER BY course_order_number ASC;', (path_id,))
            existing_ids = [r[0] for r in cur.fetchall()]
            candidates = courses_by_level(level.upper()) or []
            next_id = None
            for cid in candidates:
                if cid not in existing_ids:
                    next_id = cid
                    break
            if not next_id:
                return {"error": "Không còn môn mới cho level này trong manifest."}
            # verify course exists in DB
            cur.execute('SELECT 1 FROM course WHERE id = %s;', (next_id,))
            if not cur.fetchone():
                return {"error": f"Môn {next_id} chưa có trong CSDL."}
            # append to end
            cur.execute('SELECT COALESCE(MAX(course_order_number), 0) FROM course_learning_path WHERE learning_path_id = %s;', (path_id,))
            last_order = cur.fetchone()[0]
            cur.execute('INSERT INTO course_learning_path (course_id, learning_path_id, course_order_number) VALUES (%s, %s, %s);', (next_id, path_id, last_order + 1))
            cur.execute('UPDATE learning_path SET last_updated_at = NOW() WHERE id = %s;', (path_id,))
            conn.commit()
            return {"success": True, "added_course_id": next_id, "new_order": last_order + 1}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

@tool
def check_course_availability_for_range(start_level: str, end_level: str) -> dict:
    """
    Kiểm tra danh sách môn theo manifest từ start_level -> end_level (inclusive),
    trả về các môn có trong DB và các môn còn thiếu.
    Dùng để cảnh báo khi hệ thống chưa cập nhật đủ môn ở level cao (ví dụ N2).
    """
    try:
        start = start_level.upper()
        end = end_level.upper()
        seq = course_sequence_between(start, end) or []
        if not seq:
            return {"error": "Không tìm thấy danh sách môn theo manifest."}
        # Xây map thiếu theo từng level trong khoảng
        order = ["N5","N4","N3","N2","N1"]
        try:
            i_start = order.index(start)
            i_end = order.index(end)
        except ValueError:
            i_start, i_end = 0, 0
        levels_in_range = order[min(i_start, i_end):max(i_start, i_end)+1]
        level_to_courses = {lv: set(courses_by_level(lv) or []) for lv in levels_in_range}
        # Kiểm tra tồn tại trong DB
        conn = get_db_connection()
        if not conn:
            return {"error": "Không thể kết nối DB"}
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM course WHERE id = ANY(%s);', (seq,))
            existing = {r[0] for r in cur.fetchall()}
        missing = [cid for cid in seq if cid not in existing]
        missing_by_level = {}
        for lv, lv_courses in level_to_courses.items():
            missing_by_level[lv] = [cid for cid in lv_courses if cid in set(missing)]
        return {
            "success": True,
            "wanted_courses": seq,
            "available_courses": [cid for cid in seq if cid in existing],
            "missing_courses": missing,
            "missing_by_level": {lv: len(ids) for lv, ids in missing_by_level.items()}
        }
    except Exception as e:
        return {"error": str(e)}

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
    """Trả về level hiện tại của user từ bảng users.level (ví dụ 'N4_M')."""
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
    """Cho level hiện tại (ví dụ 'N5_M'), trả danh sách course_id để đạt +2 level (N3_M)."""
    try:
        import re as _re
        main_level = _re.split(r"[-_]", current_level.upper())[0]
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
            # Chỉ tính thời lượng dựa trên các môn có thật trong DB
            cur.execute('SELECT id FROM course WHERE id = ANY(%s);', (course_ids,))
            existing_ids = [r[0] for r in cur.fetchall()]
            total = _sum_course_duration(cur, existing_ids)
        weeks = ceil(total / hours_per_week) if hours_per_week else None
        return {"success": True, "total_hours": total, "estimated_weeks": weeks}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

@tool
def get_latest_exam_result_for_exam(user_id: Union[str, int], exam_id: Union[str, int]) -> dict:
    """Trả về lần làm bài gần nhất của user cho một kỳ thi.

    - Nhận `exam_id` là mã số kỳ thi hoặc tiêu đề kỳ thi (title). Nếu là title, sẽ join bảng `exam`.
    - Thứ tự ưu tiên: bản ghi đã nộp (submitted) trước, sau đó theo thời gian gần nhất.
    """
    user_id_int = _SESSION_USER_ID if _SESSION_USER_ID is not None else _parse_user_id(user_id)
    if user_id_int is None:
        return {"error": "user_id không hợp lệ"}

    exam_id_num = None
    exam_title = None
    try:
        exam_id_num = int(str(exam_id))
    except Exception:
        exam_title = str(exam_id)

    if exam_title is not None:
        # Lọc theo tiêu đề kỳ thi
        rows = execute_sql_query(
            """
            SELECT er.id, er.score, er.status, er.exam_id, er.started_at, er.submitted_at, e.title AS exam_title
            FROM exam_result er
            JOIN exam e ON e.id = er.exam_id
            WHERE er.user_id = %(user_id)s AND e.title = %(exam_title)s
            ORDER BY (er.submitted_at IS NULL), er.submitted_at DESC, er.started_at DESC
            LIMIT 1;
            """,
            {"user_id": user_id_int, "exam_title": exam_title}
        )
    else:
        rows = execute_sql_query(
            """
            SELECT er.id, er.score, er.status, er.exam_id, er.started_at, er.submitted_at, e.title AS exam_title
            FROM exam_result er
            JOIN exam e ON e.id = er.exam_id
            WHERE er.user_id = %(user_id)s AND er.exam_id = %(exam_id)s
            ORDER BY (er.submitted_at IS NULL), er.submitted_at DESC, er.started_at DESC
            LIMIT 1;
            """,
            {"user_id": user_id_int, "exam_id": exam_id_num}
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
