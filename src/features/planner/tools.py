import re
from langchain.tools import tool
from pydantic import BaseModel, Field
import psycopg2
from typing import List, Dict, Any, Optional
from src.core.vector_store_interface import get_db_connection
from ...core.database import execute_sql_query
from ...core.context_manager import save_task_context, load_task_context, clear_task_context
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

class PathPlannerInput(BaseModel):
    current_level: str = Field(description="Trình độ hiện tại của người học, ví dụ: 'N5', 'N4', 'Mới bắt đầu'")
    learning_goal: str = Field(
        description="Mục đích học tập, ví dụ: 'Thi JLPT N2', 'Du lịch Nhật Bản', 'Làm việc trong công ty IT'")
    personal_interests: str = Field(
        description="Các sở thích hoặc kỹ năng muốn tập trung, ví dụ: 'anime, manga, luyện nói, kanji'")
    weekly_study_time: int = Field(description="Số giờ có thể học mỗi tuần, ví dụ: 5, 10")

def execute_sql_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    if not conn: return []
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            for row in rows:
                results.append(dict(zip(colnames, row)))
    except psycopg2.Error as e:
        print(f"Lỗi thực thi SQL: {e}")
    finally:
        if conn: conn.close()
    return results


# @tool(args_schema=PathPlannerInput)
# def create_full_learning_path(current_level: str, learning_goal: str, personal_interests: str,
#                               weekly_study_time: int) -> str:
#     """
#     Tự động tạo ra một lộ trình học tập chi tiết và cá nhân hóa dựa trên 4 thông tin đầu vào:
#     trình độ hiện tại, mục tiêu học tập, sở thích cá nhân, và thời gian học hàng tuần.
#     """
#     print("--- Tool Lập Kế Hoạch được kích hoạt ---")
#     print(
#         f"Input: Level={current_level}, Mục tiêu={learning_goal}, Sở thích={personal_interests}, Thời gian={weekly_study_time}h/tuần")
#
#     # 1. Xác định Level mục tiêu
#     levels = ['N5', 'N4', 'N3', 'N2', 'N1']
#     try:
#         current_idx = levels.index(current_level.upper())
#     except ValueError:
#         current_idx = -1
#
#     # Tìm level cao nhất được đề cập trong mục tiêu
#     target_level_match = re.search(r'N[1-5]', learning_goal.upper())
#     if target_level_match:
#         target_idx = levels.index(target_level_match.group(0))
#     else:
#         # Nếu mục tiêu không rõ ràng (du lịch, sở thích), chỉ cần học 1-2 level tiếp theo
#         target_idx = min(current_idx + 2, len(levels) - 1)
#
#     path_levels = levels[current_idx + 1: target_idx + 1]
#     if not path_levels:
#         return f"Tuyệt vời! Dường như bạn đã đạt được mục tiêu học tập của mình rồi. Nếu muốn học lên cao hơn, hãy cho tôi biết mục tiêu mới nhé!"
#
#     print(f"Lộ trình level cần học: {path_levels}")
#
#     # 2. Lấy tất cả các môn học ứng viên
#     query = "SELECT * FROM subjects WHERE level IN %s;"
#     all_subjects = execute_sql_query(query, (tuple(path_levels),))
#
#     if not all_subjects:
#         return "Xin lỗi, hiện tại chưa có môn học nào trong database phù hợp với lộ trình của bạn."
#
#     # 3. Chấm điểm và lựa chọn môn học (Logic "AI" nằm ở đây)
#     interest_keywords = [kw.strip().lower() for kw in personal_interests.split(',')]
#     scored_subjects = []
#     for subject in all_subjects:
#         score = 0
#         description = (subject.get('description', '') or '').lower()
#         subject_name = (subject.get('subject_name', '') or '').lower()
#         topic = (subject.get('topic', '') or '').lower()
#         for keyword in interest_keywords:
#             if keyword in description or keyword in subject_name or keyword in topic:
#                 score += 5  # Rất liên quan đến sở thích
#         if topic and topic in learning_goal.lower():
#             score += 10  # Rất liên quan đến mục tiêu (ví dụ: topic 'IT' và mục tiêu 'Làm việc IT')
#
#         scored_subjects.append({"subject": subject, "score": score})
#
#     # Sắp xếp và chọn ra 2-3 môn có điểm cao nhất
#     top_subjects_data = sorted(scored_subjects, key=lambda x: x['score'], reverse=True)[:3]
#
#     # 4. Lấy bài học và tổng hợp thành lộ trình
#     final_plan = f"Dựa trên thông tin bạn cung cấp, đây là lộ trình học tập được cá nhân hóa cho bạn:\n"
#     final_plan += f"Mục tiêu: {learning_goal}\nLộ trình: Từ {current_level} -> {levels[target_idx]}\n\n"
#
#     total_lessons = 0
#     for item in top_subjects_data:
#         subject = item['subject']
#         final_plan += f"--- Môn học gợi ý: {subject['subject_name']} (Level: {subject['level']}) ---\n"
#         final_plan += f"Lý do gợi ý: Môn học này phù hợp với sở thích ({personal_interests}) và mục tiêu của bạn.\n"
#
#         lessons = execute_sql_query("SELECT lesson_name, lesson_type FROM lessons WHERE subject_id = %s;",
#                                     (subject['id'],))
#         if lessons:
#             final_plan += "Các bài học chính bạn nên tập trung:\n"
#             for lesson in lessons:
#                 final_plan += f"  - [{lesson['lesson_type']}] {lesson['lesson_name']}\n"
#             total_lessons += len(lessons)
#         final_plan += "\n"
#
#     # 5. Phân tích thời gian (tùy chọn)
#     if total_lessons > 0 and weekly_study_time > 0:
#         estimated_weeks = round((total_lessons * 2) / weekly_study_time)  # Giả sử mỗi bài học cần 2 giờ
#         final_plan += f"** Ước tính thời gian hoàn thành lộ trình trên với {weekly_study_time} giờ/tuần là khoảng {estimated_weeks} tuần. **\n"
#
#     return final_plan


class CourseSearchInput(BaseModel):
    level: str = Field(description="Cấp độ mục tiêu, ví dụ: 'N3'.")
    goal: str = Field(description="Mục tiêu chính, ví dụ: 'Thi JLPT'.")

@tool(args_schema=CourseSearchInput)
def find_relevant_courses(level: str, goal: str) -> List[Dict[str, Any]]:
    """
    Tìm các khóa học (Courses) phù hợp nhất bằng cách JOIN với bảng Topic
    và tìm kiếm trong các trường khác.
    """
    print(f"--- Tool: Đang tìm các khóa học Level '{level}' cho mục tiêu '{goal}' ---")

    # === SỬA LỖI Ở ĐÂY: Câu lệnh SQL mới sử dụng JOIN ===
    # Nó sẽ tìm kiếm cả trong tên của Topic và trong các trường của Course
    query = """
        SELECT DISTINCT
            C.id,
            C.title,
            C.description
        FROM
            "Course" C
        LEFT JOIN
            "CourseTopic" CT ON C.id = CT.course_id
        LEFT JOIN
            "Topic" T ON CT.topic_id = T.id
        WHERE
            LOWER(C.level) = LOWER(%s)
            AND (
                LOWER(T.name) LIKE LOWER(%s)
                OR LOWER(C.requirement) LIKE LOWER(%s)
                OR LOWER(C.description) LIKE LOWER(%s)
            );
    """
    search_term = f"%{goal}%"
    return execute_sql_query(query, (level, search_term, search_term, search_term))

# --- Tool 2: Lấy cấu trúc chi tiết của một Khóa học ---
class CourseStructureInput(BaseModel):
    course_id: str = Field(description="ID của khóa học cần lấy cấu trúc.")

@tool(args_schema=CourseStructureInput)
def get_course_structure(course_id: str) -> List[Dict[str, Any]]:
    """
    Lấy toàn bộ cấu trúc của một khóa học, bao gồm tất cả các Chapters, Units,
    và Materials (cùng với skill_type của chúng).
    """
    print(f"--- Tool: Đang lấy cấu trúc của khóa học ID '{course_id}' ---")
    query = """
        SELECT
            CH.id AS chapter_id,
            CH.title AS chapter_title,
            U.id AS unit_id,
            U.title AS unit_title,
            M.id AS material_id,
            M.type AS skill_type    
        FROM
            "Chapter" CH
        JOIN
            "Unit" U ON CH.id = U.chapter_id
        JOIN
            "Material" M ON U.id = M.unit_id
        WHERE
            CH.course_id = %s;       
    """
    return execute_sql_query(query, (course_id,))

class SavePathInput(BaseModel):
    user_id: str = Field(description="ID của người dùng.")
    path_title: str = Field(description="Tên của lộ trình học.")
    unit_ids: List[str] = Field(description="Danh sách ID của các Unit cần đưa vào lộ trình.")

@tool(args_schema=SavePathInput)
def save_learning_path_with_units(user_id: str, path_title: str, unit_ids: List[str]) -> str:
    """Tạo một lộ trình học mới và liên kết các Unit đã được chọn vào đó."""
    print(f"--- Tool: Đang lưu lộ trình '{path_title}' cho người dùng '{user_id}' ---")
    conn = get_db_connection()
    if not conn: return "Lỗi: Không thể kết nối database."

    try:
        with conn.cursor() as cur:
            # 1. Tạo bản ghi trong LearningPath
            cur.execute(
                """
                INSERT INTO "LearningPath" (user_id, title)
                VALUES (%s, %s) RETURNING id;
                """,
                (user_id, path_title)
            )
            path_id = cur.fetchone()[0]

            # 2. Chèn các liên kết vào UnitProgress
            for index, unit_id in enumerate(unit_ids):
                cur.execute(
                    """
                    INSERT INTO "UnitProgress" (learning_path_id, unit_id, course_order_number)
                    VALUES (%s, %s, %s);
                    """,
                    (path_id, unit_id, index + 1)
                )

            conn.commit()
            return f"Đã lưu thành công lộ trình mới với ID là {path_id}, bao gồm {len(unit_ids)} bài học."
    except Exception as e:
        conn.rollback()
        return f"Lỗi khi lưu lộ trình: {e}"
    finally:
        if conn: conn.close()


class UpdateProfileInput(BaseModel):
    user_id: str = Field(description="ID của người dùng cần cập nhật hồ sơ.")
    updates: Dict[str, Any] = Field(
        description="Một dictionary chứa các trường cần cập nhật, ví dụ: {'level': 'N4', 'target': 'Đi làm'}.")


@tool(args_schema=UpdateProfileInput)
def update_user_profile(user_id: str, updates: Dict[str, Any]) -> str:
    """
    Cập nhật thông tin hồ sơ người dùng (level, target, hobby) vào database ngay lập tức.
    Sử dụng tool này NGAY SAU KHI bạn nhận được một mẩu thông tin cá nhân từ người dùng.
    """
    print(f"--- Tool: Đang cập nhật hồ sơ cho user '{user_id}' với dữ liệu: {updates} ---")
    conn = get_db_connection()
    if not conn:
        return "Lỗi: Không thể kết nối database."
    set_clauses = []
    params = []
    allowed_columns = ['level', 'target', 'hobby']

    for key, value in updates.items():
        if key in allowed_columns:
            set_clauses.append(f'"{key}" = %s')
            params.append(value)

    if not set_clauses:
        return "Không có trường hợp lệ nào để cập nhật."

    params.append(user_id)
    query = f'UPDATE "User" SET {", ".join(set_clauses)}, last_updated_at = NOW() WHERE id = %s;'

    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            updated_rows = cur.rowcount
            conn.commit()
            if updated_rows > 0:
                return f"Đã cập nhật thành công các trường: {list(updates.keys())} cho người dùng {user_id}."
            else:
                # Có thể người dùng chưa tồn tại, cân nhắc tạo mới hoặc báo lỗi
                return f"Không tìm thấy người dùng với ID {user_id} để cập nhật."
    except Exception as e:
        conn.rollback()
        return f"Lỗi khi cập nhật hồ sơ: {e}"
    finally:
        if conn:
            conn.close()

@tool
def get_ongoing_plan_context(session_id: int) -> Optional[Dict[str, Any]]:
    """
    Sử dụng tool này ĐẦU TIÊN trong một phiên tư vấn lộ trình để kiểm tra xem
    có một kế hoạch nào đang được xây dựng dở dang từ trước không.
    """
    print(f"--- Tool: Đang kiểm tra context dang dở cho session {session_id} ---")
    return load_task_context(session_id, "PLANNER")

@tool
def save_plan_context(session_id: int, status: str, collected_data: Dict[str, Any]) -> str:
    """
    Sử dụng tool này SAU MỖI LƯỢT nói chuyện để lưu lại tiến trình
    thu thập thông tin cho việc tạo lộ trình.
    """
    save_task_context(session_id, "PLANNER", status, collected_data)
    return "Đã lưu lại tiến trình."

@tool
def clear_plan_context(session_id: int) -> str:
    """
    Sử dụng tool này CUỐI CÙNG, sau khi đã tạo và lưu lộ trình thành công,
    để xóa context của nhiệm vụ đã hoàn thành.
    """
    clear_task_context(session_id, "PLANNER")
    return "Đã hoàn thành và xóa context nhiệm vụ."


class TimeConstraintInput(BaseModel):
    deadline_info: str = Field(
        description="Mô tả về thời hạn của người dùng, ví dụ: 'tháng 12 này', 'trong 3 tháng tới'.")


@tool(args_schema=TimeConstraintInput)
def calculate_time_constraints(deadline_info: str) -> Optional[Dict[str, Any]]:
    """
    Phân tích một chuỗi mô tả thời gian để tính toán ngày hết hạn và số ngày còn lại.
    """
    print(f"--- Tool: Đang phân tích thời gian cho: '{deadline_info}' ---")
    today = datetime.date.today()
    deadline_info_lower = deadline_info.lower()
    target_date = None

    try:
        # Xử lý các trường hợp phổ biến
        if "tháng" in deadline_info_lower and "này" in deadline_info_lower:
            month_match = re.search(r'tháng\s*(\d{1,2})', deadline_info_lower)
            if month_match:
                target_month = int(month_match.group(1))
                target_date = datetime.date(today.year, target_month, 1)
        elif "tháng" in deadline_info_lower and "sau" in deadline_info_lower:
            months_to_add = int(re.search(r'(\d+)', deadline_info_lower).group(1))
            target_date = today + relativedelta(months=months_to_add)
        elif "hè" in deadline_info_lower:
            target_date = datetime.date(today.year, 6, 1)  # Mặc định hè là tháng 6
            if target_date < today:  # Nếu đã qua hè năm nay thì tính hè năm sau
                target_date = datetime.date(today.year + 1, 6, 1)
        else:
            # Thử dùng thư viện parse mạnh mẽ hơn cho các trường hợp khác
            # Cần cài đặt: pip install python-dateutil
            target_date = parse(deadline_info, fuzzy=True, default=datetime.datetime(today.year, today.month, 1)).date()

    except (ValueError, TypeError, AttributeError):
        return {"error": "Không thể xác định ngày hết hạn từ thông tin bạn cung cấp."}

    if target_date:
        days_remaining = (target_date - today).days
        if days_remaining > 0:
            return {
                "days_remaining": days_remaining,
                "target_date_iso": target_date.isoformat()
            }

    return {"error": "Thông tin thời gian không hợp lệ hoặc đã ở trong quá khứ."}

@tool
def list_user_learning_paths(user_id: str) -> List[Dict[str, Any]]:
    """
    Sử dụng tool này ĐẦU TIÊN để kiểm tra xem người dùng đã có lộ trình nào được lưu từ trước chưa.
    """
    print(f"--- Tool: Đang lấy danh sách lộ trình cho user '{user_id}' ---")
    query = """
        SELECT id, title, description, last_updated_at
        FROM "LearningPath"
        WHERE user_id = %s
        ORDER BY last_updated_at DESC;
    """
    return execute_sql_query(query, (user_id,))

# --- Tool 2: Lấy thông tin hồ sơ người dùng ---
@tool
def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Sử dụng tool này để lấy thông tin hồ sơ (level, target, hobby) đã được lưu
    trong bảng "User".
    """
    print(f"--- Tool: Đang lấy hồ sơ của user '{user_id}' ---")
    query = 'SELECT level, hobby FROM "User" WHERE id = %s;'
    results = execute_sql_query(query, (user_id,))
    return results[0] if results else None

class SavePathInput(BaseModel):
    user_id: str = Field(...)
    path_title: str = Field(...)
    unit_ids: List[str] = Field(...)

@tool(args_schema=SavePathInput)
def save_new_learning_path(user_id: str, path_title: str, unit_ids: List[str]) -> str:
    """Lưu một lộ trình học hoàn chỉnh vào database."""
    # Logic để lưu vào bảng LearningPath và UnitProgress sẽ được triển khai ở đây
    print(f"--- Tool: (Chưa hoạt động) Đang lưu lộ trình '{path_title}' ---")
    return f"Đã lưu thành công lộ trình '{path_title}' với {len(unit_ids)} bài học."


@tool
def list_user_learning_paths(user_id: str) -> List[Dict[str, Any]]:
    """
    Sử dụng tool này để kiểm tra xem người dùng đã có những lộ trình nào được lưu từ trước.
    Trả về một danh sách các lộ trình, bao gồm cả trạng thái (active/archived).
    """
    print(f"--- Tool: Đang lấy danh sách lộ trình cho user '{user_id}' ---")
    query = """
            SELECT id, title, status, last_updated_at
            FROM "learning_path"
            WHERE user_id = %s
            ORDER BY last_updated_at DESC; \
            """
    return execute_sql_query(query, (user_id,))


# --- Tool 2: Tìm kiếm các Khóa học phù hợp ---
class CourseSearchInput(BaseModel):
    level: str = Field(description="Cấp độ mục tiêu của người dùng, ví dụ: 'N3'.")
    goal: str = Field(description="Mục tiêu chính của người dùng, ví dụ: 'Thi JLPT', 'Giao tiếp'.")


@tool(args_schema=CourseSearchInput)
def find_relevant_courses(level: str, goal: str) -> List[Dict[str, Any]]:
    """
    Tìm các khóa học (Courses) phù hợp nhất dựa trên cấp độ và mục tiêu học tập.
    """
    print(f"--- Tool: Đang tìm các khóa học Level '{level}' cho mục tiêu '{goal}' ---")
    # Câu lệnh này có thể được cải tiến với full-text search hoặc JOIN với bảng Topic
    query = """
            SELECT id, title, description, requirement, duration
            FROM "course"
            WHERE LOWER(level) = LOWER(%s)
              AND (
                LOWER(requirement) LIKE LOWER(%s)
                    OR LOWER(description) LIKE LOWER(%s)
                    OR LOWER(title) LIKE LOWER(%s)
                ); \
            """
    search_term = f"%{goal}%"
    return execute_sql_query(query, (level, search_term, search_term, search_term))


# --- Tool 3: Tạo và Lưu một Lộ trình học mới ---
class CreatePathInput(BaseModel):
    user_id: str = Field(description="ID của người dùng.")
    title: str = Field(description="Tên của lộ trình học mới.")
    description: str = Field(description="Mô tả ngắn gọn về lộ trình.")
    target_level: str = Field(description="Cấp độ mục tiêu của lộ trình.")
    primary_goal: str = Field(description="Mục tiêu chính của lộ trình.")
    focus_skill: str = Field(description="Kỹ năng chính muốn tập trung.")
    course_ids: List[str] = Field(description="Danh sách ID của các Course cần đưa vào lộ trình.")


@tool(args_schema=CreatePathInput)
def create_new_learning_path(user_id: str, title: str, description: str, target_level: str, primary_goal: str,
                             focus_skill: str, course_ids: List[str]) -> str:
    """
    Tạo một lộ trình học mới và liên kết các Khóa học (Courses) đã được chọn vào đó.
    Tool này sẽ tự động chuyển các lộ trình cũ của người dùng sang trạng thái 'ARCHIVED'.
    """
    print(f"--- Tool: Đang tạo lộ trình mới '{title}' cho user '{user_id}' ---")
    conn = get_db_connection()
    if not conn: return "Lỗi: Không thể kết nối database."

    try:
        with conn.cursor() as cur:
            # Bước 1: Deactivate tất cả các lộ trình cũ của user này
            cur.execute('UPDATE "learning_path" SET status = \'ARCHIVED\' WHERE user_id = %s;', (user_id,))
            print(f"Đã lưu trữ các lộ trình cũ của user '{user_id}'.")

            # Bước 2: Tạo bản ghi mới trong learning_path với status là 'ACTIVE'
            cur.execute(
                """
                INSERT INTO "learning_path" (user_id, title, description, target_level, primary_goal, focus_skill,
                                             status)
                VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE') RETURNING id;
                """,
                (user_id, title, description, target_level, primary_goal, focus_skill)
            )
            path_id = cur.fetchone()[0]

            # Bước 3: Chèn các liên kết vào course_learning_path
            if course_ids:
                values_to_insert = [(course_id, path_id, index + 1) for index, course_id in enumerate(course_ids)]
                from psycopg2.extras import execute_values
                execute_values(
                    cur,
                    'INSERT INTO "course_learning_path" (course_id, learning_path_id, course_order_number) VALUES %s;',
                    values_to_insert
                )

            conn.commit()
            return f"Đã tạo và kích hoạt thành công lộ trình mới '{title}' (ID: {path_id}) với {len(course_ids)} khóa học."
    except Exception as e:
        conn.rollback()
        return f"Lỗi khi tạo lộ trình: {e}"
    finally:
        if conn: conn.close()


# --- Tool 4: Xóa một Lộ trình học ---
@tool
def delete_learning_path(path_id: int) -> str:
    """
    Xóa một lộ trình học cụ thể dựa trên ID của nó.
    Hành động này không thể hoàn tác.
    """
    print(f"--- Tool: Đang xóa lộ trình ID: {path_id} ---")
    conn = get_db_connection()
    if not conn: return "Lỗi: Không thể kết nối database."

    try:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM "learning_path" WHERE id = %s;', (path_id,))
            deleted_rows = cur.rowcount
            conn.commit()
            if deleted_rows > 0:
                return f"Đã xóa thành công lộ trình ID {path_id}."
            else:
                return f"Không tìm thấy lộ trình nào có ID {path_id} để xóa."
    except Exception as e:
        conn.rollback()
        return f"Lỗi khi xóa lộ trình: {e}"
    finally:
        if conn: conn.close()


@tool
def list_user_learning_paths(user_id: str) -> List[Dict[str, Any]]:
    """Lấy danh sách tóm tắt các lộ trình học của một người dùng."""
    print(f"--- Tool: Đang lấy danh sách lộ trình cho user '{user_id}' ---")
    query = 'SELECT id, title, status FROM "learning_path" WHERE user_id = %s ORDER BY last_updated_at DESC;'
    return execute_sql_query(query, (user_id,))


@tool
def get_learning_path_details(path_id: int) -> Dict[str, Any]:
    """Lấy thông tin chi tiết của một lộ trình học, bao gồm cả các khóa học bên trong."""
    print(f"--- Tool: Đang lấy chi tiết lộ trình ID: {path_id} ---")
    path_details_query = 'SELECT * FROM "learning_path" WHERE id = %s;'
    courses_query = """
                    SELECT C.id, C.title, CLP.course_order_number
                    FROM "course" C
                             JOIN "course_learning_path" CLP ON C.id = CLP.course_id
                    WHERE CLP.learning_path_id = %s
                    ORDER BY CLP.course_order_number ASC; \
                    """
    path_info = execute_sql_query(path_details_query, (path_id,))
    if not path_info:
        return {"error": f"Không tìm thấy lộ trình với ID {path_id}."}

    courses_in_path = execute_sql_query(courses_query, (path_id,))
    path_info[0]['courses'] = courses_in_path
    return path_info[0]


@tool
def add_courses_to_learning_path(path_id: int, course_ids: List[str]) -> str:
    """Thêm một hoặc nhiều khóa học mới vào cuối một lộ trình đã có."""
    print(f"--- Tool: Đang thêm các khóa học {course_ids} vào lộ trình ID: {path_id} ---")
    conn = get_db_connection()
    if not conn: return "Lỗi: Không thể kết nối database."
    try:
        with conn.cursor() as cur:
            # Lấy order number lớn nhất hiện có
            cur.execute(
                'SELECT COALESCE(MAX(course_order_number), 0) FROM "course_learning_path" WHERE learning_path_id = %s;',
                (path_id,))
            last_order = cur.fetchone()[0]

            # Chèn các khóa học mới
            values_to_insert = [(course_id, path_id, last_order + i + 1) for i, course_id in enumerate(course_ids)]
            execute_values(cur,
                           'INSERT INTO "course_learning_path" (course_id, learning_path_id, course_order_number) VALUES %s;',
                           values_to_insert)

            # Cập nhật thời gian
            cur.execute('UPDATE "learning_path" SET last_updated_at = NOW() WHERE id = %s;', (path_id,))
            conn.commit()
            return f"Đã thêm thành công {len(course_ids)} khóa học vào lộ trình."
    except Exception as e:
        conn.rollback();
        return f"Lỗi khi thêm khóa học: {e}"
    finally:
        if conn: conn.close()


@tool
def reorder_courses_in_learning_path(path_id: int, ordered_course_ids: List[str]) -> str:
    """Cập nhật lại toàn bộ thứ tự của các khóa học trong một lộ trình."""
    print(f"--- Tool: Đang sắp xếp lại các khóa học trong lộ trình ID: {path_id} ---")
    conn = get_db_connection()
    if not conn: return "Lỗi: Không thể kết nối database."
    try:
        with conn.cursor() as cur:
            # Dùng vòng lặp để UPDATE từng dòng
            for index, course_id in enumerate(ordered_course_ids):
                cur.execute(
                    'UPDATE "course_learning_path" SET course_order_number = %s WHERE learning_path_id = %s AND course_id = %s;',
                    (index + 1, path_id, course_id)
                )
            cur.execute('UPDATE "learning_path" SET last_updated_at = NOW() WHERE id = %s;', (path_id,))
            conn.commit()
            return "Đã cập nhật thành công thứ tự các khóa học."
    except Exception as e:
        conn.rollback();
        return f"Lỗi khi sắp xếp lại khóa học: {e}"
    finally:
        if conn: conn.close()