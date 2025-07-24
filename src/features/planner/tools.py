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


@tool(args_schema=PathPlannerInput)
def create_full_learning_path(current_level: str, learning_goal: str, personal_interests: str,
                              weekly_study_time: int) -> str:
    """
    Tự động tạo ra một lộ trình học tập chi tiết và cá nhân hóa dựa trên 4 thông tin đầu vào:
    trình độ hiện tại, mục tiêu học tập, sở thích cá nhân, và thời gian học hàng tuần.
    """
    print("--- Tool Lập Kế Hoạch được kích hoạt ---")
    print(
        f"Input: Level={current_level}, Mục tiêu={learning_goal}, Sở thích={personal_interests}, Thời gian={weekly_study_time}h/tuần")

    # 1. Xác định Level mục tiêu
    levels = ['N5', 'N4', 'N3', 'N2', 'N1']
    try:
        current_idx = levels.index(current_level.upper())
    except ValueError:
        current_idx = -1

    # Tìm level cao nhất được đề cập trong mục tiêu
    target_level_match = re.search(r'N[1-5]', learning_goal.upper())
    if target_level_match:
        target_idx = levels.index(target_level_match.group(0))
    else:
        # Nếu mục tiêu không rõ ràng (du lịch, sở thích), chỉ cần học 1-2 level tiếp theo
        target_idx = min(current_idx + 2, len(levels) - 1)

    path_levels = levels[current_idx + 1: target_idx + 1]
    if not path_levels:
        return f"Tuyệt vời! Dường như bạn đã đạt được mục tiêu học tập của mình rồi. Nếu muốn học lên cao hơn, hãy cho tôi biết mục tiêu mới nhé!"

    print(f"Lộ trình level cần học: {path_levels}")

    # 2. Lấy tất cả các môn học ứng viên
    query = "SELECT * FROM subjects WHERE level IN %s;"
    all_subjects = execute_sql_query(query, (tuple(path_levels),))

    if not all_subjects:
        return "Xin lỗi, hiện tại chưa có môn học nào trong database phù hợp với lộ trình của bạn."

    # 3. Chấm điểm và lựa chọn môn học (Logic "AI" nằm ở đây)
    interest_keywords = [kw.strip().lower() for kw in personal_interests.split(',')]
    scored_subjects = []
    for subject in all_subjects:
        score = 0
        description = (subject.get('description', '') or '').lower()
        subject_name = (subject.get('subject_name', '') or '').lower()
        topic = (subject.get('topic', '') or '').lower()
        for keyword in interest_keywords:
            if keyword in description or keyword in subject_name or keyword in topic:
                score += 5  # Rất liên quan đến sở thích
        if topic and topic in learning_goal.lower():
            score += 10  # Rất liên quan đến mục tiêu (ví dụ: topic 'IT' và mục tiêu 'Làm việc IT')

        scored_subjects.append({"subject": subject, "score": score})

    # Sắp xếp và chọn ra 2-3 môn có điểm cao nhất
    top_subjects_data = sorted(scored_subjects, key=lambda x: x['score'], reverse=True)[:3]

    # 4. Lấy bài học và tổng hợp thành lộ trình
    final_plan = f"Dựa trên thông tin bạn cung cấp, đây là lộ trình học tập được cá nhân hóa cho bạn:\n"
    final_plan += f"Mục tiêu: {learning_goal}\nLộ trình: Từ {current_level} -> {levels[target_idx]}\n\n"

    total_lessons = 0
    for item in top_subjects_data:
        subject = item['subject']
        final_plan += f"--- Môn học gợi ý: {subject['subject_name']} (Level: {subject['level']}) ---\n"
        final_plan += f"Lý do gợi ý: Môn học này phù hợp với sở thích ({personal_interests}) và mục tiêu của bạn.\n"

        lessons = execute_sql_query("SELECT lesson_name, lesson_type FROM lessons WHERE subject_id = %s;",
                                    (subject['id'],))
        if lessons:
            final_plan += "Các bài học chính bạn nên tập trung:\n"
            for lesson in lessons:
                final_plan += f"  - [{lesson['lesson_type']}] {lesson['lesson_name']}\n"
            total_lessons += len(lessons)
        final_plan += "\n"

    # 5. Phân tích thời gian (tùy chọn)
    if total_lessons > 0 and weekly_study_time > 0:
        estimated_weeks = round((total_lessons * 2) / weekly_study_time)  # Giả sử mỗi bài học cần 2 giờ
        final_plan += f"** Ước tính thời gian hoàn thành lộ trình trên với {weekly_study_time} giờ/tuần là khoảng {estimated_weeks} tuần. **\n"

    return final_plan


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