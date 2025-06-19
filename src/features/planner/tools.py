import re
from langchain.tools import tool
from pydantic import BaseModel, Field
import psycopg2
from typing import List, Dict, Any
from src.core.vector_store_interface import get_db_connection

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

        # Chấm điểm dựa trên sở thích và mục tiêu
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