# src/features/qna/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Import các thành phần cốt lõi
from ...core.database import execute_sql_query
from ...core.embedding import get_embedding_model
from ...core.session_manager import update_session_context
from ...core.database import get_db_connection
from ...config import settings

embedding_model = get_embedding_model()

# --- Quiz/Answer session binding ---
_QNA_SESSION_ID: Optional[int] = None


def set_qna_session_id(session_id: int) -> None:
    """
    Gắn `session_id` hiện tại cho QnA tools để có thể lưu state quiz vào context của phiên.
    """
    global _QNA_SESSION_ID
    _QNA_SESSION_ID = int(session_id)


@tool
def get_session_context_tool() -> Dict[str, Any]:
    """
    Lấy `session.context` của phiên QnA hiện tại (dựa trên session_id đã gắn).
    Dùng để đọc `suggested_exam_id`, `exam_completed`, v.v.
    """
    if _QNA_SESSION_ID is None:
        return {"error": "session_id chưa được gắn cho QnA."}
    rows = execute_sql_query("SELECT context FROM session WHERE id = %s;", (_QNA_SESSION_ID,))
    if not rows:
        return {"error": "Không tìm thấy phiên."}
    return {"success": True, "context": rows[0].get("context") or {}}


# --- Tool 1: Lấy hồ sơ người dùng ---
@tool
def get_user_profile_tool(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Sử dụng tool này để lấy thông tin hồ sơ (level, hobby, target) đã được lưu
    của một người dùng cụ thể từ database.
    """
    print(f"--- Tool: Đang lấy hồ sơ của user '{user_id}' ---")
    query = 'SELECT level, hobby, target FROM "users" WHERE id = %s;'
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

    # Một số DB không có cột level/skill_type => fallback vào metadata_json
    base_query = 'SELECT chunk_text, course_id FROM "content_chunks"'
    where_clauses = []
    params = []
    if course_id:
        where_clauses.append('"course_id" = %s')
        params.append(course_id)
    if level:
        # dùng COALESCE để hỗ trợ cả cột level hoặc metadata_json->>'level'
        where_clauses.append("LOWER(COALESCE(level, metadata_json->>'level')) = LOWER(%s)")
        params.append(level)
    if skill_type:
        where_clauses.append("LOWER(COALESCE(skill_type, metadata_json->>'skill_type')) = LOWER(%s)")
        params.append(skill_type)

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)

    base_query += " ORDER BY embedding <=> %s LIMIT 3;"
    params.append(str(query_embedding))

    results = execute_sql_query(base_query, tuple(params))

    if not results:
        return "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

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
    query = 'SELECT title, description FROM "course" WHERE id = %s;'
    results = execute_sql_query(query, (course_id,))

    if not results:
        return f"Không tìm thấy thông tin cho khóa học có ID {course_id}."

    course = results[0]
    return f"Môn học '{course.get('title')}' (ID: {course_id})."


# ============================
# Level Test utilities (QnA)
# ============================

@tool
def list_available_level_tests_tool(level: Optional[str] = None) -> Dict[str, Any]:
    """
    Liệt kê các đề test JLPT có sẵn trong DB (bảng exam). Nếu không có, fallback ra danh sách mặc định.
    """
    normalized = (level or "").upper()
    try:
        query = """
            SELECT id, title
            FROM exam
            WHERE type ILIKE 'JLPT' OR title ILIKE 'Test-JLPT%'
            ORDER BY id ASC;
        """
        rows = execute_sql_query(query, ())
        items = [{"exam_id": r.get("id"), "title": r.get("title")} for r in rows] if rows else []
        if normalized:
            items = [it for it in items if normalized in str(it.get("title") or "").upper()]
        if items:
            return {"success": True, "tests": items}
    except Exception:
        pass
    # Fallback danh sách mặc định
    defaults = [
        {"exam_id": "Test-JLPT-N5-exam01", "title": "JLPT N5 - Đề luyện tập 01"},
        {"exam_id": "Test-JLPT-N4-exam01", "title": "JLPT N4 - Đề luyện tập 01"},
        {"exam_id": "Test-JLPT-N3-exam01", "title": "JLPT N3 - Đề luyện tập 01"},
    ]
    if normalized:
        defaults = [d for d in defaults if normalized in d["exam_id"].upper()]
    return {"success": True, "tests": defaults}


@tool
def generate_level_test_link_tool(level: str) -> str:
    """
    Trả về link làm bài test theo level: {FRONTEND_BASE_URL}/exam/{examId}/preparation.
    Tự chọn examId phù hợp (ưu tiên DB, fallback danh sách mặc định).
    """
    lv = (level or "").upper()
    items = list_available_level_tests_tool(lv).get("tests", [])
    exam_id = None
    if items:
        exam_id = items[0].get("exam_id")
    if not exam_id:
        exam_id = f"Test-JLPT-{lv}-exam01"
    base = settings.FRONTEND_BASE_URL.rstrip('/')
    return f"Hãy mở link để làm bài test: {base}/exam/{exam_id}/preparation"


@tool
def get_user_level_tool(user_id: str) -> Dict[str, Any]:
    """
    Lấy level hiện tại từ bảng users.level
    """
    try:
        rows = execute_sql_query('SELECT level FROM "users" WHERE id = %s;', (user_id,))
        level_val = rows[0]["level"] if rows else None
        return {"success": True, "level": level_val}
    except Exception as e:
        return {"error": str(e)}


@tool
def update_user_level_tool(user_id: str, new_level: str) -> str:
    """
    Cập nhật users.level
    """
    try:
        conn = get_db_connection()
        if not conn:
            return "Lỗi: không kết nối DB"
        with conn.cursor() as cur:
            cur.execute('UPDATE "users" SET level = %s WHERE id = %s;', (new_level, user_id))
            conn.commit()
        return "Đã cập nhật level người dùng."
    except Exception as e:
        return f"Lỗi: {str(e)}"
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


# ============================
# Quiz batch save & grading
# ============================

class QuizQuestionItem(BaseModel):
    question_id: str = Field(...)
    scope: Optional[str] = Field(None)
    topic: Optional[str] = Field(None)
    choices: Optional[List[str]] = Field(None)
    correct_answer: Optional[str] = Field(None)


class SaveQuizBatchInput(BaseModel):
    questions: List[QuizQuestionItem]


@tool(args_schema=SaveQuizBatchInput)
def save_quiz_batch_tool(questions: List[Dict[str, Any]]) -> str:
    """
    Lưu danh sách câu hỏi quiz vào session.context.qna_quiz.questions và đặt current_index = 1.
    """
    if _QNA_SESSION_ID is None:
        return "Lỗi: session_id chưa được gắn cho QnA."
    # Chuẩn hóa questions -> list[dict] thuần để JSON serialize được
    normalized: List[Dict[str, Any]] = []
    try:
        for q in questions:
            if hasattr(q, 'dict') and callable(getattr(q, 'dict')):
                qd = q.dict()
            elif isinstance(q, dict):
                qd = dict(q)
            else:
                # Fallback best-effort
                qd = {"question_id": str(q)}
            # ép kiểu các trường cần thiết
            qd["question_id"] = str(qd.get("question_id", ""))
            if qd.get("choices") is not None:
                qd["choices"] = [str(x) for x in qd.get("choices")]
            if qd.get("scope") is not None:
                qd["scope"] = str(qd.get("scope"))
            if qd.get("topic") is not None:
                qd["topic"] = str(qd.get("topic"))
            if qd.get("correct_answer") is not None:
                qd["correct_answer"] = str(qd.get("correct_answer"))
            normalized.append(qd)
    except Exception:
        return "Lỗi: không thể chuẩn hóa danh sách câu hỏi."

    payload: Dict[str, Any] = {
        "qna_quiz": {
            "questions": normalized,
            "current_index": 1
        }
    }
    ok = update_session_context(_QNA_SESSION_ID, payload)
    return "Đã lưu danh sách câu hỏi." if ok else "Lỗi: không thể lưu danh sách câu hỏi."


class GradeAnswersInput(BaseModel):
    raw: str = Field(..., description="Chuỗi đáp án người dùng, ví dụ: '1.B, 2.C, 3.A'")


@tool(args_schema=GradeAnswersInput)
def grade_answers_tool(raw: str) -> Dict[str, Any]:
    """
    Chấm đáp án dựa trên session.context.qna_quiz.questions. Hỗ trợ định dạng '1.B, 2.C, 3.A'.
    Trả về: {results: [{question_id, scope, user_answer, correct_answer, is_correct}], summary:{correct,total}}
    """
    if _QNA_SESSION_ID is None:
        return {"error": "session_id chưa được gắn cho QnA."}
    rows = execute_sql_query("SELECT context FROM session WHERE id = %s;", (_QNA_SESSION_ID,))
    if not rows:
        return {"error": "Không tìm thấy phiên."}
    context = rows[0].get("context") or {}
    questions = ((context.get("qna_quiz") or {}).get("questions") or [])
    if not questions:
        return {"error": "Chưa có bộ câu hỏi nào được lưu trong phiên."}

    import re as _re
    pairs = _re.findall(r"(\d+)\s*\.\s*([A-Za-z])", raw)
    index_to_answer: Dict[int, str] = {int(i): a.upper() for i, a in pairs}
    results: List[Dict[str, Any]] = []
    correct_count = 0
    for idx, q in enumerate(questions, start=1):
        ua = index_to_answer.get(idx)
        ca = (q.get("correct_answer") or "").strip().upper() or None
        is_correct = (ua == ca) if (ua is not None and ca is not None) else None
        if is_correct is True:
            correct_count += 1
        results.append({
            "order": idx,
            "question_id": q.get("question_id"),
            "scope": q.get("scope"),
            "user_answer": ua,
            "correct_answer": ca,
            "is_correct": is_correct,
        })

    summary = {"correct": correct_count, "total": len(questions)}
    # Lưu last_grading vào context
    payload = {"qna_quiz": {"last_grading": {"results": results, "summary": summary}}}
    update_session_context(_QNA_SESSION_ID, payload)
    return {"results": results, "summary": summary}


class GetQuestionByIndexInput(BaseModel):
    index: int = Field(..., description="Số thứ tự câu hỏi (bắt đầu từ 1)")


@tool(args_schema=GetQuestionByIndexInput)
def get_saved_question_by_index_tool(index: int) -> Dict[str, Any]:
    """
    Trả về nội dung câu hỏi đã lưu trong session.context.qna_quiz.questions theo thứ tự (1-based).
    Kết quả gồm: {order, question_id, scope, topic, choices, correct_answer}
    """
    if _QNA_SESSION_ID is None:
        return {"error": "session_id chưa được gắn cho QnA."}
    rows = execute_sql_query("SELECT context FROM session WHERE id = %s;", (_QNA_SESSION_ID,))
    if not rows:
        return {"error": "Không tìm thấy phiên."}
    context = rows[0].get("context") or {}
    questions = ((context.get("qna_quiz") or {}).get("questions") or [])
    if not questions:
        return {"error": "Chưa có bộ câu hỏi nào trong phiên."}
    if index < 1 or index > len(questions):
        return {"error": f"Chỉ số câu không hợp lệ. Tổng số câu: {len(questions)}"}
    q = questions[index - 1]
    return {
        "order": index,
        "question_id": q.get("question_id"),
        "scope": q.get("scope"),
        "topic": q.get("topic"),
        "choices": q.get("choices"),
        "correct_answer": q.get("correct_answer"),
    }


@tool
def get_last_grading_tool() -> Dict[str, Any]:
    """
    Trả về kết quả chấm gần nhất trong session.context.qna_quiz.last_grading nếu có.
    """
    if _QNA_SESSION_ID is None:
        return {"error": "session_id chưa được gắn cho QnA."}
    rows = execute_sql_query("SELECT context FROM session WHERE id = %s;", (_QNA_SESSION_ID,))
    if not rows:
        return {"error": "Không tìm thấy phiên."}
    context = rows[0].get("context") or {}
    last_grading = (context.get("qna_quiz") or {}).get("last_grading")
    if not last_grading:
        return {"error": "Chưa có kết quả chấm gần đây."}
    return {"success": True, "data": last_grading}


# ============================
# Quiz state helpers for QnA
# ============================

class SaveCurrentQuestionInput(BaseModel):
    question_id: str = Field(..., description="ID nội bộ cho câu hỏi (do agent tạo hoặc rút từ tài liệu)")
    scope: Optional[str] = Field(None, description="Phân loại: KANJI | VOCAB | GRAMMAR | LISTENING | READING | OTHER")
    topic: Optional[str] = Field(None, description="Chủ đề ngắn gọn của câu hỏi")
    choices: Optional[List[str]] = Field(None, description="Danh sách đáp án lựa chọn nếu là trắc nghiệm")
    correct_answer: Optional[str] = Field(None, description="Đáp án đúng nếu có (để auto-chấm)")


@tool(args_schema=SaveCurrentQuestionInput)
def save_current_question_tool(question_id: str, scope: Optional[str] = None, topic: Optional[str] = None,
                               choices: Optional[List[str]] = None, correct_answer: Optional[str] = None) -> str:
    """
    Lưu câu hỏi hiện tại vào session context để ràng buộc các câu trả lời tiếp theo.
    Sử dụng ngay sau khi bạn tạo câu hỏi/quiz để lần sau người dùng trả lời, có thể biết câu trả lời thuộc câu nào.
    """
    if _QNA_SESSION_ID is None:
        return "Lỗi: session_id chưa được gắn cho QnA."
    payload: Dict[str, Any] = {
        "qna_quiz": {
            "current_question": {
                "question_id": question_id,
                "scope": scope,
                "topic": topic,
                "choices": choices,
                "correct_answer": correct_answer,
            }
        }
    }
    ok = update_session_context(_QNA_SESSION_ID, payload)
    return "Đã lưu ngữ cảnh câu hỏi hiện tại." if ok else "Lỗi: không thể lưu ngữ cảnh câu hỏi."


class RecordUserAnswerInput(BaseModel):
    answer: str = Field(..., description="Câu trả lời của người dùng (text hoặc nhãn đáp án)")


@tool(args_schema=RecordUserAnswerInput)
def record_user_answer_tool(answer: str) -> Dict[str, Any]:
    """
    Ghi nhận câu trả lời của người dùng cho câu hỏi hiện tại, dựa trên context đã lưu bằng save_current_question_tool.
    Trả về: {question_id, scope, user_answer, is_correct?, correct_answer?}
    """
    if _QNA_SESSION_ID is None:
        return {"error": "session_id chưa được gắn cho QnA."}

    # Lấy lại context hiện tại
    # Không có API đọc trực tiếp context ở đây, nên dùng SQL để lấy context phiên
    rows = execute_sql_query("SELECT context FROM session WHERE id = %s;", (_QNA_SESSION_ID,))
    if not rows:
        return {"error": "Không tìm thấy phiên."}
    context = rows[0].get("context") or {}
    qctx = (context.get("qna_quiz") or {}).get("current_question") or {}
    if not qctx:
        return {"error": "Chưa có câu hỏi hiện tại trong context."}

    question_id = qctx.get("question_id")
    scope = qctx.get("scope")
    correct = qctx.get("correct_answer")
    is_correct = None
    if correct is not None:
        is_correct = str(answer).strip().lower() == str(correct).strip().lower()

    # Lưu lịch sử trả lời gần nhất và xóa current_question để tránh nhầm lẫn lượt sau
    history_entry = {
        "question_id": question_id,
        "scope": scope,
        "user_answer": answer,
        "is_correct": is_correct,
        "correct_answer": correct,
    }
    payload = {
        "qna_quiz": {
            "last_answer": history_entry,
            "current_question": None
        }
    }
    update_session_context(_QNA_SESSION_ID, payload)
    return history_entry


@tool
def clear_current_question_tool() -> str:
    """
    Xóa `current_question` khỏi context nếu đang bị kẹt hoặc tạo câu hỏi mới.
    """
    if _QNA_SESSION_ID is None:
        return "Lỗi: session_id chưa được gắn cho QnA."
    payload = {"qna_quiz": {"current_question": None}}
    ok = update_session_context(_QNA_SESSION_ID, payload)
    return "Đã xóa current_question." if ok else "Lỗi: không thể xóa current_question."