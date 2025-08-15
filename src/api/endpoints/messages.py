from fastapi import APIRouter, Body, HTTPException, Path, status, Response
import re
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage

from ...core.session_manager import load_session_data, update_session_context
from ...core.database import get_db_connection
from ...features.qna.agent import initialize_qna_agent
from ...features.qna.tools import set_qna_session_id
from ...features.planner.agent import initialize_planning_agent
from ...features.learning.agent import initialize_learning_agent
from ...features.reviewer.agent import initialize_reviewer_agent
from ...features.speaking.agent import initialize_speaking_agent
from ...features.planner.tools import set_session_user_id

router = APIRouter()

class MessageCreateRequest(BaseModel):
    session_id: int = Field(...)
    user_input: str = Field(...)

class MessageCreateResponse(BaseModel):
    session_id: int
    human_message_id: int
    ai_message_id: int
    ai_response: str


def _insert_message(session_id: int, messenger_type: str, content: str) -> Optional[int]:
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(\"order\"), 0) + 1 FROM message WHERE session_id = %s;", (session_id,))
            next_order = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO message (session_id, type, content, "order")
                VALUES (%s, %s, %s, %s) RETURNING id;
                """,
                (session_id, messenger_type, content, next_order)
            )
            message_id = cur.fetchone()[0]
            cur.execute("UPDATE session SET updated_at = NOW() WHERE id = %s;", (session_id,))
            conn.commit()
            return message_id
    except Exception:
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()


@router.post("/", response_model=MessageCreateResponse)
async def create_message(request: MessageCreateRequest = Body(...)):
    session_data = load_session_data(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Phiên {request.session_id} không tồn tại.")

    human_id = _insert_message(request.session_id, "human", request.user_input)
    if human_id is None:
        raise HTTPException(status_code=500, detail="Không thể lưu tin nhắn người dùng.")

    # Nếu người dùng trả lời ngắn gọn 'có/không', set context để agent hiểu nhánh cần thực hiện
    normalized = request.user_input.strip().lower()
    flag_value: Optional[str] = None
    if normalized in {"có", "co", "yes", "y", "ok", "đúng", "dung"}:
        flag_value = "yes"
        update_session_context(request.session_id, {"confirm_previous_question": flag_value})
    elif normalized in {"không", "khong", "no", "n", "không cần", "khong can"}:
        flag_value = "no"
        update_session_context(request.session_id, {"confirm_previous_question": flag_value})

    # Nhận diện người dùng đã làm xong bài test và muốn tiếp tục
    exam_completed_flag = False
    if normalized in {
        "tiếp tục", "tiep tuc", "đã hoàn thành", "da hoan thanh", "hoàn thành", "hoan thanh",
        "đã xong", "da xong", "xong", "done", "hoàn tất", "hoan tat", "đã làm xong", "da lam xong"
    }:
        exam_completed_flag = True
        update_session_context(request.session_id, {"exam_completed": "yes"})

    session_type = (session_data.get("type") or "qna").lower()
    agent_map = {
        "qna": initialize_qna_agent(),
        "planner": initialize_planning_agent(),
        "learning": initialize_learning_agent(),
        "reviewer": initialize_reviewer_agent(),
        "speaking": initialize_speaking_agent(),
    }
    agent = agent_map.get(session_type)
    if not agent:
        raise HTTPException(status_code=500, detail="Agent không khả dụng.")

    # Reload session_data to get updated context
    session_data = load_session_data(request.session_id) or session_data

    input_data = {
        "user_id": session_data["user_id"],
        "input": request.user_input,
        "chat_history": session_data["history"] + [HumanMessage(content=request.user_input)],
        "context": session_data.get("context", {})
    }
    if session_type == "planner":
        set_session_user_id(session_data["user_id"])  # đảm bảo tool dùng đúng user_id
    if session_type == "qna":
        try:
            set_qna_session_id(int(request.session_id))
        except Exception:
            pass

    result = agent.invoke(input_data)
    ai_response_text = result.get("output", "Lỗi: Agent không có output.")
    # Sanitize: chỉ trả Final Answer cho frontend, ẩn Thought/Action nếu lỡ in ra
    try:
        # Nếu có "Final Answer:" thì chỉ lấy phần sau đó
        m = re.search(r"Final Answer:\s*(.*)", ai_response_text, re.DOTALL | re.IGNORECASE)
        if m:
            ai_response_text = m.group(1).strip()
        else:
            # Loại bỏ các dòng bắt đầu bằng "Thought:" hoặc "Action:"
            lines = []
            for line in ai_response_text.splitlines():
                if re.match(r"\s*(Thought|Action)\s*:\s*", line, flags=re.IGNORECASE):
                    continue
                lines.append(line)
            ai_response_text = "\n".join(lines).strip()
    except Exception:
        pass

    # Nếu agent vừa tạo một bộ câu hỏi, tự động lưu vào session.context để lần sau chấm đáp án được
    def _extract_quiz_from_text(text: str) -> List[Dict[str, Any]]:
        try:
            lines = [ln.rstrip() for ln in text.splitlines()]
            questions: List[Dict[str, Any]] = []
            current: Dict[str, Any] = {}
            for ln in lines:
                # Match question like: "1. ..." (tiếp theo có thể là văn bản)
                m_q = re.match(r"^\s*(\d+)\s*\.(.*)$", ln)
                if m_q:
                    # push previous
                    if current:
                        questions.append(current)
                        current = {}
                    idx = int(m_q.group(1))
                    qtext = m_q.group(2).strip()
                    current = {
                        "question_id": f"q{idx}",
                        "topic": qtext,
                        "scope": None,
                        "choices": []
                    }
                    # Heuristic: detect scope by header keywords
                    if re.search(r"kanji", text, re.IGNORECASE):
                        current["scope"] = "KANJI"
                    elif re.search(r"vocab|từ vựng", text, re.IGNORECASE):
                        current["scope"] = "VOCAB"
                    elif re.search(r"ngữ pháp|grammar", text, re.IGNORECASE):
                        current["scope"] = "GRAMMAR"
                    continue
                # Match choice like: "A. ..."
                m_c = re.match(r"^\s*([A-Da-d])\s*\.(.*)$", ln)
                if m_c and current:
                    opt = m_c.group(1).upper()
                    ctext = m_c.group(2).strip()
                    current.setdefault("choices", []).append(f"{opt}. {ctext}")
            if current:
                questions.append(current)
            # lọc bỏ entries thiếu topic/choices
            questions = [q for q in questions if q.get("topic")]
            return questions
        except Exception:
            return []

    detected_questions = _extract_quiz_from_text(ai_response_text)
    if detected_questions:
        try:
            update_session_context(request.session_id, {"qna_quiz": {"questions": detected_questions, "current_index": 1}})
        except Exception:
            pass

    # Clear flag để tránh xử lý lặp lại ở lượt sau
    if flag_value is not None:
        update_session_context(request.session_id, {"confirm_previous_question": ""})

    # Bắt examId từ link AI đã gửi để lưu vào context, phục vụ lần sau tự check kết quả
    try:
        # dạng link: localhost:5173/exam/<examId>/preparation
        m = re.search(r"exam/([^/]+)/preparation", ai_response_text)
        if m:
            exam_id_captured = m.group(1)
            update_session_context(request.session_id, {"suggested_exam_id": exam_id_captured})
    except Exception:
        pass

    # Nếu trước đó set exam_completed, sau khi đã phản hồi thì xóa cờ để tránh lặp
    if exam_completed_flag:
        update_session_context(request.session_id, {"exam_completed": ""})

    ai_id = _insert_message(request.session_id, "ai", ai_response_text)
    if ai_id is None:
        raise HTTPException(status_code=500, detail="Không thể lưu phản hồi AI.")

    return MessageCreateResponse(
        session_id=request.session_id,
        human_message_id=human_id,
        ai_message_id=ai_id,
        ai_response=ai_response_text
    )


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(message_id: int = Path(...)):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Không thể kết nối DB.")
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM message WHERE id = %s RETURNING session_id;", (message_id,))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                raise HTTPException(status_code=404, detail="Không tìm thấy message để xoá.")
            session_id = row[0]
            cur.execute(
                """
                UPDATE session SET updated_at = (
                    SELECT MAX(timestamp) FROM message WHERE session_id = %s
                ) WHERE id = %s;
                """,
                (session_id, session_id)
            )
            conn.commit()
    finally:
        if conn:
            conn.close()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
