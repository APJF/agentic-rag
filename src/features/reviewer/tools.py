# src/features/reviewer/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

from ...core.database import execute_sql_query


class ExamDataInput(BaseModel):
    exam_result_id: Union[str, int] = Field(description="ID của bài làm (exam_result) cần lấy dữ liệu để chấm.")


@tool(args_schema=ExamDataInput)
def get_exam_submission_details(exam_result_id: Union[str, int]) -> Dict[str, Any]:
    """Lấy toàn bộ thông tin bài làm: exam_id, điểm, thời gian, danh sách câu hỏi, đáp án đúng và lựa chọn của người dùng."""
    exam_info_query = """
        SELECT er.id,
               er.score,
               er.user_id,
               er.exam_id,
               er.started_at,
               er.submitted_at,
               er.status,
               e.title AS exam_title
        FROM exam_result er
        JOIN exam e ON er.exam_id = e.id
        WHERE er.id = %(exam_result_id)s;
    """
    exam_info = execute_sql_query(exam_info_query, {"exam_result_id": exam_result_id})
    if not exam_info:
        return {"error": "Không tìm thấy bài làm."}

    exam_id = exam_info[0]['exam_id']

    questions_query = """
        SELECT
            t.question_id,
            t.question_index,
            t.question_text,
            t.correct_option_id,
            t.correct_option_content,
            t.selected_option_id,
            t.selected_option_content,
            t.is_correct
        FROM (
            SELECT
                eq.question_id,
                ROW_NUMBER() OVER (ORDER BY
                    CASE
                        WHEN eq.question_id ILIKE %(kv)s THEN 1
                        WHEN eq.question_id ILIKE %(g)s THEN 2
                        ELSE 99
                    END,
                    COALESCE(NULLIF(substring(eq.question_id from '.*-(\\d+)$'), '')::int, 0)
                ) AS question_index,
                q.content AS question_text,
                co.id AS correct_option_id,
                co.content AS correct_option_content,
                erd.selected_option_id,
                uo.content AS selected_option_content,
                (uo.id IS NOT NULL AND uo.id = co.id) AS is_correct
            FROM exam_question eq
            LEFT JOIN question q ON q.id = eq.question_id
            LEFT JOIN (
                SELECT o.question_id, o.id, o.content
                FROM option o
                WHERE o.is_correct = true
            ) co ON co.question_id = eq.question_id
            LEFT JOIN exam_result_detail erd ON erd.exam_result_id = %(exam_result_id)s AND erd.question_id = eq.question_id
            LEFT JOIN option uo ON uo.id = erd.selected_option_id
            WHERE eq.exam_id = %(exam_id)s
        ) t
        ORDER BY t.question_index ASC;
    """
    questions = execute_sql_query(questions_query, {
        "exam_result_id": exam_result_id,
        "exam_id": exam_id,
        "kv": "%-KV-%",
        "g": "%-G-%",
    })

    submission_details = exam_info[0]
    submission_details['questions'] = questions
    return submission_details


class QuestionByIndexInput(BaseModel):
    exam_id: Union[str, int] = Field(...)
    index: int = Field(..., description="Số thứ tự câu (1-based)")


@tool(args_schema=QuestionByIndexInput)
def get_question_by_index(exam_id: Union[str, int], index: int) -> Dict[str, Any]:
    """Resolve câu hỏi theo thứ tự (1-based) trong một đề thi (KV trước G), trả về question_id và index."""
    rows = execute_sql_query(
        """
        SELECT question_id, question_index FROM (
            SELECT
                eq.question_id,
                ROW_NUMBER() OVER (ORDER BY
                    CASE
                        WHEN eq.question_id ILIKE %(kv)s THEN 1
                        WHEN eq.question_id ILIKE %(g)s THEN 2
                        ELSE 99
                    END,
                    COALESCE(NULLIF(substring(eq.question_id from '.*-(\\d+)$'), '')::int, 0)
                ) AS question_index
            FROM exam_question eq
            WHERE eq.exam_id = %(exam_id)s
        ) t
        WHERE t.question_index = %(index)s;
        """,
        {"exam_id": exam_id, "index": index, "kv": "%-KV-%", "g": "%-G-%"}
    )
    if not rows:
        return {"error": "Không tìm thấy câu theo index."}
    return {"success": True, "question_id": rows[0]["question_id"], "index": rows[0]["question_index"]}


class UserAnswersInput(BaseModel):
    exam_result_id: Union[str, int] = Field(...)


@tool(args_schema=UserAnswersInput)
def get_user_answers(exam_result_id: Union[str, int]) -> Dict[str, Any]:
    """Trả về mapping question_id -> selected_option_id và is_correct cho một lần làm bài."""
    rows = execute_sql_query(
        'SELECT question_id, selected_option_id, is_correct FROM exam_result_detail WHERE exam_result_id = %(exam_result_id)s;',
        {"exam_result_id": exam_result_id}
    )
    return {
        "success": True,
        "answers": {r["question_id"]: {
            "selected_option_id": r["selected_option_id"],
            "is_correct": r["is_correct"]
        } for r in rows}
    }


class ExplainQuestionInput(BaseModel):
    exam_id: Union[str, int]
    question_id: Union[str, int]
    selected_option_id: Optional[Union[str, int]] = None


@tool(args_schema=ExplainQuestionInput)
def explain_question(exam_id: Union[str, int], question_id: Union[str, int], selected_option_id: Optional[Union[str, int]] = None) -> Dict[str, Any]:
    """Trình bày câu hỏi, đáp án đúng (từ option.is_correct) và nếu có, so sánh với lựa chọn của người dùng (selected_option_id)."""
    q = execute_sql_query(
        'SELECT content AS question_text FROM question WHERE id = %(question_id)s;',
        {"question_id": question_id}
    )
    question_text = q[0]["question_text"] if q else None

    correct = execute_sql_query(
        'SELECT id, content FROM option WHERE question_id = %(question_id)s AND is_correct = true LIMIT 1;',
        {"question_id": question_id}
    )
    correct_id = correct[0]["id"] if correct else None
    correct_content = correct[0]["content"] if correct else None

    user_content = None
    is_correct = None
    if selected_option_id:
        uo = execute_sql_query('SELECT id, content FROM option WHERE id = %(id)s;', {"id": selected_option_id})
        if uo:
            user_content = uo[0]["content"]
            is_correct = (uo[0]["id"] == correct_id)

    return {
        "success": True,
        "question": question_text,
        "correct_option": {"id": correct_id, "content": correct_content},
        "user_option": ({"id": selected_option_id, "content": user_content} if selected_option_id else None),
        "is_correct": is_correct
    }
