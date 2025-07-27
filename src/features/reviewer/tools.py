# src/features/reviewer/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from ...core.database import execute_sql_query


class ExamDataInput(BaseModel):
    exam_result_id: int = Field(description="ID của bài làm (exam_result) cần lấy dữ liệu để chấm.")


@tool(args_schema=ExamDataInput)
def get_exam_submission_details(exam_result_id: int) -> Dict[str, Any]:
    """
    Lấy toàn bộ thông tin chi tiết của một lần nộp bài, bao gồm đề bài,
    đáp án đúng, và bài làm của người dùng.
    """
    print(f"--- Tool: Đang lấy chi tiết bài làm ID: {exam_result_id} ---")

    # Lấy thông tin chung về bài làm và đề thi
    exam_info_query = """
                      SELECT ER.id, \
                             ER.score, \
                             ER.user_id, \
                             E.name AS exam_name, \
                             E.id   AS exam_id
                      FROM "exam_result" ER
                               JOIN "Exam" E ON ER.exam_id = E.id
                      WHERE ER.id = %s; \
                      """
    exam_info = execute_sql_query(exam_info_query, (exam_result_id,))
    if not exam_info:
        return {"error": "Không tìm thấy bài làm."}

    # Lấy toàn bộ câu hỏi của đề thi và câu trả lời của người dùng
    questions_and_answers_query = """
                                  SELECT Q.id       AS question_id, \
                                         Q.question AS question_text, \
                                         Q.answer   AS correct_answer, \
                                         Q.explain  AS explanation, \
                                         AA.user_answer
                                  FROM "Question_Exam" QE
                                           JOIN "Question" Q ON QE.question_id = Q.id
                                           LEFT JOIN "AttemptAnswers" AA ON Q.id = AA.question_id AND AA.attempt_id = %s
                                  WHERE QE.exam_id = %s; \
                                  """
    # Cần exam_id từ kết quả truy vấn trước
    exam_id = exam_info[0]['exam_id']
    questions_data = execute_sql_query(questions_and_answers_query, (exam_result_id, exam_id))

    submission_details = exam_info[0]
    submission_details['questions'] = questions_data

    return submission_details
