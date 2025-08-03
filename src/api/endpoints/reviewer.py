# src/api/endpoints/reviewer.py

from fastapi import APIRouter, Body, HTTPException
from ..schemas import ExamGradeRequest, ExamGradeResponse, ExamResultDetailResponse, ExamAdviceRequest, ExamAdviceResponse, ExamReviewChatRequest, ChatResponse, EssayGradeRequest, EssayGradeResponse, EssayReviewChatRequest
from ...features.reviewer.agent import initialize_reviewer_agent
from ...core.session_manager import create_new_session, add_new_messages
from ...core.database import execute_sql_query
from langchain_core.messages import HumanMessage, AIMessage
import json

reviewer_agent_executor = initialize_reviewer_agent()
router = APIRouter()

@router.post("/grade", response_model=ExamGradeResponse)
async def grade_exam_submission(request: ExamGradeRequest = Body(...)):
    user_id = request.user_id
    exam_id = request.exam_id
    answers = request.answers  # {question_id: user_answer}

    # 1. Tạo exam_result mới
    exam_result_id = execute_sql_query(
        "INSERT INTO exam_result (user_id, exam_id, score, advice, status) VALUES (%s, %s, 0, '', 'SUBMITTED') RETURNING id",
        (user_id, exam_id)
    )[0]['id']
    score = 0
    advice_parts = []
    details = []
    for qid, user_answer in answers.items():
        correct = execute_sql_query("SELECT correct_answer FROM question WHERE id = %s", (qid,))[0]['correct_answer']
        is_correct = (user_answer == correct)
        if not is_correct:
            advice_parts.append(f"Câu {qid}: Bạn nên xem lại phần này.")
        # Lưu vào exam_result_detail
        execute_sql_query(
            "INSERT INTO exam_result_detail (exam_result_id, question_id, user_answer, is_correct) VALUES (%s, %s, %s, %s)",
            (exam_result_id, qid, user_answer, is_correct)
        )
        details.append({"question_id": qid, "user_answer": user_answer, "is_correct": is_correct, "correct_answer": correct})
        if is_correct:
            score += 1

    # 4. Sinh advice tổng thể bằng AI
    advice = reviewer_agent_executor.invoke({
        "context": {"exam_result_id": exam_result_id, "score": score, "advice_parts": advice_parts}
    }).get('output', "Hãy xem lại các câu sai và ôn tập thêm.")

    # 5. Update exam_result với score và advice
    execute_sql_query(
        "UPDATE exam_result SET score = %s, advice = %s WHERE id = %s",
        (score, advice, exam_result_id)
    )

    # 6. Tạo session chat chữa bài
    session_id = create_new_session(user_id, f"Chữa bài {exam_id}", session_type="EXAM_REVIEW", context={"exam_result_id": exam_result_id})
    # 7. Lưu lịch sử hỏi đáp (nếu có)
    add_new_messages(session_id, [
        HumanMessage(content="Tôi muốn nhận xét về bài làm này."),
        AIMessage(content=advice)
    ])

    return ExamGradeResponse(
        exam_result_id=exam_result_id,
        score=score,
        advice=advice,
        session_id=session_id
    )

@router.post("/advice", response_model=ExamAdviceResponse)
async def get_exam_advice(request: ExamAdviceRequest = Body(...)):
    context = request.dict()
    advice_json = reviewer_agent_executor.invoke({"context": context}).get('output', {})
    execute_sql_query(
        "UPDATE exam_result SET advice = %s WHERE id = %s",
        (json.dumps(advice_json), request.exam_result_id)
    )
    return ExamAdviceResponse(advice=advice_json)

@router.post("/essay/grade", response_model=EssayGradeResponse)
async def grade_essay(request: EssayGradeRequest = Body(...)):
    context = request.dict()
    result = reviewer_agent_executor.invoke({"context": context}).get('output', {})
    # Lưu result vào bảng essay_result nếu muốn
    # execute_sql_query("INSERT INTO essay_result ...", (...))
    return EssayGradeResponse(**result)

@router.post("/essay/chat", response_model=ChatResponse)
async def chat_essay_review(request: EssayReviewChatRequest = Body(...)):
    # Tìm hoặc tạo session chat cho bài tự luận này
    session = execute_sql_query(
        "SELECT id FROM chat_session WHERE context->>'essay_result_id' = %s", (request.essay_result_id,)
    )
    if not session:
        session_id = create_new_session(request.user_id, f"Chữa bài tự luận {request.essay_result_id}", session_type="ESSAY_REVIEW", context={"essay_result_id": request.essay_result_id})
    else:
        session_id = session[0]['id']
    # Truyền đầy đủ ngữ cảnh bài làm, điểm từng tiêu chí, advice, nội dung bài văn, v.v. vào agent
    context = request.dict()
    ai_response = reviewer_agent_executor.invoke({"context": context}).get('output', "Xin hãy cung cấp thêm thông tin.")
    add_new_messages(session_id, [
        HumanMessage(content=request.user_input),
        AIMessage(content=ai_response)
    ])
    return ChatResponse(session_id=session_id, ai_response=ai_response)

@router.get("/result/{exam_result_id}", response_model=ExamResultDetailResponse)
async def get_exam_result_detail(exam_result_id: str):
    exam_result = execute_sql_query("SELECT * FROM exam_result WHERE id = %s", (exam_result_id,))[0]
    details = execute_sql_query("SELECT * FROM exam_result_detail WHERE exam_result_id = %s", (exam_result_id,))
    session = execute_sql_query("SELECT id FROM chat_session WHERE context->>'exam_result_id' = %s", (exam_result_id,))
    chat_history = []
    if session:
        chat_history = execute_sql_query("SELECT * FROM chat_messenger WHERE session_id = %s ORDER BY messenger_order", (session[0]['id'],))
    return ExamResultDetailResponse(
        score=exam_result['score'],
        advice=exam_result['advice'],
        details=details,
        chat_history=chat_history
    )
