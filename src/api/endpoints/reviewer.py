# src/api/endpoints/reviewer.py

from fastapi import APIRouter, Body
from ..schemas import ExamGradeRequest, ExamGradeResponse

# Import các thành phần cần thiết
from ...features.reviewer.agent import initialize_reviewer_agent

# Khởi tạo ReviewerAgent một lần duy nhất
reviewer_agent_executor = initialize_reviewer_agent()

router = APIRouter()


@router.post("/grade", response_model=ExamGradeResponse)
async def grade_exam_submission(request: ExamGradeRequest = Body(...)):
    """
    Endpoint chuyên dụng để kích hoạt Agent chấm bài, chữa lỗi và đưa ra nhận xét.
    Đây là một tác vụ chạy một lần, không có tính hội thoại.
    """
    exam_result_id = request.exam_result_id

    print(f"--- Reviewer API nhận yêu cầu chấm bài cho exam_result_id: {exam_result_id} ---")

    # Dữ liệu đầu vào cho agent chỉ cần ID của bài làm
    input_data = {
        "context": {"exam_result_id": exam_result_id}
    }

    # Gọi agent để thực hiện việc chấm bài và tạo nhận xét
    result = reviewer_agent_executor.invoke(input_data)
    ai_feedback_text = result.get('output', "Lỗi: ReviewerAgent không có output.")

    # (Giả định rằng sau khi chạy, agent đã cập nhật điểm và feedback vào database)
    # TODO: Bạn có thể thêm logic để lấy lại điểm số đã được cập nhật từ DB ở đây

    return ExamGradeResponse(
        exam_result_id=exam_result_id,
        overall_score=85.5,  # Tạm thời hardcode, sẽ lấy từ DB
        ai_feedback=ai_feedback_text
    )
