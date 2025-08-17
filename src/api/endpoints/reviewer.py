# src/api/endpoints/reviewer.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from ...core.database import execute_sql_query
import json

router = APIRouter()

class OverviewRequest(BaseModel):
    exam_result_id: str | int = Field(...)

class SectionStat(BaseModel):
    section: str
    total: int
    correct: int
    wrong: int
    accuracy_percent: float

class OverviewResponse(BaseModel):
    exam_result_id: str | int
    advice: Dict[str, Any]

def _get_exam_id(exam_result_id: str | int) -> str | int:
    info = execute_sql_query(
        "SELECT exam_id FROM exam_result WHERE id = %s;",
        (exam_result_id,)
    )
    if not info:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài làm.")
    return info[0]["exam_id"]


def _get_totals(exam_result_id: str | int, exam_id: str | int) -> Dict[str, Any]:
    stats = execute_sql_query(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN erd.selected_option_id IS NOT NULL AND erd.selected_option_id = co.id THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN erd.selected_option_id IS NULL OR erd.selected_option_id <> co.id THEN 1 ELSE 0 END) AS wrong
        FROM exam_question eq
        LEFT JOIN question q ON q.id = eq.question_id
        LEFT JOIN (
            SELECT o.question_id, o.id
            FROM option o
            WHERE o.is_correct = true
        ) co ON co.question_id = eq.question_id
        LEFT JOIN exam_result_detail erd ON erd.exam_result_id = %s AND erd.question_id = eq.question_id
        WHERE eq.exam_id = %s;
        """,
        (exam_result_id, exam_id)
    )
    if not stats:
        raise HTTPException(status_code=404, detail="Không có chi tiết bài làm.")
    return stats[0]


def _get_section_rows(exam_result_id: str | int, exam_id: str | int) -> List[Dict[str, Any]]:
    return execute_sql_query(
        """
        SELECT
            UPPER(COALESCE(q.scope, 'OTHER')) AS section,
            COUNT(*) AS total,
            SUM(CASE WHEN erd.selected_option_id IS NOT NULL AND erd.selected_option_id = co.id THEN 1 ELSE 0 END) AS correct,
            SUM(CASE WHEN erd.selected_option_id IS NULL OR erd.selected_option_id <> co.id THEN 1 ELSE 0 END) AS wrong
        FROM exam_question eq
        LEFT JOIN question q ON q.id = eq.question_id
        LEFT JOIN (
            SELECT o.question_id, o.id
            FROM option o
            WHERE o.is_correct = true
        ) co ON co.question_id = eq.question_id
        LEFT JOIN exam_result_detail erd ON erd.exam_result_id = %s AND erd.question_id = eq.question_id
        WHERE eq.exam_id = %s
        GROUP BY UPPER(COALESCE(q.scope, 'OTHER'))
        ORDER BY section;
        """,
        (exam_result_id, exam_id)
    )


def _build_by_section(section_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_section: List[Dict[str, Any]] = []
    for r in section_rows:
        total = r.get("total") or 0
        correct = r.get("correct") or 0
        wrong = r.get("wrong") or 0
        acc = round((correct / total) * 100, 2) if total else 0.0
        by_section.append(
            SectionStat(
                section=r.get("section"), total=total, correct=correct, wrong=wrong, accuracy_percent=acc
            ).model_dump()
        )
    return by_section


def _build_advice(total_stats: Dict[str, Any], by_section: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = total_stats.get("total") or 0
    correct = total_stats.get("correct") or 0
    wrong = total_stats.get("wrong") or 0
    accuracy = round((correct / total) * 100, 2) if total else 0.0

    strengths = [s["section"] for s in by_section if s["accuracy_percent"] >= 70]
    weaknesses = [s["section"] for s in by_section if s["accuracy_percent"] < 50]

    notes: List[str] = []
    for w in weaknesses:
        if w == "KANJI":
            notes.append("Kanji yếu: ôn bảng Kanji N cấp, tập trung chữ hay nhầm; luyện đọc ghép âm, on-kun.")
        elif w == "VOCAB":
            notes.append("Từ vựng yếu: mở rộng chủ điểm tần suất cao, áp dụng flashcard SRS.")
        elif w == "GRAMMAR":
            notes.append("Ngữ pháp yếu: luyện trợ từ, cấu trúc N cấp; làm đề theo chủ điểm.")
        else:
            notes.append("Cần bổ sung kiến thức phần khác (OTHER).")
    if strengths:
        notes.append(f"Phần làm tốt: {', '.join(strengths)}. Duy trì và luyện đề nâng cao.")
    if not notes:
        notes.append("Tiếp tục duy trì tốc độ học và luyện đề để cải thiện độ ổn định.")

    return {
        "summary": {
            "total_questions": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy_percent": accuracy,
        },
        "by_section": by_section,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "notes": notes,
    }


@router.post("/overview", response_model=OverviewResponse)
async def generate_overview(request: OverviewRequest):
    exam_id = _get_exam_id(request.exam_result_id)
    totals_row = _get_totals(request.exam_result_id, exam_id)
    section_rows = _get_section_rows(request.exam_result_id, exam_id)
    by_section = _build_by_section(section_rows)
    advice = _build_advice(totals_row, by_section)

    # Lưu vào exam_result.advice
    execute_sql_query(
        "UPDATE exam_result SET advice = %s WHERE id = %s RETURNING id;",
        (json.dumps(advice, ensure_ascii=False), request.exam_result_id)
    )

    return OverviewResponse(exam_result_id=request.exam_result_id, advice=advice)
