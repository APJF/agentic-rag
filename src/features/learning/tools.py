# src/features/learning/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from ...core.database import execute_sql_query
from ...utils.storage_client import download_file_from_r2
import os


class MaterialContextInput(BaseModel):
    material_id: int


@tool(args_schema=MaterialContextInput)
def get_material_context(material_id: int) -> Dict[str, Any]:
    """Lấy metadata tài liệu và danh sách câu hỏi/bài tập (id, index, tiêu đề ngắn)."""
    meta = execute_sql_query('SELECT id, title, description, file_key FROM material WHERE id = %s;', (material_id,))
    if not meta:
        return {"error": "Không tìm thấy material."}
    questions = execute_sql_query(
        'SELECT id AS question_id, question_index, short_title FROM material_question WHERE material_id = %s ORDER BY question_index ASC;',
        (material_id,)
    )
    return {"success": True, "material": meta[0], "questions": questions}


class MaterialDownloadInput(BaseModel):
    material_id: int


@tool(args_schema=MaterialDownloadInput)
def fetch_material_file(material_id: int) -> Dict[str, Any]:
    """Tải file material từ MinIO (dựa vào material.file_key) về thư mục tạm và trả đường dẫn local. Hỗ trợ PDF/MD/TXT."""
    row = execute_sql_query('SELECT file_key FROM material WHERE id = %s;', (material_id,))
    if not row or not row[0].get('file_key'):
        return {"error": "Material chưa có file_key."}
    file_key = row[0]['file_key']
    tmp_path = os.path.join("/tmp", os.path.basename(file_key))
    ok = download_file_from_r2(file_key, tmp_path)
    if not ok:
        return {"error": "Không thể tải file từ MinIO."}
    return {"success": True, "local_path": tmp_path}


class MaterialExtractInput(BaseModel):
    local_path: str


@tool(args_schema=MaterialExtractInput)
def extract_text_from_material(local_path: str) -> Dict[str, Any]:
    """Đọc nội dung văn bản từ file PDF/MD/TXT đã tải. Trả về text (cắt ngắn nếu quá dài)."""
    if not os.path.exists(local_path):
        return {"error": "File không tồn tại."}
    text = ""
    if local_path.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(local_path)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n".join(pages)
        except Exception as e:
            return {"error": f"Không đọc được PDF: {e}"}
    elif local_path.lower().endswith((".md", ".txt")):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return {"error": f"Không đọc được file văn bản: {e}"}
    else:
        return {"error": "Định dạng file chưa hỗ trợ (chỉ PDF/MD/TXT)."}

    # Cắt gọn nếu quá dài
    if len(text) > 20000:
        text = text[:20000] + "\n... (đã cắt ngắn)"
    return {"success": True, "text": text}


class MaterialQuestionByIndexInput(BaseModel):
    material_id: int
    index: int = Field(..., description="Số thứ tự câu trong material (1-based)")


@tool(args_schema=MaterialQuestionByIndexInput)
def get_material_question_by_index(material_id: int, index: int) -> Dict[str, Any]:
    """Resolve câu hỏi theo index trong tài liệu, trả question_id và tiêu đề ngắn."""
    rows = execute_sql_query(
        'SELECT id AS question_id, question_index, short_title FROM material_question WHERE material_id = %s AND question_index = %s;',
        (material_id, index)
    )
    if not rows:
        return {"error": "Không tìm thấy câu hỏi theo index."}
    return {"success": True, "question_id": rows[0]["question_id"], "index": rows[0]["question_index"], "short_title": rows[0].get("short_title")}


class ExplainMaterialQuestionInput(BaseModel):
    material_id: int
    question_id: int


@tool(args_schema=ExplainMaterialQuestionInput)
def explain_material_question(material_id: int, question_id: int) -> Dict[str, Any]:
    """Giải thích câu hỏi trong tài liệu dựa trên nội dung/metadata liên quan."""
    q = execute_sql_query(
        'SELECT question, explanation FROM material_question WHERE id = %s AND material_id = %s;',
        (question_id, material_id)
    )
    if not q:
        return {"error": "Không tìm thấy câu hỏi trong tài liệu."}
    return {"success": True, "question": q[0]["question"], "explanation": q[0]["explanation"]}
