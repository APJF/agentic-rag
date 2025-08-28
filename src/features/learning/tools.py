# src/features/learning/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from ...core.database import execute_sql_query, get_db_connection
from src.core.embedding import encode_text
import os
import re


class MaterialContextInput(BaseModel):
    material_id: str


@tool(args_schema=MaterialContextInput)
def get_material_context(material_id: str) -> Dict[str, Any]:
    """
    Lấy tóm tắt ngữ cảnh material từ bảng content_chunks theo material_id (string).
    Trả về tổng số chunks, các loại skill_type hiện có, và một vài chunk đầu để tham chiếu.
    """
    try:
        # Tổng số chunk theo material_id
        cnt_rows = execute_sql_query(
            'SELECT COUNT(*) AS cnt FROM content_chunks WHERE material_id = %s OR (metadata_json::jsonb ->> %s) = %s;',
            (material_id, 'material_id', material_id)
        )
        total = cnt_rows[0]['cnt'] if cnt_rows else 0
        # Các loại skill_type
        skill_rows = execute_sql_query(
            'SELECT DISTINCT skill_type FROM content_chunks WHERE material_id = %s OR (metadata_json::jsonb ->> %s) = %s;',
            (material_id, 'material_id', material_id)
        )
        skills = [r.get('skill_type') for r in skill_rows if r.get('skill_type')]
        # Lấy 3 chunk mẫu theo thứ tự chunk_index (nếu có)
        sample_rows = execute_sql_query(
            '''
            SELECT chunk_text, metadata_json
            FROM content_chunks
            WHERE material_id = %s OR (metadata_json::jsonb ->> %s) = %s
            ORDER BY COALESCE((metadata_json::jsonb ->> 'chunk_index')::int, 1) ASC
            LIMIT 3;
            ''',
            (material_id, 'material_id', material_id)
        )
        return {"success": True, "total_chunks": total, "skills": skills, "samples": sample_rows}
    except Exception as e:
        return {"error": str(e)}


class MaterialChunksInput(BaseModel):
    material_id: str
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)


@tool(args_schema=MaterialChunksInput)
def get_material_chunks(material_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """Lấy các chunks theo material_id từ content_chunks (ưu tiên đúng tài liệu hiện tại)."""
    try:
        rows = execute_sql_query(
            '''
            SELECT chunk_text, skill_type, metadata_json
            FROM content_chunks
            WHERE material_id = %s OR (metadata_json::jsonb ->> %s) = %s
            ORDER BY COALESCE((metadata_json::jsonb ->> 'chunk_index')::int, 1) ASC
            LIMIT %s OFFSET %s;
            ''',
            (material_id, 'material_id', material_id, limit, offset)
        )
        return {"success": True, "chunks": rows}
    except Exception as e:
        return {"error": str(e)}


class MaterialSearchInput(BaseModel):
    material_id: str
    query: str
    top_k: int = Field(5, ge=1, le=20)


@tool(args_schema=MaterialSearchInput)
def search_material_chunks(material_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Tìm các chunks liên quan trong material hiện tại bằng vector similarity nếu có; fallback ILIKE.
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Không kết nối DB."}
    try:
        with conn.cursor() as cur:
            # Ưu tiên dùng ORDER BY embedding <=> query_embedding
            try:
                embedding = encode_text(query)
                embedding_str = str(list(embedding))
                cur.execute(
                    '''
                    SELECT chunk_text, skill_type, metadata_json
                    FROM content_chunks
                    WHERE material_id = %s OR (metadata_json::jsonb ->> %s) = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                    ''',
                    (material_id, 'material_id', material_id, embedding_str, top_k)
                )
                rows = cur.fetchall()
                res = [{"chunk_text": r[0], "skill_type": r[1], "metadata_json": r[2]} for r in rows]
                return {"success": True, "chunks": res}
            except Exception:
                # Fallback: ILIKE search
                like = f"%{query}%"
                cur.execute(
                    '''
                    SELECT chunk_text, skill_type, metadata_json
                    FROM content_chunks
                    WHERE (material_id = %s OR (metadata_json::jsonb ->> %s) = %s)
                      AND chunk_text ILIKE %s
                    ORDER BY LENGTH(chunk_text) ASC
                    LIMIT %s;
                    ''',
                    (material_id, 'material_id', material_id, like, top_k)
                )
                rows = cur.fetchall()
                res = [{"chunk_text": r[0], "skill_type": r[1], "metadata_json": r[2]} for r in rows]
                return {"success": True, "chunks": res}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ListeningScriptInput(BaseModel):
    material_id: str


@tool(args_schema=ListeningScriptInput)
def get_listening_script(material_id: str) -> Dict[str, Any]:
    """
    Lấy transcript (JA) và bản dịch (VI) cho tài liệu nghe theo material_id từ DB.
    Nguồn chính: bảng material (id, file_url, script, translation, type, unit_id).
    Trả về script_ja/script_vi nếu tìm thấy.
    """
    # Query bảng material theo unit_id, id::text hoặc file_url chứa material_id
    try:
        rows = execute_sql_query(
            'SELECT id, unit_id, script, translation, type, file_url FROM material WHERE unit_id = %s OR id::text = %s LIMIT 1;',
            (material_id, material_id)
        ) or []
        if not rows:
            rows = execute_sql_query(
                'SELECT id, unit_id, script, translation, type, file_url FROM material WHERE file_url ILIKE %s LIMIT 1;',
                (f"%{material_id}%",)
            ) or []
        if rows:
            r0 = rows[0]
            return {
                "success": True,
                "script_ja": r0.get('script'),
                "script_vi": r0.get('translation'),
                "type": r0.get('type'),
                "unit_id": r0.get('unit_id'),
                "material_row_id": r0.get('id'),
                "file_url": r0.get('file_url'),
                "source": "material"
            }
    except Exception:
        pass

    return {"error": "Không tìm thấy transcript/bản dịch cho tài liệu nghe trong bảng material."}


class MaterialQuestionByIndexInput(BaseModel):
    material_id: str
    index: int = Field(..., description="Số thứ tự câu trong material (1-based)")


@tool(args_schema=MaterialQuestionByIndexInput)
def get_material_question_by_index(material_id: str, index: int) -> Dict[str, Any]:
    """
    Resolve câu hỏi theo số thứ tự trực tiếp từ content_chunks bằng regex.
    Trả về question_id = index (synthetic), kèm question_text, options, answer (nếu tìm thấy trong tài liệu).
    """
    # Thử dùng bảng material_question trước (nếu có dữ liệu sẵn)
    try:
        rows = execute_sql_query(
            'SELECT id AS question_id, question_index, short_title FROM material_question WHERE material_id = %s AND question_index = %s;',
            (material_id, index)
        )
        if rows:
            return {
                "success": True,
                "question_id": rows[0]["question_id"],
                "index": rows[0]["question_index"],
                "short_title": rows[0].get("short_title")
            }
    except Exception:
        pass

    # RAG-only: tìm trong content_chunks
    chunks = execute_sql_query(
        '''
        SELECT chunk_text, COALESCE((metadata_json::jsonb ->> 'chunk_index')::int, 0) AS idx
        FROM content_chunks
        WHERE material_id = %s OR (metadata_json::jsonb ->> %s) = %s
        ORDER BY idx ASC;
        ''',
        (material_id, 'material_id', material_id)
    )
    if not chunks:
        return {"error": "Không có dữ liệu cho material này."}

    # Tìm block bắt đầu từ dòng 'index.' tới trước (index+1). hoặc đến hết chunk
    qnum = index
    pattern_start = re.compile(rf"^\s*{qnum}\s*[\.、\)]\s*(.*)", re.MULTILINE)
    pattern_next = re.compile(rf"^\s*{qnum + 1}\s*[\.、\)]\s*", re.MULTILINE)
    block_text = None
    start_found = False
    collected = []

    for i, ch in enumerate(chunks):
        text = ch.get("chunk_text") or ""
        m = pattern_start.search(text)
        if m and not start_found:
            start_found = True
            # từ vị trí start tới hết chunk hoặc tới next marker
            start_pos = m.start()
            sub = text[start_pos:]
            m_next = pattern_next.search(sub)
            if m_next:
                collected.append(sub[:m_next.start()])
                break
            else:
                collected.append(sub)
                # nối thêm chunk kế tiếp đến khi gặp next
                for j in range(i + 1, min(i + 3, len(chunks))):
                    t2 = chunks[j].get("chunk_text") or ""
                    m2 = pattern_next.search(t2)
                    if m2:
                        collected.append(t2[:m2.start()])
                        break
                    else:
                        collected.append(t2)
                break
    if start_found:
        block_text = "\n".join(collected).strip()
    else:
        return {"error": "Không tìm thấy câu hỏi theo index trong nội dung tài liệu."}

    # Parse options và đáp án
    # Options kiểu: "A. ..." hoặc "A ..."; Đáp án: "✅ Đáp án: A" hoặc "✅ Đáp án: A. ..."
    option_pat = re.compile(r"^\s*([A-DＡ-Ｄa-dａ-ｄ])\s*[\.\)]?\s*(.+)$", re.MULTILINE)
    answer_pat = re.compile(r"Đáp\s*án\s*:\s*([A-DＡ-Ｄa-dａ-ｄ])", re.IGNORECASE)
    options: Dict[str, str] = {}
    for om in option_pat.finditer(block_text):
        letter = om.group(1).upper()
        text_opt = om.group(2).strip()
        options[letter] = text_opt
    am = answer_pat.search(block_text)
    answer_letter = am.group(1).upper() if am else None
    answer_text = options.get(answer_letter) if answer_letter else None

    # Câu hỏi (nội dung sau số thứ tự trên dòng đầu)
    first_line = block_text.splitlines()[0]
    qm = pattern_start.match(first_line)
    question_text = qm.group(1).strip() if qm else first_line.strip()

    return {
        "success": True,
        "question_id": index,
        "index": index,
        "question_text": question_text,
        "options": options,
        "answer": {"letter": answer_letter, "text": answer_text},
        "raw_block": block_text,
    }


class ExplainMaterialQuestionInput(BaseModel):
    material_id: str
    question_id: int


@tool(args_schema=ExplainMaterialQuestionInput)
def explain_material_question(material_id: str, question_id: int) -> Dict[str, Any]:
    """
    Trả về khối văn bản câu hỏi + đáp án trích từ content_chunks để agent dựa vào đó giải thích.
    """
    # Nếu có bảng material_question với sẵn lời giải, ưu tiên dùng
    try:
        q = execute_sql_query(
            'SELECT question, explanation FROM material_question WHERE id = %s AND material_id = %s;',
            (question_id, material_id)
        )
        if q:
            return {"success": True, "question": q[0]["question"], "explanation": q[0]["explanation"]}
    except Exception:
        pass

    # RAG-only: lấy từ chunks
    res = get_material_question_by_index(material_id=material_id, index=question_id)
    if not res or not res.get("success"):
        return {"error": "Không thể trích xuất câu hỏi/đáp án từ tài liệu."}
    return {
        "success": True,
        "question": res.get("question_text"),
        "options": res.get("options"),
        "answer": res.get("answer"),
        "raw_block": res.get("raw_block"),
    }
