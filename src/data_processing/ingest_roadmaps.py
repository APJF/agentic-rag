import os
import json
import argparse
import re
from typing import Dict, Optional

from src.core.embedding import encode_text
from src.core.vector_store_interface import get_db_connection
from src.config import settings


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def _resolve_default_path(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path
    env_path = os.getenv("RAG_ROADMAP_FILE")
    if env_path:
        return env_path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", "roadmaps", "roadmap_templates.jsonl")


def _build_chunk_text(doc: Dict) -> str:
    title = doc.get("title") or "Lộ trình học"
    start_level = doc.get("start_level") or "Unknown"
    target_level = doc.get("target_level") or doc.get("end_level") or "Unknown"
    primary_goal = doc.get("primary_goal") or ""
    focus_skills = ", ".join(doc.get("focus_skills") or [])
    courses = ", ".join(doc.get("course_sequence") or [])
    notes = doc.get("notes") or ""
    return _normalize_text(
        f"{title}. Xuất phát: {start_level} -> Mục tiêu: {target_level}. "
        f"Mục tiêu chính: {primary_goal}. Trọng tâm: {focus_skills}. "
        f"Khoá học: {courses}. Ghi chú: {notes}."
    )


def ingest_roadmap_templates(
    jsonl_path: Optional[str] = None,
    table_name: str = settings.RAG_CONTENT_CHUNK_TABLE,
) -> int:
    jsonl_path = _resolve_default_path(jsonl_path)
    if not os.path.isfile(jsonl_path):
        print(f"[Roadmap Ingest] Không tìm thấy file: {jsonl_path}")
        return 0

    conn = get_db_connection()
    if not conn:
        print("[Roadmap Ingest] Không thể kết nối DB.")
        return 0

    inserted = 0
    try:
        with conn.cursor() as cur, open(jsonl_path, "r", encoding="utf-8") as f:
            page_counter = 0
            buffer = ""
            brace_depth = 0
            for raw_line in f:
                s = (raw_line or "").strip()
                if not s:
                    continue

                # Thu gom JSON object nhiều dòng (dựa trên số lượng ngoặc nhọn)
                if not buffer:
                    # Nếu dòng là một JSON object hoàn chỉnh
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            doc = json.loads(s)
                        except json.JSONDecodeError:
                            print("[Roadmap Ingest] Bỏ qua dòng JSON không hợp lệ.")
                            continue
                    else:
                        # Nếu bắt đầu object nhưng chưa kết thúc
                        if s.startswith("{"):
                            buffer = s
                            brace_depth = s.count("{") - s.count("}")
                            continue
                        # Bất kỳ dòng không phải JSON object, bỏ qua an toàn
                        try:
                            tmp = json.loads(s)
                            if not isinstance(tmp, dict):
                                # JSON hợp lệ nhưng không phải object (vd: chuỗi) → bỏ qua
                                print("[Roadmap Ingest] Bỏ qua JSON không phải object.")
                                continue
                            doc = tmp
                        except json.JSONDecodeError:
                            print("[Roadmap Ingest] Bỏ qua dòng JSON không hợp lệ.")
                            continue
                else:
                    buffer += " " + s
                    brace_depth += s.count("{") - s.count("}")
                    if brace_depth > 0:
                        continue
                    try:
                        doc = json.loads(buffer)
                    except json.JSONDecodeError:
                        print("[Roadmap Ingest] Bỏ qua block JSON không hợp lệ.")
                        buffer = ""
                        brace_depth = 0
                        continue
                    buffer = ""
                    brace_depth = 0

                if not isinstance(doc, dict):
                    print("[Roadmap Ingest] Bỏ qua JSON không phải object.")
                    continue

                chunk_text = _build_chunk_text(doc)
                source_name = os.path.basename(jsonl_path)
                page_counter += 1

                # Dedup check: theo text chuẩn hoá + source + page
                cur.execute(
                    f"SELECT 1 FROM {table_name} WHERE source_document_name=%s AND original_page_number=%s AND chunk_text=%s LIMIT 1;",
                    (source_name, page_counter, chunk_text),
                )
                if cur.fetchone():
                    continue

                # Xác định metadata
                metadata: Dict = {
                    "doc_type": "ROADMAP_TEMPLATE",
                    "source_path": jsonl_path.replace("\\", "/"),
                    "raw": doc,
                }

                # Level & skill
                level = doc.get("target_level") or doc.get("end_level")
                skill = None
                if isinstance(doc.get("focus_skills"), list) and doc["focus_skills"]:
                    skill = ",".join(doc["focus_skills"])[:50]

                # Embedding
                vector = encode_text(chunk_text)
                vector_str = str(list(vector))

                # Lấy danh sách cột để build INSERT động
                cur.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = %s;",
                    (table_name,),
                )
                available_cols = {r[0] for r in cur.fetchall()}

                metadata_col = None
                for candidate in ("metadata_json", "metadata", "meta_data"):
                    if candidate in available_cols:
                        metadata_col = candidate
                        break

                columns = ["chunk_text", "source_document_name", "original_page_number"]
                values = [chunk_text, source_name, page_counter]
                if "level" in available_cols:
                    columns.append("level")
                    values.append(level)
                else:
                    metadata["level"] = level
                if "skill_type" in available_cols:
                    columns.append("skill_type")
                    values.append(skill)
                else:
                    metadata["skill_type"] = skill
                if metadata_col:
                    columns.append(metadata_col)
                    values.append(json.dumps(metadata, ensure_ascii=False))
                columns.append("embedding")
                values.append(vector_str)

                cols_sql = ", ".join(columns)
                placeholders = ", ".join(["%s"] * len(values))
                cur.execute(f"INSERT INTO {table_name} ({cols_sql}) VALUES ({placeholders});", tuple(values))
                inserted += 1

            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[Roadmap Ingest] Lỗi khi ingest: {e}")
    finally:
        if conn:
            conn.close()

    print(f"[Roadmap Ingest] Đã thêm {inserted} roadmap templates từ {jsonl_path} vào {table_name}")
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest roadmap templates JSONL vào bảng content_chunks")
    parser.add_argument("--file", dest="jsonl_path", default=None, help="Đường dẫn file JSONL. Mặc định: data/roadmaps/roadmap_templates.jsonl")
    parser.add_argument("--table", dest="table_name", default=settings.RAG_CONTENT_CHUNK_TABLE, help="Tên bảng chunk (mặc định lấy từ settings)")
    args = parser.parse_args()
    ingest_roadmap_templates(jsonl_path=args.jsonl_path, table_name=args.table_name)


