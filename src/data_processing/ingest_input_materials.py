import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from psycopg2.extras import execute_values

from src.core.vector_store_interface import get_db_connection
from src.core.embedding import get_embedding_model
from src.config import settings


SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "txt", "md"}


def parse_material_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse tên file theo quy ước:
    {COURSE}__{CHAPTER}__{UNIT}__{TYPE}__{LANG}__{SEQ}.{EXT}
    Trả None nếu không khớp.
    """
    name = os.path.basename(filename)
    match = re.match(
        r"^(?P<course>JPD\d{3})__(?P<chapter>CHAPTER_\d{2})__(?P<unit>UNIT_\d{2})__(?P<type>[A-Z_]+)__(?P<lang>[A-Z]{2}(?:_[A-Z]{2})?)__(?P<seq>\d{4})\.(?P<ext>[A-Za-z0-9]+)$",
        name,
    )
    if not match:
        return None
    parts = match.groupdict()
    parts["material_id"] = name.rsplit(".", 1)[0]
    parts["ext"] = parts["ext"].lower()
    return parts


def derive_level_from_course(course_id: str) -> Optional[str]:
    """Suy ra level từ mã course (heuristic): 1xx->N5, 2xx->N4, 3xx->N3, 4xx->N2, 5xx->N1"""
    try:
        num = int(re.sub(r"^JPD", "", course_id))
        hundreds = num // 100
        mapping = {1: "N5", 2: "N4", 3: "N3", 4: "N2", 5: "N1"}
        return mapping.get(hundreds)
    except Exception:
        return None


def extract_text_from_file(file_path: str, ext: str) -> str:
    ext_l = ext.lower()
    if ext_l == "txt" or ext_l == "md":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    if ext_l == "docx":
        try:
            from docx import Document
            doc = Document(file_path)
            texts: List[str] = []
            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    texts.append(t)
            # tables (naively)
            for table in getattr(doc, "tables", []) or []:
                for row in table.rows:
                    row_text = " ".join([cell.text.strip() for cell in row.cells if cell.text])
                    if row_text:
                        texts.append(row_text)
            return "\n".join(texts)
        except Exception:
            return ""
    if ext_l == "pptx":
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            texts: List[str] = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text_frame") and shape.text_frame:
                        for p in shape.text_frame.paragraphs:
                            t = "".join([run.text for run in p.runs]).strip()
                            if t:
                                texts.append(t)
                    elif hasattr(shape, "text"):
                        t = str(getattr(shape, "text", "")).strip()
                        if t:
                            texts.append(t)
            return "\n".join(texts)
        except Exception:
            return ""
    if ext_l == "pdf":
        try:
            import pdfplumber
            texts: List[str] = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    if t.strip():
                        texts.append(t)
            return "\n\n".join(texts)
        except Exception:
            return ""
    return ""


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=False,
    )
    return [c.strip() for c in splitter.split_text(text) if c and len(c.strip()) >= 30]


def collect_material_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(dirpath, fn))
    return files


def upsert_material_chunks(record_dicts: List[Dict[str, Any]], material_id: str, conn) -> None:
    """
    Xóa chunks cũ theo material_id (ưu tiên theo metadata_json->>'material_id'), sau đó chèn mới.
    records: list of tuples phù hợp câu INSERT ở dưới.
    """
    table_name = settings.RAG_CONTENT_CHUNK_TABLE
    with conn.cursor() as cur:
        # Thử xóa theo cột material_id nếu có; nếu không, xóa theo metadata_json
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name=%s AND table_schema='public';
            """,
            (table_name,)
        )
        cols = {r[0] for r in cur.fetchall()}
        try:
            if "material_id" in cols:
                cur.execute(f"DELETE FROM {table_name} WHERE material_id = %s;", (material_id,))
            elif "metadata_json" in cols:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE metadata_json::jsonb ->> 'material_id' = %s;",
                    (material_id,),
                )
        except Exception:
            # Bỏ qua lỗi xóa để không chặn việc chèn mới
            pass

        # Xác định danh sách cột chèn dựa trên schema thực tế
        base_cols = [
            "chunk_text",
            "embedding",
            "course_id",
            "source_document_name",
            "original_page_number",
        ]
        # 'level' nếu có
        if "level" in cols:
            base_cols.append("level")
        # 'skill_type' nếu có
        if "skill_type" in cols:
            base_cols.append("skill_type")
        # 'metadata_json' nếu có
        if "metadata_json" in cols:
            base_cols.append("metadata_json")
        # 'material_id' nếu có
        if "material_id" in cols:
            base_cols.append("material_id")

        # Build values tương ứng
        values: List[Tuple] = []
        for d in record_dicts:
            row: List[Any] = [
                d.get("chunk_text"),
                d.get("embedding"),
                d.get("course_id"),
                d.get("source_document_name"),
                d.get("original_page_number"),
            ]
            if "level" in cols:
                row.append(d.get("level"))
            if "skill_type" in cols:
                row.append(d.get("skill_type"))
            if "metadata_json" in cols:
                row.append(d.get("metadata_json"))
            if "material_id" in cols:
                row.append(d.get("material_id"))
            values.append(tuple(row))

        insert_query = f"INSERT INTO {table_name} ({', '.join(base_cols)}) VALUES %s;"
        execute_values(cur, insert_query, values)
        conn.commit()


def main():
    # Xác định project_root bất kể current working directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="Ingest data/input_materials vào bảng content_chunks, xóa trùng theo material_id")
    parser.add_argument("--root", default=os.path.join(project_root, "data", "input_materials"), help="Thư mục gốc input_materials")
    args = parser.parse_args()

    # Chuẩn hóa đường dẫn root: nếu là đường dẫn tương đối, ghép với project_root
    root_dir = args.root
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(project_root, root_dir)
    root_dir = os.path.normpath(root_dir)

    if not os.path.isdir(root_dir):
        print(f"Không tìm thấy thư mục đầu vào: {root_dir}")
        return

    material_files = collect_material_files(root_dir)
    if not material_files:
        print(f"Không tìm thấy file đầu vào trong {root_dir}")
        return

    embedding_model = get_embedding_model()
    conn = get_db_connection()
    if not conn:
        print("Không thể kết nối DB")
        return

    try:
        print(f"Tìm thấy {len(material_files)} file trong {root_dir}")
        for fp in material_files:
            meta = parse_material_filename(os.path.basename(fp))
            if not meta:
                print(f"BỎ QUA (không đúng quy ước tên): {fp}")
                continue

            course_id = meta["course"]
            chapter_id = meta["chapter"]
            unit_id = meta["unit"]
            material_type = meta["type"]
            language = meta["lang"]
            material_id = meta["material_id"]
            ext = meta["ext"]

            # trích xuất văn bản
            raw_text = extract_text_from_file(fp, ext)
            if not raw_text or len(raw_text.strip()) < 30:
                print(f"BỎ QUA (không có nội dung hoặc quá ngắn): {fp}")
                continue

            # chunking và embedding
            chunks = chunk_text(raw_text)
            if not chunks:
                print(f"BỎ QUA (không tạo được chunk): {fp}")
                continue

            level = derive_level_from_course(course_id)

            record_dicts: List[Dict[str, Any]] = []
            for idx, chunk_text_value in enumerate(chunks, start=1):
                embedding = embedding_model.encode(chunk_text_value).tolist()
                metadata = {
                    "material_id": material_id,
                    "course_id": course_id,
                    "chapter_id": chapter_id,
                    "unit_id": unit_id,
                    "material_type": material_type,
                    "language": language,
                    "seq": meta["seq"],
                    "level": level,
                    "source_path": fp.replace("\\", "/"),
                    "chunk_index": idx,
                }
                # original_page_number: không áp dụng cho docx/pptx/txt (đặt None)
                record_dicts.append(
                    {
                        "chunk_text": chunk_text_value,
                        "embedding": embedding,
                        "course_id": course_id,
                        "source_document_name": os.path.basename(fp),
                        "original_page_number": None,
                        "level": level,
                        "skill_type": material_type,
                        "metadata_json": json.dumps(metadata, ensure_ascii=False),
                        "material_id": material_id,
                    }
                )

            # xóa cũ theo material_id và chèn mới
            upsert_material_chunks(record_dicts, material_id, conn)
            print(f"OK: {material_id} -> {len(record_dicts)} chunks")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()


