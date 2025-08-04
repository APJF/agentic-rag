# src/data_processing/manifest_processor.py

import json
import os
import sys
import pdfplumber
from psycopg2.extras import execute_values
from typing import List, Tuple, Dict, Any

# Thêm thư mục gốc vào system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.embedding import get_embedding_model
from src.core.database import get_db_connection

# Constants
MANIFEST_FILE = 'manifest.json'
DATA_DIR = 'data'
DEFAULT_SKILL_TYPE = 'GENERAL'

# --- KHỞI TẠO CÁC THÀNH PHẦN CỐT LÕI ---
embedding_model = get_embedding_model()


def _load_manifest_data() -> List[Dict[str, Any]]:
    """Load and validate manifest data."""
    manifest_path = os.path.join(project_root, DATA_DIR, MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Không tìm thấy file '{MANIFEST_FILE}' tại '{manifest_path}'")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _extract_page_text(pdf, page_index: int) -> str:
    """Extract text from a specific page."""
    if page_index >= len(pdf.pages):
        return ""

    page = pdf.pages[page_index]
    return page.extract_text() or ""


def _create_chunk_data(chunk_text: str, course_id: str, pdf_filename: str,
                      page_number: int, skill_type: str) -> Tuple:
    """Create chunk data tuple for database insertion."""
    embedding = embedding_model.encode(chunk_text).tolist()
    return (
        chunk_text,
        embedding,
        course_id,
        pdf_filename,
        page_number,
        skill_type
    )


def _process_course_pages(pdf, course_info: Dict[str, Any], pdf_filename: str) -> List[Tuple]:
    """Process pages for a specific course and return chunk data."""
    course_id = course_info['course_id']
    start_page = course_info['start_page']
    end_page = course_info.get('end_page', 'all')
    skill_type = course_info.get('skill_type', DEFAULT_SKILL_TYPE)

    print(f"Đang xử lý cho Course ID: {course_id} | Trang: {start_page}-{end_page}")

    start_idx = start_page - 1
    end_idx = len(pdf.pages) if end_page == 'all' else end_page

    chunks = []
    for i in range(start_idx, end_idx):
        page_text = _extract_page_text(pdf, i)

        if page_text.strip():
            chunk_data = _create_chunk_data(
                page_text, course_id, pdf_filename, i + 1, skill_type
            )
            chunks.append(chunk_data)

    return chunks


def _process_pdf_file(item: Dict[str, Any]) -> List[Tuple]:
    """Process a single PDF file and return all chunks."""
    pdf_path = os.path.join(project_root, item['pdf_path'])
    pdf_filename = os.path.basename(pdf_path)

    print(f"\n--- Bắt đầu xử lý file: {pdf_filename} ---")

    if not os.path.exists(pdf_path):
        print(f"Cảnh báo: Bỏ qua vì không tìm thấy file tại '{pdf_path}'")
        return []

    all_chunks = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for course_info in item['courses']:
                course_chunks = _process_course_pages(pdf, course_info, pdf_filename)
                all_chunks.extend(course_chunks)
    except Exception as e:
        print(f"Lỗi khi xử lý file {pdf_filename}: {e}")
        return []

    return all_chunks


def _insert_chunks_to_database(chunks: List[Tuple]) -> bool:
    """Insert all chunks into database."""
    if not chunks:
        print("Không có chunk nào được tạo ra.")
        return True

    conn = get_db_connection()
    if not conn:
        print("Lỗi: Không thể kết nối DB.")
        return False

    try:
        print(f"\nChuẩn bị chèn {len(chunks)} chunks vào database...")
        with conn.cursor() as cur:
            execute_values(
                cur,
                'INSERT INTO "content_chunks" (chunk_text, embedding, course_id, source_document_name, original_page_number, skill_type) VALUES %s',
                chunks,
                template=None,
                page_size=100
            )
            conn.commit()
            print(f"--- THÀNH CÔNG! Đã chèn {cur.rowcount} chunks. ---")
        return True
    except Exception as e:
        print(f"Lỗi khi chèn dữ liệu: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def process_and_insert_data():
    """
    Hàm chính để đọc manifest, xử lý các file PDF và nạp dữ liệu vào database.
    """
    try:
        manifest_data = _load_manifest_data()
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        return

    all_chunks_to_insert = []

    # Process each PDF file in manifest
    for item in manifest_data:
        pdf_chunks = _process_pdf_file(item)
        all_chunks_to_insert.extend(pdf_chunks)

    # Insert all chunks into database
    _insert_chunks_to_database(all_chunks_to_insert)


if __name__ == "__main__":
    process_and_insert_data()