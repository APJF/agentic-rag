# src/data_processing/manifest_processor.py

import json
import os
import sys
import pdfplumber
from psycopg2.extras import execute_values

# Thêm thư mục gốc vào system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.embedding import get_embedding_model
from src.core.database import get_db_connection

# --- KHỞI TẠO CÁC THÀNH PHẦN CỐT LÕI ---
embedding_model = get_embedding_model()


def process_and_insert_data():
    """
    Hàm chính để đọc manifest, xử lý các file PDF và nạp dữ liệu vào database.
    """
    manifest_path = os.path.join(project_root, 'data', 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f"Lỗi: Không tìm thấy file 'manifest.json' tại '{manifest_path}'")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)

    all_chunks_to_insert = []

    # Duyệt qua từng mục trong manifest (từng file PDF)
    for item in manifest_data:
        pdf_path = os.path.join(project_root, item['pdf_path'])
        print(f"\n--- Bắt đầu xử lý file: {os.path.basename(pdf_path)} ---")

        if not os.path.exists(pdf_path):
            print(f"Cảnh báo: Bỏ qua vì không tìm thấy file tại '{pdf_path}'")
            continue

        with pdfplumber.open(pdf_path) as pdf:
            for course_info in item['courses']:
                course_id = course_info['course_id']
                start_page = course_info['start_page']
                end_page = course_info.get('end_page', 'all')
                skill_type = course_info.get('skill_type', 'GENERAL')

                print(f"Đang xử lý cho Course ID: {course_id} | Trang: {start_page}-{end_page}")

                start_idx = start_page - 1
                end_idx = len(pdf.pages) if end_page == 'all' else end_page

                for i in range(start_idx, end_idx):
                    page = pdf.pages[i]
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        chunk_text = page_text
                        embedding = embedding_model.encode(chunk_text).tolist()

                        all_chunks_to_insert.append((
                            chunk_text,
                            embedding,
                            course_id,
                            os.path.basename(pdf_path),
                            i + 1,
                            skill_type
                        ))

    # Chèn toàn bộ dữ liệu vào database
    conn = get_db_connection()
    if not conn:
        print("Lỗi: Không thể kết nối DB.")
        return

    if all_chunks_to_insert:
        print(f"\nChuẩn bị chèn {len(all_chunks_to_insert)} chunks vào database...")
        try:
            with conn.cursor() as cur:
                query = """
                        INSERT INTO "content_chunks" (chunk_text, embedding, course_id, source_document_name, \
                                                      original_page_number, skill_type) \
                        VALUES %s; \
                        """
                execute_values(cur, query, all_chunks_to_insert)
                conn.commit()
                print(f"--- THÀNH CÔNG! Đã chèn {cur.rowcount} chunks. ---")
        except Exception as e:
            print(f"Lỗi khi chèn dữ liệu: {e}")
            conn.rollback()
    else:
        print("Không có chunk nào được tạo ra.")

    if conn:
        conn.close()


if __name__ == "__main__":
    process_and_insert_data()