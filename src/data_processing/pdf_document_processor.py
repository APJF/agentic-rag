# src/data_processing/pdf_document_processor.py
import os
import re
import pdfplumber
import psycopg2
import json
from psycopg2.extras import execute_values
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings
from src.core.embedding_manager import get_embedding_model
from src.core.vector_store_interface import get_db_connection

# --- Khởi tạo các đối tượng dùng chung một lần duy nhất khi module được tải ---
# Tải model embedding một lần duy nhất để tái sử dụng, tăng hiệu năng.
print("Initializing document processor dependencies...")
embedding_model = get_embedding_model()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,  # Tăng overlap một chút để giữ ngữ cảnh tốt hơn
    length_function=len,
    add_start_index=True,
)
print("Document processor dependencies initialized.")


def process_pdf_to_chunks(pdf_path: str, external_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Hàm này chỉ tập trung vào một việc: Đọc file PDF, chunking, và tạo embedding.
    Nó sẽ trả về một list các dictionary đã sẵn sàng để được ghi vào database.
    Nó không trực tiếp ghi vào DB.

    :param pdf_path: Đường dẫn đến file PDF cần xử lý.
    :param external_metadata: Một dict chứa metadata từ bên ngoài (ví dụ: từ CMS-service)
                             như {'external_subject_id': 'S123', 'external_chapter_id': 'C456'}
    :return: Một danh sách các chunks, mỗi chunk là một dictionary.
    """
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file tại {pdf_path}")
        return []

    if external_metadata is None:
        external_metadata = {}

    doc_name = os.path.basename(pdf_path)
    print(f"Đang xử lý PDF: {doc_name}...")

    all_chunks_data = []
    current_lesson_identifier = "Unknown Lesson"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

                # Cố gắng tìm lesson identifier trên trang
                lines_for_lesson_check = page_text.split('\n', 10)
                for line in lines_for_lesson_check:
                    lesson_match = re.search(r"^(第\s*([0-9一二三四五六七八九十０-９]+)\s*課)", line.strip())
                    if lesson_match:
                        current_lesson_identifier = lesson_match.group(1).strip()
                        break

                # Chunking text của trang
                page_chunks = text_splitter.split_text(page_text)

                for chunk_text in page_chunks:
                    chunk_text = chunk_text.strip()
                    if len(chunk_text) < 50:  # Bỏ qua các chunk quá ngắn
                        continue

                    # Tạo embedding cho chunk
                    embedding_vector = embedding_model.encode(chunk_text).tolist()

                    # Xây dựng bản ghi hoàn chỉnh cho chunk
                    chunk_record = {
                        "chunk_text": chunk_text,
                        "embedding": embedding_vector,
                        "source_document_name": doc_name,
                        "original_page_number": page_num,
                        "metadata_json": {"lesson_identifier_scanned": current_lesson_identifier},
                        # Gộp metadata từ bên ngoài vào
                        "external_subject_id": external_metadata.get("external_subject_id"),
                        "external_chapter_id": external_metadata.get("external_chapter_id"),
                        "external_lesson_detail_id": external_metadata.get("external_lesson_detail_id"),
                        "external_skill_id": external_metadata.get("external_skill_id")
                    }
                    all_chunks_data.append(chunk_record)

        print(f"Hoàn tất xử lý file {doc_name}. Tạo ra được {len(all_chunks_data)} chunks.")
        return all_chunks_data

    except Exception as e:
        print(f"Đã có lỗi xảy ra khi xử lý PDF {doc_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def batch_insert_chunks_to_db(chunks_data: List[Dict[str, Any]], conn):
    """
    Hàm này nhận một danh sách các chunk đã xử lý và thực hiện ghi hàng loạt vào DB.
    *** PHIÊN BẢN ĐÃ SỬA LỖI 'can't adapt type 'dict' ***
    """
    if not chunks_data:
        print("Không có dữ liệu chunk để ghi vào database.")
        return

    # Chuẩn bị dữ liệu để insert bằng execute_values
    records_to_insert = [
        (
            chunk['chunk_text'],
            chunk['embedding'],
            chunk.get('source_document_name'),
            chunk.get('original_page_number'),

            # ===== THAY ĐỔI CHÍNH Ở ĐÂY =====
            # Chuyển đổi dictionary metadata thành một chuỗi JSON
            # để psycopg2 có thể hiểu và lưu vào cột kiểu JSONB.
            json.dumps({
                "scanned_lesson": chunk.get('metadata_json', {}).get('lesson_identifier_scanned'),
                "subject_id": chunk.get('external_subject_id'),
                "chapter_id": chunk.get('external_chapter_id'),
                "lesson_detail_id": chunk.get('external_lesson_detail_id'),
                "skill_id": chunk.get('external_skill_id'),
            }),
            # =================================

            chunk.get('external_subject_id'),
            chunk.get('external_chapter_id'),
            chunk.get('external_lesson_detail_id'),
            chunk.get('external_skill_id')
        ) for chunk in chunks_data
    ]

    try:
        with conn.cursor() as cur:
            # Câu lệnh INSERT này khớp với schema DB đã được chuẩn hóa
            insert_query = f"""
                INSERT INTO {settings.RAG_CONTENT_CHUNK_TABLE} (
                    chunk_text, embedding, source_document_name, original_page_number,
                    metadata_json, external_subject_id, external_chapter_id,
                    external_lesson_detail_id, external_skill_id
                ) VALUES %s;
            """
            execute_values(cur, insert_query, records_to_insert)
            conn.commit()
            print(f"Đã ghi thành công {cur.rowcount} chunks vào bảng '{settings.RAG_CONTENT_CHUNK_TABLE}'.")
    except Exception as e:
        print(f"Lỗi khi ghi chunks vào database: {e}")
        conn.rollback()

if __name__ == "__main__":

    # 1. Xác định thư mục chứa các file PDF cần xử lý
    input_pdfs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "input_pdfs"
    )

    if not os.path.exists(input_pdfs_dir):
        print(f"Lỗi: Thư mục input PDF không tồn tại tại: {input_pdfs_dir}")
    else:
        pdf_files_to_process = [f for f in os.listdir(input_pdfs_dir) if f.lower().endswith(".pdf")]

        if not pdf_files_to_process:
            print(f"Không tìm thấy file PDF nào trong: {input_pdfs_dir}")
        else:
            print(f"Tìm thấy {len(pdf_files_to_process)} file PDF để xử lý: {pdf_files_to_process}")

            # Mở kết nối DB một lần cho toàn bộ quá trình
            db_conn = get_db_connection()
            if not db_conn:
                print("Dừng xử lý do không thể kết nối đến DB.")
                exit()

            try:
                for pdf_filename in pdf_files_to_process:
                    full_pdf_path = os.path.join(input_pdfs_dir, pdf_filename)
                    print(f"\n{'=' * 20} Bắt đầu xử lý: {pdf_filename} {'=' * 20}")

                    # Giả lập metadata từ CMS-service cho mục đích test
                    mock_metadata = {
                        "external_subject_id": f"SUBJ_{os.path.splitext(pdf_filename)[0]}",
                        "external_skill_id": "General"
                    }

                    # Bước 1: Parse file PDF thành các chunks dữ liệu có cấu trúc
                    processed_chunks = process_pdf_to_chunks(full_pdf_path, external_metadata=mock_metadata)

                    # Bước 2: Ghi các chunks đã xử lý vào database
                    batch_insert_chunks_to_db(processed_chunks, db_conn)

                    print(f"--- Hoàn tất xử lý: {pdf_filename} ---")
            finally:
                # Luôn đóng kết nối DB sau khi xong việc
                if db_conn:
                    db_conn.close()
                    print(f"\n{'=' * 20} Đã đóng kết nối Database. {'=' * 20}")

    print("\nQuy trình xử lý tài liệu chung đã kết thúc.")