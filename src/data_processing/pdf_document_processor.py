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

# --- Khởi tạo các đối tượng dùng chung ---
embedding_model = get_embedding_model()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Giảm chunk size để mỗi chunk tập trung hơn vào một chủ đề
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

# === THAY ĐỔI LỚN: Hàm xử lý giờ nhận thêm level và skill_type ===
def process_pdf_to_chunks(
    pdf_path: str,
    level: str, # 'N5', 'N4', 'N3'
    skill_type: str = 'Vocabulary' # Mặc định là từ vựng cho các file hiện tại
) -> List[Dict[str, Any]]:
    """
    Đọc PDF, chunking, tạo embedding và gán metadata quan trọng (level, skill_type).
    """
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file tại {pdf_path}")
        return []

    doc_name = os.path.basename(pdf_path)
    print(f"Đang xử lý PDF: {doc_name} với Level={level}, Skill={skill_type}...")

    all_chunks_data = []
    current_lesson_identifier = "Unknown Lesson"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

                lesson_match = re.search(r"^(第\s*([0-9一二三四五六七八九十０-９]+)\s*課)", page_text.split('\n', 1)[0])
                if lesson_match:
                    current_lesson_identifier = lesson_match.group(1).strip()

                page_chunks = text_splitter.split_text(page_text)

                for chunk_text in page_chunks:
                    chunk_text = chunk_text.strip()
                    if len(chunk_text) < 30: continue

                    embedding_vector = embedding_model.encode(chunk_text).tolist()

                    chunk_record = {
                        "chunk_text": chunk_text,
                        "embedding": embedding_vector,
                        "source_document_name": doc_name,
                        "original_page_number": page_num,
                        "level": level, # <<< THÊM LEVEL
                        "skill_type": skill_type, # <<< THÊM SKILL TYPE
                        "metadata_json": json.dumps({ # <<< Chuyển sang JSON string
                            "lesson": current_lesson_identifier
                        }),
                    }
                    all_chunks_data.append(chunk_record)

        print(f"Hoàn tất {doc_name}. Tạo ra {len(all_chunks_data)} chunks.")
        return all_chunks_data

    except Exception as e:
        print(f"Lỗi khi xử lý PDF {doc_name}: {e}")
        return []

def batch_insert_chunks_to_db(chunks_data: List[Dict[str, Any]], conn):
    if not chunks_data:
        print("Không có dữ liệu để ghi.")
        return

    records_to_insert = [
        (
            chunk['chunk_text'],
            chunk['embedding'],
            chunk.get('source_document_name'),
            chunk.get('original_page_number'),
            chunk['level'],
            chunk['skill_type'],
            chunk.get('metadata_json') # Đã là JSON string
        ) for chunk in chunks_data
    ]

    try:
        with conn.cursor() as cur:
            # === SỬA TÊN BẢNG VÀ CÁC CỘT CHO KHỚP SCHEMA MỚI ===
            table_name = settings.RAG_CONTENT_CHUNK_TABLE # Dùng tên bảng từ settings
            insert_query = f"""
                INSERT INTO {table_name} (
                    chunk_text, embedding, source_document_name, original_page_number,
                    level, skill_type, metadata_json
                ) VALUES %s;
            """
            execute_values(cur, insert_query, records_to_insert)
            conn.commit()
            print(f"Đã ghi thành công {cur.rowcount} chunks vào bảng '{table_name}'.")
    except Exception as e:
        print(f"Lỗi khi ghi vào database: {e}")
        conn.rollback()


# === THAY ĐỔI LOGIC CHẠY CHÍNH ĐỂ XỬ LÝ 3 FILE CỤ THỂ ===
if __name__ == "__main__":
    input_pdfs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "input_pdfs"
    )

    # Ánh xạ tên file tới level tương ứng
    file_to_level_map = {
        "tuvung_quyendo.pdf": "N5",
        "tuvung_quyenvang.pdf": "N4",
        "tuvung_quyenxanh.pdf": "N3"
    }

    db_conn = get_db_connection()
    if not db_conn:
        print("Dừng xử lý do không thể kết nối DB.")
        exit()

    try:
        for filename, level in file_to_level_map.items():
            full_pdf_path = os.path.join(input_pdfs_dir, filename)
            if os.path.exists(full_pdf_path):
                print(f"\n{'='*20} Bắt đầu xử lý: {filename} (Level: {level}) {'='*20}")
                # Giả định tất cả đều là sách từ vựng
                processed_chunks = process_pdf_to_chunks(full_pdf_path, level=level, skill_type="Vocabulary")
                batch_insert_chunks_to_db(processed_chunks, db_conn)
            else:
                print(f"\nCảnh báo: Không tìm thấy file {filename} trong thư mục input_pdfs.")

    finally:
        if db_conn:
            db_conn.close()
            print(f"\n{'='*20} Đã đóng kết nối Database. {'='*20}")

    print("\nQuy trình xử lý dữ liệu đã kết thúc.")