# pdf_document_processor.py
import os
import re
import pdfplumber
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# --- Cấu hình ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))

# Tên bảng mới cho các chunks
CHUNK_TABLE_NAME = "pdf_document_chunks"

# Lấy thư mục gốc của dự án và thư mục chứa các file PDF input
# Giả định script này nằm trong AgenticRAG/src/data_processing/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # Đi lên 2 cấp để tới AgenticRAG/
INPUT_PDFS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "input_pdfs") # Đường dẫn tới AgenticRAG/data/input_pdfs/


# --- Kết nối DB ---
def get_db_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        print("Chunk Processor: Database connection successful.")
        return conn
    except psycopg2.Error as e:
        print(f"Chunk Processor: Error connecting to database: {e}")
        return None


# --- Hàm xử lý và nạp chunk (Hàm này giữ nguyên logic, chỉ thay đổi cách gọi nó) ---
def process_and_load_pdf(pdf_path: str, conn):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    doc_name = os.path.basename(pdf_path)
    print(f"Processing PDF: {doc_name}...")

    # Khởi tạo embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    all_chunk_records_to_insert = []
    current_lesson_for_pdf = "Unknown Lesson"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num_actual, page in enumerate(pdf.pages):
                page_number_for_db = page_num_actual + 1
                print(f"  Processing Page {page_number_for_db}/{len(pdf.pages)} for {doc_name}...")

                page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                if not page_text:
                    print(f"    No text extracted from page {page_number_for_db}.")
                    continue

                page_specific_lesson = current_lesson_for_pdf
                lines_for_lesson_check = page_text.split('\n', 10)
                for line in lines_for_lesson_check:
                    lesson_match = re.search(r"^(第\s*([0-9一二三四五六七八九十０-９]+)\s*課)", line.strip())
                    if lesson_match:
                        new_lesson_name = lesson_match.group(1).strip()
                        if new_lesson_name != current_lesson_for_pdf:
                            print(
                                f"    ==> Lesson identified/changed to: {new_lesson_name} on Page {page_number_for_db}")
                        page_specific_lesson = new_lesson_name
                        current_lesson_for_pdf = new_lesson_name
                        break

                page_document_chunks = text_splitter.create_documents(
                    [page_text],
                    metadatas=[{
                        "source_document": doc_name,
                        "page": page_number_for_db,
                        "lesson": page_specific_lesson
                    }]
                )

                for chunk_doc in page_document_chunks:
                    chunk_content = chunk_doc.page_content.strip()
                    if not chunk_content or len(chunk_content) < 20:
                        continue

                    embedding = model.encode(chunk_content).tolist()
                    all_chunk_records_to_insert.append((
                        doc_name,
                        page_number_for_db,
                        chunk_doc.metadata.get('lesson', "Unknown Lesson"),
                        chunk_content,
                        embedding
                    ))

        if all_chunk_records_to_insert:
            print(f"\nTotal chunks to insert for {doc_name}: {len(all_chunk_records_to_insert)}")
            with conn.cursor() as cur:
                insert_query = f"""
                INSERT INTO {CHUNK_TABLE_NAME}
                    (document_name, page_number, lesson_identifier, chunk_text, embedding)
                VALUES %s
                ON CONFLICT (chunk_text) DO NOTHING;
                """
                execute_values(cur, insert_query, all_chunk_records_to_insert)
                conn.commit()
                print(f"{cur.rowcount} chunks for {doc_name} inserted/updated successfully.")
        else:
            print(f"No valid chunks generated for {doc_name}.")

    except Exception as e:
        print(f"An error occurred while processing PDF {doc_name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    db_conn = get_db_connection()
    if db_conn:
        # Tạo bảng (và extension) nếu chưa có
        try:
            with db_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {CHUNK_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    page_number INTEGER,
                    lesson_identifier TEXT,
                    chunk_text TEXT NOT NULL UNIQUE,
                    embedding VECTOR({EMBEDDING_DIMENSION})
                );""")
                db_conn.commit()
                print(f"Table '{CHUNK_TABLE_NAME}' checked/created.")
        except Exception as e_table:
            print(f"Error setting up table '{CHUNK_TABLE_NAME}': {e_table}")
            db_conn.rollback()
            # Nếu lỗi tạo bảng thì không nên tiếp tục, nên đóng kết nối và thoát
            db_conn.close()
            print("PDF chunk processing aborted due to table setup error.")
            exit() # Hoặc return nếu đây là một hàm

        # ---- THAY ĐỔI CHÍNH Ở ĐÂY ----
        # Lấy danh sách tất cả các file PDF trong thư mục INPUT_PDFS_DIR
        if not os.path.exists(INPUT_PDFS_DIR):
            print(f"Error: Input PDF directory not found at {INPUT_PDFS_DIR}")
        else:
            pdf_files_to_process = [f for f in os.listdir(INPUT_PDFS_DIR) if f.lower().endswith(".pdf")]

            if not pdf_files_to_process:
                print(f"No PDF files found in {INPUT_PDFS_DIR}")
            else:
                print(f"Found {len(pdf_files_to_process)} PDF file(s) to process: {pdf_files_to_process}")
                for pdf_filename in pdf_files_to_process:
                    full_pdf_path = os.path.join(INPUT_PDFS_DIR, pdf_filename)
                    print(f"\n======================================================================")
                    print(f"Starting processing for PDF file: {pdf_filename}")
                    print(f"Full path: {full_pdf_path}")
                    print(f"======================================================================")
                    process_and_load_pdf(full_pdf_path, db_conn) # Gọi hàm xử lý cho từng file
                    print(f"--- Finished processing for: {pdf_filename} ---")
        # ---- KẾT THÚC THAY ĐỔI ----

        db_conn.close()
        print("\nChunk Processor: Database connection closed.")
    print("\nPDF document processing finished for all specified files.")