# pdf_chunk_processor.py
import os
import re
import pdfplumber
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Công cụ chunking

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

# Đường dẫn đến file PDF bạn muốn xử lý
# Bạn có thể làm cho nó linh hoạt hơn, ví dụ nhận đường dẫn từ command line argument
PDF_TO_PROCESS_PATH = "../input/JPD113.pdf"  # << THAY ĐỔI NẾU CẦN


# --- Kết nối DB ---
def get_db_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        print("Chunk Processor: Database connection successful.")
        return conn
    except psycopg2.Error as e:
        print(f"Chunk Processor: Error connecting to database: {e}")
        return None


# --- Hàm xử lý và nạp chunk ---
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

    # Khởi tạo Text Splitter từ LangChain
    # chunk_size: kích thước tối đa của mỗi chunk (theo số ký tự)
    # chunk_overlap: số ký tự chồng lấn giữa các chunk liên tiếp (giúp giữ ngữ cảnh)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Bạn có thể thử nghiệm với các kích thước khác nhau (500-1500)
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,  # Thêm vị trí bắt đầu của chunk trong văn bản gốc (metadata)
    )

    all_chunk_records_to_insert = []
    current_lesson_for_pdf = "Unknown Lesson"  # Sẽ được cập nhật

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num_actual, page in enumerate(pdf.pages):
                page_number_for_db = page_num_actual + 1  # Số trang bắt đầu từ 1
                print(f"  Processing Page {page_number_for_db}/{len(pdf.pages)} for {doc_name}...")

                page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                if not page_text:
                    print(f"    No text extracted from page {page_number_for_db}.")
                    continue

                # Cố gắng xác định bài học cho trang hiện tại
                # (Logic này có thể cần cải thiện nếu bài học kéo dài nhiều trang không có tiêu đề lặp lại)
                page_specific_lesson = current_lesson_for_pdf  # Mặc định là bài của trang trước
                lines_for_lesson_check = page_text.split('\n', 10)  # Kiểm tra 10 dòng đầu
                for line in lines_for_lesson_check:
                    lesson_match = re.search(r"^(第\s*([0-9一二三四五六七八九十０-９]+)\s*課)", line.strip())
                    if lesson_match:
                        new_lesson_name = lesson_match.group(1).strip()
                        if new_lesson_name != current_lesson_for_pdf:
                            print(
                                f"    ==> Lesson identified/changed to: {new_lesson_name} on Page {page_number_for_db}")
                        page_specific_lesson = new_lesson_name
                        current_lesson_for_pdf = new_lesson_name  # Cập nhật cho các trang sau
                        break

                # Chia văn bản của trang này thành các chunks
                # LangChain text_splitter trả về list các đối tượng Document, mỗi Document có page_content và metadata
                # Chúng ta truyền metadata để có thể lưu lại nguồn gốc của chunk
                page_document_chunks = text_splitter.create_documents(
                    [page_text],  # Phải là list các text
                    metadatas=[{
                        "source_document": doc_name,
                        "page": page_number_for_db,
                        "lesson": page_specific_lesson
                    }]
                )

                for chunk_doc in page_document_chunks:
                    chunk_content = chunk_doc.page_content.strip()
                    if not chunk_content or len(chunk_content) < 20:  # Bỏ qua các chunk quá ngắn
                        continue

                    # print(f"      Chunk created from lesson '{chunk_doc.metadata.get('lesson')}': {chunk_content[:100]}...") # Debug
                    embedding = model.encode(chunk_content).tolist()
                    all_chunk_records_to_insert.append((
                        doc_name,
                        page_number_for_db,
                        chunk_doc.metadata.get('lesson', "Unknown Lesson"),  # Lấy lesson từ metadata
                        chunk_content,
                        embedding
                    ))

        # Nạp các chunk vào DB
        if all_chunk_records_to_insert:
            print(f"\nTotal chunks to insert for {doc_name}: {len(all_chunk_records_to_insert)}")
            with conn.cursor() as cur:
                insert_query = f"""
                INSERT INTO {CHUNK_TABLE_NAME} 
                    (document_name, page_number, lesson_identifier, chunk_text, embedding)
                VALUES %s
                ON CONFLICT (chunk_text) DO NOTHING; 
                -- Giả sử chunk_text là UNIQUE để tránh trùng lặp hoàn toàn.
                -- Nếu bạn muốn cho phép chunk_text trùng từ các nguồn khác nhau, bỏ ON CONFLICT
                -- hoặc làm UNIQUE trên (document_name, chunk_text)
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
        # Bạn nên chạy lệnh CREATE TABLE ở Bước 1 bằng pgAdmin một lần.
        # Hoặc thêm code tạo bảng ở đây:
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
                # Tạo index nếu chưa có, có thể thêm kiểm tra
                # cur.execute(f"""
                # CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding
                # ON {CHUNK_TABLE_NAME}
                # USING hnsw (embedding vector_cosine_ops);
                # """)
                db_conn.commit()
                print(f"Table '{CHUNK_TABLE_NAME}' checked/created.")
        except Exception as e_table:
            print(f"Error setting up table '{CHUNK_TABLE_NAME}': {e_table}")
            db_conn.rollback()

        # Xử lý file PDF được chỉ định
        process_and_load_pdf(PDF_TO_PROCESS_PATH, db_conn)

        db_conn.close()
        print("Chunk Processor: Database connection closed.")
    print("PDF chunk processing finished.")