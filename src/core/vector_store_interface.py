# src/core/vector_store_interface.py
import psycopg2
from psycopg2.extensions import AsIs # Quan trọng cho query pgvector
from src.config import settings
from src.core.embedding_manager import encode_text # Sử dụng hàm encode tập trung

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def retrieve_relevant_documents_from_db(
    query_text: str,
    top_k: int = 3,
    table_name: str = settings.CHUNK_TABLE_NAME
) -> list[dict]:
    query_embedding = encode_text(query_text) # Dùng hàm encode
    conn = get_db_connection()
    if not conn: return []
    retrieved_items = []
    try:
        with conn.cursor() as cur:
            # Sử dụng AsIs cho tên bảng và vector
            cur.execute(
                """
                SELECT chunk_text, document_name, page_number, lesson_identifier
                FROM %s ORDER BY embedding <=> %s LIMIT %s;
                """,
                (AsIs(table_name), AsIs(f"'{query_embedding}'"), top_k)
            )
            results = cur.fetchall()
            for row in results:
                retrieved_items.append({
                    "text": row[0],
                    "metadata": {"document": row[1], "page": row[2], "lesson": row[3]}
                })
    except psycopg2.Error as e:
        print(f"Lỗi truy xuất chunk từ bảng {table_name}: {e}")
    finally:
        if conn: conn.close()
    return retrieved_items

def find_precise_definitional_source_from_db(
    japanese_term: str,
    table_name: str = settings.CHUNK_TABLE_NAME # Hoặc một bảng chuyên biệt hơn
) -> dict | None:
    if not japanese_term: return None
    targeted_query = f"Định nghĩa và vị trí của từ tiếng Nhật: {japanese_term}"
    definitional_chunks = retrieve_relevant_documents_from_db(targeted_query, top_k=1, table_name=table_name)
    if definitional_chunks:
        return definitional_chunks[0].get("metadata")
    return None