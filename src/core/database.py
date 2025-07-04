# src/core/vector_store_interface.py
import psycopg2
from psycopg2.sql import SQL, Identifier, Literal
from src.config import settings
from src.core.embedding import encode_text
from typing import List, Dict, Any

def get_db_connection():
    """
    Tạo và trả về một kết nối mới đến database PostgreSQL.
    Đọc thông tin kết nối từ file settings.
    """
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Lỗi kết nối database (OperationalError): {e}")
        return None
    except psycopg2.Error as e:
        print(f"Lỗi database không xác định: {e}")
        return None


def execute_sql_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    """
    Hàm trợ giúp chung để thực thi một câu lệnh SELECT và trả về kết quả
    dưới dạng một danh sách các dictionary.

    Args:
        query: Chuỗi SQL cần thực thi.
        params: Tuple chứa các tham số cho câu lệnh SQL để tránh SQL injection.

    Returns:
        Một danh sách các dictionary, mỗi dictionary đại diện cho một dòng kết quả.
        Trả về danh sách rỗng nếu có lỗi hoặc không có kết quả.
    """
    conn = get_db_connection()
    if not conn:
        # Không thể kết nối, trả về danh sách rỗng
        return []

    results = []
    try:
        # Sử dụng 'with' để đảm bảo cursor được đóng đúng cách
        with conn.cursor() as cur:
            cur.execute(query, params or ())

            # Kiểm tra xem truy vấn có trả về kết quả không
            if cur.description:
                # Lấy tên các cột từ cur.description để làm key cho dictionary
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                for row in rows:
                    results.append(dict(zip(colnames, row)))
    except psycopg2.Error as e:
        print(f"Lỗi khi thực thi câu lệnh SQL: {e}")
        # Không cần rollback với lệnh SELECT, nhưng vẫn nên có
        conn.rollback()
    finally:
        # Luôn đóng kết nối sau khi hoàn tất
        if conn:
            conn.close()

    return results


def retrieve_relevant_documents_from_db(
    query_text: str,
    top_k: int = 3,
    table_name: str = "contentchunks", # Sử dụng tên bảng trực tiếp hoặc từ settings
    filters: dict = None
) -> list[dict]:
    """
    Truy xuất các tài liệu liên quan từ CSDL với logic xây dựng query đã được sửa lỗi.
    """
    query_embedding = encode_text(query_text)
    # Chuyển embedding thành chuỗi string theo định dạng của pgvector
    embedding_str = str(list(query_embedding))

    conn = get_db_connection()
    if not conn:
        return []

    retrieved_items = []

    # --- SỬA LỖI LOGIC TẠI ĐÂY ---
    # 1. Xây dựng mệnh đề WHERE và các tham số của nó một cách riêng biệt
    where_clauses = []
    where_params = []
    if filters:
        for key, value in filters.items():
            # Thêm "Identifier(key)" để tránh SQL injection cho tên cột
            where_clauses.append(SQL("{}=%s").format(Identifier(key)))
            where_params.append(value)

    where_sql = SQL(" WHERE {}").format(SQL(" AND ").join(where_clauses)) if where_clauses else SQL("")

    # 2. Xây dựng câu truy vấn cuối cùng với các placeholder
    # Lưu ý: Identifier(table_name) để bảo vệ tên bảng
    sql_query = SQL("""
        SELECT chunk_text, source_document_name, original_page_number, level, skill_type, metadata_json
        FROM {table}
        {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s;
    """).format(
        table=Identifier(table_name),
        where_clause=where_sql
    )

    # 3. Kết hợp các tham số theo đúng thứ tự
    # Thứ tự: [params_for_where, param_for_orderby, param_for_limit]
    final_params = where_params + [embedding_str, top_k]
    # --- KẾT THÚC SỬA LỖI ---

    try:
        with conn.cursor() as cur:
            cur.execute(sql_query, final_params)
            results = cur.fetchall()
            for row in results:
                 retrieved_items.append({
                    "text": row[0],
                    "metadata": {"document": row[1], "page": row[2], "level": row[3], "skill": row[4], "lesson": (row[5] or {}).get('lesson')}
                })
    except Exception as e:
        # In ra câu lệnh và tham số để debug dễ hơn
        # print("SQL Query Failed:", cur.mogrify(sql_query, final_params).decode())
        print(f"Lỗi truy xuất chunk từ bảng {table_name}: {e}")
    finally:
        if conn:
            conn.close()

    return retrieved_items

def find_precise_definitional_source_from_db(
    japanese_term: str,
    # SỬA Ở ĐÂY: Dùng tên biến đã chuẩn hóa
    table_name: str = settings.RAG_CONTENT_CHUNK_TABLE
) -> dict | None:
    if not japanese_term: return None
    targeted_query = f"Định nghĩa và vị trí của từ tiếng Nhật: {japanese_term}"
    definitional_chunks = retrieve_relevant_documents_from_db(targeted_query, top_k=1, table_name=table_name)
    if definitional_chunks:
        return definitional_chunks[0].get("metadata")
    return None