import sys
import os
import psycopg2

# Thêm thư mục gốc của dự án vào system path để có thể import các module khác
# Điều này đảm bảo script có thể chạy được từ bất kỳ đâu
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import hàm get_db_connection từ cấu trúc dự án của bạn
# Giả sử nó nằm trong src/core/vector_store_interface.py
try:
    from src.core.vector_store_interface import get_db_connection
except ImportError:
    print("[LỖI] Không thể import get_db_connection. Hãy chắc chắn rằng file check_db.py")
    print("đang được đặt ở thư mục gốc của dự án và cấu trúc thư mục là chính xác.")
    sys.exit(1)

def check_database_connection_and_data():
    """
    Hàm này thực hiện 3 bước kiểm tra chính:
    1. Kiểm tra kết nối tới PostgreSQL.
    2. Kiểm tra và lấy dữ liệu từ bảng 'subjects' (dành cho lộ trình).
    3. Kiểm tra và lấy dữ liệu từ bảng 'contentchunks' (dành cho RAG/Quiz).
    """
    print("--- Bắt đầu kiểm tra kết nối và dữ liệu Database ---")
    conn = None
    try:
        # --- BƯỚC 1: KIỂM TRA KẾT NỐI ---
        print("\n1. Đang thử kết nối tới PostgreSQL...")
        conn = get_db_connection()

        if conn and not conn.closed:
            print("   => THÀNH CÔNG: Đã kết nối tới PostgreSQL!")
        else:
            print("   => LỖI: Không thể thiết lập kết nối. Vui lòng kiểm tra các biến môi trường")
            print("      (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) trong file .env")
            print("      và file cấu hình src/config/settings.py.")
            return

        # --- BƯỚC 2: KIỂM TRA BẢNG 'subjects' ---
        print("\n2. Đang thử truy vấn bảng 'subjects'...")
        with conn.cursor() as cur:
            cur.execute("SELECT id, subject_name, level, topic FROM subjects LIMIT 11;")
            subjects_rows = cur.fetchall()

            if subjects_rows:
                print(f"   => THÀNH CÔNG: Tìm thấy {len(subjects_rows)} dòng trong bảng 'subjects'.")
                print("   Dữ liệu mẫu:")
                for row in subjects_rows:
                    print(f"     - ID: {row[0]}, Tên: {row[1]}, Level: {row[2]}, Topic: {row[3]}")
            else:
                print("   => CẢNH BÁO: Kết nối thành công nhưng bảng 'subjects' đang rỗng.")
                print("   => Gợi ý: Bạn đã chạy script SQL INSERT để nạp dữ liệu demo cho lộ trình chưa?")

        # --- BƯỚC 3: KIỂM TRA BẢNG 'contentchunks' ---
        print("\n3. Đang thử truy vấn bảng 'contentchunks'...")
        with conn.cursor() as cur:
            cur.execute("SELECT id, source_document_name, level, topic FROM contentchunks LIMIT 20;")
            chunks_rows = cur.fetchall()

            if chunks_rows:
                print(f"   => THÀNH CÔNG: Tìm thấy {len(chunks_rows)} dòng trong bảng 'contentchunks'.")
                print("   Dữ liệu mẫu:")
                for row in chunks_rows:
                    print(f"     - ID: {row[0]}, Nguồn: {row[1]}, Level: {row[2]}, Topic: {row[3]}")
            else:
                print("   => CẢNH BÁO: Kết nối thành công nhưng bảng 'contentchunks' đang rỗng.")
                print("   => Gợi ý: Bạn đã chạy script `src/data_processing/pdf_document_processor.py` để xử lý file PDF chưa?")

    except psycopg2.OperationalError as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] Lỗi Kết nối (OperationalError): {e}")
        print("Gợi ý: Database của bạn đã khởi động chưa? Tên host, port, user, password, dbname có chính xác không?")
    except psycopg2.Error as e:
        # Bắt các lỗi khác của psycopg2, ví dụ như bảng không tồn tại
        print(f"\n[LỖI NGHIÊM TRỌNG] Lỗi PostgreSQL (psycopg2.Error): {e}")
        print("Gợi ý: Tên bảng có đúng không? Cấu trúc bảng có khớp với câu lệnh SELECT không?")
    except Exception as e:
        print(f"\n[LỖI KHÁC] Một lỗi không xác định đã xảy ra: {e}")
    finally:
        if conn and not conn.closed:
            conn.close()
            print("\n--- Kiểm tra hoàn tất. Đã đóng kết nối. ---")


if __name__ == "__main__":
    check_database_connection_and_data()