import psycopg2
from src.config import settings

def print_db_schema():
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )
    cur = conn.cursor()
    print("Tất cả các bảng:")
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = cur.fetchall()
    for t in tables:
        table = t[0]
        print(f"\nCấu trúc bảng: {table}")
        cur.execute(f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = '{table}'
        """)
        for row in cur.fetchall():
            print("  ", row)
        # Liệt kê khóa chính
        cur.execute(f"""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = '{table}' AND tc.constraint_type = 'PRIMARY KEY'
        """)
        pk = [r[0] for r in cur.fetchall()]
        if pk:
            print("  Khóa chính:", pk)
        # Liệt kê khóa ngoại
        cur.execute(f"""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name='{table}';
        """)
        fk = cur.fetchall()
        if fk:
            print("  Khóa ngoại:")
            for f in fk:
                print("    ", f)
    cur.close()
    conn.close()

if __name__ == "__main__":
    print_db_schema()