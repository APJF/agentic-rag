from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import psycopg2
from .database import get_db_connection


def get_or_create_user(user_id: str, display_name: str = None) -> bool:
    """
    Kiểm tra user_id có tồn tại không, nếu không thì tạo mới.
    Đã được cập nhật để khớp với cấu trúc bảng "User" mới.
    """
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            query = 'INSERT INTO "User" (id) VALUES (%s) ON CONFLICT (id) DO NOTHING;'

            cur.execute(query, (user_id,))

            conn.commit()
            print(f"Đã xác thực hoặc tạo người dùng: {user_id}")
            return True
    except psycopg2.Error as e:
        print(f"Lỗi khi get/create user: {e}")
        conn.rollback()
        return False
    finally:
        if conn: conn.close()


def list_sessions_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Lấy danh sách các phiên làm việc của một người dùng,
    sắp xếp theo thời gian cập nhật gần nhất."""
    conn = get_db_connection()
    if not conn: return []
    sessions = []
    try:
        with conn.cursor() as cur:
            query = "SELECT id, session_name, updated_at FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC;"
            cur.execute(query, (user_id,))
            rows = cur.fetchall()
            for row in rows:
                sessions.append({"id": row[0], "session_name": row[1], "updated_at": row[2]})
    except psycopg2.Error as e:
        print(f"Lỗi khi liệt kê các phiên: {e}")
    finally:
        if conn: conn.close()
    return sessions


def create_new_session(user_id: str, session_name: str) -> Optional[int]:
    """Tạo một phiên mới cho người dùng và trả về ID của phiên đó."""
    conn = get_db_connection()
    if not conn: return None
    session_id = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (user_id, session_name) VALUES (%s, %s) RETURNING id;",
                (user_id, session_name)
            )
            session_id = cur.fetchone()[0]
            conn.commit()
            print(f"Đã tạo phiên mới '{session_name}' với ID: {session_id}")
    except psycopg2.Error as e:
        print(f"Lỗi khi tạo phiên mới: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return session_id


def load_chat_history(session_id: int) -> List[BaseMessage]:
    """Tải lịch sử chat của một phiên từ database bằng ID số của phiên."""
    conn = get_db_connection()
    if not conn: return []
    history = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT message_type, content FROM chat_messages WHERE session_id = %s ORDER BY message_order ASC;",
                (session_id,)
            )
            for row in cur.fetchall():
                msg_type, content = row
                if msg_type == 'human':
                    history.append(HumanMessage(content=content))
                else:
                    history.append(AIMessage(content=content))
    except psycopg2.Error as e:
        print(f"Lỗi khi tải lịch sử chat: {e}")
    finally:
        if conn: conn.close()
    return history


def add_new_messages(session_id: int, new_messages: List[BaseMessage]):
    """Thêm các tin nhắn mới vào một phiên đã có."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(message_order), 0) FROM chat_messages WHERE session_id = %s;",
                        (session_id,))
            last_order = cur.fetchone()[0]

            for i, msg in enumerate(new_messages):
                message_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
                cur.execute(
                    "INSERT INTO chat_messages (session_id, message_type, content, message_order) VALUES (%s, %s, %s, %s);",
                    (session_id, message_type, msg.content, last_order + i + 1)
                )

            # Cập nhật lại thời gian 'updated_at' của session cha
            cur.execute("UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s;", (session_id,))
            conn.commit()
    except psycopg2.Error as e:
        print(f"Lỗi khi thêm tin nhắn mới: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()


def format_history_for_prompt(chat_history: List[BaseMessage]) -> str:
    """
    Chuyển đổi một danh sách các đối tượng BaseMessage (HumanMessage, AIMessage)
    thành một chuỗi văn bản duy nhất, dễ đọc, để đưa vào prompt.

    Hàm này đóng vai trò "phiên dịch" giữa cấu trúc dữ liệu nội bộ và
    dữ liệu dạng văn bản mà LLM có thể hiểu dễ dàng.

    Args:
        chat_history: Danh sách các đối tượng tin nhắn từ LangChain.

    Returns:
        Một chuỗi văn bản định dạng cuộc hội thoại.
    """
    if not chat_history:
        return "Không có lịch sử trò chuyện."
    formatted_lines = []

    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_lines.append(f"Người dùng: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_lines.append(f"Trợ lý: {message.content}")
        else:
            pass

    return "\n".join(formatted_lines)


def delete_session(session_id: int) -> bool:
    """
    Xóa một phiên trò chuyện và tất cả các tin nhắn liên quan khỏi database.
    Trả về True nếu xóa thành công, False nếu không tìm thấy phiên.
    """
    conn = get_db_connection()
    if not conn: return False

    deleted_rows = 0
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE id = %s;", (session_id,))
            deleted_rows = cur.rowcount
            conn.commit()
            if deleted_rows > 0:
                print(f"[Thông báo] Đã xóa thành công phiên có ID: {session_id}")
    except psycopg2.Error as e:
        print(f"[Lỗi] Không thể xóa phiên {session_id}: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()

    return deleted_rows > 0


# === THÊM HÀM MỚI ĐỂ SỬA TÊN PHIÊN ===
def rename_session(session_id: int, new_name: str) -> bool:
    """
    Cập nhật lại tên của một phiên trò chuyện.
    Trả về True nếu cập nhật thành công, False nếu không tìm thấy phiên.
    """
    conn = get_db_connection()
    if not conn: return False

    updated_rows = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET session_name = %s, updated_at = NOW() WHERE id = %s;",
                (new_name, session_id)
            )
            updated_rows = cur.rowcount
            conn.commit()
            if updated_rows > 0:
                print(f"[Thông báo] Đã đổi tên phiên {session_id} thành '{new_name}'")
    except psycopg2.Error as e:
        print(f"[Lỗi] Không thể đổi tên phiên {session_id}: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()

    return updated_rows > 0


def rewind_last_turn(session_id: int) -> bool:
    """
    Xóa 2 tin nhắn cuối cùng (một cặp Human-AI) khỏi một phiên trong database.
    Hàm này thực hiện hành động "tua lại" một lượt nói.
    Trả về True nếu xóa thành công, False nếu có lỗi hoặc không có đủ tin nhắn để xóa.
    """
    conn = get_db_connection()
    if not conn:
        print("[Lỗi] Không thể kết nối DB để tua lại phiên.")
        return False

    success = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM chat_messages WHERE session_id = %s ORDER BY message_order DESC LIMIT 2;",
                (session_id,)
            )
            rows_to_delete = cur.fetchall()

            if len(rows_to_delete) >= 2:
                ids_to_delete = tuple(row[0] for row in rows_to_delete)
                cur.execute(
                    "DELETE FROM chat_messages WHERE id IN %s;",
                    (ids_to_delete,)
                )
                cur.execute("""
                            UPDATE chat_sessions
                            SET updated_at = (SELECT timestamp
                            FROM chat_messages
                            WHERE session_id = %s
                            ORDER BY message_order DESC
                                LIMIT 1
                                )
                            WHERE id = %s;
                            """, (session_id, session_id))

                conn.commit()
                print(f"[Thông báo] Đã tua lại lượt nói cuối cùng cho phiên {session_id}")
                success = True
            else:
                print("[Cảnh báo] Không có đủ tin nhắn để thực hiện thao tác sửa.")
                conn.rollback()

    except Exception as e:
        print(f"[Lỗi] Không thể tua lại phiên: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

    return success

