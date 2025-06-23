# src/core/session_manager.py

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import psycopg2

# Chúng ta sẽ giả định bạn đã có một file quản lý kết nối DB tập trung
# Nếu chưa, bạn có thể tạo file src/core/database.py và chuyển hàm get_db_connection vào đó
from .database import get_db_connection


def get_or_create_user(user_id: str, display_name: str = None) -> bool:
    """Kiểm tra user_id có tồn tại không, nếu không thì tạo mới."""
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            # Dùng ON CONFLICT DO NOTHING để tránh lỗi nếu user đã tồn tại
            cur.execute(
                "INSERT INTO users (user_id, display_name) VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING;",
                (user_id, display_name or user_id)
            )
            conn.commit()
            return True
    except psycopg2.Error as e:
        print(f"Lỗi khi get/create user: {e}")
        conn.rollback()
        return False
    finally:
        if conn: conn.close()


def list_sessions_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Lấy danh sách các phiên làm việc của một người dùng."""
    conn = get_db_connection()
    if not conn: return []
    sessions = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, session_name, updated_at FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC;",
                (user_id,)
            )
            rows = cur.fetchall()
            for row in rows:
                sessions.append({"id": row[0], "name": row[1], "updated_at": row[2]})
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