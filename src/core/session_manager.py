import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import psycopg2
from .database import get_db_connection


def get_user(user_id: str) -> bool:
    user_id_int = int(user_id)
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            query = 'SELECT 1 FROM "users" WHERE id = %s;'
            cur.execute(query, (user_id_int,))
            exists = cur.fetchone() is not None
            return exists
    except psycopg2.Error as e:
        print(f"Lỗi khi kiểm tra user: {e}")
        return False
    finally:
        if conn: conn.close()


def list_sessions_for_user(user_id: str) -> List[Dict[str, Any]]:
    user_id_int = int(user_id)
    conn = get_db_connection()
    if not conn: return []
    sessions = []
    try:
        with conn.cursor() as cur:
            query = "SELECT id, name, type, created_at, updated_at FROM session WHERE user_id = %s ORDER BY updated_at DESC;"
            cur.execute(query, (user_id_int,))
            rows = cur.fetchall()
            for row in rows:
                sessions.append({"id": row[0], "session_name": row[1], "type": row[2], "created_at": row[3], "updated_at": row[4]})
    except psycopg2.Error as e:
        print(f"Lỗi khi liệt kê các phiên: {e}")
    finally:
        if conn: conn.close()
    return sessions


def create_new_session(
        user_id: str,
        session_name: str,
        session_type: str = 'GENERAL',
        context: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    print(f"[DEBUG] Bắt đầu tạo session cho user_id={user_id}, session_name={session_name}, session_type={session_type}")
    try:
        user_id_int = int(user_id)
    except Exception as e:
        print(f"[ERROR] Lỗi ép kiểu user_id: {user_id} -> {e}")
        return None

    conn = get_db_connection()
    if not conn:
        print("[ERROR] Không kết nối được database!")
        return None
    session_id = None
    try:
        with conn.cursor() as cur:
            context_json = json.dumps(context) if context else None
            query = """
                    INSERT INTO session (user_id, name, type, context)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                    """
            cur.execute(query, (user_id_int, session_name, session_type.upper(), context_json))
            session_id = cur.fetchone()[0]
            conn.commit()
            print(f"[INFO] Đã tạo phiên '{session_name}' (Loại: {session_type}) với ID: {session_id}")
    except Exception as e:
        print(f"[ERROR] Lỗi khi tạo phiên mới: {e}")
        if hasattr(e, 'pgerror'):
            print(f"[ERROR] pgerror: {e.pgerror}")
        if hasattr(e, 'diag') and getattr(e.diag, 'message_detail', None):
            print(f"[ERROR] Chi tiết: {e.diag.message_detail}")
        if conn:
            conn.rollback()
    finally:
        print("[DEBUG] Đóng kết nối database sau khi tạo session.")
        if conn: conn.close()
    if session_id is None:
        print("[ERROR] create_new_session trả về None!")
    return session_id


def load_session_data(session_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if not conn: return None

    session_data = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, type, context, created_at FROM session WHERE id = %s;",
                (session_id,)
            )
            session_info = cur.fetchone()
            if not session_info:
                print(f"[Lỗi] Không tìm thấy session với ID {session_id}")
                return None

            user_id, session_type, context, created_at = session_info

            history = []
            cur.execute(
                "SELECT type, content FROM message WHERE session_id = %s ORDER BY \"order\" ASC;",
                (session_id,)
            )
            for row in cur.fetchall():
                msg_type, content = row
                if msg_type == 'human':
                    history.append(HumanMessage(content=content))
                else:
                    history.append(AIMessage(content=content))

            session_data = {
                "user_id": user_id,
                "type": session_type,
                "context": context or {},
                "history": history,
                "created_at": created_at
            }
            print(f"[Thông báo] Đã tải thành công dữ liệu cho phiên {session_id} (Loại: {session_type})")

    except psycopg2.Error as e:
        print(f"Lỗi] Không thể tải dữ liệu phiên: {e}")
    finally:
        if conn: conn.close()

    return session_data


def load_chat_history(session_id: int) -> List[BaseMessage]:
    conn = get_db_connection()
    if not conn: return []
    history = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT type, content FROM message WHERE session_id = %s ORDER BY \"order\" ASC;",
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
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(\"order\"), 0) FROM message WHERE session_id = %s;",
                        (session_id,))
            last_order = cur.fetchone()[0]

            for i, msg in enumerate(new_messages):
                message_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
                cur.execute(
                    "INSERT INTO message (session_id, type, content, \"order\") VALUES (%s, %s, %s, %s);",
                    (session_id, message_type, msg.content, last_order + i + 1)
                )

            cur.execute("UPDATE session SET updated_at = NOW() WHERE id = %s;", (session_id,))
            conn.commit()
    except psycopg2.Error as e:
        print(f"Lỗi khi thêm tin nhắn mới: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()


def delete_session(session_id: int) -> bool:
    conn = get_db_connection()
    if not conn: return False

    deleted_rows = 0
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM session WHERE id = %s;", (session_id,))
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


def rename_session(session_id: int, new_name: str) -> bool:
    conn = get_db_connection()
    if not conn: return False

    updated_rows = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE session SET name = %s, updated_at = NOW() WHERE id = %s;",
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
    conn = get_db_connection()
    if not conn:
        print("[Lỗi] Không thể kết nối DB để tua lại phiên.")
        return False

    success = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM message WHERE session_id = %s ORDER BY \"order\" DESC LIMIT 2;",
                (session_id,)
            )
            rows_to_delete = cur.fetchall()

            if len(rows_to_delete) >= 2:
                ids_to_delete = tuple(row[0] for row in rows_to_delete)
                cur.execute(
                    "DELETE FROM message WHERE id IN %s;",
                    (ids_to_delete,)
                )
                cur.execute("""
                            UPDATE session
                            SET updated_at = (SELECT timestamp
                            FROM message
                            WHERE session_id = %s
                            ORDER BY \"order\" DESC
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


def find_session(
        user_id: int,
        session_type: str,
        context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if not conn: return None

    session_info = None
    try:
        with conn.cursor() as cur:
            query = """
                    SELECT id, name, updated_at
                    FROM "session"
                    WHERE user_id = %s \
                      AND type = %s \
                    """
            params = [user_id, session_type.upper()]

            if context:
                query += " AND context @> %s"
                params.append(json.dumps(context))

            query += " ORDER BY updated_at DESC LIMIT 1;"

            cur.execute(query, tuple(params))
            result = cur.fetchone()

            if result:
                session_info = {
                    "id": result[0],
                    "session_name": result[1],
                    "updated_at": result[2]
                }
                print(
                    f"[Session Manager] Đã tìm thấy phiên '{session_type}' tồn tại cho user '{user_id}' với ID: {result[0]}")

    except psycopg2.Error as e:
        print(f"[Lỗi] Không thể tìm phiên: {e}")
    finally:
        if conn: conn.close()

    return session_info


def update_session_context(session_id: int, updates: Dict[str, Any]) -> bool:
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE session
                SET context = COALESCE(context, '{}'::jsonb) || %s::jsonb,
                    updated_at = NOW()
                WHERE id = %s;
                """,
                (json.dumps(updates), session_id)
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        print(f"[Lỗi] update_session_context: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
