# src/core/context_manager.py

import json
from typing import Optional, Dict, Any

# Giả định bạn có file quản lý DB tập trung
from .database import get_db_connection


def save_task_context(session_id: int, intent_name: str, status: str, context_data: Dict[str, Any]):
    """
    Lưu hoặc cập nhật trạng thái của một nhiệm vụ vào database.
    Sử dụng ON CONFLICT để tự động UPDATE nếu đã tồn tại.
    """
    conn = get_db_connection()
    if not conn: return

    try:
        with conn.cursor() as cur:
            query = """
                    INSERT INTO task_contexts (session_id, intent_name, status, context_data, updated_at)
                    VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT (session_id, intent_name) DO \
                    UPDATE SET
                        status = EXCLUDED.status, \
                        context_data = EXCLUDED.context_data, \
                        updated_at = NOW(); \
                    """
            cur.execute(query, (session_id, intent_name, status, json.dumps(context_data)))
            conn.commit()
            print(f"[Context Manager] Đã lưu context cho session {session_id}, intent {intent_name}")
    except Exception as e:
        print(f"[Lỗi] Không thể lưu task context: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()


def load_task_context(session_id: int, intent_name: str) -> Optional[Dict[str, Any]]:
    """
    Tải trạng thái của một nhiệm vụ cụ thể từ database.
    """
    conn = get_db_connection()
    if not conn: return None

    context = None
    try:
        with conn.cursor() as cur:
            query = """
                    SELECT status, context_data
                    FROM task_contexts
                    WHERE session_id = %s \
                      AND intent_name = %s; \
                    """
            cur.execute(query, (session_id, intent_name))
            result = cur.fetchone()
            if result:
                status, context_data = result
                context = {"status": status, "data": context_data or {}}
                print(f"[Context Manager] Đã tải context cho session {session_id}, intent {intent_name}")
    except Exception as e:
        print(f"[Lỗi] Không thể tải task context: {e}")
    finally:
        if conn: conn.close()

    return context


def clear_task_context(session_id: int, intent_name: str):
    """
    Xóa trạng thái của một nhiệm vụ sau khi nó đã hoàn thành.
    """
    conn = get_db_connection()
    if not conn: return

    try:
        with conn.cursor() as cur:
            query = "DELETE FROM task_contexts WHERE session_id = %s AND intent_name = %s;"
            cur.execute(query, (session_id, intent_name))
            conn.commit()
            print(f"[Context Manager] Đã xóa context cho session {session_id}, intent {intent_name}")
    except Exception as e:
        print(f"[Lỗi] Không thể xóa task context: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()