import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app


class TestSessionsEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.api.endpoints.sessions.initialize_qna_agent")
    @patch("src.api.endpoints.sessions.add_new_messages")
    @patch("src.api.endpoints.sessions.create_new_session")
    @patch("src.api.endpoints.sessions.get_user")
    def test_create_session_qna_basic(self, mock_get_user, mock_create, mock_add_msgs, mock_init_qna):
        mock_get_user.return_value = True
        mock_create.return_value = 123
        mock_add_msgs.return_value = None

        class FakeAgent:
            def invoke(self, input_data):
                return {"output": "Xin chào!"}

        mock_init_qna.return_value = FakeAgent()

        payload = {
            "user_id": 1,
            "first_message": "Xin chào",
            "session_type": "qna",
            "context": {}
        }
        resp = self.client.post("/api/sessions/", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], 123)
        self.assertIn("ai_first_response", data)

    @patch("src.api.endpoints.sessions.list_sessions_for_user")
    def test_list_sessions(self, mock_list):
        from datetime import datetime, timezone
        mock_list.return_value = [
            {
                "id": 10,
                "session_name": "Phiên A",
                "name": "Phiên A",
                "type": "qna",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        ]
        resp = self.client.get("/api/sessions", params={"user_id": "1"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["user_id"], "1")
        self.assertEqual(len(body["sessions"]), 1)
        self.assertEqual(body["sessions"][0]["id"], 10)

    @patch("src.api.endpoints.sessions.get_db_connection")
    def test_get_session_detail(self, mock_conn):
        # Fake cursor/connection
        class FakeCursor:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def execute(self, sql, params=None):
                self._last_sql = sql
                self._last_params = params
                # For column detection
                if "information_schema.columns" in sql:
                    self._rows = [("created_at",)]
                # Fetch messages with created_at
                elif "FROM message" in sql and "ORDER BY \"order\" ASC" in sql and "id, type" in sql:
                    self._rows = [
                        (1, "human", "Hi", 1, None),
                        (2, "ai", "Hello", 2, None),
                    ]
                else:
                    # session exists check is not required by endpoint directly
                    self._rows = []
            def fetchall(self):
                return list(self._rows)
            def fetchone(self):
                try:
                    return self._rows[0]
                except Exception:
                    return None

        class FakeConn:
            def cursor(self):
                return FakeCursor()
            def close(self):
                pass

        mock_conn.return_value = FakeConn()

        resp = self.client.get("/api/sessions/42")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], 42)
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0]["type"], "human")

    @patch("src.api.endpoints.sessions.get_session_info")
    @patch("src.api.endpoints.sessions.update_session_context")
    @patch("src.api.endpoints.sessions.rename_session")
    def test_put_and_patch_session(self, mock_rename, mock_update_ctx, mock_get_info):
        from datetime import datetime, timezone
        mock_rename.return_value = True
        mock_update_ctx.return_value = True
        mock_get_info.return_value = {
            "id": 55,
            "name": "New name",
            "type": "qna",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        payload = {"name": "New name", "context": {"x": 1}}
        r1 = self.client.put("/api/sessions/55", json=payload)
        self.assertEqual(r1.status_code, 200)
        r2 = self.client.patch("/api/sessions/55", json=payload)
        self.assertEqual(r2.status_code, 200)

    @patch("src.api.endpoints.sessions.delete_session")
    def test_delete_session(self, mock_delete):
        mock_delete.return_value = True
        resp = self.client.delete("/api/sessions/77")
        self.assertEqual(resp.status_code, 204)


if __name__ == "__main__":
    unittest.main()


