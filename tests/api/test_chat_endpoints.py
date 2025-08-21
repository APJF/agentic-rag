import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app


class TestChatEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.api.endpoints.messages.update_session_context")
    @patch("src.api.endpoints.messages.add_new_messages")
    @patch("src.api.endpoints.messages.load_session_data")
    @patch("src.api.endpoints.messages.initialize_qna_agent")
    def test_create_message_qna(self, mock_init_qna, mock_load, mock_add, mock_update_ctx):
        mock_load.return_value = {
            "user_id": 1,
            "type": "qna",
            "context": {},
            "history": []
        }

        class FakeAgent:
            def invoke(self, input_data):
                return {"output": "Trả lời từ QnA"}

        mock_init_qna.return_value = FakeAgent()

        payload = {"session_id": 1, "user_input": "Xin chào"}
        resp = self.client.post("/api/messages/", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], 1)
        self.assertIn("ai_response", data)


if __name__ == "__main__":
    unittest.main()


