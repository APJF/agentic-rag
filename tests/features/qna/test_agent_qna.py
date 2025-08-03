import unittest
from src.features.qna.agent import initialize_qna_agent

class TestQnAAgent(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_qna_agent()

    def test_agent_init(self):
        self.assertIsNotNone(self.agent)

    def test_agent_invoke(self):
        if self.agent:
            input_data = {
                "user_id": "test_user",
                "input": "Xin chào, hãy dịch câu này sang tiếng Nhật",
                "chat_history": []
            }
            try:
                result = self.agent.invoke(input_data)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Agent invoke raised exception: {e}")

if __name__ == "__main__":
    unittest.main()