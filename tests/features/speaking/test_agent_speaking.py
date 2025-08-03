import unittest
from src.features.speaking.agent import initialize_speaking_agent

class TestSpeakingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_speaking_agent()

    def test_agent_init(self):
        self.assertIsNotNone(self.agent)

    def test_agent_invoke(self):
        if self.agent:
            input_data = {
                "user_id": "test_user",
                "input": "Luyện nói chủ đề du lịch",
                "chat_history": [],
                "level": "N3"
            }
            try:
                result = self.agent.invoke(input_data)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Agent invoke raised exception: {e}")

if __name__ == "__main__":
    unittest.main()