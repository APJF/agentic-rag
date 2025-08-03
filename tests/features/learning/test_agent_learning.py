import unittest
from src.features.learning.agent import initialize_learning_agent

class TestLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_learning_agent()

    def test_agent_init(self):
        self.assertIsNotNone(self.agent)

    def test_agent_invoke(self):
        if self.agent:
            input_data = {
                "user_id": "test_user",
                "input": "Giải thích ngữ pháp bài 1",
                "chat_history": [],
                "context": {"material_id": "M101"}
            }
            try:
                result = self.agent.invoke(input_data)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Agent invoke raised exception: {e}")

if __name__ == "__main__":
    unittest.main()