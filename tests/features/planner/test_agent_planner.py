import unittest
from src.features.planner.agent import initialize_planning_agent

class TestPlannerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_planning_agent()

    def test_agent_init(self):
        self.assertIsNotNone(self.agent)

    def test_agent_invoke(self):
        if self.agent:
            input_data = {
                "user_id": "test_user",
                "input": "Tạo lộ trình học tiếng Nhật từ N5 đến N3",
                "chat_history": []
            }
            try:
                result = self.agent.invoke(input_data)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Agent invoke raised exception: {e}")

if __name__ == "__main__":
    unittest.main()