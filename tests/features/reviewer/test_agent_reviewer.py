import unittest
from src.features.reviewer.agent import initialize_reviewer_agent

class TestReviewerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_reviewer_agent()

    def test_agent_init(self):
        self.assertIsNotNone(self.agent)

    def test_agent_invoke(self):
        if self.agent:
            input_data = {
                "context": {"exam_result_id": "EXAM123"}
            }
            try:
                result = self.agent.invoke(input_data)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"Agent invoke raised exception: {e}")

if __name__ == "__main__":
    unittest.main()