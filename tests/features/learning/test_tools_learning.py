import unittest
import src.features.learning.tools as learning_tools

class TestLearningTools(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(learning_tools)

if __name__ == "__main__":
    unittest.main()