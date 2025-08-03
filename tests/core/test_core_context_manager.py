import unittest
import src.core.context_manager as context_manager

class TestContextManager(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(context_manager)

if __name__ == "__main__":
    unittest.main()