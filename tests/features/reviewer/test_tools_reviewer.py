import unittest
import src.features.reviewer.tools as reviewer_tools

class TestReviewerTools(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(reviewer_tools)

if __name__ == "__main__":
    unittest.main()