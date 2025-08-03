import unittest
from src.core.llm import get_llm

class TestLLM(unittest.TestCase):
    def test_llm_load(self):
        llm = get_llm()
        self.assertIsNotNone(llm)

if __name__ == "__main__":
    unittest.main()