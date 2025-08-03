import unittest
from src.core.embedding import get_embedding_model

class TestEmbeddingModel(unittest.TestCase):
    def test_model_load(self):
        model = get_embedding_model()
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()