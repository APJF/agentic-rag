import unittest
from src.utils.storage_client import StorageClient

class TestStorageClient(unittest.TestCase):
    def test_client_init(self):
        client = StorageClient()
        self.assertIsNotNone(client)

if __name__ == "__main__":
    unittest.main()