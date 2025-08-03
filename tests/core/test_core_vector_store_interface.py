import unittest
import src.core.vector_store_interface as vsi

class TestVectorStoreInterface(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(vsi)
    def test_get_db_connection(self):
        if hasattr(vsi, 'get_db_connection'):
            conn = vsi.get_db_connection()
            self.assertTrue(conn is None or hasattr(conn, 'cursor'))

if __name__ == "__main__":
    unittest.main()