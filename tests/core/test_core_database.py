import unittest
import src.core.database as database

class TestDatabase(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(database)
    def test_execute_sql_query(self):
        if hasattr(database, 'execute_sql_query'):
            result = database.execute_sql_query('SELECT 1')
            self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()