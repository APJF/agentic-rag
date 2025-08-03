import unittest
import src.data_processing.vocabulary_parser as vocabulary_parser

class TestVocabularyParser(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(vocabulary_parser)
    def test_parse_vocabulary(self):
        if hasattr(vocabulary_parser, 'parse_vocabulary'):
            result = vocabulary_parser.parse_vocabulary('')
            self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()