import unittest
import src.data_processing.pdf_document_processor as pdf_document_processor

class TestPDFDocumentProcessor(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(pdf_document_processor)
    def test_parse_pdf(self):
        if hasattr(pdf_document_processor, 'parse_pdf'):
            result = pdf_document_processor.parse_pdf('')
            self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()