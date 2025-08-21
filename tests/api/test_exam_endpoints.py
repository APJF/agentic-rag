import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app


class TestExamEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.api.endpoints.reviewer.execute_sql_query")
    def test_exam_overview_ok(self, mock_exec):
        # Mock sequence of DB calls for _get_exam_id, _get_totals, _get_section_rows, and UPDATE
        # 1) SELECT exam_id FROM exam_result WHERE id = %s;
        # 2) big SELECT for totals
        # 3) section rows SELECT
        # 4) UPDATE exam_result SET advice = %s WHERE id = %s RETURNING id;
        mock_exec.side_effect = [
            [{"exam_id": 9}],
            [{"total": 10, "correct": 7, "wrong": 3}],
            [
                {"section": "KANJI", "total": 4, "correct": 2, "wrong": 2},
                {"section": "GRAMMAR", "total": 3, "correct": 3, "wrong": 0},
                {"section": "VOCAB", "total": 3, "correct": 2, "wrong": 1},
            ],
            [{"id": 1}],
        ]

        resp = self.client.post("/api/exam/overview", json={"exam_result_id": 123})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["exam_result_id"], 123)
        self.assertIn("advice", data)
        self.assertIn("summary", data["advice"])

    @patch("src.api.endpoints.reviewer.execute_sql_query")
    def test_exam_overview_not_found(self, mock_exec):
        # exam_result không có
        mock_exec.side_effect = [
            [],  # _get_exam_id -> 404
        ]
        resp = self.client.post("/api/exam/overview", json={"exam_result_id": 999})
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()


