import unittest
from src.features.planner.tools import calculate_time_constraints

class TestPlannerTools(unittest.TestCase):
    def test_calculate_time_constraints_month(self):
        result = calculate_time_constraints("tháng 12")
        self.assertIn("target_date_iso", result)
        self.assertIn("days_remaining", result)
        self.assertGreater(result["days_remaining"], 0)

    def test_calculate_time_constraints_full(self):
        result = calculate_time_constraints("tháng 7 năm 2025")
        self.assertIn("target_date_iso", result)
        self.assertIn("days_remaining", result)

    def test_calculate_time_constraints_invalid(self):
        result = calculate_time_constraints("ngày không hợp lệ")
        self.assertIn("error", result)

if __name__ == "__main__":
    unittest.main()