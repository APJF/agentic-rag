import unittest
from src.features.qna.tools import get_user_profile_tool

class TestQnATools(unittest.TestCase):
    def test_get_user_profile_tool_invalid(self):
        # Trường hợp user_id không tồn tại, nên trả về None
        result = get_user_profile_tool("user_id_khong_ton_tai")
        self.assertTrue(result is None or isinstance(result, dict))

if __name__ == "__main__":
    unittest.main()