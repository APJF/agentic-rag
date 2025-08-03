import unittest
import src.config.settings as settings

class TestConfigSettings(unittest.TestCase):
    def test_settings_exist(self):
        self.assertTrue(hasattr(settings, '__file__') or hasattr(settings, '__doc__'))

if __name__ == "__main__":
    unittest.main()