"""Unit test ensuring scripts/verify_isolated.py syntax and integrity."""
import unittest
import py_compile
from pathlib import Path

class TestVerifyIsolatedScript(unittest.TestCase):
    def test_verify_isolated_syntax(self):
        script_path = Path("scripts/verify_isolated.py")
        self.assertTrue(script_path.is_file(), "scripts/verify_isolated.py must exist")
        # Ensure it compiles without syntax errors
        py_compile.compile(str(script_path), doraise=True)

if __name__ == "__main__":
    unittest.main()
