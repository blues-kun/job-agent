import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTests(unittest.TestCase):
    def test_env_example_has_no_secret(self):
        env_example = ROOT / ".env.example"
        self.assertTrue(env_example.is_file())

        values = {}
        for raw_line in env_example.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key] = value

        self.assertIn("API_KEY", values)
        self.assertEqual(values["API_KEY"], "")
        self.assertFalse(
            any(re.search(r"(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}", value) for value in values.values())
        )

    def test_local_env_is_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", gitignore)
        self.assertIn("!.env.example", gitignore)


if __name__ == "__main__":
    unittest.main()
