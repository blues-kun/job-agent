"""虚构数据的路径保护、字段兼容及当前检索可用性。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/create_demo_data.py"
spec = importlib.util.spec_from_file_location("publication_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # /tmp在当前机器受外层Git管理，试验工作簿只写入仓库外私有缓存。
        root = Path.home() / ".cache/job-agent/publication-tests"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        cls.temporary = tempfile.TemporaryDirectory(prefix="demo-", dir=root)
        cls.root = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_fictional_workbook_loads_in_current_corpus(self):
        from job_agent.corpus import Corpus
        from job_agent.workflow import Workflow
        output = self.root / "compatible.xlsx"
        result = demo.create_demo_data(output)
        self.assertEqual(result["jobs"], 16)
        book = load_workbook(output, read_only=True)
        try:
            rows = list(book.active.values)
            self.assertEqual(list(rows[0]), demo.HEADERS)
            self.assertTrue(all(row[-1] == demo.DISCLAIMER for row in rows[1:]))
        finally:
            book.close()
        corpus = Corpus(output)
        self.assertEqual(len(corpus.jobs), 16)
        self.assertEqual(corpus.duplicates, 0)
        self.assertTrue(any(job.salary["status"] != "月薪" for job in corpus.jobs))
        self.assertTrue(any(group["kind"] == "any" for job in corpus.jobs for group in job.groups))
        with patch.dict("os.environ", {"JOB_AGENT_RANKER_DIR": "", "JOB_AGENT_COACH_URL": ""}):
            result = Workflow(corpus).recommend("本科应届生。我使用Python开发订单接口，使用SQL查询订单，编写接口测试并完成项目文档。",
                    {"city": "深圳", "intent": "Python后端开发", "education": "本科", "experience_years": 0})
        self.assertEqual(result["action"], "recommend")
        self.assertGreater(len(result["jobs"]), 0)
        self.assertTrue(all(row["evidence_verified"] for row in result["jobs"]))

    def test_rejects_relative_repository_and_existing_targets(self):
        with self.assertRaises(ValueError):
            demo.create_demo_data(Path("relative.xlsx"))
        with self.assertRaises(ValueError):
            demo.create_demo_data(demo.REPOSITORY_ROOT / "inside.xlsx")
        other = self.root / "another-repo"
        other.mkdir(); (other / ".git").mkdir()
        with self.assertRaises(ValueError):
            demo.create_demo_data(other / "inside.xlsx")
        target = self.root / "existing.xlsx"
        target.write_bytes(b"do-not-change")
        with self.assertRaises(FileExistsError):
            demo.create_demo_data(target)
        self.assertEqual(target.read_bytes(), b"do-not-change")

    def test_rejects_target_and_parent_symlinks(self):
        destination = self.root / "destination.xlsx"
        destination.write_bytes(b"untouched")
        link = self.root / "linked.xlsx"
        parent_link = self.root / "linked-directory"
        try:
            link.symlink_to(destination)
            parent_link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("当前系统不允许创建测试符号链接")
        with self.assertRaises(ValueError):
            demo.create_demo_data(link)
        with self.assertRaises(ValueError):
            demo.create_demo_data(parent_link / "new.xlsx")
        self.assertEqual(destination.read_bytes(), b"untouched")


if __name__ == "__main__":
    unittest.main()
