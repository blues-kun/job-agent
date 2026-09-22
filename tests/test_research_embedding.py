"""训练数据隔离、假负例屏蔽、官方池化的必要回归验证；无 GPU。"""
import json
import tempfile
import unittest
from pathlib import Path

from research.train_embedding import prepare_data, masks_for_batch, last_token_pool


def row(index, split, family=None, title=None, description=None):
    return {"job_id": str(index), "split": split, "job_family_id": family or str(index),
            "title": title or f"职位{index}", "category": "技术",
            "requirements": f"岗位 {index} 要求独立分析业务需求并完成规范化数据处理工作。",
            "description": description or f"负责第 {index} 类业务的工程建设与代码质量检查。"}


class DataTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.jsonl"
            path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in data), encoding="utf-8")
            return prepare_data(path)

    def test_family_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "分区泄漏"):
            self.load([row(1, "train", "same"), row(2, "dev", "same")])

    def test_duplicate_content_is_rejected(self):
        a, b = row(1, "train"), row(2, "dev")
        b["requirements"], b["description"] = a["requirements"], a["description"]
        with self.assertRaisesRegex(ValueError, "分区泄漏"):
            self.load([a, b])

    def test_query_holdout_retains_candidate_documents(self):
        data, stats = self.load([row(1, "train", title="数据分析"), row(2, "dev", title="数据分析"),
                                 row(3, "dev", title="机器学习"), row(4, "test", title="机器学习"),
                                 row(5, "test", title="系统开发")])
        self.assertEqual(len(data["corpora"]["dev"]), 2)
        self.assertEqual(len(data["queries"]["dev"]), 1)
        self.assertEqual(len(data["queries"]["test"]), 1)
        self.assertEqual(stats["family_overlap"], 0)

    def test_no_cross_family_negative_for_same_query(self):
        import torch
        batch = [{"query_key": "a", "family": "f1"}, {"query_key": "a", "family": "f2"},
                 {"query_key": "b", "family": "f1"}]
        positive, allowed, informative = masks_for_batch(batch, torch, "cpu")
        self.assertTrue(bool(positive[0, 1]))
        self.assertFalse(bool(allowed[0, 2]))
        self.assertEqual(informative.tolist(), [False, True, True])

    def test_last_token_pool_left_and_right_padding(self):
        import torch
        hidden = torch.arange(12).reshape(2, 3, 2).float()
        left = last_token_pool(hidden, torch.tensor([[0, 1, 1], [1, 1, 1]]), torch)
        right = last_token_pool(hidden, torch.tensor([[1, 1, 0], [1, 1, 1]]), torch)
        torch.testing.assert_close(left, hidden[:, -1])
        torch.testing.assert_close(right, torch.stack([hidden[0, 1], hidden[1, 2]]))


if __name__ == "__main__":
    unittest.main()
