"""只用完全虚构小图验证泄漏边界、对照和数值安全，不加载模型或真实简历。"""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from research.train_graph import Edges, GraphEncoder, alignment_loss, build_edges, file_hash, load_inputs, masked_logits, randomize_edges


def fixture_jobs():
    jobs = []
    for index, split in enumerate(["train", "train", "train", "train", "dev", "dev", "test", "test"]):
        names = ["Python", "SQL"] if index % 2 == 0 else ["Java", "SQL"]
        jobs.append({"job_id": f"{index:020d}", "legacy_id": "不参与特征映射", "job_family_id": f"family-{index}",
                     "snapshot": "完全虚构烟测", "split": split, "title": "虚构岗位", "category": "虚构职类", "category_parent": "技术",
                     "requirements": "虚构要求", "description": "虚构职责",
                     "skills": {name: {"level": "熟悉", "field": "requirements"} for name in names},
                     "groups": [{"kind": "any", "skills": names, "preferred": False}]})
    return jobs


class ResearchGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)
        self.jobs = fixture_jobs()
        self.names = ["Python", "Java", "SQL"]
        self.edges, _ = build_edges(self.jobs, self.names)
        self.text = F.normalize(torch.randn(8, 8), dim=-1)
        self.skills = F.normalize(torch.randn(3, 8), dim=-1)
        self.train = torch.tensor([True] * 4 + [False] * 4)

    def test_unseen_jobs_cannot_change_shared_train_states(self):
        model = GraphEncoder(8, hidden=16, dropout=0).eval()
        before = model(self.text, self.skills, self.edges, self.train)
        changed = self.text.clone()
        changed[~self.train] = 1000 * torch.randn_like(changed[~self.train])
        after = model(changed, self.skills, self.edges, self.train)
        torch.testing.assert_close(before["graph_vectors"][self.train], after["graph_vectors"][self.train], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(before["skill_graph_vectors"], after["skill_graph_vectors"], atol=1e-6, rtol=1e-6)

    def test_train_loss_has_no_gradient_through_unseen_text(self):
        model = GraphEncoder(8, hidden=16, dropout=0)
        vectors = self.text.detach().clone().requires_grad_()
        result = model(vectors, self.skills, self.edges, self.train)
        result["aligned_vectors"][self.train, 0].sum().backward()
        self.assertEqual(float(vectors.grad[~self.train].abs().sum()), 0)
        self.assertGreater(float(vectors.grad[self.train].abs().sum()), 0)

    def test_near_duplicate_family_is_not_negative(self):
        student = torch.tensor([[1.0, 0.0]])
        teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        families = torch.tensor([7, 7, 8])
        logits, allowed = masked_logits(student, teacher, families[:1], families, torch.tensor([0]), 1.0)
        self.assertEqual(allowed.tolist(), [[True, False, True]])
        self.assertTrue(torch.isneginf(logits[0, 1]))
        loss, count = alignment_loss(student, teacher, families[:1], families, torch.tensor([0]), 1.0)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(loss), float(torch.log1p(torch.exp(torch.tensor(-1.0)))), places=6)

    def test_random_edges_preserve_weighted_degrees_and_train_independence(self):
        splits = [job["split"] for job in self.jobs]
        shuffled = randomize_edges(self.edges, splits, 42)
        def degrees(edges):
            out = defaultdict(float)
            for job, skill, relation, weight in zip(edges.jobs.tolist(), edges.skills.tolist(), edges.relations.tolist(), edges.weights.tolist()):
                out[(splits[job], "job", job, relation)] += weight
                out[(splits[job], "skill", skill, relation)] += weight
            return dict(out)
        self.assertEqual(degrees(self.edges), degrees(shuffled))
        self.assertTrue(bool((self.edges.skills != shuffled.skills).any()))
        train_edges = self.edges.subset(self.train[self.edges.jobs])
        train_shuffled = randomize_edges(train_edges, splits[:4], 42)
        torch.testing.assert_close(shuffled.skills[self.train[shuffled.jobs]], train_shuffled.skills)

    def test_empty_neighbours_are_exact_text_fallback(self):
        empty = Edges(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0))
        for mode in ("graphsage", "pool_mlp"):
            model = GraphEncoder(8, hidden=16, mode=mode, dropout=0).eval()
            output = model(self.text, self.skills, empty, self.train)
            expected = F.normalize(model.job_projection(self.text), dim=-1)
            self.assertTrue(bool(torch.isfinite(output["graph_vectors"]).all()))
            torch.testing.assert_close(output["graph_vectors"], expected)

    def test_cross_split_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            bad_jobs = fixture_jobs()
            bad_jobs[-1]["job_family_id"] = bad_jobs[0]["job_family_id"]
            (path / "jobs.jsonl").write_text("\n".join(json.dumps(job) for job in bad_jobs), encoding="utf-8")
            (path / "manifest.json").write_text(json.dumps({"snapshot": "完全虚构烟测", "jobs_sha256": file_hash(path / "jobs.jsonl")}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "跨分区"):
                load_inputs(path, path / "not-needed.npz")

    def test_source_and_feature_hash_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs_file = root / "jobs.jsonl"
            jobs_file.write_text("\n".join(json.dumps(job) for job in self.jobs), encoding="utf-8")
            manifest = {"source_sha256": "完全虚构烟测", "config": {"snapshot": "完全虚构烟测"},
                        "files": {"jobs.jsonl": {"sha256": file_hash(jobs_file)}}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            features = root / "features.npz"
            np.savez(features, job_ids=np.array([job["job_id"] for job in self.jobs]), text_vectors=self.text.numpy(),
                     skills=np.array(self.names), skill_vectors=self.skills.numpy())
            feature_manifest = {"jobs_sha256": file_hash(jobs_file), "features_sha256": file_hash(features)}
            features.with_suffix(".manifest.json").write_text(json.dumps(feature_manifest), encoding="utf-8")
            loaded = load_inputs(root, features)
            self.assertEqual(loaded[-1]["snapshot"], "完全虚构烟测")
            feature_manifest["features_sha256"] = "tampered"
            features.with_suffix(".manifest.json").write_text(json.dumps(feature_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_inputs(root, features)
            manifest["config"]["snapshot"] = "another-snapshot"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot/source_sha256"):
                load_inputs(root, features)

    def test_three_modes_export_real_smoke_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"; dataset.mkdir()
            (dataset / "jobs.jsonl").write_text("\n".join(json.dumps(job) for job in self.jobs), encoding="utf-8")
            (dataset / "manifest.json").write_text(json.dumps({"source_sha256": "完全虚构烟测", "files": {"jobs.jsonl": {"sha256": file_hash(dataset / "jobs.jsonl")}}}), encoding="utf-8")
            np.savez(root / "features.npz", job_ids=np.array([job["job_id"] for job in self.jobs]),
                     text_vectors=self.text.numpy(), skills=np.array(self.names), skill_vectors=self.skills.numpy())
            (root / "features.manifest.json").write_text(json.dumps({"jobs_sha256": file_hash(dataset / "jobs.jsonl"), "features_sha256": file_hash(root / "features.npz")}), encoding="utf-8")
            for mode in ("graphsage", "pool_mlp", "random_edges"):
                output = root / mode
                run = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "research/train_graph.py"),
                    "--dataset", str(dataset), "--features", str(root / "features.npz"), "--output", str(output),
                    "--mode", mode, "--epochs", "2", "--hidden", "16", "--batch-size", "2",
                    "--max-steps-per-epoch", "1", "--cpu-threads", "2"], capture_output=True, text=True, timeout=45)
                self.assertEqual(run.returncode, 0, run.stderr)
                metrics = json.loads((output / "metrics.json").read_text("utf-8"))
                self.assertTrue(metrics["not_person_job_evaluation"])
                self.assertNotIn("test", metrics["before_training"])
                self.assertIn("test", metrics["selected_checkpoint"])
                self.assertEqual(len(metrics["history"]), 2)
                self.assertTrue(all("test" not in epoch for epoch in metrics["history"]))
                with np.load(output / "graph_vectors.npz", allow_pickle=False) as archive:
                    self.assertEqual(archive["graph_vectors"].shape, (8, 16))
                    self.assertTrue(np.isfinite(archive["graph_vectors"]).all())
                self.assertTrue((output / "model.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
