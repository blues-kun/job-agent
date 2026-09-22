"""要求组作用域、归纳状态隔离、标签与未知负例门禁的必要回归检查。"""
import copy
import unittest

import pytest
torch = pytest.importorskip("torch")

from research.model_fixtures import counterexample_asts, group, leaf, make_fixtures
from research.group_graph import (FrozenTextFeatures, GroupGraphRanker, build_graph,
    degree_preserving_randomization, degree_signature, logic_baseline, validate_supervision)
from research.train_person_job_embedding import reviewed_masks, multipositive_loss, prepare_rendering, load_module, segment_scores


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def graphs(self):
        first, second = counterexample_asts()
        return [build_graph({"job_id": "same-id", "title": "开发岗位", "requirement_ast": ast}) for ast in (first, second)]

    def test_two_or_scopes_have_different_graph_representations(self):
        a, b = self.graphs()
        model = GroupGraphRanker(FrozenTextFeatures(), hidden=32).eval()
        x, y = model.encode_job(a), model.encode_job(b)
        self.assertNotEqual(a.ast_hash, b.ast_hash)
        self.assertGreater(float(torch.linalg.vector_norm(x - y).detach()), 1e-5)
        pooled = GroupGraphRanker(FrozenTextFeatures(), hidden=32, mode="pool").eval()
        self.assertTrue(torch.equal(pooled.encode_job(a), pooled.encode_job(b)))
        profile = {"skills": ["Python", "Java"], "tasks": []}
        first, second = counterexample_asts()
        self.assertEqual(logic_baseline(first, profile)["status"], "unknown")
        self.assertEqual(logic_baseline(second, profile)["status"], "pass")

    def test_random_graph_preserves_typed_degree(self):
        graph = self.graphs()[0]
        altered = [degree_preserving_randomization(graph, seed) for seed in (17, 42, 73)]
        for item in altered:
            self.assertEqual(degree_signature(graph), degree_signature(item))
            self.assertEqual(graph.nodes, item.nodes)
        self.assertTrue(any(set(item.edges) != set(graph.edges) for item in altered))

    def test_eval_graph_does_not_update_train_shared_state(self):
        model = GroupGraphRanker(FrozenTextFeatures()).eval()
        state = copy.deepcopy(model.state_dict())
        features = copy.deepcopy(model.features.__dict__)
        with torch.inference_mode():
            model.encode_job(self.graphs()[1])
            model.encode_profile({"text": "独立留出画像", "skills": ["Rust"], "tasks": ["数据审计"]})
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(state[key], tensor))
        self.assertEqual(features, model.features.__dict__)

    def test_unknown_relation_and_preferred_not_false_evidence(self):
        unknown = group("all", leaf("Python"), parse_status="unknown")
        self.assertEqual(logic_baseline(unknown, {"skills": ["Python"]})["status"], "unknown")
        preferred = group("any", leaf("SQL"), leaf("Python", modality="preferred"))
        self.assertEqual(logic_baseline(preferred, {"skills": ["Python"]})["status"], "unknown")
        negative = {"skills": {"SQL": {"level": "否定"}}}
        self.assertEqual(logic_baseline(leaf("SQL"), negative)["status"], "fail")
        self.assertEqual(logic_baseline(leaf("SQL"), {"skills": []})["status"], "unknown")
        for extra in ({"parse_status": "unknown"}, {"conflict": True}, {"actor": "background"}):
            self.assertEqual(logic_baseline(leaf("SQL"), {"skills": {"SQL": {"level": "实践", **extra}}})["status"], "unknown")

    def test_task_nodes_are_not_dropped(self):
        ast = group("all", leaf("数据清洗", "task"), group("any", leaf("Python"), leaf("SQL")))
        graph = build_graph({"job_id": "task-job", "title": "数据岗", "requirement_ast": ast})
        self.assertIn("task", [node.kind for node in graph.nodes])
        self.assertEqual(logic_baseline(ast, {"skills": ["Python"], "tasks": ["数据清洗"]})["status"], "pass")

    def test_unknown_leaf_and_empty_requirement_keep_uncertainty(self):
        for ast in ({**leaf("Python"), "parse_status": "unknown"}, group("all")):
            self.assertEqual(logic_baseline(ast, {"skills": ["Python"]})["status"], "unknown")
            graph = build_graph({"job_id": "uncertain", "title": "尚待结构化岗位", "requirement_ast": ast})
            self.assertTrue(any(edge.status == "unknown" for edge in graph.edges))
            for mode in ("pool", "graph"):
                self.assertTrue(bool(torch.isfinite(GroupGraphRanker(FrozenTextFeatures(), mode=mode).encode_job(graph)).all()))


class GateTests(unittest.TestCase):
    def test_fixture_requires_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            validate_supervision(*make_fixtures())
        _, _, _, contract = validate_supervision(*make_fixtures(), allow_fixture=True)
        self.assertTrue(contract["fixture_only"])

    def test_empty_human_labels_cannot_train(self):
        jobs, profiles, _ = make_fixtures()
        with self.assertRaisesRegex(ValueError, "没有已审核"):
            validate_supervision(jobs, profiles, [])

    def test_unknown_grade_is_not_negative(self):
        jobs, profiles, labels = make_fixtures()
        labels[0]["grade"] = None
        with self.assertRaisesRegex(ValueError, "未知"):
            validate_supervision(jobs, profiles, labels, True)

    def test_family_leak_is_rejected(self):
        jobs, profiles, labels = make_fixtures()
        profiles[2]["profile_family_id"] = profiles[0]["profile_family_id"]
        with self.assertRaisesRegex(ValueError, "泄漏"):
            validate_supervision(jobs, profiles, labels, True)

    def test_fake_human_without_adjudication_is_rejected(self):
        jobs, profiles, labels = make_fixtures()
        labels[0]["label_source"] = "human_adjudicated"
        with self.assertRaisesRegex(ValueError, "血缘"):
            validate_supervision(jobs, profiles, labels, True)

    def model_labels(self):
        jobs, profiles, labels = make_fixtures()
        for row in labels:
            row.update(label_source="llm_reviewed", model="fixture-model", prompt_hash="a"*64,
                       input_hash="b"*64, review_run_id="fixture-weak-run")
            row["model_reviews"] = [{"channel": channel, "model": "fixture-model", "model_revision": "fixture-v1",
                "prompt_hash": "a"*64, "input_hash": "b"*64, "request_id": channel, "grade": row["grade"]}
                for channel in ("support", "transfer")]
            row["rule_validation"] = {"passed": True, "input_hash": "b"*64, "validator_version": "fixture-rules",
                                      "checks": {"quotes": True}}
        return jobs, profiles, labels

    def test_model_labels_require_flag_and_complete_review_contract(self):
        jobs, profiles, labels = self.model_labels()
        with self.assertRaisesRegex(ValueError, "allow-model-labels"):
            validate_supervision(jobs, profiles, labels)
        *_, contract = validate_supervision(jobs, profiles, labels, allow_model_labels=True)
        self.assertTrue(contract["model_weak_supervision"])
        self.assertFalse(contract["eligible_for_production"])
        self.assertFalse(contract["fixture_only"])
        labels[0]["model_reviews"][1]["grade"] = 2
        with self.assertRaisesRegex(ValueError, "模型评审"):
            validate_supervision(jobs, profiles, labels, allow_model_labels=True)

    def test_failed_rules_or_mixed_teacher_human_are_rejected(self):
        jobs, profiles, labels = self.model_labels()
        labels[0]["rule_validation"]["checks"]["quotes"] = False
        with self.assertRaisesRegex(ValueError, "命名规则"):
            validate_supervision(jobs, profiles, labels, allow_model_labels=True)
        labels[0].update(label_source="human_adjudicated", reviewer_ids=["fixture-a", "fixture-b"], adjudication_id="fixture-adjudication")
        labels[0].update(task_hash="fixture-task", adjudicator_id="fixture-a", adjudication_sha256="fixture-hash",
                         registry_sha256="fixture-registry", reviewer_person_ids=["fixture-person-a", "fixture-person-b"],
                         review_ids=["fixture-review-a", "fixture-review-b"],
                         review_hashes={"fixture-review-a": "fixture-a", "fixture-review-b": "fixture-b"})
        with self.assertRaisesRegex(ValueError, "不得混"):
            validate_supervision(jobs, profiles, labels, allow_model_labels=True)


class EmbeddingTests(unittest.TestCase):
    def test_project_score_aggregation_matches_service_formula(self):
        from job_agent.encoder import aggregate_dense_scores
        queries = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [-1., 0.]], requires_grad=True)
        documents = torch.tensor([[1., 0.], [0., 1.]])
        scores = segment_scores(queries, documents, [3, 1], torch)
        expected = aggregate_dense_scores((queries[:1] @ documents.T)[0], (queries[1:3] @ documents.T).max(dim=0).values)
        torch.testing.assert_close(scores[0], expected)
        torch.testing.assert_close(scores[1], (queries[-1:] @ documents.T)[0])
        scores.sum().backward()
        self.assertGreater(float(queries.grad[1:3].abs().sum()), 0)

    def test_unknown_in_batch_pairs_never_enter_negative_denominator(self):
        labels = {("q", "positive"): {"grade": 3}, ("q", "negative"): {"grade": 0, "hard_negative": True},
                  ("q", "partial"): {"grade": 1}, ("q", "soft-zero"): {"grade": 0}}
        positive, negative, informative = reviewed_masks(["q"], ["positive", "negative", "partial", "unknown", "soft-zero"], labels, torch)
        self.assertTrue(bool(informative.all()))
        self.assertEqual(negative.tolist(), [[False, True, False, False, False]])
        query = torch.tensor([[1., 0.]], requires_grad=True)
        documents = torch.tensor([[1., 0.], [0., 1.], [100., 0.], [100., 0.], [100., 0.]], requires_grad=True)
        loss = multipositive_loss(query, documents, positive, negative, .05, torch)
        loss.backward()
        self.assertTrue(torch.equal(documents.grad[2:], torch.zeros_like(documents.grad[2:])))

    def test_shared_pooling_left_right_and_gradient(self):
        from job_agent.encoder import pool_hidden, wrap_texts
        hidden = torch.tensor([[[1., 0.], [0., 3.], [4., 4.]], [[4., 4.], [1., 0.], [0., 3.]]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        output = pool_hidden(hidden, mask, "last_token")
        torch.testing.assert_close(output[0], output[1])
        output[:, 0].sum().backward()
        self.assertGreater(float(hidden.grad[0, 1].abs().sum()), 0)
        self.assertEqual(float(hidden.grad[0, 2].abs().sum()), 0)
        self.assertEqual(wrap_texts(["正文"], True, "last_token", "检索"), ["Instruct: 检索\nQuery: 正文"])

    def test_unified_renderer_keeps_project_context(self):
        from pathlib import Path
        import job_agent.retrieval_contract as renderer
        self.assertIs(load_module(Path(renderer.__file__), "test_renderer"), renderer)
        jobs, profiles, _ = make_fixtures()
        queries, documents = prepare_rendering({row["job_id"]: row for row in jobs},
                                               {row["query_id"]: row for row in profiles}, renderer)
        self.assertIn("本人使用Python和Java完成练习", queries["train-query-0"])
        self.assertIn("这些材料不是任何真实招聘信息", documents["train-job-0"])


if __name__ == "__main__":
    unittest.main()
