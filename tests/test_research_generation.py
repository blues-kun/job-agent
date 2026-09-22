"""纯CPU门槛测试；虚构审核员仅用于单测，正式CLI明确拒绝fixture_only。"""
from copy import deepcopy
import unittest

import torch

from research.train_generation_sft import collate, prepare_example, review_payload_sha256, validate_records


def record(split="train", task="extract"):
    value = {
        "fixture_only": True,
        "sample_id": f"fixture-{split}", "split": split, "task": task,
        "source_kind": "human_authored", "review_status": "human_adjudicated",
        "reviewer_ids": ["fictional-test-reviewer"],
        "adjudication": {"record_id": f"fictional-{split}", "reviewer_id": "fictional-test-reviewer",
                         "decision": "approved", "reviewed_at": "2026-09-22T12:00:00+08:00"},
        "source_job_family": [f"job-{split}"], "profile_family": [f"profile-{split}"],
        "messages": [{"role": "system", "content": "只处理已给出的事实。"},
                     {"role": "user", "content": f"{split}样例：参与数据整理。"}],
        "label": "参与数据整理。",
    }
    value["adjudication"]["payload_sha256"] = review_payload_sha256(value)
    return value


class CharacterTokenizer:
    """字符级tokenizer用来独立验证loss契约，不模拟模型质量。"""
    is_fast = True
    all_special_tokens = ["<CONTROL>"]
    poison_prompt_mask = False

    def render(self, messages, generation=False):
        value = "".join(f"<{message['role']}>" + message["content"] + "<end>" for message in messages)
        return value + ("<assistant>" if generation else "")

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, **kwargs):
        value = self.render(messages, add_generation_prompt)
        if not tokenize:
            return value
        prefix = self.render(messages[:-1], True)
        left, right = len(prefix), len(prefix) + len(messages[-1]["content"])
        mask = [int(left <= index < right) for index in range(len(value))]
        if self.poison_prompt_mask:
            mask[0] = 1
        return {"input_ids": [ord(char) for char in value], "assistant_masks": mask}

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class TrainingGatesTest(unittest.TestCase):
    def test_assistant_only_mask_padding_and_no_silent_truncation(self):
        tokenizer, example = CharacterTokenizer(), record()
        prepared = prepare_example(tokenizer, example, 1000)
        selected = "".join(chr(value) for value in prepared["labels"] if value != -100)
        self.assertEqual(selected, example["label"])
        self.assertEqual(prepared["assistant_tokens"], len(example["label"]))
        shorter = deepcopy(prepared)
        shorter["input_ids"] = shorter["input_ids"][:-1]
        shorter["labels"] = shorter["labels"][:-1]
        batch = collate([prepared, shorter], 0, torch.device("cpu"))
        self.assertEqual(int(batch["labels"][1, -1]), -100)
        self.assertEqual(int(batch["attention_mask"][1, -1]), 0)
        with self.assertRaisesRegex(ValueError, "不会截断"):
            prepare_example(tokenizer, example, 5)
        tokenizer.poison_prompt_mask = True
        with self.assertRaisesRegex(ValueError, "mask与字符跨度不一致"):
            prepare_example(tokenizer, example, 1000)

    def test_reject_unreviewed_null_or_fixture_training(self):
        examples = [record(), record("dev")]
        with self.assertRaisesRegex(ValueError, "单测虚构"):
            validate_records(examples)
        checked, summary = validate_records(examples, allow_test_fixtures=True)
        self.assertEqual(summary["splits"], {"train": 1, "dev": 1})
        self.assertFalse(summary["semantic_facts_proven"])
        for field, bad in [("review_status", "synthetic_unreviewed"), ("source_kind", "synthetic_unreviewed"),
                           ("label", None), ("reviewer_ids", []), ("adjudication", None)]:
            with self.subTest(field=field):
                rows = deepcopy(examples); rows[0][field] = bad
                with self.assertRaises(ValueError):
                    validate_records(rows, allow_test_fixtures=True)
        rows = deepcopy(examples); rows[0]["messages"].append("invalid")
        with self.assertRaisesRegex(ValueError, "messages仅允许"):
            validate_records(rows, allow_test_fixtures=True)
        rows = deepcopy(examples); rows[0]["label"] = "未经重新仲裁的标签变更。"
        with self.assertRaisesRegex(ValueError, "payload_sha256"):
            validate_records(rows, allow_test_fixtures=True)

    def test_job_profile_and_exact_prompt_split_leakage(self):
        for field in ["source_job_family", "profile_family", "messages"]:
            with self.subTest(field=field):
                rows = [record(), record("dev")]
                rows[1][field] = deepcopy(rows[0][field])
                with self.assertRaisesRegex(ValueError, "跨train/dev"):
                    validate_records(rows, allow_test_fixtures=True)
        rows = [record(), record("dev")]
        rows[1]["profile_family"] = [" profile-dev "]
        with self.assertRaises(ValueError):
            validate_records(rows, allow_test_fixtures=True)

    def test_preserving_rewrite_requires_full_exact_alignment(self):
        source = "了解Python。参与SQL练习。"
        rewrite = record(task="rewrite")
        rewrite.update({"rewrite_mode": "sentence_preserving", "source_text": source, "label": source,
                        "alignment_spans": [{"source_start": 0, "source_end": len(source),
                                             "target_start": 0, "target_end": len(source)}]})
        rewrite["messages"][-1]["content"] = "整理以下简历：" + source
        rewrite["adjudication"]["payload_sha256"] = review_payload_sha256(rewrite)
        rows = [rewrite, record("dev")]
        checked, _ = validate_records(rows, allow_test_fixtures=True)
        self.assertTrue(checked[0]["structural_checks"]["alignment_complete"])
        self.assertFalse(checked[0]["structural_checks"]["semantic_facts_proven"])
        for change in [{"alignment_spans": []}, {"label": source.replace("了解", "精通")},
                       {"programmatically_proven_facts": True}]:
            bad = deepcopy(rows); bad[0].update(change)
            with self.assertRaises(ValueError):
                validate_records(bad, allow_test_fixtures=True)
        complex_rows = deepcopy(rows)
        complex_rows[0].update({"rewrite_mode": "human_reviewed_complex", "label": "经真人审核的复杂改写。"})
        complex_rows[0]["adjudication"]["payload_sha256"] = review_payload_sha256(complex_rows[0])
        checked, _ = validate_records(complex_rows, allow_test_fixtures=True)
        self.assertEqual(checked[0]["structural_checks"], {"semantic_facts_proven": False})


if __name__ == "__main__":
    unittest.main()
