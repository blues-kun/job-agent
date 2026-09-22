"""解析版本桥接的原文保全与运行时一致性回归。"""
import pytest
pytest.importorskip("torch")

import copy
import json
from pathlib import Path
import tempfile
import unittest

from job_agent.domain import Job, PARSER_VERSION
from research.prepare_group_vectors import SEMANTIC_JOB_FIELDS, refresh_semantics, semantic_changes


class BridgeTests(unittest.TestCase):
    def fixture(self):
        row = {"job_id":"fictional-job", "job_version_id":"fictional-version", "job_family_id":"fictional-family",
            "title":"虚构开发岗位", "category":"软件开发", "category_parent":"技术", "company":"虚构公司",
            "salary_raw":"10-15K", "requirements":"需要Python或者Java，并需要SQL。", "description":"本人负责接口开发。",
            "address":"深圳", "split":"train", "snapshot":"fictional"}
        job = Job(id=row["job_id"],version=row["job_version_id"],row=0,family=row["category_parent"],
            **{key:row[key] for key in ("title","company","category","salary_raw","requirements","description","address")})
        row.update({key:getattr(job,key) for key in SEMANTIC_JOB_FIELDS})
        profile={"query_id":"fictional-query","profile_family_id":"fictional-profile","split":"train",
            "text":"本人使用Python编写接口，负责订单校验。","preferences":{"city":"深圳","intent":"后台开发"},
            "skills":{},"tasks":{},"parser_version":"fictional-old"}
        return row, profile

    def test_raw_change_and_missing_null_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"非解析字段"):
            semantic_changes({"description":"原文"},{"description":"另一段"},SEMANTIC_JOB_FIELDS)
        with self.assertRaisesRegex(ValueError,"非解析字段"):
            semantic_changes({}, {"company":None}, SEMANTIC_JOB_FIELDS)

    def test_bridge_keeps_source_text_and_preferences(self):
        current, profile=self.fixture()
        previous={**copy.deepcopy(current),"skills":{},"parser_version":"fictional-old"}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"semantic.jsonl"
            path.write_text(json.dumps(current,ensure_ascii=False)+"\n")
            jobs,profiles,contract=refresh_semantics({current["job_id"]:previous},[profile],path)
        self.assertEqual(profiles[0]["text"],profile["text"])
        self.assertEqual(profiles[0]["preferences"],profile["preferences"])
        self.assertEqual(profiles[0]["parser_version"],PARSER_VERSION)
        self.assertEqual(jobs[current["job_id"]],current)
        self.assertTrue(contract["raw_fields_unchanged"])
        self.assertIn("Python",profiles[0]["skills"])

    def test_matching_version_but_forged_parse_is_rejected(self):
        current,profile=self.fixture()
        altered=copy.deepcopy(current);altered["skills"]={}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"semantic.jsonl"
            path.write_text(json.dumps(altered,ensure_ascii=False)+"\n")
            with self.assertRaisesRegex(ValueError,"逐字段复现"):
                refresh_semantics({current["job_id"]:current},[profile],path)


if __name__=="__main__":unittest.main()
