"""只用临时合成 XLSX 验证血缘、隔离、证据、统计与发布保护。"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook

from research import data_pipeline as pipeline

REPOSITORY = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def fixture(path):
    columns = ["岗位名称", "企业", "岗位薪资", "岗位要求", "岗位职责", "岗位地址", "职位类型名称", "二级分类", "内部标记"]
    rows = []
    for index in range(120):
        rows.append([f"虚构后端岗位{index}", f"虚构企业{index}", "10-20K", "1-3年本科",
                     f"熟悉Python或Java至少一种，优先掌握SQL。负责第{index:04d}号系统接口设计与自动化测试，编写项目说明。",
                     "深圳南山区", "Python", "后端开发", "原始"])
    rows.append(copy.deepcopy(rows[0]))  # 完整重复，应聚合血缘。
    changed = copy.deepcopy(rows[0]); changed[4] = "熟悉Python，负责另一个系统的接口开发与测试，编写独立文档。"
    rows.append(changed)  # 同业务键不同职责，必须留下。
    changed = copy.deepcopy(rows[0]); changed[0] = "跨企业职位"; changed[1] = "另一虚构企业"
    rows.append(changed)  # 相同职责跨企业，必须同家族同split。
    changed = copy.deepcopy(rows[0]); changed[8] = "额外列内容发生变化"
    rows.append(changed)  # 未映射列也参与完整内容版本。
    book = Workbook(); sheet = book.active
    sheet.append(columns)
    for row in rows:
        sheet.append(row)
    book.save(path); book.close()


class PipelineTests(unittest.TestCase):
    def test_full_build_lineage_and_rerun_protection(self):
        with tempfile.TemporaryDirectory(prefix="job-pipeline-test-") as folder:
            root = Path(folder); source = root/"synthetic.xlsx"; fixture(source)
            before = pipeline.file_hash(source)
            run, manifest = pipeline.build_dataset(source, root/"research", REPOSITORY, annotation_counts=(6,2,4))
            jobs = load_jsonl(run/"jobs.jsonl"); edges = load_jsonl(run/"edges.jsonl"); tasks = load_jsonl(run/"annotation_tasks.jsonl")
            self.assertEqual(manifest["data"]["raw_records"],124)
            self.assertEqual(len(jobs),123)
            self.assertEqual(sum(len(job["source_record_ids"]) for job in jobs),124)
            self.assertEqual(manifest["data"]["legacy_key_additional_versions"],2)
            original = [job for job in jobs if job["title"] == "虚构后端岗位0"]
            self.assertEqual(len(original),3)
            self.assertEqual(len({job["job_family_id"] for job in original}),1)
            self.assertEqual(len({job["split"] for job in original}),1)
            other = next(job for job in jobs if job["title"] == "跨企业职位")
            self.assertEqual(other["job_family_id"],original[0]["job_family_id"])
            self.assertTrue(all(job["job_id"] == job["job_version_id"] == job["content_sha256"][:20] for job in jobs))
            family_splits = {}
            by_id = {job["job_id"]:job for job in jobs}
            for job in jobs:
                family_splits.setdefault(job["job_family_id"],set()).add(job["split"])
            self.assertTrue(all(len(values)==1 for values in family_splits.values()))
            self.assertEqual(Counter(task["split"] for task in tasks),Counter(train=6,dev=2,test=4))
            self.assertTrue(all(task["label"] is None and task["split"] == by_id[task["job_id"]]["split"] for task in tasks))
            for edge in edges:
                if edge["status"] != "hypothesis":
                    citation = edge["evidence"]; job = by_id[edge["job_id"]]
                    self.assertEqual(job[citation["field"]][citation["start"]:citation["end"]],citation["quote"])
            or_edge = next(edge for edge in edges if edge["relation"] == "has_requirement_group")
            options = [edge for edge in edges if edge["source_id"] == or_edge["target_id"]]
            self.assertEqual({edge["target_id"] for edge in options},{"skill:Python","skill:Java"})
            self.assertTrue(all(edge["relation"] == "option" and edge["status"] == "machine_parsed" for edge in options))
            direct_required = [edge for edge in edges if edge["source_id"] == or_edge["source_id"] and edge["relation"] == "single_requirement"]
            self.assertFalse(any(edge["target_id"] in {"skill:Python","skill:Java"} for edge in direct_required))
            hypothesis = [edge for edge in edges if edge["status"] == "hypothesis"]
            self.assertTrue(hypothesis)
            self.assertTrue(all(edge["split"] == "train" and edge["evidence"] is None and not edge["fact_citable"] for edge in hypothesis))
            train_jobs = [job for job in jobs if job["split"] == "train"]
            for edge in hypothesis:
                first = edge["source_id"].removeprefix("skill:"); second = edge["target_id"].removeprefix("skill:")
                actual = sum(first in job["skills"] and second in job["skills"] for job in train_jobs)
                self.assertEqual(edge["support_count"],actual)
            self.assertLessEqual(max(Counter(edge["source_id"] for edge in hypothesis).values()),10)
            self.assertEqual(pipeline.file_hash(source),before)
            rerun, again = pipeline.build_dataset(source, root/"research", REPOSITORY, annotation_counts=(6,2,4))
            self.assertEqual(run,rerun); self.assertEqual(manifest,again)
            # 完全相同数据在另一输出根目录重跑，训练/标注等内容必须一致。
            second, reproduced = pipeline.build_dataset(source, root/"another", REPOSITORY, annotation_counts=(6,2,4))
            self.assertEqual(manifest["files"],reproduced["files"])
            self.assertNotEqual(run,second)
            with (run/"jobs.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError,"校验失败"):
                pipeline.build_dataset(source, root/"research", REPOSITORY, annotation_counts=(6,2,4))
            self.assertEqual(pipeline.file_hash(source),before)

    def test_repo_output_and_partial_publication_rejected(self):
        with tempfile.TemporaryDirectory(prefix="job-pipeline-test-") as folder:
            root = Path(folder); source = root/"synthetic.xlsx"; fixture(source)
            before = pipeline.file_hash(source)
            with self.assertRaisesRegex(ValueError,"仓库外"):
                pipeline.build_dataset(source,REPOSITORY/"forbidden-research-output",REPOSITORY,annotation_counts=(0,0,0))
            output = root/"research"
            with self.assertRaisesRegex(ValueError,"片段不足"):
                pipeline.build_dataset(source,output,REPOSITORY,annotation_counts=(300,100,200))
            self.assertFalse(list(output.iterdir()))
            self.assertEqual(pipeline.file_hash(source),before)

    def test_test_only_skill_cannot_generate_training_cooccurrence(self):
        jobs = []
        for index in range(12):
            split = "train" if index < 10 else "test"
            skills = {skill:{"field":"description","start":start,"end":end,"quote":quote,"level":"熟悉","preferred":False}
                      for skill,start,end,quote in (("Python",0,6,"Python"),("Java",7,11,"Java"))}
            if split == "test":
                skills["SQL"] = {"field":"description","start":12,"end":15,"quote":"SQL","level":"熟悉","preferred":False}
            jobs.append({"job_id":str(index),"split":split,"skills":skills,"groups":[],"category":"","category_parent":"",
                         "description":"Python Java SQL","source_record_ids":[str(index)],"field_sources":{"description":"岗位职责"}})
        edges,_ = pipeline.build_edges(jobs,minimum_support=2,top_k=10)
        hypothesis = [edge for edge in edges if edge["status"] == "hypothesis"]
        self.assertEqual(len(hypothesis),2)
        self.assertFalse(any("skill:SQL" in (edge["source_id"],edge["target_id"]) for edge in hypothesis))
        self.assertTrue(all(edge["support_count"]==10 for edge in hypothesis))


if __name__ == "__main__":
    unittest.main(verbosity=2)
