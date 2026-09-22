"""固定工程矩阵可复跑，且真实行为回退时报告会失败。"""
import json

import pytest

from research import evaluate_decision_innovations as benchmark


def test_all_fixed_dimensions_match_independently_declared_expectations():
    report = benchmark.build_report()
    assert set(report["dimensions"]) == {"动作证据对齐", "主动追问", "逻辑行动规划", "区间与多目标比较"}
    assert report["summary"]["cases"] >= 23
    assert report["summary"]["current_expected_met"] == report["summary"]["cases"]
    assert report["summary"]["baseline_false_support"] == 6
    assert report["summary"]["current_false_support"] == 0
    assert report["real_person_job_effect_measured"] is False
    assert report["human_gold_count"] == report["external_api_calls"] == report["training_runs"] == 0
    assert all(len(value) == 64 for value in report["source_files_sha256"].values())


def test_reintroduced_global_binding_error_is_exposed(monkeypatch):
    original = benchmark.align_requirements
    def broken(profile, job):
        result = original(profile, job)
        for group in result["groups"]:
            group["status"] = group["global_status"]
        return result
    monkeypatch.setattr(benchmark, "align_requirements", broken)
    report = benchmark.build_report()
    assert report["summary"]["current_false_support"] == 6
    failed = [case for case in report["cases"] if not case["current"]["expected_met"]]
    assert len(failed) >= 6
    assert any(case["case_id"] == "alignment_separate_actions" for case in failed)


def test_report_preserves_scope_and_refuses_overwrite(tmp_path):
    output = tmp_path/"new-report"
    report = benchmark.run(output)
    stored = json.loads((output/"report.json").read_text("utf-8"))
    assert stored["fixture_spec_sha256"] == report["fixture_spec_sha256"]
    assert stored["all_inputs_synthetic"] is True
    document = (output/"REPORT.md").read_text("utf-8")
    assert "不是独立人岗效果评测" in document and "逐例归因" in document
    before = (output/"report.json").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.run(output)
    assert (output/"report.json").read_bytes() == before
