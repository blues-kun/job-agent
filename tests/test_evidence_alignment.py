"""只用固定虚构反例验证动作绑定；不是实际人岗效果评测。"""
from copy import deepcopy

import pytest

from job_agent.domain import Job, parse_profile
from job_agent.evidence_alignment import LIMITS, align_requirements


PREFERENCES = {"city": "深圳", "intent": "数据分析", "education": "本科", "experience_years": 1}


def job(description="负责使用Python完成数据清洗。"):
    return Job("fixture-alignment", "v1", "虚构数据岗位", "虚构公司", "数据分析", "技术", "10-20K",
               "本科；经验不限。", description, "深圳南山区", 1)


def align(text, target=None):
    return align_requirements(parse_profile(text, PREFERENCES), target or job())


def bound_groups(result):
    return [group for group in result["groups"] if group["task_binding"]["required"]]


@pytest.mark.parametrize("text,expected", [
    ("项目一：使用Python完成数据清洗。", "pass"),
    ("项目一：使用Python编写订单接口。\n\n项目二：使用Excel完成数据清洗。", "unknown"),
    ("项目一：使用Python编写订单接口，使用Excel完成数据清洗。", "unknown"),
    ("项目一：熟悉Python。负责数据清洗，使用Excel制作报表。", "unknown"),
    ("项目一：团队使用Python完成数据清洗，我负责项目协调。", "unknown"),
    ("项目一：计划使用Python完成数据清洗。", "unknown"),
    ("项目一：不会使用Python完成数据清洗。", "fail"),
    ("任职要求：使用Python完成数据清洗。", "unknown"),
    ("项目一：阅读教程使用Python完成数据清洗的示例。", "unknown"),
])
def test_same_action_and_semantic_counterexamples(text, expected):
    result = align(text)
    groups = bound_groups(result)
    assert groups
    assert all(group["status"] == expected for group in groups)
    assert result["summary"]["ability_verified"] is False


def test_fixed_global_vs_bound_control_reduces_constructed_false_support():
    negatives = [
        "项目一：使用Python开发订单接口。\n\n项目二：使用Excel完成数据清洗。",
        "使用Python开发接口，使用Excel完成数据清洗。",
        "熟悉Python。负责数据清洗。",
        "使用Python编写接口。负责数据清洗，但处理时只使用Excel。",
        "技能清单：熟悉Python。\n\n项目经历：使用Excel完成数据清洗。",
        "项目一：使用Python编写自动化测试。\n\n项目二：开展数据清洗。",
    ]
    old_false, new_false = 0, 0
    for text in negatives:
        groups = bound_groups(align(text))
        old_false += all(group["global_status"] == "pass" for group in groups)
        new_false += all(group["status"] == "pass" for group in groups)
    assert old_false == 6
    assert new_false == 0
    assert all(group["status"] == "pass" for group in bound_groups(align("使用Python完成数据清洗。")))


def test_nonbinding_and_can_be_supported_by_separate_blocks():
    result = align("项目一：使用Python开发接口。\n\n项目二：使用SQL查询订单。", job("熟悉Python和SQL。"))
    assert result["groups"][0]["status"] == "pass"
    assert result["groups"][0]["task_binding"]["status"] == "not_required"
    assert len(result["groups"][0]["candidate_blocks"]) == 2


def test_or_tools_accepts_one_branch_but_never_infers_both():
    result = align("使用Java完成数据清洗。", job("使用Python或Java完成数据清洗。"))
    assert all(group["status"] == "pass" for group in bound_groups(result))
    result = align("使用Python或Java完成数据清洗。", job("使用Python完成数据清洗。"))
    assert all(group["task_binding"]["status"] == "unknown" for group in bound_groups(result))
    result = align("使用Python完成数据清洗。", job("使用Python和SQL完成数据清洗。"))
    assert all(group["status"] != "pass" for group in bound_groups(result))


def test_parenthesized_tool_expression_keeps_nested_alternatives():
    target = job("使用(Python或Java)和(SQL或MySQL)完成数据清洗。")
    positive = align("使用Python和SQL完成数据清洗。", target)
    assert all(group["task_binding"]["status"] == "pass" for group in bound_groups(positive))
    # 旧任务分支在括号场景可能为unknown；新的动作证据不能绕过这个解析边界。
    assert positive["groups"][0]["status"] == "pass"
    assert all(group["status"] == "unknown" for group in positive["groups"] if group["global_status"] == "unknown")
    incomplete = align("使用Python完成数据清洗。", target)
    assert all(group["status"] != "pass" for group in bound_groups(incomplete))


def test_original_spans_display_redaction_and_complete_project_block():
    text = "项目一：清洗项目\r\n2024.01-2024.06\r\n  使用Python完成数据清洗。联系13812345678。\r\n\r\n项目二：使用SQL查询订单。"
    target = job()
    result = align(text, target)
    for group in bound_groups(result):
        evidence = group["job_evidence"]
        assert getattr(target, evidence["field"])[evidence["start"]:evidence["end"]] == evidence["quote"]
        block = group["best_block"]
        assert block and "项目一" in block["quote"] and "2024.01" in block["quote"]
        assert text[block["start"]:block["end"]] == block["quote"]
        assert "13812345678" not in block["display_quote"]
        assert "[联系方式已隐藏]" in block["display_quote"]
        for proof in block["task_bindings"]:
            cite = proof["resume_evidence"]
            assert text[cite["start"]:cite["end"]] == cite["quote"]


def test_empty_unknown_and_candidate_limits():
    empty = align("做过项目。", job("负责尚未收录的专业业务。"))
    assert empty["groups"] == [] and empty["summary"]["no_parsed_requirements"]
    unknown = align("使用Python和Java完成数据清洗。", job("使用Python/Java完成数据清洗。"))
    assert all(group["status"] == "unknown" for group in unknown["groups"])
    many = align("\n\n".join(f"项目{i}：使用Python完成数据清洗。" for i in range(8)))
    assert all(len(group["candidate_blocks"]) == 3 for group in many["groups"])
    assert align("字"*(LIMITS["resume_chars"]+1))["summary"]["status"] == "not_evaluated"


def test_invalid_citation_cannot_become_support_and_inputs_are_unchanged():
    target = job(); original = deepcopy(target.__dict__)
    profile = parse_profile("使用Python完成数据清洗。", PREFERENCES); before = deepcopy(profile)
    align_requirements(profile, target)
    assert profile == before and target.__dict__ == original
    target.groups[0]["evidence"]["quote"] = "伪造跨度"
    result = align_requirements(profile, target)
    assert result["groups"][0]["status"] == "unknown"
    assert result["groups"][0]["job_evidence"] is None
