"""最小行动规划的分支、三态、引用、去重及预算回归；全部使用虚构材料。"""
from copy import deepcopy

import pytest

from job_agent import action_planner as planner
from job_agent.domain import Job, parse_profile


def job(description, requirements="经验不限本科", salary="10-20K", address="深圳南山区"):
    return Job("fixture", "v", "虚构岗位", "虚构企业", "技术", "技术", salary, requirements, description, address, 1)


def profile(text="本科毕业。", **preferences):
    return parse_profile(text, {"city": "深圳", "intent": "技术", "education": "本科", "experience_years": 0, **preferences})


def keysets(result):
    return {frozenset(action["key"] for action in plan["actions"]) for plan in result["plans"]}


def test_nested_or_requires_one_complete_and_branch_not_four_skills():
    target = job("必须掌握(Python和SQL)或(Java和MySQL)。")
    result = planner.plan_actions(profile(), target)
    assert keysets(result) == {frozenset(["Python", "SQL"]), frozenset(["Java", "MySQL"])}
    assert all(plan["action_count"] == 2 and plan["selected_branches"] for plan in result["plans"])
    assert result["truncated"] is False


def test_existing_python_satisfies_or_without_suggesting_java():
    result = planner.plan_actions(profile("使用Python完成服务开发。"), job("必须掌握Python或Java。"))
    assert keysets(result) == {frozenset()}
    assert result["plans"][0]["status"] == "no_action_for_current_self_report"
    assert result["plans"][0]["conditional"] is True


def test_local_shortest_branch_must_not_destroy_global_overlap_optimum():
    target = job("必须掌握(Python和SQL)或Java。\n必须掌握Python和SQL。")
    result = planner.plan_actions(profile(), target)
    assert keysets(result) == {frozenset(["Python", "SQL"])}
    python = next(action for action in result["plans"][0]["actions"] if action["key"] == "Python")
    assert len(python["group_ids"]) == 2


def test_deduplicates_same_skill_and_preserves_all_exact_references():
    target = job("必须掌握Python和SQL。", "经验不限本科，熟悉Python。")
    result = planner.plan_actions(profile(), target)
    actions = result["plans"][0]["actions"]
    assert len(actions) == 2
    python = next(action for action in actions if action["key"] == "Python")
    assert {ref["job_evidence"]["field"] for ref in python["requirements"]} == {"requirements", "description"}
    for action in actions:
        for ref in action["requirements"]:
            cite = ref["job_evidence"]
            assert getattr(target, cite["field"])[cite["start"]:cite["end"]] == cite["quote"]
            assert ref["group_id"] in action["group_ids"]


def test_unknown_requests_evidence_but_explicit_negative_requests_learning():
    result = planner.plan_actions(profile("我不会Python。"), job("必须掌握Python和SQL。"))
    actions = {action["key"]: action for action in result["plans"][0]["actions"]}
    assert actions["Python"]["kind"] == "learning"
    assert actions["SQL"]["kind"] == "evidence_request"
    assert all(action["conditional"] and action["completion_is_not_verified_competence"] for action in actions.values())


def test_team_background_is_unknown_and_never_rewritten_as_personal_fact():
    person = profile("团队使用Python，我负责项目协调。")
    before = deepcopy(person)
    result = planner.plan_actions(person, job("必须掌握Python。"))
    assert result["plans"][0]["actions"][0]["kind"] == "evidence_request"
    assert person == before


def test_preferred_or_group_stays_optional_and_preserves_alternatives():
    target = job("必须掌握Python。\n加分项：熟悉Java或Go。")
    result = planner.plan_actions(profile("使用Python完成脚本。"), target)
    assert keysets(result) == {frozenset()}
    assert len(result["optional_actions"]) == 1
    optional = result["optional_actions"][0]
    assert optional["preferred"]
    assert {frozenset(action["key"] for action in plan["actions"]) for plan in optional["plans"]} == {frozenset(["Java"]), frozenset(["Go"])}


def test_negated_job_requirement_is_not_a_learning_action():
    result = planner.plan_actions(profile(), job("不要求Python。必须掌握SQL。"))
    assert keysets(result) == {frozenset(["SQL"])}


def test_ambiguous_logic_requests_employer_confirmation_not_all_skills():
    result = planner.plan_actions(profile(), job("熟悉Python/Java。"))
    assert result["status"] == "needs_requirement_confirmation"
    assert not any(plan["actions"] for plan in result["plans"])
    assert all(not plan["all_required_groups_accounted_for"] for plan in result["plans"])
    assert any(row["kind"] == "requirement_clarification" and row["recipient"] == "employer" for row in result["non_actionable_constraints"])


def test_hard_constraints_remain_separate_and_do_not_become_quick_skill_tasks():
    target = job("必须掌握SQL。", "本科，3年以上工作经验。", salary="10-15K")
    result = planner.plan_actions(profile(education="大专", experience_years=0, salary_min=25000), target)
    failed = {row["name"] for row in result["non_actionable_constraints"] if row["status"] == "fail"}
    assert {"学历要求", "经验要求", "最低月薪"} <= failed
    assert keysets(result) == {frozenset(["SQL"])}
    assert not result["plans"][0]["hard_constraints_resolved"]
    assert all(not row["resolvable_by_skill_actions"] for row in result["non_actionable_constraints"])


def test_unknown_job_fields_ask_employer_but_missing_personal_field_asks_user():
    target = job("必须掌握SQL。", "要求待确认", salary="面议", address="地点待确认")
    result = planner.plan_actions(profile(salary_min=10000), target)
    assert all(row["recipient"] == "employer" for row in result["non_actionable_constraints"])
    target = job("必须掌握SQL。")
    result = planner.plan_actions(profile("简历尚未补全。", education="", experience_years=None), target)
    assert all(row["recipient"] == "user" for row in result["non_actionable_constraints"])


def test_task_actions_work_without_dictionary_skills():
    target = job("负责客户维护，开展客户需求分析。")
    result = planner.plan_actions(profile(), target)
    assert result["plans"] and all(action["type"] == "task" for action in result["plans"][0]["actions"])
    assert result["plans"][0]["cost"]["distinct_skills"] == 0


def test_empty_graph_does_not_mean_every_requirement_is_met():
    result = planner.plan_actions(profile(), job("其他职责由面谈确认。"))
    assert result["status"] == "no_parsed_requirements" and result["plans"] == []


def test_invalid_citation_is_rejected_before_planning():
    target = job("必须掌握Python。")
    target.groups[0]["ast"]["evidence"]["quote"] = "伪造引用"
    result = planner.plan_actions(profile(), target)
    assert result["plans"] == [] and result["truncated"]
    assert "invalid_citation" in result["truncation_reasons"]


def test_node_and_expansion_limits_return_explicit_truncation(monkeypatch):
    target = job("必须掌握(Python和SQL)或(Java和MySQL)。")
    monkeypatch.setattr(planner, "MAX_NODES", 2)
    result = planner.plan_actions(profile(), target)
    assert not result["plans"] and "graph_budget" in result["truncation_reasons"]
    monkeypatch.setattr(planner, "MAX_NODES", 512)
    monkeypatch.setattr(planner, "MAX_EXPANSIONS", 2)
    result = planner.plan_actions(profile(), target)
    assert not result["plans"] and "expansion_budget" in result["truncation_reasons"]


def test_action_and_display_limits_do_not_return_incomplete_branch():
    target = job("和".join("必须掌握" + name for name in ["Python", "Java", "SQL", "MySQL", "Go", "Docker", "Linux", "Redis", "Git"]) + "。")
    result = planner.plan_actions(profile(), target)
    assert result["plans"] == [] and "action_limit" in result["truncation_reasons"]
    target = job("必须掌握Python或Java或SQL或Go。")
    result = planner.plan_actions(profile(), target)
    assert len(result["plans"]) == 3 and "plan_display_limit" in result["truncation_reasons"]
    assert all(plan["action_count"] <= 8 for plan in result["plans"])


@pytest.mark.parametrize("value", [0, 4, True, 1.5])
def test_invalid_max_plans(value):
    with pytest.raises(ValueError):
        planner.plan_actions(profile(), job("必须掌握Python。"), value)


def test_unknown_preferred_logic_does_not_block_supported_required_group():
    target = job("必须掌握Python。\n加分项：熟悉Java/Go。")
    result = planner.plan_actions(profile("使用Python完成脚本。"), target)
    assert not result["unresolved_group_ids"] and result["optional_unresolved_group_ids"]
    assert result["plans"][0]["all_required_groups_accounted_for"]
    assert result["optional_actions"][0]["plans"][0]["status"] == "optional_pending_requirement_confirmation"


def test_separate_python_interface_and_excel_cleaning_need_binding_evidence():
    from job_agent.evidence_alignment import align_requirements
    target = job("使用Python完成数据清洗。")
    person = profile("项目一：接口\n使用Python完成接口开发。\n\n项目二：整理\n使用Excel完成数据清洗。")
    alignment = align_requirements(person, target)
    assert any(row["global_status"] == "pass" and row["status"] == "unknown" for row in alignment["groups"])
    result = planner.plan_actions(person, target, alignment=alignment)
    assert result["plans"] and result["plans"][0]["action_count"] == 1
    action = result["plans"][0]["actions"][0]
    assert action["type"] == "binding" and action["kind"] == "task_binding"
    assert action["task_key"] == "data_cleaning" and len(action["group_ids"]) == 2
    assert result["plans"][0]["has_unresolved_binding"]
    for ref in action["requirements"]:
        for field in ("job_evidence", "job_scope"):
            cite = ref[field]
            assert getattr(target, cite["field"])[cite["start"]:cite["end"]] == cite["quote"]


def test_same_scope_explicit_binding_needs_no_new_action():
    from job_agent.evidence_alignment import align_requirements
    target = job("使用Python完成数据清洗。")
    person = profile("使用Python完成数据清洗。")
    result = planner.plan_actions(person, target, alignment=align_requirements(person, target))
    assert keysets(result) == {frozenset()}
    assert not result["plans"][0]["has_unresolved_binding"]


def test_preferred_binding_only_adds_optional_actions():
    from job_agent.evidence_alignment import align_requirements
    target = job("必须掌握SQL。\n加分项：使用Python完成数据清洗。")
    person = profile("使用SQL和Python完成接口开发。\n\n使用Excel完成数据清洗。")
    result = planner.plan_actions(person, target, alignment=align_requirements(person, target))
    assert keysets(result) == {frozenset()}
    assert any(action["type"] == "binding" for item in result["optional_actions"] for plan in item["plans"] for action in plan["actions"])


def test_unevaluated_or_stale_alignment_cannot_claim_all_groups_accounted():
    from job_agent.evidence_alignment import align_requirements
    target = job("使用Python完成数据清洗。")
    person = profile("使用Python完成数据清洗。")
    for alignment in ({"summary": {"status": "not_evaluated"}}, {**align_requirements(person, target), "profile_version": "old"}):
        result = planner.plan_actions(person, target, alignment=alignment)
        assert result["alignment_status"] == "not_evaluated"
        assert all(not plan["all_required_groups_accounted_for"] and plan["has_unresolved_binding"] for plan in result["plans"])


def test_binding_counts_toward_maximum_actions(monkeypatch):
    from job_agent.evidence_alignment import align_requirements
    target = job("使用Python完成数据清洗。")
    person = profile()
    monkeypatch.setattr(planner, "MAX_ACTIONS", 2)
    result = planner.plan_actions(person, target, alignment=align_requirements(person, target))
    assert result["plans"] == [] and "action_limit" in result["truncation_reasons"]
