"""以明确反例验证追问影响、未知传播与候选取舍。"""
from copy import deepcopy

from job_agent.domain import Job, parse_profile
from job_agent.decision_support import evidence_bounds, plan_questions, comparison_frontier


def job(key="j1", description="掌握Python或Java。", requirements="本科，3年工作经验", salary="10-20K", address="深圳南山区"):
    return Job(key, "v1", "开发工程师", "样例企业", "开发", "技术", salary, requirements, description, address, 1)


def person(text="我使用Python完成接口开发。", **preferences):
    return parse_profile(text, {"city": "深圳", "intent": "开发", **preferences})


def test_already_satisfied_alternative_does_not_trigger_skill_question():
    output = plan_questions(person(), [job()])
    assert not any(row["kind"] == "experience_evidence" for row in output["questions"])
    assert {q["field"] for q in output["questions"]} == {"education", "experience_years"}


def test_unknown_employer_condition_cannot_be_answered_by_candidate():
    output = plan_questions(person(), [job(requirements="待遇面议", salary="面议", address="")])
    assert not output["questions"]
    assert set(output["employer_checks"][0]["fields"]) == {"工作城市", "可比月薪范围", "学历门槛", "总年资要求"}


def test_question_priority_changes_with_candidate_pool_and_budget():
    profile = person(education="本科")
    jobs = [job("a", requirements="本科，3年工作经验"), job("b", requirements="本科，5年工作经验")]
    output = plan_questions(profile, jobs, max_questions=1)
    q = output["questions"][0]
    assert q["field"] == "experience_years" and q["affected_job_ids"] == ["a", "b"]
    assert q["decision_partitions"] == 3
    assert plan_questions(profile, jobs, max_questions=0)["questions"] == []
    assert not plan_questions(person(education="本科", experience_years=5), jobs)["questions"]


def test_counterfactuals_do_not_modify_user_profile_or_mark_abilities_verified():
    profile = person("了解Python；我使用SQL维护报表。", education="本科", experience_years=3)
    original = deepcopy(profile)
    output = plan_questions(profile, [job(description="掌握Python。")])
    assert profile == original
    assert output["questions"][0]["field"] == "Python"
    assert output["questions"][0]["answer_schema"]["type"] == "experience_block"
    assert output["questions"][0]["evidence"][0]["evidence"]["quote"] == "Python"


def test_no_recognized_requirements_keep_full_unknown_interval():
    target = job(description="负责创意策划以及团队联络。")
    assert evidence_bounds(person(), target)["coverage_bounds"] == [0, 1]
    assert evidence_bounds(person(), target)["requirement_count"] == 0


def test_unproven_task_binding_removes_false_lower_bound():
    target, profile = job(description="使用Python完成数据清洗。"), person("使用Python写接口。使用Excel完成数据清洗。")
    assert evidence_bounds(profile, target)["coverage_bounds"] == [1, 1]
    alignment = {"groups": [{"group_id": group["group_id"], "status": "unknown"} for group in target.groups]}
    assert evidence_bounds(profile, target, alignment)["coverage_bounds"] == [0, 1]


def test_preferred_requirements_do_not_block_coverage():
    assert evidence_bounds(person(), job(description="掌握Python。Java优先。"))["coverage_bounds"] == [1, 1]


def test_full_time_question_only_when_job_requires_it():
    assert any(q["field"] == "education_full_time" for q in plan_questions(person(education="本科", experience_years=5), [job(requirements="全日制本科，3年工作经验")])["questions"])
    assert not any(q["field"] == "education_full_time" for q in plan_questions(person(education="本科", experience_years=5), [job()])["questions"])


def row(key, bounds, salary, district="南山区", conflicts=()):
    target = job(key, salary=salary)
    return {**target.public(), "district": district, "decision_certificate": {"coverage_bounds": bounds, "hard_conflicts": list(conflicts)}}


def test_frontier_keeps_salary_evidence_tradeoff_and_unknown():
    rows = [row("supported", [1, 1], "10-12K"), row("higher_pay", [0.4, 0.7], "20-25K"), row("unknown", [0, 1], "面议")]
    assert set(comparison_frontier(person(), rows)["frontier_ids"]) == {"supported", "higher_pay", "unknown"}


def test_robust_dominance_and_irrelevant_candidate_invariance():
    strong, weak = row("strong", [1, 1], "20-25K"), row("weak", [.3, .5], "10-12K")
    original = comparison_frontier(person(), [strong, weak])
    assert original["frontier_ids"] == ["strong"]
    changed = comparison_frontier(person(), [strong, weak, row("extra", [0, .1], "5-6K")])
    assert original["points"][0]["axes"] == changed["points"][0]["axes"]
    assert changed["points"][1]["dominated_by"] == ["strong"]


def test_salary_overlap_cannot_prove_dominance_and_hard_fail_is_excluded():
    rows = [row("a", [1, 1], "10-20K"), row("b", [1, 1], "15-25K"), row("fail", [1, 1], "50-80K", conflicts=["学历要求"])]
    assert set(comparison_frontier(person(), rows)["frontier_ids"]) == {"a", "b"}


def test_unknown_district_is_not_assumed_close_and_zeros_are_confirmed():
    profile = person(experience_years=0, education="本科", district="南山区")
    assert not any(q["field"] == "experience_years" for q in plan_questions(profile, [job()])["questions"])
    result = comparison_frontier(profile, [row("a", [1, 1], "10-20K", district=None)])
    assert result["points"][0]["axes"]["preferred_district"] == [0, 1]


def test_unknown_eligibility_cannot_dominate_confirmed_eligibility_by_salary():
    higher = row("high", [1, 1], "30-40K")
    higher["decision_certificate"]["unknown_constraints"] = ["学历要求"]
    lower = row("known", [1, 1], "10-12K")
    assert set(comparison_frontier(person(), [higher, lower])["frontier_ids"]) == {"high", "known"}
