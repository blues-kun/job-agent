"""独立虚构反例：追问责任方、用户回填合同和稳健比较性质。"""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import json
import random
import re

from job_agent.decision_support import comparison_frontier, evidence_bounds, plan_questions
from job_agent.domain import Job, constraints, parse_profile
from job_agent.evidence_alignment import align_requirements


ROOT = Path(__file__).resolve().parents[1]


def _job(key="fictional", requirements="本科，5年工作经验", description="掌握Python。"):
    return Job(key, "fictional-v1", "虚构开发岗位", "虚构企业", "开发", "技术",
               "10-20K", requirements, description, "深圳南山区", 1)


def _person(text="我使用Python完成接口开发。", **overrides):
    return parse_profile(text, {"city": "深圳", "intent": "开发", "education": "本科",
                                "experience_years": 5, **overrides})


def test_specialized_duration_unknown_needs_candidate_question_not_total_years():
    target = _job(requirements="本科，工作经验5年，需3年Python开发经验")
    profile = _person()
    checks = constraints(profile, target)
    assert next(row for row in checks if row["name"] == "经验要求")["status"] == "pass"
    assert next(row for row in checks if row.get("scope") == "specialized")["status"] == "unknown"
    result = plan_questions(profile, [target])
    candidates = [q for q in result["questions"] if q["target_side"] == "candidate"]
    assert any("Python" in q["question"] and ("年" in q["question"] or "时长" in q["question"])
               and q["control"] == "resume-text" for q in candidates), "总年资不能替代Python专项时长；应提示在真实经历中补充"
    assert all(q["field"] != "experience_years" for q in candidates)


def test_unspecified_specialized_scope_requires_employer_clarification():
    target = _job(requirements="本科，工作经验5年，需3年专项经验")
    result = plan_questions(_person(), [target])
    employer = next(item for item in result["employer_checks"] if item["job_id"] == target.id)
    assert any("专项" in field for field in employer["fields"]), "岗位没有说明是哪项经验，不能要求用户猜测"


class _Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"input", "select", "textarea"} and attributes.get("id"):
            self.ids.add(attributes["id"])


def test_advertised_full_time_answer_control_is_editable_and_invalidates_confirmation():
    """静态DOM连接检查；不把它当作真实浏览器端到端测试。"""
    output = plan_questions(_person(education_full_time=None),
                            [_job(requirements="全日制本科，5年工作经验")])
    control = next(q["control"] for q in output["questions"] if q["field"] == "education_full_time")
    parser = _Controls()
    parser.feed((ROOT / "web/platform/index.html").read_text("utf-8"))
    assert control in parser.ids
    source = (ROOT / "web/platform/app.js").read_text("utf-8")
    invalidating_controls = set()
    # 验证现有统一监听入口覆盖了系统实际要求回填的字段，不固定字段顺序。
    registry = json.loads(re.search(r'const fields\s*=\s*(\[[^\n]+\]);', source).group(1))
    for match in re.finditer(r'for\s*\(const\s+id\s+of\s+(\[[^\n]+?\])\)\s*\$\(id\)\.addEventListener\(([^\n]+)', source):
        if 'invalidate()' in match.group(2):
            invalidating_controls.update(re.findall(r'"([^\"]+)"', match.group(1)))
            if 'fields.map(([id])=>id)' in match.group(1):
                invalidating_controls.update(row[0] for row in registry)
    assert control in invalidating_controls, "全日制字段变化必须撤销旧确认令牌并重新确认画像"


def test_known_specialized_duration_does_not_request_duplicate_evidence():
    target = _job(requirements="本科，工作经验5年，需3年Python开发经验")
    profile = _person("我有3年Python开发经验，使用Python完成接口开发。")
    assert next(row for row in constraints(profile, target) if row.get("scope") == "specialized")["status"] == "pass"
    output = plan_questions(profile, [target])
    assert not any("Python" in q["question"] and ("年" in q["question"] or "时长" in q["question"])
                   for q in output["questions"])


def test_positive_binding_answer_recomputes_text_support_without_mutating_previous_profile():
    target = _job(description="使用Python完成数据清洗。")
    before = _person("项目一：我使用Python完成接口开发。\n\n项目二：我使用Excel完成数据清洗。")
    preserved = deepcopy(before)
    alignment = align_requirements(before, target)
    output = plan_questions(before, [target], alignments={target.id: alignment})
    assert any(q["kind"] == "task_binding" and q["control"] == "resume-text" for q in output["questions"])
    assert before == preserved
    after = _person(before["text"] + "\n\n项目三：我使用Python完成数据清洗。")
    revised = align_requirements(after, target)
    assert evidence_bounds(before, target, alignment)["coverage_bounds"] == [0, 1]
    assert evidence_bounds(after, target, revised)["coverage_bounds"] == [1, 1]
    assert revised["summary"]["ability_verified"] is False


def test_unassessed_alignment_cannot_reuse_keyword_perfect_coverage():
    target, profile = _job(), _person()
    unassessed = {"groups": [], "summary": {"status": "not_evaluated"}}
    result = evidence_bounds(profile, target, unassessed)
    assert result["coverage_bounds"] == [0, 1]
    assert result["state"] == "needs_evidence"


def test_filled_or_alternative_cannot_gain_resolution_from_other_branch():
    target = _job(description="掌握Python或Java。")
    output = plan_questions(_person(), [target])
    assert not any(q["field"] in {"Python", "Java"} for q in output["questions"])


def test_frontier_matches_independent_endpoint_dominance_under_order_changes():
    """直接在原始薪资端点比较，不重用生产代码的薪资归一化函数。"""
    rng = random.Random(4923)
    raw, rows = {}, []
    for index in range(15):
        coverage = sorted(rng.sample(range(11), 2))
        salary = sorted(rng.sample(range(1, 40), 2))
        raw[str(index)] = [(x / 10) for x in coverage], [1000 * x for x in salary]
        rows.append({"id": str(index), "salary": {"status": "月薪", "monthly_low": salary[0] * 1000,
                      "monthly_high": salary[1] * 1000}, "district": "南山区",
                     "decision_certificate": {"coverage_bounds": raw[str(index)][0], "hard_conflicts": []}})
    expected = {}
    for key, axes in raw.items():
        dominators = set()
        for other_key, other_axes in raw.items():
            if key == other_key:
                continue
            pairs = [(other_axis[0], axis[1]) for other_axis, axis in zip(other_axes, axes)]
            if all(low >= high for low, high in pairs) and any(low > high for low, high in pairs):
                dominators.add(other_key)
        expected[key] = dominators
    for candidates in (rows, list(reversed(rows))):
        output = comparison_frontier(_person(), candidates)
        assert {p["job_id"]: set(p["dominated_by"]) for p in output["points"]} == expected
        assert set(output["frontier_ids"]) == {key for key, values in expected.items() if not values}


def test_unknown_pay_and_unknown_requirements_do_not_produce_false_dominance():
    unknown = {"id": "unknown", "salary": {"status": "未知"}, "district": None,
               "decision_certificate": {"coverage_bounds": [0, 1], "hard_conflicts": []}}
    known = {"id": "known", "salary": {"status": "月薪", "monthly_low": 20000, "monthly_high": 25000},
             "district": "南山区", "decision_certificate": {"coverage_bounds": [1, 1], "hard_conflicts": []}}
    assert set(comparison_frontier(_person(district="南山区"), [unknown, known])["frontier_ids"]) == {"unknown", "known"}
