"""运行固定合成工程反例矩阵，输出JSON和中文说明；不读取私有数据、不调用模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from job_agent.action_planner import plan_actions
from job_agent import action_planner, decision_support, evidence_alignment, domain, semantics
from job_agent.decision_support import comparison_frontier, evidence_bounds, plan_questions
from job_agent.domain import Job, group_support, parse_profile
from job_agent.evidence_alignment import align_requirements


VERSION = "decision-innovation-fixtures-v1"
DISCLAIMER = "本报告只验证固定合成反例的工程行为；不是独立人岗效果评测，不证明真实用户收益或统计显著性。"
PREFERENCES = {"city": "深圳", "intent": "数据分析", "education": "本科", "experience_years": 1}


def _job(description, requirements="本科；经验不限。", identifier="fixture-job", salary="10-20K"):
    return Job(identifier, "fixture-v1", "虚构研究岗位", "虚构研究企业", "数据分析", "技术", salary,
               requirements, description, "深圳南山区", 1)


def _profile(text, **changes):
    return parse_profile(text, {**PREFERENCES, **changes})


def _matches(actual, expected):
    return all(actual.get(key) == value for key, value in expected.items())


def _case(identifier, dimension, scenario, inputs, expected, baseline, current, attribution):
    return {"case_id": identifier, "dimension": dimension, "scenario": scenario, "inputs": inputs,
            "expected": expected, "baseline": {**baseline, "expected_met": _matches(baseline["actual"], expected)},
            "current": {**current, "expected_met": _matches(current["actual"], expected)}, "attribution": attribution}


def _state(rows, field):
    values = [row[field] for row in rows]
    return "fail" if "fail" in values else "pass" if values and all(value == "pass" for value in values) else "unknown"


def _alignment_cases():
    # 预期来自显式场景定义，不从待测函数推导。六个负例都有Python和清洗自述，缺的是动作绑定。
    fixtures = [
        ("separate_projects", "使用Python开发订单接口。\n\n项目二：使用Excel完成数据清洗。", "unknown", "跨项目工具与任务不能拼成同一操作"),
        ("separate_actions", "使用Python开发接口，使用Excel完成数据清洗。", "unknown", "同一段中的两种操作仍有不同工具"),
        ("skill_and_task", "熟悉Python。负责数据清洗。", "unknown", "掌握工具和承担任务分别成立，不证明工具用于该任务"),
        ("different_tool", "使用Python编写接口。负责数据清洗，但处理时只使用Excel。", "unknown", "清洗使用的工具与岗位要求不一致，不能确认Python清洗经历"),
        ("skill_section", "技能清单：熟悉Python。\n\n项目经历：使用Excel完成数据清洗。", "unknown", "技能清单不能替代具体任务中的操作证据"),
        ("different_work", "项目一：使用Python编写自动化测试。\n\n项目二：开展数据清洗。", "unknown", "另一个项目只说明任务，没有说明使用Python"),
        ("positive_control", "项目一：使用Python完成数据清洗。", "pass", "同一动作中明确记录工具和任务，文本支持成立"),
        ("negative_control", "项目一：不会使用Python完成数据清洗。", "fail", "明确否定不能当成支持"),
        ("team_control", "项目一：团队使用Python完成数据清洗，我负责项目协调。", "unknown", "团队行为不能直接归于本人"),
        ("future_control", "项目一：计划使用Python完成数据清洗。", "unknown", "计划不代表已完成经历"),
    ]
    target = _job("负责使用Python完成数据清洗。")
    cases = []
    for identifier, text, expected, reason in fixtures:
        aligned = align_requirements(_profile(text), target)
        groups = [row for row in aligned["groups"] if row["task_binding"]["required"]]
        cases.append(_case("alignment_"+identifier, "动作证据对齐", reason,
            {"resume": text, "jd": target.description}, {"state": expected},
            {"method": "现有全局技能/任务三态匹配", "actual": {"state": _state(groups, "global_status")}},
            {"method": "要求组与同动作工具—任务绑定", "actual": {"state": _state(groups, "status"),
                "bound_groups": len(groups), "downgraded_groups": aligned["summary"]["downgraded_groups"]}}, reason))
    return cases


def _question_cases():
    cases = []
    configurations = [
        ("tenure_only", "本科。使用Python编写接口。", {"experience_years": None}, "本科；要求3年以上工作经验。",
         ["experience_years"], [], "学历已知，候选岗位之间的年资判断仍未知，应先核对本人总年资"),
        ("education_only", "使用Python编写接口。", {"education": "", "experience_years": 1}, "本科；经验不限。",
         ["education"], [], "本人学历未知而岗位门槛明确，学历追问在这个控制样例中有用"),
        ("employer_unknown", "本科。使用Python编写接口。", {}, "本科。",
         [], ["总年资要求"], "缺失来自JD，不能通过反问求职者补出招聘方门槛"),
        ("or_satisfied", "本科。使用Python编写接口。", {}, "本科；经验不限。",
         [], [], "Python已满足OR分支，不应强制追问Java或重复询问已知学历"),
    ]
    for identifier, text, changes, requirements, relevant, employer_required, reason in configurations:
        target = _job("熟悉Python或Java。" if identifier == "or_satisfied" else "熟悉Python。", requirements)
        person = _profile(text, **changes); before = deepcopy(person)
        result = plan_questions(person, [target], max_questions=3)
        fields = [row["field"] for row in result["questions"]]
        employer = [field for row in result["employer_checks"] for field in row["fields"]]
        actual = {"useful_questions": sum(field in relevant for field in fields),
                  "irrelevant_questions": sum(field not in relevant for field in fields),
                  "employer_routed": all(field in employer for field in employer_required),
                  "profile_unchanged": person == before, "question_fields": fields}
        baseline_fields = ["education"]
        baseline = {"useful_questions": sum(field in relevant for field in baseline_fields),
                    "irrelevant_questions": sum(field not in relevant for field in baseline_fields),
                    "employer_routed": not employer_required, "profile_unchanged": True, "question_fields": baseline_fields}
        cases.append(_case("question_"+identifier, "主动追问", reason,
            {"resume": text, "preferences": changes, "jd": requirements+target.description},
            {"useful_questions": len(relevant), "irrelevant_questions": 0, "employer_routed": True, "profile_unchanged": True},
            {"method": "对照启发式：固定先问学历（不是旧平台实际策略）", "actual": baseline},
            {"method": "候选条件与逻辑区间驱动的追问", "actual": actual}, reason))
    return cases


def _plan_cases():
    cases = []
    fixtures = [
        ("and", "熟悉Python和SQL。", "暂无相关项目描述。", [], 2,
         [{"Python", "SQL"}], "AND需要两个技能的条件式行动，不能只补一个"),
        ("or", "熟悉Python或Java。", "暂无相关项目描述。", [], 1,
         [{"Python"}, {"Java"}], "OR只需一个完整分支，不应把两个选项都列为必要行动"),
        ("nested", "熟悉(Python或Java)和(SQL或MySQL)。", "暂无相关项目描述。", [], 2,
         [{"Python", "SQL"}, {"Python", "MySQL"}, {"Java", "SQL"}, {"Java", "MySQL"}], "两个OR组必须各选一项，平铺会多要求两项"),
        ("supported_or", "熟悉Python或Java。", "使用Python开发接口。", ["Python"], 0,
         [{"Python"}, {"Java"}], "一个分支已有自述支持，不再将另一选项当必要缺口"),
    ]
    for identifier, jd, text, already, minimum, legal_branches, reason in fixtures:
        target = _job(jd); person = _profile(text); before = deepcopy(person)
        result = plan_actions(person, target, max_plans=3, alignment=align_requirements(person, target))
        flat_missing = sorted({leaf["key"] for group in group_support(person, target) for leaf in group["leaf_support"]
                               if leaf["type"] == "skill" and leaf["status"] != "pass"})
        plans = result["plans"]
        sets = [{row["key"] for row in plan["actions"] if row["type"] == "skill"} for plan in plans]
        complete = bool(plans) and all(any(branch <= (actions | set(already)) for branch in legal_branches) for actions in sets)
        counts = [plan["action_count"] for plan in plans]
        actual = {"minimum_actions": min(counts) if counts else None, "logical_completeness": complete,
            "conditional_only": all(plan["conditional"] and all(row["conditional"] for row in plan["actions"]) for plan in plans),
            "profile_unchanged": before == person, "action_sets": [sorted(keys) for keys in sets], "planner_status": result["status"]}
        baseline = {"minimum_actions": len(flat_missing),
                    "logical_completeness": any(branch <= (set(flat_missing) | set(already)) for branch in legal_branches),
                    "conditional_only": True, "profile_unchanged": True, "action_sets": [flat_missing]}
        cases.append(_case("plan_"+identifier, "逻辑行动规划", reason, {"resume": text, "jd": jd},
            {"minimum_actions": minimum, "logical_completeness": True, "conditional_only": True, "profile_unchanged": True},
            {"method": "对照启发式：平铺未出现的全部技能选项", "actual": baseline},
            {"method": "保留AND/OR的最小条件式行动集", "actual": actual}, reason))
    target = _job("使用Python完成数据清洗。")
    for name, text, expected_count in [
        ("binding_gap", "使用Python开发接口。\n\n使用Excel完成数据清洗。", 1),
        ("binding_control", "使用Python完成数据清洗。", 0),
    ]:
        person = _profile(text)
        baseline = plan_actions(person, target)
        current = plan_actions(person, target, alignment=align_requirements(person, target))
        def binding_counts(result):
            actions = result["plans"][0]["actions"] if result["plans"] else []
            return {"binding_actions": sum(row["type"] == "binding" for row in actions),
                    "conditional_only": all(row["conditional"] for row in actions)}
        cases.append(_case("plan_"+name, "逻辑行动规划", "已出现的技能与任务不应掩盖缺少动作关联的补证需求",
            {"resume": text, "jd": target.description}, {"binding_actions": expected_count, "conditional_only": True},
            {"method": "只按全局叶支持规划", "actual": binding_counts(baseline)},
            {"method": "加入动作关联的条件式补证规划", "actual": binding_counts(current)},
            "关联缺口只要求核对既有经历，不把行动清单写成已经会做；同一绑定在两个要求组中去重。"))
    return cases


def _interval_cases():
    fixtures = [
        ("missing_skill", "使用Python开发接口。", "熟悉Python。掌握SQL。", [.5, 1.0],
         "SQL没有材料是未知，上界应保留，不能当作明确不会"),
        ("negative_skill", "使用Python开发接口。不会SQL。", "熟悉Python。掌握SQL。", [.5, .5],
         "明确否定与缺证据不同，SQL分支不再保留可支持上界"),
        ("no_requirements", "使用Python开发接口。", "负责未收录的专业工作。", [0.0, 1.0],
         "没有解析出要求应标未知，不能解释为全部满足或完全不匹配"),
        ("bound_gap", "使用Python写接口。\n\n使用Excel完成数据清洗。", "使用Python完成数据清洗。", [0.0, 1.0],
         "原全局分数满支持，但跨动作证据不能关闭逻辑不确定区间"),
    ]
    cases = []
    for identifier, text, jd, expected_bounds, reason in fixtures:
        person = _profile(text); target = _job(jd)
        certificate = evidence_bounds(person, target, align_requirements(person, target))
        supports = group_support(person, target)
        baseline_point = sum(row["lower"] for row in supports)/len(supports) if supports else 0.0
        cases.append(_case("interval_"+identifier, "区间与多目标比较", reason, {"resume": text, "jd": jd},
            {"coverage_bounds": expected_bounds},
            {"method": "对照启发式：独立条件计分，缺失当0且不检查动作绑定", "actual": {"coverage_bounds": [baseline_point, baseline_point]}},
            {"method": "要求组逻辑区间＋动作绑定", "actual": {"coverage_bounds": certificate["coverage_bounds"],
                "state": certificate["state"], "requirement_count": certificate["requirement_count"]}}, reason))

    person = _profile("使用Python开发接口。")
    # 人工设定的可比较候选，不从排名函数反推预期：强证据/高薪互有取舍，未知项不能填0淘汰。
    specs = [("strong-evidence", [1.0, 1.0], "10-10K"), ("higher-pay", [.5, .5], "30-30K"),
             ("unknown-record", [0.0, 1.0], "面议"), ("dominated", [.25, .25], "5-5K")]
    rows = []
    for identifier, coverage, salary in specs:
        row = _job("熟悉Python。", identifier=identifier, salary=salary).public()
        row["decision_certificate"] = {"coverage_bounds": coverage, "hard_conflicts": []}
        rows.append(row)
    frontier = comparison_frontier(person, rows)
    points = {row["job_id"]: row for row in frontier["points"]}
    baseline_points = {identifier: [coverage[0], 0 if salary == "面议" else float(salary.split('-')[0])] for identifier, coverage, salary in specs}
    baseline_frontier = []
    for identifier, axes in baseline_points.items():
        if not any(all(a >= b for a, b in zip(other, axes)) and any(a > b for a, b in zip(other, axes))
                   for other_id, other in baseline_points.items() if other_id != identifier):
            baseline_frontier.append(identifier)
    current = {"frontier_ids": sorted(frontier["frontier_ids"]),
               "unknown_salary_preserved": points["unknown-record"]["axes"]["advertised_salary"] == [0.0, 1.0],
               "dominated_removed": not points["dominated"]["on_frontier"]}
    cases.append(_case("frontier_tradeoffs", "区间与多目标比较", "保留证据—薪资取舍与未知岗位，剔除明确被支配岗位",
        {"candidates": [{"id": identifier, "evidence_interval": bounds, "salary": salary} for identifier, bounds, salary in specs]},
        {"frontier_ids": ["higher-pay", "strong-evidence", "unknown-record"], "unknown_salary_preserved": True, "dominated_removed": True},
        {"method": "对照启发式：缺失当0后的点值Pareto", "actual": {"frontier_ids": sorted(baseline_frontier),
             "unknown_salary_preserved": False, "dominated_removed": "dominated" not in baseline_frontier}},
        {"method": "逻辑区间上的稳健Pareto", "actual": current},
        "未知记录不一定优秀；在证据不足时不能证明其被支配。前沿只表示当前候选间的取舍，不保证适岗。"))
    return cases


def build_report():
    cases = [*_alignment_cases(), *_question_cases(), *_plan_cases(), *_interval_cases()]
    dimensions = list(dict.fromkeys(case["dimension"] for case in cases))
    aggregates = {dimension: {"cases": sum(case["dimension"] == dimension for case in cases),
        "baseline_expected_met": sum(case["dimension"] == dimension and case["baseline"]["expected_met"] for case in cases),
        "current_expected_met": sum(case["dimension"] == dimension and case["current"]["expected_met"] for case in cases)} for dimension in dimensions}
    negatives = cases[:6]
    sources = {"research/evaluate_decision_innovations.py": Path(__file__),
        **{"job_agent/"+Path(module.__file__).name: Path(module.__file__)
           for module in [action_planner, decision_support, evidence_alignment, domain, semantics]}}
    inputs = [{"case_id": case["case_id"], "inputs": case["inputs"], "expected": case["expected"]} for case in cases]
    return {"schema": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": DISCLAIMER, "all_inputs_synthetic": True, "human_gold_count": 0,
        "real_person_job_effect_measured": False, "statistical_significance": "未估计，不适用于该固定工程矩阵",
        "external_api_calls": 0, "training_runs": 0,
        "versions": {"alignment": evidence_alignment.VERSION, "decision": decision_support.VERSION,
                     "planner": action_planner.VERSION, "parser": domain.PARSER_VERSION},
        "source_files_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "fixture_spec_sha256": hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "summary": {"cases": len(cases), "current_expected_met": sum(case["current"]["expected_met"] for case in cases),
                    "baseline_expected_met": sum(case["baseline"]["expected_met"] for case in cases),
                    "constructed_binding_negatives": len(negatives),
                    "baseline_false_support": sum(case["baseline"]["actual"]["state"] == "pass" for case in negatives),
                    "current_false_support": sum(case["current"]["actual"]["state"] == "pass" for case in negatives)},
        "dimensions": aggregates, "cases": cases,
        "limitations": ["每项预期由固定场景人工式定义，不来自真实用户标注；禁止将通过数解释为人岗准确率。",
            "对照中只有全局技能/任务匹配对应已有代码；固定先问学历、平铺缺口、缺失当0都是明示的对照启发式。",
            "样本选择旨在暴露特定工程错误，不具有随机性、总体代表性或统计显著性。",
            "最小行动数只计算条件式行动条目，不估计学习成本、真实能力变化或录用机会。",
            "逻辑上下界不是统计置信区间；稳健前沿不是全库最优或录用概率。"]}


def markdown(report):
    lines = ["# 决策组件固定工程反例评测", "", DISCLAIMER, "", "## 汇总", "",
             "| 检查维度 | 固定案例数 | 对照符合预期 | 当前组件符合预期 |", "|---|---:|---:|---:|"]
    for name, values in report["dimensions"].items():
        lines.append(f"| {name} | {values['cases']} | {values['baseline_expected_met']} | {values['current_expected_met']} |")
    lines += ["", f"六个指定动作混配反例：旧全局判定错误支持 {report['summary']['baseline_false_support']} 个，当前绑定判定错误支持 {report['summary']['current_false_support']} 个。该计数只适用于这些固定输入。", "", "## 逐例归因", ""]
    for case in report["cases"]:
        lines += [f"### {case['case_id']}", "", case["scenario"], "",
                  "- 预期：`"+json.dumps(case["expected"], ensure_ascii=False, sort_keys=True)+"`",
                  "- 对照（"+case["baseline"]["method"]+"）：`"+json.dumps(case["baseline"]["actual"], ensure_ascii=False, sort_keys=True)+"`",
                  "- 当前（"+case["current"]["method"]+"）：`"+json.dumps(case["current"]["actual"], ensure_ascii=False, sort_keys=True)+"`",
                  "- 检查："+("符合预期" if case["current"]["expected_met"] else "未符合预期，需要检查"),
                  "- 归因："+case["attribution"], ""]
    lines += ["## 来源与解释范围", "", "固定样例摘要：`"+report["fixture_spec_sha256"]+"`。源码摘要、版本与全部输入保存在同目录JSON中。", ""]
    lines.extend("- "+item for item in report["limitations"])
    return "\n".join(lines)+"\n"


def run(output_dir: Path):
    output_dir = Path(output_dir)
    if any(path.is_symlink() for path in (output_dir, *output_dir.parents)):
        raise ValueError("输出目录及其父目录不能是符号链接")
    if output_dir.exists():
        raise FileExistsError("输出目录已存在，请使用新的版本目录")
    report = build_report()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output_dir/"REPORT.md").write_text(markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="新的评测输出目录；只写固定虚构输入与聚合结果")
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({"summary": report["summary"], "scope": report["scope"]}, ensure_ascii=False))
    if report["summary"]["current_expected_met"] != report["summary"]["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
