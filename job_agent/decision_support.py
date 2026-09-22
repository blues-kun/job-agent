"""从可解析要求的逻辑区间生成追问与多目标比较，不估计录用概率。"""
from __future__ import annotations

from copy import deepcopy
from math import log2

from .domain import Job, constraints, display_text, group_support

VERSION = "decision-support-v1"
MAX_CANDIDATES = 20


def _required(profile, job):
    return [row for row in group_support(profile, job)
            if not row["preferred"] and row.get("modality") != "negated"]


def evidence_bounds(profile: dict, job: Job, alignment: dict | None = None) -> dict:
    """上下界是未知叶取真/假后的逻辑边界，绝非统计置信区间。"""
    groups = _required(profile, job)
    aligned = {row["group_id"]: row for row in (alignment or {}).get("groups", [])}
    not_evaluated = alignment is not None and alignment.get("summary", {}).get("status") == "not_evaluated"
    lows, highs, unresolved = [], [], []
    for group in groups:
        lower, upper = group["lower"], group["upper"]
        if not_evaluated:
            lower, upper = 0.0, 1.0
        proof = aligned.get(group["group_id"])
        if proof and proof["status"] != "pass":
            lower = 0.0
            if proof["status"] == "fail":
                upper = 0.0
        lows.append(lower)
        highs.append(upper)
        if lower != upper:
            unresolved.append(group["group_id"])
    count = len(groups)
    checks = constraints(profile, job)
    failed = [row["name"] for row in checks if row["status"] == "fail"]
    unknown = [row["name"] for row in checks if row["status"] == "unknown"]
    bounds = [round(sum(lows) / count, 6), round(sum(highs) / count, 6)] if count else [0.0, 1.0]
    return {"version": VERSION, "coverage_bounds": bounds, "requirement_count": count,
            "unresolved_group_ids": unresolved, "hard_conflicts": failed, "unknown_constraints": unknown,
            "state": "conflict" if failed else "needs_evidence" if unknown or not count or bounds[0] < 1 else "text_supported",
            "note": "区间仅覆盖当前解析器识别的必要要求；没有识别出要求时保持[0,1]。文本支持不等于经历已外部核实。"}


def _employer_unknown(profile, job):
    result = []
    if profile.get("city") and not job.city:
        result.append("工作城市")
    if profile.get("district") and not job.district:
        result.append("工作区域")
    if job.salary.get("status") != "月薪":
        result.append("可比月薪范围")
    if job.education_min is None or job.education_requirements.get("conflict"):
        result.append("学历门槛")
    if job.experience_min is None or job.experience_requirements.get("conflict"):
        result.append("总年资要求")
    if any(not row.get("skills") and not row.get("optional") and not row.get("negated") for row in job.experience_requirements.get("specialized", [])):
        result.append("专项经验的具体对象")
    return result


def plan_questions(profile: dict, jobs: list[Job], max_questions: int = 3, alignments: dict | None = None) -> dict:
    """计算不同回答可能改变的判断数；没有回答概率时不冒称期望信息收益。"""
    jobs = list({job.id: job for job in jobs}.values())[:MAX_CANDIDATES]
    questions = []
    fields = [
        ("education", "学历要求", range(6), "你的最高已取得学历是什么？", "education"),
        ("experience_years", "经验要求", sorted({0.0, 60.0, *[float(job.experience_min) for job in jobs if job.experience_min is not None],
          *[max(0.0, float(job.experience_requirements.get("total_maximum")) + .1) for job in jobs if job.experience_requirements.get("total_maximum") is not None]}),
         "你的总工作年资是多少？请与专项技能使用时长分开填写。", "years"),
        ("education_full_time", "全日制要求", [False, True], "你已取得的相关学历是否为全日制？", "education-full-time"),
    ]
    for field, name, values, title, control in fields:
        if profile.get(field) is not None:
            continue
        affected, scenarios = set(), []
        for value in values:
            simulated = {**profile, field: value}
            decisions = {}
            for job in jobs:
                before = next((row for row in constraints(profile, job) if row["name"] == name), None)
                after = next((row for row in constraints(simulated, job) if row["name"] == name), None)
                if before and after and before["status"] == "unknown" and after["status"] != "unknown":
                    affected.add(job.id)
                    decisions[job.id] = after["status"]
            scenarios.append({"assumed_value": value, "decisions": decisions})
        if not affected:
            continue
        partitions = len({tuple(sorted(row["decisions"].items())) for row in scenarios})
        questions.append({"id": "field:" + field, "kind": "profile_field", "target_side": "candidate",
                          "field": field, "control": control, "question": title,
                          "affected_job_ids": sorted(affected), "potential_resolution": len(affected),
                          "decision_partitions": partitions, "priority": round(len(affected) * (1 + log2(max(1, partitions))), 4),
                          "answer_schema": {"type": "confirmed_profile_field", "allows_unknown": True},
                          "scenarios": scenarios, "note": "假设回答仅用于挑选问题，未写入画像；修改后需重新确认。"})

    specialized = {}
    for job in jobs:
        unknown_names = {row["name"] for row in constraints(profile, job) if row["status"] == "unknown"}
        for requirement in job.experience_requirements.get("specialized", []):
            keys = sorted(requirement.get("skills", []))
            label = " / ".join(keys)
            if not keys or requirement.get("optional") or requirement.get("negated") or f"专项经验：{label}" not in unknown_names:
                continue
            entry = specialized.setdefault(label, {"job_ids": set(), "evidence": []})
            entry["job_ids"].add(job.id)
            entry["evidence"].append({"job_id": job.id, "evidence": requirement["evidence"]})
    for label, item in sorted(specialized.items()):
        questions.append({"id": "specialized:" + label, "kind": "specialized_experience", "target_side": "candidate",
                          "field": "specialized_experience", "control": "resume-text",
                          "question": f"你实际使用“{label}”的专项年资是多少？请在简历中明确，例如“3年{label}开发经验”；总工作年资不能替代这项条件。",
                          "affected_job_ids": sorted(item["job_ids"]), "potential_resolution": len(item["job_ids"]),
                          "priority": float(2 * len(item["job_ids"])), "evidence": item["evidence"],
                          "answer_schema": {"type": "experience_block", "allows_unknown": True},
                          "note": "请按真实经历填写；示例数字只是输入格式，不会自动写入画像。"})
    # 相同技能跨岗位合并；在完整布尔树中模拟，已满足OR分支不会重复追问另一选项。
    uncertain = {}
    for job in jobs:
        for group in _required(profile, job):
            if group["status"] != "unknown" or group.get("parse_status") != "known":
                continue
            for leaf in group["leaf_support"]:
                if leaf["status"] == "unknown" and leaf.get("modality") not in {"negated", "preferred", "unknown"}:
                    uncertain.setdefault((leaf["type"], leaf["key"]), leaf["text"])
    for (kind, key), label in sorted(uncertain.items())[:80]:
        positive, negative = deepcopy(profile), deepcopy(profile)
        field = "skills" if kind == "skill" else "tasks"
        positive.setdefault(field, {})[key] = {"weight": 1, "parse_status": "known", "actor": "self", "polarity": "positive"}
        negative.setdefault(field, {})[key] = {"weight": 0, "parse_status": "known", "actor": "self", "polarity": "negative"}
        impact, affected, evidence = 0.0, [], []
        for job in jobs:
            before = _required(profile, job)
            yes, no = _required(positive, job), _required(negative, job)
            resolution = sum(max(0, (old["upper"] - old["lower"]) - (new["upper"] - new["lower"]))
                             for old, new in zip(before, yes))
            resolution += sum(max(0, (old["upper"] - old["lower"]) - (new["upper"] - new["lower"]))
                              for old, new in zip(before, no))
            if resolution:
                impact += resolution / (2 * max(1, len(before)))
                affected.append(job.id)
                evidence.extend({"job_id": job.id, "group_id": group["group_id"], "evidence": leaf["job_evidence"]}
                                for group in before if group["status"] == "unknown" for leaf in group["leaf_support"]
                                if leaf["type"] == kind and leaf["key"] == key)
        if affected:
            questions.append({"id": f"evidence:{kind}:{key}", "kind": "experience_evidence", "target_side": "candidate",
                              "field": key, "control": "resume-text",
                              "question": (f"你是否亲自使用过“{display_text(label)}”完成具体工作？" if kind == "skill" else f"你是否亲自完成过“{display_text(label)}”？") + "请补充任务、操作和结果；没有相关经历可如实说明。",
                              "affected_job_ids": affected, "potential_resolution": round(impact, 4),
                              "priority": round(impact, 4), "evidence": evidence[:20],
                              "answer_schema": {"type": "experience_block", "allows_unknown": True},
                              "note": "正反两种假设的逻辑区间缩减均值，仅为追问优先级代理；不代表回答概率或学习收益。"})
    # 原子技能都命中但工具与任务未绑定时，追问关系，不再次要求罗列技能。
    bindings = {}
    for job in jobs:
        required_ids = {row["group_id"] for row in _required(profile, job)}
        for group in (alignments or {}).get(job.id, {}).get("groups", []):
            binding = group.get("task_binding", {})
            if group["group_id"] not in required_ids or not binding.get("required") or binding.get("status") == "pass":
                continue
            for relation in binding.get("requirements", []):
                quote = relation["job_evidence"]["quote"]
                key = display_text(quote)
                entry = bindings.setdefault(key, {"job_ids": set(), "evidence": []})
                entry["job_ids"].add(job.id)
                entry["evidence"].append({"job_id": job.id, "group_id": group["group_id"], "evidence": relation["job_evidence"]})
    for index, (quote, item) in enumerate(sorted(bindings.items())):
        questions.append({"id": f"binding:{index}", "kind": "task_binding", "target_side": "candidate", "field": "experience_block",
                          "control": "resume-text", "question": f"对于“{quote}”，你是否亲自在同一个任务中使用过这些工具？请补充对应经历。",
                          "affected_job_ids": sorted(item["job_ids"]), "potential_resolution": len(item["job_ids"]),
                          "priority": float(len(item["job_ids"])), "evidence": item["evidence"][:20],
                          "answer_schema": {"type": "experience_block", "allows_unknown": True},
                          "note": "多个技能分别出现不能证明工具与任务的对应关系；回答不会自动记为已核验能力。"})
    questions.sort(key=lambda row: (-row["priority"], row["id"]))
    employer = [{"job_id": job.id, "fields": _employer_unknown(profile, job), "active_status": "unknown"} for job in jobs]
    return {"version": VERSION, "questions": questions[:max(0, min(max_questions, 3))],
            "candidate_count": len(jobs), "question_count_before_budget": len(questions),
            "employer_checks": employer,
            "method": "有界反事实回答的判断影响，未学习回答概率；岗位未知字段向招聘方确认。"}


def _pay_utility(value):
    return value / (value + 20000.0)


def comparison_frontier(profile: dict, rows: list[dict]) -> dict:
    """只在本次返回的、已通过硬过滤的候选中比较。缺失字段保留区间。"""
    points = []
    for row in rows[:MAX_CANDIDATES]:
        certificate = row["decision_certificate"]
        if certificate["hard_conflicts"]:
            continue
        salary = row.get("salary", {})
        pay = [_pay_utility(salary["monthly_low"]), _pay_utility(salary["monthly_high"])] if salary.get("status") == "月薪" else [0.0, 1.0]
        district = ([float(row.get("district") == profile["district"])] * 2 if row.get("district") else [0.0, 1.0]) if profile.get("district") else None
        axes = {"evidence": certificate["coverage_bounds"], "advertised_salary": pay,
                "hard_conditions": [0.0, 1.0] if certificate.get("unknown_constraints") else [1.0, 1.0]}
        if district is not None:
            axes["preferred_district"] = district
        points.append({"job_id": row["id"], "axes": axes, "dominated_by": []})
    for point in points:
        for other in points:
            if point is other:
                continue
            pairs = [(other["axes"][key][0], bounds[1]) for key, bounds in point["axes"].items()]
            if all(low >= high for low, high in pairs) and any(low > high for low, high in pairs):
                point["dominated_by"].append(other["job_id"])
        point["on_frontier"] = not point["dominated_by"]
    return {"version": VERSION, "points": points, "frontier_ids": [p["job_id"] for p in points if p["on_frontier"]],
            "scope": "本次展示候选；薪资为广告区间，区域匹配不估计通勤时间。",
            "note": "只有一个候选在所有比较维度的下界都不低于另一个的上界时才判为稳健占优；保留取舍与未知。前沿不代表录用概率或全库最优。"}
