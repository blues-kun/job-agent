"""保留 AND/OR 作用域的有限补证规划；行动只是待办，不是能力或录用承诺。"""
from __future__ import annotations

from copy import deepcopy

from .domain import Job, canonical, constraints, digest, group_support

VERSION = "evidence-action-planner-v1"
MAX_ACTIONS = 8
MAX_STATES = 256
MAX_EXPANSIONS = 4096
MAX_NODES = 512
MAX_DEPTH = 16


class _BudgetExceeded(Exception):
    pass


def _citation(job, evidence):
    if not isinstance(evidence, dict) or evidence.get("field") not in {"description", "requirements"}:
        return None
    start, end = evidence.get("start"), evidence.get("end")
    text = getattr(job, evidence["field"])
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        return None
    if text[start:end] != evidence.get("quote"):
        return None
    return {key: evidence[key] for key in ("field", "start", "end", "quote")}


def _constraint_rows(profile, job):
    rows = []
    for check in constraints(profile, job):
        if check["status"] == "pass":
            continue
        name = check["name"]
        unknown_job = (
            name == "目标城市" and not job.city
            or name == "最低月薪" and job.salary.get("status") != "月薪"
            or name == "学历要求" and (job.education_min is None or job.education_requirements.get("conflict"))
            or name == "经验要求" and (job.experience_min is None or job.experience_requirements.get("conflict"))
            or name.startswith("专项经验") and "未明确专项" in name
        )
        recipient = "employer" if unknown_job else "user"
        rows.append({**deepcopy(check), "kind": "hard_constraint", "recipient": recipient,
                     "action": "向招聘方确认原文门槛或计薪口径" if unknown_job else
                               "当前存在明确条件冲突；补技能或补文字不能立即改变该条件" if check["status"] == "fail" else
                               "核对本人实际条件；未填写不等同满足或不满足",
                     "resolvable_by_skill_actions": False})
    return rows


def _preflight(job):
    """先限制递归规模，再调用现有三态计算，避免复杂图绕过搜索预算。"""
    count = 0
    stack = [(group.get("ast"), 0) for group in job.groups]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            return False, "graph_budget", count
        if not isinstance(node, dict):
            return False, "invalid_graph", count
        if "type" in node:
            if node.get("type") not in {"skill", "task"} or not isinstance(node.get("key"), str):
                return False, "invalid_leaf", count
            if not _citation(job, node.get("evidence")):
                return False, "invalid_citation", count
        else:
            children = node.get("children")
            if node.get("op") not in {"all", "any"} or not isinstance(children, list) or not children:
                return False, "invalid_operator", count
            stack.extend((child, depth + 1) for child in children)
    root = job.requirement_ast
    if not isinstance(root, dict) or root.get("op") != "all" or root.get("children") != [g["ast"] for g in job.groups]:
        return False, "group_root_mismatch", count
    return True, None, count


def plan_actions(profile: dict, job: Job, max_plans: int = 3, alignment: dict | None = None) -> dict:
    """返回条件式最小行动集，不改画像；未知先补证，明确否定才列学习行动。

    中间状态仅按集合包含关系安全剪枝，最终按不同技能/任务/绑定行动数做 Pareto。
    未知逻辑、无效引用和预算不足不假装得到完整解；硬条件单独核对。
    """
    if type(max_plans) is not int or not 1 <= max_plans <= 3:
        raise ValueError("max_plans 必须是 1 到 3 的整数")
    if alignment is not None and not isinstance(alignment, dict):
        raise ValueError("alignment 必须是对齐结果对象或 None")
    output = {"version": VERSION, "job_id": job.id, "job_version": job.version,
              "profile_version": profile.get("version"), "parser_version": job.parser_version,
              "plans": [], "optional_actions": [], "non_actionable_constraints": _constraint_rows(profile, job),
              "truncated": False, "truncation_reasons": [], "unresolved_group_ids": [], "optional_unresolved_group_ids": [],
              "alignment_status": "not_requested" if alignment is None else "not_evaluated",
              "notes": ["行动是待办和条件式路径，不代表已经具备能力、岗位已匹配或能够录用。",
                        "未知技能先请求真实已有项目证据；没有经历时应保留未知，不能为了补证编造经历。",
                        "学习后仍需实际操作、证据核对及招聘方评估；不估计学习时长或录用概率。",
                        "最小只指不同技能/任务行动数量，不代表学习难度或时间最少；学历、薪资与年资另行核对。"]}
    output["search"] = {"nodes": 0, "expansions": 0, "max_nodes": MAX_NODES, "max_depth": MAX_DEPTH,
                        "max_states": MAX_STATES, "max_expansions": MAX_EXPANSIONS, "max_actions": MAX_ACTIONS}
    if not job.groups:
        output["status"] = "no_parsed_requirements"
        output["notes"].append("没有可解析的要求组不等于岗位没有要求；请向招聘方确认关键职责及门槛。")
        return output
    valid, reason, nodes = _preflight(job)
    output["search"]["nodes"] = nodes
    if not valid:
        output.update(status="needs_requirement_confirmation", truncated=True, truncation_reasons=[reason])
        output["notes"].append("需求图、引用或复杂度未通过检查，未生成假定完整的技能方案。")
        return output

    support = {row["group_id"]: row for row in group_support(profile, job)}
    registry = {}
    optional = []
    unresolved = set()
    optional_unresolved = set()
    processed_optional = set()
    binding_keys = {}
    aligned_groups = {}
    if alignment is not None:
        rows = alignment.get("groups", []) if isinstance(alignment, dict) else []
        if (isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
                and alignment.get("job_id") == job.id and alignment.get("profile_version") == profile.get("version")
                and alignment.get("summary", {}).get("status") == "completed"
                and len(rows) == len(job.groups) and {row.get("group_id") for row in rows} == set(support)):
            output["alignment_status"] = "completed"
            aligned_groups = {row["group_id"]: row for row in rows}
        else:
            output["notes"].append("工具与任务的对齐未完成或版本不一致，不能据全局技能命中声称全部要求已覆盖。")

    def mark_truncated(reason):
        output["truncated"] = True
        if reason not in output["truncation_reasons"]:
            output["truncation_reasons"].append(reason)

    def tick():
        output["search"]["expansions"] += 1
        if output["search"]["expansions"] > MAX_EXPANSIONS:
            mark_truncated("expansion_budget")
            raise _BudgetExceeded

    def prune(candidates):
        # 局部按条目数删方案会错过后续重复技能带来的更优全局解。
        result = []
        for keys, choices in sorted(candidates, key=lambda row: (len(row[0]), sorted(row[0]), row[1])):
            if len(keys) > MAX_ACTIONS:
                mark_truncated("action_limit")
                continue
            if any(old_keys <= keys for old_keys, _ in result):
                continue
            result.append((keys, choices))
        if len(result) > MAX_STATES:
            mark_truncated("state_budget")
            result = result[:MAX_STATES]
        return result

    def unresolved_node(group_id, node, path, preferred=False):
        (optional_unresolved if preferred else unresolved).add(group_id)
        evidence = _citation(job, node.get("scope_evidence") or node.get("evidence"))
        output["non_actionable_constraints"].append({"kind": "requirement_clarification", "name": "需求作用域或必需程度",
            "status": "unknown", "group_id": group_id, "node_path": list(path), "job_evidence": evidence, "preferred": preferred,
            "recipient": "employer", "resolvable_by_skill_actions": False,
            "action": "向招聘方确认替代分支、组合要求或必需程度；目前不能断言补哪几项就足够"})

    def prepare_bindings():
        for group in job.groups:
            group_id = group["group_id"]
            binding = aligned_groups.get(group_id, {}).get("task_binding", {})
            if not binding.get("required") or binding.get("status") == "pass":
                continue
            relations = binding.get("requirements")
            if not isinstance(relations, list) or not relations:
                output["alignment_status"] = "not_evaluated"
                continue
            for index, relation in enumerate(relations):
                tick()
                if not isinstance(relation, dict):
                    output["alignment_status"] = "not_evaluated"
                    continue
                citation = _citation(job, relation.get("job_evidence"))
                scope = _citation(job, relation.get("job_scope"))
                if not citation or not scope or not isinstance(relation.get("tools"), dict) or not relation.get("task_key"):
                    output["alignment_status"] = "not_evaluated"
                    continue
                identity = [relation["task_key"], relation["tools"], citation, scope]
                key = ("binding", digest(canonical(identity))[:20])
                reference = {"group_id": group_id, "node_path": ["binding", index], "job_evidence": citation, "job_scope": scope}
                if key not in registry:
                    registry[key] = {"action_id": digest(canonical([VERSION, *key]))[:20], "type": "binding", "key": key[1],
                        "label": str(relation.get("task") or relation["task_key"]) + "的工具—任务关联",
                        "kind": "task_binding", "current_status": "unknown", "conditional": True,
                        "task_key": relation["task_key"], "tool_requirement": deepcopy(relation["tools"]),
                        "group_ids": [], "requirements": [], "completion_is_not_verified_competence": True,
                        "instruction": "若你确实在同一操作场景中使用这些工具完成该任务，请补充本人步骤、工具选择与结果依据；若未做过，不把不同项目拼接成已有实践，可另列真实实践目标"}
                action = registry[key]
                if group_id not in action["group_ids"]:
                    action["group_ids"].append(group_id)
                if reference not in action["requirements"]:
                    action["requirements"].append(reference)
                binding_keys.setdefault(group_id, set()).add(key)

    try:
        prepare_bindings()
    except _BudgetExceeded:
        output["status"] = "bounded"
        output["notes"].append("绑定关系数量超出搜索预算，未生成不完整方案。")
        return output

    def attach_bindings(candidates, group_id):
        keys = frozenset(binding_keys.get(group_id, ()))
        return prune([(old_keys | keys, choices) for old_keys, choices in candidates])

    def action(node, group_id, path):
        leaf_rows = support[group_id]["leaf_support"]
        leaf = next(row for row in leaf_rows if row["type"] == node["type"] and row["key"] == node["key"]
                    and row["job_evidence"] == node["evidence"])
        if leaf["status"] == "pass":
            return [(frozenset(), ())]
        key = (node["type"], node["key"])
        kind = "learning" if leaf["status"] == "fail" else "evidence_request"
        reference = {"group_id": group_id, "node_path": list(path), "modality": node.get("modality"),
                     "job_evidence": _citation(job, node["evidence"])}
        if key not in registry:
            registry[key] = {"action_id": digest(canonical([VERSION, *key]))[:20], "type": key[0], "key": key[1],
                "label": node["text"], "kind": kind, "current_status": leaf["status"], "conditional": True,
                "group_ids": [], "requirements": [],
                "instruction": "当前材料明确表示尚未掌握；可列为学习与实践目标，完成真实操作并保留证据后再核对岗位要求" if kind == "learning" else
                               "若你已有相关经历，请补充本人操作、任务背景及可核验结果；若没有，请保留未知或另列学习目标，不编写已有经历",
                "completion_is_not_verified_competence": True}
        entry = registry[key]
        if group_id not in entry["group_ids"]:
            entry["group_ids"].append(group_id)
        if reference not in entry["requirements"]:
            entry["requirements"].append(reference)
        return [(frozenset([key]), ())]

    def solve(node, group_id, path=(), include_preferred=False):
        tick()
        modality = node.get("modality", "unknown")
        if modality == "negated":
            return [(frozenset(), ())]
        if modality == "preferred" and not include_preferred:
            marker = (group_id, path)
            if marker not in processed_optional:
                processed_optional.add(marker)
                optional.append((group_id, path, attach_bindings(solve(node, group_id, path, True), group_id)))
            return [(frozenset(), ())]
        if node.get("parse_status") == "unknown" or ("type" in node and modality == "unknown"):
            unresolved_node(group_id, node, path, include_preferred)
            return [(frozenset(), ())]
        if "type" in node:
            return action(node, group_id, path)
        children = node["children"]
        # 可选分支不替代必需分支；保持与当前三态匹配的模态约定一致。
        mandatory = [(index, child) for index, child in enumerate(children) if child.get("modality") not in {"preferred", "negated"}]
        chosen = list(enumerate(children)) if include_preferred else mandatory
        if not include_preferred:
            for index, child in enumerate(children):
                if child.get("modality") == "preferred":
                    solve(child, group_id, path + (index,))
        if not chosen:
            return [(frozenset(), ())]
        if node["op"] == "any":
            candidates = []
            for index, child in chosen:
                for keys, choices in solve(child, group_id, path + (index,), include_preferred):
                    candidates.append((keys, choices + ((group_id, path, index),)))
            return prune(candidates)
        states = [(frozenset(), ())]
        for index, child in chosen:
            candidates = solve(child, group_id, path + (index,), include_preferred)
            merged = []
            for old_keys, old_choices in states:
                for keys, choices in candidates:
                    tick()
                    merged.append((old_keys | keys, old_choices + choices))
            states = prune(merged)
        return states

    def cost(keys):
        return tuple(sum(kind == target for kind, _ in keys) for target in ("skill", "task", "binding"))

    def pareto(candidates):
        result = []
        for candidate in prune(candidates):
            value = cost(candidate[0])
            if any(all(a <= b for a, b in zip(cost(other[0]), value)) and cost(other[0]) != value for other in candidates):
                continue
            result.append(candidate)
        return sorted(result, key=lambda row: (sum(cost(row[0])), cost(row[0]), sorted(row[0])))

    def present(candidates, optional_group=None, optional_path=()):
        candidates = pareto(candidates)
        if len(candidates) > max_plans:
            mark_truncated("plan_display_limit")
        plans = []
        for keys, choices in candidates[:max_plans]:
            actions = [deepcopy(registry[key]) for key in sorted(keys)]
            identity = [VERSION, job.id, profile.get("version"), sorted(keys), choices, optional_group, optional_path]
            plans.append({"plan_id": digest(canonical(identity))[:20], "conditional": True,
                          "status": "conditional_actions" if actions else "no_action_for_current_self_report",
                          "actions": actions, "action_count": len(actions),
                          "cost": {"distinct_skills": cost(keys)[0], "distinct_tasks": cost(keys)[1], "distinct_bindings": cost(keys)[2]},
                          "selected_branches": [{"group_id": group_id, "node_path": list(path), "child_index": index} for group_id, path, index in choices],
                          "hard_constraints_resolved": not any(row["kind"] == "hard_constraint" for row in output["non_actionable_constraints"]),
                          "has_unresolved_binding": any(action["type"] == "binding" for action in actions) or output["alignment_status"] == "not_evaluated",
                          "all_required_groups_accounted_for": not unresolved and output["alignment_status"] != "not_evaluated"})
            if optional_group in optional_unresolved:
                plans[-1]["status"] = "optional_pending_requirement_confirmation"
        return plans

    try:
        combined = [(frozenset(), ())]
        for group in job.groups:
            candidates = solve(group["ast"], group["group_id"])
            if not group["preferred"]:
                candidates = attach_bindings(candidates, group["group_id"])
            merged = []
            for old_keys, old_choices in combined:
                for keys, choices in candidates:
                    tick()
                    merged.append((old_keys | keys, old_choices + choices))
            combined = prune(merged)
        output["plans"] = present(combined)
        output["optional_actions"] = [{"group_id": group_id, "node_path": list(path), "preferred": True,
                                       "plans": present(candidates, group_id, path)} for group_id, path, candidates in optional]
    except _BudgetExceeded:
        output["plans"] = []
        output["optional_actions"] = []
        output["notes"].append("搜索预算已用尽，未把未完成的分支拼成完整方案；请缩小目标要求后重试。")
    output["unresolved_group_ids"] = sorted(unresolved)
    output["optional_unresolved_group_ids"] = sorted(optional_unresolved)
    output["status"] = "needs_requirement_confirmation" if unresolved else "alignment_not_evaluated" if output["alignment_status"] == "not_evaluated" else "bounded" if output["truncated"] else "planned"
    if unresolved:
        for plan in output["plans"]:
            plan["status"] = "partial_pending_requirement_confirmation"
        output["notes"].append("仍有需求待招聘方确认；已列行动只针对可解释部分，不构成全部要求的满足方案。")
    return output
