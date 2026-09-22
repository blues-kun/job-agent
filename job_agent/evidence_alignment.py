"""要求组与经历块的保守文本对齐；文本支持不代表真实能力已经核验。

只识别同一动作作用域内显式的“使用工具完成任务”关系。普通 AND 要求仍可由
不同经历支持。输出 quote 保持原文坐标；界面应渲染 display_quote，避免显示联系方式。
"""
from __future__ import annotations

from itertools import product
import re

from . import semantics
from .domain import PATTERNS, TASK_PATTERNS, TASK_NAMES, display_text, group_support


VERSION = "requirement-block-alignment-v1"
LIMITS = {"resume_chars": 20000, "job_chars": 60000, "groups": 96,
          "blocks": 64, "block_chars": 6000, "ast_nodes": 2048,
          "ast_depth": 24, "tool_alternatives": 32, "candidate_blocks": 3}
_HEADING = re.compile(r"^(?:项目(?:一|二|三|四|五|六|[0-9]+|名称)|工作经历|项目经历|教育经历|教育背景|任职经历|实习经历|20\d{2}[./年-])")
_BREAK = re.compile(r"但是|不过|然而|但|[。；;\n，,]")
_USE = re.compile(r"使用|运用|采用|利用|借助|用")
_BRIDGE = re.compile(r"^[\s)）]*(?:(?:来|以|用于|用来)\s*)?(?:完成|进行|开展|执行|实现|负责)\s*(?:日常|业务|自动化|批量)?\s*$")
_ACTION_TASK = re.compile(r"^(?:清洗|标注|分析|梳理|维护|处理|筛选|协调|跟踪|编写|撰写|执行|设计|部署|排查|定位)")
_CONTEXT = re.compile(r"希望|计划|准备|打算|将要|拟学习|待学习|招聘|任职要求|岗位要求|阅读|教程|示例|文档中")


def _citation(text, field, start, end):
    quote = text[start:end]
    return {"field": field, "start": start, "end": end, "quote": quote,
            "display_quote": display_text(quote)}


def _checked_citation(evidence, text, field):
    if not isinstance(evidence, dict) or evidence.get("field") != field:
        return None
    start, end = evidence.get("start"), evidence.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        return None
    if text[start:end] != evidence.get("quote") or end-start > LIMITS["block_chars"]:
        return None
    return _citation(text, field, start, end)


def _leaves(ast, depth=0, budget=None):
    budget = [LIMITS["ast_nodes"]] if budget is None else budget
    budget[0] -= 1
    if depth > LIMITS["ast_depth"] or budget[0] < 0 or not isinstance(ast, dict):
        raise ValueError("要求树超出限制或结构不合法")
    if "type" in ast:
        return [ast]
    if ast.get("op") not in {"all", "any"} or not isinstance(ast.get("children"), list):
        raise ValueError("要求树结构不合法")
    return [leaf for child in ast["children"] for leaf in _leaves(child, depth+1, budget)]


def _blocks(text):
    """保留原始换行与空格；标题和紧邻日期属于同一完整经历块。"""
    result, start, end, previous, offset = [], None, None, "", 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n"); stripped = content.strip()
        follows_title = bool(start is not None and re.match(r"^20\d{2}[./年-]", stripped)
                             and _HEADING.search(previous) and not re.match(r"^20\d{2}", previous))
        if not stripped or (start is not None and _HEADING.search(stripped) and not follows_title):
            if start is not None:
                result.append(_citation(text, "resume", start, end))
            start = end = None
        if stripped:
            if start is None:
                start = offset
            end = offset + len(content)
            previous = stripped
        offset += len(line)
    if start is not None:
        result.append(_citation(text, "resume", start, end))
    for block in result:
        block["block_id"] = f"resume:{block['start']}:{block['end']}"
    return result


def _tool_tree(ast):
    if "type" in ast:
        return {"skill": ast["key"]}
    return {"op": ast["op"], "unknown": ast.get("parse_status") != "known",
            "children": [_tool_tree(child) for child in ast["children"]]}


def _options(tree):
    """列出工具表达式的有限可能分支，避免把‘Python或Java’当作两者都会。"""
    if "skill" in tree:
        return [frozenset([tree["skill"]])]
    if tree.get("unknown"):
        return None
    children = [_options(child) for child in tree["children"]]
    if not children or any(child is None for child in children):
        return None
    if tree["op"] == "any":
        values = [option for child in children for option in child]
    else:
        count = 1
        for child in children:
            count *= len(child)
            if count > LIMITS["tool_alternatives"]:
                return None
        values = [frozenset().union(*items) for items in product(*children)]
    return list(dict.fromkeys(values)) if len(values) <= LIMITS["tool_alternatives"] else None


def _practiced(item):
    return (item.get("actor") == "self" and item.get("polarity") == "positive"
            and item.get("parse_status") == "known" and item.get("claim_type") == "self_reported_action"
            and item.get("weight", 0) >= .7)


def _relations(text, field):
    skills = semantics.mentions(text, PATTERNS, field, "skill")
    tasks = semantics.mentions(text, TASK_PATTERNS, field, "task", TASK_NAMES)
    bounds, offset = [], 0
    for separator in _BREAK.finditer(text):
        bounds.append((offset, separator.start())); offset = separator.end()
    bounds.append((offset, len(text)))
    relations = []
    for start, end in bounds:
        for task in (item for item in tasks if start <= item["start"] < end):
            uses = list(_USE.finditer(text, start, task["start"]))
            if not uses:
                continue
            use = uses[-1]
            tools = [item for item in skills if use.end() <= item["start"] and item["end"] <= task["start"]]
            if not tools or not re.fullmatch(r"[\s(（]*", text[use.end():tools[0]["start"]]):
                continue
            bridge = text[tools[-1]["end"]:task["start"]]
            if not (_BRIDGE.fullmatch(bridge) or (not bridge.strip(" )）") and _ACTION_TASK.search(task["quote"]))):
                continue
            local = [{**item, "start": item["start"]-use.end(),
                      "end": item["end"]-use.end()} for item in tools]
            expression = semantics.parse_expression(text[use.end():task["start"]], local)
            tree = _tool_tree(expression); options = _options(tree)
            practiced = ({item["key"] for item in tools if _practiced(item)} if field == "resume" else set())
            context_only = bool(_CONTEXT.search(text[start:use.start()])) if field == "resume" else False
            verified = (field == "resume" and _practiced(task) and not context_only and options is not None
                        and all(option <= practiced for option in options))
            relations.append({"task_key": task["key"], "task_name": task["text"], "tools": tree,
                "tool_keys": sorted({item["key"] for item in tools}), "options": options,
                "job_modality": task["modality"], "verified_text_support": verified,
                "evidence": _citation(text, field, use.start(), task["end"]),
                "scope": _citation(text, field, start, end),
                "mentions": [*tools, task]})
    return relations, skills, tasks


def _applies(relation, leaves):
    if relation["job_modality"] == "negated":
        return False
    return any(leaf["type"] == item["type"] and leaf["key"] == item["key"]
               and leaf["evidence"]["field"] == item["field"]
               and leaf["evidence"]["start"] == item["start"]
               for leaf in leaves for item in relation["mentions"])


def _proves(required, observed):
    if (not observed["verified_text_support"] or required["task_key"] != observed["task_key"]
            or required["options"] is None or observed["options"] is None):
        return False
    # 简历每种可能工具组合都须满足岗位一个完整分支。
    return all(any(need <= actual for need in required["options"]) for actual in observed["options"])


def _public_relation(relation):
    return {"task_key": relation["task_key"], "task": relation["task_name"], "tools": relation["tools"],
            "job_evidence": relation["evidence"], "job_scope": relation["scope"]}


def align_requirements(profile: dict, job) -> dict:
    """输出版本、要求组与摘要；超限或不合法树返回未评估，不虚构支持结论。"""
    text = profile.get("text", "")
    output = {"version": VERSION, "profile_version": profile.get("version"), "job_id": job.id,
        "groups": [], "summary": {"status": "completed", "groups_total": len(job.groups),
        "groups_evaluated": 0, "global_supported_groups": 0, "supported_groups": 0,
        "binding_required_groups": 0, "binding_supported_groups": 0, "downgraded_groups": 0,
        "claim_scope": "textual_support_only", "ability_verified": False}, "limits": dict(LIMITS),
        "note": "对齐只说明简历自述与要求的文本关系；经历和能力未经外部核验。界面展示display_quote，quote用于原文定位。"}
    summary = output["summary"]
    if not isinstance(text, str):
        summary.update(status="not_evaluated", reason="invalid_resume_text"); return output
    if (len(text) > LIMITS["resume_chars"] or len(job.requirements)+len(job.description) > LIMITS["job_chars"]
            or len(job.groups) > LIMITS["groups"]):
        summary.update(status="not_evaluated", reason="input_length_limit"); return output
    blocks = _blocks(text)
    if len(blocks) > LIMITS["blocks"] or any(block["end"]-block["start"] > LIMITS["block_chars"] for block in blocks):
        summary.update(status="not_evaluated", reason="experience_block_limit"); return output
    try:
        leaves = [_leaves(group["ast"]) for group in job.groups]
        baseline = group_support(profile, job)
    except (ValueError, KeyError, TypeError, RecursionError):
        summary.update(status="not_evaluated", reason="invalid_requirement_tree"); return output
    observed, resume_skills, resume_tasks = _relations(text, "resume")
    requirements = [item for field in ("requirements", "description")
                    for item in _relations(getattr(job, field), field)[0]]
    block_support = []
    for block in blocks:
        inside = lambda item: block["start"] <= item["start"] and item["end"] <= block["end"]
        partial = {**profile, "skills": semantics.compact_mentions([item for item in resume_skills if inside(item)]),
                   "tasks": semantics.compact_mentions([item for item in resume_tasks if inside(item)])}
        block_support.append(group_support(partial, job))
    for index, (group, old, group_leaves) in enumerate(zip(job.groups, baseline, leaves)):
        evidence = group.get("evidence", {}); field = evidence.get("field", "")
        citation = _checked_citation(evidence, getattr(job, field, ""), field) if field in {"requirements", "description"} else None
        applicable = [relation for relation in requirements if _applies(relation, group_leaves)]
        candidates, proof_blocks = [], [set() for _ in applicable]
        for block_index, (block, supports) in enumerate(zip(blocks, block_support)):
            support = supports[index]
            proof = []
            for relation_index, relation in enumerate(applicable):
                for actual in observed:
                    if (block["start"] <= actual["scope"]["start"] and actual["scope"]["end"] <= block["end"]
                            and _proves(relation, actual)):
                        proof_blocks[relation_index].add(block_index)
                        proof.append({"requirement_index": relation_index, "resume_evidence": actual["evidence"],
                                      "resume_scope": actual["scope"]})
            mentioned = [leaf for leaf in support["leaf_support"] if leaf.get("resume_evidence")]
            if not mentioned and not proof:
                continue
            passed = sum(leaf["status"] == "pass" for leaf in support["leaf_support"])
            complete_binding = not applicable or len({item["requirement_index"] for item in proof}) == len(applicable)
            block_status = "unknown" if support["status"] == "pass" and not complete_binding else support["status"]
            candidates.append({**block, "status": block_status, "global_status": support["status"], "supported_leaves": passed,
                "mentioned_leaves": [{"type": leaf["type"], "key": leaf["key"], "status": leaf["status"]} for leaf in mentioned],
                "task_bindings": proof, "_order": (len({item["requirement_index"] for item in proof}),
                                                    support["status"] == "pass", passed, -block["start"])})
        candidates.sort(key=lambda row: row["_order"], reverse=True)
        candidates = [{key: value for key, value in row.items() if key != "_order"}
                      for row in candidates[:LIMITS["candidate_blocks"]]]
        binding_status = "not_required" if not applicable else "pass" if all(proof_blocks) else "unknown"
        status = old["status"]
        if citation is None or group.get("parse_status") != "known":
            status = "unknown"
        elif status == "pass" and binding_status == "unknown":
            status = "unknown"
        reasons = []
        if citation is None:
            reasons.append("原JD要求跨度无效或超出展示限制")
        if applicable and binding_status != "pass":
            reasons.append("技能和任务分别出现，尚不能证明在同一动作作用域中完成该任务")
        if group.get("parse_status") != "known":
            reasons.append("要求逻辑尚不明确，保留未知")
        if not applicable:
            reasons.append("未识别到显式工具—任务绑定；普通AND允许不同经历共同支持")
        output["groups"].append({"group_id": group["group_id"], "label": group["label"], "kind": group["kind"],
            "status": status, "global_status": old["status"], "job_evidence": citation,
            "best_block": candidates[0] if candidates else None, "candidate_blocks": candidates,
            "task_binding": {"required": bool(applicable), "status": binding_status,
                             "requirements": [_public_relation(item) for item in applicable],
                             "supported_requirements": sum(bool(item) for item in proof_blocks)},
            "reasons": reasons, "ability_verified": False})
        summary["groups_evaluated"] += 1
        summary["global_supported_groups"] += old["status"] == "pass"
        summary["supported_groups"] += status == "pass"
        summary["binding_required_groups"] += bool(applicable)
        summary["binding_supported_groups"] += bool(applicable) and binding_status == "pass"
        summary["downgraded_groups"] += old["status"] == "pass" and status != "pass"
    summary["no_parsed_requirements"] = not bool(job.groups)
    return output
