"""固定随机小布尔树的独立穷举验证；不使用生产解析器或搜索器生成期望值。

48种深度不超过3的树，每棵遍历4个技能的16种已有能力组合。
穷举全部剩余技能子集，分别核对方案可行性与全局最小动作数。
这里只验证已知必需AND/OR逻辑的技能规划，不声称覆盖模糊语义或真实能力。
"""
from copy import deepcopy
from itertools import combinations
import random

import pytest

from job_agent import action_planner
from job_agent.domain import Job


SKILLS = ("Python", "Java", "SQL", "Go")
SEED = 20260923
TREE_COUNT = 48


def truth(tree, available):
    """独立的二值解释器，仅解释测试自己的tuple文法。"""
    if isinstance(tree, str):
        return tree in available
    operator, left, right = tree
    left_value, right_value = truth(left, available), truth(right, available)
    if operator == "AND":
        return left_value and right_value
    if operator == "OR":
        return left_value or right_value
    raise AssertionError("测试树包含未知运算符")


def subsets(values):
    values = tuple(values)
    for count in range(len(values) + 1):
        yield from (frozenset(items) for items in combinations(values, count))


def minimum_additions(tree, already_supported):
    """直接枚举2^4以内的剩余技能集，不调用生产规划逻辑。"""
    remaining = [skill for skill in SKILLS if skill not in already_supported]
    feasible = [candidate for candidate in subsets(remaining)
                if truth(tree, already_supported | candidate)]
    assert feasible, "有限正布尔式在四技能全部具备时应可满足"
    return min(len(candidate) for candidate in feasible)


def depth(tree):
    return 0 if isinstance(tree, str) else 1 + max(depth(tree[1]), depth(tree[2]))


def occurrences(tree):
    return [tree] if isinstance(tree, str) else occurrences(tree[1]) + occurrences(tree[2])


def generated_trees():
    rng = random.Random(SEED)
    # 这些固定种子反例确保重复技能跨AND/OR的集合剪枝受到检查。
    trees = [
        ("AND", ("OR", ("AND", "Python", "SQL"), "Java"), ("AND", "Python", "SQL")),
        ("AND", ("OR", "Python", "Java"), ("OR", "Python", "Go")),
        ("OR", ("AND", "Python", "Python"), ("AND", "SQL", "SQL")),
        ("OR", ("AND", "Python", "SQL"), ("AND", "Python", "Java")),
        ("AND", ("AND", "Python", "Java"), ("AND", "SQL", "Go")),
    ]

    def sample(remaining_depth):
        if remaining_depth == 0 or rng.random() < .28:
            return rng.choice(SKILLS)
        return (rng.choice(("AND", "OR")), sample(remaining_depth - 1), sample(remaining_depth - 1))

    while len(trees) < TREE_COUNT:
        candidate = sample(3)
        if candidate not in trees:
            trees.append(candidate)
    assert len(set(trees)) == TREE_COUNT and all(depth(tree) <= 3 for tree in trees)
    return tuple(trees)


TREES = generated_trees()
SUPPORTED_SUBSETS = tuple(subsets(SKILLS))


def materialize_job(tree, identifier):
    """直接构造带真实跨度的Job要求树；Job默认抽取结果随后全部替换。"""
    fragments = []
    position = 0

    def append(text):
        nonlocal position
        fragments.append(text)
        position += len(text)

    def build(node):
        if isinstance(node, str):
            start = position
            append(node)
            return {"type": "skill", "key": node, "text": node, "required": True,
                    "modality": "required", "parse_status": "known",
                    "evidence": {"field": "description", "start": start, "end": position, "quote": node}}
        append("（")
        left = build(node[1])
        append("且" if node[0] == "AND" else "或")
        right = build(node[2])
        append("）")
        return {"op": "all" if node[0] == "AND" else "any", "children": [left, right],
                "modality": "required", "parse_status": "known"}

    append("必须掌握")
    ast = build(tree)
    append("。")
    description = "".join(fragments)
    target = Job(identifier, "v", "虚构验证岗位", "虚构企业", "技术", "技术", "10-20K",
                 "经验不限本科", description, "深圳南山区", 1)
    group = {"group_id": identifier + "-group", "label": description,
             "kind": "single" if isinstance(tree, str) else "all" if tree[0] == "AND" else "any",
             "skills": sorted(set(occurrences(tree))), "tasks": [], "preferred": False,
             "modality": "required", "parse_status": "known", "ast": ast,
             "evidence": {"field": "description", "start": 0, "end": len(description), "quote": description}}
    target.groups = [group]
    target.requirement_ast = {"op": "all", "children": [ast], "modality": "required", "parse_status": "known"}
    return target


def materialize_profile(supported):
    """直接给定哪些叶被支持，避免用生产文本解析器决定期望真值。"""
    text = "\n".join("本人使用" + skill + "完成练习。" for skill in sorted(supported))
    skills = {}
    for skill in supported:
        start = text.index(skill)
        skills[skill] = {"field": "resume", "start": start, "end": start + len(skill), "quote": skill,
                         "weight": 1.0, "level": "实践", "parse_status": "known", "polarity": "positive",
                         "actor": "self", "claim_type": "self_reported_action"}
    return {"text": text, "version": "fixture:" + ",".join(sorted(supported)), "skills": skills, "tasks": {},
            "city": "深圳", "intent": "技术", "education": 3, "experience_years": 0,
            "experience_details": {"specialized": []}}


@pytest.mark.parametrize("index,tree", list(enumerate(TREES)), ids=[f"seed{SEED}-tree{i:02d}" for i in range(TREE_COUNT)])
def test_every_returned_plan_is_feasible_and_globally_minimal(index, tree):
    target = materialize_job(tree, f"tree-{index}")
    original_groups = deepcopy(target.groups)
    for supported in SUPPORTED_SUBSETS:
        profile = materialize_profile(supported)
        original_profile = deepcopy(profile)
        optimum = minimum_additions(tree, supported)
        result = action_planner.plan_actions(profile, target)
        context = {"seed": SEED, "tree": tree, "supported": sorted(supported), "independent_optimum": optimum}
        assert result["plans"] and len(result["plans"]) <= 3, context
        assert set(result["truncation_reasons"]) <= {"plan_display_limit"}, context
        for plan in result["plans"]:
            additions = frozenset(action["key"] for action in plan["actions"])
            assert additions <= set(SKILLS) - supported, context
            assert len(additions) == len(plan["actions"]) == plan["action_count"], context
            assert truth(tree, supported | additions), {**context, "returned": sorted(additions)}
            assert len(additions) == optimum, {**context, "returned": sorted(additions)}
            assert plan["conditional"] and plan["all_required_groups_accounted_for"], context
            for action in plan["actions"]:
                assert action["type"] == "skill" and action["kind"] == "evidence_request", context
                for reference in action["requirements"]:
                    citation = reference["job_evidence"]
                    assert target.description[citation["start"]:citation["end"]] == citation["quote"], context
                    assert reference["group_id"] == target.groups[0]["group_id"], context
        assert profile == original_profile and target.groups == original_groups, context


@pytest.mark.parametrize("limit,value,reason", [
    ("MAX_EXPANSIONS", 1, "expansion_budget"),
    ("MAX_NODES", 2, "graph_budget"),
    ("MAX_DEPTH", 1, "graph_budget"),
    ("MAX_ACTIONS", 1, "action_limit"),
])
def test_budget_exhaustion_never_labels_incomplete_branch_as_full_plan(monkeypatch, limit, value, reason):
    tree = ("AND", ("AND", "Python", "Java"), ("AND", "SQL", "Go"))
    assert minimum_additions(tree, frozenset()) == 4
    monkeypatch.setattr(action_planner, limit, value)
    result = action_planner.plan_actions(materialize_profile(frozenset()), materialize_job(tree, "limited"))
    assert result["truncated"] and reason in result["truncation_reasons"]
    assert result["plans"] == []
    assert result["status"] != "planned"


def test_state_truncation_preserves_feasibility_but_does_not_claim_exact_search(monkeypatch):
    tree = TREES[0]
    supported = frozenset()
    assert minimum_additions(tree, supported) == 2
    monkeypatch.setattr(action_planner, "MAX_STATES", 1)
    result = action_planner.plan_actions(materialize_profile(supported), materialize_job(tree, "state-limited"))
    assert result["truncated"] and "state_budget" in result["truncation_reasons"]
    assert result["status"] == "bounded"
    for plan in result["plans"]:
        additions = frozenset(action["key"] for action in plan["actions"])
        assert truth(tree, additions)
        assert len(additions) >= minimum_additions(tree, supported)
