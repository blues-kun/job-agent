"""归纳要求组图：保留 AND/OR 作用域，不把不确定关系或缺失证据写成事实。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

GRAPH_VERSION = "requirement-group-inductive-v2"
NODE_TYPES = ("job", "all", "any", "unknown", "skill", "task")
MODALITIES = ("required", "preferred", "unknown", "mixed", "negated")


def canonical_text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def private_output(path):
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("输出必须位于 Git 工作区外")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("输出目录非空，拒绝覆盖")
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical_ast(ast, depth=0):
    if depth > 6 or not isinstance(ast, dict):
        raise ValueError("要求 AST 必须为字典且嵌套深度不超过6")
    modality = ast.get("modality", "unknown")
    if modality not in MODALITIES:
        raise ValueError("未知要求模态")
    evidence = ast.get("evidence")
    parse_status = ast.get("parse_status", "known")
    if parse_status not in ("known", "unknown"):
        raise ValueError("未知解析状态")
    if "op" in ast:
        if ast["op"] not in ("all", "any") or not isinstance(ast.get("children"), list):
            raise ValueError("要求组必须有 all/any 与 children列表")
        # 词典未覆盖或没有可抽取要求时保留未知节点，不能因空AND而判满足。
        if not ast["children"]:
            parse_status = "unknown"
        children = [canonical_ast(child, depth + 1) for child in ast["children"]]
        # AND/OR 子项本身无先后语义，但组作用域绝不展开或打平。
        children.sort(key=object_sha)
        return {"op": ast["op"], "children": children, "modality": modality,
                "parse_status": parse_status, "evidence": evidence}
    if ast.get("type") not in ("skill", "task") or not canonical_text(ast.get("key")):
        raise ValueError("叶子必须具有 skill/task 类型及规范 key")
    if modality == "negated":
        raise ValueError("否定要求不应进入行动 AST；请保留在 requirement_mentions")
    return {"type": ast["type"], "key": canonical_text(ast["key"]),
            "text": canonical_text(ast.get("text") or ast["key"]), "modality": modality,
            "required": bool(ast.get("required", modality == "required")), "parse_status": parse_status,
            "evidence": evidence}


def declared_support(profile, kind, key):
    values = profile.get("skills" if kind == "skill" else "tasks", {})
    if isinstance(values, list):
        # 列表接口表示显式自述实践；不视作经第三方验证的能力事实。
        return (1.0, 1.0) if key in values else (0.0, 1.0)
    item = values.get(key) if isinstance(values, dict) else None
    if not isinstance(item, dict):
        return (0.0, 1.0)
    if item.get("parse_status") == "unknown" or item.get("conflict") is True or item.get("actor") in ("background", "unknown", "other"):
        return (0.0, 1.0)
    if item.get("level") == "否定" or item.get("status") == "fail":
        return (0.0, 0.0)
    if item.get("level") == "实践" or item.get("status") == "pass" or item.get("practice_supported") is True:
        return (1.0, 1.0)
    return (0.0, 1.0)


def logic_baseline(ast, profile):
    ast = canonical_ast(ast)
    def visit(node):
        if node["modality"] == "preferred":
            return None
        if node["modality"] == "unknown" or node.get("parse_status") == "unknown":
            return 0.0, 1.0
        if "op" not in node:
            if not node["required"]:
                return 0.0, 1.0
            return declared_support(profile, node["type"], node["key"])
        values = [value for child in node["children"] if (value := visit(child)) is not None]
        if not values:
            return 1.0, 1.0
        aggregate = min if node["op"] == "all" else max
        return aggregate(value[0] for value in values), aggregate(value[1] for value in values)
    lower, upper = visit(ast) or (1.0, 1.0)
    return {"supported_lower": lower, "possible_upper": upper,
            "status": "pass" if lower == 1 else "fail" if upper == 0 else "unknown",
            "note": "pass表示简历存在自述实践证据；unknown不是不相关或能力不足"}


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    text: str
    modality: str


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    relation: int
    status: str


@dataclass(frozen=True)
class RequirementGraph:
    job_id: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    ast_hash: str
    successful_swaps: int = 0


def build_graph(job):
    ast = canonical_ast(job["requirement_ast"])
    nodes = [Node("job", "job", canonical_text(job.get("title")), "required")]
    edges = []
    def append(node, path, parent, uncertain=False):
        uncertain = uncertain or node["modality"] == "unknown" or node.get("parse_status") == "unknown"
        kind = ("unknown" if node.get("parse_status") == "unknown" else node["op"]) if "op" in node else node["type"]
        text = node.get("text", {"all": "全部要求", "any": "任选要求", "unknown": "连接关系待确认"}.get(kind, ""))
        index = len(nodes)
        nodes.append(Node(path, kind, text, node["modality"]))
        destination = nodes[parent].kind
        relation = {"all": 0, "any": 1, "unknown": 2, "job": 3}[destination] + (4 if uncertain else 0)
        edges.append(Edge(index, parent, relation, "unknown" if uncertain else "machine_parsed"))
        for child_index, child in enumerate(node.get("children", [])):
            append(child, f"{path}.{child_index}", index, uncertain)
    append(ast, "root", 0)
    return RequirementGraph(str(job["job_id"]), tuple(nodes), tuple(edges), object_sha(ast))


def degree_signature(graph):
    return Counter(("out", e.source, e.relation, e.status) for e in graph.edges) + Counter(("in", e.target, e.relation, e.status) for e in graph.edges)


def degree_preserving_randomization(graph, seed, swaps_per_edge=5):
    """仅交换同类型叶子→同类型组的边端点；保留逐节点入/出度、模态、关系和不确定性。"""
    edges = list(graph.edges)
    groups = defaultdict(list)
    for index, edge in enumerate(edges):
        node = graph.nodes[edge.source]
        if node.kind in ("skill", "task"):
            groups[(node.kind, node.modality, graph.nodes[edge.target].kind, edge.relation, edge.status)].append(index)
    rng = random.Random(f"{seed}:{graph.job_id}:{graph.ast_hash}")
    successes = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        for _ in range(swaps_per_edge * len(members)):
            a, b = rng.sample(members, 2)
            first, second = edges[a], edges[b]
            if first.source == second.source or first.target == second.target:
                continue
            existing = {(edge.source, edge.target, edge.relation, edge.status) for i, edge in enumerate(edges) if i not in (a, b)}
            x, y = replace(first, target=second.target), replace(second, target=first.target)
            if (x.source, x.target, x.relation, x.status) in existing or (y.source, y.target, y.relation, y.status) in existing:
                continue
            edges[a], edges[b] = x, y
            successes += 1
    result = replace(graph, edges=tuple(edges), successful_swaps=successes)
    if degree_signature(result) != degree_signature(graph):
        raise RuntimeError("随机图未保持节点关系入/出度")
    return result


class FrozenTextFeatures:
    """固定文本初始化；不从 train/dev/test 拟合词表或更新共享特征。"""
    def __init__(self, dimension=128, external=None):
        self.external = external
        self.dimension = int(external["dimension"] if external else dimension)
        if self.dimension < 8:
            raise ValueError("文本初始化维度至少8")
        self.contract = {"method": "external_frozen_vectors" if external else "deterministic_text_hash",
                         "dimension": self.dimension, "is_pretrained_semantic": bool(external),
                         "external_contract": external.get("contract") if external else None,
                         "fit_on_evaluation": False}
        if external and not external.get("contract"):
            raise ValueError("外部文本向量必须声明模型/版本/模板合同")

    def __call__(self, text):
        text = canonical_text(text)
        if self.external:
            key = hashlib.sha256(text.encode()).hexdigest()
            if key not in self.external["vectors"]:
                raise ValueError("外部冻结初始化缺少当前节点文本，拒绝混用hash替代")
            vector = torch.tensor(self.external["vectors"][key], dtype=torch.float32)
        else:
            vector = torch.zeros(self.dimension)
            parts = re.findall(r"[a-zA-Z0-9_+#]+|[\u4e00-\u9fff]", text.lower())
            units = parts + ["|".join(parts[i:i + 2]) for i in range(max(0, len(parts) - 1))]
            for unit in units or ["<empty>"]:
                digest = hashlib.sha256(unit.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1 if digest[4] % 2 else -1
        if vector.shape != (self.dimension,) or not torch.isfinite(vector).all() or not bool(vector.norm() > 0):
            raise ValueError("文本初始化向量的形状、有限性或范数非法")
        return F.normalize(vector, dim=0)


def profile_text(profile):
    fields = [canonical_text(profile.get("text"))]
    for field in ("skills", "tasks"):
        values = profile.get(field, {})
        names = values.keys() if isinstance(values, dict) else values
        fields.append(f"{field}：" + "、".join(sorted(str(key) for key in names)))
    return "\n".join(fields)


class GroupGraphRanker(nn.Module):
    def __init__(self, features, hidden=32, mode="graph"):
        super().__init__()
        if mode not in ("graph", "pool"):
            raise ValueError("模型模式必须为graph/pool")
        self.features, self.mode = features, mode
        self.input = nn.Linear(features.dimension, hidden)
        self.types = nn.Embedding(len(NODE_TYPES), hidden)
        self.modalities = nn.Embedding(len(MODALITIES), hidden)
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(8))
        self.update = nn.Linear(3 * hidden, hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.readout = nn.Linear(hidden, hidden)
        self.query = nn.Sequential(nn.Linear(features.dimension, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.log_scale = nn.Parameter(torch.tensor(1.0))

    def encode_job(self, graph):
        device = self.input.weight.device
        x = torch.stack([self.features(node.text) for node in graph.nodes]).to(device)
        types = torch.tensor([NODE_TYPES.index(node.kind) for node in graph.nodes], device=device)
        modalities = torch.tensor([MODALITIES.index(node.modality) for node in graph.nodes], device=device)
        h = torch.tanh(self.input(x) + self.types(types) + self.modalities(modalities))
        if self.mode == "pool":
            # 保留与图模型相同的岗位标题和叶子文本/模态，仅移除组及连接关系。
            leaves = [i for i, node in enumerate(graph.nodes) if node.kind in ("skill", "task")]
            # 保证相同多重叶集合以同顺序求和，避免浮点微差伪造组结构收益。
            leaves.sort(key=lambda index: (graph.nodes[index].kind, graph.nodes[index].text, graph.nodes[index].modality))
            return F.normalize(self.readout(h[[0, *leaves]].mean(0)), dim=0)
        incoming = defaultdict(list)
        for edge in graph.edges:
            incoming[edge.target].append(edge)
        states, visiting = {}, set()
        def propagate(index):
            if index in states:
                return states[index]
            if index in visiting:
                raise ValueError("要求图含环，不能按要求依赖归纳编码")
            visiting.add(index)
            if not incoming[index]:
                state = h[index]
            else:
                messages = torch.stack([self.messages[edge.relation](propagate(edge.source)) for edge in incoming[index]])
                aggregate = messages.mean(0)
                # 非线性组内聚合使同一叶集合的不同OR分组能产生不同表示。
                # 这是可学习排序表示，不替代上面的三值逻辑事实校验。
                update_input = torch.cat((h[index], aggregate, aggregate.square()))
                state = torch.tanh(self.node_norm(self.update(update_input)) + h[index])
            visiting.remove(index)
            states[index] = state
            return state
        return F.normalize(self.readout(propagate(0)), dim=0)

    def encode_profile(self, profile):
        vector = self.features(profile_text(profile)).to(self.input.weight.device)
        return F.normalize(self.query(vector), dim=0)

    def score(self, profile_vector, job_vectors):
        return (job_vectors @ profile_vector) * self.log_scale.exp().clamp(max=30)


def validate_supervision(jobs, profiles, labels, allow_fixture=False, require_train_dev=True, allow_model_labels=False):
    def index(rows, key):
        if any(not row.get(key) for row in rows) or len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"{key}缺失或重复")
        return {row[key]: row for row in rows}
    job_index, query_index = index(jobs, "job_id"), index(profiles, "query_id")
    family_splits = defaultdict(set)
    profile_text_splits = defaultdict(set)
    snapshots = set()
    for kind, rows, family_key in (("job", jobs, "job_family_id"), ("profile", profiles, "profile_family_id")):
        for row in rows:
            if row.get("split") not in ("train", "dev", "test") or not row.get(family_key):
                raise ValueError("画像和岗位必须有固定split及家族")
            family_splits[(kind, row[family_key])].add(row["split"])
            if kind == "job":
                if not row.get("snapshot"):
                    raise ValueError("岗位缺少snapshot来源版本")
                snapshots.add(row["snapshot"])
            elif canonical_text(row.get("text")):
                profile_text_splits[canonical_text(row["text"])].add(row["split"])
    if len(snapshots) != 1 or any(len(parts) > 1 for parts in family_splits.values()) or any(len(parts) > 1 for parts in profile_text_splits.values()):
        raise ValueError("快照不一致或岗位/画像家族/画像原文跨分区泄漏")
    if not labels:
        raise ValueError("没有已审核人岗标签；不生成伪human标签")
    seen, sources, groups = set(), Counter(), defaultdict(list)
    for row in labels:
        query, job = query_index.get(row.get("query_id")), job_index.get(row.get("job_id"))
        if query is None or job is None or row.get("split") != query["split"] or row["split"] != job["split"]:
            raise ValueError("qrel的画像/岗位不存在或split不匹配")
        if type(row.get("grade")) is not int or row["grade"] not in range(4):
            raise ValueError("grade必须为已审核的0..3；未知不能当作0")
        source = row.get("label_source")
        if source == "human_adjudicated":
            if not isinstance(row.get("reviewer_ids"), list) or not row["reviewer_ids"] or not row.get("adjudication_id"):
                raise ValueError("人工标签缺少评审/仲裁血缘")
            from .review_contracts import validate_qrel
            validate_qrel(row)
        elif source == "llm_reviewed":
            if not allow_model_labels:
                raise ValueError("模型标签须显式allow-model-labels；不能冒充human")
            if not canonical_text(row.get("model")) or not canonical_text(row.get("review_run_id")):
                raise ValueError("模型标签缺少model或review_run_id")
            if any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) for key in ("prompt_hash", "input_hash")):
                raise ValueError("模型标签缺少有效prompt_hash/input_hash")
            from .review_contracts import validate_qrel
            validate_qrel(row, allow_model_labels=True)
        elif source != "synthetic_fixture" or not allow_fixture:
            raise ValueError("拒绝未审标签；fixture必须显式开启")
        key = (row["query_id"], row["job_id"])
        if key in seen:
            raise ValueError("qrel重复")
        seen.add(key)
        sources[source] += 1
        groups[(row["split"], row["query_id"])].append(row)
    if len(sources) > 1:
        raise ValueError("fixture、模型弱监督与人工标签不得混为同一次监督实验")
    if require_train_dev and not {"train", "dev"}.issubset({split for split, _ in groups}):
        raise ValueError("监督实验需要独立train/dev；不能把已有test搬进train")
    for (split, _), group in groups.items():
        if len(group) < 2 or (split == "train" and len({row["grade"] for row in group}) < 2):
            raise ValueError("每个查询至少两个已审核候选；训练组需要至少两档标签")
    return job_index, query_index, groups, {"snapshot": next(iter(snapshots)), "label_sources": dict(sources),
              "fixture_only": "synthetic_fixture" in sources, "family_overlap": 0, "profile_text_overlap": 0,
              "model_weak_supervision": "llm_reviewed" in sources,
              "eligible_for_production": False,
              "teacher_models": sorted({row["model"] for row in labels if row["label_source"] == "llm_reviewed"}),
              "teacher_provenance_hash": object_sha(sorted((row["query_id"], row["job_id"], row["model"], row["prompt_hash"], row["input_hash"], row["review_run_id"]) for row in labels if row["label_source"] == "llm_reviewed")),
              "candidate_contract_sha256": object_sha(sorted((split, query_id, sorted(row["job_id"] for row in group)) for (split, query_id), group in groups.items())),
              "note": "评估仅使用每个查询同源已审标签的固定候选池；模型教师标签不是人工金标，池外仍未知，不计算全库Recall"}
