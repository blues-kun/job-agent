"""为固定教师研究候选的要求图准备冻结BGE节点属性；不读取或生成相关性标签。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import uuid

import numpy as np

from research.group_graph import (GRAPH_VERSION, build_graph, canonical_text, file_sha,
                                 object_sha, profile_text, read_jsonl)

SEMANTIC_JOB_FIELDS = frozenset({"groups", "skills", "tasks", "requirement_ast", "parser_version",
    "experience_requirements", "education_requirements", "experience_min", "education_min"})
SEMANTIC_PROFILE_FIELDS = frozenset({"skills", "tasks", "parser_version"})


def semantic_changes(before, after, allowed):
    """缺字段与null不同；只允许显式解析字段变化。"""
    changed = sorted(key for key in before.keys() | after.keys()
                     if key not in before or key not in after or before[key] != after[key])
    forbidden = sorted(set(changed) - allowed)
    if forbidden:
        raise ValueError("来源桥接修改了非解析字段：" + ",".join(forbidden))
    return changed


def selected_rows(path, selected_ids):
    selected = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("job_id") in selected_ids:
                if row["job_id"] in selected:
                    raise ValueError("岗位ID重复")
                selected[row["job_id"]] = row
    if set(selected) != selected_ids:
        raise ValueError("研究任务引用不存在的岗位")
    return selected


def jsonl_chunks(rows):
    return (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def jsonl_sha(rows):
    digest = hashlib.sha256()
    for chunk in jsonl_chunks(rows):
        digest.update(chunk.encode())
    return digest.hexdigest()


def refresh_semantics(selected, profiles, semantic_path):
    from job_agent import domain, semantics
    from job_agent.domain import Job, PARSER_VERSION, parse_profile
    refreshed = selected_rows(semantic_path, set(selected))
    job_changes, profile_changes, provenance = Counter(), Counter(), {"jobs": [], "profiles": []}
    for key, current in refreshed.items():
        previous = selected[key]
        changes = semantic_changes(previous, current, SEMANTIC_JOB_FIELDS)
        if current.get("parser_version") != PARSER_VERSION:
            raise ValueError("目标语义快照与运行时解析器版本不一致")
        runtime = Job(id=current["job_id"], version=current["job_version_id"], row=0,
            family=current["category_parent"], **{field: current[field] for field in
            ("title", "company", "category", "salary_raw", "requirements", "description", "address")})
        if any(current[field] != getattr(runtime, field) for field in SEMANTIC_JOB_FIELDS):
            raise ValueError("目标语义快照不能由当前运行时代码逐字段复现")
        job_changes.update(changes)
        provenance["jobs"].append({"job_id":key,"parent_record_sha256":object_sha(previous),
            "runtime_record_sha256":object_sha(current),"changed_fields":changes})
    updated_profiles = []
    for previous in profiles:
        parsed = parse_profile(previous["text"], previous.get("preferences", {}))
        current = {**previous, **{key:parsed[key] for key in SEMANTIC_PROFILE_FIELDS}}
        changes = semantic_changes(previous, current, SEMANTIC_PROFILE_FIELDS)
        profile_changes.update(changes)
        provenance["profiles"].append({"query_id":previous["query_id"],
            "parent_record_sha256":object_sha(previous),"runtime_record_sha256":object_sha(current),"changed_fields":changes})
        updated_profiles.append(current)
    bridge = {"schema":"job-agent-parsed-fields-bridge-v1", "parser_version":PARSER_VERSION,
        "parent_jobs_content_sha256":object_sha(sorted((key,object_sha(row)) for key,row in selected.items())),
        "semantic_jobs_file_sha256":file_sha(semantic_path),
        "parser_source_sha256":{"domain.py":file_sha(domain.__file__),"semantics.py":file_sha(semantics.__file__)},
        "original_task_hashes_verified":True,"raw_fields_unchanged":True,"runtime_reproduction_verified":True,
        "job_changed_fields":dict(job_changes),"profile_changed_fields":dict(profile_changes),
        "record_provenance":provenance}
    return refreshed, updated_profiles, bridge


def request_json(endpoint, path, body=None):
    request = urllib.request.Request(endpoint + path,
        data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def new_file(path):
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path.parent, *path.parents)):
        raise ValueError("冻结文本向量与岗位子集必须写入Git工作区外")
    if path.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    if not path.parent.is_dir():
        raise ValueError("请提供已存在的私有研究目录")
    return path


def atomic_write(path, writer):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            writer(stream)
        # 同文件系统硬链接确保不覆盖已存在结果；不使用可覆盖的rename。
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args):
    endpoint = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path:
        raise ValueError("本命令仅使用已授权的本地HTTP编码服务")
    output = new_file(args.output)
    jobs_output = new_file(args.jobs_output)
    profiles_output = new_file(args.profiles_output) if args.profiles_output else None
    if args.semantic_jobs and profiles_output is None:
        raise ValueError("语义版本桥接必须同时输出运行时画像")
    outputs = [path for path in (output, jobs_output, profiles_output) if path]
    if len(set(outputs)) != len(outputs) or not 1 <= args.batch_size <= 64:
        raise ValueError("输出不能重名，批次需为1至64")
    profiles = read_jsonl(args.profiles)
    tasks = [row for row in read_jsonl(args.tasks) if row.get("kind") == "relevance"]
    if not tasks or len({row["task_id"] for row in tasks}) != len(tasks):
        raise ValueError("相关性任务为空或ID重复")
    profile_index = {row["query_id"]: row for row in profiles}
    if len(profile_index) != len(profiles) or set(profile_index) != {row["query_id"] for row in tasks}:
        raise ValueError("画像与任务查询集合不一致")
    selected_ids = {row["job_id"] for row in tasks}
    selected = selected_rows(args.jobs, selected_ids)
    for task in tasks:
        job, profile = selected[task["job_id"]], profile_index[task["query_id"]]
        if task.get("source_job_sha256") != object_sha(job) or task.get("source_query_sha256") != object_sha(profile):
            raise ValueError("任务岗位/画像版本摘要不一致")
        if task["split"] != job["split"] or task["split"] != profile["split"]:
            raise ValueError("候选与画像跨分区")
        if task.get("input_hash") != object_sha(task["material"]):
            raise ValueError("任务输入材料摘要不一致")
    bridge = None
    if args.semantic_jobs:
        selected, profiles, bridge = refresh_semantics(selected, profiles, args.semantic_jobs)
        bridge["parent_jobs_file_sha256"] = file_sha(args.jobs)
        bridge["parent_profiles_file_sha256"] = file_sha(args.profiles)
    family_splits, profile_splits = defaultdict(set), defaultdict(set)
    for kind, rows, family in (("job", selected.values(), "job_family_id"),
                               ("profile", profiles, "profile_family_id")):
        for row in rows:
            if not row.get(family) or row.get("split") not in ("train", "dev", "test"):
                raise ValueError("家族或分区缺失")
            family_splits[kind, row[family]].add(row["split"])
            if kind == "profile":
                text = canonical_text(row.get("text"))
                if not text:
                    raise ValueError("画像正文为空")
                profile_splits[text].add(row["split"])
    if any(len(values) > 1 for values in (*family_splits.values(), *profile_splits.values())):
        raise ValueError("岗位/画像家族或完全相同画像文本跨分区")
    graphs = [build_graph(selected[key]) for key in sorted(selected)]
    texts = {canonical_text(node.text) for graph in graphs for node in graph.nodes}
    texts.update(canonical_text(profile_text(profile)) for profile in profiles)
    if any(len(text) > 24000 for text in texts):
        raise ValueError("节点属性超出本地服务文本上限；必须调整显式编码合同")
    summary = {"tasks": len(tasks), "jobs": len(selected), "profiles": len(profiles),
               "unique_texts": len(texts),
               "job_splits": dict(Counter(row["split"] for row in selected.values())),
               "profile_splits": dict(Counter(row["split"] for row in profiles)),
               "parser_versions":dict(Counter(row.get("parser_version") for row in selected.values())),
               "node_kinds": dict(Counter(node.kind for graph in graphs for node in graph.nodes)),
               "edges": sum(len(graph.edges) for graph in graphs),
               "unknown_edges": sum(edge.status == "unknown" for graph in graphs for edge in graph.edges),
               "family_overlap": 0, "canonical_ast_compatible": True, "labels_read": 0,
               "semantic_bridge": {key:bridge[key] for key in ("parser_version","raw_fields_unchanged","runtime_reproduction_verified","job_changed_fields","profile_changed_fields")} if bridge else None}
    if args.validate_only:
        print(json.dumps({"状态":"只验证来源和语义，未调用编码服务",**summary},ensure_ascii=False),flush=True)
        return summary
    health = request_json(endpoint, "/health")
    required = ("encoder_contract_hash", "weights_sha256", "revision", "pooling", "dimension", "max_tokens", "normalized")
    if any(key not in health for key in required) or health["normalized"] is not True:
        raise ValueError("服务未提供完整固定权重、池化与归一化合同")
    dimension, vectors = int(health["dimension"]), {}
    ordered = sorted(texts)
    for start in range(0, len(ordered), args.batch_size):
        batch = ordered[start:start + args.batch_size]
        result = request_json(endpoint, "/encode", {"texts": batch, "query": False})
        if result.get("encoder_contract_hash") != health["encoder_contract_hash"]:
            raise ValueError("编码期间服务权重或模板发生变化")
        values = np.asarray(result["vectors"], dtype=np.float32)
        if values.shape != (len(batch), dimension) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=.001):
            raise ValueError("冻结文本向量维度、数值或归一化不合法")
        vectors.update((hashlib.sha256(text.encode()).hexdigest(), vector.tolist()) for text, vector in zip(batch, values))
        print(json.dumps({"已编码节点文本": len(vectors), "总数": len(ordered)}, ensure_ascii=False), flush=True)
    if request_json(endpoint, "/health") != health:
        raise ValueError("运行前后编码服务合同不一致")
    job_rows = [selected[key] for key in sorted(selected)]
    contract = {"schema": "job-agent-group-node-vectors-v2", "graph_version": GRAPH_VERSION,
                "encoder": health, "query_wrapper_applied": False,
                "purpose": "冻结节点/画像文本属性初始化；不是已训练GNN也不是人工金标",
                "selected_job_ids_sha256": object_sha(sorted(selected)),
                "selected_job_versions_sha256": object_sha(sorted((key, object_sha(row)) for key, row in selected.items())),
                "profile_sha256": file_sha(args.profiles), "tasks_sha256": file_sha(args.tasks),
                "node_texts_sha256": object_sha(ordered), "script_sha256": file_sha(__file__),
                "group_graph_sha256": file_sha(Path(__file__).with_name("group_graph.py")),
                "selected_jobs_file_sha256":jsonl_sha(job_rows),
                "runtime_profiles_file_sha256":jsonl_sha(profiles),
                "semantic_bridge":bridge,
                "fit_on_evaluation": False, "label_accessed": False}
    summary["dimension"] = dimension
    atomic_write(output, lambda stream: json.dump({"dimension":dimension,"contract":contract,"vectors":vectors,"summary":summary}, stream, ensure_ascii=False, allow_nan=False))
    atomic_write(jobs_output, lambda stream: stream.writelines(jsonl_chunks(job_rows)))
    if profiles_output:
        atomic_write(profiles_output, lambda stream: stream.writelines(jsonl_chunks(profiles)))
    print(json.dumps({"状态":"完成","向量输出":str(output),"候选岗位输出":str(jobs_output),
        "运行时画像输出":str(profiles_output) if profiles_output else None,
        "output_sha256":{"vectors":file_sha(output),"jobs":file_sha(jobs_output),
                         "profiles":file_sha(profiles_output) if profiles_output else None},**summary}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("jobs", "profiles", "tasks", "output", "jobs-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8093")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--semantic-jobs", type=Path, help="仅解析字段更新的目标岗位快照；原任务仍先由--jobs验证")
    parser.add_argument("--profiles-output", type=Path, help="保留原简历和偏好、用当前解析器更新的画像")
    parser.add_argument("--validate-only", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
