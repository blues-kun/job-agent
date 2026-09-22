"""审核、冻结与回放共享的数据合同；只验证来源记录，不能认证真人身份。"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL每行必须是对象")
    return rows


def new_private_directory(path):
    path = Path(path).expanduser().resolve()
    def git_marker(parent):
        marker = parent/".git"
        # 空目录不是Git仓库；有效worktree文件或含HEAD的Git目录才是标记。
        return marker.is_file() or marker.is_dir() and (marker/"HEAD").is_file()
    if any(git_marker(parent) for parent in [path, *path.parents]):
        raise ValueError("私有输出不得位于Git工作区")
    if path.exists():
        raise FileExistsError("输出目录已存在，拒绝覆盖")
    os.umask(0o077)
    path.mkdir(parents=True, mode=0o700)
    return path


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n")
    Path(path).chmod(0o600)


def identifier(value):
    return isinstance(value, str) and bool(value) and value == value.strip() and value.lower() not in {"null", "none", "unknown", "todo", "待填写"}


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("缺少时区")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("时间必须为含时区的ISO字符串") from exc
    return result


def unique_rows(rows, key):
    output = {}
    for row in rows:
        value = row.get(key)
        if not identifier(value) or value in output:
            raise ValueError(f"{key}缺失、占位或重复")
        output[value] = row
    return output


def profile_family(row):
    value = row.get("profile_family_id", row.get("profile_family"))
    if not identifier(value):
        raise ValueError("缺少profile_family_id；必须先导入独立画像家族，不能按query_id补造")
    if row.get("profile_family") and row.get("profile_family_id") and row["profile_family"] != row["profile_family_id"]:
        raise ValueError("两种画像家族字段不一致")
    return value


def validate_profiles(rows):
    lookup = unique_rows(rows, "query_id")
    assignments, text_families = {}, {}
    for row in rows:
        family = profile_family(row)
        if row.get("split") not in {"train", "dev", "test"}:
            raise ValueError("画像缺少合法分区")
        identity = (row["split"], row.get("purpose"))
        if assignments.setdefault(family, identity) != identity:
            raise ValueError("画像家族跨分区或用途")
        text = " ".join(str(row.get("text", "")).split())
        if not text:
            raise ValueError("画像正文不能为空")
        key = digest(text)
        if text_families.setdefault(key, family) != family:
            raise ValueError("完全相同画像正文被赋予不同家族")
    return lookup


def validate_model_review(review, expected_input, allow_abstention=False):
    if not isinstance(review, dict) or any(not identifier(review.get(key)) for key in ("model", "model_revision", "prompt_hash", "input_hash", "request_id")):
        raise ValueError("模型评审缺model/revision/prompt_hash/input_hash/request_id")
    if review["input_hash"] != expected_input:
        raise ValueError("模型评审输入血缘不一致")
    if allow_abstention and "grade" in review and review["grade"] is None:
        return
    if type(review.get("grade")) is not int or review["grade"] not in range(4):
        raise ValueError("模型评审等级必须为0至3；初审可显式null，最终裁定不可不确定")


def validate_qrel(row, allow_model_labels=False):
    source = row.get("label_source")
    if source == "llm_reviewed":
        if not allow_model_labels:
            raise ValueError("模型标签默认禁止；仅显式allow_model_labels可计算教师相对指标或研究训练")
        reviews = row.get("model_reviews")
        if not identifier(row.get("input_hash")) or not isinstance(reviews, list) or len(reviews) < 2:
            raise ValueError("模型弱标需要两个通道及输入hash")
        if len({review.get("channel") for review in reviews}) != len(reviews) or any(not identifier(review.get("channel")) for review in reviews):
            raise ValueError("模型评审通道必须明确且不重复")
        for review in reviews:
            validate_model_review(review, row["input_hash"], allow_abstention=True)
        rule = row.get("rule_validation", {})
        if rule.get("passed") is not True or rule.get("input_hash") != row["input_hash"] or not identifier(rule.get("validator_version")) or not isinstance(rule.get("checks"), dict) or not rule["checks"] or any(value is not True for value in rule["checks"].values()):
            raise ValueError("模型弱标必须通过有版本和输入血缘的命名规则检查")
        adjudication = row.get("model_adjudication")
        if len({review["grade"] for review in reviews}) > 1 or adjudication is not None:
            validate_model_review(adjudication, row["input_hash"])
            if adjudication["grade"] != row.get("grade") or not identifier(adjudication.get("rationale")):
                raise ValueError("模型分歧必须显式仲裁并解释，不默认多数票")
        elif row.get("grade") != reviews[0]["grade"]:
            raise ValueError("最终弱标签与一致的两个评审不同")
    elif source == "human_adjudicated":
        fields = ("task_hash", "adjudication_id", "adjudicator_id", "adjudication_sha256", "registry_sha256")
        if any(not identifier(row.get(key)) for key in fields):
            raise ValueError("人工qrels缺材料、身份登记或显式仲裁血缘")
        reviewers, people, review_ids = (row.get(key) for key in ("reviewer_ids", "reviewer_person_ids", "review_ids"))
        if any(not isinstance(values, list) or len(values) < 2 or any(not identifier(value) for value in values)
               or len(set(values)) != len(values) for values in (reviewers, people, review_ids)) or len({len(values) for values in (reviewers, people, review_ids)}) != 1:
            raise ValueError("人工qrels需要至少两位已登记独立评审及完整原始记录")
        hashes = row.get("review_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(review_ids) or any(not identifier(value) for value in hashes.values()):
            raise ValueError("人工qrels未绑定全部原始评审hash")
    else:
        raise ValueError("仅接受带评审及显式人工仲裁血缘的qrels")
    if type(row.get("grade")) is not int or row["grade"] not in range(4):
        raise ValueError("标签必须为人工审核后的0至3整数；未标注不是0")
    if row.get("kind", "relevance") != "relevance":
        raise ValueError("追问/抽取标签不能混入相关性qrels")


def build_evaluation_contract(queries, pools, qrels, source_hashes=None, k=10, allow_model_labels=False):
    if type(k) is not int or k < 1:
        raise ValueError("K必须为正整数")
    lookup = validate_profiles(queries)
    if not lookup or set(pools) != set(lookup):
        raise ValueError("共同查询与候选池集合必须完全一致")
    for row in queries:
        if row.get("scenario_group") == "clarification_stress" or row.get("purpose") == "ranker_training":
            raise ValueError("追问压力任务或排序训练画像不得混入推荐评测")
    clean_pools = {}
    for query, ids in pools.items():
        values = list(ids)
        if not values or any(not identifier(value) for value in values) or len(values) != len(set(values)):
            raise ValueError("共同候选池必须非空且岗位ID唯一")
        clean_pools[query] = sorted(values)
    labels = {}
    for row in qrels:
        validate_qrel(row, allow_model_labels)
        pair = (row["query_id"], row["job_id"])
        if pair in labels:
            raise ValueError("相关性标签重复")
        if row.get("profile_family_id") is not None and row["profile_family_id"] != profile_family(lookup.get(row["query_id"], {})):
            raise ValueError("标签与评测画像家族不同")
        if row.get("split") is not None and row["split"] != lookup.get(row["query_id"], {}).get("split"):
            raise ValueError("标签与评测画像分区不同")
        labels[pair] = row["grade"]
    expected = {(query, job) for query, jobs in clean_pools.items() for job in jobs}
    if set(labels) != expected:
        raise ValueError("正式冻结需要全共同池已审，且qrels不能含池外或其他查询")
    query_contract = [{"query_id": row["query_id"], "profile_family_id": profile_family(row), "split": row["split"],
                       "query_payload_sha256": digest({"text": row["text"], "preferences": row.get("preferences", {})})}
                      for row in sorted(queries, key=lambda row: row["query_id"])]
    payload = {"schema": "job-agent-evaluation-v2", "k": k, "queries": query_contract, "pools": clean_pools,
               "query_set_sha256": digest(query_contract), "pool_sha256": digest(clean_pools),
               "qrels_content_sha256": digest(sorted(qrels, key=lambda row: (row["query_id"], row["job_id"]))),
               "source_files": source_hashes or {}, "label_sources": sorted({row["label_source"] for row in qrels}),
               "metrics_scope": "教师相对的弱标共同池；不代表人工人岗效果" if any(row["label_source"] == "llm_reviewed" for row in qrels) else "人工已审共同池；非全库Recall"}
    return {**payload, "contract_sha256": digest(payload)}


def verify_evaluation_contract(contract, qrels, allow_model_labels=False):
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if contract.get("schema") != "job-agent-evaluation-v2" or digest(payload) != contract.get("contract_sha256"):
        raise ValueError("冻结评测合同hash不一致")
    if digest(contract["queries"]) != contract["query_set_sha256"] or digest(contract["pools"]) != contract["pool_sha256"]:
        raise ValueError("冻结查询或共同池hash失配")
    if digest(sorted(qrels, key=lambda row: (row["query_id"], row["job_id"]))) != contract["qrels_content_sha256"]:
        raise ValueError("qrels已变更，必须重新冻结评测版本")
    expected = {(query, job) for query, jobs in contract["pools"].items() for job in jobs}
    if {(row["query_id"], row["job_id"]) for row in qrels} != expected or len(qrels) != len(expected):
        raise ValueError("冻结qrels不完整或重复")
    for row in qrels:
        validate_qrel(row, allow_model_labels)
    return contract
