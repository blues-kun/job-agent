"""从冻结标签和固定检索回放物化精排样本；不现场补造检索特征或负例。"""
import argparse
import json
from pathlib import Path
import math

from job_agent.ranking import FEATURE_NAMES, FEATURE_VERSION
from research.review_contracts import (digest, file_sha, identifier, new_private_directory, profile_family,
    read_jsonl, unique_rows, validate_profiles, validate_qrel, write_json, write_jsonl)


def query_payload_sha256(query):
    return digest({"text": query["text"], "preferences": query.get("preferences", {})})


def materialize(dataset, profiles, qrels, replay=None, replay_manifest=None, *, allow_model_labels=False, evaluation_profiles=None):
    if replay is None or replay_manifest is None:
        raise ValueError("必须提供固定retrieval replay和manifest；不能把未重放的检索特征默认缺失")
    if evaluation_profiles is None:
        raise ValueError("必须提供独立evaluation_profiles排除清单，证明本次训练来源与评測来源隔离")
    dataset, profiles, qrels, replay, replay_manifest = map(Path, (dataset, profiles, qrels, replay, replay_manifest))
    data_manifest = json.loads((dataset/"manifest.json").read_text("utf-8"))
    jobs_sha = file_sha(dataset/"jobs.jsonl")
    if jobs_sha != data_manifest["files"]["jobs.jsonl"]["sha256"]:
        raise ValueError("岗位数据hash失配")
    jobs = unique_rows(read_jsonl(dataset/"jobs.jsonl"), "job_id")
    profile_rows = read_jsonl(profiles); queries = validate_profiles(profile_rows)
    excluded = read_jsonl(evaluation_profiles); validate_profiles(excluded)
    excluded_families = {profile_family(row) for row in excluded}
    excluded_ids = {row["query_id"] for row in excluded}
    excluded_text = {digest(" ".join(row["text"].split())) for row in excluded}
    if not excluded:
        raise ValueError("独立评测画像排除清单不能为空")
    for query in queries.values():
        if query.get("purpose") != "ranker_training" or query["split"] not in {"train", "dev"}:
            raise ValueError("仅接受独立ranker_training用途的train/dev画像；不得拿评測test画像训练")
        if query["query_id"] in excluded_ids or profile_family(query) in excluded_families or digest(" ".join(query["text"].split())) in excluded_text:
            raise ValueError("排序训练与独立评测画像ID/家族/正文重合")
    contract = json.loads(replay_manifest.read_text("utf-8"))
    if contract.get("schema") != "job-agent-ranker-replay-v2":
        raise ValueError("不支持的检索回放合同")
    expected = {"replay_sha256": file_sha(replay), "jobs_sha256": jobs_sha, "profiles_sha256": file_sha(profiles),
                "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES)}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("回放数据、画像、岗位或特征合同SHA失配")
    if not identifier(contract.get("retrieval_contract_hash")) or not contract.get("encoder_contract"):
        raise ValueError("回放缺检索/编码器合同")
    replay_by_pair = {}
    for row in read_jsonl(replay):
        pair = (row.get("query_id"), row.get("job_id"))
        if pair in replay_by_pair:
            raise ValueError("同一人岗对有多条回放，不能任意挑选")
        replay_by_pair[pair] = row
    rows = []
    for label in read_jsonl(qrels):
        validate_qrel(label, allow_model_labels)
        query, source = queries.get(label.get("query_id")), jobs.get(label.get("job_id"))
        if query is None or source is None:
            raise ValueError("标签引用未知画像或岗位")
        if query["split"] != source["split"] or source["split"] not in {"train", "dev"}:
            raise ValueError("查询和岗位分区不一致或含test训练数据")
        item = replay_by_pair.get((query["query_id"], source["job_id"]))
        if item is None:
            raise ValueError("已审人岗对缺固定检索回放；不补零、不重算")
        identities = {"query_id": query["query_id"], "profile_family_id": profile_family(query), "job_id": source["job_id"],
            "job_family_id": source["job_family_id"], "split": query["split"], "snapshot": source["snapshot"],
            "feature_version": FEATURE_VERSION, "query_payload_sha256": query_payload_sha256(query),
            "retrieval_contract_hash": contract["retrieval_contract_hash"], "encoder_contract": contract["encoder_contract"]}
        if any(item.get(key) != value for key, value in identities.items()):
            raise ValueError("回放身份、原文输入、分区、snapshot或检索编码器合同不一致")
        for key in ("profile_family_id", "job_family_id", "split"):
            if key in label and label[key] != identities[key]:
                raise ValueError("标签血缘与物化身份不一致")
        features = item.get("features", {})
        if set(features) != set(FEATURE_NAMES):
            raise ValueError("回放特征名必须与当前运行时完整合同一致")
        if any(value is not None and (type(value) not in {int, float} or not math.isfinite(value)) for value in features.values()):
            raise ValueError("回放特征仅允许有限数值或显式null")
        rows.append({**label, **identities, "features": features, "purpose": "ranker_training",
            "replay_sha256": expected["replay_sha256"], "replay_manifest_sha256": file_sha(replay_manifest),
            "profiles_sha256": expected["profiles_sha256"], "qrels_sha256": file_sha(qrels),
            "excluded_evaluation_profiles_sha256": file_sha(evaluation_profiles), "eligible_for_production": False})
    from research.train_ranker import validate_rows
    validate_rows(rows, allow_model_labels=allow_model_labels)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "profiles", "qrels", "replay", "replay-manifest", "evaluation-profiles", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--allow-model-labels", action="store_true")
    args = parser.parse_args()
    rows = materialize(args.dataset, args.profiles, args.qrels, args.replay, args.replay_manifest,
                       allow_model_labels=args.allow_model_labels, evaluation_profiles=args.evaluation_profiles)
    target = new_private_directory(args.output)
    write_jsonl(target/"ranker_rows.jsonl", rows)
    write_json(target/"manifest.json", {"schema": "job-agent-ranker-materialized-v2", "rows_sha256": file_sha(target/"ranker_rows.jsonl"),
        "input_hashes": {str(path): file_sha(path) for path in [args.profiles, args.qrels, args.replay, args.replay_manifest, args.evaluation_profiles]},
        "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION,
        "retrieval_contract_hash": rows[0]["retrieval_contract_hash"], "encoder_contract": rows[0]["encoder_contract"],
        "label_sources": sorted({row["label_source"] for row in rows}), "eligible_for_production": False})
    print(json.dumps({"rows": len(rows), "output": str(target), "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
