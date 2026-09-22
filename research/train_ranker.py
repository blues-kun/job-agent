"""固定检索回放上的LambdaRank；人工与教师弱监督严格分源，仅作影子研究。"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np

from job_agent.ranking import FEATURE_NAMES, FEATURE_VERSION
from research.metrics import ranking_metrics
from research.review_contracts import digest, file_sha, identifier, new_private_directory, read_jsonl, validate_qrel, write_json


def validate_rows(rows, allow_fixture=False, allow_model_labels=False):
    seen = set(); families = defaultdict(set); grouped = defaultdict(list)
    snapshots = set(); versions = set(); contracts = set(); encoders = set(); query_identity = {}; job_identity = {}
    for row in rows:
        required = {"query_id", "profile_family_id", "job_id", "job_family_id", "split", "grade", "label_source", "features", "snapshot", "feature_version"}
        if not required.issubset(row):
            raise ValueError("缺少排序样本来源、profile_family_id、标签或特征合同")
        if any(not identifier(row[key]) for key in ("query_id", "profile_family_id", "job_id", "job_family_id")):
            raise ValueError("查询/岗位及家族ID缺失")
        fixture = allow_fixture and row["label_source"] == "synthetic_fixture"
        if not fixture:
            if row["label_source"] not in {"human_adjudicated", "llm_reviewed"}:
                raise ValueError("不接受旧合成标签、未审核标签或跳过即负例")
            validate_qrel(row, allow_model_labels)
            if row.get("purpose") != "ranker_training" or row["split"] not in {"train", "dev"}:
                raise ValueError("正式训练只允许独立train/dev用途数据；test另由冻结评测器评估")
            provenance = ("replay_sha256", "replay_manifest_sha256", "query_payload_sha256", "profiles_sha256",
                          "qrels_sha256", "excluded_evaluation_profiles_sha256", "retrieval_contract_hash")
            if any(not identifier(row.get(key)) for key in provenance) or not row.get("encoder_contract"):
                raise ValueError("排序样本缺固定replay、原文、独立评测排除或检索/编码器来源")
            contracts.add(row["retrieval_contract_hash"]); encoders.add(digest(row["encoder_contract"]))
        if type(row["grade"]) is not int or row["grade"] not in range(4):
            raise ValueError("标签须为0至3整数；未知不等于0")
        split = row["split"]
        if split not in {"train", "dev", "test"}:
            raise ValueError("分区不合法")
        pair = (row["query_id"], row["job_id"])
        if pair in seen:
            raise ValueError("查询岗位对重复")
        seen.add(pair)
        if query_identity.setdefault(row["query_id"], (row["profile_family_id"], split)) != (row["profile_family_id"], split):
            raise ValueError("同画像ID更换家族或分区")
        if job_identity.setdefault(row["job_id"], (row["job_family_id"], split)) != (row["job_family_id"], split):
            raise ValueError("同岗位ID更换家族或分区")
        families["profile:"+row["profile_family_id"]].add(split); families["job:"+row["job_family_id"]].add(split)
        snapshots.add(row["snapshot"]); versions.add(row["feature_version"])
        if set(row["features"]) != set(FEATURE_NAMES):
            raise ValueError("排序特征名不符合当前固定合同")
        if any(value is not None and (type(value) not in {int, float} or not math.isfinite(value)) for value in row["features"].values()):
            raise ValueError("特征仅允许有限数值或显式null；不能用字符串、布尔或非标准NaN")
        grouped[(split, row["query_id"])].append(row)
    if any(row["label_source"] == "synthetic_fixture" for row in rows) and not all(row["label_source"] == "synthetic_fixture" for row in rows):
        raise ValueError("夹具不能与真实研究标签混合训练")
    if not rows or len(snapshots) != 1 or versions != {FEATURE_VERSION}:
        raise ValueError("数据快照或特征版本不一致")
    if len(contracts) > 1 or len(encoders) > 1:
        raise ValueError("不能混合不同检索/编码器合同训练")
    if any(len(value) > 1 for value in families.values()):
        raise ValueError("画像或岗位家族跨分区泄漏")
    if not {"train", "dev"}.issubset({key[0] for key in grouped}):
        raise ValueError("需要独立训练集和开发集")
    for (split, query), group in grouped.items():
        if len(group) < 2 or split == "train" and len({row["grade"] for row in group}) < 2:
            raise ValueError("每个训练查询须有至少两档已审核候选；不静默筛除无信息查询")
    return grouped


def train(input_path, output, allow_fixture=False, dataset=None, allow_model_labels=False):
    rows = read_jsonl(input_path); groups = validate_rows(rows, allow_fixture, allow_model_labels)
    fixture_only = all(row["label_source"] == "synthetic_fixture" for row in rows)
    if not fixture_only:
        if dataset is None:
            raise ValueError("正式研究训练必须提供数据版本目录")
        dataset = Path(dataset); contract = json.loads((dataset/"manifest.json").read_text("utf-8"))
        if file_sha(dataset/"jobs.jsonl") != contract["files"]["jobs.jsonl"]["sha256"]:
            raise ValueError("岗位版本hash不符")
        jobs = {row["job_id"]: row for row in read_jsonl(dataset/"jobs.jsonl")}
        for row in rows:
            job = jobs.get(row["job_id"])
            if job is None or any(row[key] != job[key] for key in ("job_family_id", "split", "snapshot")):
                raise ValueError("训练标签岗位不属于声明版本/分区")
    import lightgbm as lgb
    target = new_private_directory(output)
    ordered = {split: [group for (part, _), group in sorted(groups.items()) if part == split] for split in ("train", "dev", "test")}
    def arrays(part):
        flat = [row for group in part for row in group]
        return (np.asarray([[np.nan if row["features"][key] is None else row["features"][key] for key in FEATURE_NAMES] for row in flat]),
                np.asarray([row["grade"] for row in flat]), [len(group) for group in part])
    x, y, g = arrays(ordered["train"]); dx, dy, dg = arrays(ordered["dev"])
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=200, learning_rate=.04, num_leaves=15,
                          max_depth=5, min_child_samples=10, reg_lambda=1.0, colsample_bytree=.9, random_state=42, n_jobs=4, verbosity=-1)
    model.fit(x, y, group=g, eval_set=[(dx, dy)], eval_group=[dg], eval_at=[10], feature_name=list(FEATURE_NAMES), callbacks=[lgb.early_stopping(20, verbose=False)])
    model.booster_.save_model(str(target/"model.txt"))
    results = {}
    for split in ("dev", "test"):
        values = []
        for group in ordered[split]:
            inputs, _, _ = arrays([group]); scores = model.predict(inputs)
            ranked = [group[index]["job_id"] for index in np.argsort(-scores, kind="stable")]
            values.append(ranking_metrics(ranked, {row["job_id"]: row["grade"] for row in group}))
        ndcg = [value["ndcg"] for value in values if value["ndcg"] is not None]
        results[split] = {"queries": len(values), "ndcg_queries": len(ndcg), "ndcg_at_10": float(np.mean(ndcg)) if ndcg else None}
    teacher_relative = any(row["label_source"] == "llm_reviewed" for row in rows)
    write_json(target/"metrics.json", {"task": "固定回放候选的按查询精排", "fixture_only": fixture_only, "teacher_relative": teacher_relative,
        "results": results, "eligible_for_production": False, "note": "教师指标不是人工效果；训练内部候选指标不等于所有基线共同池的正式比较。"})
    write_json(target/"manifest.json", {"schema": "job-agent-ranker-v2", "input_sha256": file_sha(input_path), "snapshot": rows[0]["snapshot"],
        "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES), "retrieval_contract_hash": rows[0].get("retrieval_contract_hash"),
        "encoder_contract": rows[0].get("encoder_contract"), "label_sources": sorted({row["label_source"] for row in rows}), "teacher_relative": teacher_relative,
        "seed": 42, "model_sha256": file_sha(target/"model.txt"), "best_iteration": model.best_iteration_, "eligible_for_production": False,
        "shadow_load_allowed": not fixture_only, "fixture_only": fixture_only, "note": "仅允许明确研究影子加载；默认推荐路径不得自动晋升。"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path); parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--allow-model-labels", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.allow_fixture, args.dataset, args.allow_model_labels), ensure_ascii=False))


if __name__ == "__main__":
    main()
