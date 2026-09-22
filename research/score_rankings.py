"""固定共同查询和已审池评测；正式比较拒绝漏查询、池外候选和变更标签。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from research.metrics import ranking_metrics, paired_bootstrap
from research.review_contracts import (digest, file_sha, read_jsonl, validate_qrel,
                                      verify_evaluation_contract, new_private_directory, write_json)


def score(rankings, qrels, pools, k=10, *, expected_queries=None, strict=False, allow_model_labels=False):
    if type(k) is not int or k < 1:
        raise ValueError("K必须为正整数")
    grades = defaultdict(dict)
    for row in qrels:
        validate_qrel(row, allow_model_labels)
        if row["job_id"] in grades[row["query_id"]]:
            raise ValueError("同一对不能重复提供标签")
        grades[row["query_id"]][row["job_id"]] = row["grade"]
    expected_queries = set(pools) if expected_queries is None else set(expected_queries)
    if set(pools) != expected_queries:
        raise ValueError("共同查询与池集合不一致")
    by_query = {}
    for row in rankings:
        query = row["query_id"]
        if query in by_query:
            raise ValueError("同一模型的查询排名重复")
        by_query[query] = row
    if set(by_query) != expected_queries:
        raise ValueError("模型缺查询或含额外查询，拒绝改变评测分母")
    rows = []
    for query in sorted(expected_queries):
        ids = [job["job_id"] for job in by_query[query]["ranking"]]
        if len(ids) != len(set(ids)):
            raise ValueError("排名包含重复岗位")
        expected = set(pools[query])  # 冻结共同池，不并入单模型topK。
        outside = set(ids[:k])-expected
        if outside:
            raise ValueError("TopK含共同池外岗位；必须先合并所有系统候选、补标并重新冻结")
        missing = expected-set(grades[query])
        if missing or not expected:
            if strict:
                raise ValueError("共同池尚未全部人工仲裁，拒绝正式比较")
            rows.append({"query_id": query, "status": "待补标", "missing": len(missing), "metrics": None})
        else:
            metrics = ranking_metrics(ids, {key: grades[query][key] for key in expected}, k)
            rows.append({"query_id": query, "status": "已评分", "missing": 0, "metrics": metrics})
    complete = bool(rows) and all(row["metrics"] is not None for row in rows)
    summary = {"queries": len(rows), "fully_judged_queries": sum(row["metrics"] is not None for row in rows),
               "ndcg_at_k": None, "pool_recall_at_k": None, "mrr_at_k": None, "metric_query_counts": {}}
    if complete:
        for key, field in [("ndcg_at_k", "ndcg"), ("pool_recall_at_k", "pool_recall"), ("mrr_at_k", "mrr")]:
            values = [row["metrics"][field] for row in rows if row["metrics"][field] is not None]
            summary[key] = float(np.mean(values)) if values else None
            summary["metric_query_counts"][key] = len(values)
    return {"rows": rows, "summary": summary, "all_queries_judged": complete,
            "common_pool_sha256": digest({query: sorted(pools[query]) for query in sorted(pools)}),
            "note": "固定K，少返回缺位增益为0；零IDCG的nDCG为N/A，grade>=2为空时Recall/MRR为N/A；仅已审共同池。"}


def compare_models(models, qrels, contract, allow_model_labels=False):
    verify_evaluation_contract(contract, qrels, allow_model_labels)
    if len(models) < 2:
        raise ValueError("正式比较至少需要两个明确命名的模型")
    families = {row["query_id"]: row["profile_family_id"] for row in contract["queries"]}
    if any(not value for value in families.values()):
        raise ValueError("缺少画像家族，不能作独立样本推断")
    results = {name: score(rows, qrels, contract["pools"], contract["k"], expected_queries=families, strict=True, allow_model_labels=allow_model_labels)
               for name, rows in models.items()}
    # 已提供的候选宇宙与输入合同必须一致；缺元数据时不伪称完成输入公平性认证。
    contexts = defaultdict(list)
    query_payloads = {row["query_id"]: row["query_payload_sha256"] for row in contract["queries"]}
    fields = ("query_payload_sha256", "candidate_universe_count", "candidate_universe_sha256", "query_sha256", "document_view_sha256", "constraint_policy_sha256")
    for rows in models.values():
        for row in rows:
            if row.get("query_payload_sha256") is not None and row["query_payload_sha256"] != query_payloads[row["query_id"]]:
                raise ValueError("模型实际画像输入与冻结text/preferences不一致")
            for field in fields:
                contexts[(row["query_id"], field)].append(row.get(field))
    if any(len(set(values)) > 1 for values in contexts.values()):
        raise ValueError("跨模型候选宇宙、查询或输入字段合同不一致")
    names = list(models)
    comparisons = []
    reference = {row["query_id"]: row["metrics"]["ndcg"] for row in results[names[0]]["rows"]}
    for name in names[1:]:
        current = {row["query_id"]: row["metrics"]["ndcg"] for row in results[name]["rows"]}
        eligible = [key for key in sorted(reference) if reference[key] is not None and current[key] is not None]
        result = paired_bootstrap([reference[key] for key in eligible], [current[key] for key in eligible],
                                  clusters=[families[key] for key in eligible])
        comparisons.append({"reference": names[0], "candidate": name, "ndcg_delta": result,
                            "excluded_zero_idcg_queries": len(reference)-len(eligible)})
    return {"models": results, "comparisons": comparisons, "contract_sha256": contract["contract_sha256"],
            "label_sources": contract["label_sources"], "teacher_relative": "llm_reviewed" in contract["label_sources"],
            "eligible_for_production": False, "stage": contract.get("stage", "frozen"),
            "pool_sha256": contract["pool_sha256"], "query_set_sha256": contract["query_set_sha256"],
            "input_context_fully_recorded": all(all(value is not None for value in values) for values in contexts.values()),
            "scope": "共同池与查询分母已验证；输入上下文缺记录时，不能仅凭本表声称检索输入完全可比"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, nargs="+", required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--allow-model-labels", action="store_true", help="明确评估教师相对弱标；不能称为人工金标")
    parser.add_argument("--output", type=Path, required=True, help="新的仓库外结果目录")
    args = parser.parse_args()
    if len({path.name for path in args.rankings}) != len(args.rankings):
        raise ValueError("排名文件basename重复，请明确区分模型")
    qrels = read_jsonl(args.qrels)
    contract = json.loads(args.contract.read_text("utf-8"))
    expected_sha = contract.get("source_files", {}).get("qrels.jsonl")
    if expected_sha and expected_sha != file_sha(args.qrels):
        raise ValueError("qrels文件SHA与冻结版本不一致")
    result = compare_models({path.name: read_jsonl(path) for path in args.rankings}, qrels, contract, args.allow_model_labels)
    result["source_files"] = {str(path): file_sha(path) for path in [args.contract, args.qrels, *args.rankings]}
    target = new_private_directory(args.output)
    write_json(target/"metrics.json", result)
    print(json.dumps({key: value["summary"] for key, value in result["models"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
