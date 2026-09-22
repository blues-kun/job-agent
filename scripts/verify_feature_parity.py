"""用明确虚构样例，核对HTTP推荐与独立进程离线回放的35维特征。

同一全库、偏好、查询模式与编码器下比较；不测排名质量、不读取标注答案。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from job_agent.corpus import Corpus
from job_agent.ranking import FEATURE_NAMES, FEATURE_VERSION
from job_agent.retrieval_contract import contract_info
from job_agent.workflow import Workflow
from research.common import private_path, read_jsonl, sha256, write_json


def verify(url, source, dataset, dense_dir, output):
    if urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("验收只允许访问本机服务")
    output = private_path(output)
    if output.exists():
        raise FileExistsError("验收版本已存在，请使用新文件")
    manifest = json.loads((dataset / "manifest.json").read_text())
    source_sha = sha256(source)
    if manifest["source_sha256"] != source_sha or sha256(dataset / "jobs.jsonl") != manifest["files"]["jobs.jsonl"]["sha256"]:
        raise ValueError("岗位来源摘要不一致")
    if manifest["config"]["retrieval_contract"]["hash"] != contract_info()["hash"]:
        raise ValueError("岗位版本与当前查询合同不一致")
    corpus = Corpus(source, dense_dir=dense_dir, records=read_jsonl(dataset / "jobs.jsonl"))
    if corpus.dense is None:
        raise ValueError("真实编码器不可用，不能用回退引擎完成一致性验证")
    workflow = Workflow(corpus)
    rows = []
    with httpx.Client(base_url=url, timeout=90, trust_env=False) as client:
        def call(path, payload=None, method=None):
            response = client.request(method or ("GET" if payload is None else "POST"), path, json=payload)
            response.raise_for_status()
            return response.json()
        health = call("/api/v2/health")
        if health["snapshot"] != source_sha[:16] or health["jobs"] != len(corpus.jobs):
            raise ValueError("HTTP与离线语料不一致")
        samples = call("/api/v2/samples")["samples"]
        try:
            for sample in samples:
                if sample["id"] not in {"python-junior", "transition-data", "frontend"}:
                    continue
                payload = {"text": sample["text"], "preferences": sample["preferences"], "limit": 10,
                           "research_consent": False, "method": "hybrid"}
                preview = call("/api/v2/profile/preview", payload)
                if preview["conflicts"]:
                    raise ValueError("虚构验收画像发生字段冲突")
                payload["preferences"] = preview["preferences"]
                confirmed = call("/api/v2/profile/confirm", {**payload, "user_confirmed": True, "acknowledged_conflicts": []})
                payload["confirmation_token"] = confirmed["confirmation_token"]
                for mode in ("intent", "experience", "structured"):
                    online = call("/api/v2/recommend", {**payload, "query_mode": mode})
                    if online["action"] != "recommend" or not online["jobs"]:
                        raise ValueError("虚构验收样例没有返回待比较岗位")
                    if online["retrieval_contract"]["hash"] != contract_info()["hash"] or online["feature_version"] != FEATURE_VERSION:
                        raise ValueError("HTTP使用的特征/解析版本不同")
                    if online["encoder_contract"] != corpus.dense.encoder_contract or any("fallback" in name for name in online.get("source_counts", {})):
                        raise ValueError("HTTP编码器与离线版本不同")
                    job_ids = [row["id"] for row in online["jobs"]]
                    offline = workflow.replay_features(sample["text"], payload["preferences"], job_ids, query_mode=mode)
                    maximum_difference = 0.0
                    missing = 0
                    for left, right in zip(online["jobs"], offline):
                        if left["id"] != right["job_id"] or not right["eligible"]:
                            raise ValueError("离线岗位身份或硬条件资格不一致")
                        if "dense" not in right["retrieval"]:
                            raise ValueError("离线回放编码失败，不能混用回退特征")
                        if set(left["features"]) != set(FEATURE_NAMES) or set(right["features"]) != set(FEATURE_NAMES):
                            raise ValueError("排序特征名称不一致")
                        for name in FEATURE_NAMES:
                            a, b = left["features"][name], right["features"][name]
                            if a is None or b is None:
                                if a is not None or b is not None:
                                    raise ValueError(f"未知值被替换：{name}")
                                missing += 1
                                continue
                            if not np.isclose(a, b, atol=1e-6, rtol=1e-6):
                                raise ValueError(f"HTTP/离线特征不一致：{name}")
                            maximum_difference = max(maximum_difference, abs(a-b))
                    rows.append({"sample_id": sample["id"], "query_mode": mode, "pairs": len(job_ids),
                                 "checked_features": len(job_ids)*len(FEATURE_NAMES), "unknown_values_preserved": missing,
                                 "max_absolute_difference": maximum_difference})
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            if call("/api/v2/research-events")["events"]:
                raise ValueError("默认调用意外保留研究事件")
        finally:
            call("/api/v2/journey", method="DELETE")
            call("/api/v2/feedback", method="DELETE")
    if len(rows) != 9 or sha256(source) != source_sha:
        raise ValueError("样例验收不完整或原表变化")
    result = {"status": "passed", "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "三份虚构画像、三种查询模式、相同已知全库的HTTP/离线特征一致性；不是人岗效果验证。",
              "source_sha256": source_sha, "jobs_sha256": sha256(dataset / "jobs.jsonl"),
              "retrieval_contract_hash": contract_info()["hash"], "feature_version": FEATURE_VERSION,
              "encoder_contract_hash": corpus.dense.encoder_contract["encoder_contract_hash"],
              "script_sha256": sha256(__file__), "atol": 1e-6, "rtol": 1e-6, "rows": rows,
              "pairs": sum(row["pairs"] for row in rows), "checked_features": sum(row["checked_features"] for row in rows),
              "max_absolute_difference": max(row["max_absolute_difference"] for row in rows),
              "raw_resume_stored": False, "human_gold_created": False}
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8094")
    for name in ("source", "dataset", "dense-dir", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.url, args.source, args.dataset, args.dense_dir, args.output)
