#!/usr/bin/env python3
"""在虚构简历画像上生成 Qwen 基座/LoRA 排名；无人工金标，不计算相关性指标。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import time


RESUME_INSTRUCTION = "根据简历中的经历、技能与求职意向，检索职责和任职要求相符的招聘描述。"
QUERY_TEMPLATE_VERSION = "resume-text-intent-v1"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_id_sha(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def protected_output(path):
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("拒绝向 Git 工作区内写入私有研究产物")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("输出路径非空或不是目录，拒绝覆盖")
    return path


def load_contract(args):
    training = json.loads((args.training_run / "manifest.json").read_text())
    if training.get("status") != "completed":
        raise ValueError("LoRA 训练未完成")
    frozen_script = args.training_run / "frozen_train_embedding.py"
    if file_sha(frozen_script) != training["script_sha256"]:
        raise ValueError("冻结训练代码与训练 manifest 不一致")
    spec = importlib.util.spec_from_file_location("frozen_embedding_contract", frozen_script)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    source = contract.dataset_provenance(args.jobs_jsonl)
    if source["jobs_sha256"] != training["dataset_sha256"]:
        raise ValueError("召回岗位快照与 LoRA 训练源快照不同")
    _, integrity = contract.prepare_data(args.jobs_jsonl)
    adapter_sha = file_sha(args.training_run / "adapter/adapter_model.safetensors")
    report = json.loads((args.training_run / "run_analysis.json").read_text())
    if adapter_sha != report["adapter_sha256"]:
        raise ValueError("adapter 权重与已校验训练产物不同")
    benchmark = json.loads((args.benchmark_dir / "manifest.json").read_text())
    if benchmark.get("schema") != "job-agent-benchmark-v1":
        raise ValueError("不支持的评测画像 schema")
    if benchmark["config"]["jobs_sha256"] != source["jobs_sha256"]:
        raise ValueError("benchmark 与召回岗位快照不同")
    for name in ("queries.jsonl", "pools.jsonl", "tasks.jsonl", "qrels.jsonl"):
        if file_sha(args.benchmark_dir / name) != benchmark["files"][name]["sha256"]:
            raise ValueError(f"benchmark {name} 已变化；请生成新版本后执行")
    if (args.benchmark_dir / "qrels.jsonl").stat().st_size:
        raise ValueError("本运行预定为无金标排名生成；检测到 qrels，请改用新版本方案")
    tasks = read_jsonl(args.benchmark_dir / "tasks.jsonl")
    if any(row.get("label") is not None for row in tasks):
        raise ValueError("人工任务存在已填标签，与本次无金标运行前提不同")
    queries = sorted(read_jsonl(args.benchmark_dir / "queries.jsonl"), key=lambda row: row["query_id"])
    pools = {row["query_id"]: row for row in read_jsonl(args.benchmark_dir / "pools.jsonl")}
    if len({row["query_id"] for row in queries}) != len(queries) or set(pools) != {row["query_id"] for row in queries}:
        raise ValueError("查询 ID 重复或人工池与查询集合不一致")
    if any(row["split"] not in ("dev", "test") for row in queries):
        raise ValueError("画像只能属于 dev/test")
    if any(row.get("is_real_resume") or row.get("is_human_gold") for row in queries):
        raise ValueError("本运行仅针对已指定的虚构、未审核画像")
    jobs = sorted(read_jsonl(args.jobs_jsonl), key=lambda row: row["job_id"])
    family_representatives = {"dev": {}, "test": {}}
    for row in jobs:
        if row["split"] in family_representatives:
            family_representatives[row["split"]].setdefault(row["job_family_id"], row)
    corpora = {split: sorted(group.values(), key=lambda row: row["job_id"])
               for split, group in family_representatives.items()}
    for split, rows in corpora.items():
        if len(rows) != benchmark["candidate_universe_counts"][split]:
            raise ValueError("家族代表候选数与 benchmark 不一致")
    catalog = {split: {row["job_id"]: row["job_family_id"] for row in rows} for split, rows in corpora.items()}
    for query in queries:
        pool = pools[query["query_id"]]
        if pool["split"] != query["split"]:
            raise ValueError("人工池与画像分区不同")
        for candidate in pool["candidates"]:
            if catalog[query["split"]].get(candidate["job_id"]) != candidate["job_family_id"]:
                raise ValueError("人工池候选不是同分区的最小 job_id 家族代表")
    return contract, training, source, integrity, adapter_sha, benchmark, queries, pools, corpora


def resume_query(row, contract):
    intent = row.get("preferences", {}).get("intent", "")
    return f"求职意向：{contract.normalized(intent)}\n简历正文：{contract.normalized(row.get('text'))}"


def encode(model, tokenizer, texts, query, args, torch, contract):
    if query:
        texts = [f"Instruct: {RESUME_INSTRUCTION}\nQuery: {text}" for text in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    output = model(**inputs, return_dict=True, use_cache=False)
    vectors = contract.last_token_pool(output.last_hidden_state, inputs["attention_mask"], torch)
    return torch.nn.functional.normalize(vectors.float(), p=2, dim=1)


def generate_rankings(name, model, tokenizer, queries, pools, corpora, args, torch, contract):
    model.eval()
    results = []
    split_stats = {}
    for split in ("dev", "test"):
        rows = corpora[split]
        texts = [contract.document_text(row) for row in rows]
        valid_indices = [index for index, text in enumerate(texts) if text.strip()]
        missing_indices = [index for index, text in enumerate(texts) if not text.strip()]
        model_start = time.perf_counter()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            vectors = []
            for start in range(0, len(valid_indices), args.batch_size):
                indices = valid_indices[start:start + args.batch_size]
                vectors.append(encode(model, tokenizer, [texts[index] for index in indices], False,
                                      args, torch, contract).cpu())
            document_vectors = torch.cat(vectors)
            torch.cuda.synchronize()
            index_seconds = time.perf_counter() - started
            # 固定一次预热，不计入逐查询延迟，也不缓存待测查询向量。
            encode(model, tokenizer, ["求职意向：软件开发\n简历正文：具有基础编程与项目实践经历。"],
                   True, args, torch, contract)
            torch.cuda.synchronize()
            for query in (row for row in queries if row["split"] == split):
                torch.cuda.synchronize()
                started = time.perf_counter()
                vector = encode(model, tokenizer, [resume_query(query, contract)], True,
                                args, torch, contract).cpu()
                torch.cuda.synchronize()
                encoded = time.perf_counter()
                scores = (vector @ document_vectors.T)[0]
                ranked = torch.argsort(scores, descending=True, stable=True)[:args.top_k].tolist()
                ended = time.perf_counter()
                pool_ids = {row["job_id"] for row in pools[query["query_id"]]["candidates"]}
                hits = [{"rank": rank + 1, "job_id": rows[valid_indices[index]]["job_id"],
                         "job_family_id": rows[valid_indices[index]]["job_family_id"],
                         "cosine": float(scores[index]),
                         "in_existing_annotation_pool": rows[valid_indices[index]]["job_id"] in pool_ids,
                         "human_relevance_label": None}
                        for rank, index in enumerate(ranked)]
                if any(not math.isfinite(hit["cosine"]) for hit in hits):
                    raise RuntimeError("发现非有限相似度")
                outside = [hit["job_id"] for hit in hits[:10] if not hit["in_existing_annotation_pool"]]
                results.append({"query_id": query["query_id"], "split": split,
                                "scenario_group": query.get("scenario_group"), "model": name,
                                "query_sha256": hashlib.sha256(resume_query(query, contract).encode()).hexdigest(),
                                "is_real_resume": False, "is_human_gold": False,
                                "candidate_universe_count": len(rows), "indexable_candidate_count": len(valid_indices),
                                "top_k": args.top_k, "ranking": hits,
                                "top10_outside_existing_pool_count": len(outside),
                                "top10_outside_existing_pool_job_ids": outside,
                                "top10_without_human_labels_count": min(10, len(hits)),
                                "latency_ms": {"query_encoding_with_device_sync": (encoded - started) * 1000,
                                               "cpu_cosine_and_stable_sort": (ended - encoded) * 1000,
                                               "total_without_corpus_index_or_model_loading": (ended - started) * 1000}})
        split_records = [row for row in results if row["split"] == split]
        split_stats[split] = {"queries": len(split_records), "candidate_universe_count": len(rows),
                              "indexable_candidate_count": len(valid_indices),
                              "empty_document_job_ids": [rows[index]["job_id"] for index in missing_indices],
                              "corpus_encoding_seconds": index_seconds,
                              "total_split_seconds": time.perf_counter() - model_start,
                              "top10_outside_existing_pool_pairs": sum(row["top10_outside_existing_pool_count"] for row in split_records),
                              "top10_pairs": sum(min(10, len(row["ranking"])) for row in split_records),
                              "all_top10_pairs_without_human_labels": sum(row["top10_without_human_labels_count"] for row in split_records),
                              "mean_query_latency_ms": sum(row["latency_ms"]["total_without_corpus_index_or_model_loading"] for row in split_records) / len(split_records)}
        print(json.dumps({"model": name, "split": split, **split_stats[split]}, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: row["query_id"])
    path = args.output_dir / f"{name}_rankings.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    return results, split_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs-jsonl", required=True, type=Path)
    p.add_argument("--benchmark-dir", required=True, type=Path)
    p.add_argument("--training-run", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--gpu", default="3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.top_k < 10 or args.max_length < 32:
        raise ValueError("batch-size≥1、top-k≥10、max-length≥32")
    args.output_dir = protected_output(args.output_dir)
    contract, training, source, integrity, adapter_sha, benchmark, queries, pools, corpora = load_contract(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": "虚构简历格式召回工程验证，非人岗收益评测", "status": "validated",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "script_sha256": file_sha(__file__), "training_script_sha256": training["script_sha256"],
                "adapter_sha256": adapter_sha, "source_snapshot": source,
                "benchmark_manifest_sha256": file_sha(args.benchmark_dir / "manifest.json"),
                "benchmark_config_hash": benchmark["config_hash"],
                "queries_sha256": file_sha(args.benchmark_dir / "queries.jsonl"),
                "pools_sha256": file_sha(args.benchmark_dir / "pools.jsonl"),
                "query_count": len(queries), "query_split_counts": dict(Counter(row["split"] for row in queries)),
                "candidate_universe_counts": {split: len(rows) for split, rows in corpora.items()},
                "candidate_id_hashes": {split: ordered_id_sha([row["job_id"] for row in rows]) for split, rows in corpora.items()},
                "candidate_policy": "同 split 所有家族各取字典序最小 job_id，与 benchmark 人工池代表口径一致",
                "empty_document_policy": "保留候选宇宙记录；空正文没有可定义的文本向量，不参与余弦排名，等价于排在所有可编码候选之后；不填充标题",
                "short_document_policy": "非空短正文照常编码；不改动训练时的去标题正文构造",
                "document_template": contract.DOCUMENT_VERSION,
                "query_template": QUERY_TEMPLATE_VERSION,
                "query_source_fields": ["text", "preferences.intent"],
                "query_instruction": RESUME_INSTRUCTION,
                "training_instruction": training["instruction"],
                "instruction_change": "训练为标题类别→JD；本运行改为简历正文与意向→JD，无额外训练，存在任务分布变化",
                "official_query_wrapper": "Instruct: {instruction}\\nQuery: {query}",
                "pooling": "官方最后有效token，L2归一化，精确余弦与稳定排序",
                "labels_status": "qrels为空；池内和池外候选都未人工标注，池外不视作负例",
                "is_real_resume": False, "is_human_gold": False,
                "relevance_metrics_computed": False, "test_rankings_only_no_tuning": True,
                "structural_isolation_checks": integrity,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "limitations": ["48份画像完全虚构且未经人工审阅，不是实际用户简历",
                                "仅验证模型可服务简历格式输入，不证明微调召回更好",
                                "没有应用薪资/学历等硬过滤；不能直接作为最终产品推荐",
                                "意图缺失、指令注入等stress画像仍生成排名，仅用于审阅，不表示应推荐",
                                "查询最大512tokens；延迟为当前机器共享负载下单次测量，非性能基准"]}
    write_json(args.output_dir / "manifest.json", manifest)
    if args.validate_only:
        print(json.dumps({key: manifest[key] for key in ("query_count", "query_split_counts", "candidate_universe_counts", "candidate_id_hashes")}, ensure_ascii=False))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("指定 GPU 不可用，不自动切换 GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.device = "cuda:0"
    model_dir = Path(training["arguments"]["model_dir"])
    provenance = json.loads((model_dir / "job_agent_provenance.json").read_text())
    if provenance != training["model_source"] or file_sha(model_dir / "model.safetensors") != provenance["model_safetensors_sha256"]:
        raise ValueError("官方基座来源或权重已发生变化")
    smoke = torch.randn(8, 8, device=args.device, dtype=torch.bfloat16)
    if not bool(torch.isfinite(smoke @ smoke.T).all()):
        raise RuntimeError("BF16 GPU smoke失败")
    del smoke
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                     dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    model.requires_grad_(False)
    baseline, baseline_stats = generate_rankings("frozen_qwen", model, tokenizer, queries, pools, corpora, args, torch, contract)
    model = PeftModel.from_pretrained(model, args.training_run / "adapter", is_trainable=False)
    adapted, adapted_stats = generate_rankings("adapted_qwen", model, tokenizer, queries, pools, corpora, args, torch, contract)
    summary = {"no_relevance_labels": True, "no_relevance_metrics": True,
               "frozen_qwen": baseline_stats, "adapted_qwen": adapted_stats,
               "pool_note": "池外数量是待补充审阅范围，不表示不相关；目前所有池内条目也未标注。"}
    supplements = {}
    for result in baseline + adapted:
        for hit in result["ranking"][:10]:
            if not hit["in_existing_annotation_pool"]:
                key = (result["query_id"], hit["job_id"])
                row = supplements.setdefault(key, {"query_id": result["query_id"], "split": result["split"],
                         "job_id": hit["job_id"], "job_family_id": hit["job_family_id"], "label": None,
                         "annotation_status": "not_yet_in_pool", "is_human_gold": False, "private_pool_sources": []})
                row["private_pool_sources"].append({"model": result["model"], "rank": hit["rank"], "cosine": hit["cosine"]})
    supplement_path = args.output_dir / "supplemental_top10_review_candidates.jsonl"
    supplement_path.write_text("".join(json.dumps(supplements[key], ensure_ascii=False) + "\n" for key in sorted(supplements)), encoding="utf-8")
    summary["unique_supplemental_query_job_pairs"] = len(supplements)
    summary["supplemental_note"] = "仅输出待审阅候选，不修改既有人工池；展示给标注者时必须隐藏模型来源、分数和排名。"
    write_json(args.output_dir / "summary.json", summary)
    manifest.update({"status": "completed", "model_source": provenance,
                     "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
                     "hardware": {"physical_gpu": args.gpu, "name": torch.cuda.get_device_name(0)},
                     "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "outputs": {path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
                                 for path in args.output_dir.iterdir() if path.name != "manifest.json"}})
    write_json(args.output_dir / "manifest.json", manifest)
    shutil.copyfile(__file__, args.output_dir / "frozen_retrieve_resumes.py")
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir),
                      "unique_supplemental_query_job_pairs": len(supplements)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
