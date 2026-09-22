#!/usr/bin/env python3
"""标题/任务→JD 弱监督领域适配；指标不代表真实简历—岗位匹配效果。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

OFFICIAL_REPO = "Qwen/Qwen3-Embedding-0.6B"
OFFICIAL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_MODEL = "/home/xukunbo/.cache/job-agent/models/qwen3-embedding-0.6b"
QUERY_INSTRUCTION = "根据目标职位名称和岗位类别，检索职责与技能要求相符的招聘描述。"
QUERY_VERSION = "title-category-v1"
DOCUMENT_VERSION = "requirements-description-without-title-v1"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protect_output_path(path: Path) -> Path:
    """私有模型与原文派生产物不能写入任何 Git 工作区（含符号链接）。"""
    resolved = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("输出目录位于 Git 工作区内，拒绝写入；请使用仓库外私有目录")
    return resolved


def dataset_provenance(path: Path) -> dict:
    """若输入由正式管道生成，校验输入文件、原始快照及完整 lineage。"""
    result = {"jobs_sha256": digest_file(path)}
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return {**result, "status": "未提供数据快照 manifest（仅适用于独立样例）"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "job-agent-research-data-v1":
        raise ValueError("不支持的数据快照 manifest schema")
    if manifest.get("files", {}).get(path.name, {}).get("sha256") != result["jobs_sha256"]:
        raise ValueError("jobs.jsonl 与数据快照 manifest 的 SHA256 不一致")
    source_path = Path(manifest["source_path"])
    if not source_path.is_file():
        raise ValueError("数据快照的原始源文件不可读，无法验证源快照")
    source_sha = digest_file(source_path)
    if source_sha != manifest.get("source_sha256") or source_sha != manifest.get("config", {}).get("snapshot"):
        raise ValueError("原始文件与数据快照记录的 SHA256 不一致")
    if not manifest.get("checks", {}).get("source_lineage_complete"):
        raise ValueError("数据快照未通过完整来源链检查")
    return {**result, "status": "已校验输入文件与源快照",
            "manifest_path": str(manifest_path.resolve()), "manifest_sha256": digest_file(manifest_path),
            "source_path": str(source_path.resolve()), "source_sha256": source_sha,
            "config_hash": manifest.get("config_hash"), "schema_version": manifest["schema_version"]}


def normalized(value) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def query_text(row: dict) -> str:
    return f"目标职位：{normalized(row['title'])}\n岗位类别：{normalized(row.get('category'))}"


def document_text(row: dict) -> str:
    # 仅用于弱监督适配；不让重复标题本身直接提供答案。
    title = normalized(row["title"])
    chunks = []
    for name, label in [("requirements", "岗位要求"), ("description", "工作职责")]:
        body = normalized(row.get(name))
        if title:
            body = re.sub(re.escape(title), "", body, flags=re.IGNORECASE).strip()
        if body:
            chunks.append(f"{label}：{body}")
    return "\n".join(chunks)


def prepare_data(path: Path) -> tuple[dict, dict]:
    rows = {"train": [], "dev": [], "test": []}
    identifiers = set()
    family_splits, content_splits, encoded_content_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    discarded = Counter()
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            for required in ("job_id", "title", "job_family_id", "split"):
                if not normalized(row.get(required)):
                    raise ValueError(f"第 {line_number} 行缺少必需字段 {required}")
            job_id = str(row["job_id"])
            if job_id in identifiers:
                raise ValueError(f"第 {line_number} 行 job_id 重复；拒绝训练")
            identifiers.add(job_id)
            split = {"validation": "dev", "valid": "dev", "val": "dev"}.get(row["split"], row["split"])
            if split not in rows:
                raise ValueError(f"第 {line_number} 行 split 不受支持")
            family_splits[str(row["job_family_id"])].add(split)
            raw_content = normalized(row.get("requirements")) + "\n" + normalized(row.get("description"))
            if raw_content.strip():
                content_splits[hashlib.sha256(raw_content.encode()).hexdigest()].add(split)
            document = document_text(row)
            if document:
                encoded_content_splits[hashlib.sha256(normalized(document).casefold().encode()).hexdigest()].add(split)
            if len(document) < 20:
                discarded[f"{split}_内容过短"] += 1
                continue
            query = query_text(row)
            rows[split].append({"job_id": job_id, "family": str(row["job_family_id"]),
                                "query": query, "query_key": normalized(query).casefold(),
                                "document": document})
    leaked_families = sum(len(value) > 1 for value in family_splits.values())
    leaked_content = sum(len(value) > 1 for value in content_splits.values())
    leaked_encoded = sum(len(value) > 1 for value in encoded_content_splits.values())
    if leaked_families or leaked_content or leaked_encoded:
        raise ValueError(f"分区泄漏：{leaked_families} 个岗位家族、{leaked_content} 个原始正文、{leaked_encoded} 个模型输入正文跨 split；拒绝训练")
    for split in rows:
        rows[split].sort(key=lambda r: r["job_id"])
    # 数据行不移动。评测查询另行排除训练中已经出现的相同标题+类别。
    # 候选文档保持该 split 的完整岗位集合，留出范围写入 manifest。
    seen_queries = set(r["query_key"] for r in rows["train"])
    queries = {"train": rows["train"]}
    for split in ("dev", "test"):
        unique = {}
        for row in rows[split]:
            if row["query_key"] in seen_queries:
                discarded[f"{split}_查询与先前分区相同"] += 1
            else:
                unique.setdefault(row["query_key"], row)
        queries[split] = list(unique.values())
        seen_queries.update(r["query_key"] for r in rows[split])
    stats = {"input_rows": total, "corpus_counts": {s: len(r) for s, r in rows.items()},
             "eligible_query_counts": {s: len(r) for s, r in queries.items()},
             "excluded": dict(discarded), "family_overlap": leaked_families,
             "exact_document_overlap": leaked_content,
             "encoded_document_overlap": leaked_encoded,
             "evaluation_query_policy": "先冻结分区；dev 排除与 train 相同查询；test 排除与 train/dev 相同查询；每种留出查询仅取一个代表",
             "split_scope": "训练只用 train；dev/test 的候选库各自独立；不是全库人岗匹配评测"}
    return {"corpora": rows, "queries": queries}, stats


def masks_for_batch(batch, torch, device):
    positive = torch.tensor([[a["query_key"] == b["query_key"] for b in batch] for a in batch],
                            dtype=torch.bool, device=device)
    same_family = torch.tensor([[a["family"] == b["family"] for b in batch] for a in batch],
                               dtype=torch.bool, device=device)
    # 同一完整查询的多个 JD 为此代理任务的多正例；同家族其他查询的 JD 排除出负例。
    allowed = (~same_family) | positive
    informative = (allowed & ~positive).any(dim=1)
    return positive, allowed, informative


def last_token_pool(last_hidden_state, attention_mask, torch):
    # 与官方实现一致：左 padding 时最后一位就是最后有效 token。
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), lengths]


def encode(model, tokenizer, texts, is_query, max_length, device, torch):
    if is_query:
        texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{value}" for value in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model(**inputs, return_dict=True, use_cache=False)
    vectors = last_token_pool(output.last_hidden_state, inputs["attention_mask"], torch)
    return torch.nn.functional.normalize(vectors.float(), p=2, dim=1)


def evaluate(model, tokenizer, query_rows, corpus, args, torch, split_name):
    # 每次从相同有序留出集合以独立局部 RNG 抽样，基座/适配/重载严格一致。
    query_rows = list(query_rows)
    if args.eval_limit and len(query_rows) > args.eval_limit:
        query_rows = random.Random(f"{args.seed}:{split_name}:eval-query-v1").sample(query_rows, args.eval_limit)
    query_ids_sha256 = hashlib.sha256("\n".join(row["job_id"] for row in query_rows).encode()).hexdigest()
    if not query_rows:
        return {"status": "未执行：无严格留出查询", "split": split_name}
    if not corpus:
        return {"status": "未执行：无候选文档", "split": split_name}
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        document_vectors = []
        for start in range(0, len(corpus), args.eval_batch_size):
            document_vectors.append(encode(model, tokenizer,
                [r["document"] for r in corpus[start:start + args.eval_batch_size]], False,
                args.max_length, args.device, torch).cpu())
        documents = torch.cat(document_vectors)
        scores = []
        for start in range(0, len(query_rows), args.eval_batch_size):
            q = encode(model, tokenizer, [r["query"] for r in query_rows[start:start + args.eval_batch_size]],
                       True, args.max_length, args.device, torch).cpu()
            scores.append(q @ documents.T)
        similarities = torch.cat(scores)
    k = min(10, len(corpus))
    indices = torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :k].tolist()
    records = []
    for query, ranked in zip(query_rows, indices):
        relevant = {i for i, r in enumerate(corpus)
                    if r["query_key"] == query["query_key"] or r["family"] == query["family"]}
        hits = [i in relevant for i in ranked]
        recall = sum(hits) / len(relevant)
        mrr = next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0)
        dcg = sum(float(hit) / math.log2(rank + 2) for rank, hit in enumerate(hits))
        idcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
        records.append({"query_job_id": query["job_id"], "relevant_count": len(relevant),
                        "retrieved_ids": [corpus[i]["job_id"] for i in ranked],
                        "recall_at_10": recall, "mrr_at_10": mrr, "ndcg_at_10": dcg / idcg})
    mean = lambda name: sum(r[name] for r in records) / len(records)
    return {"status": "已执行", "task": "弱监督标题类别到JD检索，非人岗金标", "split": split_name,
            "queries": len(records), "corpus_size": len(corpus), "requested_k": 10, "effective_k": k,
            "query_sampling": "固定 seed 的无放回随机查询抽样；候选文档不抽样",
            "query_ids_sha256": query_ids_sha256,
            "recall_at_10": mean("recall_at_10"), "mrr_at_10": mean("mrr_at_10"),
            "ndcg_at_10": mean("ndcg_at_10"), "seconds": time.monotonic() - started,
            "relevance_rule": "与查询相同的标题+类别，或同岗位近重复家族；是代理标签，不是人工适岗判断",
            "per_query": records}


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument("--expected-revision", default=OFFICIAL_REVISION)
    p.add_argument("--gpu", default="3", help="进程可见的物理 GPU；在导入 torch 前设置")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8, help="实际 InfoNCE 查询数；每查询一正例，无梯度累积")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=0, help="仅 smoke 使用；0=全量训练分区")
    p.add_argument("--eval-limit", type=int, default=0, help="评测查询数上限；候选库不截断；0=全部")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--evaluate-test", action="store_true", help="仅配置冻结后作一次最终留出评测")
    p.add_argument("--validate-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.batch_size < 2 or args.epochs < 1 or args.temperature <= 0 or args.max_length < 32:
        raise ValueError("batch_size≥2、epochs≥1、temperature>0、max_length≥32")
    if args.eval_batch_size < 1 or min(args.train_limit, args.eval_limit, args.max_steps) < 0:
        raise ValueError("数量参数必须非负，eval_batch_size≥1")
    args.output_dir = protect_output_path(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录非空，拒绝覆盖；请选择新的运行目录")
    source_provenance = dataset_provenance(args.jobs_jsonl)
    data, stats = prepare_data(args.jobs_jsonl)
    stats["source_snapshot"] = source_provenance
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "data_validation.json", stats)
    if args.validate_only:
        print(json.dumps(stats, ensure_ascii=False))
        return
    provenance_path = args.model_dir / "job_agent_provenance.json"
    if not provenance_path.is_file():
        raise ValueError("缺少官方模型 provenance；拒绝使用未经固定 revision 的模型目录")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("repo_id") != OFFICIAL_REPO or provenance.get("revision") != args.expected_revision:
        raise ValueError("模型来源或 revision 与预期不一致")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    args.device = "cpu" if args.cpu else "cuda:0"
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("指定 GPU 不可用；不自动占用其他 GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cpu:
        torch.cuda.manual_seed_all(args.seed)
        smoke = torch.randn(8, 8, device=args.device, dtype=torch.bfloat16, requires_grad=True)
        (smoke @ smoke.T).float().mean().backward()
        del smoke
        torch.cuda.reset_peak_memory_stats()
    train_rows = list(data["corpora"]["train"])
    random.Random(args.seed).shuffle(train_rows)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if len({r["query_key"] for r in train_rows}) < 2:
        raise ValueError("至少需要两种训练查询构成合法负例")
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {"task": "weak_title_to_jd_domain_adaptation", "not_person_job_fit_gold": True,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset_sha256": digest_file(args.jobs_jsonl), "script_sha256": digest_file(Path(__file__)),
                "source_snapshot": source_provenance,
                "model_source": provenance, "arguments": args_dict, "data": stats,
                "query_template": QUERY_VERSION, "document_template": DOCUMENT_VERSION,
                "instruction": QUERY_INSTRUCTION, "training_rows_used": len(train_rows),
                "actual_infonce_query_batch": args.batch_size, "gradient_accumulation_steps": 1,
                "explicit_hard_negatives": 0, "loss": "带多正例与同家族屏蔽的 in-batch InfoNCE",
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                "dependencies": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "peft", "accelerate", "safetensors")},
                "limitations": ["只有岗位数据，监督来自标题/类别，不是用户适岗反馈",
                                "同标题类别可能含不同职责，代理标签有噪声",
                                "未显式标为正例的其他家族仍可能是假负例",
                                "当前原型按最大长度截断，需单独测长文覆盖率",
                                "小步 smoke 仅验证工程，不能证明微调有效"]}
    if not args.cpu:
        manifest["hardware"] = {"visible_physical_gpu": args.gpu, "name": torch.cuda.get_device_name(0),
                                "capability": torch.cuda.get_device_capability(0),
                                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    write_json(args.output_dir / "manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                                     dtype=torch.float32 if args.cpu else torch.bfloat16,
                                     attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    evaluation_sets = ["dev", "test"] if args.evaluate_test else ["dev"]
    baseline = {s: evaluate(model, tokenizer, data["queries"][s], data["corpora"][s], args, torch, s)
                for s in evaluation_sets}
    write_json(args.output_dir / "baseline_metrics.json", baseline)
    print(json.dumps({"stage": "baseline_complete", "dev_queries": baseline["dev"].get("queries", 0)}, ensure_ascii=False), flush=True)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
            lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    manifest["trainable_parameters"] = sum(p.numel() for p in trainable)
    manifest["all_parameters"] = sum(p.numel() for p in model.parameters())
    write_json(args.output_dir / "manifest.json", manifest)
    steps = 0
    skipped = 0
    started = time.monotonic()
    log_path = args.output_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(train_rows)
            model.train()
            for offset in range(0, len(train_rows), args.batch_size):
                batch = train_rows[offset:offset + args.batch_size]
                if len(batch) < 2:
                    skipped += 1
                    continue
                positive, allowed, informative = masks_for_batch(batch, torch, args.device)
                if not bool(informative.any().item()):
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                query_vectors = encode(model, tokenizer, [r["query"] for r in batch], True,
                                       args.max_length, args.device, torch)
                document_vectors = encode(model, tokenizer, [r["document"] for r in batch], False,
                                          args.max_length, args.device, torch)
                logits = (query_vectors @ document_vectors.T) / args.temperature
                numerators = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
                denominators = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
                loss = (denominators - numerators)[informative].mean()
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("损失非有限值，停止训练")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not bool(torch.isfinite(grad_norm).item()):
                    raise RuntimeError("梯度非有限值，停止训练")
                optimizer.step()
                steps += 1
                record = {"step": steps, "epoch": epoch + 1, "loss": float(loss.detach()),
                          "gradient_norm": float(grad_norm), "actual_query_batch": len(batch),
                          "informative_queries": int(informative.sum()),
                          "positive_pairs": int(positive.sum()), "negative_pairs": int((allowed & ~positive).sum()),
                          "learning_rate": args.learning_rate, "elapsed_seconds": time.monotonic() - started}
                if not args.cpu:
                    record["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if args.max_steps and steps >= args.max_steps:
                    break
            if args.max_steps and steps >= args.max_steps:
                break
    if steps == 0:
        raise RuntimeError("所有批次都缺少合法负例，未训练；拒绝生成成功标记")
    training_seconds = time.monotonic() - started
    model.eval()
    model.save_pretrained(args.output_dir / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / "adapter")
    final = {s: evaluate(model, tokenizer, data["queries"][s], data["corpora"][s], args, torch, s)
             for s in evaluation_sets}
    write_json(args.output_dir / "adapted_metrics.json", final)
    manifest.update({"status": "completed", "optimizer_steps": steps, "skipped_batches": skipped,
                     "training_seconds": training_seconds,
                     "test_evaluated": args.evaluate_test,
                     "selection_policy": "固定训练预算的最终检查点；不按 test 选择；大实验需额外实现 dev 早停"})
    if not args.cpu:
        manifest["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir), "steps": steps}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
