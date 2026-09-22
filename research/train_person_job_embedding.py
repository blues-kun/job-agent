"""基于已审核人岗对微调Qwen；仅显式已审核难负例参与InfoNCE，未知关系全部屏蔽。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def load_module(path, name):
    path = Path(path).resolve()
    package, parent = [], path.parent
    while (parent / "__init__.py").is_file():
        package.insert(0, parent.name)
        parent = parent.parent
    if package:
        if str(parent) not in sys.path:
            sys.path.append(str(parent))
        module = importlib.import_module(".".join([*package, path.stem]))
        if Path(module.__file__).resolve() != path:
            raise ValueError("已加载模块与显式合同文件不一致")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def training_groups(groups):
    eligible, excluded = {}, {}
    for (split, query), rows in groups.items():
        if split != "train":
            continue
        positive = [row["job_id"] for row in rows if row["grade"] >= 2]
        negative = [row["job_id"] for row in rows if row["grade"] == 0 and row.get("hard_negative") is True]
        if positive and negative:
            eligible[query] = {"positive": sorted(positive), "negative": sorted(negative)}
        else:
            excluded[query] = {"positive_count": len(positive), "explicit_hard_negative_count": len(negative)}
    if not eligible:
        raise ValueError("没有同时具备已审核正例和显式已审核难负例的训练查询")
    return eligible, excluded


def reviewed_masks(query_ids, document_ids, lookup, torch, device="cpu"):
    positive = torch.tensor([[lookup.get((query, job), {}).get("grade", -1) >= 2 for job in document_ids]
                             for query in query_ids], dtype=torch.bool, device=device)
    negative = torch.tensor([[lookup.get((query, job), {}).get("grade") == 0 and
                              lookup[(query, job)].get("hard_negative") is True for job in document_ids]
                             for query in query_ids], dtype=torch.bool, device=device)
    informative = positive.any(1) & negative.any(1)
    return positive, negative, informative


def segment_scores(query_vectors, document_vectors, view_counts, torch):
    from job_agent.encoder import aggregate_dense_scores
    similarities = query_vectors @ document_vectors.T
    rows, offset = [], 0
    for count in view_counts:
        if count < 1:
            raise ValueError("每个画像至少一个完整文本视图")
        values = similarities[offset:offset + count]
        rows.append(aggregate_dense_scores(values[0], values[1:].max(dim=0).values if count > 1 else None))
        offset += count
    if offset != len(query_vectors):
        raise ValueError("分块长度与实际编码数不一致")
    return torch.stack(rows)


def multipositive_loss(query_vectors, document_vectors, positive, negative, temperature, torch, scores=None):
    if temperature <= 0:
        raise ValueError("温度必须为正")
    if not bool((positive.any(1) & negative.any(1)).all()):
        raise ValueError("每个训练查询必须有已审核正例与负例")
    logits = (query_vectors @ document_vectors.T if scores is None else scores) / temperature
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~(positive | negative), -torch.inf), dim=1)
    return (denominator - numerator).mean()


def prepare_rendering(jobs, profiles, renderer):
    queries = {key: renderer.render_query(row, mode="structured") for key, row in profiles.items()}
    documents = {key: renderer.render_document(row) for key, row in jobs.items()}
    for name, values, records in (("query", queries, profiles), ("document", documents, jobs)):
        splits = defaultdict(set)
        for key, text in values.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{name}渲染为空，拒绝使用标题/占位文本补造")
            normalized = " ".join(text.split()).casefold()
            splits[hashlib.sha256(normalized.encode()).hexdigest()].add(records[key]["split"])
        if any(len(parts) > 1 for parts in splits.values()):
            raise ValueError(f"{name}经过统一模板后出现跨分区相同输入")
    return queries, documents


def evaluate(model, encode, queries, documents, groups, split, torch, query_views=None):
    from research.train_group_graph import metrics_for_group, mean_metric
    model.eval()
    output, cached = [], {}
    with torch.inference_mode():
        for (part, query_id), rows in sorted(groups.items()):
            if part != split:
                continue
            ids = sorted(row["job_id"] for row in rows)
            missing = [job for job in ids if job not in cached]
            for start in range(0, len(missing), 8):
                batch = missing[start:start + 8]
                vectors = encode([documents[job] for job in batch], False).cpu()
                cached.update(zip(batch, vectors))
            views = query_views[query_id] if query_views else [queries[query_id]]
            query = encode(views, True).cpu()
            scores = segment_scores(query, torch.stack([cached[job] for job in ids]), [len(views)], torch)[0].tolist()
            ranked = [ids[index] for index in sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))]
            output.append({"query_id": query_id, "candidate_ids": ids, "ranked_ids": ranked,
                           "candidate_scores": scores, **metrics_for_group(ranked, {row["job_id"]: row["grade"] for row in rows})})
    return {"split": split, "queries": len(output),
            **{key: mean_metric(output, key) for key in ("ndcg_at_10", "mrr_at_10", "pool_recall_at_10")},
            "scope": "固定已审核候选池，非全库Recall；标签来源必须一起解读", "per_query": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("jobs", "profiles", "qrels", "output", "retrieval-contract"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--base-helper", type=Path, default=Path(__file__).with_name("train_embedding.py"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/xukunbo/.cache/job-agent/models/qwen3-embedding-0.6b"))
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2, help="实际query批次；不是累计梯度负例池")
    parser.add_argument("--positives-per-query", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--fixture-encoder", action="store_true", help="仅fixture使用的小型CPU投影，不是Qwen或LoRA结果")
    parser.add_argument("--allow-model-labels", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.positives_per_query, args.negatives_per_query, args.epochs, args.max_steps) < 1 or args.max_length < 32 or min(args.learning_rate, args.temperature) <= 0:
        raise ValueError("训练数量和学习率/温度必须为正，max-length≥32")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu or args.fixture_encoder else args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from research.group_graph import (FrozenTextFeatures, file_sha, object_sha, private_output,
                                      read_jsonl, validate_supervision, write_json)
    output = private_output(args.output)
    labels = read_jsonl(args.qrels)
    jobs, profiles, groups, label_contract = validate_supervision(read_jsonl(args.jobs), read_jsonl(args.profiles), labels,
                                   args.allow_fixture, allow_model_labels=args.allow_model_labels)
    eligible, excluded = training_groups(groups)
    if args.fixture_encoder and not label_contract["fixture_only"]:
        raise ValueError("小型fixture encoder不能训练human或模型教师标签")
    renderer = load_module(args.retrieval_contract, "person_job_retrieval_contract")
    from job_agent.encoder import LocalEncoder, wrap_texts
    instruction = renderer.RESUME_INSTRUCTION
    queries, documents = prepare_rendering(jobs, profiles, renderer)
    query_views = {key: renderer.query_segments(row, mode="structured") for key, row in profiles.items()}
    manifest = {"task": "已审核简历画像→岗位对比学习", "status": "validated", "label_contract": label_contract,
                "input_hashes": {name: file_sha(getattr(args, name)) for name in ("jobs", "profiles", "qrels")},
                "script_sha256": file_sha(__file__), "review_contract_sha256": file_sha(Path(__file__).with_name("review_contracts.py")),
                "retrieval_contract": renderer.contract_info(), "retrieval_contract_file_sha256": file_sha(args.retrieval_contract),
                "rendered_query_hash": object_sha(sorted(queries.items())), "rendered_document_hash": object_sha(sorted(documents.items())),
                "rendered_query_views_hash": object_sha(sorted(query_views.items())),
                "training_and_inference_aggregation": "0.6*global+0.4*best_project；仅一个视图时使用global",
                "document_version_hash": object_sha(sorted((key, job.get("job_version_id"), job["snapshot"]) for key, job in jobs.items())),
                "instruction": instruction, "official_wrapper": "Instruct: {instruction}\\nQuery: {text}",
                "training_queries": len(eligible), "excluded_training_queries": excluded,
                "positive_rule": "当前query对该document的已审核grade>=2",
                "negative_rule": "当前query对该document已审核grade=0且hard_negative=true；其它关系不进分母",
                "unknown_pair_policy": "完全屏蔽；同批其它query正例不自动成为负例",
                "gradient_accumulation_steps": 1, "actual_query_batch_upper_bound": args.batch_size,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "selection": "固定预算最终检查点，无test调参；默认只评dev",
                "eligible_for_production": False,
                "limitations": ["模型教师弱标签不是人工金标，教师相对增益不证明真实人岗收益",
                                "当前评估仅已审核候选池；未评岗位不作负例",
                                "小数据结果不能代替多seed、外部人工留出和线上验证"]}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    if args.validate_only:
        print(json.dumps({"status": "validated", "eligible_queries": len(eligible), "label_contract": label_contract}, ensure_ascii=False))
        return
    import torch
    from torch import nn
    from torch.nn import functional as F
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = "cpu" if args.cpu or args.fixture_encoder else "cuda:0"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("指定GPU不可用，不自动使用其它GPU")
    if args.fixture_encoder:
        features = FrozenTextFeatures(dimension=64)
        model = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 32)).to(device)
        def encode(texts, query):
            values = wrap_texts(texts, query, "last_token", instruction)
            return F.normalize(model(torch.stack([features(value) for value in values]).to(device)), dim=1)
        manifest["backend"] = "CPU hash投影fixture；不是Qwen/LoRA验证"
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        helper = load_module(args.base_helper, "qwen_official_pooling_helper")
        provenance = json.loads((args.model_dir / "job_agent_provenance.json").read_text())
        if provenance["repo_id"] != helper.OFFICIAL_REPO or provenance["revision"] != helper.OFFICIAL_REVISION or file_sha(args.model_dir / "model.safetensors") != provenance["model_safetensors_sha256"]:
            raise ValueError("官方模型revision或权重hash不一致")
        encoder = LocalEncoder(args.model_dir, device=device, max_tokens=args.max_length, pooling="last_token")
        model, tokenizer = encoder.model, encoder.tokenizer
        model.config.use_cache = False
        def encode(texts, query):
            return encoder.encode_tensors(texts, query=query, trainable=torch.is_grad_enabled())
        manifest.update({"backend": "Qwen3-Embedding-0.6B LoRA", "model_source": provenance,
                         "encoder_contract": encoder.info,
                         "encoder_source_sha256": file_sha(Path(sys.modules[LocalEncoder.__module__].__file__)),
                         "official_provenance_helper_sha256": file_sha(args.base_helper), "dtype": str(next(model.parameters()).dtype)})
    evaluation_sets = ["dev", "test"] if args.evaluate_test else ["dev"]
    baseline = {split: evaluate(model, encode, queries, documents, groups, split, torch, query_views) for split in evaluation_sets}
    write_json(output / "baseline_metrics.json", baseline)
    if not args.fixture_encoder:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                   lora_dropout=.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        encoder.model = model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        manifest["lora"] = {"r": 16, "alpha": 32, "dropout": .05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=.01)
    manifest["trainable_parameters"] = sum(p.numel() for p in trainable)
    lookup = {(row["query_id"], row["job_id"]): row for row in labels}
    history, step = [], 0
    started = time.perf_counter()
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            rng = random.Random(args.seed + epoch)
            ordered = sorted(eligible)
            rng.shuffle(ordered)
            for offset in range(0, len(ordered), args.batch_size):
                query_ids = ordered[offset:offset + args.batch_size]
                document_ids = set()
                for query in query_ids:
                    candidates = eligible[query]
                    document_ids.update(rng.sample(candidates["positive"], min(args.positives_per_query, len(candidates["positive"]))))
                    document_ids.update(rng.sample(candidates["negative"], min(args.negatives_per_query, len(candidates["negative"]))))
                document_ids = sorted(document_ids)
                positive, negative, informative = reviewed_masks(query_ids, document_ids, lookup, torch, device)
                if not bool(informative.all()):
                    raise RuntimeError("已审核batch失去正例或难负例")
                model.train()
                optimizer.zero_grad(set_to_none=True)
                views = [text for key in query_ids for text in query_views[key]]
                view_counts = [len(query_views[key]) for key in query_ids]
                q = encode(views, True)
                d = encode([documents[key] for key in document_ids], False)
                scores = segment_scores(q, d, view_counts, torch)
                loss = multipositive_loss(q, d, positive, negative, args.temperature, torch, scores=scores)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("InfoNCE loss非有限值")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("梯度非有限值")
                optimizer.step()
                step += 1
                row = {"step": step, "epoch": epoch + 1, "loss": float(loss.detach()), "gradient_norm": float(norm),
                       "actual_queries": len(query_ids), "actual_documents": len(document_ids),
                       "actual_query_views": len(views),
                       "positive_pairs": int(positive.sum()), "explicit_negative_pairs": int(negative.sum()),
                       "unknown_or_nonnegative_pairs_masked": int((~(positive | negative)).sum()),
                       "seconds": time.perf_counter() - started}
                history.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if step >= args.max_steps:
                    break
            if step >= args.max_steps:
                break
    if not step:
        raise RuntimeError("没有执行任何训练步骤")
    trained_seconds = time.perf_counter() - started
    if args.fixture_encoder:
        torch.save(model.state_dict(), output / "fixture_projection.pt")
    else:
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "adapter")
    final = {split: evaluate(model, encode, queries, documents, groups, split, torch, query_views) for split in evaluation_sets}
    write_json(output / "adapted_metrics.json", final)
    if not args.fixture_encoder:
        import numpy as np
        # 验证服务采用的LocalEncoder能读取刚保存的adapter，且同输入输出向量相同。
        query_probe = [queries[key] for key in sorted(queries) if profiles[key]["split"] == "dev"][:2]
        document_probe = [documents[key] for key in sorted(documents) if jobs[key]["split"] == "dev"][:2]
        expected = {"query": encoder.encode(query_probe, query=True),
                    "document": encoder.encode(document_probe, query=False)}
        reloaded = LocalEncoder(args.model_dir, device=device, adapter=output / "adapter", max_tokens=args.max_length, pooling="last_token")
        observed = {"query": reloaded.encode(query_probe, query=True),
                    "document": reloaded.encode(document_probe, query=False)}
        reload_report = {"passed": all(np.allclose(expected[key], observed[key], atol=1e-5, rtol=1e-5) for key in expected),
                         "max_absolute_difference": max(float(np.max(np.abs(expected[key]-observed[key]))) for key in expected),
                         "query_probes": len(query_probe), "document_probes": len(document_probe),
                         "probe_input_hash": object_sha({"queries":query_probe,"documents":document_probe}),
                         "inference_encoder_contract": reloaded.info,
                         "scope": "保存后独立LocalEncoder重载；HTTP路由连通性需由服务集成测试另验"}
        write_json(output / "reload_verification.json", reload_report)
        if not reload_report["passed"]:
            raise RuntimeError("保存后服务编码器重载向量不一致，拒绝标记实验完成")
        manifest["reload_verification"] = reload_report
        del reloaded
    manifest.update({"status": "completed", "optimizer_steps": step, "training_seconds": trained_seconds,
                     "test_evaluated": args.evaluate_test, "fixture_only": label_contract["fixture_only"],
                     "model_weak_supervision": label_contract["model_weak_supervision"],
                     "candidate_contract_sha256": label_contract["candidate_contract_sha256"]})
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "output": str(output), "optimizer_steps": step,
                      "fixture_only": label_contract["fixture_only"], "model_weak_supervision": label_contract["model_weak_supervision"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
