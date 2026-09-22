"""比较确定性逻辑、无图池化、真实组图与度数保持随机图；无人工标签时拒绝正式训练。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from research.group_graph import (GRAPH_VERSION, FrozenTextFeatures, GroupGraphRanker, build_graph,
    canonical_ast, degree_preserving_randomization, file_sha, logic_baseline, object_sha,
    private_output, read_jsonl, validate_supervision, write_json)


def metrics_for_group(ranked, grades):
    from research.metrics import ranking_metrics
    scores = ranking_metrics(ranked, grades, k=10)
    return {"ndcg_at_10": scores["ndcg"], "mrr_at_10": scores["mrr"],
            "pool_recall_at_10": scores["pool_recall"]}


def mean_metric(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


def evaluate(model, graphs, profiles, groups, split, jobs):
    rows = []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        vectors = {}
        for (part, query_id), labels in sorted(groups.items()):
            if part != split:
                continue
            candidate_ids = sorted(row["job_id"] for row in labels)
            if model is None:
                scores = [logic_baseline(jobs[job_id]["requirement_ast"], profiles[query_id])["supported_lower"] for job_id in candidate_ids]
            else:
                for job_id in candidate_ids:
                    if job_id not in vectors:
                        vectors[job_id] = model.encode_job(graphs[job_id])
                query_vector = model.encode_profile(profiles[query_id])
                scores = model.score(query_vector, torch.stack([vectors[job_id] for job_id in candidate_ids])).tolist()
            ranked = [candidate_ids[index] for index in sorted(range(len(candidate_ids)), key=lambda index: (-scores[index], candidate_ids[index]))]
            grades = {row["job_id"]: row["grade"] for row in labels}
            rows.append({"query_id": query_id, "candidate_ids": candidate_ids, "ranked_ids": ranked,
                         "candidate_scores": scores, **metrics_for_group(ranked, grades)})
    return {"split": split, "queries": len(rows),
            **{key: mean_metric(rows, key) for key in ("ndcg_at_10", "mrr_at_10", "pool_recall_at_10")},
            "scope": "固定已审核候选池；fixture运行的指标仅核验工程，不是人岗实证", "per_query": rows}


def train_variant(mode, seed, jobs, profiles, groups, features, args, output):
    random.seed(seed)
    torch.manual_seed(seed)
    graphs = {key: build_graph(job) for key, job in jobs.items()}
    randomization = {"eligible_graphs": 0, "changed_graphs": 0, "successful_edge_swaps": 0}
    if mode == "random_graph":
        for key, graph in graphs.items():
            randomized = degree_preserving_randomization(graph, seed)
            randomization["eligible_graphs"] += randomized.successful_swaps > 0
            randomization["changed_graphs"] += set(randomized.edges) != set(graph.edges)
            randomization["successful_edge_swaps"] += randomized.successful_swaps
            graphs[key] = randomized
    model = GroupGraphRanker(features, hidden=args.hidden, mode="pool" if mode == "pool" else "graph")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    history, best, best_epoch, best_state = [], -math.inf, 0, None
    training_groups = [group for (split, _), group in sorted(groups.items()) if split == "train"]
    train_job_ids = sorted({row["job_id"] for group in training_groups for row in group})
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # 共享的是训练梯度图；不预计算或写回任何dev/test节点状态。
        job_vectors = {key: model.encode_job(graphs[key]) for key in train_job_ids}
        losses = []
        for labels in training_groups:
            profile = profiles[labels[0]["query_id"]]
            scores = model.score(model.encode_profile(profile), torch.stack([job_vectors[row["job_id"]] for row in labels]))
            pairs = [(i, j) for i in range(len(labels)) for j in range(len(labels)) if labels[i]["grade"] > labels[j]["grade"]]
            losses.append(torch.stack([F.softplus(-(scores[i] - scores[j])) for i, j in pairs]).mean())
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("训练loss非有限值")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("训练梯度非有限值")
        optimizer.step()
        state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        dev = evaluate(model, graphs, profiles, groups, "dev", jobs)
        if any(not torch.equal(state_before[key], value) for key, value in model.state_dict().items()):
            raise RuntimeError("评估改变了共享训练参数或buffer")
        score = dev["ndcg_at_10"]
        if score is None:
            raise ValueError("开发池没有正向已审核等级，无法选择检查点")
        history.append({"epoch": epoch, "loss": float(loss.detach()), "gradient_norm": float(norm), "dev_ndcg_at_10": score})
        if score > best + 1e-12:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    path = output / f"{mode}-seed{seed}"
    path.mkdir()
    torch.save(model.state_dict(), path / "model.pt")
    reports = {split: evaluate(model, graphs, profiles, groups, split, jobs)
               for split in (["dev", "test"] if args.evaluate_test else ["dev"])}
    info = {"mode": mode, "seed": seed, "best_epoch": best_epoch, "executed_epochs": len(history),
            "seconds": time.perf_counter() - started, "parameter_count": sum(p.numel() for p in model.parameters()),
            "history": history, "evaluation": reports, "randomization": randomization,
            "state_unchanged_by_evaluation": True, "candidate_pool_hash": object_sha(sorted((part, query, sorted(row["job_id"] for row in group)) for (part, query), group in groups.items()))}
    write_json(path / "metrics.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("jobs", "profiles", "qrels", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--text-vectors", type=Path)
    parser.add_argument("--seeds", default="17,42,73")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--allow-model-labels", action="store_true", help="允许有完整来源的llm_reviewed弱标签，不是人工金标")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.patience) < 1 or args.learning_rate <= 0:
        raise ValueError("训练参数必须为正")
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seed列表为空或重复")
    output = private_output(args.output)
    jobs, profiles, groups, label_contract = validate_supervision(read_jsonl(args.jobs), read_jsonl(args.profiles), read_jsonl(args.qrels), args.allow_fixture, allow_model_labels=args.allow_model_labels)
    for row in jobs.values():
        canonical_ast(row["requirement_ast"])
    features = FrozenTextFeatures(external=json.loads(args.text_vectors.read_text()) if args.text_vectors else None)
    manifest = {"graph_version": GRAPH_VERSION, "label_contract": label_contract,
                "input_hashes": {name: file_sha(getattr(args, name)) for name in ("jobs", "profiles", "qrels")},
                "script_hashes": {"trainer": file_sha(__file__), "group_graph": file_sha(Path(__file__).with_name("group_graph.py"))},
                "document_version_hash": object_sha(sorted((key, job.get("job_version_id"), object_sha(canonical_ast(job["requirement_ast"]))) for key, job in jobs.items())),
                "text_feature_contract": features.contract, "text_vectors_sha256": file_sha(args.text_vectors) if args.text_vectors else None,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "selection": "仅dev选择epoch；每模型每seed相同训练预算；test在最终检查点上一次评估",
                "graph_control": "同类型叶子到组的度数保持端点交换，不随机改标签；changed_graphs=0时该随机对照无判别力",
                "not_person_job_evidence_when_fixture": label_contract["fixture_only"],
                "no_transductive_updates": True, "no_full_jd_self_retrieval": True,
                "limitations": ["hash初始化不是预训练语义；外部冻结向量需明确模型和模板",
                                "逻辑pass只表示文本已有自述证据；不证明实际技能或录用可能",
                                "固定已审核候选池指标不能代替全库召回或线上指标"]}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    if args.validate_only:
        print(json.dumps(label_contract, ensure_ascii=False))
        return
    torch.set_num_threads(1)
    baseline = {split: evaluate(None, {}, profiles, groups, split, jobs)
                for split in (["dev", "test"] if args.evaluate_test else ["dev"])}
    write_json(output / "logic_baseline.json", baseline)
    results = []
    for seed in seeds:
        for mode in ("pool", "graph", "random_graph"):
            result = train_variant(mode, seed, jobs, profiles, groups, features, args, output)
            results.append(result)
            print(json.dumps({"mode": mode, "seed": seed, "dev_ndcg": result["evaluation"]["dev"]["ndcg_at_10"], "fixture_only": label_contract["fixture_only"]}, ensure_ascii=False), flush=True)
    write_json(output / "summary.json", {"status": "completed", "fixture_only": label_contract["fixture_only"],
                "model_weak_supervision": label_contract["model_weak_supervision"], "eligible_for_production": False,
                "logic_baseline": baseline, "variants": results, "test_evaluated": args.evaluate_test})


if __name__ == "__main__":
    main()
