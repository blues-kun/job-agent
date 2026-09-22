#!/usr/bin/env python3
"""岗位文本—图自监督研究原型；不训练人岗适配标签，不晋升线上排序器。

仅依赖 numpy、torch。岗位与技能必须由同一冻结文本模型编码。
dev/test 岗位只读取 train 岗位形成的技能状态，不能写回共享技能节点。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RELATIONS = ("required", "preferred", "alternative", "mentioned")
RELATION_SCALES = (1.0, 0.5, 0.75, 0.25)
SPLITS = ("train", "dev", "test")


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_seed(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "little")


@dataclass
class Edges:
    jobs: torch.Tensor
    skills: torch.Tensor
    relations: torch.Tensor
    weights: torch.Tensor

    def to(self, device: torch.device) -> "Edges":
        return Edges(*(getattr(self, name).to(device) for name in ("jobs", "skills", "relations", "weights")))

    def subset(self, mask: torch.Tensor) -> "Edges":
        return Edges(*(getattr(self, name)[mask] for name in ("jobs", "skills", "relations", "weights")))


def build_edges(jobs: list[dict], skill_names: list[str]) -> tuple[Edges, dict]:
    """将原文技能关联投影成软消息边；没有冒充完整AND/OR逻辑执行器。"""
    lookup = {name: i for i, name in enumerate(skill_names)}
    records: list[tuple[int, int, int, float]] = []
    omitted = Counter()
    group_count = Counter()
    for index, job in enumerate(jobs):
        evidence = job.get("skills") or {}
        selected: dict[str, tuple[int, float, int]] = {}
        for group in job.get("groups") or []:
            names = list(dict.fromkeys(group.get("skills") or group.get("options") or []))
            kind = str(group.get("kind", group.get("operator", "single"))).lower()
            group_count[kind] += 1
            preferred = bool(group.get("preferred")) or group.get("priority") == "preferred"
            is_any = kind in {"any", "or"}
            # 未明确必需的组保持 mentioned；不将现规则生成的单项组全部当作必需。
            relation = "preferred" if preferred else "alternative" if is_any else "required" if group.get("priority") == "required" else "mentioned"
            precedence = {"required": 4, "alternative": 3, "mentioned": 2, "preferred": 1}[relation]
            weight = 1.0 / max(len(names), 1) if is_any else 1.0
            for name in names:
                item = evidence.get(name, {})
                if item.get("level") == "否定" or item.get("polarity") == "negative":
                    continue
                if name not in selected or precedence > selected[name][2]:
                    selected[name] = (RELATIONS.index(relation), weight, precedence)
        for name, item in evidence.items():
            if item.get("level") == "否定" or item.get("polarity") == "negative":
                continue
            selected.setdefault(name, (RELATIONS.index("mentioned"), 1.0, 2))
        for name, (relation, weight, _) in selected.items():
            if name not in lookup:
                omitted["feature_vocab_missing"] += 1
                continue
            records.append((index, lookup[name], relation, weight))
    if records:
        columns = list(zip(*records))
        edges = Edges(torch.tensor(columns[0], dtype=torch.long), torch.tensor(columns[1], dtype=torch.long),
                      torch.tensor(columns[2], dtype=torch.long), torch.tensor(columns[3], dtype=torch.float32))
    else:
        edges = Edges(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                      torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32))
    return edges, {"edges": len(records), "relations": dict(Counter(RELATIONS[item[2]] for item in records)),
                   "groups": dict(group_count), "omitted": dict(omitted)}


def randomize_edges(edges: Edges, splits: list[str], seed: int) -> Edges:
    """分区内、同关系和同权重内交换技能端点，保留两端加权度数。"""
    result = Edges(edges.jobs.clone(), edges.skills.clone(), edges.relations.clone(), edges.weights.clone())
    buckets: dict[tuple[str, int, float], list[int]] = defaultdict(list)
    for i, (job, relation, weight) in enumerate(zip(edges.jobs.tolist(), edges.relations.tolist(), edges.weights.tolist())):
        buckets[(splits[job], relation, float(weight))].append(i)
    for key, positions in sorted(buckets.items()):
        # 每个桶独立随机种子；增加dev/test节点不会改变train图随机化。
        rng = np.random.default_rng(stable_seed(seed, repr(key)))
        source = result.skills[positions].clone()
        result.skills[positions] = source[torch.as_tensor(rng.permutation(len(positions)), dtype=torch.long)]
    return result


def weighted_mean(values: torch.Tensor, destinations: torch.Tensor, weights: torch.Tensor, count: int) -> torch.Tensor:
    result = values.new_zeros((count, values.shape[-1]))
    denominator = values.new_zeros((count, 1))
    if destinations.numel():
        result.index_add_(0, destinations, values * weights[:, None])
        denominator.index_add_(0, destinations, weights[:, None])
    return result / denominator.clamp_min(1e-12)


class RelationLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.job_self = nn.Linear(hidden, hidden)
        self.skill_self = nn.Linear(hidden, hidden)
        self.to_job = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in RELATIONS)
        self.to_skill = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in RELATIONS)
        self.job_norm = nn.LayerNorm(hidden)
        self.skill_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, jobs: torch.Tensor, skills: torch.Tensor, edges: Edges, train_mask: torch.Tensor):
        job_value, skill_value = self.job_self(jobs), self.skill_self(skills)
        for relation, scale in enumerate(RELATION_SCALES):
            selected = edges.relations == relation
            edge = edges.subset(selected)
            incoming = weighted_mean(skills[edge.skills], edge.jobs, edge.weights, len(jobs))
            job_value = job_value + scale * self.to_job[relation](incoming)
            # 唯一能更新共享技能状态的源节点是训练岗位。
            train_edges = edge.subset(train_mask[edge.jobs])
            incoming = weighted_mean(jobs[train_edges.jobs], train_edges.skills, train_edges.weights, len(skills))
            skill_value = skill_value + scale * self.to_skill[relation](incoming)
        return (self.dropout(F.gelu(self.job_norm(job_value))),
                self.dropout(F.gelu(self.skill_norm(skill_value))))


class GraphEncoder(nn.Module):
    def __init__(self, text_dimension: int, hidden: int = 128, mode: str = "graphsage", dropout: float = 0.2, beta: float = 0.3):
        super().__init__()
        if mode not in {"graphsage", "random_edges", "pool_mlp"}:
            raise ValueError("未知图实验模式")
        self.config = {"text_dimension": text_dimension, "hidden": hidden, "mode": mode, "dropout": dropout, "beta": beta}
        self.mode, self.beta = mode, beta
        self.job_projection = nn.Linear(text_dimension, hidden)
        self.skill_projection = nn.Linear(text_dimension, hidden)
        if mode == "pool_mlp":
            self.pool = nn.Sequential(nn.Linear(hidden * (len(RELATIONS) + 1), hidden), nn.LayerNorm(hidden),
                                      nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        else:
            self.layers = nn.ModuleList(RelationLayer(hidden, dropout) for _ in range(2))
        self.graph_projection = nn.Linear(hidden, hidden, bias=False)
        self.alignment = nn.Linear(hidden, text_dimension, bias=False)

    def forward(self, text_vectors: torch.Tensor, skill_vectors: torch.Tensor, edges: Edges, train_mask: torch.Tensor):
        initial = self.job_projection(text_vectors)
        jobs, skills = initial, self.skill_projection(skill_vectors)
        if self.mode == "pool_mlp":
            summaries = [initial]
            for relation, scale in enumerate(RELATION_SCALES):
                selected = edges.subset(edges.relations == relation)
                summaries.append(scale * weighted_mean(skills[selected.skills], selected.jobs, selected.weights, len(jobs)))
            jobs = self.pool(torch.cat(summaries, dim=-1))
        else:
            for layer in self.layers:
                jobs, skills = layer(jobs, skills, edges, train_mask)
        has_edges = torch.zeros(len(jobs), dtype=torch.bool, device=jobs.device)
        has_edges[edges.jobs] = True
        # 词典无覆盖的岗位只保留文本残差；空邻居不是全零可用向量。
        vectors = F.normalize(initial + self.beta * self.graph_projection(jobs) * has_edges[:, None], dim=-1)
        return {"graph_vectors": vectors, "aligned_vectors": F.normalize(self.alignment(vectors), dim=-1),
                "skill_graph_vectors": F.normalize(skills, dim=-1)}


def masked_logits(student: torch.Tensor, teachers: torch.Tensor, anchor_families: torch.Tensor,
                  candidate_families: torch.Tensor, targets: torch.Tensor, temperature: float):
    logits = student @ teachers.T / temperature
    allowed = anchor_families[:, None] != candidate_families[None, :]
    allowed[torch.arange(len(targets), device=targets.device), targets] = True
    return logits.masked_fill(~allowed, -torch.inf), allowed


def alignment_loss(student: torch.Tensor, teachers: torch.Tensor, anchor_families: torch.Tensor,
                   candidate_families: torch.Tensor, targets: torch.Tensor, temperature: float):
    logits, allowed = masked_logits(student, teachers, anchor_families, candidate_families, targets, temperature)
    valid = allowed.sum(dim=1) >= 2
    if not valid.any():
        return None, 0
    return F.cross_entropy(logits[valid], targets[valid]), int(valid.sum())


def load_inputs(dataset: Path, feature_file: Path):
    manifest_file = dataset / "manifest.json"
    manifest = json.loads(manifest_file.read_text("utf-8"))
    snapshots = [value for value in (manifest.get("snapshot"), manifest.get("source_sha256"),
                                    manifest.get("config", {}).get("snapshot")) if value]
    if not snapshots or len(set(snapshots)) != 1:
        raise ValueError("dataset manifest 必须提供一致的 snapshot/source_sha256")
    snapshot = snapshots[0]
    jobs_file = dataset / "jobs.jsonl"
    jobs_hash = file_hash(jobs_file)
    declared_jobs_hash = manifest.get("files", {}).get("jobs.jsonl", {}).get("sha256") or manifest.get("jobs_sha256")
    if declared_jobs_hash != jobs_hash:
        raise ValueError("jobs.jsonl 与 dataset manifest 声明的SHA-256不一致或缺失")
    jobs = [json.loads(line) for line in jobs_file.read_text("utf-8").splitlines() if line.strip()]
    if any(job.get("snapshot") != snapshot for job in jobs):
        raise ValueError("job.snapshot 与 dataset 源快照不一致或缺失")
    ids = [str(job["job_id"]) for job in jobs]
    if not jobs or len(ids) != len(set(ids)):
        raise ValueError("岗位ID缺失或重复；不能按legacy_id覆盖内容版本")
    if any(job.get("split") not in SPLITS or not job.get("job_family_id") for job in jobs):
        raise ValueError("每个岗位必须包含 train/dev/test split 和 job_family_id")
    family_splits: dict[str, set[str]] = defaultdict(set)
    for job in jobs:
        family_splits[job["job_family_id"]].add(job["split"])
    if any(len(value) > 1 for value in family_splits.values()):
        raise ValueError("发现岗位近重复家族跨分区；拒绝训练")
    feature_manifest_file = feature_file.with_suffix(".manifest.json")
    if not feature_manifest_file.exists():
        raise ValueError("features旁必须有同名.manifest.json，记录jobs_sha256和features_sha256")
    feature_manifest = json.loads(feature_manifest_file.read_text("utf-8"))
    if feature_manifest.get("jobs_sha256") != jobs_hash or feature_manifest.get("features_sha256") != file_hash(feature_file):
        raise ValueError("特征manifest与当前jobs/features文件SHA-256不一致")
    if any(feature_manifest[key] != snapshot for key in ("snapshot", "source_sha256") if feature_manifest.get(key)):
        raise ValueError("特征manifest的源快照与dataset不一致")
    with np.load(feature_file, allow_pickle=False) as archive:
        feature_ids = archive["job_ids"].astype(str).tolist()
        if len(feature_ids) != len(set(feature_ids)) or set(feature_ids) != set(ids):
            raise ValueError("features.npz 的 job_ids 必须与当前内容ID集合完全一致且唯一")
        order = {value: i for i, value in enumerate(feature_ids)}
        vectors = np.asarray(archive["text_vectors"], dtype=np.float32)[[order[value] for value in ids]]
        skill_names = archive["skills"].astype(str).tolist()
        skill_vectors = np.asarray(archive["skill_vectors"], dtype=np.float32)
    if vectors.ndim != 2 or skill_vectors.ndim != 2 or vectors.shape[0] != len(jobs) or skill_vectors.shape != (len(skill_names), vectors.shape[1]):
        raise ValueError("文本/技能向量维度不一致")
    for key, expected in (("jobs", len(jobs)), ("skills", len(skill_names)), ("dimension", vectors.shape[1])):
        if key in feature_manifest and feature_manifest[key] != expected:
            raise ValueError("特征manifest的数量/维度与NPZ不一致")
    if len(skill_names) != len(set(skill_names)) or not np.isfinite(vectors).all() or not np.isfinite(skill_vectors).all():
        raise ValueError("技能名重复或向量包含非有限值")
    if (np.linalg.norm(vectors, axis=1) <= 0).any() or (np.linalg.norm(skill_vectors, axis=1) <= 0).any():
        raise ValueError("不接受全零文本向量；缺边可退回文本，缺文本编码不可伪装有效")
    families = {value: i for i, value in enumerate(sorted(family_splits))}
    manifest = {**manifest, "snapshot": snapshot, "feature_manifest": feature_manifest,
                "feature_manifest_sha256": file_hash(feature_manifest_file)}
    return jobs, skill_names, F.normalize(torch.from_numpy(vectors), dim=-1), F.normalize(torch.from_numpy(skill_vectors), dim=-1), torch.tensor([families[j["job_family_id"]] for j in jobs]), manifest


@torch.no_grad()
def evaluate(model: GraphEncoder, vectors: torch.Tensor, skill_vectors: torch.Tensor, edges: Edges,
             train_mask: torch.Tensor, families: torch.Tensor, splits: list[str], limit: int, seed: int, temperature: float,
             include_test: bool = False):
    model.eval()
    output = model(vectors, skill_vectors, edges, train_mask)
    results = {}
    for split in SPLITS:
        if split == "test" and not include_test:
            continue
        candidates = torch.tensor([i for i, value in enumerate(splits) if value == split], dtype=torch.long, device=vectors.device)
        if not len(candidates):
            results[split] = {"status": "无此分区", "candidate_count": 0, "queries": 0}
            continue
        rng = np.random.default_rng(stable_seed(seed, "evaluation:" + split))
        positions = np.sort(rng.choice(len(candidates), size=min(limit, len(candidates)), replace=False))
        anchors = candidates[torch.as_tensor(positions, device=vectors.device)]
        targets = torch.as_tensor(positions, device=vectors.device)
        logits, allowed = masked_logits(output["aligned_vectors"][anchors], vectors[candidates], families[anchors], families[candidates], targets, temperature)
        valid = allowed.sum(dim=1) >= 2
        if not valid.any():
            results[split] = {"status": "没有独立负例，指标不适用", "candidate_count": len(candidates), "queries": 0}
            continue
        logits, targets, allowed = logits[valid], targets[valid], allowed[valid]
        positive = logits[torch.arange(len(targets), device=vectors.device), targets]
        # 并列时把并列负项排在正项之前，避免全相同向量被虚报满分。
        ranks = 1 + ((logits >= positive[:, None]) & allowed).sum(dim=1) - 1
        ranks = ranks.float()
        results[split] = {"status": "岗位自监督；非人岗相关性", "candidate_count": len(candidates), "queries": len(targets),
                          "loss": float(F.cross_entropy(logits, targets)),
                          "self_text_recall_at_1": float((ranks <= 1).float().mean()),
                          "self_text_recall_at_10": float((ranks <= 10).float().mean()),
                          "self_text_mrr_at_10": float(torch.where(ranks <= 10, 1 / ranks, 0).mean()),
                          "mean_eligible_candidates": float(allowed.sum(dim=1).float().mean())}
    return results, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("graphsage", "pool_mlp", "random_edges"), default="graphsage")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-queries", type=int, default=256)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="0为全部训练batch；短烟测可设2")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden, args.eval_queries, args.patience, args.cpu_threads) < 1 or args.temperature <= 0 or args.beta < 0 or not 0 <= args.dropout < 1 or args.max_steps_per_epoch < 0:
        parser.error("训练参数必须为有效范围")
    repo = args.repo_root.resolve()
    for path in (args.dataset, args.features, args.output):
        if path.resolve().is_relative_to(repo):
            parser.error("数据、特征与训练产物必须位于仓库外")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空；请使用新的实验目录避免覆盖结果")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("指定CUDA但当前环境不可用")
    if device.type == "cuda":
        # 在任何CUDA随机数或统计初始化前选择目标卡，避免默认cuda:0分配上下文。
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    jobs, skill_names, vectors, skill_vectors, families, data_manifest = load_inputs(args.dataset, args.features)
    splits = [job["split"] for job in jobs]
    edges, edge_stats = build_edges(jobs, skill_names)
    if args.mode == "random_edges":
        shuffled = randomize_edges(edges, splits, args.seed)
        edge_stats["random_endpoint_changed"] = int((edges.skills != shuffled.skills).sum())
        edges = shuffled
    edge_stats["by_split"] = dict(Counter(splits[index] for index in edges.jobs.tolist()))
    train_mask = torch.tensor([split == "train" for split in splits], dtype=torch.bool, device=device)
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    if len(set(families[train_mask.cpu()].tolist())) < 2:
        raise ValueError("训练集至少需要两个不同岗位家族")
    vectors, skill_vectors, families, edges = vectors.to(device), skill_vectors.to(device), families.to(device), edges.to(device)
    model = GraphEncoder(vectors.shape[1], args.hidden, args.mode, args.dropout, args.beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    before, _ = evaluate(model, vectors, skill_vectors, edges, train_mask, families, splits, args.eval_queries, args.seed, args.temperature)
    history, best_loss, best_epoch, best_weights, stale = [], float("inf"), None, None, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        permutation = np.random.default_rng(stable_seed(args.seed, f"train:{epoch}")).permutation(len(train_indices))
        total_loss, anchors_seen, steps = 0.0, 0, 0
        model.train()
        for offset in range(0, len(permutation), args.batch_size):
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
            positions = torch.tensor(permutation[offset:offset + args.batch_size], dtype=torch.long, device=device)
            anchors = train_indices[positions]
            output = model(vectors, skill_vectors, edges, train_mask)
            loss, count = alignment_loss(output["aligned_vectors"][anchors], vectors[train_indices], families[anchors], families[train_indices], positions, args.temperature)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("出现非有限训练损失；不发布该模型")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * count
            anchors_seen += count; steps += 1
        metrics, _ = evaluate(model, vectors, skill_vectors, edges, train_mask, families, splits, args.eval_queries, args.seed, args.temperature)
        # test指标不进入训练日志或模型选择；只在最终选定后一次性报告。
        selection_split = "dev" if "loss" in metrics["dev"] else "train"
        selection_loss = metrics[selection_split]["loss"]
        row = {"epoch": epoch, "train_loss": total_loss / anchors_seen if anchors_seen else None,
               "anchors": anchors_seen, "steps": steps, "seconds": round(time.perf_counter() - epoch_started, 3),
               "train": metrics["train"], "dev": metrics["dev"], "selection_split": selection_split}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if selection_loss < best_loss - 1e-6:
            best_loss, best_epoch = selection_loss, epoch
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_weights is None:
        raise ValueError("没有有效训练结果")
    model.load_state_dict(best_weights)
    final, output = evaluate(model, vectors, skill_vectors, edges, train_mask, families, splits, args.eval_queries, args.seed, args.temperature, include_test=True)
    dataset_hash = file_hash(args.dataset / "jobs.jsonl")
    feature_hash = file_hash(args.features)
    checkpoint = {"state_dict": best_weights, "config": model.config, "seed": args.seed, "best_epoch": best_epoch,
                  "dataset_sha256": dataset_hash, "features_sha256": feature_hash, "relations": list(RELATIONS)}
    torch.save(checkpoint, args.output / "model.pt")
    np.savez_compressed(args.output / "graph_vectors.npz", job_ids=np.asarray([job["job_id"] for job in jobs]),
                        graph_vectors=output["graph_vectors"].cpu().numpy(), aligned_vectors=output["aligned_vectors"].cpu().numpy(),
                        skills=np.asarray(skill_names), skill_graph_vectors=output["skill_graph_vectors"].cpu().numpy(),
                        splits=np.asarray(splits), job_family_ids=np.asarray([job["job_family_id"] for job in jobs]))
    manifest = {"status": "仅研究产物，禁止据此晋升人岗排序器", "task": "岗位文本—图自监督对齐", "mode": args.mode,
                "dataset_snapshot": data_manifest["snapshot"], "dataset_sha256": dataset_hash, "features_sha256": feature_hash,
                "feature_manifest_sha256": data_manifest["feature_manifest_sha256"], "feature_model": data_manifest["feature_manifest"].get("model"),
                "feature_template": data_manifest["feature_manifest"].get("template"),
                "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"), "script_sha256": file_hash(Path(__file__)),
                "model_config": model.config, "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "split_counts": dict(Counter(splits)), "family_count": len(set(families.cpu().tolist())), "edge_stats": edge_stats,
                "best_epoch": best_epoch, "selection_split": history[-1]["selection_split"], "parameter_count": sum(p.numel() for p in model.parameters()),
                "torch_version": torch.__version__, "numpy_version": np.__version__, "device": str(device),
                "seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "graph_contract": {"node_types": ["job", "skill"], "relations": list(RELATIONS),
                                   "degree_normalization": "每关系加权均值，另用固定关系尺度", "shared_state_sources": "仅train岗位",
                                   "or_handling": "将选项投影为单独alternative软消息边；未实现组节点传播或布尔满足逻辑",
                                   "unused_fields": ["category", "category_parent", "legacy_id", "edges.jsonl", "hypothesis关系"],
                                   "random_control": "分区/关系/相同边权内交换技能端点；保留加权度，可能生成平行边"},
                "limitations": ["输入已包含完整JD文本，文本对齐可能通过文本残差与自环完成；必须与pool_mlp/random_edges比较。",
                                "没有masked技能任务、没有人工qrels、没有真实用户监督；自监督Recall不能称为人岗Recall或推荐增益。",
                                "本原型未训练简历查询塔，图向量不能直接接入线上简历向量索引。",
                                "该脚本检查家族跨分区和内容ID对应；特征编码器及词典来源是否使用测试监督由上游manifest审计。",
                                "CUDA scatter归约可能存在浮点非确定性；固定种子不是逐bit一致性承诺。"]}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "metrics.json", {"task": manifest["task"], "not_person_job_evaluation": True, "before_training": before,
                                             "selected_checkpoint": final, "history": history, "selection_uses_test": False})
    print(json.dumps({"finished": True, "output": str(args.output), "best_epoch": best_epoch, "metrics": final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
