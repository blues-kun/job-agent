#!/usr/bin/env python3
"""经人工仲裁数据的生成LoRA续接入口。默认只校验，不加载模型权重或训练。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any

import torch


TASKS = {"extract": "extract", "diagnose": "diagnose", "rewrite": "rewrite", "抽取": "extract", "诊断": "diagnose", "改写": "rewrite"}
SOURCE_KINDS = {"human_authored", "human_reviewed_synthetic", "authorized_real"}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path: Path, value: Any):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip() and value.lower() not in {"null", "none", "unknown", "todo", "待填", "待审核", "未审核"} and "<" not in value and ">" not in value


def families(value: Any, name: str) -> set[str]:
    if value is None or value == []:
        return set()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or any(not identifier(item) for item in values):
        raise ValueError(f"{name}须为真实家族标识或显式空列表")
    return set(values)


def review_payload_sha256(record: dict) -> str:
    """仲裁必须绑定被审内容、原文跨度及分区血缘，避免标签变更后复用旧记录。"""
    keys = ("sample_id", "split", "source_kind", "source_job_family", "profile_family", "messages", "label",
            "rewrite_mode", "source_text", "alignment_spans")
    payload = {key: record.get(key) for key in keys}
    payload["task"] = TASKS.get(record.get("task"), record.get("task"))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])|\n", text) if part.strip()]


def verify_preserving_rewrite(record: dict) -> dict:
    """只核验文本保全/跨度；句子重排也可能改变语义，绝不返回事实已证明。"""
    source, target = record.get("source_text"), record["label"]
    spans = record.get("alignment_spans")
    if not isinstance(source, str) or not source.strip() or not isinstance(spans, list) or not spans:
        raise ValueError("保全改写必须有source_text及非空alignment_spans，缺失不自动推断")
    if source not in record["messages"][-1]["content"]:
        raise ValueError("保全改写的source_text必须逐字出现在用户输入中")
    if Counter(sentences(source)) != Counter(sentences(target)):
        raise ValueError("保全改写的句子多重集不一致，不能通过整理制造内容")
    if re.search(r"前者|后者|上述|下述|该项目|它|其|随后|之后|接着|首先|其次|最后", source) and sentences(source) != sentences(target):
        raise ValueError("存在指代/顺序词的保全任务不得重排原句")
    source_covered, target_covered = set(), set()
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("对齐跨度必须是对象")
        values = [span.get(key) for key in ("source_start", "source_end", "target_start", "target_end")]
        if any(type(value) is not int for value in values):
            raise ValueError("对齐跨度必须为Unicode码点整数偏移")
        ss, se, ts, te = values
        if not (0 <= ss < se <= len(source) and 0 <= ts < te <= len(target)) or source[ss:se] != target[ts:te]:
            raise ValueError("对齐跨度越界或两侧原文不相同")
        source_chars = {i for i in range(ss, se) if not source[i].isspace()}
        target_chars = {i for i in range(ts, te) if not target[i].isspace()}
        if source_chars & source_covered or target_chars & target_covered:
            raise ValueError("保全对齐跨度不得重复覆盖正文字符")
        source_covered.update(source_chars); target_covered.update(target_chars)
    if source_covered != {i for i, char in enumerate(source) if not char.isspace()} or target_covered != {i for i, char in enumerate(target) if not char.isspace()}:
        raise ValueError("对齐跨度必须覆盖源文本与目标文本全部非空白字符")
    return {"sentence_multiset_equal": True, "alignment_complete": True, "semantic_facts_proven": False}


def validate_records(records: list[dict], *, allow_test_fixtures: bool = False) -> tuple[list[dict], dict]:
    checked, seen_ids, seen_family_splits, seen_prompt_splits = [], set(), {}, {}
    checks = Counter()
    for line_number, original in enumerate(records, 1):
        try:
            if not isinstance(original, dict):
                raise ValueError("JSONL每行必须是对象")
            record = dict(original)
            if record.get("fixture_only") and not allow_test_fixtures:
                raise ValueError("单测虚构审核记录不能进入正式CLI")
            if not identifier(record.get("sample_id")) or record["sample_id"] in seen_ids:
                raise ValueError("sample_id缺失、占位或重复")
            seen_ids.add(record["sample_id"])
            if record.get("split") not in {"train", "dev"} or record.get("task") not in TASKS:
                raise ValueError("仅接受train/dev和抽取/诊断/改写任务")
            record["task"] = TASKS[record["task"]]
            if record.get("review_status") != "human_adjudicated" or record.get("source_kind") not in SOURCE_KINDS:
                raise ValueError("只接受有人工仲裁记录的数据，synthetic_unreviewed不可训练")
            reviewers, adjudication = record.get("reviewer_ids"), record.get("adjudication")
            if not isinstance(reviewers, list) or not reviewers or any(not identifier(item) for item in reviewers) or len(reviewers) != len(set(reviewers)):
                raise ValueError("必须提供非空、去重的真人审核员标识")
            if not isinstance(adjudication, dict) or adjudication.get("decision") != "approved" or not identifier(adjudication.get("record_id")) or adjudication.get("reviewer_id") not in reviewers:
                raise ValueError("缺少可关联审核员的approved仲裁记录")
            reviewed_at = adjudication.get("reviewed_at")
            if not isinstance(reviewed_at, str):
                raise ValueError("缺少仲裁时间")
            try:
                parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("缺时区")
            except ValueError as exc:
                raise ValueError("仲裁时间必须为含时区ISO格式") from exc
            if not isinstance(record.get("label"), str) or not record["label"].strip() or record["label"].strip().lower() in {"null", "none"}:
                raise ValueError("label必须为审核后的非空assistant文本，null不能默认补标签")
            messages = record.get("messages")
            if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages) or [message.get("role") for message in messages] not in [["user"], ["system", "user"]]:
                raise ValueError("messages仅允许一个user，或system+user；assistant目标只来自label")
            if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
                raise ValueError("提示内容必须为非空文本")
            if "source_job_family" not in record or "profile_family" not in record:
                raise ValueError("必须显式声明岗位与画像家族，不适用时用空列表")
            namespaces = {"job": families(record["source_job_family"], "source_job_family"),
                          "profile": families(record["profile_family"], "profile_family")}
            if not any(namespaces.values()):
                raise ValueError("每条样本至少有岗位或画像家族，用于防泄漏")
            for namespace, values in namespaces.items():
                for value in values:
                    key = (namespace, value)
                    if seen_family_splits.setdefault(key, record["split"]) != record["split"]:
                        raise ValueError("岗位或画像家族跨train/dev分区")
            prompt_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if seen_prompt_splits.setdefault(prompt_hash, record["split"]) != record["split"]:
                raise ValueError("相同提示跨train/dev分区，即使家族标识不同也拒绝")
            structural = {"semantic_facts_proven": False}
            if record["task"] == "rewrite":
                if record.get("rewrite_mode") == "sentence_preserving":
                    structural = verify_preserving_rewrite(record)
                    checks["sentence_preserving_checked"] += 1
                elif record.get("rewrite_mode") != "human_reviewed_complex":
                    raise ValueError("改写须声明sentence_preserving或human_reviewed_complex")
            if record.get("programmatically_proven_facts") is True:
                raise ValueError("不接受将任意生成文本标为程序已证明事实")
            if adjudication.get("payload_sha256") != review_payload_sha256(record):
                raise ValueError("仲裁payload_sha256缺失或与标签/原文/分区血缘不一致；必须重新人工核对")
            record["structural_checks"] = structural
            checked.append(record)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"第{line_number}条未通过：{exc}") from exc
    counts = Counter(record["split"] for record in checked)
    if not counts["train"] or not counts["dev"]:
        raise ValueError("必须同时有独立且经过审核的train和dev样本")
    return checked, {"records": len(checked), "splits": dict(counts), "tasks": dict(Counter(row["task"] for row in checked)),
                     "family_count": len(seen_family_splits), "structural_checks": dict(checks),
                     "review_gate": "仅校验审核记录结构与关联，不能认证审核员身份或替代真实人工审核", "semantic_facts_proven": False}


def prepare_example(tokenizer, record: dict, max_length: int) -> dict:
    """原生generation mask与独立字符偏移双重校验，任何不确定边界都拒绝。"""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("需要fast tokenizer提供字符偏移；不猜测assistant边界")
    messages = record["messages"] + [{"role": "assistant", "content": record["label"]}]
    special_literals = [token for token in getattr(tokenizer, "all_special_tokens", []) if token]
    if any(token in message["content"] for message in messages for token in special_literals):
        raise ValueError("审核文本包含模型控制token字面值，必须先人工处理")
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                            return_dict=True, return_assistant_tokens_mask=True, truncation=False)
    ids = encoded["input_ids"]
    native_mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if native_mask is None or len(native_mask) != len(ids) or not any(native_mask):
        raise ValueError("chat template没有有效generation块；请提供经核对的assistant-mask模板")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
    if not rendered.startswith(prefix + record["label"]):
        raise ValueError("assistant训练模板与生成前缀不一致；拒绝猜测token边界")
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if offsets["input_ids"] != ids:
        raise ValueError("chat template与独立tokenize不一致，不能确认loss边界")
    start, end = len(prefix), len(prefix) + len(record["label"])
    expected_mask = []
    for left, right in offsets["offset_mapping"]:
        overlaps = right > start and left < end
        if overlaps and not (start <= left < right <= end):
            raise ValueError("token跨越提示与assistant边界，拒绝把提示计入loss")
        expected_mask.append(int(overlaps))
    if list(map(int, native_mask)) != expected_mask or not any(expected_mask):
        raise ValueError("原生assistant mask与字符跨度不一致，可能含提示或漏掉监督")
    if len(ids) > max_length:
        raise ValueError(f"完整样本长度{len(ids)}超过max_length={max_length}，不会截断或静默丢弃监督")
    labels = [token if active else -100 for token, active in zip(ids, expected_mask)]
    if not any(value != -100 for value in labels[1:]):
        raise ValueError("因果shift后没有assistant监督token")
    return {"input_ids": ids, "labels": labels, "assistant_tokens": sum(expected_mask), "split": record["split"], "task": record["task"]}


def collate(examples: list[dict], pad_id: int, device: torch.device):
    length = max(len(item["input_ids"]) for item in examples)
    ids, labels, attention = [], [], []
    for item in examples:
        padding = length - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention.append([1] * len(item["input_ids"]) + [0] * padding)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device), "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device)}


@torch.no_grad()
def evaluate(model, examples: list[dict], pad_id: int, device: torch.device, batch_size: int):
    model.eval(); total, tokens = 0.0, 0
    for offset in range(0, len(examples), batch_size):
        batch = collate(examples[offset:offset + batch_size], pad_id, device)
        count = int((batch["labels"][:, 1:] != -100).sum())
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise ValueError("开发loss非有限值；不发布适配器")
        total += float(loss) * count; tokens += count
    return {"assistant_token_loss": total / tokens, "assistant_tokens": tokens, "examples": len(examples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-template-file", type=Path)
    parser.add_argument("--mode", choices=("validate", "train"), default="validate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if min(args.max_length, args.max_steps, args.batch_size, args.gradient_accumulation, args.eval_every, args.lora_r, args.lora_alpha) < 1 or args.lr <= 0 or not 0 <= args.lora_dropout < 1:
        parser.error("训练参数超出有效范围")
    repo = args.repo_root.resolve()
    if args.data.resolve().is_relative_to(repo) or args.output.resolve().is_relative_to(repo):
        parser.error("审核数据与输出必须位于仓库外")
    if args.output.exists() or args.output.is_symlink():
        parser.error("输出路径已存在，拒绝复用或覆盖；校验与训练使用新的不同目录")
    raw = [json.loads(line) for line in args.data.read_text("utf-8").splitlines() if line.strip()]
    records, validation = validate_records(raw)
    # 所有审核/家族检查在模型及tokenizer加载之前执行。
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False, use_fast=True)
    if args.chat_template_file:
        tokenizer.chat_template = args.chat_template_file.read_text("utf-8")
    if not tokenizer.chat_template:
        raise ValueError("模型没有chat template；不得套用不经检查的提示模板")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer既没有pad也没有eos")
        tokenizer.pad_token = tokenizer.eos_token
    prepared = [prepare_example(tokenizer, record, args.max_length) for record in records]
    validation.update({"assistant_tokens": sum(item["assistant_tokens"] for item in prepared),
                       "max_full_sequence_tokens": max(len(item["input_ids"]) for item in prepared),
                       "prompt_tokens_supervised": 0, "truncated_examples": 0, "discarded_examples": 0})
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    save_json(args.output / "validation.json", validation)
    revision_file = args.model_dir / "revision.txt"
    config_file = args.model_dir / "config.json"
    manifest = {"status": "数据与loss边界已校验；不等于审核身份或输出事实已获认证", "mode": args.mode,
                "data_sha256": sha256(args.data), "script_sha256": sha256(Path(__file__)),
                "model_dir": str(args.model_dir.resolve()), "base_revision": revision_file.read_text().strip() if revision_file.exists() else "未提供revision文件",
                "base_config_sha256": sha256(config_file) if config_file.exists() else None,
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "validation": validation, "semantic_facts_proven": False,
                "limitations": ["审核门槛核对记录结构，不能认证记录所称的人类真实进行了审核。",
                                "句子多重集与跨度保全不证明上下文语义、归属或任意事实正确。",
                                "loss仅覆盖经审核assistant正文；模板角色头、提示、结束标记和padding不参与loss。",
                                "开发assistant-token loss不是事实性、推荐质量或简历改写效果的充分评测。"]}
    save_json(args.output / "manifest.json", manifest)
    if args.mode == "validate":
        print(json.dumps({"validated": True, "training_started": False, "summary": validation}, ensure_ascii=False))
        return
    # 仅显式--mode train且所有门槛通过，才加载权重与占用训练设备。
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA不可用")
        torch.cuda.set_device(device)
    random.seed(args.seed); torch.manual_seed(args.seed)
    dtype = torch.float32 if args.dtype == "float32" or (args.dtype == "auto" and device.type == "cpu") else torch.bfloat16
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("指定设备不支持BF16；请明确使用float32或更换设备")
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=dtype).to(device)
    targets = [value.strip() for value in args.lora_targets.split(",") if value.strip()]
    module_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not targets or any(target not in module_suffixes for target in targets):
        raise ValueError("LoRA目标层与模型实际模块名不一致")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                                          lora_dropout=args.lora_dropout, target_modules=targets, bias="none"))
    train, dev = [item for item in prepared if item["split"] == "train"], [item for item in prepared if item["split"] == "dev"]
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=0.01)
    baseline = evaluate(model, dev, tokenizer.pad_token_id, device, args.batch_size)
    best_loss, best_step = baseline["assistant_token_loss"], 0
    best_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
    history, cursor, order, rng, examples_seen = [], 0, list(range(len(train))), random.Random(args.seed), 0
    rng.shuffle(order)
    for step in range(1, args.max_steps + 1):
        groups = []
        for _ in range(args.gradient_accumulation):
            batch_items = []
            for _ in range(args.batch_size):
                if cursor == len(order):
                    rng.shuffle(order); cursor = 0
                batch_items.append(train[order[cursor]]); cursor += 1
            groups.append(batch_items)
        total_tokens = sum(sum(item["assistant_tokens"] for item in batch) for batch in groups)
        optimizer.zero_grad(set_to_none=True); model.train(); weighted_loss = 0.0
        for items in groups:
            batch = collate(items, tokenizer.pad_token_id, device)
            token_count = int((batch["labels"][:, 1:] != -100).sum())
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("出现非有限训练loss；停止且不保存未验证适配器")
            (loss * token_count / total_tokens).backward()
            weighted_loss += float(loss.detach()) * token_count / total_tokens
            examples_seen += len(items)
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        row = {"step": step, "train_assistant_token_loss": weighted_loss, "assistant_tokens": total_tokens, "examples_seen": examples_seen}
        if step % args.eval_every == 0 or step == args.max_steps:
            score = evaluate(model, dev, tokenizer.pad_token_id, device, args.batch_size)
            row["dev"] = score
            if score["assistant_token_loss"] < best_loss:
                best_loss, best_step = score["assistant_token_loss"], step
                best_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        save_json(args.output / "training_log.json", history)
    set_peft_model_state_dict(model, best_state)
    final = evaluate(model, dev, tokenizer.pad_token_id, device, args.batch_size)
    model.save_pretrained(args.output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "adapter")
    manifest.update({"status": "LoRA训练完成，仍需独立事实/任务评测后决定是否使用", "best_step": best_step,
                     "trained_steps": args.max_steps, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "seconds": round(time.perf_counter() - started, 3), "torch_version": torch.__version__})
    save_json(args.output / "manifest.json", manifest)
    save_json(args.output / "metrics.json", {"baseline_dev": baseline, "selected_dev": final, "best_step": best_step,
                                            "human_evaluation_completed": False, "semantic_facts_proven": False})


if __name__ == "__main__":
    main()
