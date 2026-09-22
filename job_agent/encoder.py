"""本地离线与HTTP服务共用编码器，绑定实际权重、池化、截断和指令。"""
from pathlib import Path
import hashlib
import json

from .retrieval_contract import RESUME_INSTRUCTION, stable_hash


def aggregate_dense_scores(global_scores, best_project_scores=None):
    """对NumPy数组与可微Tensor均适用的统一画像/项目分数聚合。"""
    return global_scores if best_project_scores is None else .6 * global_scores + .4 * best_project_scores


def wrap_texts(texts, query, pooling, instruction=None):
    """训练、离线与HTTP使用同一个指令包装函数。"""
    if not query:
        return list(texts)
    instruction = instruction or (RESUME_INSTRUCTION if pooling == "last_token" else "为这个句子生成表示以用于检索相关文章：")
    return [f"Instruct: {instruction}\nQuery: {text}" if pooling == "last_token" else instruction + text for text in texts]


def pool_hidden(hidden, attention_mask, pooling):
    """最后有效token同时兼容左/右填充；保持可微，归一化统一为float32。"""
    import torch
    if pooling == "cls":
        values = hidden[:, 0, :]
    elif pooling == "last_token":
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
        indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        if bool((indices < 0).any()):
            raise ValueError("编码输入没有有效token")
        values = hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    else:
        raise ValueError("未知池化方法")
    return torch.nn.functional.normalize(values.float(), p=2, dim=1)


def weight_signature(directory):
    paths = sorted(Path(directory).glob("*.safetensors"))
    if not paths:
        raise ValueError("模型目录必须包含安全张量权重")
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    return stable_hash(result)


class LocalEncoder:
    def __init__(self, model_dir, device="cpu", adapter=None, max_tokens=512, pooling="auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        directory = Path(model_dir)
        self.device, self.max_tokens = device, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        config = json.loads((directory/"config.json").read_text())
        is_qwen = "qwen" in config.get("model_type", "").lower()
        self.pooling = ("last_token" if is_qwen else "cls") if pooling == "auto" else pooling
        if self.pooling not in {"last_token", "cls"}:
            raise ValueError("仅支持已验证的CLS或最后有效token池化")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True, dtype=dtype).to(device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, local_files_only=True).to(device)
        self.model.eval()
        revision_file = directory/"revision.txt"
        revision = revision_file.read_text().strip() if revision_file.exists() else "local-weights"
        self.info = {"model":config.get("_name_or_path") or config.get("model_type", "local"),
                     "revision":revision,"weights_sha256":weight_signature(directory),
                     "adapter_sha256":weight_signature(adapter) if adapter else None,
                     "pooling":self.pooling,"dimension":config["hidden_size"],"normalized":True,
                     "max_tokens":max_tokens,"padding_side":self.tokenizer.padding_side,
                     "query_instruction":RESUME_INSTRUCTION if self.pooling == "last_token" else "为这个句子生成表示以用于检索相关文章："}
        self.info["encoder_contract_hash"] = stable_hash(self.info)
        self.info["device"] = device

    def encode_tensors(self, texts, query=False, trainable=False):
        """trainable=True保留梯度；调用方显式设置train/eval，不修改模型状态。"""
        from contextlib import nullcontext
        import torch
        texts = wrap_texts(texts, query, self.pooling, self.info["query_instruction"])
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt").to(self.device)
        with nullcontext() if trainable else torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            return pool_hidden(hidden, encoded["attention_mask"], self.pooling)

    def encode(self, texts, query=False):
        return self.encode_tensors(texts, query=query).cpu().numpy()
