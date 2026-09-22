"""真实预训练向量缓存与本地编码服务；模型失配时禁止混用索引。"""
from pathlib import Path
import hashlib
import json
import os
from urllib.parse import urlparse

import httpx
import numpy as np
from .retrieval_contract import contract_info
from .encoder import aggregate_dense_scores


class DenseIndex:
    def __init__(self, directory: Path, snapshot: str, ids: list[str], documents: list[str]):
        self.url = os.environ.get("JOB_AGENT_EMBEDDING_URL", "http://127.0.0.1:8091").rstrip("/")
        if urlparse(self.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("本版编码器只允许本机服务，避免把岗位或简历发送到未核实的外部端点。")
        self.client = httpx.Client(timeout=120, trust_env=False)
        response = self.client.get(self.url + "/health")
        response.raise_for_status()
        self.model = response.json()
        if not self.model.get("encoder_contract_hash"):
            raise ValueError("编码器缺少完整权重/模板约定，请使用新版编码服务")
        self.encoder_contract = {key:value for key,value in self.model.items() if key not in {"device", "status"}}
        signature = {"snapshot": snapshot, "ids": ids, "encoder_contract": self.encoder_contract,
                     "text_contract": contract_info(),
                     "document_hash": hashlib.sha256("\0".join(documents).encode()).hexdigest()}
        # 元组经JSON落盘变成列表；统一内存表示，否则正确缓存也会在每次启动时重编码。
        signature = json.loads(json.dumps(signature, ensure_ascii=False))
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        matrix_file, meta_file = directory / f"{key}.npy", directory / f"{key}.json"
        if matrix_file.exists() and meta_file.exists() and json.loads(meta_file.read_text()) == signature:
            self.matrix = np.load(matrix_file, allow_pickle=False)
        else:
            vectors = []
            for offset in range(0, len(documents), 64):
                vectors.extend(self.encode(documents[offset:offset+64], query=False))
                if offset % 2048 == 0:
                    print(f"向量索引：{min(offset+64,len(documents))}/{len(documents)}", flush=True)
            self.matrix = np.asarray(vectors, dtype=np.float32)
            # 完成后才发布缓存，避免中断时读取半个矩阵。
            temporary = matrix_file.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, self.matrix, allow_pickle=False)
            temporary.replace(matrix_file)
            meta_file.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
        if self.matrix.shape != (len(ids), self.model["dimension"]) or not np.isfinite(self.matrix).all():
            raise ValueError("向量缓存形状或数值不合法，不能用于检索。")
        norms = np.linalg.norm(self.matrix, axis=1)
        if not np.allclose(norms, 1, atol=.002):
            raise ValueError("向量没有正确归一化。")

    def encode(self, texts: list[str], query: bool) -> list:
        response = self.client.post(self.url + "/encode", json={"texts": texts, "query": query})
        response.raise_for_status()
        value = response.json()
        if value.get("encoder_contract_hash") != self.model["encoder_contract_hash"]:
            raise ValueError("查询模型版本与索引不一致，拒绝返回相似度。")
        return value["vectors"]

    def score(self, query: str) -> np.ndarray:
        vector = np.asarray(self.encode([query], query=True)[0], dtype=np.float32)
        return self.matrix @ vector

    def score_segments(self, queries: list[str]) -> np.ndarray:
        vectors = np.asarray(self.encode(queries, query=True), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.matrix.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("查询向量形状或数值不合法")
        scores = self.matrix @ vectors.T
        if len(queries) == 1:
            return scores[:, 0]
        return aggregate_dense_scores(scores[:, 0], np.max(scores[:, 1:], axis=1))
