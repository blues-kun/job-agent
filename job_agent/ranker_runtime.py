"""学习排序器的影子加载与受证据约束的晋升；弱监督默认不替换规则排序。"""
import json
from pathlib import Path
import hashlib
import numpy as np
from .ranking import FEATURE_NAMES, FEATURE_VERSION
from .retrieval_contract import contract_info


class RankerRuntime:
    def __init__(self, directory: Path, corpus):
        import lightgbm as lgb
        metadata=json.loads((directory/"manifest.json").read_text())
        if metadata.get("feature_version")!=FEATURE_VERSION or metadata.get("feature_names")!=list(FEATURE_NAMES):
            raise ValueError("排序特征版本不一致")
        if metadata.get("snapshot") != corpus.snapshot or metadata.get("retrieval_contract_hash")!=contract_info()["hash"]:
            raise ValueError("排序器数据或文本约定不一致")
        if not metadata.get("shadow_load_allowed") or "synthetic_fixture" in metadata.get("label_sources",[]):
            raise ValueError("排序器不允许服务加载，fixture不可用于用户推荐")
        encoder=getattr(corpus.dense,"encoder_contract",{"model":"字符TF-IDF"})
        if metadata.get("encoder_contract")!=encoder:
            raise ValueError("排序器训练与服务编码器不同")
        if hashlib.sha256((directory/"model.txt").read_bytes()).hexdigest()!=metadata["model_sha256"]:
            raise ValueError("排序权重摘要不匹配")
        self.promoted=bool(metadata.get("eligible_for_production"))
        if self.promoted and (not metadata.get("promotion_evidence") or metadata.get("label_sources")!=["human_adjudicated"]):
            raise ValueError("缺少独立人工评测晋升依据")
        self.model=lgb.Booster(model_file=str(directory/"model.txt"))
        self.status="已验证学习排序" if self.promoted else "规则排序＋学习模型影子预测（未晋升）"

    def predict(self, vector):
        value=float(self.model.predict(np.asarray(vector).reshape(1,-1),num_threads=1)[0])
        if not np.isfinite(value):raise ValueError("排序器输出不合法")
        return value
