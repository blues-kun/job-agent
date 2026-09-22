"""训练、回放与服务共用的文本约定；身份信息不作为推荐信号。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

TEMPLATE_VERSION = "person-job-evidence-v2"
RESUME_INSTRUCTION = "根据简历中的经历、技能与求职意向，检索职责和任职要求相符的招聘描述。"
QUERY_MODES = ("intent", "experience", "structured")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _get(value, key, default=""):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def experience_blocks(text):
    """按空行、经历标题保留完整块；不把组织/时间标题与其职责任意拆开。"""
    blocks, current = [], []
    heading = re.compile(r"^(?:项目(?:一|二|三|四|五|六|[0-9]+|名称)|工作经历|项目经历|教育经历|教育背景|任职经历|实习经历|20\d{2}[./年-])")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append("\n".join(current)); current = []
        else:
            # 紧跟项目/任职标题的日期属于同一经历块，不能把标题留在原位置。
            date_after_title = bool(current and re.match(r"^20\d{2}[./年-]",line)
                                    and heading.search(current[-1]) and not re.match(r"^20\d{2}",current[-1]))
            if current and heading.search(line) and not date_after_title:
                blocks.append("\n".join(current)); current = []
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def semantic_text(text):
    """仅剔除显式身份/联系方式行和指令行；保留任务、对象及结果。"""
    from .domain import CONTACT
    identity = re.compile(r"^(?:姓名|性别|年龄|出生|民族|婚姻|籍贯|身份证|手机|电话|邮箱|联系地址)\s*[:：]")
    injection = re.compile(r"忽略.{0,10}(?:指令|要求|规则)|(?:system|assistant)\s*:|系统提示词|保证录用|排名第一|给我满分", re.I)
    lines = [CONTACT.sub("", line).strip() for line in text.splitlines()
             if not identity.search(line.strip()) and not injection.search(line)]
    return "\n".join(line for line in lines if line)


def render_document(job):
    return "\n".join(str(value).strip() for value in [
        _get(job, "title"), _get(job, "category"), _get(job, "requirements"), _get(job, "description")
    ] if str(value).strip())


def _parts(profile):
    preferences = profile.get("preferences", profile)
    return semantic_text(profile.get("text", "")), semantic_text(preferences.get("intent", ""))


def render_query(profile, mode="structured"):
    if mode not in QUERY_MODES:
        raise ValueError("未知查询表示")
    text, intent = _parts(profile)
    if mode == "intent":
        return intent
    if mode == "experience":
        return text
    return f"求职方向：{intent}\n实际经历：\n{text}".strip()


def query_segments(profile, mode="structured"):
    """完整画像加完整经历块；长块仍由同一编码器截断，并记录截断配置。"""
    query = render_query(profile, mode)
    if mode == "intent":
        return [query]
    _, intent = _parts(profile)
    blocks = [semantic_text(block) for block in experience_blocks(profile.get("text", ""))]
    blocks = sorted(set(block for block in blocks if len(block) >= 15))
    # 每个块都是独立编码输入，避免只取简历开头；最多8块以限制交互时延。
    return list(dict.fromkeys([query] + [f"求职方向：{intent}\n实际经历：\n{block}" for block in blocks[:8]]))


def contract_info():
    from .domain import PARSER_VERSION
    root = Path(__file__).parent
    payload = {"template_version": TEMPLATE_VERSION, "parser_version": PARSER_VERSION,
               "query_modes": QUERY_MODES, "query_chunk_limit": 8,
               "dense_aggregation": "0.6*global+0.4*best_project",
               "instruction": RESUME_INSTRUCTION,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "semantics_sha256": hashlib.sha256((root / "semantics.py").read_bytes()).hexdigest(),
               "parser_sha256": hashlib.sha256((root / "domain.py").read_bytes()).hexdigest()}
    payload["hash"] = stable_hash(payload)
    return payload
