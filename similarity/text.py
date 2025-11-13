"""
基础文本处理与相似度计算
"""
import re
from typing import List, Set, Tuple


def normalize(text: str) -> str:
    if not text:
        return ""
    return str(text).lower().strip()


def tokenize(text: str) -> List[str]:
    s = normalize(text)
    if not s:
        return []
    tokens = []
    for m in re.finditer(r"[\u4e00-\u9fa5]+|[a-zA-Z0-9_+.#]+", s):
        tokens.append(m.group(0))
    return tokens


def to_set(tokens: List[str]) -> Set[str]:
    return set(t for t in tokens if t)


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa = to_set(a)
    sb = to_set(b)
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return round(len(inter) / len(union), 6)


def cosine_count_similarity(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    freq_a = {}
    freq_b = {}
    for t in a:
        freq_a[t] = freq_a.get(t, 0) + 1
    for t in b:
        freq_b[t] = freq_b.get(t, 0) + 1
    vocab = set(freq_a.keys()).union(freq_b.keys())
    dot = 0.0
    na = 0.0
    nb = 0.0
    for v in vocab:
        va = freq_a.get(v, 0)
        vb = freq_b.get(v, 0)
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0.0 or nb == 0.0:
        return 0.0
    return round(dot / ((na ** 0.5) * (nb ** 0.5)), 6)

