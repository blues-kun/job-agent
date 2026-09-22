"""独立指标与按画像家族成对重采样；缺标、无相关项均不伪造。"""
import math
import numpy as np


def ranking_metrics(ranked, qrels, k=10):
    if type(k) is not int or k < 1:
        raise ValueError("K必须为正整数")
    if len(ranked) != len(set(ranked)):
        raise ValueError("排名包含重复岗位")
    if any(job not in qrels for job in ranked[:k]):
        raise ValueError("TopK有未标注岗位，请先补标；不能默认作为负例")
    if any(type(value) is not int or value not in range(4) for value in qrels.values()):
        raise ValueError("相关性等级必须0至3整数")
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(ideal))
    relevant = {job for job, grade in qrels.items() if grade >= 2}
    dcg = sum((2**qrels[job]-1)/math.log2(i+2) for i, job in enumerate(ranked[:k]))
    return {"ndcg": dcg/idcg if idcg else None,
            "pool_recall": len(relevant.intersection(ranked[:k]))/len(relevant) if relevant else None,
            "mrr": next((1/(i+1) for i, job in enumerate(ranked[:k]) if job in relevant), 0) if relevant else None,
            "judged_at_k": 1.0 if ranked[:k] else None, "returned": len(ranked[:k]), "k": k,
            "pool_size": len(qrels), "relevant_in_pool": len(relevant), "idcg": idcg}


def weighted_kappa(first, second, levels=4):
    if len(first) != len(second) or not first:
        return None
    if levels < 2:
        raise ValueError("等级数至少为2")
    observed = np.zeros((levels, levels), dtype=float)
    for a, b in zip(first, second):
        if type(a) is not int or type(b) is not int or a not in range(levels) or b not in range(levels):
            raise ValueError("标注等级超出范围")
        observed[a, b] += 1
    observed /= len(first)
    expected = np.outer(observed.sum(1), observed.sum(0))
    weights = np.fromfunction(lambda i, j: ((i-j)/(levels-1))**2, (levels, levels))
    denominator = (weights*expected).sum()
    return float(1-(weights*observed).sum()/denominator) if denominator else None


def paired_bootstrap(first, second, seed=42, iterations=2000, clusters=None):
    """有clusters时整组重采样；点估计仍为每查询均值，不混成家族宏平均。"""
    if len(first) != len(second) or not first:
        return None
    if iterations < 1:
        raise ValueError("重采样次数必须为正")
    delta = np.asarray(second, dtype=float)-np.asarray(first, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("成对指标必须为有限值")
    if clusters is None:
        clusters = list(range(len(delta)))
        unit = "query"
    else:
        if len(clusters) != len(delta) or any(not isinstance(value, str) or not value.strip() for value in clusters):
            raise ValueError("每个查询必须提供非空profile_family_id")
        unit = "profile_family"
    groups = {}
    for index, family in enumerate(clusters):
        groups.setdefault(family, []).append(index)
    indices = list(groups.values())
    result = {"mean_delta": float(delta.mean()), "ci95": None, "queries": len(delta),
              "clusters": len(groups), "resampling_unit": unit, "estimand": "query_weighted_mean_delta",
              "seed": seed, "iterations": iterations}
    if len(groups) < 2:
        result["note"] = "只有一个独立重采样单位，不能估计推广置信区间"
        return result
    sums = np.asarray([delta[group].sum() for group in indices])
    sizes = np.asarray([len(group) for group in indices])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(iterations, len(groups)))
    samples = sums[draws].sum(1)/sizes[draws].sum(1)
    result["ci95"] = np.quantile(samples, [.025, .975]).tolist()
    return result
