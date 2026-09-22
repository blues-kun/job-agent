"""训练与推理共用的特征合同；缺失用NaN，不把未知条件编码为已满足。"""
import math
import numpy as np

from .domain import constraints, supported_matches, group_support
from .corpus import tokens

FEATURE_NAMES = (
    "bm25_log", "dense_cosine", "graph_cosine", "rrf", "title_overlap", "category_overlap",
    "skill_jaccard", "evidence_coverage", "practice_ratio", "self_report_ratio", "weak_skill_ratio",
    "uncovered_group_ratio", "alternative_coverage", "preferred_gap_ratio", "education_delta",
    "experience_delta", "salary_headroom", "salary_iou", "city_match", "district_match",
    "hard_pass_ratio", "hard_unknown_ratio", "hard_fail_ratio", "job_skill_count", "profile_skill_count",
    "from_bm25", "from_dense", "from_graph", "from_structured", "salary_known", "experience_known", "education_known",
    "task_alignment", "requirement_support_ratio", "requirement_unknown_ratio",
)
FEATURE_VERSION = "evidence-ranker-v2-35"


def feature_vector(profile, job, retrieval=None):
    retrieval = retrieval or {}
    matched, gaps, coverage = supported_matches(profile, job)
    checks = constraints(profile, job)
    positive = {name for name, evidence in profile["skills"].items() if evidence["weight"] >= .7}
    demands = {name for name, evidence in job.skills.items() if evidence["level"] != "否定"}
    words = set(tokens(profile["intent"]))
    overlap = lambda text: sum(word in text.lower() for word in words) / max(len(words), 1)
    ratio = lambda predicate: sum(predicate(item) for item in matched) / max(len(matched), 1)
    support = group_support(profile, job)
    groups = max(len(job.groups), 1)
    alternatives = sum(group["kind"] == "any" for group in job.groups)
    monthly = job.salary.get("status") == "月薪"
    lower, upper = profile.get("salary_min"), profile.get("salary_max")
    missing = float("nan")
    salary_iou = missing
    if monthly and lower is not None and upper is not None and upper >= lower:
        low, high = job.salary["monthly_low"], job.salary["monthly_high"]
        salary_iou = max(0, min(upper,high)-max(lower,low)) / max(max(upper,high)-min(lower,low),1)
    values = [
        math.log1p(max(0,retrieval["bm25"])) if "bm25" in retrieval else missing,
        retrieval.get("dense",missing),retrieval.get("graph",missing),retrieval.get("rrf",missing),
        overlap(job.title),overlap(job.category),len(positive&demands)/max(len(positive|demands),1),coverage,
        ratio(lambda item:item["level"]=="实践"),ratio(lambda item:item["level"] in {"精通","熟悉"}),
        ratio(lambda item:item["strength"]<.7),len(gaps)/groups,
        sum(item["kind"]=="any" and item["strength"]>=.7 for item in matched)/alternatives if alternatives else missing,
        sum(item["preferred"] for item in gaps)/groups,
        profile["education"]-job.education_min if profile["education"] is not None and job.education_min is not None else missing,
        profile["experience_years"]-job.experience_min if profile["experience_years"] is not None and job.experience_min is not None else missing,
        (job.salary["monthly_high"]-lower)/max(lower,1) if monthly and lower is not None else missing,salary_iou,
        float(profile["city"]==job.city) if profile["city"] and job.city else missing,
        float(profile["district"]==job.district) if profile["district"] and job.district else missing,
        *[sum(check["status"]==status for check in checks)/max(len(checks),1) for status in ["pass","unknown","fail"]],
        len(demands),len(positive),*[float(name in retrieval.get("sources",[])) for name in ["bm25","dense","graph","structured"]],
        float(monthly),float(job.experience_min is not None),float(job.education_min is not None),
        retrieval.get("task_alignment",missing),
        sum(item["status"] == "pass" for item in support)/groups,
        sum(item["status"] == "unknown" for item in support)/groups,
    ]
    result = np.asarray(values,dtype=np.float32)
    if result.shape!=(len(FEATURE_NAMES),) or np.isinf(result).any():raise ValueError("排序特征不合法")
    return result
