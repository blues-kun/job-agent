"""
相似度引擎
"""
from typing import Dict, Any, Tuple
from models import ResumeProfile
from .text import jaccard_similarity, cosine_count_similarity
from .extractor import resume_tokens, job_tokens, job_title_tokens


class SimilarityEngine:
    def text_similarity(self, resume: ResumeProfile, job: Dict[str, Any]) -> Tuple[float, float, float]:
        rt = resume_tokens(resume)
        jt = job_tokens(job)
        j = jaccard_similarity(rt, jt)
        c = cosine_count_similarity(rt, jt)
        s = round(0.5 * j + 0.5 * c, 6)
        return s, j, c

    def title_intent_match(self, resume: ResumeProfile, job: Dict[str, Any]) -> Tuple[bool, int]:
        title_ts = set(job_title_tokens(job))
        intent = set()
        try:
            for p in resume.work_preferences.position_type_name:
                for t in job_title_tokens({'岗位名称': p}):
                    intent.add(t)
        except Exception:
            pass
        inter = title_ts.intersection(intent)
        return (len(inter) > 0), len(inter)

