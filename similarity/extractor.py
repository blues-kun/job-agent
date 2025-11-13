"""
文本特征提取
"""
from typing import Dict, Any, List
from models import ResumeProfile
from .text import tokenize


def resume_tokens(resume: ResumeProfile) -> List[str]:
    parts = []
    try:
        for p in resume.work_preferences.position_type_name:
            parts.append(str(p))
    except Exception:
        pass
    try:
        parts.append(resume.full_resume_text or "")
    except Exception:
        pass
    return tokenize(" ".join(parts))


def job_tokens(job: Dict[str, Any]) -> List[str]:
    fields = [
        str(job.get('岗位名称', '') or ''),
        str(job.get('岗位要求', '') or ''),
        str(job.get('岗位职责', '') or ''),
        str(job.get('二级分类', '') or ''),
        str(job.get('三级分类', '') or ''),
    ]
    return tokenize(" ".join(fields))


def job_title_tokens(job: Dict[str, Any]) -> List[str]:
    return tokenize(str(job.get('岗位名称', '') or ''))

