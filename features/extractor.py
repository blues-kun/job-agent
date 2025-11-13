from typing import Dict, Any
from models import ResumeProfile
from similarity.engine import SimilarityEngine
from search.scorer import MatchConfig

def extract(resume: ResumeProfile, job: Dict[str, Any]) -> Dict[str, float]:
    e = SimilarityEngine()
    s, j, c = e.text_similarity(resume, job)
    title_hit, _ = e.title_intent_match(resume, job)
    city = resume.personal_info.current_city
    addr = str(job.get('岗位地址', '') or '')
    job_city = str(job.get('城市', '') or '')
    loc = 1.0 if (city and (city in addr or city in job_city)) else 0.0
    salary = str(job.get('岗位薪资', '') or '')
    negotiable = 1.0 if ('面议' in salary) else 0.0
    min_expect = float(resume.work_preferences.salary_expectation.min_annual_package or 0)
    from data_preprocess.cleaner import DataCleaner
    sal = DataCleaner.parse_salary_range(salary) if salary else {'min': None, 'max': None, 'months': 12}
    sal_max = float(sal.get('max') or 0)
    sal_ratio = (sal_max / min_expect) if (min_expect and sal_max) else 0.0
    edu_req = 0
    req = str(job.get('岗位要求', '') or '')
    if '博士' in req:
        edu_req = 4
    elif '硕士' in req:
        edu_req = 3
    elif '本科' in req:
        edu_req = 2
    elif '大专' in req:
        edu_req = 1
    edu_user = 0
    level = str(resume.professional_summary.education_level or '')
    if level == '博士':
        edu_user = 4
    elif level == '硕士':
        edu_user = 3
    elif level == '本科':
        edu_user = 2
    elif level == '大专':
        edu_user = 1
    edu_match = 1.0 if (edu_user >= edu_req) else 0.0
    import re
    m = re.search(r'(\d+)', req)
    req_years = float(m.group(1)) if m else 0.0
    user_years = float(resume.professional_summary.total_experience_years or 0.0)
    exp_ratio = (user_years / max(req_years, 1.0)) if user_years else 0.0
    return {
        'text_sim': float(s),
        'jaccard': float(j),
        'cosine_cnt': float(c),
        'title_intent': 1.0 if title_hit else 0.0,
        'location_match': float(loc),
        'salary_ratio': float(sal_ratio),
        'salary_negotiable': float(negotiable),
        'education_match': float(edu_match),
        'experience_ratio': float(exp_ratio),
    }

