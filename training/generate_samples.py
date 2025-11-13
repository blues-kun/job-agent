import json
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JOBS_FILE, RESUME_FILE
from data_preprocess import JobDataLoader
from resume_extract.storage import ResumeStorage
from models import ResumeProfile
from search.scorer import MatchScorer

def clone_resume(base: ResumeProfile, city: str, salary: int, intents: list[str]) -> ResumeProfile:
    d = base.model_dump()
    d['personal_info'] = d.get('personal_info') or {}
    d['personal_info']['current_city'] = city
    d['work_preferences'] = d.get('work_preferences') or {}
    d['work_preferences']['salary_expectation'] = d['work_preferences'].get('salary_expectation') or {}
    d['work_preferences']['salary_expectation']['min_annual_package'] = salary
    d['work_preferences']['position_type_name'] = intents
    return ResumeProfile.model_validate(d)

def main():
    random.seed(42)
    loader = JobDataLoader(JOBS_FILE)
    jobs = loader.to_dict_list()
    rs = ResumeStorage(RESUME_FILE)
    base = rs.get()
    cities = ['上海','深圳','北京','南京','杭州']
    salaries = [240000, 300000, 360000, 420000]
    ter_set = set()
    for j in jobs[:2000]:
        v = str(j.get('三级分类', '') or '')
        if v:
            ter_set.add(v)
    intents_pool = list(ter_set) or ['Java','Python','Golang']
    out = Path('logs'); out.mkdir(parents=True, exist_ok=True)
    fp = out / 'recommend_events.jsonl'
    w = fp.open('w', encoding='utf-8')
    total = 0
    for city in cities:
        for sal in salaries:
            intents = random.sample(intents_pool, k=min(3, len(intents_pool)))
            resume = clone_resume(base, city, sal, intents)
            scored = []
            for j in jobs:
                s, reasons = MatchScorer.calculate_score(resume, j)
                scored.append((s, j, reasons))
            scored.sort(key=lambda x: x[0], reverse=True)
            pos_k = 30
            neg_k = 90
            for s, j, reasons in scored[:pos_k]:
                rec = {'resume': resume.model_dump(), 'job': j, 'label': 1, 'score': s, 'reasons': reasons}
                w.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total += 1
            for s, j, reasons in scored[-neg_k:]:
                rec = {'resume': resume.model_dump(), 'job': j, 'label': 0, 'score': s, 'reasons': reasons}
                w.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total += 1
    w.close()
    print('样本已生成:', fp, '共', total, '条')

if __name__ == '__main__':
    main()

