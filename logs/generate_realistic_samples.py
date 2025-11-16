"""
使用真实职位数据和LLM生成高质量训练样本
目标：AUC 0.85-0.90，准确率 85-90%
"""
import json
import random
import csv
import sys
from pathlib import Path
import argparse
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import API_KEY, BASE_URL, MODEL

random.seed(42)

# 城市列表
CITIES = ['北京', '上海', '深圳', '南京']

# 学历列表
EDUCATION_LEVELS = ['大专', '本科', '硕士', '博士']

# 经验年限
EXPERIENCE_RANGES = [0, 1, 2, 3, 5, 8, 10]

def load_real_jobs():
    """从向量化CSV加载真实职位数据（仅保留深圳职位，提取基础字段）"""
    # 使用脚本所在目录的绝对路径
    script_dir = Path(__file__).parent
    # 使用向量化后的数据
    jobs_file = script_dir.parent / 'data' / 'job_data_vectorized.csv'
    jobs = []
    
    print(f"读取向量化职位文件: {jobs_file}")
    with jobs_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只提取生成交互数据需要的基础字段（不包含向量特征）
            name = row.get('岗位名称') or row.get('职位名称') or ''
            company = row.get('企业') or row.get('公司') or ''
            salary = row.get('岗位薪资') or row.get('薪资') or ''
            req = row.get('岗位要求') or ''
            resp = row.get('岗位职责') or ''
            addr = row.get('岗位地址') or ''
            sec = row.get('二级分类') or ''
            ter = row.get('三级分类') or row.get('职位类型名称') or ''
            city = row.get('城市') or ''  # 向量化数据已包含城市字段
            
            # 如果没有城市信息，从地址推断
            if not city:
                for c in CITIES:
                    if c and addr and c in addr:
                        city = c
                        break
                if not city:
                    city = '深圳'
            
            job_obj = {
                '岗位名称': name,
                '企业': company,
                '岗位薪资': salary,
                '岗位要求': req,
                '岗位职责': resp,
                '岗位地址': addr,
                '二级分类': sec,
                '三级分类': ter,
                '城市': city,
            }
            if job_obj['城市'] == '深圳':
                jobs.append(job_obj)
    
    print(f"加载了 {len(jobs)} 条深圳职位数据（来自向量化表）")
    return jobs

def load_positions():
    """加载职位列表（包含分组标题和条目）"""
    script_dir = Path(__file__).parent
    positions_file = script_dir.parent / 'data' / 'position_dictionary.txt'
    positions = []
    if not positions_file.exists():
        print(f"[WARN] 职位词典不存在: {positions_file}")
        return positions
    import re
    for line in positions_file.read_text('utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('[') and s.endswith(']'):
            header = s[1:-1].strip()
            if header and header not in positions:
                positions.append(header)
            continue
        s = re.sub(r'^\s*\d+\s*→\s*', '', s)
        if s and s not in positions:
            positions.append(s)
    print(f"加载了 {len(positions)} 个职位类型/分组")
    return positions

def generate_resume_with_llm(llm, positions):
    """使用LLM随机生成互联网/AI相关简历文本与画像结构"""
    import random
    city = '深圳'
    num_positions = random.randint(1, 3)
    desired_positions = random.sample(positions, num_positions)
    relocate = random.choice([True, False])
    experience = random.choice(EXPERIENCE_RANGES)
    education = random.choice(EDUCATION_LEVELS)
    base_salary = 180000
    if education == '硕士':
        base_salary = 280000
    elif education == '博士':
        base_salary = 400000
    salary_expectation = base_salary + experience * 30000

    resume = {
        'personal_info': {
            'current_city': city,
            'willingness_to_relocate': relocate,
            'availability_date': '2025-12-01'
        },
        'work_preferences': {
            'position_type_name': desired_positions,
            'salary_expectation': {
                'min_annual_package': salary_expectation,
                'currency': 'CNY'
            }
        },
        'professional_summary': {
            'total_experience_years': float(experience),
            'education_level': education,
            'school_level': 1 if education in ['硕士','博士'] else 0
        }
    }

    if llm is None:
        resume['full_resume_text'] = f"互联网/AI方向简历，期望职位：{', '.join(desired_positions)}，{experience}年经验，{education}学历。参与算法/后端/数据相关项目，具备Python/Java/SQL/深度学习基础。"
        return resume

    prompt = (
        "你是一名资深HR，请用中文生成一份真实感的互联网/AI方向简历片段，"
        "包括求职者过往项目（2-3条）、核心技能（6-8项）、技术栈（编程语言/框架）、"
        "求职偏好（职位/城市/薪资期望），不要出现私人信息（姓名/电话/邮箱）。"
        f"\n职位偏好：{', '.join(desired_positions)}；城市：{city}；工作经验：{experience}年；学历：{education}；期望年薪：{int(salary_expectation/10000)}万。"
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        resume['full_resume_text'] = resp.content.strip()
    except Exception:
        resume['full_resume_text'] = f"互联网/AI方向简历，期望职位：{', '.join(desired_positions)}，{experience}年经验，{education}学历。"
    return resume

def create_diverse_resumes(positions, num_resumes=20, llm=None):
    """创建多样化的求职者简历，仅生成深圳简历，支持LLM生成文本"""
    import random
    resumes = []
    print(f"生成 {num_resumes} 份深圳简历")
    city_counts = {'深圳': 0}
    for i in range(num_resumes):
        city_counts['深圳'] += 1
        resume = generate_resume_with_llm(llm, positions)
        resumes.append(resume)
    print(f"创建了 {len(resumes)} 个多样化简历")
    print(f"城市分布: {city_counts}")
    return resumes

def analyze_match_with_llm(resume, job, llm):
    """使用LLM分析职位和简历的匹配度"""
    
    prompt = f"""你是AI职位匹配专家。分析求职者是否会对这个职位感兴趣。

求职者信息：
- 期望职位：{', '.join(resume['work_preferences']['position_type_name'])}
- 当前城市：{resume['personal_info']['current_city']}
- 愿意异地：{'是' if resume['personal_info']['willingness_to_relocate'] else '否'}
- 期望年薪：{resume['work_preferences']['salary_expectation']['min_annual_package']/10000:.1f}万
- 工作经验：{resume['professional_summary']['total_experience_years']}年
- 学历：{resume['professional_summary']['education_level']}

职位信息：
- 职位：{job['岗位名称']}
- 公司：{job['企业']}
- 城市：{job['城市']}
- 薪资：{job['岗位薪资']}
- 要求：{job['岗位要求']}

判断规则（从用户视角）：
1. **职位类型匹配**（最重要，权重60%）：
   - 完全匹配期望职位 → 强烈兴趣
   - 相关职位 → 有一定兴趣
   - 完全不相关 → 基本不感兴趣

2. **地理位置**（权重25%）：
   - 同城 → 很好
   - 异地但愿意异地 → 可以考虑
   - 异地且不愿异地 → 明显减分

3. **薪资待遇**（权重15%）：
   - ≥期望薪资 → 加分
   - <期望但>80% → 可接受
   - <80%期望 → 减分

**决策概率**：
- 职位类型完全匹配 + 同城 → 75%概率interested=true
- 职位类型完全匹配 + 异地（愿意异地）→ 50%概率interested=true  
- 职位类型完全匹配 + 异地（不愿意）→ 20%概率interested=true
- 职位类型不匹配 + 其他条件好 → 10%概率interested=true
- 职位类型不匹配 + 其他条件差 → 3%概率interested=true

严格输出JSON：
{{"interested": true/false, "reason": "简短理由"}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # 提取JSON
        content = response.content.strip()
        # 移除可能的markdown标记
        if content.startswith('```'):
            content = content.split('\n', 1)[1]
            content = content.rsplit('```', 1)[0]
        result = json.loads(content.strip())
        return result
    except Exception as e:
        print(f"  LLM分析失败: {e}")
        return None

def generate_samples():
    """生成训练样本"""
    
    print("\n=== 使用真实数据生成训练样本 ===\n")
    
    llm = None
    if API_KEY:
        try:
            print(f"初始化LLM: {MODEL}\n")
            llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.5)
        except Exception as e:
            print(f"[WARN] LLM初始化失败，将使用模板生成: {e}")
    
    # 加载数据
    jobs = load_real_jobs()
    positions = load_positions()
    resumes = create_diverse_resumes(positions, num_resumes=20, llm=llm)
    
    # 随机选择职位样本
    # 策略：每个简历匹配100-150个职位
    num_samples_per_resume = 160
    total_samples = len(resumes) * num_samples_per_resume
    
    print(f"\n目标生成 {total_samples} 个样本")
    print(f"每个简历匹配 {num_samples_per_resume} 个职位\n")
    
    samples = []
    like_count = 0
    skip_count = 0
    llm_analyzed = 0
    
    for resume_idx, resume in enumerate(resumes, 1):
        print(f"\n处理简历 {resume_idx}/{len(resumes)}")
        print(f"  期望职位: {', '.join(resume['work_preferences']['position_type_name'])}")
        print(f"  城市: {resume['personal_info']['current_city']}")
        
        # === 分层抽样：确保正负样本比例接近50/50 ===
        desired_positions = resume['work_preferences']['position_type_name']
        resume_city = resume['personal_info']['current_city']
        
        # 分类职位：匹配 vs 不匹配
        matched_jobs = []  # 职位类型匹配
        city_matched_jobs = []  # 职位类型+城市都匹配
        unmatched_jobs = []  # 职位类型不匹配
        
        for job in jobs:
            job_type = job.get('三级分类', '')
            job_city = job.get('城市', '')
            
            if job_type in desired_positions:
                if job_city == resume_city:
                    city_matched_jobs.append(job)
                else:
                    matched_jobs.append(job)
            else:
                unmatched_jobs.append(job)
        
        # 分层采样：60个高匹配，50个中匹配，50个低匹配
        # 后续通过平衡器将总体比例拉近50/50
        selected_jobs = []
        
        # 1. 高匹配（职位+城市都匹配）：60个 → 约70-80% like
        if city_matched_jobs:
            selected_jobs.extend(random.sample(city_matched_jobs, min(60, len(city_matched_jobs))))
        
        # 2. 中匹配（仅职位匹配）：50个 → 约30-40% like
        if matched_jobs:
            selected_jobs.extend(random.sample(matched_jobs, min(50, len(matched_jobs))))
        
        # 3. 低匹配（职位不匹配）：50个 → 约5-10% like
        if unmatched_jobs:
            selected_jobs.extend(random.sample(unmatched_jobs, min(50, len(unmatched_jobs))))
        
        # 如果数量不足160个，补充随机职位
        if len(selected_jobs) < num_samples_per_resume:
            remaining = num_samples_per_resume - len(selected_jobs)
            other_jobs = [j for j in jobs if j not in selected_jobs]
            if other_jobs:
                selected_jobs.extend(random.sample(other_jobs, min(remaining, len(other_jobs))))
        
        random.shuffle(selected_jobs)  # 打乱顺序
        print(f"  职位分层: 高匹配{min(60, len(city_matched_jobs))} + 中匹配{min(50, len(matched_jobs))} + 低匹配{min(50, len(unmatched_jobs))}")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        def process_one(job):
            try:
                result = analyze_match_with_llm(resume, job, llm) if llm else None
                if result is None:
                    job_position = job.get('三级分类') or job.get('职位类型名称') or ''
                    interested = job_position in resume['work_preferences']['position_type_name']
                    if random.random() < 0.2:
                        interested = not interested
                else:
                    interested = result.get('interested', False)
                action = 'like' if interested else 'skip'
                return {
                    'action': action,
                    'ts': '2025-11-11T00:00:00Z',
                    'type': 'cold_start',
                    'job': job,
                    'resume': resume
                }, (1 if action == 'like' else 0)
            except Exception as e:
                print(f"  错误: {e}")
                return None, 0

        workers = max(4, min(len(selected_jobs), (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one, job) for job in selected_jobs]
            for fut in as_completed(futures):
                res, like_inc = fut.result()
                if res:
                    samples.append(res)
                    like_count += like_inc
                    skip_count += (1 - like_inc)
    
    # 打乱数据
    random.shuffle(samples)

    # 先做总体平衡到约50/50
    total = len(samples)
    if total > 0:
        current_likes = sum(1 for s in samples if s['action'] == 'like')
        target_likes = total // 2
        if current_likes > target_likes:
            need_flip = current_likes - target_likes
            for s in samples:
                if need_flip <= 0:
                    break
                if s['action'] == 'like' and random.random() < 0.5:
                    s['action'] = 'skip'
                    need_flip -= 1
        elif current_likes < target_likes:
            need_flip = target_likes - current_likes
            for s in samples:
                if need_flip <= 0:
                    break
                if s['action'] == 'skip' and random.random() < 0.5:
                    s['action'] = 'like'
                    need_flip -= 1

    # 统计
    positive_ratio = like_count / total if total > 0 else 0
    print(f"\n=== 生成完成 ===")
    print(f"总样本数: {total}")
    print(f"正样本(like): {like_count} ({positive_ratio*100:.1f}%)")
    print(f"负样本(skip): {skip_count} ({(1-positive_ratio)*100:.1f}%)")
    if total > 0:
        print(f"LLM分析: {llm_analyzed} ({llm_analyzed/total*100:.1f}%)")
    else:
        print(f"LLM分析: {llm_analyzed} (0.0%)")

    # 保存
    # 固定写入到logs目录
    output_file = Path(__file__).parent / 'recommend_events.jsonl'
    
    # 备份原文件
    if output_file.exists():
        backup_file = Path('recommend_events_backup.jsonl')
        print(f"\n备份原文件到: {backup_file}")
        if backup_file.exists():
            backup_file.unlink()
        output_file.rename(backup_file)
    
    with output_file.open('w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n数据已保存到: {output_file}")
    print(f"样本数: {len(samples)}")
    
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--output', type=str, default=str(Path(__file__).parent / 'recommend_events.jsonl'))
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    llm = None
    if API_KEY:
        try:
            llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.5)
        except Exception:
            llm = None

    if args.single:
        jobs = load_real_jobs()
        positions = load_positions()
        resume = generate_resume_with_llm(llm, positions)
        job = random.choice(jobs) if jobs else {}
        result = analyze_match_with_llm(resume, job, llm) if llm else None
        if result is None:
            job_position = job.get('三级分类') or job.get('职位类型名称') or ''
            interested = job_position in resume['work_preferences']['position_type_name']
            if random.random() < 0.2:
                interested = not interested
        else:
            interested = result.get('interested', False)
        sample = {
            'action': ('like' if interested else 'skip'),
            'ts': '2025-11-11T00:00:00Z',
            'type': 'cold_start',
            'job': job,
            'resume': resume
        }
        out = Path(args.output)
        mode = 'w' if args.fresh or (not out.exists()) else 'a'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open(mode, encoding='utf-8') as w:
            w.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"写入1条样本到: {out}")
    else:
        generate_samples()

