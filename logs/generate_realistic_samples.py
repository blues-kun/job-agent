"""
使用真实职位数据和LLM生成高质量训练样本
目标：AUC 0.85-0.90，准确率 85-90%
"""
import json
import random
import csv
import sys
from pathlib import Path
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
    """从CSV加载真实职位数据"""
    # 使用脚本所在目录的绝对路径
    script_dir = Path(__file__).parent
    jobs_file = script_dir.parent / 'data' / 'job_data.csv'
    jobs = []
    
    print(f"读取职位文件: {jobs_file}")
    with jobs_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(row)
    
    print(f"加载了 {len(jobs)} 条真实职位数据")
    return jobs

def load_positions():
    """加载职位列表"""
    # 使用脚本所在目录的绝对路径
    script_dir = Path(__file__).parent
    positions_file = script_dir.parent / 'data' / 'builtin_positions.txt'
    positions = []
    
    for line in positions_file.read_text('utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            positions.append(line)
    
    print(f"加载了 {len(positions)} 个职位类型")
    return positions

def create_diverse_resumes(positions, num_resumes=20):
    """创建多样化的求职者简历，确保城市均匀分布"""
    resumes = []
    
    # 确保每个城市的简历数量均匀
    resumes_per_city = num_resumes // len(CITIES)
    remaining = num_resumes % len(CITIES)
    
    print(f"每个城市生成 {resumes_per_city} 个简历，剩余 {remaining} 个随机分配")
    
    city_counts = {city: 0 for city in CITIES}
    
    for i in range(num_resumes):
        # 随机选择1-3个期望职位
        num_positions = random.randint(1, 3)
        desired_positions = random.sample(positions, num_positions)
        
        # 均匀分配城市
        if i < len(CITIES) * resumes_per_city:
            city = CITIES[i % len(CITIES)]
        else:
            city = random.choice(CITIES)
        city_counts[city] += 1
        
        # 是否愿意异地
        relocate = random.choice([True, False])
        
        # 经验年限
        experience = random.choice(EXPERIENCE_RANGES)
        
        # 学历
        education = random.choice(EDUCATION_LEVELS)
        
        # 期望薪资（基于经验）
        base_salary = 150000 if education == '大专' else 200000
        if education == '硕士':
            base_salary = 280000
        elif education == '博士':
            base_salary = 400000
        
        salary_expectation = base_salary + experience * 30000
        
        resume = {
            'personal_info': {
                'current_city': city,
                'willingness_to_relocate': relocate,
                'availability_date': '2024-08-01'
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
                'school_level': 0 if education in ['大专', '本科'] else 1
            },
            'full_resume_text': f"求职者{i+1}，期望职位：{', '.join(desired_positions)}，{experience}年经验，{education}学历"
        }
        
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
    
    if not API_KEY:
        print("错误：未配置API_KEY")
        return
    
    # 初始化LLM
    print(f"初始化LLM: {MODEL}\n")
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.5)
    
    # 加载数据
    jobs = load_real_jobs()
    positions = load_positions()
    resumes = create_diverse_resumes(positions, num_resumes=15)
    
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
        
        # === 分层抽样：确保正负样本比例 ===
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
        # 这样能确保约35-40%的正样本
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
        
        for job_idx, job in enumerate(selected_jobs, 1):
            try:
                # 使用LLM分析
                result = analyze_match_with_llm(resume, job, llm)
                
                if result is None:
                    # LLM失败，使用简单规则
                    job_position = job['三级分类']
                    interested = job_position in resume['work_preferences']['position_type_name']
                    
                    # 增加噪声
                    if random.random() < 0.2:
                        interested = not interested
                else:
                    interested = result.get('interested', False)
                    llm_analyzed += 1
                
                action = 'like' if interested else 'skip'
                
                sample = {
                    'action': action,
                    'ts': '2025-11-11T00:00:00Z',
                    'type': 'cold_start',
                    'job': job,
                    'resume': resume
                }
                samples.append(sample)
                
                if action == 'like':
                    like_count += 1
                else:
                    skip_count += 1
                
                # 每50个样本打印进度
                if len(samples) % 50 == 0:
                    print(f"  进度: {len(samples)}/{total_samples} - like:{like_count}, skip:{skip_count}, LLM:{llm_analyzed}")
                
            except Exception as e:
                print(f"  错误: {e}")
                continue
    
    # 打乱数据
    random.shuffle(samples)
    
    # 统计
    total = len(samples)
    positive_ratio = like_count / total if total > 0 else 0
    
    print(f"\n=== 生成完成 ===")
    print(f"总样本数: {total}")
    print(f"正样本(like): {like_count} ({positive_ratio*100:.1f}%)")
    print(f"负样本(skip): {skip_count} ({(1-positive_ratio)*100:.1f}%)")
    print(f"LLM分析: {llm_analyzed} ({llm_analyzed/total*100:.1f}%)")
    
    # 保存
    output_file = Path('recommend_events.jsonl')
    
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
    generate_samples()

