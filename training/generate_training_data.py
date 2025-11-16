"""
生成训练数据（基于深圳数据）
使用向量化特征 + 用户交互标注
"""
import json
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import JOBS_FILE
from data_preprocess.loader import JobDataLoader
from data_preprocess.data_processor import JobDataProcessor
from models import ResumeProfile


def generate_user_profiles(base_profile: ResumeProfile, n_profiles: int = 20) -> list:
    """
    生成多样化的用户画像
    
    Args:
        base_profile: 基础简历模板
        n_profiles: 生成数量
        
    Returns:
        用户画像列表
    """
    profiles = []
    
    # 深圳的区域
    shenzhen_districts = ['南山区', '福田区', '罗湖区', '龙华区', '龙岗区', '宝安区']
    
    # 薪资范围（年薪，单位：元）
    salary_ranges = [
        150000, 180000, 210000, 240000, 280000, 320000, 
        360000, 400000, 500000, 600000
    ]
    
    # 职位意向（基于数据中的三级分类）
    position_intents = [
        ['Java', '后端开发', 'Spring'],
        ['Python', '后端开发', 'Django'],
        ['前端开发', 'Vue', 'React'],
        ['算法工程师', '机器学习', 'AI'],
        ['测试工程师', '自动化测试'],
        ['产品经理', '项目管理'],
        ['运维工程师', 'DevOps'],
        ['数据分析', 'BI'],
    ]
    
    # 经验年限
    experience_years = [1, 2, 3, 5, 7, 10]
    
    # 学历
    education_levels = ['本科', '硕士', '博士']
    
    # 院校层次 (0=普通, 1=211, 2=985)
    school_levels = [0, 1, 2]
    
    for i in range(n_profiles):
        # 复制基础模板
        profile_dict = base_profile.model_dump()
        
        # 随机化参数
        profile_dict['personal_info']['current_city'] = '深圳'
        
        profile_dict['work_preferences']['salary_expectation']['min_annual_package'] = random.choice(salary_ranges)
        profile_dict['work_preferences']['position_type_name'] = random.choice(position_intents)
        
        profile_dict['professional_summary']['total_experience_years'] = float(random.choice(experience_years))
        profile_dict['professional_summary']['education_level'] = random.choice(education_levels)
        profile_dict['professional_summary']['school_level'] = random.choice(school_levels)
        
        # 生成简单的简历文本
        intent_text = '、'.join(profile_dict['work_preferences']['position_type_name'])
        exp_years = profile_dict['professional_summary']['total_experience_years']
        edu_level = profile_dict['professional_summary']['education_level']
        
        profile_dict['full_resume_text'] = f"""
        求职意向: {intent_text}
        工作经验: {exp_years}年
        学历: {edu_level}
        期望薪资: {profile_dict['work_preferences']['salary_expectation']['min_annual_package']/10000:.0f}万/年
        """
        
        profiles.append(ResumeProfile.model_validate(profile_dict))
    
    return profiles


def label_job_for_profile(profile: ResumeProfile, job: dict) -> int:
    """
    为用户画像和职位打标签（模拟用户选择）
    
    规则：
    1. 职位意向完全不匹配 -> label=0
    2. 薪资太低（低于期望50%）-> label=0
    3. 经验要求差距太大 -> label=0
    4. 学历不符 -> label=0
    5. 其他情况综合评分决定
    
    Args:
        profile: 用户简历
        job: 职位信息
        
    Returns:
        0 (不感兴趣) 或 1 (感兴趣)
    """
    score = 0.0
    
    # 1. 职位意向匹配（最重要）
    job_title = str(job.get('岗位名称', '') or '')
    job_type = str(job.get('三级分类', '') or '')
    user_intents = profile.work_preferences.position_type_name
    
    intent_match = False
    for intent in user_intents:
        if intent.lower() in job_title.lower() or intent.lower() in job_type.lower():
            intent_match = True
            break
    
    if not intent_match:
        return 0  # 职位不匹配，直接拒绝
    
    score += 0.4
    
    # 2. 薪资匹配
    from data_preprocess.salary_normalizer import SalaryNormalizer
    salary_info = SalaryNormalizer.normalize(job.get('岗位薪资', ''))
    
    user_expect = profile.work_preferences.salary_expectation.min_annual_package
    job_avg_salary = salary_info.get('avg_annual')
    
    if job_avg_salary and user_expect:
        salary_ratio = job_avg_salary / user_expect
        if salary_ratio < 0.5:
            return 0  # 薪资太低
        elif salary_ratio >= 1.0:
            score += 0.3
        elif salary_ratio >= 0.8:
            score += 0.2
        else:
            score += 0.1
    
    # 3. 经验年限
    import re
    req_text = str(job.get('岗位要求', '') or '')
    exp_match = re.search(r'(\d+)[-~年]', req_text)
    
    if exp_match:
        req_years = int(exp_match.group(1))
        user_years = profile.professional_summary.total_experience_years
        
        if user_years >= req_years * 0.8:
            score += 0.2
        elif user_years < req_years * 0.5:
            return 0  # 经验差距太大
    else:
        score += 0.1  # 无明确要求
    
    # 4. 学历匹配
    edu_order = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}
    user_edu = edu_order.get(profile.professional_summary.education_level, 0)
    
    req_edu = 0
    for edu, level in edu_order.items():
        if edu in req_text:
            req_edu = max(req_edu, level)
    
    if req_edu > 0:
        if user_edu >= req_edu:
            score += 0.1
        else:
            return 0  # 学历不符
    
    # 最终决策
    # 添加一些随机性，模拟真实用户行为
    noise = random.uniform(-0.1, 0.1)
    final_score = score + noise
    
    return 1 if final_score >= 0.6 else 0


def main():
    """主流程"""
    random.seed(42)
    
    print("="*80)
    print("生成训练数据（基于深圳职位数据）")
    print("="*80)
    
    # 1. 加载职位数据
    print("\n[1/4] 加载职位数据...")
    loader = JobDataLoader(JOBS_FILE)
    jobs = loader.to_dict_list()
    print(f"  ✓ 加载了 {len(jobs)} 个职位")
    
    # 过滤深圳职位
    shenzhen_jobs = [
        job for job in jobs 
        if '深圳' in str(job.get('岗位地址', '') or '') or 
           '深圳' in str(job.get('城市', '') or '')
    ]
    print(f"  ✓ 深圳职位: {len(shenzhen_jobs)} 个")
    
    if len(shenzhen_jobs) < 100:
        print(f"  警告: 深圳职位数量较少，使用全部数据")
        shenzhen_jobs = jobs
    
    # 2. 生成用户画像
    print("\n[2/4] 生成用户画像...")
    # 创建基础模板
    base_template = ResumeProfile.model_validate({
        'personal_info': {
            'current_city': '深圳',
            'willingness_to_relocate': False
        },
        'work_preferences': {
            'position_type_name': ['Java'],
            'salary_expectation': {'min_annual_package': 200000}
        },
        'professional_summary': {
            'total_experience_years': 3.0,
            'education_level': '本科',
            'school_level': 0
        },
        'full_resume_text': ''
    })
    
    profiles = generate_user_profiles(base_template, n_profiles=30)
    print(f"  ✓ 生成了 {len(profiles)} 个用户画像")
    
    # 3. 生成训练样本
    print("\n[3/4] 生成训练样本...")
    samples = []
    
    for profile in profiles:
        # 为每个用户随机选择一些职位进行标注
        sampled_jobs = random.sample(shenzhen_jobs, min(100, len(shenzhen_jobs)))
        
        for job in sampled_jobs:
            label = label_job_for_profile(profile, job)
            
            sample = {
                'resume': profile.model_dump(),
                'job': job,
                'label': label
            }
            samples.append(sample)
    
    print(f"  ✓ 生成了 {len(samples)} 个样本")
    
    # 统计
    positive = sum(1 for s in samples if s['label'] == 1)
    negative = len(samples) - positive
    print(f"  ✓ 正样本: {positive} ({positive/len(samples)*100:.1f}%)")
    print(f"  ✓ 负样本: {negative} ({negative/len(samples)*100:.1f}%)")
    
    # 4. 保存数据
    print("\n[4/4] 保存训练数据...")
    output_dir = Path('logs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'recommend_events.jsonl'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 已保存到: {output_path}")
    print(f"  ✓ 共 {len(samples)} 条记录")
    
    print("\n" + "="*80)
    print("训练数据生成完成！")
    print("="*80)
    print(f"\n下一步:")
    print(f"  1. 运行向量化预处理: python -m data_preprocess.data_processor")
    print(f"  2. 训练模型: python -m training.train_xgb")


if __name__ == '__main__':
    main()

