import pandas as pd
import json
import random

random.seed(42)

cities = ['南京', '上海', '北京', '深圳']
districts = {
    '南京': ['玄武区', '鼓楼区', '雨花台区', '江宁区'],
    '上海': ['浦东新区', '徐汇区', '闵行区', '长宁区'],
    '北京': ['海淀区', '朝阳区', '东城区', '丰台区'],
    '深圳': ['南山区', '福田区', '宝安区', '龙华区']
}
roads = ['科技大道', '创新路', '软件园路', '高新区一路', '滨江大道', '中心大道', '星河路', '云谷路']
buildings = ['国际中心', '科创大厦', '智谷大厦', '云计算中心', '数据中心', '人才大厦', '产业园A座', '研发中心']
company_prefix = ['星宸', '云鲲', '智谱', '蓝图', '极光', '数联', '辰曜', '博睿', '乾元', '凌云', '曜石', '光瀚', '钛舟', '澜芯', '星河']
company_suffix = ['科技', '网络', '信息', '软件', '数据', '智能', '云科', '数科', '系统', '科技集团']
company_tail = ['有限公司', '股份有限公司']
sec_ter = {
    '后端开发': ['Java', 'C/C++', 'PHP', 'Python', 'Golang', 'Node.js', '全栈工程师'],
    '前端/移动开发': ['前端开发工程师', 'Android', 'iOS', '鸿蒙开发工程师'],
    '测试': ['测试工程师', '自动化测试', '测试开发', '性能测试'],
    '运维/技术支持': ['运维工程师', 'IT技术支持', '网络工程师', '系统工程师', 'DBA'],
    '人工智能': ['算法工程师', '机器学习', '深度学习', '推荐算法', '数据挖掘'],
    '数据': ['数据分析师', '数据开发', '数据仓库', '数据治理']
}
levels = ['初级', '中级', '高级', '资深', '架构师']
skills = ['Java', 'SpringBoot', 'SpringCloud', 'MySQL', 'Redis', 'Kafka', 'RabbitMQ', 'Docker', 'Kubernetes', '微服务', '多线程', 'JVM', 'React', 'Vue', 'PyTorch', 'TensorFlow', 'Go', 'Node.js']


def gen_company():
    return f"{random.choice(company_prefix)}{random.choice(company_suffix)}{random.choice(company_tail)}"


def gen_title(ter):
    lvl = random.choice(levels)
    if ter in ['Java', 'Python', 'Golang', 'C/C++', 'PHP', 'Node.js', '全栈工程师']:
        return f"{ter}{lvl}开发工程师"
    elif ter in ['前端开发工程师', 'Android', 'iOS', '鸿蒙开发工程师']:
        return f"{ter}"
    elif ter in ['测试工程师', '自动化测试', '测试开发', '性能测试']:
        return f"{ter}"
    elif ter in ['运维工程师', 'IT技术支持', '网络工程师', '系统工程师', 'DBA']:
        return f"{ter}"
    else:
        return f"{ter}{lvl}"


def gen_salary():
    if random.random() < 0.1:
        return '面议'
    low = random.choice([10, 12, 15, 18, 20, 22, 25, 28, 30])
    high = low + random.choice([5, 7, 8, 10, 12, 15])
    months = random.choice(['·12薪', '·13薪', 'x13薪'])
    return f"{low}-{high}K{months}"


def gen_requirements():
    exp = random.choice(['经验不限', '0-1年', '1-3年', '3-5年', '5-10年'])
    edu = random.choice(['本科', '大专', '硕士'])
    extra = random.choice(['', ' 985优先', ' 211优先'])
    return f"{exp}{edu}{extra}".strip()


def gen_responsibilities():
    ks = ', '.join(random.sample(skills, k=random.randint(5, 8)))
    lines = [
        f"参与系统设计与核心模块开发，使用{ks}",
        "编写接口与技术文档，保证代码质量与稳定性",
        "优化性能与可靠性，参与上线与故障排查",
        "协作跨团队沟通，推进项目按期交付"
    ]
    return '； '.join(lines)


def gen_address(city):
    d = random.choice(districts[city])
    r = random.choice(roads)
    b = random.choice(buildings)
    num = random.choice(['A座8层', 'B座12层', 'C座6层', 'D座10层', 'E座15层'])
    return f"{city}{d}{r}{b}{num}"


def main():
    rows = []
    jsonl_objs = []
    N = 10000
    for i in range(N):
        city = cities[i % len(cities)]
        sec = random.choice(list(sec_ter.keys()))
        ter = random.choice(sec_ter[sec])
        company = gen_company()
        title = gen_title(ter)
        salary = gen_salary()
        req = gen_requirements()
        resp = gen_responsibilities()
        addr = gen_address(city)
        row = {
            '岗位名称': title,
            '企业': company,
            '岗位薪资': salary,
            '岗位要求': req,
            '岗位职责': resp,
            '岗位地址': addr,
            '二级分类': sec,
            '三级分类': ter,
            '城市': city,
        }
        rows.append(row)
        comp = {'salary_negotiable': ('面议' in salary)}
        if '面议' not in salary:
            import re
            m = re.search(r'(\d+)[-~](\d+)K', salary)
            months = 12
            mm = re.search(r'[·x×]\s*(\d+)\s*薪', salary)
            if mm:
                months = int(mm.group(1))
            if m:
                min_k = int(m.group(1)) * 1000 * months
                max_k = int(m.group(2)) * 1000 * months
                comp['salary_range_annual'] = {'min': min_k, 'max': max_k}
        obj = {
            'job_identity': {
                'original_job_title': title,
                'company_name': company,
                'position_type_name': ter,
                'secondary_category': sec,
            },
            'original_text': {
                'raw_requirements': req,
                'raw_responsibilities': resp,
            },
            'location': {
                'city': city,
                'address': addr,
            },
            'compensation': comp,
            'requirements': {
                'min_experience_years': 0 if ('经验不限' in req) else (int(req.split('年')[0].split('-')[0]) if '-' in req else (int(req.split('年')[0]) if '年' in req else 0)),
                'education_level_min': ('本科' if '本科' in req else ('硕士' if '硕士' in req else ('大专' if '大专' in req else None))),
                'school_level_min': (2 if '985' in req else (1 if '211' in req else 0)),
            },
        }
        jsonl_objs.append(obj)
    df = pd.DataFrame(rows)
    df.to_csv('data/job_data.csv', index=False, encoding='utf-8')
    df.to_excel('data/job_data.xlsx', index=False)
    with open('data/job_data.jsonl', 'w', encoding='utf-8') as f:
        for o in jsonl_objs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print('生成完成', N)


if __name__ == '__main__':
    main()

