"""证据和逻辑分离的需求模型；接口兼容v1，未知不视为不会或满足。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from scripts.profile_data import SKILLS, DISTRICTS, parse_salary
from . import semantics as semantic

PATTERNS={key:re.compile(value,re.I|re.ASCII) for key,value in SKILLS.items()}
TASK_PATTERNS={key:re.compile(value[1],re.I) for key,value in semantic.TASK_PATTERNS.items()}
TASK_NAMES={key:value[0] for key,value in semantic.TASK_PATTERNS.items()}
EDUCATION={'初中':0,'高中':1,'中专':1,'大专':2,'专科':2,'本科':3,'硕士':4,'研究生':4,'博士':5}
LEVELS=semantic.LEVELS
CONTACT=re.compile(r'(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.\w+|(?<!\d)1[3-9]\d{9}(?!\d)')


def digest(value: str)->str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


PARSER_VERSION=semantic.SEMANTICS_VERSION+'-'+digest(canonical({'skills':SKILLS,'tasks':semantic.TASK_PATTERNS}))[:12]


def display_text(text: str)->str:
    return CONTACT.sub('[联系方式已隐藏]',text)


def _evidence(text,patterns,field_name,kind,names=None):
    items=semantic.mentions(text,patterns,field_name,kind,names)
    for item in items:
        item['context']=display_text(text[item['context_start']:item['context_end']][:180])
    return semantic.compact_mentions(items,field_name in {'requirements','description'})


def skill_evidence(text: str,field_name: str='resume')->dict[str,dict]:
    return _evidence(text,PATTERNS,field_name,'skill')


def task_evidence(text: str,field_name: str='resume')->dict[str,dict]:
    """小型、透明的业务任务词表；未覆盖任务保持未知，不推断零能力。"""
    return _evidence(text,TASK_PATTERNS,field_name,'task',TASK_NAMES)


def _occurrences(evidence):
    return sorted([dict(item) for entry in evidence.values() for item in entry.get('mentions',[entry])],key=lambda item:(item['start'],item['end']))


def _format_ast(ast):
    if 'type' in ast:
        return ast['text']
    separator=' 或 ' if ast['op']=='any' else ' 且 '
    return separator.join('('+_format_ast(child)+')' if 'type' not in child else _format_ast(child) for child in ast['children'])


def _group(ast):
    leaves=semantic.ast_leaves(ast)
    kind='single' if 'type' in ast else 'any' if ast['op']=='any' else 'all'
    evidence=ast.get('scope_evidence') or ast.get('evidence')
    modalities={item['modality'] for item in leaves}
    modality=next(iter(modalities)) if len(modalities)==1 else 'mixed'
    return {'kind':kind,'skills':list(dict.fromkeys(item['key'] for item in leaves if item['type']=='skill')),
            'tasks':list(dict.fromkeys(item['key'] for item in leaves if item['type']=='task')),
            'label':_format_ast(ast),'evidence':evidence,'preferred':modality=='preferred','modality':modality,
            'parse_status':ast.get('parse_status','unknown'),'ast':ast,
            'group_id':digest(canonical({'parser':PARSER_VERSION,'ast':ast}))[:20]}


def requirement_groups(text: str,evidence: dict,field_name: str='description')->list[dict]:
    # 对旧调用默认field=resume的evidence重新按岗位文本解释，防止模态全部unknown。
    entries=_occurrences(evidence)
    if entries and any(item.get('field')!=field_name for item in entries):
        entries=_occurrences(skill_evidence(text,field_name))
    return [_group(ast) for ast in semantic.requirement_asts(text,entries)]


def education(text: str,is_job: bool=False)->int|None:
    return semantic.education_details(text,EDUCATION,is_job)['level']


def experience_details(text: str,is_job: bool=False)->dict:
    return semantic.experience_details(text,PATTERNS,is_job)


def experience(text: str,is_job: bool=False)->float|None:
    """只返回明确总年资；专项技能时长另存，不冒充总工作年资。"""
    return experience_details(text,is_job)['total_years']


def experience_blocks(text: str)->list[dict]:
    return semantic.experience_blocks(text)


def _semantic_identity(ast):
    if 'type' in ast:
        return (ast['type'],ast['key'],ast['modality'])
    return (ast['op'],ast.get('parse_status'),tuple(_semantic_identity(child) for child in ast['children']))


@dataclass
class Job:
    id:str
    version:str
    title:str
    company:str
    category:str
    family:str
    salary_raw:str
    requirements:str
    description:str
    address:str
    row:int
    sheet:int=1
    salary:dict=field(default_factory=dict)
    skills:dict=field(default_factory=dict)
    groups:list=field(default_factory=list)
    city:str|None=None
    district:str|None=None
    experience_min:float|None=None
    education_min:int|None=None
    job_family_id:str=''
    tasks:dict=field(default_factory=dict)
    requirement_ast:dict=field(default_factory=dict)
    requirement_mentions:list=field(default_factory=list)
    experience_requirements:dict=field(default_factory=dict)
    education_requirements:dict=field(default_factory=dict)
    parser_version:str=PARSER_VERSION

    def __post_init__(self):
        self.salary=parse_salary(self.salary_raw)
        self.groups=[];all_skills=[];all_tasks=[]
        # 不以同技能为由删除另一字段的整个OR组；只合并语义完全相同的重复条件。
        seen=set()
        for field_name in ('requirements','description'):
            text=getattr(self,field_name)
            skills=skill_evidence(text,field_name);tasks=task_evidence(text,field_name)
            all_skills.extend(_occurrences(skills));all_tasks.extend(_occurrences(tasks))
            for entries in (_occurrences(skills),_occurrences(tasks)):
                for ast in semantic.requirement_asts(text,entries):
                    identity=_semantic_identity(ast)
                    if identity not in seen:
                        self.groups.append(_group(ast));seen.add(identity)
        self.skills=semantic.compact_mentions(all_skills,True)
        self.tasks=semantic.compact_mentions(all_tasks,True)
        self.requirement_mentions=all_skills+all_tasks
        self.requirement_ast={'op':'all','children':[group['ast'] for group in self.groups],
                              'parser_version':PARSER_VERSION,'modality':'mixed','parse_status':'known' if all(group['parse_status']=='known' for group in self.groups) else 'unknown'}
        self.city=next((name for name in ['深圳','广州','北京','上海','杭州','东莞','惠州'] if name in self.address),None)
        self.district=next((name for name in DISTRICTS if name in self.address),None)
        self.experience_requirements=experience_details(self.requirements,True)
        self.education_requirements=semantic.education_details(self.requirements,EDUCATION,True)
        self.experience_min=self.experience_requirements['total_years']
        self.education_min=self.education_requirements['level']

    def public(self)->dict:
        return {'id':self.id,'version':self.version,'job_family_id':self.job_family_id,'title':display_text(self.title),'company':display_text(self.company),
                'category':self.category,'family':self.family,'salary_raw':display_text(self.salary_raw),'city':self.city,'district':self.district,
                'requirements':display_text(self.requirements),'salary':self.salary,'experience_min':self.experience_min,'education_min':self.education_min,
                'skills':list(self.skills),'tasks':[TASK_NAMES[key] for key in self.tasks],'active_status':'unknown','parser_version':PARSER_VERSION}


def parse_profile(text: str,preferences: dict)->dict:
    parsed_experience=experience_details(text)
    parsed_education=semantic.education_details(text,EDUCATION)
    explicit_years=preferences.get('experience_years')
    explicit_education=preferences.get('education')
    return {'text':text,'version':digest(canonical({'text':text,'preferences':preferences,'parser_version':PARSER_VERSION}))[:16],
            'parser_version':PARSER_VERSION,'skills':skill_evidence(text),'tasks':task_evidence(text),
            'experience_years':explicit_years if explicit_years is not None else parsed_experience['total_years'],
            'experience_details':parsed_experience,'experience_blocks':experience_blocks(text),
            'education':EDUCATION.get(explicit_education) if explicit_education else parsed_education['level'],
            'education_details':parsed_education,'education_full_time':preferences.get('education_full_time'),
            'city':preferences.get('city') or None,'intent':(preferences.get('intent') or '').strip(),
            'salary_min':preferences.get('salary_min'),'salary_max':preferences.get('salary_max'),'district':preferences.get('district') or None,
            'claim_note':'简历文本属于本人自述，规则跨度不能验证经历真实或能力水平。'}


def constraints(profile: dict,job: Job)->list[dict]:
    checks=[]
    def add(name,known,passed,value,**extra):
        checks.append({'name':name,'status':('pass' if passed else 'fail') if known else 'unknown','detail':display_text(value),**extra})
    if profile.get('city'):
        add('目标城市',bool(job.city),job.city==profile['city'],job.city or '岗位未提供可确认城市')
    if profile.get('salary_min') is not None:
        monthly=job.salary.get('status')=='月薪'
        add('最低月薪',monthly,monthly and job.salary['monthly_high']>=profile['salary_min'],job.salary_raw or '薪资未知')
    edu_known=profile.get('education') is not None and job.education_min is not None and not job.education_requirements.get('conflict')
    add('学历要求',edu_known,edu_known and profile['education']>=job.education_min,job.requirements or '未知')
    if job.education_requirements.get('full_time_required'):
        add('全日制要求',profile.get('education_full_time') is not None,profile.get('education_full_time') is True,'岗位存在全日制条件，须由用户确认')
    years=profile.get('experience_years');known=years is not None and job.experience_min is not None and not job.experience_requirements.get('conflict')
    upper=job.experience_requirements.get('total_maximum')
    add('经验要求',known,known and years>=job.experience_min and (upper is None or years<=upper),job.requirements or '未知',scope='total')
    for index,requirement in enumerate(job.experience_requirements.get('specialized',[])):
        if requirement['optional'] or requirement['negated']:
            continue
        skills=set(requirement['skills']);candidates=[item for item in profile.get('experience_details',{}).get('specialized',[]) if skills and skills==set(item['skills']) and not item['negated']]
        # 总年资不会代替专项年资；没有专项跨度必须为unknown。
        value=max((item['minimum'] for item in candidates),default=None)
        quote=requirement['evidence']['quote']
        label=' / '.join(sorted(skills)) or '未明确专项'
        add(f'专项经验：{label}',value is not None,value is not None and value>=requirement['minimum'],quote,scope='specialized',evidence=requirement['evidence'])
    return checks


def _evaluate_ast(ast,profile):
    if 'type' in ast:
        source=profile.get('skills' if ast['type']=='skill' else 'tasks',{}).get(ast['key'])
        negative=source is not None and source.get('polarity')=='negative' and source.get('parse_status')=='known'
        positive=source is not None and source.get('weight',0)>=.7 and source.get('parse_status')=='known' and source.get('actor')!='background'
        status='fail' if negative else 'pass' if positive else 'unknown'
        if ast.get('modality')=='unknown' or ast.get('parse_status')=='unknown':
            status='unknown'
        strength=source.get('weight',0) if source and not negative else 0.0
        if source and (source.get('actor')=='background' or source.get('conflict')):
            strength=0.0
        result={'type':ast['type'],'key':ast['key'],'text':ast['text'],'status':status,'strength':strength,
                'modality':ast['modality'],'required':ast['required'],'resume_evidence':source,'job_evidence':ast['evidence']}
        return {'status':status,'strength':strength,'lower':1.0 if status=='pass' else 0.0,'upper':0.0 if status=='fail' else 1.0,'leaf_support':[result],
                'chosen_leaves':[result] if source and not negative and source.get('actor')!='background' else [],'modality':ast['modality']}
    children=[_evaluate_ast(child,profile) for child in ast.get('children',[])]
    if not children:
        return {'status':'unknown','strength':0.0,'lower':0.0,'upper':1.0,'leaf_support':[],'chosen_leaves':[],'modality':ast.get('modality','unknown')}
    considered=[child for child in children if child['modality'] not in {'preferred','negated'}] or children
    if ast['op']=='any':
        chosen=max(considered,key=lambda child:(child['status']=='pass',child['strength']))
        status='pass' if any(child['status']=='pass' for child in considered) else 'fail' if all(child['status']=='fail' for child in considered) else 'unknown'
        strength=max(child['strength'] for child in considered);lower=max(child['lower'] for child in considered);upper=max(child['upper'] for child in considered)
        selected=chosen['chosen_leaves']
    else:
        status='fail' if any(child['status']=='fail' for child in considered) else 'pass' if all(child['status']=='pass' for child in considered) else 'unknown'
        strength=min(child['strength'] for child in considered);lower=min(child['lower'] for child in considered);upper=min(child['upper'] for child in considered)
        selected=[leaf for child in considered for leaf in child['chosen_leaves']]
    if ast.get('parse_status')=='unknown':
        status='unknown';lower=0.0;upper=1.0
    return {'status':status,'strength':strength,'lower':lower,'upper':upper,'leaf_support':[leaf for child in children for leaf in child['leaf_support']],
            'chosen_leaves':selected,'modality':ast.get('modality','mixed')}


def group_support(profile: dict,job: Job)->list[dict]:
    return [{'group_id':group['group_id'],'group':group['label'],'kind':group['kind'],'preferred':group['preferred'],
             'parse_status':group['parse_status'],**_evaluate_ast(group['ast'],profile)} for group in job.groups]


def supported_matches(profile: dict,job: Job)->tuple[list,list,float]:
    matches=[];gaps=[];scores=[];preferred_scores=[]
    for group,support in zip(job.groups,group_support(profile,job)):
        (preferred_scores if group['preferred'] else scores).append(support['strength'])
        selected=support['chosen_leaves']
        if selected:
            representative=max(selected,key=lambda leaf:leaf['strength']);source=representative['resume_evidence']
            matches.append({'skill':representative['text'],'type':representative['type'],'group':group['label'],'kind':group['kind'],
                            'group_id':group['group_id'],'level':source['level'],'strength':support['strength'],'status':support['status'],
                            'resume_evidence':source,'job_evidence':representative['job_evidence'],'leaf_support':support['leaf_support'],
                            'support_bounds':[support['lower'],support['upper']],
                            'note':'简历自述了相关操作，尚未核验经历真实性' if source['claim_type']=='self_reported_action' else '仅有技能或任务自述，建议补充实际操作依据'})
        if support['status']!='pass':
            message='简历明确否定了至少一项必要要求' if support['status']=='fail' else '当前材料尚不能确认这组要求'
            if group['kind']=='any':
                message+='，可补充其中一个完整分支的依据'
            gaps.append({'skill':group['label'],'options':group['skills'],'tasks':group['tasks'],'kind':group['kind'],'group_id':group['group_id'],
                         'preferred':group['preferred'],'evidence':group['evidence'],'status':support['status'],'leaf_support':support['leaf_support'],
                         'message':message,'parse_status':group['parse_status']})
    values=scores or preferred_scores
    coverage=sum(values)/len(values) if values else 0.0
    return matches,gaps,coverage
