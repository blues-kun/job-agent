"""保守的中文需求语义：证据跨度、逻辑作用域与条件修饰。

规则命中仅表示文本支持，不验证经历真假；不确定作用域显式标记unknown。
"""
from __future__ import annotations

import re

SEMANTICS_VERSION = 'demand-semantics-v2.0.2-actor-scope'
LEVELS = {'否定':0.0, '提及':0.25, '了解':0.4, '熟悉':0.7, '实践':1.0, '精通':0.7}
TASK_PATTERNS = {
    'data_cleaning': ('数据清洗', r'数据清洗|清洗数据|数据去重|清洗[^，。；\n]{0,6}数据'),
    'data_labeling': ('数据标注', r'数据标注|标注数据|图像标注|文本标注'),
    'customer_discovery': ('客户需求分析', r'客户需求分析|分析客户需求|梳理客户需求|客户需求调研'),
    'customer_success': ('客户维护', r'客户维护|维护客户关系|客户回访|客户续费|处理客户问题'),
    'recruiting': ('招聘筛选', r'招聘工作|招聘流程|简历筛选|筛选简历|候选人沟通'),
    'project_coordination': ('项目协调', r'项目协调|协调项目|跨部门协调|跟踪项目进度|项目进度管理'),
    'requirements_analysis': ('需求分析', r'需求分析|需求澄清|需求梳理|梳理[^，。；\n]{0,5}需求'),
    'documentation': ('文档编写', r'编写[^，。；\n]{0,6}文档|撰写[^，。；\n]{0,6}文档|文档编写'),
    'test_execution': ('测试执行', r'执行测试|测试用例设计|编写测试用例|设计测试用例|缺陷复现'),
    'deployment': ('系统部署', r'系统部署|部署系统|服务部署|部署服务|部署应用'),
    'troubleshooting': ('故障排查', r'故障排查|排查故障|定位故障|排查[^，。；\n]{0,6}问题'),
    'training_delivery': ('业务培训', r'业务培训|用户培训|客户培训|实施培训'),
}
NEGATIVE = r'不会|不熟悉|不掌握|未掌握|未使用(?:过)?|不了解|没有.{0,5}(?:经验|接触)|尚未|未学'
NOT_REQUIRED = r'不要求|无需(?:掌握|了解|熟悉|具备|精通)?|不必(?:掌握|了解|熟悉|具备|精通)?|无须'
PREFERRED = r'优先|加分|更佳|优选|非必需|非必须|可选项'
REQUIRED = r'必须|必需|必备|要求|熟悉|掌握|精通|熟练|了解|具备|能够|能独立|负责|使用'
HEADINGS = re.compile(r'(加分项|优先条件|优先项|必备条件|必备技能|必须项|必需项|必需|必须|必备|核心要求|任职要求|岗位要求|项目经历|工作经历|技能清单)\s*[：:]')


def clause_bounds(text, start, end):
    delimiters = tuple('。；;\n，,') + ('但是','不过','然而','但')
    left = 0; right = len(text)
    for delimiter in delimiters:
        index = text.rfind(delimiter,0,start)
        if index >= 0:
            left = max(left,index+len(delimiter))
        index = text.find(delimiter,end)
        if index >= 0:
            right = min(right,index)
    return left,right


def longest_spans(text, patterns):
    """同跨度优先最长概念，不把Spring Boot中的Spring另造一个独立要求。"""
    candidates = [(m.start(),m.end(),key,m.group()) for key,pattern in patterns.items() for m in pattern.finditer(text)]
    kept=[]
    for item in sorted(candidates,key=lambda value:(-(value[1]-value[0]),value[0],value[2])):
        if not any(item[0]<old[1] and old[0]<item[1] for old in kept):
            kept.append(item)
    return sorted(kept,key=lambda value:(value[0],value[1],value[2]))


def section_heading(text, position):
    matches=list(HEADINGS.finditer(text,0,position))
    return matches[-1].group(1) if matches else ''


_ACTOR_ACTION = re.compile(
    r'(?P<actor>团队成员|项目团队|研发团队|技术团队|后端团队|数据团队|其他成员|'
    r'本人|我(?!们)|个人|独立|团队|同事|队友|候选人|导师|老师|同学|组员|他们|她们|我们|他|她|公司|部门|'
    r'(?:系统|平台|服务|项目)(?=\s*(?:使用|采用|运用|利用|借助|通过)))'
    r'\s*(?:曾经|目前|当前|曾|已经|已|也|则|独立|主要|仅|只|共同|一起|直接|实际|正在|熟练|还|并){0,4}\s*'
    r'(?P<action>不会|未使用|不熟悉|不掌握|未掌握|不了解|负责|使用|运用|采用|利用|用过|借助|通过|'
    r'完成|实现|搭建|编写|构建|开发|掌握|熟悉|精通|了解|学习|计划|准备|希望|打算|'
    r'进行|执行|开展|参与|协调|维护|分析|跟踪|整理|设计|训练|部署|撰写)'
)


def _resume_actor_scope(text,start,end):
    """同一句内按最近的显式动作主体归因；逗号省略主体可继承，句号换行不继承。

    不把“团队用了X，我负责Y”中的X归给本人。团队、系统和不明确的集合主体
    只支持背景提及；只有明确属于本人的动作才可能升级为个人实践。
    """
    left,right=clause_bounds(text,start,end)
    sentence_start=max(text.rfind(mark,0,start)+1 for mark in '。；;\n')
    sentence_end=len(text)
    for mark in '。；;\n':
        boundary=text.find(mark,end)
        if boundary>=0:sentence_end=min(sentence_end,boundary)
    anchors=list(_ACTOR_ACTION.finditer(text,sentence_start,sentence_end))
    previous=[match for match in anchors if match.start()<=start]
    owner=previous[-1] if previous else None
    # “Python由同事使用”不因技能词位于主体前就默认成为本人动作。
    following=next((match for match in anchors if match.start()>=end),None)
    if following is not None and re.fullmatch(r'\s*由\s*',text[end:following.start()]):
        owner=following
    if owner is not None and owner.start()<=start:
        left=max(left,owner.start())
    following=next((match for match in anchors if match.start()>=end and match is not owner),None)
    if following is not None and owner is not None and owner.start()<=start:
        right=min(right,following.start())
    actor='background' if owner is not None and owner.group('actor') not in {'我','本人','个人','独立'} else 'self'
    evidence=None if owner is None else {'field':'resume','start':owner.start(),'end':owner.end(),'quote':owner.group()}
    return left,right,actor,evidence


def classify_mention(text,start,end,field,kind='skill'):
    is_job=field in {'requirements','description'}
    if is_job:
        left,right=clause_bounds(text,start,end);actor='job';actor_evidence=None
    else:
        left,right,actor,actor_evidence=_resume_actor_scope(text,start,end)
    clause=text[left:right]; prefix=text[left:start];suffix=text[end:right]
    heading=section_heading(text,start)
    double_negative=bool(re.search(r'并非不|不是不|不能说不|并不是不',prefix))
    negated=bool(re.search(NEGATIVE,prefix[-28:]) or re.match(r'\s*(?:还|暂时)?(?:不会|不熟悉|不掌握|未掌握|不了解)',suffix))
    waived=is_job and bool(re.search(NOT_REQUIRED,prefix[-30:]) or re.match(r'\s*(?:经验)?(?:不要求|非必需|无需)',suffix))
    preferred=bool(re.search(PREFERRED,prefix[-30:]) or re.search(PREFERRED,suffix[:24]) or heading in {'加分项','优先条件','优先项'})
    future=bool(re.search(r'希望|计划|准备|想要|打算|将要|拟学习|待学习',prefix[-28:]))
    recruitment_background=kind=='skill' and bool(re.search(r'(?:招聘|招募|猎头|人才搜寻|候选人筛选)',clause))
    direct_use=bool(re.search(r'(?:我|本人|独立)?(?:使用过?|运用|采用|利用|用过?|借助|通过)[^。；;，,\n]{0,24}$',prefix))
    copied_role=(not is_job and heading in {'任职要求','岗位要求'}) or bool(re.search(r'招聘广告|招聘信息|岗位职责|阅读.{0,8}(?:要求|文档)',clause))
    background=(not is_job) and (copied_role or actor=='background' or (recruitment_background and not direct_use))
    if double_negative:
        level='提及'; claim='ambiguous'; polarity='unknown'
    elif background:
        level='提及';claim='background';polarity='unknown'
    elif negated or waived:
        level='否定';claim='explicit_negative';polarity='negative'
    elif future:
        level='提及';claim='planned';polarity='unknown'
    elif re.search(r'仅了解|了解|入门|学习中',prefix[-24:]):
        level='了解';claim='self_report';polarity='positive'
    elif direct_use or (kind=='task' and (re.search(r'完成|负责|执行|开展|独立|参与|进行了',prefix[-24:]) or re.match(r'编写|维护|分析|梳理|清洗|执行|协调|排查|部署|处理|撰写|设计|跟踪|筛选',text[start:end]))) or re.search(r'开发了|实现了|搭建了|编写了|构建了',prefix[-24:]):
        level='实践';claim='self_reported_action';polarity='positive'
    elif re.search(r'精通',prefix[-24:]):
        level='精通';claim='self_report';polarity='positive'
    elif re.search(r'熟悉|掌握|熟练',prefix[-24:]):
        level='熟悉';claim='self_report';polarity='positive'
    else:
        level='提及';claim='mention';polarity='unknown'
    modality=('negated' if negated or waived else 'preferred' if preferred else 'required' if re.search(REQUIRED,prefix[-40:]) or re.match(r'\s*(?:必须|必需|必备)',suffix) else 'unknown') if is_job else 'unknown'
    if double_negative:
        modality='unknown'
    return {'level':level,'weight':LEVELS[level],'preferred':modality=='preferred' if is_job else preferred,
            'modality':modality,'claim_type':('job_text' if is_job else claim),'polarity':polarity,
            'actor':'job' if is_job else ('background' if background else 'unknown' if double_negative else 'self'),
            'parse_status':'unknown' if double_negative else 'known','context_start':left,'context_end':right,
            'actor_evidence':actor_evidence}


def mentions(text,patterns,field='resume',kind='skill',names=None):
    output=[]
    for start,end,key,quote in longest_spans(text,patterns):
        info=classify_mention(text,start,end,field,kind)
        output.append({'skill':key,'key':key,'type':kind,'text':names.get(key,key) if names else key,
                       'field':field,'start':start,'end':end,'quote':quote,**info})
    return output


def compact_mentions(items,is_job=False):
    by_key={}
    for item in items:
        by_key.setdefault(item['key'],[]).append(item)
    output={}
    for key,occurrences in by_key.items():
        chosen=max(occurrences,key=lambda item:(item['weight'],item['start']))
        negatives=[item for item in occurrences if item['polarity']=='negative']
        positives=[item for item in occurrences if item['polarity']=='positive' and item['weight']>=.7]
        chosen=dict(chosen)
        if not is_job and negatives and positives:
            chosen.update(weight=min(chosen['weight'],.4),parse_status='unknown',polarity='unknown',conflict=True)
        chosen['mentions']=[dict(item) for item in occurrences]
        output[key]=chosen
    return output


def leaf(item):
    evidence={key:item[key] for key in ('field','start','end','quote')}
    return {'type':item['type'],'key':item['key'],'text':item.get('text',item['key']),
            'required':item['modality']=='required','modality':item['modality'],
            'parse_status':item['parse_status'],'evidence':evidence,'level':item['level'],'claim':item}


def ast_leaves(ast):
    if 'type' in ast:
        return [ast]
    return [item for child in ast.get('children',[]) for item in ast_leaves(child)]


def node(op,children,unknown=False):
    if len(children)==1 and not unknown:
        return children[0]
    modalities={child.get('modality','unknown') for child in children}
    return {'op':op,'children':children,'modality':next(iter(modalities)) if len(modalities)==1 else 'mixed',
            'parse_status':'unknown' if unknown or any(child.get('parse_status')=='unknown' for child in children) else 'known'}


def parse_expression(text,items):
    """AND高于OR；保留括号。模糊连接不强制解读为AND。"""
    if not items:
        return None
    tokens=[]; previous=0; ambiguous=False
    for item in items:
        gap=text[previous:item['start']]
        connectors=[]
        for match in re.finditer(r'[()（）]|或者|或|以及|并且|同时|和|及|且|、|[,，]|/',gap):
            value=match.group()
            connectors.append('(' if value in '(（' else ')' if value in ')）' else 'or' if value in {'或','或者'} else 'and')
            if value=='/':
                ambiguous=True
        # 逗号后跟“或者”属于同一个OR连接，不能插入空操作数。
        reduced=[]
        for connector in connectors:
            if connector in {'and','or'} and reduced and reduced[-1] in {'and','or'}:
                reduced[-1]='or' if connector=='or' or reduced[-1]=='or' else 'and'
            else:
                reduced.append(connector)
        if tokens and reduced and reduced[0]=='(' and (isinstance(tokens[-1],dict) or tokens[-1] not in {'and','or','('}):
            reduced.insert(0,'and');ambiguous=True
        if tokens and not reduced:
            reduced=['and'];ambiguous=True
        if not tokens:
            reduced=[value for value in reduced if value=='(']
        tokens.extend(reduced);tokens.append(leaf(item));previous=item['end']
    suffix=text[previous:]
    tokens.extend(')' for value in suffix if value in ')）')
    # 顿号/斜杠列表后明确“至少一种”才转为OR，不能把“和”的组合拆开。
    first,last=items[0]['start'],items[-1]['end']
    between=text[first:last]
    quantifier=bool(re.search(r'之一|一种|任一|任意|至少一|任选',suffix))
    if quantifier and not re.search(r'和|并且|同时|以及|[()（）]',between):
        return node('any',[leaf(item) for item in items])
    position=0
    def atom(depth=0):
        nonlocal position,ambiguous
        if depth>16 or position>=len(tokens):
            raise ValueError('表达式不完整')
        value=tokens[position];position+=1
        if isinstance(value,dict):
            return value
        if value=='(':
            value=parse_or(depth+1)
            if position>=len(tokens) or tokens[position]!=')':
                raise ValueError('括号不匹配')
            position+=1;return value
        raise ValueError('连接词无操作数')
    def parse_and(depth=0):
        nonlocal position
        children=[atom(depth)]
        while position<len(tokens) and tokens[position]=='and':
            position+=1;children.append(atom(depth))
        return node('all',children)
    def parse_or(depth=0):
        nonlocal position
        children=[parse_and(depth)]
        while position<len(tokens) and tokens[position]=='or':
            position+=1;children.append(parse_and(depth))
        return node('any',children)
    try:
        result=parse_or()
        if position!=len(tokens):
            raise ValueError('无法完全解析')
    except (ValueError,RecursionError):
        result=node('all',[leaf(item) for item in items],True)
        ambiguous=True
    if ambiguous:
        if 'type' in result:
            result=node('all',[result],True)
        result['parse_status']='unknown'
    return result


def requirement_asts(text,items):
    roots=[]
    # 顶层句号/分号/换行是独立条件；括号内不拆，逗号交给表达式解析。
    segments=[];start=0;depth=0
    for index,char in enumerate(text):
        if char in '(（':depth+=1
        elif char in ')）':depth=max(0,depth-1)
        elif char in '。；;\n' and depth==0:
            segments.append((start,index));start=index+1
        elif char in '，,' and depth==0 and re.search(r'或|至少一|任意|任一|一种|之一',text[start:index]) and not re.match(r'\s*(?:或者|或)',text[index+1:]):
            # “A或B，使用C”是OR组加另一项；“A和B，或者C和D”仍是两分支。
            segments.append((start,index));start=index+1
    segments.append((start,len(text)))
    for start,end in segments:
        candidates=[item for item in items if start<=item['start'] and item['end']<=end and item['modality']!='negated']
        if not candidates:
            continue
        # 转换局部坐标仅用于文法tokenization；叶子引用始终恢复原字段位置。
        local=[{**item,'start':item['start']-start,'end':item['end']-start} for item in candidates]
        ast=parse_expression(text[start:end],local)
        for item in ast_leaves(ast):
            item['evidence']['start']+=start;item['evidence']['end']+=start
            item['claim']['start']+=start;item['claim']['end']+=start
        ast['scope_evidence']={'field':candidates[0]['field'],'start':start,'end':end,'quote':text[start:end]}
        roots.append(ast)
    return roots


def education_details(text,levels,is_job=False):
    items=[];unlimited=False
    for match in re.finditer('|'.join(sorted(levels,key=len,reverse=True)),text):
        left,right=clause_bounds(text,match.start(),match.end())
        prefix=text[left:match.start()];suffix=text[match.end():right]
        negated=bool(re.search(r'不接受|不考虑|排除|谢绝|不是|并非',prefix[-12:]) or re.match(r'.{0,5}(?:不符合|不接受|勿投|不要|不考虑)',suffix))
        optional=bool(re.search(r'优先|可放宽|优秀者|加分',prefix[-20:]) or re.search(r'优先|更佳|可放宽',suffix[:10]))
        background=not is_job and (section_heading(text,match.start()) in {'任职要求','岗位要求'} or bool(re.search(r'目标岗位|招聘要求|职位要求|招聘广告',prefix)))
        in_progress=not is_job and bool(re.search(r'在读|攻读|预计|计划|拟考|报考',prefix[-12:]+suffix[:12]))
        items.append({'level':levels[match.group()],'degree':match.group(),'negated':negated,'optional':optional,'in_progress':in_progress,'background':background,
                      'evidence':{'field':'requirements' if is_job else 'resume','start':match.start(),'end':match.end(),'quote':match.group()}})
    if is_job:
        unlimited=bool(re.search(r'学历(?:不限|不要求)|不限学历|不要求学历',text))
    valid=[item['level'] for item in items if not item['negated'] and not item['in_progress'] and not item['background'] and (not is_job or not item['optional'])]
    level=(-1 if unlimited and not valid else min(valid) if is_job and valid else max(valid) if valid else None)
    return {'level':level,'mentions':items,'status':'known' if level is not None else 'unknown',
            'full_time_required':is_job and bool(re.search(r'(?<!非)全日制',text)) and not bool(re.search(r'全日制.{0,8}(?:不限|不要求|优先|更佳)',text)),
            'conflict':unlimited and bool(valid)}


def experience_details(text,patterns,is_job=False):
    items=[];total=[];specialized=[];unlimited=False;upper=None
    number=re.compile(r'(\d+(?:\.\d+)?)\s*(?:[-~至—–]\s*(\d+(?:\.\d+)?))?\s*年')
    matches=list(number.finditer(text))
    for index,match in enumerate(matches):
        lower=float(match[1]);high=float(match[2]) if match[2] else None
        if lower>60 or (high is not None and high>60):
            continue
        left,right=clause_bounds(text,match.start(),match.end())
        before=text[max(left,matches[index-1].end() if index else 0):match.start()]
        after=text[match.end():min(right,matches[index+1].start() if index+1<len(matches) else len(text))]
        if re.match(r'\s*\d+月',after) or re.search(r'成立|创立|营收|毕业于',before+after):
            continue
        if not is_job and (section_heading(text,match.start()) in {'任职要求','岗位要求'} or re.search(r'目标岗位|招聘要求|职位要求|招聘广告',before)):
            continue
        negated=bool(re.search(r'没有|不足|不到|未满|不够',before[-10:]))
        optional=bool(re.search(r'优先|加分|可放宽',after[:16]) or re.search(r'优先|加分|可放宽',before[-16:]))
        explicit_total=bool(re.search(r'总(?:共|计|工作)?|累计|总体',before[-14:]) or re.match(r'\s*(?:以上|以内|以下|的)?\s*(?:工作|从业|相关工作)?经验',after))
        nearby=before[-24:]+after[:24]
        scopes=sorted({key for key,pattern in patterns.items() if pattern.search(nearby)})
        shorthand=is_job and bool(re.match(r'\s*(?:以上|以内|以下)?\s*(?:本科|大专|硕士|博士|高中|中专|学历|$)',after))
        generic=bool(re.match(r'\s*(?:以上|以内|以下|的)?\s*(?:开发|行业|相关|相关开发)经验',after))
        if not (is_job or '经验' in after[:20] or '经验' in before[-12:] or scopes or explicit_total):
            continue
        if negated:
            scope='unknown'
        elif explicit_total or shorthand or (generic and not scopes):
            scope='total'
        else:
            scope='specialized' if scopes or re.search(r'经验|从事|开发',nearby) else 'unknown'
        maximum=bool(re.match(r'\s*(?:以内|以下)',after))
        item={'minimum':0.0 if maximum else lower,'maximum':lower if maximum else high,'upper_is_hard':maximum,
              'scope':scope,'skills':scopes,'optional':optional,'negated':negated,
              'evidence':{'field':'requirements' if is_job else 'resume','start':match.start(),'end':match.end(),'quote':match.group()}}
        items.append(item)
        if scope=='total' and not optional and not negated:
            total.append(item)
        elif scope=='specialized':
            specialized.append(item)
    if is_job:
        for clause in re.split(r'[。；;\n，,]',text):
            if re.search(r'经验不限|不限经验|无需经验|无经验|在校|应届',clause) and not re.search(r'勿投|勿扰|不招|不接受|不要|不考虑|谢绝|拒绝',clause):
                unlimited=True
    known=[item['minimum'] for item in total]
    value=max(known) if known else 0.0 if unlimited else None
    profile_conflict=not is_job and len(set(known))>1
    if not is_job and total and (any(item['maximum'] is not None for item in total) or profile_conflict):
        value=None
    hard_upper=[item['maximum'] for item in total if item['upper_is_hard']]
    upper=min(hard_upper) if hard_upper else None
    return {'total_years':value,'total_maximum':upper,'specialized':specialized,'mentions':items,
            'status':'known' if value is not None else 'unknown','unlimited':unlimited,'conflict':profile_conflict or (unlimited and any(v>0 for v in known))}


def experience_blocks(text):
    """只划分有原文跨度的经历块；不从重叠日期或项目时长累加工作年资。"""
    blocks=[]
    for match in re.finditer(r'[^\n]+',text):
        raw=match.group();stripped=raw.strip()
        if not stripped:
            continue
        start=match.start()+len(raw)-len(raw.lstrip());end=start+len(stripped)
        role='project' if re.search(r'项目|课程|练习',stripped) else 'employment' if re.search(r'工作经历|任职|就职|公司',stripped) else 'education' if re.search(r'学历|毕业|在读|本科|硕士|博士',stripped) else 'unclassified'
        blocks.append({'kind':role,'field':'resume','start':start,'end':end,'quote':stripped,'status':'text_only_unverified'})
    return blocks
