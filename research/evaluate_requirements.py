"""需求片段教师相对评测；只读冻结材料，不调用接口，不输出岗位原文。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

from job_agent.domain import CONTACT,PARSER_VERSION,PATTERNS,requirement_groups,skill_evidence
from job_agent.retrieval_contract import contract_info,stable_hash
from research.common import private_path,write_json,write_jsonl

VERSION='requirement-teacher-relative-v1'
MODALITIES={'required','preferred','negated','unknown'}


class InvalidLabel(ValueError):
    """只返回稳定错误码，不回显原文。"""


def normalized_name(name):
    if not isinstance(name,str) or not name.strip():raise InvalidLabel('empty_skill')
    normalized=unicodedata.normalize('NFKC',name).strip()
    aliases=[key for key,pattern in PATTERNS.items() if pattern.fullmatch(normalized)]
    if len(aliases)>1:raise InvalidLabel('ambiguous_alias')
    return (aliases[0],True) if aliases else (normalized.casefold(),False)


def checked_teacher(groups,fragment):
    """校验逐字引用、重复跨度、算子元数及递归；不验证教师语义是否真实正确。"""
    if not isinstance(groups,list):raise InvalidLabel('groups_not_list')
    used=set();nodes=0
    def visit(group,depth=0):
        nonlocal nodes
        nodes+=1
        if depth>12 or nodes>256:raise InvalidLabel('tree_too_large')
        if not isinstance(group,dict):raise InvalidLabel('group_not_object')
        logic,modality=group.get('logic'),group.get('modality')
        if logic not in {'single','all','any','unknown'}:raise InvalidLabel('invalid_logic')
        if modality not in MODALITIES:raise InvalidLabel('invalid_modality')
        skills,children=group.get('skills',[]),group.get('children',[])
        if not isinstance(skills,list) or not isinstance(children,list):raise InvalidLabel('operands_not_list')
        count=len(skills)+len(children)
        if count==0 or logic=='single' and count!=1 or logic in {'all','any'} and count<2:raise InvalidLabel('invalid_arity')
        leaves=[]
        for item in skills:
            if not isinstance(item,dict):raise InvalidLabel('skill_not_object')
            name,covered=normalized_name(item.get('skill'));quote=item.get('quote')
            if not isinstance(quote,str) or not quote:raise InvalidLabel('empty_quote')
            locations=[m.start() for m in re.finditer('(?='+re.escape(quote)+')',fragment)]
            if not locations:raise InvalidLabel('quote_not_literal')
            occurrence=item.get('occurrence')
            if occurrence is None:
                if len(locations)!=1:raise InvalidLabel('ambiguous_quote_occurrence')
                occurrence=0
            if type(occurrence) is not int or not 0<=occurrence<len(locations):raise InvalidLabel('invalid_occurrence')
            start=locations[occurrence];end=start+len(quote)
            identity=(name,start,end)
            if identity in used:raise InvalidLabel('duplicate_leaf_span')
            used.add(identity)
            leaves.append({'name':name,'covered':covered,'modality':modality,'start':start,'end':end})
        return {'logic':logic,'modality':modality,'children':leaves+[visit(child,depth+1) for child in children]}
    return [visit(group) for group in groups]


def leaves(node):
    if 'name' in node:return [node]
    return [leaf for child in node['children'] for leaf in leaves(child)]


def tree_unknown(node):
    return 'name' not in node and (node['logic']=='unknown' or any(tree_unknown(child) for child in node['children']))


def evaluate_logic(node,assignment):
    if 'name' in node:return assignment[node['name']]
    if node['logic']=='unknown':raise ValueError('未知逻辑不可硬转AND')
    values=[evaluate_logic(child,assignment) for child in node['children']]
    return any(values) if node['logic']=='any' else all(values)


def equivalent_logic(one,two,max_leaves=12):
    names=sorted({leaf['name'] for leaf in leaves(one)})
    if set(names)!={leaf['name'] for leaf in leaves(two)}:return None,'different_leaf_set'
    if tree_unknown(one) or tree_unknown(two):return None,'unknown_logic'
    if len(names)>max_leaves:return None,'too_many_leaves'
    for values in itertools.product((False,True),repeat=len(names)):
        assignment=dict(zip(names,values))
        if evaluate_logic(one,assignment)!=evaluate_logic(two,assignment):return False,'comparable'
    return True,'comparable'


def parser_output(fragment,field):
    evidence=skill_evidence(fragment,field)
    raw=requirement_groups(fragment,evidence,field)
    def convert(node):
        if 'type' in node:
            cite=node['evidence']
            if fragment[cite['start']:cite['end']]!=cite['quote']:raise ValueError('冻结解析器引用异常')
            return {'name':node['key'],'covered':True,'modality':node['modality'],'start':cite['start'],'end':cite['end']}
        return {'logic':'unknown' if node.get('parse_status')=='unknown' else node['op'],
                'modality':node.get('modality','unknown'),'children':[convert(child) for child in node['children']]}
    roots=[]
    for group in raw:
        value=convert(group['ast'])
        if 'name' in value:value={'logic':'single','modality':value['modality'],'children':[value]}
        roots.append(value)
    # 实体指标包括显式否定提及；要求树本身按现有实现排除否定要求。
    mentions=[{'name':name,'modality':mention['modality']} for name,item in evidence.items() for mention in item['mentions']]
    return roots,mentions


def compare_one(teacher,predicted,mentions):
    gold_leaves=[leaf for root in teacher for leaf in leaves(root)]
    gold_names={leaf['name'] for leaf in gold_leaves};pred_names={leaf['name'] for leaf in mentions}
    covered={leaf['name'] for leaf in gold_leaves if leaf['covered']}
    entity={'tp':len(gold_names&pred_names),'fp':len(pred_names-gold_names),'fn':len(gold_names-pred_names),
            'teacher_entities':len(gold_names),'teacher_dictionary_entities':len(covered),'parser_entities':len(pred_names),
            'dictionary_tp':len(covered&pred_names),'dictionary_fp':len(pred_names-covered),'dictionary_fn':len(covered-pred_names),
            'teacher_out_of_dictionary_entities':len(gold_names-covered)}
    counts=Counter();pairs=[]
    def group_map(roots):
        result={}
        for root in roots:result.setdefault(frozenset(leaf['name'] for leaf in leaves(root)),[]).append(root)
        return result
    gold_map,pred_map=group_map(teacher),group_map(predicted)
    counts['teacher_groups']=len(teacher);counts['parser_groups']=len(predicted)
    counts['teacher_unknown_logic']=sum(tree_unknown(root) for root in teacher)
    counts['parser_unknown_logic']=sum(tree_unknown(root) for root in predicted)
    counts['teacher_unknown_modality_leaves']=sum(leaf['modality']=='unknown' for leaf in gold_leaves)
    counts['teacher_leaves']=len(gold_leaves)
    counts['parser_unknown_modality_mentions']=sum(leaf['modality']=='unknown' for leaf in mentions)
    counts['parser_mentions']=len(mentions)
    for signature,gs in gold_map.items():
        ps=pred_map.get(signature,[])
        if not ps:counts['unmatched_teacher_groups']+=len(gs);continue
        if len(gs)!=1 or len(ps)!=1:counts['ambiguous_same_leaf_groups']+=len(gs);continue
        one,two=gs[0],ps[0];counts['same_leaf_set_groups']+=1
        equal,reason=equivalent_logic(one,two)
        counts['logic_'+reason]+=1
        if equal is not None:counts['logic_equal']+=int(equal)
        # 模态只在相同叶集合上按(实体,模态)多重集比较，未知单独记录。
        a=Counter((leaf['name'],leaf['modality']) for leaf in leaves(one))
        b=Counter((leaf['name'],leaf['modality']) for leaf in leaves(two))
        counts['modality_same_leaf_comparable']+=1;counts['modality_same_leaf_equal']+=int(a==b)
        pairs.append({'logic_equal':equal,'logic_reason':reason,'modality_equal':a==b,'leaves':len(signature)})
    return {'entities':entity,'groups':dict(counts),'group_comparisons':pairs}


def ratio(n,d):return n/d if d else None


def aggregate(cases):
    entity=Counter();groups=Counter()
    for case in cases:
        entity.update(case['entities']);groups.update(case['groups'])
    tp,fp,fn=(entity[key] for key in ('tp','fp','fn'))
    dtp,dfp,dfn=(entity[key] for key in ('dictionary_tp','dictionary_fp','dictionary_fn'))
    return {'entities':{**dict(entity),'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'f1':ratio(2*tp,2*tp+fp+fn),
                        'teacher_dictionary_coverage':ratio(entity['teacher_dictionary_entities'],entity['teacher_entities']),
                        'dictionary_only_precision':ratio(dtp,dtp+dfp),'dictionary_only_recall':ratio(dtp,dtp+dfn),
                        'dictionary_only_f1':ratio(2*dtp,2*dtp+dfp+dfn)},
            'groups':{**dict(groups),'logic_agreement_on_comparable_same_leaf_groups':ratio(groups['logic_equal'],groups['logic_comparable']),
                      'modality_agreement_on_same_leaf_groups':ratio(groups['modality_same_leaf_equal'],groups['modality_same_leaf_comparable']),
                      'teacher_unknown_logic_rate':ratio(groups['teacher_unknown_logic'],groups['teacher_groups']),
                      'parser_unknown_logic_rate':ratio(groups['parser_unknown_logic'],groups['parser_groups']),
                      'teacher_unknown_modality_rate':ratio(groups['teacher_unknown_modality_leaves'],groups['teacher_leaves']),
                      'parser_unknown_modality_rate':ratio(groups['parser_unknown_modality_mentions'],groups['parser_mentions'])}}


def file_json(path,hashes):
    data=Path(path).read_bytes();hashes[str(Path(path).resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)


def selected_records(dataset,ids,hashes):
    manifest=file_json(dataset/'manifest.json',hashes);records={};hasher=hashlib.sha256()
    with (dataset/'jobs.jsonl').open('rb') as stream:
        for line in stream:
            hasher.update(line)
            if not line.strip():continue
            row=json.loads(line)
            if row['job_id'] in ids:
                if row['job_id'] in records:raise ValueError('岗位ID重复')
                records[row['job_id']]=row
    actual=hasher.hexdigest();expected=manifest['files']['jobs.jsonl']['sha256']
    if actual!=expected or set(records)!=ids:raise ValueError('数据摘要或所需岗位不完整')
    hashes[str((dataset/'jobs.jsonl').resolve())]=actual
    return manifest,records


def run(study,dataset,annotation,output,allow_partial=False):
    study,dataset,annotation=map(lambda p:Path(p).resolve(),(study,dataset,annotation));output=private_path(output)
    if output.exists():raise FileExistsError('评测输出必须是新目录，禁止覆盖旧版本')
    hashes={};study_manifest=file_json(study/'manifest.json',hashes)
    tasks_bytes=(study/'extraction_tasks.jsonl').read_bytes();tasks=[json.loads(line) for line in tasks_bytes.splitlines() if line.strip()]
    task_sha=hashlib.sha256(tasks_bytes).hexdigest();hashes[str(study/'extraction_tasks.jsonl')]=task_sha
    if study_manifest['files']['extraction_tasks.jsonl']!=task_sha:raise ValueError('任务文件摘要不一致')
    task_map={t['task_id']:t for t in tasks}
    if len(task_map)!=len(tasks) or any(t['input_hash']!=stable_hash(t['material']) or t['kind']!='requirement_extraction' for t in tasks):raise ValueError('任务版本或ID不合法')
    annotation_manifest=file_json(annotation/'manifest.json',hashes)
    if annotation_manifest.get('tasks_hash')!=stable_hash(tasks):raise ValueError('接口标注与任务版本不一致')
    ids={task['job_id'] for task in tasks}
    dataset_manifest,current=selected_records(dataset,ids,hashes)
    if dataset_manifest['config'].get('parser_version')!=PARSER_VERSION or dataset_manifest['config'].get('retrieval_contract',{}).get('hash')!=contract_info()['hash']:
        raise ValueError('当前解析代码不是指定冻结版本，拒绝使用新规则冒充r4')
    _,original=selected_records(Path(study_manifest['dataset']),ids,hashes)
    raw_fields=['job_version_id','snapshot','job_family_id','split','title','company','requirements','description','salary_raw','address']
    for task in tasks:
        old,new=original[task['job_id']],current[task['job_id']]
        if stable_hash(old)!=task['source_job_sha256'] or any(old[key]!=new[key] for key in raw_fields) or task['split']!=new['split']:raise ValueError('旧教师来源与当前岗位正文或分区不一致')
        field=task['material']['field'];fragment=task['material']['fragment']
        if field not in {'requirements','description'} or not fragment:raise ValueError('片段字段非法')
        sanitized=CONTACT.sub('[联系方式已删除]',new[field])
        if new['company']:sanitized=sanitized.replace(new['company'],'某企业')
        if fragment not in sanitized:raise ValueError('任务片段无法逐字回溯已脱敏的岗位原字段')
    labels={};pending_files=0
    for path in sorted(annotation.glob('batch-*.json')):
        batch=file_json(path,hashes)
        if batch.get('status')!='completed':pending_files+=1;continue
        for row in batch.get('rows',[]):
            key=row.get('task_id')
            if key not in task_map or key in labels:raise ValueError('接口结果存在池外或重复任务')
            labels[key]=row
    cases=[];rejected=Counter();missing=0
    for task in tasks:
        identifier=task['task_id'];row=labels.get(identifier)
        base={'task_id':identifier,'split':task['split'],'input_hash':task['input_hash']}
        if row is None:missing+=1;cases.append({**base,'status':'pending'});continue
        try:
            if any(row.get(key)!=task[key] for key in ['kind','job_id','split','input_hash','source_job_sha256']):raise InvalidLabel('row_provenance_mismatch')
            if row.get('label_source')!='llm_reviewed' or row.get('human_reviewed') is not False or row.get('eligible_for_production') is not False:raise InvalidLabel('teacher_provenance_invalid')
            reviews=row.get('model_reviews',[])
            if len(reviews)<2 or len({v.get('channel') for v in reviews})!=len(reviews) or any(v.get('input_hash')!=task['input_hash'] for v in reviews):raise InvalidLabel('review_inputs_invalid')
            if row.get('rule_validation',{}).get('passed') is not True:raise InvalidLabel('upstream_rule_failed')
            teacher=checked_teacher(row.get('groups'),task['material']['fragment'])
        except InvalidLabel as error:
            rejected[str(error)]+=1;cases.append({**base,'status':'strict_rejected','reason':str(error)});continue
        predicted,mentions=parser_output(task['material']['fragment'],task['material']['field'])
        cases.append({**base,'status':'evaluated',**compare_one(teacher,predicted,mentions)})
    good=[row for row in cases if row['status']=='evaluated']
    complete=len(good)==len(tasks) and annotation_manifest.get('status')=='completed'
    if not complete and not allow_partial:raise ValueError('标签未完整严格通过；仅显式--allow-partial可生成进度报告')
    report={'schema':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_teacher_relative' if complete else 'progress_only',
            'teacher_relative':True,'human_gold_count':0,'parser_version':PARSER_VERSION,'tasks_expected':len(tasks),'teacher_rows_present':len(labels),
            'tasks_evaluated':len(good),'tasks_pending':missing,'strict_rejected':sum(rejected.values()),'rejection_reasons':dict(rejected),
            'annotation_manifest_status':annotation_manifest.get('status'),'metrics':aggregate(good),
            'by_split':{split:aggregate([row for row in good if row['split']==split]) for split in sorted({row['split'] for row in cases})},
            'input_files_sha256':hashes,'notes':[
                '这是同一冻结片段上的教师相对比较；双通道模型不是独立人工，不能解释为真实抽取准确率。',
                '仅严格通过的已完成任务进入当前指标；缺失、未通过和全量分母同时报告，进度结果存在完成顺序偏差。',
                '实体按规范名称集合计micro F1；只将词典完整匹配的别名归一，开放词表不强行拆分。逐字引用校验不证明引用支持实体语义。',
                '开放实体F1保留全部教师标签；封闭词典F1仅以落入93词典的教师实体为参照，不能替代开放覆盖率。教师skills含专业、业务任务等广义要求，不可直接称技术技能准确率。',
                '逻辑仅比较唯一配对的相同叶集合组，使用最多12个实体的真值表，保留嵌套语义；未知或过大组不强制算错，也不进入正确分母。',
                '模态在同叶集合组比较实体-模态多重集；未知率单报。需求树排除明确否定要求，但实体统计仍包含否定提及。',
                '本次只比较教师skills通道与冻结技能解析分支，不把任务词表节点并入技能召回；词典未覆盖不等于岗位无要求。',
                '输出不包含岗位原文、公司名、联系方式、原始引用或模型解释。']}
    output.mkdir(parents=True,mode=0o700)
    write_json(output/'report.json',report);write_jsonl(output/'cases.jsonl',cases)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','dataset','annotation','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    report=run(args.study,args.dataset,args.annotation,args.output,args.allow_partial)
    print(json.dumps({key:report[key] for key in ['status','tasks_expected','teacher_rows_present','tasks_evaluated','tasks_pending','strict_rejected','parser_version']},ensure_ascii=False))


if __name__=='__main__':main()
