"""抽取评测的分母、严格引用、逻辑作用域和版本测试，全部使用虚构材料。"""
import importlib.util
import json
from pathlib import Path
import hashlib

import pytest

MODULE=Path(__file__).parent/'research/evaluate_requirements.py'
if MODULE.exists():
    spec=importlib.util.spec_from_file_location('evaluate_requirements_audit',MODULE)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
else:
    import research.evaluate_requirements as ev


def leaf(name):return {'skill':name,'quote':name,**({'occurrence':0} if name=='SQL' else {})}


def group(logic='any',skills=None,children=None,modality='required'):
    return {'logic':logic,'modality':modality,'skills':skills or [],'children':children or []}


@pytest.mark.parametrize('groups,fragment,code',[
    ([group('single',[leaf('Python'),leaf('SQL')])],'Python和SQL','invalid_arity'),
    ([group('all',[leaf('Python')])],'Python','invalid_arity'),
    ([group('unknown')],'Python','invalid_arity'),
    ([group('single',[leaf('Java')])],'Python','quote_not_literal'),
    ([group('single',[leaf('Python')])],'Python或者Python','ambiguous_quote_occurrence'),
    ([group('all',[leaf('Python'),leaf('Python')])],'Python','duplicate_leaf_span'),
    ([{'logic':'all','modality':'required','children':'bad'}],'Python','operands_not_list'),
])
def test_strict_teacher_tree_rejects_weak_validator_loopholes(groups,fragment,code):
    with pytest.raises(ev.InvalidLabel,match=code):ev.checked_teacher(groups,fragment)


def test_nested_boolean_scope_not_only_same_leaf_bag():
    fragment='Python Java SQL MySQL'
    one=group('all',children=[group('any',[leaf('Python'),leaf('Java')]),group('any',[leaf('SQL'),leaf('MySQL')])])
    two=group('all',children=[group('any',[leaf('Python'),leaf('SQL')]),group('any',[leaf('Java'),leaf('MySQL')])])
    a,b=(ev.checked_teacher([item],fragment)[0] for item in (one,two))
    assert ev.equivalent_logic(a,b)==(False,'comparable')
    assert ev.equivalent_logic(a,a)==(True,'comparable')
    b['logic']='unknown';assert ev.equivalent_logic(a,b)==(None,'unknown_logic')


def test_alias_coverage_is_not_confused_with_no_job_requirement():
    gold=ev.checked_teacher([group('all',[leaf('Python'),leaf('陌生业务任务')])],'Python和陌生业务任务')
    predicted,mentions=ev.parser_output('熟悉Python。','description')
    result=ev.aggregate([ev.compare_one(gold,predicted,mentions)])
    assert result['entities']['teacher_dictionary_coverage']==.5
    assert result['entities']['recall']==.5
    assert result['entities']['dictionary_only_recall']==1
    assert result['entities']['dictionary_only_f1']==1
    assert result['groups']['logic_agreement_on_comparable_same_leaf_groups'] is None


def test_same_leaf_group_modality_measured_separately_from_logic():
    gold=ev.checked_teacher([group('any',[leaf('Python'),leaf('Java')],modality='preferred')],'Python或Java')
    predicted,mentions=ev.parser_output('必须掌握Python或Java。','description')
    result=ev.aggregate([ev.compare_one(gold,predicted,mentions)])
    assert result['groups']['logic_agreement_on_comparable_same_leaf_groups']==1
    assert result['groups']['modality_agreement_on_same_leaf_groups']==0


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False))
def jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def study_fixture(tmp_path):
    old=tmp_path/'r2';new=tmp_path/'r4';study=tmp_path/'study';annotation=tmp_path/'annotation'
    for directory in (old,new,study,annotation):directory.mkdir()
    jobs=[];tasks=[]
    for i,fragment in enumerate(['熟悉Python或Java。','熟悉Python和SQL。']):
        row={'job_id':str(i),'job_version_id':str(i),'snapshot':'fixture','job_family_id':str(i),'split':'train','title':'虚构职位','company':'虚构企业',
             'requirements':'本科','description':fragment,'salary_raw':'10-20K','address':'深圳'}
        jobs.append(row);material={'field':'description','fragment':fragment}
        tasks.append({'task_id':'t'+str(i),'kind':'requirement_extraction','job_id':str(i),'split':'train','material':material,
                      'input_hash':ev.stable_hash(material),'source_job_sha256':ev.stable_hash(row)})
    for directory in (old,new):
        jsonl(directory/'jobs.jsonl',jobs)
        dump(directory/'manifest.json',{'files':{'jobs.jsonl':{'sha256':file_sha(directory/'jobs.jsonl')}},'config':{'parser_version':ev.PARSER_VERSION,'retrieval_contract':ev.contract_info()}})
    jsonl(study/'extraction_tasks.jsonl',tasks)
    dump(study/'manifest.json',{'dataset':str(old),'files':{'extraction_tasks.jsonl':file_sha(study/'extraction_tasks.jsonl')}})
    dump(annotation/'manifest.json',{'tasks_hash':ev.stable_hash(tasks),'status':'running'})
    task=tasks[0]
    row={k:v for k,v in task.items() if k!='material'}
    row.update(label_source='llm_reviewed',human_reviewed=False,eligible_for_production=False,
               model_reviews=[{'channel':c,'input_hash':task['input_hash']} for c in ['support','transfer']],
               rule_validation={'passed':True},groups=[group('any',[leaf('Python'),leaf('Java')])])
    dump(annotation/'batch-0000.json',{'status':'completed','rows':[row]})
    return study,new,annotation


def test_incomplete_work_is_progress_only_and_reports_missing_denominator(tmp_path):
    study,dataset,annotation=study_fixture(tmp_path)
    with pytest.raises(ValueError,match='标签未完整'):ev.run(study,dataset,annotation,tmp_path/'rejected')
    result=ev.run(study,dataset,annotation,tmp_path/'progress',allow_partial=True)
    assert result['status']=='progress_only' and result['tasks_expected']==2
    assert result['tasks_evaluated']==1 and result['tasks_pending']==1
    assert result['teacher_relative'] and result['human_gold_count']==0
    assert result['metrics']['entities']['f1']==1
    content=(tmp_path/'progress/report.json').read_text()+(tmp_path/'progress/cases.jsonl').read_text()
    assert '熟悉Python或Java' not in content and '虚构企业' not in content
    with pytest.raises(FileExistsError):ev.run(study,dataset,annotation,tmp_path/'progress',allow_partial=True)


def test_parser_version_and_task_hash_must_match(tmp_path):
    study,dataset,annotation=study_fixture(tmp_path)
    manifest=json.loads((dataset/'manifest.json').read_text());manifest['config']['parser_version']='old-version';dump(dataset/'manifest.json',manifest)
    with pytest.raises(ValueError,match='不是指定冻结版本'):ev.run(study,dataset,annotation,tmp_path/'wrong',True)
