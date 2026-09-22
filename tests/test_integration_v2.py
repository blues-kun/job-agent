"""当前checkout的v2集成回归：仅临时虚构岗位/会话，不连接模型或真实数据。

运行时显式优先导入checkout，避免此目录的旧覆盖包干扰集成结果。
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

CHECKOUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CHECKOUT))

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook
import job_agent.api as api_module
import job_agent.domain as domain_module
from job_agent.api import create_app
from job_agent.corpus import Corpus
from job_agent.profiles import fact_fingerprint,profile_preview
from job_agent.retrieval_contract import experience_blocks,query_segments
from job_agent.workflow import Workflow

PREFS={'city':'深圳','intent':'Python开发','education':'本科','experience_years':1,'salary_min':8000,'salary_max':None,'district':''}
TEXT='本科毕业，1年工作经验。\n\n项目一：清洗项目\n使用Python完成数据清洗脚本，负责检查输入和复核结果。'
EVIDENCE='我使用SQL完成订单查询项目，负责设计查询语句并核对分组统计结果。'


@pytest.fixture
def environment(tmp_path,monkeypatch):
    assert Path(domain_module.__file__).resolve()==CHECKOUT/'job_agent/domain.py'
    for key in tuple(os.environ):
        if key.startswith('JOB_AGENT_'):
            monkeypatch.delenv(key,raising=False)
    source=tmp_path/'fiction.xlsx';book=Workbook();sheet=book.active
    sheet.append(['岗位名称','企业','岗位薪资','岗位要求','岗位职责','岗位地址','职位类型名称','二级分类'])
    fixtures=[
        ['Python订单开发','虚构企业甲','10-20K','经验不限本科','熟悉Python，必须掌握SQL。负责订单查询和交易对账。','深圳南山区','Python','后端开发'],
        ['Python视觉开发','虚构企业乙','10-20K','经验不限本科','熟悉Python，负责目标检测、图像处理和视觉模型训练。','深圳南山区','Python','人工智能'],
        ['Python文本开发','虚构企业丙','10-20K','经验不限本科','熟悉Python，负责自然语言处理、文本检索和语料处理。','深圳福田区','Python','人工智能'],
        ['Java后台开发','虚构企业丁','10-20K','经验不限本科','熟悉Java和MySQL，开发后台业务接口。','深圳福田区','Java','后端开发'],
        ['项目协调专员','虚构企业戊','10-20K','经验不限本科','负责项目协调、客户需求分析和客户维护，跟踪项目进度。','深圳南山区','项目专员/助理','技术项目管理'],
        ['接口技术专员','虚构企业己','10-20K','经验不限本科','必须掌握(Python和SQL)或(Java和MySQL)。\n优先：熟悉Go。','深圳南山区','后端开发','后端开发'],
    ]
    for row in fixtures:sheet.append(row)
    book.save(source);corpus=Corpus(source)
    app=create_app(source=source,private_root=tmp_path/'private',corpus=corpus)
    with TestClient(app,raise_server_exceptions=False) as client:
        yield SimpleNamespace(app=app,client=client,corpus=corpus,private=tmp_path/'private')


def request(text=TEXT,prefs=None):
    return {'text':text,'preferences':deepcopy(prefs or PREFS),'limit':5,'method':'hybrid','query_mode':'structured','confirmation_token':'','field_origins':{}}


def confirmed(client,payload=None):
    payload=deepcopy(payload or request())
    preview=client.post('/api/v2/profile/preview',json=payload)
    assert preview.status_code==200,preview.text
    payload['preferences']=preview.json()['preferences']
    confirmation=client.post('/api/v2/profile/confirm',json={**payload,'user_confirmed':True,'acknowledged_conflicts':preview.json()['conflicts']})
    assert confirmation.status_code==200,confirmation.text
    payload['confirmation_token']=confirmation.json()['confirmation_token']
    return payload


def target_id(environment):
    return next(job.id for job in environment.corpus.jobs if job.title=='Python订单开发')


def start_action(environment,payload=None):
    payload=confirmed(environment.client,payload)
    target=target_id(environment)
    diagnosis=environment.client.post('/api/v2/diagnose',json={**payload,'job_id':target})
    assert diagnosis.status_code==200,diagnosis.text
    gap=next(gap for gap in diagnosis.json()['gaps'] if 'SQL' in gap['options'])
    response=environment.client.post('/api/v2/journey/actions',json={**payload,'job_id':target,'group_id':gap['group_id']})
    assert response.status_code==200,response.text
    return payload,target,response.json()


@pytest.mark.parametrize('endpoint',['recommend','diagnose','rewrite','interview','coach','compare','journey/actions','journey/confirm-evidence'])
def test_sensitive_workflow_requires_current_confirmation(environment,endpoint):
    payload=request()
    if endpoint not in {'recommend','compare'}:payload['job_id']=target_id(environment)
    if endpoint=='journey/actions':payload['group_id']='missing'
    if endpoint=='journey/confirm-evidence':payload.update(action_id='a'*32,revised_text=TEXT,user_confirmed=True)
    response=environment.client.post('/api/v2/'+endpoint,json=payload)
    assert response.status_code==409,response.text


def test_confirmation_binds_text_preferences_session_and_expiry(environment,monkeypatch):
    client=environment.client;payload=confirmed(client)
    assert client.post('/api/v2/recommend',json=payload).status_code==200
    changed=deepcopy(payload);changed['preferences']['salary_min']=12000
    assert client.post('/api/v2/recommend',json=changed).status_code==409
    changed=deepcopy(payload);changed['text']+='\n使用SQL完成项目。'
    assert client.post('/api/v2/recommend',json=changed).status_code==409
    cookie=client.cookies.get('job_agent_session');client.cookies.clear()
    assert client.post('/api/v2/recommend',json=payload).status_code==409
    client.cookies.clear();client.cookies.set('job_agent_session',cookie)
    stamp=int(payload['confirmation_token'].split('.')[0])
    monkeypatch.setattr(api_module,'time',SimpleNamespace(time=lambda:stamp+86401))
    assert client.post('/api/v2/recommend',json=payload).status_code==409


def test_conflicts_and_stale_sample_fields_require_explicit_resolution(environment):
    payload=request(prefs={**PREFS,'experience_years':8,'education':'博士'})
    preview=environment.client.post('/api/v2/profile/preview',json=payload).json()
    assert {'education','experience_years'}<=set(preview['conflicts'])
    response=environment.client.post('/api/v2/profile/confirm',json={**payload,'user_confirmed':True})
    assert response.status_code==422
    payload['field_origins']={'education':'sample_stale','experience_years':'sample_stale','salary_min':'sample_stale'}
    preview=environment.client.post('/api/v2/profile/preview',json=payload).json()
    assert preview['preferences']['education']=='本科' and preview['preferences']['experience_years']==1
    assert preview['preferences']['salary_min'] is None
    assert environment.client.post('/api/v2/profile/confirm',json={**payload,'user_confirmed':True,'acknowledged_conflicts':preview['conflicts']}).status_code==422


def test_preview_confirmation_cannot_silently_confirm_different_effective_fields(environment):
    """预览已抽取的城市/意向不得在确认响应里悄悄退回空值。"""
    payload=request('目标城市：深圳\n求职方向：Python开发\n'+TEXT,prefs={**PREFS,'city':'','intent':''})
    preview=environment.client.post('/api/v2/profile/preview',json=payload).json()
    assert preview['preferences']['city']=='深圳'
    response=environment.client.post('/api/v2/profile/confirm',json={**payload,'user_confirmed':True,'acknowledged_conflicts':preview['conflicts']})
    assert response.status_code in (200,409,422)
    if response.status_code==200:
        assert response.json()['preferences']==preview['preferences'],'确认成功但确认字段与刚展示的有效画像不同'


def test_real_experience_changes_ranking_for_same_declared_skills(environment):
    workflow=Workflow(environment.corpus)
    visual='本科，1年工作经验。我使用Python完成目标检测和图像处理项目，负责视觉模型训练、图像标注及检测结果复核。'
    language='本科，1年工作经验。我使用Python完成自然语言处理项目，负责文本检索、语料处理及查询结果复核。'
    left=workflow.recommend(visual,PREFS,method='bm25',query_mode='experience')
    right=workflow.recommend(language,PREFS,method='bm25',query_mode='experience')
    assert left['jobs'] and right['jobs']
    assert left['jobs'][0]['title']=='Python视觉开发'
    assert right['jobs'][0]['title']=='Python文本开发'
    assert query_segments({'text':visual,'preferences':PREFS},'experience')!=query_segments({'text':language,'preferences':PREFS},'experience')


def test_block_reordering_preserves_titles_dates_multiset_and_facts(environment):
    text='项目一：客户协作\n2024-01至2024-12\n负责客户维护和项目协调。\n\n项目二：数据脚本\n2025-01至2025-06\n使用Python和SQL完成订单查询项目。'
    payload=confirmed(environment.client,request(text))
    response=environment.client.post('/api/v2/rewrite',json={**payload,'job_id':target_id(environment)})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['facts_preserved'] and result['new_claims']==0
    assert Counter(experience_blocks(text))==Counter(experience_blocks(result['revised']))
    assert fact_fingerprint(text,payload['preferences'])==fact_fingerprint(result['revised'],payload['preferences'])
    assert '项目一：客户协作\n2024-01至2024-12\n负责客户维护和项目协调。' in experience_blocks(result['revised']),'标题被从日期职责块拆开，事实指纹却仍通过'
    changed={**payload,'text':result['revised']}
    if result['revised']!=text:
        assert environment.client.post('/api/v2/recommend',json=changed).status_code==409


def test_journey_consent_ownership_append_only_reconfirm_and_delete(environment):
    client=environment.client;payload,target,action=start_action(environment)
    assert client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':EVIDENCE,'consent_to_store':False}).status_code==422
    cookie=client.cookies.get('job_agent_session');client.cookies.clear()
    assert client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':EVIDENCE,'consent_to_store':True}).status_code==404
    assert client.get('/api/v2/journey').json()['actions']==[]
    client.cookies.clear();client.cookies.set('job_agent_session',cookie)
    saved=client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':EVIDENCE,'consent_to_store':True})
    assert saved.status_code==200
    confirm={**payload,'job_id':target,'action_id':action['id'],'revised_text':TEXT.rstrip()+'\n\n'+EVIDENCE,'user_confirmed':False}
    assert client.post('/api/v2/journey/confirm-evidence',json=confirm).status_code==422
    confirm['user_confirmed']=True
    bad={**confirm,'revised_text':confirm['revised_text'].replace('1年工作经验','8年工作经验')}
    assert client.post('/api/v2/journey/confirm-evidence',json=bad).status_code==422
    response=client.post('/api/v2/journey/confirm-evidence',json=confirm)
    assert response.status_code==200,response.text
    assert 'SQL' in str(response.json()['closed_gaps'])
    history=client.get('/api/v2/journey').json()
    assert history['actions'][0]['status']=='confirmed' and len(history['versions'])==1
    summary=history['versions'][0]['summary']
    assert summary['confirmation']=='user_confirmed_not_externally_verified'
    assert TEXT not in json.dumps(summary,ensure_ascii=False)
    assert client.post('/api/v2/recommend',json={**payload,'text':confirm['revised_text']}).status_code==409
    assert client.delete('/api/v2/journey').status_code==200
    assert client.get('/api/v2/journey').json()['actions']==[]
    assert client.get('/api/v2/journey').json()['versions']==[]
    assert client.get('/api/v2/research-events').json()['events']==[]
    assert client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':EVIDENCE,'consent_to_store':True}).status_code==404


def test_confirmed_evidence_is_not_overwritten_under_old_version_reference(environment):
    client=environment.client;payload,target,action=start_action(environment)
    client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':EVIDENCE,'consent_to_store':True})
    response=client.post('/api/v2/journey/confirm-evidence',json={**payload,'job_id':target,'action_id':action['id'],'revised_text':TEXT.rstrip()+'\n\n'+EVIDENCE,'user_confirmed':True})
    assert response.status_code==200
    changed=client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':'我使用Java完成另一个练习项目，负责开发接口并验证请求返回。','consent_to_store':True})
    assert changed.status_code in (409,422),'同一已确认action被覆盖，旧versions.action_id失去原证据血缘'


def test_expired_journey_action_cannot_be_resurrected_without_history_read(environment):
    client=environment.client;_,_,action=start_action(environment)
    with environment.app.state.journey.connect() as db:
        db.execute('UPDATE actions SET updated=? WHERE id=?',(time.time()-31*86400,action['id']))
    response=client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':EVIDENCE,'consent_to_store':True})
    assert response.status_code in (404,409,422),'超过30天的证据行动在直接写入路径被复活'


def test_team_technical_background_is_not_personal_practice(environment):
    text='项目团队使用Python和SQL完成数据接口，我负责项目协调、客户需求分析和客户维护，跟踪项目进度并整理交付事项。'
    payload=confirmed(environment.client,request(text))
    response=environment.client.post('/api/v2/recommend',json=payload)
    assert response.status_code==200,response.text
    result=response.json()
    assert result['profile']['tasks'],'本人项目协调任务可以作为真实自述'
    for skill in ('Python','SQL'):
        assert result['profile']['skills'][skill]['weight']<.7,'团队使用技术不表示本人使用'
    assert not any(item['skill']in {'Python','SQL'} and item['level']=='实践' for job in result['jobs'] for item in job['matched'])


def test_research_features_require_consent_and_are_deleted(environment):
    client=environment.client;payload=confirmed(client)
    assert client.post('/api/v2/recommend',json=payload).status_code==200
    assert client.get('/api/v2/research-events').json()['events']==[]
    payload['research_consent']=True
    assert client.post('/api/v2/recommend',json=payload).status_code==200
    rows=client.get('/api/v2/research-events').json()['events']
    assert len(rows)==1 and rows[0]['raw_resume_stored'] is False
    assert TEXT not in json.dumps(rows,ensure_ascii=False)
    client.delete('/api/v2/journey')
    assert client.get('/api/v2/research-events').json()['events']==[]


def test_api_nested_requirements_preserve_branch_scopes_and_unknown_evidence(environment):
    """真实接口贯通嵌套逻辑、可选要求和双侧跨度，不把未写技能判作不会。"""
    target=next(job for job in environment.corpus.jobs if job.title=='接口技术专员')
    client=environment.client
    for text,expected in [('本科毕业。使用Python和SQL完成服务开发。','pass'),
                          ('本科毕业。使用Python完成服务开发。','unknown'),
                          ('本科毕业。我不会Python，也不会Java。','fail')]:
        payload=confirmed(client,request(text))
        response=client.post('/api/v2/diagnose',json={**payload,'job_id':target.id})
        assert response.status_code==200,response.text
        result=response.json()
        required=[group for group in result['groups'] if not group['preferred']]
        assert len(required)==1 and required[0]['ast']['op']=='any'
        assert [child['op'] for child in required[0]['ast']['children']]==['all','all']
        required_gaps=[gap for gap in result['gaps'] if not gap['preferred']]
        if expected=='pass':
            assert not required_gaps and result['coverage']==100
            match=next(item for item in result['matched'] if item['group_id']==required[0]['group_id'])
            assert len(match['leaf_support'])==4
            for leaf in match['leaf_support']:
                cite=leaf['job_evidence']
                assert getattr(target,cite['field'])[cite['start']:cite['end']]==cite['quote']
                if leaf['resume_evidence']:
                    cite=leaf['resume_evidence'];assert text[cite['start']:cite['end']]==cite['quote']
        else:
            assert len(required_gaps)==1 and required_gaps[0]['status']==expected
        optional_gaps=[gap for gap in result['gaps'] if gap['preferred']]
        assert optional_gaps and all(gap['status']=='unknown' for gap in optional_gaps)


def test_frontend_sample_edit_and_upload_clear_only_sample_values(tmp_path):
    node=shutil.which('node') or '/tmp/cube-node-bin/node'
    if not Path(node).exists():pytest.skip('当前环境无可用Node，不伪造前端执行结果')
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function get(id){if(!elements.has(id))elements.set(id,{value:'',checked:false,open:false,events:{},classList:{add(){},remove(){},toggle(){}},addEventListener(name,fn){this.events[name]=fn},querySelectorAll(){return[]},focus(){},close(){this.open=false},reportValidity(){return true}});return elements.get(id)}
const context={document:{getElementById:get,querySelectorAll(){return[]},querySelector(){return get('query')},modelContext:null},window:{addEventListener(){}},location:{hash:''},setTimeout(){return 1},clearTimeout(){},console,FormData:class{append(){}},fetch:async()=>({ok:true,json:async()=>({text:'真实自定义简历：使用SQL完成查询。',note:'模拟内存导入'})})};
vm.createContext(context);
const source=fs.readFileSync(process.argv[2],'utf8').replace(/\ninit\(\);\s*$/,'');vm.runInContext(source,context);
vm.runInContext(`(async()=>{
 state.samples=[{id:'sample-a',text:'虚构样例',preferences:{city:'深圳',intent:'Python',education:'博士',experience_years:10,salary_min:50000}},{id:'sample-b',text:'另一虚构样例',preferences:{city:'深圳',intent:'Java'}}];
 applySample('sample-a'); $('city').value='广州'; $('city').events.input();
 $('resume-text').value='新的真实经历'; $('resume-text').events.input();
 if($('city').value!=='广州'||$('education').value!==''||$('years').value!==''||$('salary').value!=='')throw Error('样例字段污染或误删用户编辑');
 applySample('sample-a'); applySample('sample-b');if($('education').value!==''||$('salary').value!=='')throw Error('切换样例残留字段');
 applySample('sample-a');state.confirmation='old';const input=$('resume-file');input.files=[{name:'resume.txt',size:100}];
 await input.events.change({target:input});
 if($('education').value!==''||$('years').value!==''||$('salary').value!==''||state.confirmation!=='')throw Error('上传未清空样例或旧确认');
})()`,context).then(()=>console.log('frontend_sample_flow_ok')).catch(error=>{console.error(error);process.exitCode=1});
'''
    path=tmp_path/'frontend_sample_flow.cjs';path.write_text(script)
    result=subprocess.run([node,str(path),str(CHECKOUT/'web/platform/app.js')],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'frontend_sample_flow_ok' in result.stdout

def test_api_nested_requirements_preserve_branch_scopes_and_unknown_evidence(environment):
    """真实接口贯通嵌套逻辑、可选要求和双侧跨度，不把未写技能判作不会。"""
    target=next(job for job in environment.corpus.jobs if job.title=='接口技术专员')
    client=environment.client
    for text,expected in [('本科毕业。使用Python和SQL完成服务开发。','pass'),
                          ('本科毕业。使用Python完成服务开发。','unknown'),
                          ('本科毕业。我不会Python，也不会Java。','fail')]:
        payload=confirmed(client,request(text))
        response=client.post('/api/v2/diagnose',json={**payload,'job_id':target.id})
        assert response.status_code==200,response.text
        result=response.json()
        required=[group for group in result['groups'] if not group['preferred']]
        assert len(required)==1 and required[0]['ast']['op']=='any'
        assert [child['op'] for child in required[0]['ast']['children']]==['all','all']
        required_gaps=[gap for gap in result['gaps'] if not gap['preferred']]
        if expected=='pass':
            assert not required_gaps and result['coverage']==100
            match=next(item for item in result['matched'] if item['group_id']==required[0]['group_id'])
            assert len(match['leaf_support'])==4
            for leaf in match['leaf_support']:
                cite=leaf['job_evidence']
                assert getattr(target,cite['field'])[cite['start']:cite['end']]==cite['quote']
                if leaf['resume_evidence']:
                    cite=leaf['resume_evidence'];assert text[cite['start']:cite['end']]==cite['quote']
        else:
            assert len(required_gaps)==1 and required_gaps[0]['status']==expected
        optional_gaps=[gap for gap in result['gaps'] if gap['preferred']]
        assert optional_gaps and all(gap['status']=='unknown' for gap in optional_gaps)


def test_frontend_sample_edit_and_upload_clear_only_sample_values(tmp_path):
    node=shutil.which('node') or '/tmp/cube-node-bin/node'
    if not Path(node).exists():pytest.skip('当前环境无可用Node，不伪造前端执行结果')
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function get(id){if(!elements.has(id))elements.set(id,{value:'',checked:false,open:false,events:{},classList:{add(){},remove(){},toggle(){}},addEventListener(name,fn){this.events[name]=fn},querySelectorAll(){return[]},focus(){},close(){this.open=false},reportValidity(){return true}});return elements.get(id)}
const context={document:{getElementById:get,querySelectorAll(){return[]},querySelector(){return get('query')},modelContext:null},window:{addEventListener(){}},location:{hash:''},setTimeout(){return 1},clearTimeout(){},console,FormData:class{append(){}},fetch:async()=>({ok:true,json:async()=>({text:'真实自定义简历：使用SQL完成查询。',note:'模拟内存导入'})})};
vm.createContext(context);
const source=fs.readFileSync(process.argv[2],'utf8').replace(/\ninit\(\);\s*$/,'');vm.runInContext(source,context);
vm.runInContext(`(async()=>{
 state.samples=[{id:'sample-a',text:'虚构样例',preferences:{city:'深圳',intent:'Python',education:'博士',experience_years:10,salary_min:50000}},{id:'sample-b',text:'另一虚构样例',preferences:{city:'深圳',intent:'Java'}}];
 applySample('sample-a'); $('city').value='广州'; $('city').events.input();
 $('resume-text').value='新的真实经历'; $('resume-text').events.input();
 if($('city').value!=='广州'||$('education').value!==''||$('years').value!==''||$('salary').value!=='')throw Error('样例字段污染或误删用户编辑');
 applySample('sample-a'); applySample('sample-b');if($('education').value!==''||$('salary').value!=='')throw Error('切换样例残留字段');
 applySample('sample-a');state.confirmation='old';const input=$('resume-file');input.files=[{name:'resume.txt',size:100}];
 await input.events.change({target:input});
 if($('education').value!==''||$('years').value!==''||$('salary').value!==''||state.confirmation!=='')throw Error('上传未清空样例或旧确认');
})()`,context).then(()=>console.log('frontend_sample_flow_ok')).catch(error=>{console.error(error);process.exitCode=1});
'''
    path=tmp_path/'frontend_sample_flow.cjs';path.write_text(script)
    result=subprocess.run([node,str(path),str(CHECKOUT/'web/platform/app.js')],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'frontend_sample_flow_ok' in result.stdout
