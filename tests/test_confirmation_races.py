"""最终接口复核：全部虚构临时材料，不连接模型或外部接口。"""
from copy import deepcopy
import os
import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook
from job_agent.api import create_app
from job_agent.corpus import Corpus

TEXT='本科毕业，1年工作经验。使用Python完成数据清洗项目，负责检查输入并复核结果。'
FIRST='我使用SQL完成订单查询练习，负责编写查询语句并核对分组统计结果。'
SECOND='我使用Java完成另一个接口练习，负责发送请求并核对接口返回的结果。'
PREFS={'city':'深圳','intent':'Python开发','education':'本科','experience_years':1,'salary_min':8000,'salary_max':None,'district':''}


@pytest.fixture
def env(tmp_path,monkeypatch):
    for key in tuple(os.environ):
        if key.startswith('JOB_AGENT_'):monkeypatch.delenv(key,raising=False)
    source=tmp_path/'fiction.xlsx';book=Workbook();sheet=book.active
    sheet.append(['岗位名称','企业','岗位薪资','岗位要求','岗位职责','岗位地址','职位类型名称','二级分类'])
    sheet.append(['Python订单开发','虚构企业甲','10-20K','经验不限本科','熟悉Python，必须掌握SQL。负责订单查询。','深圳南山区','Python','后端开发'])
    sheet.append(['Java后台开发','虚构企业乙','10-20K','经验不限本科','熟悉Java和MySQL。负责后台接口。','深圳南山区','Java','后端开发'])
    book.save(source);corpus=Corpus(source)
    app=create_app(source=source,private_root=tmp_path/'private',corpus=corpus)
    with TestClient(app,raise_server_exceptions=False) as client:
        payload={'text':TEXT,'preferences':deepcopy(PREFS),'limit':2}
        preview=client.post('/api/v2/profile/preview',json=payload).json()
        payload['preferences']=preview['preferences']
        confirm=client.post('/api/v2/profile/confirm',json={**payload,'user_confirmed':True,'acknowledged_conflicts':preview['conflicts']})
        assert confirm.status_code==200,confirm.text
        payload['confirmation_token']=confirm.json()['confirmation_token']
        yield SimpleNamespace(app=app,client=client,payload=payload,job=next(j for j in corpus.jobs if j.title=='Python订单开发'))


def saved_action(env):
    payload={**env.payload,'job_id':env.job.id}
    result=env.client.post('/api/v2/diagnose',json=payload)
    assert result.status_code==200,result.text
    gap=next(g for g in result.json()['gaps'] if 'SQL' in g['options'])
    action=env.client.post('/api/v2/journey/actions',json={**payload,'group_id':gap['group_id']})
    assert action.status_code==200,action.text
    action=action.json()
    saved=env.client.post('/api/v2/journey/evidence',json={'action_id':action['id'],'evidence':FIRST,'consent_to_store':True})
    assert saved.status_code==200,saved.text
    return {**payload,'action_id':action['id'],'revised_text':TEXT+'\n\n'+FIRST,'user_confirmed':True},action


def test_confirm_cannot_accept_evidence_replaced_after_api_validation(env,monkeypatch):
    """确定性模拟另一请求恰在API验证后、存储确认前修改证据。"""
    payload,action=saved_action(env);store=env.app.state.journey;original=store.confirm
    def interleave(session,identifier,summary,**kwargs):
        store.save_evidence(session,identifier,SECOND)
        return original(session,identifier,summary,**kwargs)
    monkeypatch.setattr(store,'confirm',interleave)
    result=env.client.post('/api/v2/journey/confirm-evidence',json=payload)
    assert result.status_code in (409,422),'用户只确认FIRST，但服务器将SECOND确认并写入FIRST摘要'
    history=env.client.get('/api/v2/journey').json()
    assert history['versions']==[]
    assert history['actions'][0]['status']=='evidence_submitted'
    assert history['actions'][0]['evidence']==SECOND


def test_confirmation_remains_user_self_report_and_requires_reconfirmation(env):
    payload,action=saved_action(env)
    assert env.client.post('/api/v2/journey/confirm-evidence',json={**payload,'user_confirmed':False}).status_code==422
    assert env.client.get('/api/v2/journey').json()['versions']==[]
    result=env.client.post('/api/v2/journey/confirm-evidence',json=payload)
    assert result.status_code==200,result.text
    assert '没有外部验证' in result.json()['note']
    history=env.client.get('/api/v2/journey').json()
    assert history['versions'][0]['summary']['confirmation']=='user_confirmed_not_externally_verified'
    changed={**env.payload,'text':payload['revised_text']}
    assert env.client.post('/api/v2/recommend',json=changed).status_code==409


def test_expired_research_events_are_not_exported_without_history_request(env):
    result=env.client.post('/api/v2/recommend',json={**env.payload,'research_consent':True})
    assert result.status_code==200,result.text
    assert len(env.client.get('/api/v2/research-events').json()['events'])==1
    with env.app.state.journey.connect() as db:
        db.execute('UPDATE research_events SET created=?',(time.time()-31*86400,))
    assert env.client.get('/api/v2/research-events').json()['events']==[]


def test_expired_exposure_cannot_authorize_feedback_or_company_exclusion(env):
    result=env.client.post('/api/v2/recommend',json=env.payload)
    assert result.status_code==200,result.text
    row=result.json();assert row['jobs'];job_id=row['jobs'][0]['id']
    with sqlite3.connect(env.app.state.db) as db:
        db.execute('UPDATE exposures SET created=?',(time.time()-25*3600,))
    assert env.client.post('/api/v2/feedback',json={'run_id':row['run_id'],'job_id':job_id,'action':'like'}).status_code==404
    assert env.client.post('/api/v2/preferences/company',json={'job_id':job_id,'exclude':True}).status_code==404


def test_expired_feedback_is_not_returned_by_read_only_summary(env):
    row=env.client.post('/api/v2/recommend',json=env.payload).json()
    assert env.client.post('/api/v2/feedback',json={'run_id':row['run_id'],'job_id':row['jobs'][0]['id'],'action':'like'}).status_code==200
    with sqlite3.connect(env.app.state.db) as db:
        db.execute('UPDATE feedback SET created=?',(time.time()-25*3600,))
    assert env.client.get('/api/v2/feedback').json()['counts']=={}
