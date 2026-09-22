"""针对真实故障模式的回归；仅使用临时合成岗位，不读取私人简历。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from job_agent.api import create_app
from job_agent.corpus import Corpus
from job_agent.domain import Job, constraints, education, parse_profile, skill_evidence, supported_matches
from job_agent.workflow import Workflow


def job(description="熟悉Python或Java或Go任意一种，使用SQL进行数据处理。", requirements="经验不限本科", **changes):
    values = dict(id="fixture-python", version="v1", title="Python开发工程师", company="虚构企业甲", category="Python", family="后端开发",
                  salary_raw="10-20K·14薪", requirements=requirements, description=description, address="深圳南山区", row=2)
    values.update(changes)
    return Job(**values)


PREFS = {"city":"深圳","intent":"Python","education":"本科","experience_years":1,"salary_min":10000,"district":""}
RESUME = "本科，1年工作经验。\n使用Python和SQL完成数据清洗项目。"


def test_education_modifiers_are_local():
    assert education("经验不限本科", True) == 3
    assert education("本科必须，硕士优先", True) == 3
    assert education("本科优先", True) is None
    assert education("学历不限", True) == -1
    profile = parse_profile(RESUME, {**PREFS,"education":"大专"})
    assert next(check for check in constraints(profile,job()) if check["name"]=="学历要求")["status"] == "fail"


def test_three_way_alternatives_are_one_group():
    target=job()
    group=next(group for group in target.groups if group["kind"]=="any")
    assert set(group["skills"]) == {"Python","Java","Go"}
    _,gaps,_=supported_matches(parse_profile(RESUME,PREFS),target)
    assert not any(set(gap["options"]) & {"Python","Java","Go"} for gap in gaps)
    for phrase in ["熟悉Python、Java、Go至少一种", "熟悉Python或Java或Go"]:
        assert len([g for g in job(phrase).groups if g["kind"]=="any"]) == 1


def test_negation_and_future_scope():
    evidence=skill_evidence("不会 Python 但熟悉 SQL。希望学习 Python，目前使用 SQL 完成项目。TensorFlow不熟悉。")
    assert evidence["Python"]["weight"] < .7
    assert evidence["SQL"]["level"] == "实践"
    assert evidence["TensorFlow"]["level"] == "否定"
    assert skill_evidence("不会 Python, 熟悉 SQL。")["SQL"]["level"] == "熟悉"


def test_keyword_stuffing_and_ascii_boundaries():
    evidence=skill_evidence("项目：Python、Java、C++、SQL、TensorFlow、PyTorch。")
    assert all(item["weight"] < .7 for item in evidence.values())
    assert "Java" not in skill_evidence("使用JavaScript开发页面")
    assert skill_evidence("熟悉Python")["Python"]["level"] == "熟悉"


def test_requirements_field_has_correct_provenance():
    target=job("负责业务数据处理", "本科，经验不限，必须掌握Python")
    matches,_,_=supported_matches(parse_profile(RESUME,PREFS),target)
    evidence=matches[0]["job_evidence"]
    assert evidence["field"] == "requirements"
    assert target.requirements[evidence["start"]:evidence["end"]] == evidence["quote"]


def test_unknown_location_salary_period_and_experience():
    profile=parse_profile(RESUME,PREFS)
    target=job(address="",salary_raw="200-300元/天",requirements="3天/周本科")
    checks={check["name"]:check["status"] for check in constraints(profile,target)}
    assert checks["目标城市"] == checks["最低月薪"] == checks["经验要求"] == "unknown"
    target=job(requirements="3-5年本科")
    assert next(check for check in constraints(profile,target) if check["name"]=="经验要求")["status"] == "fail"
    target=job()
    assert target.salary["annual_low"] == 140000 and target.salary["monthly_low"] == 10000


def test_no_contact_in_constraint_response():
    target=job(requirements="本科，经验不限，联系电话13800000000")
    assert "13800000000" not in str(constraints(parse_profile(RESUME,PREFS),target))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root=tmp_path_factory.mktemp("岗位库")
    source=root/"fixtures.xlsx"
    book=Workbook();sheet=book.active
    sheet.append(["岗位名称","企业","岗位薪资","岗位要求","岗位职责","岗位地址","职位类型名称","二级分类"])
    rows=[
        ["Python数据开发","虚构企业甲","10-20K","经验不限本科","使用Python和SQL完成数据清洗项目，熟悉Python或Java或Go任一种。","深圳南山区","Python","后端开发"],
        ["Python后端开发","虚构企业乙","12-22K","1-3年本科","使用Python和FastAPI开发接口，熟悉SQL。","深圳福田区","Python","后端开发"],
        ["Python架构师","虚构企业丙","35-50K","5-10年硕士","使用Python设计分布式系统，精通SQL。","深圳南山区","Python","后端开发"],
        ["Java后端开发","虚构企业丁","12-20K","1-3年本科","使用Java和MySQL开发后台接口。","深圳宝安区","Java","后端开发"],
        ["Python数据工程师","虚构企业戊","面议","经验不限本科","使用Python处理数据，熟悉Pandas和SQL。","","Python","后端开发"],
    ]
    for row in rows+rows[:1]:sheet.append(row)
    book.save(source)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    result=Corpus(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    assert result.duplicates==1 and len(result.jobs)==5
    return result


def test_ranker_filters_hard_fail_and_checks_quotes(corpus):
    result=Workflow(corpus).recommend(RESUME,PREFS)
    assert result["action"]=="recommend" and result["jobs"]
    assert all(not any(check["status"]=="fail" for check in item["constraints"]) for item in result["jobs"])
    assert all(item["experience_min"]!=5 for item in result["jobs"])
    assert result["jobs"][0]["category"]=="Python"
    for item in result["jobs"]:
        raw=corpus.by_id[item["id"]]
        for match in item["matched"]:
            jd,resume=match["job_evidence"],match["resume_evidence"]
            assert getattr(raw,jd["field"])[jd["start"]:jd["end"]]==jd["quote"]
            assert RESUME[resume["start"]:resume["end"]]==resume["quote"]


def test_clarifies_keyword_only_and_refuses_impossible_city(corpus):
    workflow=Workflow(corpus)
    assert workflow.recommend("项目：Python、SQL、Java。",PREFS)["action"]=="clarify"
    assert workflow.recommend(RESUME,{**PREFS,"city":"杭州"},strict_unknown=True)["action"]=="no_match"


def test_generic_development_word_does_not_override_specific_intent(corpus, monkeypatch):
    jobs = [job("熟悉Python、FastAPI和Django。", id="python"),
            job("熟悉Python。", id="labview", title="LabVIEW测试软件开发工程师", category="测试", company="虚构企业乙")]
    fixture = SimpleNamespace(jobs=jobs, snapshot="fixture", dense=None, overview=lambda: {"engine":"测试检索"})
    workflow = Workflow(fixture)
    # 控制两条候选召回/任务证据相同，仅检验明确意向在排序中的作用。
    contexts=[{"bm25":1.,"dense":.5,"rrf":1.,"sources":["bm25"],"task_alignment":.5} for _ in jobs]
    profile=parse_profile(RESUME,{**PREFS,"intent":"Python开发"})
    monkeypatch.setattr(workflow,"retrieval_state",lambda *a,**k:([constraints(profile,j) for j in jobs],np.ones(2,dtype=bool),np.ones(2),{"bm25":[0,1]},contexts))
    result = workflow.recommend(RESUME,{**PREFS,"intent":"Python开发"})
    assert result["jobs"][0]["id"] == "python"
    assert result["jobs"][0]["score"] > result["jobs"][1]["score"]


def test_dense_service_failure_is_visible_fallback(corpus, monkeypatch):
    class FailedEncoder:
        def score(self, query):
            raise ConnectionError("模拟本地编码进程中断")
    monkeypatch.setattr(corpus,"dense",FailedEncoder())
    result=Workflow(corpus).recommend(RESUME,PREFS)
    assert result["jobs"] and "char_tfidf_fallback" in result["source_counts"]
    assert "本次语义编码器不可用" in result["retriever"]


def test_dense_startup_failure_keeps_lexical_search(corpus, tmp_path, monkeypatch):
    import job_agent.dense
    def unavailable(*args, **kwargs):
        raise ConnectionError("模拟编码服务启动时不可用")
    monkeypatch.setattr(job_agent.dense,"DenseIndex",unavailable)
    degraded=Corpus(corpus.source,tmp_path/"dense")
    result=Workflow(degraded).recommend(RESUME,PREFS)
    assert result["jobs"] and "初始化失败" in result["retriever"]
    assert "char_tfidf" in result["source_counts"]


def test_rewrite_does_not_change_anaphora(corpus):
    text="项目甲使用 Java。项目乙使用 Python。后者负责财务核算。"
    result=Workflow(corpus).rewrite(text,PREFS,corpus.jobs[0].id)
    assert result["revised"]==text and result["new_claims"]==0



def confirmed_payload(client):
    payload={"text":RESUME,"preferences":PREFS}
    confirmation=client.post("/api/v2/profile/confirm",json={**payload,"user_confirmed":True,"acknowledged_conflicts":[]})
    assert confirmation.status_code==200
    return {**payload,"confirmation_token":confirmation.json()["confirmation_token"]}

def test_api_feedback_is_session_bound_idempotent_and_private(corpus,tmp_path):
    application=create_app(corpus.source,tmp_path/"private",corpus)
    with TestClient(application) as client:
        result=client.post("/api/v2/recommend",json=confirmed_payload(client)).json()
        payload={"run_id":result["run_id"],"job_id":result["jobs"][0]["id"],"action":"like"}
        assert client.post("/api/v2/feedback",json=payload).status_code==200
        assert client.post("/api/v2/feedback",json=payload).status_code==200
        assert client.get("/api/v2/feedback").json()["counts"]=={"like":1}
        cookie=client.cookies.get("job_agent_session")
        client.cookies.clear()
        assert client.post("/api/v2/feedback",json=payload).status_code==404
        client.cookies.set("job_agent_session",cookie)
        assert client.post("/api/v2/feedback",json=payload,headers={"Origin":"https://untrusted.invalid"}).status_code==403
        assert client.post("/api/v2/recommend",json={"text":"x"*20001}).status_code==422
        assert client.get("/assets/../../resume.txt").status_code==404
        assert client.delete("/api/v2/feedback").status_code==200
        assert client.get("/api/v2/feedback").json()["counts"]=={}
    connection=sqlite3.connect(tmp_path/"private/feedback.sqlite3")
    schema=" ".join(row[0] for row in connection.execute("SELECT sql FROM sqlite_master WHERE type='table'"))
    assert "resume" not in schema and "description" not in schema
    connection.close()


def test_sample_compare_never_claims_gold_metrics(corpus,tmp_path):
    with TestClient(create_app(corpus.source,tmp_path/"private",corpus)) as client:
        result=client.post("/api/v2/compare",json=confirmed_payload(client)).json()
        assert len(result["rows"])==3
        assert all(row["hard_violations"]==0 for row in result["rows"])
        assert "无人工金标" in result["note"]
