"""虚构材料、mock接口和本地派生产物；不调用外部API，不创建人工金标。"""
import asyncio
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import research
research.__path__.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from research import api_annotation as api
from research.annotation_workflow import seal_task
from research.annotation_snapshot import read_snapshot
from research.annotation_supervisor import completion_gate,prepare_derived_annotations,supervise
from research.common import write_json,write_jsonl
from research.review_contracts import digest,file_sha
from research.teacher_dashboard import publish_teacher_dashboard,load_teacher_dashboard


def fixture(name="f1",kind="relevance",wrong_field=False):
    material={"resume":"本人使用Python开发订单接口。","preferences":{},
        "job":{"description":"使用Python开发订单接口。","requirements":"本科毕业。"}} if kind=="relevance" else {"fragment":"熟悉专业领域甲。"}
    task=seal_task({"task_id":name,"kind":kind,"query_id":"q-"+name,"job_id":"j-"+name,
        "profile_family_id":"pf-"+name,"job_family_id":"jf-"+name,"split":"test","material":material,"input_hash":digest(material)})
    label={"task_id":name,"reason":"虚构测试判断","grade":2,"hard_negative":False,
        "evidence":[{"resume_quote":"使用Python开发订单接口","job_quote":"使用Python开发订单接口",
                     "job_field":"requirements" if wrong_field else "description"}]} if kind=="relevance" else {
        "task_id":name,"reason":"虚构抽取","groups":[{"logic":"single","modality":"required","skills":[{"skill":"专业领域甲","quote":"专业领域甲"}]}]}
    audit={"model":api.MODEL,"model_revision":"fixture-only","prompt_hash":"p"*64,
        "input_hash":task["input_hash"],"request_id":"fixture-request","grade":label.get("grade")}
    row={**{k:v for k,v in task.items() if k!="material"},**label,"label_source":"llm_reviewed",
        "model_reviews":[{**audit,"channel":"support"},{**audit,"channel":"transfer"}],"review_run_id":"fixture-run",
        "rule_validation":api.validate_label(task,label)}
    return task,label,row


def run_fixture(tmp_path,tasks,rows,status="completed",dirname="annotation"):
    run=tmp_path/dirname;run.mkdir(parents=True)
    write_json(run/"batch-0000.json",{"batch":0,"status":"completed","rows":rows,"calls":[]})
    if status=="completed":write_jsonl(run/"labels.jsonl",rows)
    write_json(run/"manifest.json",{"schema":"job-agent-api-labels-v2","run_id":"fixture-run","tasks":len(tasks),
        "tasks_hash":digest(tasks),"model":api.MODEL,"reasoning_effort":"max","batch_size":2,
        "label_source":"llm_reviewed","status":status})
    return run


def task_file(tmp_path,tasks,name="tasks.jsonl"):
    target=tmp_path/name;write_jsonl(target,tasks);return target


def mock_call(selected,channel,labels):
    return {"channel":channel,"model":api.MODEL,"returned_model":api.MODEL,
        "requested_reasoning_effort":"max","model_revision":"fixture-only","prompt_hash":api.stable_hash(api.prompt_for(channel)),
        "request_id":"mock-"+channel,"seconds":0,"labels":[deepcopy(labels[t["task_id"]]) for t in selected]}


def test_resumes_paid_call_checkpoint_without_repeating_successful_channel(tmp_path,monkeypatch):
    task,label,_=fixture();tasks=task_file(tmp_path,[task]);run=tmp_path/"api-run";calls=[]
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def interrupted(client,endpoint,key,selected,channel,previous=None):
        calls.append(channel)
        if channel=="transfer":raise RuntimeError("模拟传输中断")
        return mock_call(selected,channel,{task["task_id"]:label})
    monkeypatch.setattr(api,"request_labels",interrupted)
    first=asyncio.run(api.annotate(tasks,tmp_path/"ignored",run))
    assert first["status"]=="incomplete" and (run/"call-0000-support.json").exists()
    first_bytes=(run/"call-0000-support.json").read_bytes()
    async def resumed(client,endpoint,key,selected,channel,previous=None):
        calls.append(channel);return mock_call(selected,channel,{task["task_id"]:label})
    monkeypatch.setattr(api,"request_labels",resumed)
    result=asyncio.run(api.annotate(tasks,tmp_path/"ignored",run))
    assert result["status"]=="completed" and result["strict_usable"]==1
    assert calls.count("support")==1 and calls.count("transfer")==2
    assert first_bytes==(run/"call-0000-support.json").read_bytes()
    _,rows,_,_=read_snapshot(run,[task],require_complete=True)
    assert api.row_usable(task,rows[task["task_id"]])


def test_cache_task_identity_forgery_rejected_before_any_request(tmp_path,monkeypatch):
    task,_,row=fixture();row["job_id"]="forged-job";run=run_fixture(tmp_path,[task],[row]);tasks=task_file(tmp_path,[task])
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def forbidden(*args,**kwargs):pytest.fail("身份失配不能触发付费请求")
    monkeypatch.setattr(api,"request_labels",forbidden)
    with pytest.raises(ValueError,match="身份"):asyncio.run(api.annotate(tasks,tmp_path/"ignored",run))


def test_unique_field_correction_preserves_raw_cache_without_paid_repair(tmp_path,monkeypatch):
    task,_,row=fixture(wrong_field=True);run=run_fixture(tmp_path,[task],[row]);tasks=task_file(tmp_path,[task])
    original=(run/"batch-0000.json").read_bytes()
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def forbidden(*args,**kwargs):pytest.fail("逐字可定位字段不需要重新付费")
    monkeypatch.setattr(api,"request_labels",forbidden)
    result=asyncio.run(api.annotate(tasks,tmp_path/"ignored",run))
    assert result["status"]=="completed" and result["strict_usable"]==0
    assert original==(run/"batch-0000.json").read_bytes()
    derived=prepare_derived_annotations(tmp_path)
    assert prepare_derived_annotations(tmp_path)==derived
    _,rows,_,_=read_snapshot(derived,[task],require_complete=True)
    fixed=rows[task["task_id"]]
    assert api.row_usable(task,fixed) and fixed["grade"]==row["grade"] and fixed["model_reviews"]==row["model_reviews"]
    assert fixed["reconciliation"]["original_label_sha256"]==digest(row)
    assert fixed["reconciliation"]["original_rule_validation"]["passed"] is False
    assert original==(run/"batch-0000.json").read_bytes()


def test_null_grade_and_unquoted_evidence_cannot_be_recovered():
    task,_,row=fixture(wrong_field=True);row["grade"]=None
    assert not api.row_recoverable_without_api(task,row)
    task,_,row=fixture(wrong_field=True);row["evidence"][0]["job_quote"]="不存在的引用"
    assert not api.row_recoverable_without_api(task,row)


def test_active_resume_prefers_current_batches_and_terminal_requires_exact_aggregate(tmp_path):
    task,_,row=fixture();run=run_fixture(tmp_path,[task],[row],status="running")
    stale=deepcopy(row);stale["reason"]="旧的汇总输出";write_jsonl(run/"labels.jsonl",[stale])
    _,rows,_,info=read_snapshot(run,[task]);assert rows[task["task_id"]]["reason"]==row["reason"]
    assert info["final_labels_state"]=="stale_aggregate_ignored_while_active"
    with pytest.raises(ValueError,match="完整终态"):read_snapshot(run,[task],require_complete=True)
    manifest=json.loads((run/"manifest.json").read_text());manifest["status"]="completed";write_json(run/"manifest.json",manifest)
    with pytest.raises(ValueError,match="记录不一致"):read_snapshot(run,[task],require_complete=True)


def test_supervisor_cannot_train_from_forged_success_counts(tmp_path,monkeypatch):
    from research import annotation_supervisor as supervisor
    task,_,row=fixture();task_file(tmp_path,[task]);extract,_,erow=fixture("extract","requirement_extraction");task_file(tmp_path,[extract],"extraction_tasks.jsonl")
    run_fixture(tmp_path,[task],[row],status="running")
    run_fixture(tmp_path,[extract],[erow],dirname="extraction-annotation")
    async def fake_annotate(*args,**kwargs):return {"status":"completed","rule_passed":999,"completed_tasks":999}
    async def forbidden(*args,**kwargs):pytest.fail("进度计数不能授权训练")
    monkeypatch.setattr(supervisor,"annotate",fake_annotate)
    monkeypatch.setattr(supervisor.asyncio,"create_subprocess_exec",forbidden)
    monkeypatch.setattr(supervisor,"publish_teacher_dashboard",lambda *args,**kwargs: {})
    args=SimpleNamespace(study=tmp_path,max_rounds=1,config=tmp_path/"ignored",dataset=tmp_path/"ignored",replay=tmp_path/"ignored",output=tmp_path/"output",training_python=Path("python"),gpu="0")
    asyncio.run(supervise(args))
    assert json.loads((tmp_path/"supervisor_status.json").read_text())["stage"]=="needs_review"
    assert not (tmp_path/"reconciled-annotation").exists()


def test_completion_gate_rechecks_rules_and_model_contracts(tmp_path):
    task,_,row=fixture();task_file(tmp_path,[task]);run=run_fixture(tmp_path,[task],[row])
    extract,_,erow=fixture("extract","requirement_extraction");task_file(tmp_path,[extract],"extraction_tasks.jsonl")
    run_fixture(tmp_path,[extract],[erow],dirname="extraction-annotation")
    assert completion_gate(tmp_path)["ready"]
    row["model_reviews"][0]["input_hash"]="forged";write_jsonl(run/"labels.jsonl",[row]);write_json(run/"batch-0000.json",{"batch":0,"status":"completed","rows":[row]})
    with pytest.raises(ValueError,match="审核血缘"):completion_gate(tmp_path)


def test_single_writer_lock_blocks_second_annotator_before_requests(tmp_path):
    run=tmp_path/"locked";run.mkdir()
    with (run/"annotation.lock").open("a") as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match="已有写入进程"):asyncio.run(api.annotate(tmp_path/"absent",tmp_path/"absent",run))


def test_recursive_arity_and_open_vocabulary_rules():
    task,label,_=fixture("extract","requirement_extraction")
    assert api.validate_label(task,label)["passed"] # 未登录词典的专业照样允许
    label["groups"][0]["skills"]*=2
    assert not api.validate_label(task,label)["passed"]
    label["groups"][0]["logic"]="any"
    assert api.validate_label(task,label)["errors"][0]["code"]=="duplicate_leaf_span"
    task["material"]["fragment"]+="专业领域乙。"
    label["groups"][0]["skills"][1]={"skill":"专业领域乙","quote":"专业领域乙"}
    assert api.validate_label(task,label)["passed"]
    label["groups"][0]["skills"]=[];assert not api.validate_label(task,label)["passed"]
    label["groups"]=[];assert api.validate_label(task,label)["passed"]


def study_fixture(tmp_path,active=False):
    task,_,row=fixture();task_file(tmp_path,[task]);run=run_fixture(tmp_path,[task],[row],status="running" if active else "completed")
    write_json(tmp_path/"manifest.json",{"schema":"teacher-study-v2","source_sha256":"f"*64,"profiles":1,"pairs":1,
        "extraction_tasks":0,"files":{"tasks.jsonl":file_sha(tmp_path/"tasks.jsonl")}})
    return run


def test_annotation_publisher_retains_metrics_and_source_hashes(tmp_path):
    run=study_fixture(tmp_path);metrics=tmp_path/"fixture-ranking.json";write_json(metrics,{"ndcg":0.625,"resume":"不能公布的虚构原文","api_key":"fixture-secret"})
    publish_teacher_dashboard(tmp_path,ranking_metrics=metrics)
    original=(tmp_path/"ranking_metrics.json").read_bytes()
    published=publish_teacher_dashboard(tmp_path,[run])
    assert set(published["artifacts"])=={"ranking_metrics.json","annotation_progress.json"}
    assert original==(tmp_path/"ranking_metrics.json").read_bytes()
    loaded=load_teacher_dashboard(tmp_path)
    assert loaded["files_verified"] and loaded["metrics"]["ranking"]["ndcg"]==0.625
    assert "fixture-secret" not in json.dumps(loaded) and "不能公布" not in json.dumps(loaded,ensure_ascii=False)
    with (tmp_path/"ranking_metrics.json").open("a") as stream:stream.write(" ")
    assert load_teacher_dashboard(tmp_path)["status"]=="validation_failed"


def test_stale_active_progress_is_marked_without_invalidating_prior_metrics(tmp_path):
    run=study_fixture(tmp_path,active=True);publish_teacher_dashboard(tmp_path,[run])
    dashboard=json.loads((tmp_path/"dashboard_manifest.json").read_text())
    dashboard["artifact_published_at"]["annotation_progress.json"]=(datetime.now(timezone.utc)-timedelta(seconds=125)).isoformat()
    write_json(tmp_path/"dashboard_manifest.json",dashboard)
    result=load_teacher_dashboard(tmp_path)
    assert result["files_verified"] and result["annotation_progress_stale"]
    assert result["annotation_progress_age_seconds"]>=125


def test_changed_call_input_uses_new_checkpoint_without_overwriting_old_paid_record(tmp_path,monkeypatch):
    task,label,_=fixture();tasks=task_file(tmp_path,[task]);run=tmp_path/"api-run";run.mkdir()
    old=run/"call-0000-support.json";write_json(old,{"call_input_hash":"old-other-subset","result":{"fixture":"原调用不可覆盖"}})
    original=old.read_bytes();calls=[]
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def mocked(client,endpoint,key,selected,channel,previous=None):
        calls.append(channel);return mock_call(selected,channel,{task["task_id"]:label})
    monkeypatch.setattr(api,"request_labels",mocked)
    assert asyncio.run(api.annotate(tasks,tmp_path/"ignored",run))["strict_usable"]==1
    assert old.read_bytes()==original and len(list(run.glob("call-0000-support-*.json")))==1
    assert calls==["support","transfer"]


def test_extraction_schema_repair_gets_specific_error_and_new_channel(tmp_path,monkeypatch):
    task,good,_=fixture("extract","requirement_extraction");bad=deepcopy(good);bad["groups"][0]["skills"]*=2
    tasks=task_file(tmp_path,[task]);calls=[]
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def mocked(client,endpoint,key,selected,channel,previous=None):
        calls.append(channel)
        if channel=="repair-schema-1":
            assert previous[task["task_id"]]["checks"]["errors"][0]["code"]=="invalid_arity"
            return mock_call(selected,channel,{task["task_id"]:good})
        return mock_call(selected,channel,{task["task_id"]:bad})
    monkeypatch.setattr(api,"request_labels",mocked)
    manifest=asyncio.run(api.annotate(tasks,tmp_path/"ignored",tmp_path/"api-run"))
    assert manifest["strict_usable"]==1
    assert calls==["support","transfer","adjudicate","repair-schema-1"]
    assert api.prompt_for("repair-schema-1")!=api.prompt_for("repair-1")


def test_null_grade_has_bounded_repairs_and_never_becomes_usable(tmp_path,monkeypatch):
    task,label,_=fixture();label["grade"]=None;tasks=task_file(tmp_path,[task]);calls=[]
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def mocked(client,endpoint,key,selected,channel,previous=None):
        calls.append(channel);return mock_call(selected,channel,{task["task_id"]:label})
    monkeypatch.setattr(api,"request_labels",mocked)
    manifest=asyncio.run(api.annotate(tasks,tmp_path/"ignored",tmp_path/"api-run"))
    assert manifest["strict_usable"]==0 and manifest["usable_relevance"]==0
    assert calls==["support","transfer","repair-1","repair-2"]


def test_shared_rate_gate_stops_both_queues_after_one_429_probe(tmp_path,monkeypatch):
    tasks_a=[fixture("a"+str(i))[0] for i in range(8)]
    tasks_b=[fixture("b"+str(i),"requirement_extraction")[0] for i in range(8)]
    file_a=task_file(tmp_path,tasks_a);file_b=task_file(tmp_path,tasks_b,"extraction_tasks.jsonl");calls=[]
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    async def limited(client,endpoint,key,selected,channel,previous=None):
        calls.append(channel);await asyncio.sleep(0);raise api.RateLimitedError(300)
    monkeypatch.setattr(api,"request_labels",limited)
    async def both():
        gate=api.RateLimitGate()
        return await asyncio.gather(api.annotate(file_a,tmp_path/"ignored",tmp_path/"annotation",2,8,rate_gate=gate),
            api.annotate(file_b,tmp_path/"ignored",tmp_path/"extraction",2,8,rate_gate=gate))
    manifests=asyncio.run(both())
    assert calls==["support"]
    assert all(result["status"]=="rate_limited" and result["automatic_restart"] is False for result in manifests)
    assert all(result["completed_tasks"]==0 and result["strict_usable"]==0 for result in manifests)
    with pytest.raises(api.RateLimitedError):asyncio.run(api.annotate(file_a,tmp_path/"ignored",tmp_path/"annotation"))
    assert calls==["support"] # Retry-After期间恢复也不触发探针


def test_429_honors_retry_after_without_sleep_or_retry():
    task,_,_=fixture();calls=[]
    class Client:
        async def post(self,*args,**kwargs):
            calls.append(1);return SimpleNamespace(status_code=429,headers={"retry-after":"720"})
    with pytest.raises(api.RateLimitedError) as failure:
        asyncio.run(api.request_labels(Client(),"https://mock.invalid","fixture-no-key",[task],"support"))
    assert calls==[1] and failure.value.retry_after_seconds==720
    assert api.retry_after_seconds(None)==300


def test_supervisor_exits_first_round_on_429_without_training(tmp_path,monkeypatch):
    from research import annotation_supervisor as supervisor
    calls=[]
    async def limited(*args,**kwargs):calls.append(1);return api.RateLimitedError(300).status()
    async def forbidden(*args,**kwargs):pytest.fail("429后不得进入训练")
    monkeypatch.setattr(supervisor,"annotate",limited)
    monkeypatch.setattr(supervisor.asyncio,"create_subprocess_exec",forbidden)
    monkeypatch.setattr(supervisor,"publish_teacher_dashboard",lambda *args,**kwargs: {})
    args=SimpleNamespace(study=tmp_path,max_rounds=3,config=tmp_path/"ignored",dataset=tmp_path/"ignored",replay=tmp_path/"ignored",output=tmp_path/"output",training_python=Path("python"),gpu="0")
    asyncio.run(supervise(args));status=json.loads((tmp_path/"supervisor_status.json").read_text())
    assert status["stage"]=="rate_limited" and status["round"]==1 and len(calls)==2
    assert status["automatic_restart"] is False


def test_rate_limit_dashboard_shows_cooldown_and_preserves_checkpoint_counts(tmp_path):
    run=study_fixture(tmp_path,active=True)
    manifest=json.loads((run/"manifest.json").read_text());manifest.update(api.RateLimitedError(300).status());write_json(run/"manifest.json",manifest)
    publish_teacher_dashboard(tmp_path,[run]);loaded=load_teacher_dashboard(tmp_path)
    row=loaded["annotation_runs"][0]
    assert row["status"]=="rate_limited" and row["completed_tasks"]==1
    assert 295<=row["retry_wait_remaining_seconds"]<=300 and row["automatic_restart"] is False


def test_interrupted_repair_keeps_original_completed_rows_in_aggregate(tmp_path,monkeypatch):
    task,_,row=fixture()
    row["evidence"][0]["job_quote"]="不存在的引用，不能自动纠正"
    row["rule_validation"]=api.validate_label(task,row)
    tasks=task_file(tmp_path,[task]);run=run_fixture(tmp_path,[task],[row])
    monkeypatch.setattr(api,"read_config",lambda _: ("https://mock.invalid","fixture-no-key"))
    calls=[]
    async def limited(*args,**kwargs):
        calls.append(1)
        raise api.RateLimitedError(300)
    monkeypatch.setattr(api,"request_labels",limited)
    result=asyncio.run(api.annotate(tasks,tmp_path/"ignored",run,batch_size=2,concurrency=1))
    assert len(calls)==1 and result["status"]=="rate_limited"
    assert result["completed_tasks"]==1 and result["strict_usable"]==0
    assert json.loads((run/"labels.jsonl").read_text())==row
    _,snapshot,_,_=read_snapshot(run,[task])
    assert snapshot[task["task_id"]]==row


def test_requirement_report_failure_stops_supervised_training(tmp_path,monkeypatch):
    from research import annotation_supervisor as supervisor
    from research import evaluate_requirements
    async def completed(*args,**kwargs):return {"status":"completed"}
    async def forbidden(*args,**kwargs):pytest.fail("抽取评测来源失败后不得启动监督训练")
    def failed_report(*args,**kwargs):raise ValueError("虚构来源校验失败")
    monkeypatch.setattr(supervisor,"annotate",completed)
    monkeypatch.setattr(supervisor,"prepare_derived_annotations",lambda _: tmp_path/"derived")
    monkeypatch.setattr(supervisor,"completion_gate",lambda *args: {"ready":True})
    monkeypatch.setattr(supervisor,"publish_teacher_dashboard",lambda *args,**kwargs: {})
    monkeypatch.setattr(supervisor.asyncio,"create_subprocess_exec",forbidden)
    monkeypatch.setattr(evaluate_requirements,"run",failed_report)
    args=SimpleNamespace(study=tmp_path,max_rounds=1,config=tmp_path/"ignored",dataset=tmp_path/"ignored",
        replay=tmp_path/"ignored",output=tmp_path/"output",training_python=Path("python"),gpu="0")
    with pytest.raises(ValueError,match="来源校验失败"):asyncio.run(supervise(args))
    assert json.loads((tmp_path/"supervisor_status.json").read_text())["stage"]=="failed"
