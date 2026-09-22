"""全部为虚构材料；验证工作台导出/仲裁合同与教师快照，不执行真实标注或网络请求。"""
from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest
import job_agent
import research
job_agent.__path__.insert(0,str(Path(__file__).resolve().parents[1]/"job_agent"))
research.__path__.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from job_agent.research_store import ResearchStore
from research.annotation_workflow import seal_task, review_status, freeze_dataset, validate_tasks
from research.review_contracts import digest, file_sha, read_jsonl
from research.teacher_dashboard import publish_teacher_dashboard, load_teacher_dashboard


def rows(path,values):
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in values),encoding="utf-8")


def store_fixture(tmp_path,monkeypatch):
    for key in ("JOB_AGENT_BENCHMARK_DIR","JOB_AGENT_SYNTHESIS_DIR","JOB_AGENT_RETRIEVAL_REVIEW_DIR","JOB_AGENT_TEACHER_STUDY_DIR"):
        monkeypatch.delenv(key,raising=False)
    dataset=tmp_path/"dataset";dataset.mkdir()
    text="要求Python或Java，且掌握SQL。"
    job={"job_id":"fixture-job","job_family_id":"fixture-job-family","snapshot":"fixture-source","split":"dev",
         "title":"虚构岗位","category":"开发","requirements":"本科","description":text,
         "salary_raw":"10-20K","address":"深圳","skills":{},"groups":[],"source_record_ids":["fixture-record"]}
    extraction={"task_id":"extract","kind":"requirement_extraction","job_id":job["job_id"],"split":"dev",
                "evidence":{"field":"description","start":0,"end":len(text),"quote":text},"stage":"pilot"}
    query={"query_id":"fixture-query","profile_family_id":"fixture-profile-family","split":"dev","purpose":"evaluation",
           "text":"虚构简历：使用Python开发订单接口，使用SQL查询订单。","preferences":{"city":"深圳","intent":"开发"}}
    clarification={"query_id":"fixture-stress","profile_family_id":"fixture-stress-family","split":"dev","purpose":"evaluation",
           "text":"虚构简历：想找工作。","preferences":{},"scenario_group":"clarification_stress"}
    rows(dataset/"jobs.jsonl",[job]);rows(dataset/"annotation_tasks.jsonl",[extraction])
    (dataset/"manifest.json").write_text(json.dumps({"source_sha256":"fixture-source","files":{name:{"sha256":file_sha(dataset/name)} for name in ["jobs.jsonl","annotation_tasks.jsonl"]}}))
    benchmark=dataset/"benchmark";benchmark.mkdir()
    relevance={"task_id":"relevance","kind":"relevance","query_id":query["query_id"],"job_id":job["job_id"],"split":"dev","stage":"pilot","pool_sources":["不应对评审显示"]}
    action={"task_id":"clarify","kind":"clarification","query_id":clarification["query_id"],"split":"dev","stage":"pilot"}
    rows(benchmark/"tasks.jsonl",[relevance,action]);rows(benchmark/"queries.jsonl",[query,clarification])
    (benchmark/"manifest.json").write_text(json.dumps({"config":{"jobs_sha256":file_sha(dataset/"jobs.jsonl")},"files":{name:file_sha(benchmark/name) for name in ["tasks.jsonl","queries.jsonl"]}}))
    return ResearchStore(dataset,tmp_path/"private")


def nested_extraction():
    return {"groups":[{"logic":"all","modality":"required","skills":[{"skill":"SQL","quote":"SQL"}],
        "children":[{"logic":"any","modality":"required","skills":[{"skill":"Python","quote":"Python"},{"skill":"Java","quote":"Java"}]}]}]}


def independent_reviews(store):
    for alias in ("fixture-a","fixture-b"):
        for task_id,task in store.review_tasks.items():
            extra={"independent":True,"task_hash":task["task_hash"]}
            if task["kind"]=="relevance":store.annotate(task_id,alias,2,"","完全虚构测试",**extra)
            elif task["kind"]=="requirement_extraction":store.annotate(task_id,alias,None,"正确","完全虚构测试",nested_extraction(),**extra)
            else:store.annotate(task_id,alias,None,"","完全虚构测试",action="clarify",**extra)


def fixture_registry():
    return {"attestation":{"record_id":"fixture-only-record","attested_by":"fixture-only-coordinator",
        "attested_at":"2026-09-22T00:00:00Z","identity_check_performed":True,"independent_review_process_confirmed":True},
        "reviewers":[{"reviewer_id":"fixture-"+name,"person_id":"fixture-person-"+name,"identity_verified":True} for name in ("a","b")]}


def test_nested_workbench_export_and_explicit_arbitration_complete_roundtrip(tmp_path,monkeypatch):
    store=store_fixture(tmp_path,monkeypatch)
    assert len(store.review_tasks)==3 and not store.review_blockers
    seen=store.task("fixture-a","relevance")["task"]
    assert seen["review_contract_ready"] and len(seen["task_hash"])==64 and "pool_sources" not in seen
    independent_reviews(store)
    bundle=tmp_path/"bundle";result=store.export_review_bundle(bundle)
    assert result["reviews"]==6 and result["human_gold_created"] is False
    tasks,reviews=read_jsonl(bundle/"tasks.jsonl"),read_jsonl(bundle/"reviews.jsonl")
    validate_tasks(tasks)
    template=json.loads((bundle/"reviewer_registry.template.json").read_text())
    with pytest.raises(ValueError,match="确认记录"):review_status(tasks,reviews,template)
    registry=fixture_registry();status=review_status(tasks,reviews,registry)
    assert status["counts"]=={"agreement_pending_adjudication":3}
    with pytest.raises(ValueError,match="未完成"):
        freeze_dataset(tasks,read_jsonl(bundle/"queries.jsonl"),reviews,registry,[],tmp_path/"pending")
    decisions=[]
    for item in status["tasks"]:
        decisions.append({"task_id":item["task_id"],"task_hash":item["task_hash"],"source":"explicit_human_adjudication",
            "decision":"approved","human_confirmed":True,"review_hashes":{row["review_id"]:row["review_sha256"] for row in item["reviews"]},
            "adjudicator_id":"fixture-a","adjudication_id":"fixture-only-decision-"+item["task_id"],
            "created_at":"2026-09-22T01:00:00Z","rationale":"仅在虚构夹具中模拟明确仲裁，不是研究金标",**item["reviews"][0]["label"]})
    result=freeze_dataset(tasks,read_jsonl(bundle/"queries.jsonl"),reviews,registry,decisions,tmp_path/"fixture-frozen")
    assert result["gold_count"]==3 and result["stage"]=="pilot"
    extracted=read_jsonl(tmp_path/"fixture-frozen/extraction_gold.jsonl")[0]
    assert extracted["extraction"]["groups"][0]["children"][0]["logic"]=="any"


def test_old_unbound_or_nonindependent_reviews_are_not_upgraded(tmp_path,monkeypatch):
    store=store_fixture(tmp_path,monkeypatch)
    store.annotate("relevance","fixture-a",2,"","旧调用没有任务hash")
    result=store.export_review_bundle(tmp_path/"legacy")
    assert result["reviews"]==0 and result["legacy_reviews"]==1
    with pytest.raises(ValueError,match="任务版本"):
        store.annotate("relevance","fixture-a",2,"","",independent=True,task_hash="changed")
    store.annotate("relevance","fixture-a",2,"","未声明独立",task_hash=store.material_hashes["relevance"])
    store.export_review_bundle(tmp_path/"nonindependent")
    with pytest.raises(ValueError,match="独立盲审"):
        review_status(read_jsonl(tmp_path/"nonindependent/tasks.jsonl"),read_jsonl(tmp_path/"nonindependent/reviews.jsonl"),fixture_registry())


def test_recursive_extraction_rejects_duplicate_or_unbounded_tree(tmp_path,monkeypatch):
    store=store_fixture(tmp_path,monkeypatch);tree=nested_extraction()
    tree["groups"][0]["children"][0]["skills"][1]={"skill":"Python","quote":"Python"}
    with pytest.raises(ValueError,match="重复"):store.annotate("extract","fixture-a",None,"正确","",tree)
    nested={"logic":"single","modality":"required","skills":[{"skill":"SQL","quote":"SQL"}]}
    for _ in range(10):nested={"logic":"single","modality":"required","skills":[],"children":[nested]}
    with pytest.raises(ValueError,match="上限"):store.annotate("extract","fixture-a",None,"正确","",{"groups":[nested]})


def teacher_fixture(tmp_path):
    study=tmp_path/"study";study.mkdir()
    material={"resume":"使用Python开发订单接口。","preferences":{},"job":{"description":"使用Python开发订单接口。","requirements":"本科"}}
    task=seal_task({"task_id":"fixture-teacher","kind":"relevance","query_id":"q","profile_family_id":"pf","job_id":"j",
        "job_family_id":"jf","split":"test","material":material,"input_hash":digest(material)})
    rows(study/"tasks.jsonl",[task]);rows(study/"profiles.jsonl",[{"query_id":"q","text":material["resume"]}])
    (study/"manifest.json").write_text(json.dumps({"schema":"teacher-study-v2","source_sha256":"fixture-source","profiles":1,"pairs":1,
        "extraction_tasks":0,"splits":{"test":1},"source_kinds":{"model_authored_fiction":1},
        "files":{name:file_sha(study/name) for name in ["tasks.jsonl","profiles.jsonl"]}}))
    run=tmp_path/"api-run";run.mkdir()
    call={"model":"gpt-5.6-terra","model_revision":"provider-not-disclosed","prompt_hash":"p"*64,"input_hash":task["input_hash"],"request_id":"fixture-request","grade":2}
    label={**{key:value for key,value in task.items() if key!="material"},"grade":2,"reason":"这里不得向看板泄漏fixture-secret",
        "hard_negative":False,"evidence":[{"resume_quote":"使用Python","job_quote":"使用Python","job_field":"description"}],
        "label_source":"llm_reviewed","model_reviews":[{**call,"channel":"support"},{**call,"channel":"transfer"}],
        "rule_validation":{"passed":True,"validator_version":"fixture-rules","input_hash":task["input_hash"],"checks":{"quote":True}}}
    rows(run/"labels.jsonl",[label])
    (run/"manifest.json").write_text(json.dumps({"tasks_hash":digest([task]),"tasks":1,"status":"completed","label_source":"llm_reviewed",
        "model":"gpt-5.6-terra","reasoning_effort":"max","LLM_API_KEY":"fixture-secret"}))
    return study,run


def test_teacher_dashboard_hashes_progress_and_redacts_nonaggregate_content(tmp_path,monkeypatch):
    study,run=teacher_fixture(tmp_path)
    metrics=tmp_path/"metrics.json"
    metrics.write_text(json.dumps({"models":{"bm25":{"summary":{"ndcg_at_k":.5}}},"api_key":"fixture-secret",
        "raw_resume":"使用Python开发订单接口。","variants":[{"mode":"graph","seed":42,"evaluation":{"dev":{"ndcg_at_10":.75}}}]}))
    report=publish_teacher_dashboard(study,[run],ranking_metrics=metrics,graph_metrics=metrics)
    assert report["human_gold_created"] is False
    output=load_teacher_dashboard(study,"fixture-source")
    assert output["status"]=="dashboard_verified" and output["annotation_runs"][0]["usable_relevance"]==1
    assert output["annotation_runs"][0]["model"]=="gpt-5.6-terra"
    assert output["metrics"]["graph"]["variants"][0]["mode"]=="graph"
    assert "fixture-secret" not in json.dumps(output,ensure_ascii=False)
    assert "使用Python开发订单接口" not in json.dumps(output,ensure_ascii=False)
    (study/"ranking_metrics.json").write_text('{"metrics":{"ndcg_at_k":1.0}}')
    assert load_teacher_dashboard(study,"fixture-source")["status"]=="validation_failed"


def test_teacher_publisher_rejects_wrong_study_version_and_store_exposes_only_status(tmp_path,monkeypatch):
    study,run=teacher_fixture(tmp_path)
    manifest=json.loads((run/"manifest.json").read_text());manifest["tasks_hash"]="wrong"
    (run/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="任务hash"):publish_teacher_dashboard(study,[run])
    assert not (study/"dashboard_manifest.json").exists()
    store=store_fixture(tmp_path,monkeypatch)
    store.teacher_directory=study
    assert store.summary()["teacher_study"]["status"]=="materials_verified"
    assert load_teacher_dashboard(study,"other-source")["status"]=="validation_failed"


def test_verified_teacher_material_enters_blind_queue_without_teacher_answers(tmp_path,monkeypatch):
    store=store_fixture(tmp_path,monkeypatch)
    job=store.jobs["fixture-job"]
    study=tmp_path/"teacher-study";study.mkdir()
    query={"query_id":"teacher-new-q","profile_family_id":"teacher-new-family","split":"dev","purpose":"ranker_training",
           "source_kind":"model_authored_fiction","text":"虚构材料：使用Python完成订单接口。","preferences":{}}
    material={"resume":query["text"],"preferences":{},"job":{"description":job["description"],"requirements":job["requirements"]}}
    task=seal_task({"task_id":"teacher-new-task","kind":"relevance","query_id":query["query_id"],"profile_family_id":query["profile_family_id"],
        "job_id":job["job_id"],"job_family_id":job["job_family_id"],"split":"dev","material":material,"input_hash":digest(material),
        "source_job_sha256":digest(job),"source_query_sha256":digest(query),"stage":"teacher_relative_research"})
    rows(study/"profiles.jsonl",[query]);rows(study/"tasks.jsonl",[task])
    (study/"manifest.json").write_text(json.dumps({"schema":"teacher-study-v2","source_sha256":"fixture-source",
        "profiles":1,"pairs":1,"extraction_tasks":0,"splits":{"dev":1},"source_kinds":{"model_authored_fiction":1},
        "files":{name:file_sha(study/name) for name in ["profiles.jsonl","tasks.jsonl"]}}))
    # 原始教师答案即使存在也不进入队列。
    (study/"labels.jsonl").write_text('{"task_id":"teacher-new-task","grade":3,"reason":"教师答案不应展示"}\n')
    monkeypatch.setenv("JOB_AGENT_TEACHER_STUDY_DIR",str(study))
    loaded=ResearchStore(store.directory,tmp_path/"another-private")
    assert loaded.teacher_review_queue["status"]=="verified_blind_materials"
    assert loaded.review_tasks[task["task_id"]]["task_hash"]==task["task_hash"]
    assert "教师答案" not in json.dumps(loaded.tasks[task["task_id"]],ensure_ascii=False)
    assert loaded.teacher_summary()["human_review_queue"]["teacher_answers_exposed"] is False


def test_teacher_r2_to_r4_bridge_preserves_sealed_task_and_rejects_raw_field_change(tmp_path,monkeypatch):
    current=store_fixture(tmp_path,monkeypatch)
    old=tmp_path/"dataset-r2";old.mkdir()
    source_job={**current.jobs["fixture-job"],"parser_version":"fixture-r2","skills":{"Python":{"level":"熟悉"}}}
    rows(old/"jobs.jsonl",[source_job]);(old/"annotation_tasks.jsonl").write_bytes((current.directory/"annotation_tasks.jsonl").read_bytes())
    def source_manifest():
        (old/"manifest.json").write_text(json.dumps({"source_sha256":"fixture-source","files":{name:{"sha256":file_sha(old/name)} for name in ["jobs.jsonl","annotation_tasks.jsonl"]}}))
    source_manifest()
    study=tmp_path/"teacher-r2";study.mkdir()
    query={"query_id":"teacher-r2-query","profile_family_id":"teacher-r2-family","split":"dev","purpose":"ranker_training",
           "text":"虚构画像：使用Python完成订单接口。","preferences":{}}
    def save_study():
        material={"resume":query["text"],"preferences":{},"job":{"description":source_job["description"],"requirements":source_job["requirements"]}}
        task=seal_task({"task_id":"teacher-r2-task","kind":"relevance","query_id":query["query_id"],"profile_family_id":query["profile_family_id"],
            "job_id":source_job["job_id"],"job_family_id":source_job["job_family_id"],"split":"dev","material":material,"input_hash":digest(material),
            "source_job_sha256":digest(source_job),"source_query_sha256":digest(query)})
        rows(study/"tasks.jsonl",[task]);rows(study/"profiles.jsonl",[query])
        (study/"replay_manifest.json").write_text(json.dumps({"jobs_sha256":file_sha(old/"jobs.jsonl")}))
        (study/"manifest.json").write_text(json.dumps({"schema":"teacher-study-v2","dataset":str(old),"source_sha256":"fixture-source",
            "profiles":1,"pairs":1,"extraction_tasks":0,"splits":{"dev":1},"source_kinds":{"model_authored_fiction":1},
            "files":{name:file_sha(study/name) for name in ["tasks.jsonl","profiles.jsonl","replay_manifest.json"]}}))
        return task
    task=save_study();monkeypatch.setenv("JOB_AGENT_TEACHER_STUDY_DIR",str(study))
    loaded=ResearchStore(current.directory,tmp_path/"bridge-private")
    assert loaded.teacher_review_queue["status"]=="verified_blind_materials"
    assert loaded.review_tasks[task["task_id"]]["task_hash"]==task["task_hash"]
    lineage=loaded.task_source_lineage[task["task_id"]]
    assert lineage["teacher_source_manifest_sha256"]==file_sha(old/"manifest.json")
    assert lineage["teacher_source_job_sha256"]==digest(source_job)
    assert lineage["teacher_parser_version"]=="fixture-r2" and lineage["current_parser_version"] is None
    assert lineage["teacher_source_job_sha256"]!=lineage["current_job_sha256"]
    loaded.export_review_bundle(tmp_path/"bridge-export",[task["task_id"]])
    saved=read_jsonl(tmp_path/"bridge-export/tasks.jsonl")[0]
    assert saved["task_hash"]==task["task_hash"] and saved["source_job_sha256"]==task["source_job_sha256"]
    assert json.loads((tmp_path/"bridge-export/task_source_lineage.json").read_text())[task["task_id"]]["sealed_task_unchanged"] is True
    # 即使所有hash重新签出，原始职责改动仍不能冒充仅解析器换版。
    source_job["description"]+="新增原始职责。"
    rows(old/"jobs.jsonl",[source_job]);source_manifest();save_study()
    rejected=ResearchStore(current.directory,tmp_path/"rejected-private")
    assert rejected.teacher_review_queue["status"]=="validation_failed"
    assert task["task_id"] not in rejected.tasks
