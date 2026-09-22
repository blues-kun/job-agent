"""全部为虚构引用与API结果；字段定位纠正不等于模型判断或人工金标。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import research
research.__path__.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from research.annotation_workflow import seal_task
from research.api_annotation import validate_label
from research.reconcile_annotations import reconcile, reconcile_label, exact_offsets
from research.review_contracts import digest, file_sha, validate_qrel


def fixture(task_id="fixture-task"):
    material={"resume":"本人使用Python开发订单接口并修复事务问题。","preferences":{},
              "job":{"description":"虚构科技有限公司。任职要求：使用Python开发订单接口，定位事务问题。",
                     "requirements":"本科，1年以上工作经验。"}}
    task=seal_task({"task_id":task_id,"kind":"relevance","query_id":"fixture-q","job_id":"fixture-j",
        "profile_family_id":"fixture-pf","job_family_id":"fixture-jf","split":"test",
        "material":material,"input_hash":digest(material)})
    audit={"model":"gpt-5.6-terra","model_revision":"provider-not-disclosed","prompt_hash":"p"*64,
           "input_hash":task["input_hash"],"request_id":"fixture-request","grade":2}
    label={**{key:value for key,value in task.items() if key!="material"},"grade":2,"reason":"虚构原判断，不能被字段修复改写。",
        "hard_negative":False,"evidence":[{"resume_quote":"使用Python开发订单接口","job_quote":"使用Python开发订单接口","job_field":"requirements"}],
        "label_source":"llm_reviewed","model_reviews":[{**audit,"channel":"support"},{**audit,"channel":"transfer"}],
        "model_adjudication":{**audit,"channel":"adjudicate","rationale":"虚构仲裁，仅用于测试"},
        "review_run_id":"fixture-run","model":"gpt-5.6-terra"}
    label["rule_validation"]=validate_label(task,label)
    return task,label


def bind(task,label):
    task=deepcopy(task);task["input_hash"]=digest(task["material"]);task=seal_task(task)
    label=deepcopy(label)
    for key in ("input_hash","task_hash"):label[key]=task[key]
    for review in label["model_reviews"]+[label["model_adjudication"]]:review["input_hash"]=task["input_hash"]
    label["rule_validation"]=validate_label(task,label)
    return task,label


def make_run(tmp_path,tasks,labels,final=False):
    tasks_path=tmp_path/"tasks.jsonl"
    tasks_path.write_text("".join(json.dumps(task,ensure_ascii=False)+"\n" for task in tasks),encoding="utf-8")
    run=tmp_path/"api-run";run.mkdir()
    (run/"manifest.json").write_text(json.dumps({"schema":"job-agent-api-labels-v2","run_id":"fixture-run",
        "tasks":len(tasks),"tasks_hash":digest(tasks),"label_source":"llm_reviewed","model":"gpt-5.6-terra",
        "reasoning_effort":"max","status":"completed" if len(labels)==len(tasks) else "running"}),encoding="utf-8")
    (run/"batch-0000.json").write_text(json.dumps({"batch":0,"status":"completed","rows":labels,"calls":[]},ensure_ascii=False),encoding="utf-8")
    if final:(run/"labels.jsonl").write_text("".join(json.dumps(label,ensure_ascii=False)+"\n" for label in labels),encoding="utf-8")
    return tasks_path,run


def test_unique_exact_field_fix_preserves_model_judgment_and_original_failure():
    task,label=fixture();original=deepcopy(label)
    derived,stats=reconcile_label(task,label)
    assert stats["changed"] and stats["strict_qrel_usable"] and stats["field_corrections"]==1
    assert derived["evidence"][0]["job_field"]=="description"
    assert derived["evidence"][0]["job_quote"]==label["evidence"][0]["job_quote"]
    for key,value in original.items():
        if key not in {"evidence","rule_validation"}:assert derived[key]==value
    assert label==original
    audit=derived["reconciliation"]
    assert audit["original_label_sha256"]==digest(original) and audit["input_hash"]==task["input_hash"]
    assert audit["original_rule_validation"]["checks"]["quoted_support"] is False
    assert audit["model_audit_sha256"]==digest({key:original.get(key) for key in ("model_reviews","model_adjudication")})
    assert audit["human_confirmed"] is False and audit["grade_changed"] is False
    change=audit["changes"][0]
    for offset in change["offsets"]:
        assert task["material"]["job"][change["corrected_field"]][offset["start"]:offset["end"]]==label["evidence"][0]["job_quote"]
    validate_qrel(derived,allow_model_labels=True)


def test_ambiguous_or_nonliteral_quotes_remain_failed_without_guessing():
    task,label=fixture()
    task["material"]["job"]["requirements"]+="使用Python开发订单接口"
    label["evidence"][0]["job_field"]="任职要求"
    task,label=bind(task,label)
    derived,stats=reconcile_label(task,label)
    assert not stats["changed"] and not stats["strict_qrel_usable"]
    assert derived["evidence"][0]["job_field"]=="任职要求"
    assert stats["unresolved_reasons"]=={"quote_in_multiple_fields":1}
    task,label=fixture();label["evidence"][0]["job_quote"]="使用 Python 开发订单接口"
    derived,stats=reconcile_label(task,label)
    assert not stats["changed"] and not stats["rule_passed_after"]
    assert stats["unresolved_reasons"]=={"quote_not_found_exactly":1}


def test_repeated_occurrences_are_all_recorded_and_other_false_quote_is_not_fixed():
    assert exact_offsets("AAAA","AA")==[{"start":0,"end":2},{"start":1,"end":3},{"start":2,"end":4}]
    task,label=fixture();task["material"]["job"]["description"]+="再次使用Python开发订单接口。"
    label["evidence"][0]["resume_quote"]="并不存在的本人实践"
    task,label=bind(task,label)
    derived,stats=reconcile_label(task,label)
    assert len(derived["reconciliation"]["changes"][0]["offsets"])==2
    assert stats["changed"] and not stats["rule_passed_after"] and not stats["strict_qrel_usable"]
    assert derived["evidence"][0]["resume_quote"]==label["evidence"][0]["resume_quote"]


def test_full_derived_version_keeps_raw_files_and_exports_hashes(tmp_path):
    task,label=fixture();tasks,run=make_run(tmp_path,[task],[label],final=True)
    before={path.name:path.read_bytes() for path in run.iterdir()}
    output=tmp_path/"derived";summary=reconcile(tasks,run,output)
    assert summary["labels_with_corrected_field"]==1 and summary["strict_qrel_usable"]==1
    assert {path.name:path.read_bytes() for path in run.iterdir()}==before
    derived=json.loads((output/"labels.jsonl").read_text())
    assert derived["reconciliation"]["original_label_sha256"]==digest(label)
    manifest=json.loads((output/"manifest.json").read_text())
    assert manifest["raw_labels_overwritten"] is False and manifest["direct_training_allowed"] is False
    assert all(file_sha(output/name)==sha for name,sha in manifest["files"].items())
    assert "虚构科技有限公司" not in json.dumps(summary,ensure_ascii=False)


def test_partial_run_only_allows_progress_without_labels(tmp_path):
    first,label=fixture();second,_=fixture("fixture-second")
    tasks,run=make_run(tmp_path,[first,second],[label])
    with pytest.raises(ValueError,match="尚未齐备"):reconcile(tasks,run,tmp_path/"refused")
    assert not (tmp_path/"refused").exists()
    summary=reconcile(tasks,run,tmp_path/"progress",progress=True)
    assert summary["pending_tasks"]==1 and summary["progress_only"] is True
    assert not (tmp_path/"progress/labels.jsonl").exists()
    assert set(path.name for path in (tmp_path/"progress").iterdir())=={"progress.json","manifest.json"}


def test_input_lineage_and_batch_final_disagreement_are_rejected(tmp_path):
    task,label=fixture();changed=deepcopy(label);changed["input_hash"]="wrong"
    with pytest.raises(ValueError,match="SHA不符"):reconcile_label(task,changed)
    tasks,run=make_run(tmp_path,[task],[label],final=True)
    changed=deepcopy(label);changed["reason"]="被更改"
    (run/"labels.jsonl").write_text(json.dumps(changed)+"\n")
    with pytest.raises(ValueError,match="记录不一致"):reconcile(tasks,run,tmp_path/"refused")
    assert not (tmp_path/"refused").exists()


def test_missing_grade_or_model_arbitration_stays_unusable():
    task,label=fixture();label["grade"]=None
    derived,stats=reconcile_label(task,label)
    assert stats["changed"] and not stats["strict_qrel_usable"] and derived["grade"] is None
    task,label=fixture();label["model_reviews"][0]["grade"]=0;label.pop("model_adjudication")
    derived,stats=reconcile_label(task,label)
    assert stats["rule_passed_after"] and not stats["strict_qrel_usable"]
    assert "model_adjudication" not in derived
