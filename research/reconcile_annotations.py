"""只纠正教师引用的逐字字段定位；原始标签不变更，不推断等级或改写引用。"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from research.api_annotation import validate_label, VERSION as RULE_VERSION
from research.annotation_workflow import validate_tasks
from research.common import private_path
from research.review_contracts import (digest, file_sha, new_private_directory, read_jsonl,
                                      validate_qrel, write_json, write_jsonl)

VERSION="exact-quote-field-reconciliation-v1"
JOB_FIELDS=("description","requirements")


def exact_offsets(text,quote):
    """保留全部精确出现位置；字段唯一不等于该字段内只有一次出现。"""
    offsets=[];start=0
    while isinstance(text,str) and isinstance(quote,str) and quote:
        index=text.find(quote,start)
        if index<0:break
        offsets.append({"start":index,"end":index+len(quote)})
        start=index+1
    return offsets


def checked_rule(task,label):
    try:return validate_label(task,label)
    except (ValueError,TypeError,KeyError,AttributeError):
        return {"passed":False,"validator_version":RULE_VERSION,"input_hash":task["input_hash"],
                "checks":{"schema_exception":False}}


def reconcile_label(task,original):
    """只改变evidence[i].job_field和派生审核元数据；不改任何模型判断。"""
    if task.get("kind")!="relevance":raise ValueError("字段纠正仅处理人岗相关性任务")
    if original.get("label_source")!="llm_reviewed":raise ValueError("只接受明确的模型原始标签")
    if original.get("input_hash")!=task["input_hash"] or original.get("task_hash")!=task["task_hash"]:
        raise ValueError("标签与原始任务输入或内容SHA不符")
    for key in ("task_id","query_id","job_id","profile_family_id","job_family_id","split"):
        if original.get(key)!=task.get(key):raise ValueError("原始标签身份与任务不一致")
    if "reconciliation" in original:raise ValueError("输入必须为原始API标签，不叠加已有字段修复")
    result=deepcopy(original);changes=[];unresolved=[]
    evidence=result.get("evidence",[])
    if not isinstance(evidence,list):evidence=[];unresolved.append({"reason":"evidence_not_list"})
    fields=task["material"]["job"]
    for index,item in enumerate(evidence):
        if not isinstance(item,dict):
            unresolved.append({"evidence_index":index,"reason":"evidence_item_not_object"});continue
        old_field=item.get("job_field");quote=item.get("job_quote")
        if old_field in JOB_FIELDS and exact_offsets(fields.get(old_field,""),quote):continue
        matches={field:exact_offsets(fields.get(field,""),quote) for field in JOB_FIELDS}
        supported=[field for field,offsets in matches.items() if offsets]
        if len(supported)==1:
            new_field=supported[0]
            item["job_field"]=new_field
            changes.append({"evidence_index":index,"original_field":old_field,"corrected_field":new_field,
                "offsets":matches[new_field],"quote_sha256":digest(quote),
                "reason":"exact_quote_in_one_allowed_job_field"})
        else:
            unresolved.append({"evidence_index":index,"reason":"quote_in_multiple_fields" if len(supported)>1 else "quote_not_found_exactly"})
    before_rule=checked_rule(task,original);after_rule=checked_rule(task,result)
    audit_keys=("model_reviews","model_adjudication")
    audit_sha=digest({key:original.get(key) for key in audit_keys})
    # 明确断言允许的差异，避免后续维护意外把修复变成改标签。
    expected=deepcopy(original)
    for change in changes:
        expected["evidence"][change["evidence_index"]]["job_field"]=change["corrected_field"]
    if result!=expected or audit_sha!=digest({key:result.get(key) for key in audit_keys}):
        raise AssertionError("字段纠正越过允许修改范围")
    result["rule_validation"]=after_rule
    result["reconciliation"]={"version":VERSION,"original_label_sha256":digest(original),
        "label_sha256_method":"canonical_json_sha256","input_hash":task["input_hash"],"task_hash":task["task_hash"],
        "original_rule_validation":deepcopy(original.get("rule_validation")),"original_rule_rechecked":before_rule,
        "changes":changes,"unresolved":unresolved,"model_audit_sha256":audit_sha,"human_confirmed":False,
        "grade_changed":False,"quote_text_changed":False,
        "notice":"仅做逐字原文的字段定位更正；不验证语义支持、能力真实性或人工一致性。"}
    usable=False
    try:
        validate_qrel(result,allow_model_labels=True);usable=True
    except (ValueError,TypeError,KeyError,AttributeError):pass
    stored_rule=original.get("rule_validation") or {}
    return result,{"changed":bool(changes),"field_corrections":len(changes),
        "stored_rule_passed_before":stored_rule.get("passed") is True,
        "stored_failed_checks_before":[key for key,value in stored_rule.get("checks",{}).items() if value is not True],
        "rule_passed_before":before_rule["passed"],
        "rule_passed_after":after_rule["passed"],"strict_qrel_usable":usable,
        "unresolved_reasons":dict(Counter(row["reason"] for row in unresolved)),
        "failed_checks_before":[key for key,value in before_rule.get("checks",{}).items() if value is not True],
        "failed_checks_after":[key for key,value in after_rule.get("checks",{}).items() if value is not True]}


def read_api_snapshot(run,tasks,progress=False):
    from research.annotation_snapshot import read_snapshot
    try:return read_snapshot(run,tasks,require_complete=not progress)
    except ValueError as error:
        if "完整终态" in str(error):
            raise ValueError("任务尚未齐备或尚无完整终态；--progress只查看汇总，不生成训练标签") from None
        raise


def reconcile(tasks_path,run,output,progress=False):
    tasks_path=Path(tasks_path);tasks=read_jsonl(tasks_path)
    if not tasks:raise ValueError("任务集不能为空")
    validate_tasks(tasks)
    for task in tasks:
        if task.get("kind")!="relevance" or not isinstance(task.get("material"),dict) or task.get("input_hash")!=digest(task["material"]):
            raise ValueError("任务必须为带正确材料input_hash的人岗相关性任务")
        if not isinstance(task["material"].get("job"),dict):raise ValueError("任务缺岗位原文字段")
    source,originals,hashes,read_info=read_api_snapshot(run,tasks,progress)
    rows=[];stats=[]
    for task in tasks:
        if task["task_id"] in originals:
            derived,stat=reconcile_label(task,originals[task["task_id"]]);rows.append(derived);stats.append(stat)
    summary={"total_tasks":len(tasks),"completed_tasks":len(rows),"pending_tasks":len(tasks)-len(rows),
        "labels_with_corrected_field":sum(stat["changed"] for stat in stats),
        "field_corrections":sum(stat["field_corrections"] for stat in stats),
        "stored_rule_passed_before":sum(stat["stored_rule_passed_before"] for stat in stats),
        "stored_failed_checks_before":dict(Counter(key for stat in stats for key in stat["stored_failed_checks_before"])),
        "rule_passed_before":sum(stat["rule_passed_before"] for stat in stats),
        "rule_passed_after":sum(stat["rule_passed_after"] for stat in stats),
        "strict_qrel_usable":sum(stat["strict_qrel_usable"] for stat in stats),
        "failed_checks_before":dict(Counter(key for stat in stats for key in stat["failed_checks_before"])),
        "failed_checks_after":dict(Counter(key for stat in stats for key in stat["failed_checks_after"])),
        "unresolved_reasons":dict(sum((Counter(stat["unresolved_reasons"]) for stat in stats),Counter())),
        "human_confirmed":False,"grade_changes":0,"quote_text_changes":0,"progress_only":progress,
        "cause":"只定位字段名与原文所在字段不一致的情况；不纠正语义判断或不逐字的引用。"}
    target=new_private_directory(output)
    write_json(target/"progress.json",summary)
    if not progress:write_jsonl(target/"labels.jsonl",rows)
    write_json(target/"manifest.json",{"schema":"job-agent-quote-field-reconciliation-v1","version":VERSION,
        "created_at":datetime.now(timezone.utc).isoformat(),"tasks":len(tasks),"tasks_hash":digest(tasks),
        "tasks_file_sha256":file_sha(tasks_path),"source_files_sha256":hashes,"read_info":read_info,
        "rule_validator_module_sha256":file_sha(inspect.getsourcefile(validate_label)),
        "reconciler_module_sha256":file_sha(__file__),
        "source_run_id":source.get("run_id"),"source_status":source.get("status"),"model":source.get("model"),
        "reasoning_effort":source.get("reasoning_effort"),"label_source":"llm_reviewed",
        "status":"running" if progress else "completed","completed_tasks":len(rows),"rule_passed":summary["rule_passed_after"],
        "usable_relevance":summary["strict_qrel_usable"],"progress_only":progress,"direct_training_allowed":False,
        "requires_freeze_and_training_validation":True,"human_reviewers":0,"eligible_for_production":False,
        "raw_labels_overwritten":False,"files":{path.name:file_sha(path) for path in target.iterdir()},
        "notice":"派生标签保留原模型来源和失败检查；进度模式不产出labels。默认完整模式也必须通过原有冻结与训练门禁。"})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tasks","api-run","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--progress",action="store_true",help="只输出聚合进度，绝不生成可冻结labels.jsonl")
    args=parser.parse_args()
    print(json.dumps(reconcile(args.tasks,args.api_run,args.output,args.progress),ensure_ascii=False))


if __name__=="__main__":main()
