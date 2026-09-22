"""发布与读取带SHA的教师研究聚合快照；不向工作台提供原始请求、密钥或简历。"""
from __future__ import annotations
import argparse
import fcntl
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from research.common import private_path, write_json
from research.review_contracts import digest, file_sha, read_jsonl, validate_qrel

ARTIFACTS={"annotation_progress.json","ranking_metrics.json","embedding_metrics.json","graph_metrics.json"}
STATS={"models","summary","results","train","dev","test","comparisons","ndcg_delta","queries","pairs","seed","seeds",
       "mean_delta","ci95","clusters","iterations","k","ndcg","ndcg_at_k","ndcg_at_10","pool_recall_at_k","mrr_at_k",
       "recall_at_10","recall_at_k","mrr","loss","epoch","epochs","steps","optimizer_steps","metric_query_counts",
       "fully_judged_queries","all_queries_judged","input_context_fully_recorded","excluded_zero_idcg_queries","ndcg_queries",
       "best_iteration","training_seconds","fixture_only","teacher_relative","eligible_for_production","reload_verification",
       "max_absolute_difference","ranking_identical","absolute_tolerance","relative_tolerance","mean","std","count","n",
       "explicit_logic","text_pool","pool_mlp","graphsage","random_graph","real_graph","baseline","candidate","reference",
       "supported_lower","supported_upper","coverage","metrics","variants","runs","evaluation","dev_ndcg","test_ndcg",
       "logic_baseline","best_epoch","executed_epochs","seconds","parameter_count","gradient_norm","dev_ndcg_at_10",
       "history","randomization","changed_graphs","test_evaluated","model_weak_supervision","state_unchanged_by_evaluation"}


def verified_manifest(directory,name="manifest.json"):
    directory=private_path(directory)
    manifest=json.loads((directory/name).read_text("utf-8"))
    files=manifest.get("files")
    if not isinstance(files,dict) or not files:raise ValueError("聚合来源缺文件SHA清单")
    for filename,details in files.items():
        if Path(filename).name!=filename:raise ValueError("清单仅允许同目录文件")
        path=(directory/filename).resolve()
        if path.parent!=directory.resolve():raise ValueError("清单不能通过链接读取目录外文件")
        expected=details.get("sha256") if isinstance(details,dict) else details
        if not isinstance(expected,str) or file_sha(path)!=expected:raise ValueError("研究产物SHA校验失败")
    return manifest


def numeric_metrics(value,depth=0):
    """只保留有限数值/状态及指标结构；任意字符串与原始样本行不会发到前端。"""
    if depth>8:return None
    if value is None or type(value) is bool:return value
    if type(value) in {int,float}:return value if math.isfinite(value) else None
    if isinstance(value,list):return [numeric_metrics(item,depth+1) for item in value[:50] if not isinstance(item,str)]
    if isinstance(value,dict):
        output={}
        for index,(key,item) in enumerate(value.items()):
            if index>=100:break
            if re.search(r"secret|token|password|api.?key|authorization|resume|原文|简历|prompt|evidence|calls|source_files|path|^rows$|ranked_ids|candidate_ids",str(key),re.I):continue
            if isinstance(item,str):
                if key=="mode" and item in {"pool","graph","random_graph","explicit_logic"}:output[key]=item
                elif key=="status" and item in {"validated","running","completed","incomplete","failed","rate_limited"}:output[key]=item
                continue
            public_key=key if (key in STATS or re.fullmatch(r"系列_[0-9a-f]{8}",str(key))) or re.fullmatch(r"(?:bm25|dense|rule|ranker|bge|qwen|lora|frozen|graph|random|text|pool|lambda|hgt)[a-z0-9_.+@-]{0,40}",str(key),re.I) else "系列_"+digest(str(key))[:8]
            output[public_key]=numeric_metrics(item,depth+1)
        return output
    return None


def _annotation_snapshot(folder,study):
    from research.api_annotation import validate_label
    folder=private_path(folder)
    raw_manifest=(folder/"manifest.json").read_bytes();manifest=json.loads(raw_manifest)
    matches=[]
    for name in ("tasks.jsonl","extraction_tasks.jsonl"):
        path=study/name
        if not path.exists():continue
        tasks=read_jsonl(path);count=manifest.get("tasks")
        if type(count) is int and 0<count<=len(tasks) and digest(tasks[:count])==manifest.get("tasks_hash"):
            matches.append(tasks[:count])
    if len(matches)!=1:raise ValueError("API标注任务hash不属于当前教师材料版本")
    task_rows=matches[0]
    tasks={row["task_id"]:row for row in task_rows}
    if manifest.get("label_source")!="llm_reviewed":raise ValueError("教师看板不能把其他来源改名为模型标签")
    if manifest.get("progress_only") is True:raise ValueError("进度汇总不是标注运行，不能当作labels来源")
    from research.annotation_snapshot import read_snapshot
    manifest,by_task,source_hashes,snapshot_info=read_snapshot(folder,task_rows)
    labels=list(by_task.values())
    seen=set();valid=usable=invalid=0;grades=Counter()
    for label in labels:
        task=tasks.get(label.get("task_id"))
        if task is None or label["task_id"] in seen:raise ValueError("API标签含未知或重复任务")
        seen.add(label["task_id"])
        if label.get("input_hash")!=task.get("input_hash") or label.get("label_source")!="llm_reviewed":
            raise ValueError("API标签输入或来源变更")
        if task.get("task_hash") and label.get("task_hash")!=task["task_hash"]:
            raise ValueError("API标签与任务内容版本不同")
        try:
            passed=validate_label(task,label)["passed"] and label.get("rule_validation",{}).get("passed") is True
            valid+=int(passed)
            if task["kind"]=="relevance" and label.get("grade") is not None and passed:
                validate_qrel(label,allow_model_labels=True);usable+=1;grades[str(label["grade"])]+=1
        except (ValueError,TypeError,KeyError):invalid+=1
    status=manifest.get("status")
    if status not in {"running","completed","incomplete","failed","rate_limited"}:raise ValueError("未知标注执行状态")
    if status=="completed" and len(labels)!=len(tasks):raise ValueError("API清单声称完成但标签未齐")
    return {"task_kind":next(iter(tasks.values()))["kind"],"status":status,"tasks":len(tasks),
            "completed_tasks":len(labels),"pending_tasks":len(tasks)-len(labels),"rule_passed":valid,
            "usable_relevance":usable,"invalid_labels":invalid,"grade_counts":dict(grades),
            "model":"gpt-5.6-terra" if manifest.get("model")=="gpt-5.6-terra" else "其他配置模型",
            "reasoning_effort":"max" if manifest.get("reasoning_effort")=="max" else "未核对",
            "label_source":"llm_reviewed","human_reviewers":0,"eligible_for_production":False,
            "source_files_sha256":source_hashes,"snapshot_info":snapshot_info,
            "observed_at":datetime.now(timezone.utc).isoformat(),
            "retry_after_seconds":manifest.get("retry_after_seconds"),"resume_not_before":manifest.get("resume_not_before"),
            "automatic_restart":False}


def _publish_teacher_dashboard_locked(study,annotation_runs=(),ranking_metrics=None,embedding_metrics=None,graph_metrics=None):
    study=private_path(study);base=verified_manifest(study)
    if base.get("schema")!="teacher-study-v2":raise ValueError("不是支持的教师研究版本")
    payloads={}
    if annotation_runs:
        payloads["annotation_progress.json"]={"runs":[_annotation_snapshot(folder,study) for folder in annotation_runs],
            "notice":"模型双通道与规则校验，不是双人人工一致性。"}
    for kind,path in [("ranking",ranking_metrics),("embedding",embedding_metrics),("graph",graph_metrics)]:
        if path is None:continue
        path=private_path(path);content=path.read_bytes();value=json.loads(content)
        payloads[kind+"_metrics.json"]={"metrics":numeric_metrics(value),"source_sha256":hashlib.sha256(content).hexdigest(),
                                      "teacher_relative":True,"eligible_for_production":False}
    if not payloads:raise ValueError("请提供至少一个标注运行或研究指标文件")
    # 在共享读/独占写锁内合并同一研究版本的已有指标；进度更新不能擦掉实验索引。
    files={};times={};study_hash=file_sha(study/"manifest.json")
    if (study/"dashboard_manifest.json").exists():
        previous=verified_manifest(study,"dashboard_manifest.json")
        if previous.get("schema")!="job-agent-teacher-dashboard-v2" or not set(previous["files"]).issubset(ARTIFACTS):raise ValueError("已有看板合同非法")
        if previous.get("study_manifest_sha256")==study_hash:
            files.update(previous["files"])
            times={name:previous.get("artifact_published_at",{}).get(name,previous.get("published_at")) for name in files}
    now=datetime.now(timezone.utc).isoformat()
    for name,value in payloads.items():
        write_json(study/name,value);files[name]=file_sha(study/name);times[name]=now
    manifest={"schema":"job-agent-teacher-dashboard-v2","study_manifest_sha256":study_hash,
              "published_at":now,"artifact_published_at":times,"files":files,
              "label_source":"llm_reviewed","human_gold_created":False,"eligible_for_production":False}
    write_json(study/"dashboard_manifest.json",manifest)
    return {"published":True,"artifacts":sorted(files),"updated_artifacts":sorted(payloads),"human_gold_created":False}


def _load_teacher_dashboard_locked(directory,expected_source_sha256=None):
    if directory is None:return {"available":False,"status":"not_configured"}
    try:
        directory=private_path(directory);base=verified_manifest(directory)
        if base.get("schema")!="teacher-study-v2" or expected_source_sha256 and base.get("source_sha256")!=expected_source_sha256:
            raise ValueError("教师版本或岗位来源不同")
        def count(value):
            if type(value) is not int or value<0:raise ValueError("汇总计数非法")
            return value
        allowed_sources={"model_authored_fiction","model_synthetic_from_job","authorized_real","independent_fiction"}
        result={"available":True,"status":"materials_verified","profiles":count(base["profiles"]),"pairs":count(base["pairs"]),
                "extraction_tasks":count(base["extraction_tasks"]),
                "splits":{key:count(value) for key,value in base.get("splits",{}).items() if key in {"train","dev","test"}},
                "source_kinds":{key:count(value) for key,value in base.get("source_kinds",{}).items() if key in allowed_sources},
                "label_source":"llm_reviewed","human_reviewers":0,"eligible_for_production":False,
                "scope":"合成画像与固定共同池的教师相对研究；没有独立真人效果验证。",
                "annotation_runs":[],"metrics":{},"files_verified":True}
        if not (directory/"dashboard_manifest.json").exists():
            result["notice"]="材料已校验，尚未发布带SHA的标注进度/指标快照。"
            return result
        dashboard=verified_manifest(directory,"dashboard_manifest.json")
        if dashboard.get("schema")!="job-agent-teacher-dashboard-v2" or dashboard.get("study_manifest_sha256")!=file_sha(directory/"manifest.json") or not set(dashboard["files"]).issubset(ARTIFACTS):
            raise ValueError("看板快照与教师研究材料不一致")
        result.update(status="dashboard_verified",published_at=dashboard.get("published_at"),artifact_published_at=dashboard.get("artifact_published_at",{}))
        for name in dashboard["files"]:
            value=json.loads((directory/name).read_text("utf-8"))
            if name=="annotation_progress.json":
                for row in value.get("runs",[]):
                    result["annotation_runs"].append({key:row[key] for key in ["task_kind","status","tasks","completed_tasks","pending_tasks","rule_passed","usable_relevance","invalid_labels","grade_counts","model","reasoning_effort","label_source","human_reviewers","eligible_for_production"]})
                    if row.get("status")=="rate_limited":
                        run=result["annotation_runs"][-1]
                        run.update(retry_after_seconds=row.get("retry_after_seconds"),resume_not_before=row.get("resume_not_before"),automatic_restart=False)
                        if row.get("resume_not_before"):
                            remaining=(datetime.fromisoformat(row["resume_not_before"])-datetime.now(timezone.utc)).total_seconds()
                            run["retry_wait_remaining_seconds"]=max(0,round(remaining))
            else:result["metrics"][name.removesuffix("_metrics.json")]=numeric_metrics(value.get("metrics",{}))
        annotation_time=dashboard.get("artifact_published_at",{}).get("annotation_progress.json",dashboard.get("published_at"))
        if result["annotation_runs"] and annotation_time:
            observed=datetime.fromisoformat(annotation_time.replace("Z","+00:00"))
            age=max(0,(datetime.now(timezone.utc)-observed).total_seconds())
            active=any(row["status"] in {"running","incomplete"} for row in result["annotation_runs"])
            result["annotation_progress_age_seconds"]=round(age,1)
            result["annotation_progress_stale"]=active and age>120
            if result["annotation_progress_stale"]:result["notice"]="运行中标注进度超过120秒未更新；保留上次已核验快照，不作为当前完成状态。"
        return result
    except (ValueError,KeyError,TypeError,OSError):
        return {"available":False,"status":"validation_failed","files_verified":False,
                "notice":"教师产物未通过版本/文件SHA核验，暂不展示；请重新发布完整快照。"}


def publish_teacher_dashboard(study,annotation_runs=(),ranking_metrics=None,embedding_metrics=None,graph_metrics=None):
    study=private_path(study)
    with (study/"dashboard.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _publish_teacher_dashboard_locked(study,annotation_runs,ranking_metrics,embedding_metrics,graph_metrics)


def load_teacher_dashboard(directory,expected_source_sha256=None):
    if directory is None:return {"available":False,"status":"not_configured"}
    try:
        directory=private_path(directory)
        with (directory/"dashboard.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            return _load_teacher_dashboard_locked(directory,expected_source_sha256)
    except (ValueError,KeyError,TypeError,OSError):
        return {"available":False,"status":"validation_failed","files_verified":False,
                "notice":"教师看板快照暂不可读，未发布未核验的内容。"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--annotation-runs",type=Path,nargs="*",default=[])
    for name in ("ranking-metrics","embedding-metrics","graph-metrics"):parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    print(json.dumps(publish_teacher_dashboard(args.study,args.annotation_runs,args.ranking_metrics,args.embedding_metrics,args.graph_metrics),ensure_ascii=False))


if __name__=="__main__":main()
