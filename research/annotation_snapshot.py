"""读取一致的API标注快照；运行中使用已完成batch，终态严格核对完整标签与SHA。"""
import hashlib
import json
from pathlib import Path
from research.common import private_path
from research.review_contracts import digest


def validate_identity(task,row):
    if not isinstance(row,dict) or row.get("input_hash")!=task.get("input_hash") or row.get("label_source")!="llm_reviewed":
        raise ValueError("API标签输入或来源变更")
    for key in ("task_id","kind","query_id","job_id","profile_family_id","job_family_id","split","task_hash"):
        if key in task and row.get(key)!=task[key]:raise ValueError("API标签身份或任务版本不同")


def read_snapshot(run,tasks,require_complete=False):
    run=private_path(run)
    content=(run/"manifest.json").read_bytes();manifest=json.loads(content)
    if not tasks or manifest.get("tasks")!=len(tasks) or manifest.get("tasks_hash")!=digest(tasks):
        raise ValueError("API运行任务集与当前任务版本不一致")
    if manifest.get("label_source")!="llm_reviewed":raise ValueError("API运行不是模型标签来源")
    if require_complete and (manifest.get("status")!="completed" or manifest.get("progress_only") is True):
        raise ValueError("API尚无完整终态；进度文件不能作为训练入口")
    expected={task["task_id"]:task for task in tasks}
    if len(expected)!=len(tasks):raise ValueError("当前任务ID重复")
    active=manifest.get("status")!="completed"
    hashes={"manifest.json":hashlib.sha256(content).hexdigest()}
    batches={};completed=0
    for path in sorted(run.glob("batch-*.json")):
        content=path.read_bytes();batch=json.loads(content)
        declared=manifest.get("files",{}).get(path.name)
        declared=declared.get("sha256") if isinstance(declared,dict) else declared
        if not active and declared and hashlib.sha256(content).hexdigest()!=declared:raise ValueError("终态batch文件SHA失配")
        if batch.get("status")!="completed":continue
        hashes[path.name]=hashlib.sha256(content).hexdigest();completed+=1
        for row in batch.get("rows",[]):
            key=row.get("task_id")
            if key not in expected or key in batches:raise ValueError("API batch含未知或重复任务")
            validate_identity(expected[key],row);batches[key]=row
    final=None;state="absent"
    if (run/"labels.jsonl").exists():
        if active:
            state="stale_aggregate_ignored_while_active"
        else:
            content=(run/"labels.jsonl").read_bytes();hashes["labels.jsonl"]=hashlib.sha256(content).hexdigest()
            declared=manifest.get("files",{}).get("labels.jsonl")
            expected_sha=declared.get("sha256") if isinstance(declared,dict) else declared
            if expected_sha and expected_sha!=hashes["labels.jsonl"]:raise ValueError("终态labels文件SHA失配")
            final={}
            for line in content.decode("utf-8").splitlines():
                if not line.strip():continue
                row=json.loads(line);key=row.get("task_id")
                if key not in expected or key in final:raise ValueError("终态标签含未知或重复任务")
                validate_identity(expected[key],row);final[key]=row
            if batches and (set(final)!=set(batches) or any(digest(final[key])!=digest(row) for key,row in batches.items())):
                raise ValueError("终态labels与已完成batch记录不一致")
            state="verified_terminal_aggregate"
    values=batches if active or final is None else final
    if require_complete and (final is None or set(values)!=set(expected)):
        raise ValueError("完整终态标签未齐备，不能启动训练或冻结")
    if not active and set(values)!=set(expected):raise ValueError("API宣称完成但任务结果未齐备")
    return manifest,values,hashes,{"completed_batches":completed,"final_labels_state":state,
        "all_tasks_present":set(values)==set(expected),"terminal_manifest":not active,
        "progress_only":manifest.get("progress_only") is True}
