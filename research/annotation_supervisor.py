"""有限轮次的断点标注与实验执行；成功门禁之前绝不启动训练。"""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import uuid

from research.api_annotation import annotate,row_usable,validate_label,RateLimitGate,RateLimitedError
from research.annotation_snapshot import read_snapshot
from research.common import read_jsonl
from research.review_contracts import digest,file_sha
from research.reconcile_annotations import reconcile
from research.common import private_path,write_json
from research.teacher_dashboard import publish_teacher_dashboard


def prepare_derived_annotations(study):
    """只有完整终态才生成不可变派生目录；目录标识绑定原文件和当前纠正规则。"""
    tasks=read_jsonl(study/"tasks.jsonl")
    _,_,hashes,_=read_snapshot(study/"annotation",tasks,require_complete=True)
    import inspect
    key=digest({"source_files_sha256":hashes,"tasks_sha256":file_sha(study/"tasks.jsonl"),
        "reconciler_sha256":file_sha(inspect.getsourcefile(reconcile)),
        "validator_sha256":file_sha(inspect.getsourcefile(validate_label))})
    output=study/"reconciled-annotation"/key
    if output.exists():
        manifest,_,_,_=read_snapshot(output,tasks,require_complete=True)
        if manifest.get("source_files_sha256")!=hashes:raise ValueError("已有派生版本来源不同")
        for name,expected in manifest.get("files",{}).items():
            if Path(name).name!=name or file_sha(output/name)!=expected:raise ValueError("派生版本文件SHA失配")
    else:reconcile(study/"tasks.jsonl",study/"annotation",output)
    return output


def completion_gate(study,annotation_dir=None):
    """独立读取终态和全部任务，不相信annotate返回计数或旧的progress文件。"""
    verified=[]
    for task_name,run_name in [("tasks.jsonl","annotation"),("extraction_tasks.jsonl","extraction-annotation")]:
        tasks=read_jsonl(study/task_name)
        if any(task.get("input_hash")!=digest(task.get("material")) for task in tasks):raise ValueError("任务材料hash失配")
        manifest,rows,hashes,info=read_snapshot(annotation_dir if run_name=="annotation" and annotation_dir is not None else study/run_name,tasks,require_complete=True)
        if manifest.get("model")!="gpt-5.6-terra" or manifest.get("reasoning_effort")!="max":raise ValueError("终态并非指定模型与推理档")
        if any(not row_usable(task,rows[task["task_id"]]) for task in tasks):raise ValueError("仍有引用、审核血缘或未知等级未通过；不启动训练")
        verified.append({"tasks":len(tasks),"tasks_sha256":digest(tasks),"source_files_sha256":hashes})
    return {"ready":True,"runs":verified,"human_gold_count":0}


async def supervise(args):
    study=private_path(args.study);status_path=study/"supervisor_status.json"
    if not 1<=args.max_rounds<=3:raise ValueError("总轮次须为1至3")
    lock=(study/"supervisor.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=asyncio.Event()
    rate_gate=RateLimitGate()
    annotation_dir=study/"annotation"
    async def publish():
        while not stopped.is_set():
            try:await asyncio.to_thread(publish_teacher_dashboard,study,[annotation_dir,study/"extraction-annotation"])
            except (ValueError,OSError,KeyError):pass # 启动中尚未有manifest，下轮再核验。
            try:await asyncio.wait_for(stopped.wait(),timeout=30)
            except TimeoutError:pass
    publishing=asyncio.create_task(publish())
    try:
        for round_index in range(1,args.max_rounds+1):
            annotation_dir=study/"annotation"
            write_json(status_path,{"stage":"annotating","round":round_index,"max_rounds":args.max_rounds,
                "model":"gpt-5.6-terra","reasoning_effort":"max","human_gold_count":0})
            results=await asyncio.gather(
                annotate(study/"tasks.jsonl",args.config,study/"annotation",2,getattr(args,"annotation_concurrency",1),rate_gate=rate_gate),
                annotate(study/"extraction_tasks.jsonl",args.config,study/"extraction-annotation",2,getattr(args,"extraction_concurrency",1),rate_gate=rate_gate),return_exceptions=True)
            results=[result.status() if isinstance(result,RateLimitedError) else {"status":"failed","error_type":type(result).__name__} if isinstance(result,BaseException) else result for result in results]
            limited=[result for result in results if result.get("status")=="rate_limited"]
            if limited:
                slowest=max(limited,key=lambda result:result.get("retry_after_seconds",0))
                write_json(status_path,{"stage":"rate_limited",**slowest,"round":round_index,"results":results,
                    "human_gold_count":0,"note":"已保存所有成功checkpoint；429不进入下一轮，不自动重启。冷却后从一次成功探针与低并发恢复。"})
                return
            try:
                annotation_dir=prepare_derived_annotations(study)
                gate=completion_gate(study,annotation_dir)
            except (ValueError,KeyError,TypeError,OSError) as error:
                gate={"ready":False,"error_type":type(error).__name__}
            if gate["ready"]:break
        else:
            write_json(status_path,{"stage":"needs_review","rounds":args.max_rounds,"results":results,
                "completion_gate":gate,"note":"已耗尽有限重试；保留失败/未知与原始调用，不补标签、不启动训练。"})
            return
        from research.evaluate_requirements import run as evaluate_requirements
        requirement_output=private_path(args.output)/"requirements-evaluations"/uuid.uuid4().hex
        requirement_report=await asyncio.to_thread(evaluate_requirements,study,args.dataset,
            study/"extraction-annotation",requirement_output,False)
        requirement_summary={"status":requirement_report["status"],"tasks_evaluated":requirement_report["tasks_evaluated"],
            "report_path":str(requirement_output/"report.json")}
        write_json(status_path,{"stage":"training","completion_gate":gate,"requirements_evaluation":requirement_summary,
            "human_gold_count":0,"eligible_for_production":False})
        command=[sys.executable,"-B","-m","research.run_teacher_experiments","--study",str(study),"--dataset",str(args.dataset),
            "--annotation-dir",str(annotation_dir),"--replay",str(args.replay),"--output",str(args.output),"--training-python",str(args.training_python),"--gpu",args.gpu]
        process=await asyncio.create_subprocess_exec(*command)
        code=await process.wait()
        write_json(status_path,{"stage":"completed" if code==0 else "experiment_failed","exit_code":code,
            "requirements_evaluation":requirement_summary,"human_gold_count":0,"eligible_for_production":False})
    except Exception as error:
        write_json(status_path,{"stage":"failed","error_type":type(error).__name__,
            "human_gold_count":0,"eligible_for_production":False,"note":"执行步骤失败，保留私有日志；未发布完成状态。"})
        raise
    finally:
        stopped.set();await publishing
        try:await asyncio.to_thread(publish_teacher_dashboard,study,[annotation_dir,study/"extraction-annotation"])
        except (ValueError,OSError,KeyError):pass
        lock.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["study","config","dataset","replay","output","training-python"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--annotation-concurrency",type=int,choices=range(1,9),default=1)
    parser.add_argument("--extraction-concurrency",type=int,choices=range(1,9),default=1)
    parser.add_argument("--gpu",default="2");parser.add_argument("--max-rounds",type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.max_rounds<=3:raise ValueError("总轮次须为1至3；不无限重试接口")
    asyncio.run(supervise(args))
