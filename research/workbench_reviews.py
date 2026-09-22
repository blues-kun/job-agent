"""人工工作台原始记录导出、身份核实和显式仲裁桥接；不会自动生成金标。"""
import argparse
import json
from pathlib import Path
from job_agent.research_store import ResearchStore
from research.annotation_workflow import review_status, freeze_dataset
from research.review_contracts import (new_private_directory, file_sha, read_jsonl, write_json, write_jsonl)
from research.teacher_dashboard import verified_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    exported=sub.add_parser("export")
    for field in ("dataset","private-root","output"):exported.add_argument("--"+field,type=Path,required=True)
    exported.add_argument("--task-id",action="append",help="显式导出选定任务；不指定则保留全部可桥接任务")
    for name in ("review-status","freeze"):
        command=sub.add_parser(name)
        for field in ("bundle","registry","output"):command.add_argument("--"+field,type=Path,required=True)
        if name=="freeze":command.add_argument("--decisions",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="export":
        print(json.dumps(ResearchStore(args.dataset,args.private_root).export_review_bundle(args.output,args.task_id),ensure_ascii=False));return
    manifest=verified_manifest(args.bundle)
    if manifest.get("schema")!="job-agent-workbench-review-export-v2":raise ValueError("不是工作台v2导出包")
    tasks=read_jsonl(args.bundle/"tasks.jsonl");reviews=read_jsonl(args.bundle/"reviews.jsonl")
    registry=json.loads(args.registry.read_text("utf-8"))
    if args.command=="review-status":
        report=review_status(tasks,reviews,registry);target=new_private_directory(args.output)
        write_json(target/"review_status.json",report)
        write_jsonl(target/"pending_adjudication.jsonl",[{"task_id":row["task_id"],"task_hash":row["task_hash"],
            "review_hashes":{review["review_id"]:review["review_sha256"] for review in row["reviews"]},
            "decision":None,"human_confirmed":False,"status":row["status"]} for row in report["tasks"]])
        print(json.dumps({"counts":report["counts"],"human_gold_created":False},ensure_ascii=False))
    else:
        result=freeze_dataset(tasks,read_jsonl(args.bundle/"queries.jsonl"),reviews,registry,read_jsonl(args.decisions),args.output,
            {"bundle_manifest":file_sha(args.bundle/"manifest.json"),"registry":file_sha(args.registry),"decisions":file_sha(args.decisions)})
        print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
