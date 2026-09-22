"""保留原岗位内容ID与家族划分，生成带解析器版本和嵌套要求图的新快照。"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path

from job_agent.domain import Job, PARSER_VERSION
from job_agent.retrieval_contract import stable_hash, contract_info
from research.common import read_jsonl, write_json, private_path, sha256


def refresh(source, output):
    source=Path(source);output=private_path(output)
    if output.exists():raise FileExistsError("输出版本已存在")
    manifest=json.loads((source/"manifest.json").read_text())
    for name,details in manifest["files"].items():
        if sha256(source/name)!=details["sha256"]:raise ValueError("输入版本摘要不一致")
    output.mkdir(parents=True,mode=0o700)
    count=0;edge_count=0;skills=0;tasks=0;groups=0;coverage=0
    with (output/"jobs.jsonl").open("w") as job_file,(output/"edges.jsonl").open("w") as edge_file:
        for original in read_jsonl(source/"jobs.jsonl"):
            job=Job(id=original["job_id"],version=original["job_version_id"],row=0,family=original["category_parent"],
                    **{key:original[key] for key in ["title","company","category","salary_raw","requirements","description","address"]})
            row={**original,"skills":job.skills,"tasks":job.tasks,"groups":job.groups,"requirement_ast":job.requirement_ast,
                 "parser_version":PARSER_VERSION,"experience_requirements":job.experience_requirements,"education_requirements":job.education_requirements,
                 "experience_min":job.experience_min,"education_min":job.education_min}
            job_file.write(json.dumps(row,ensure_ascii=False)+"\n")
            count+=1;skills+=len(job.skills);tasks+=len(job.tasks);groups+=len(job.groups);coverage+=bool(job.skills or job.tasks)
            def emit(parent,node,path):
                nonlocal edge_count
                if "type" in node:
                    node_id=f"{node['type']}:{node['key']}";node_type=node['type'];children=[]
                else:
                    node_id="requirement_group:"+stable_hash([job.id,path,node])[:24];node_type="requirement_group";children=node.get("children",[])
                evidence=node.get("scope_evidence") or node.get("evidence")
                if evidence and getattr(job,evidence["field"])[evidence["start"]:evidence["end"]]!=evidence["quote"]:
                    raise ValueError("要求图引用与原文不符")
                edge={"source_id":parent,"target_id":node_id,"target_type":node_type,"job_id":job.id,"split":original["split"],
                      "relation":"has_requirement" if parent==job.id else "has_option","operator":node.get("op","leaf"),
                      "modality":node.get("modality","unknown"),"status":"machine_parsed","fact_citable":bool(evidence),
                      "evidence":evidence,"parser_version":PARSER_VERSION}
                edge["edge_id"]=stable_hash(edge)[:24]
                edge_file.write(json.dumps(edge,ensure_ascii=False)+"\n");edge_count+=1
                for i,child in enumerate(children):emit(node_id,child,path+[i])
            for i,group in enumerate(job.groups):emit(job.id,group["ast"],[i])
            if count%2000==0:print(json.dumps({"已更新岗位":count},ensure_ascii=False),flush=True)
    # 原文片段审核任务仍有效；旧机器解释不升级成人工或模型标签。
    (output/"annotation_tasks.jsonl").write_bytes((source/"annotation_tasks.jsonl").read_bytes())
    files={name:{"sha256":sha256(output/name),"bytes":(output/name).stat().st_size} for name in ["jobs.jsonl","edges.jsonl","annotation_tasks.jsonl"]}
    metadata={**manifest,"created_utc":datetime.now(timezone.utc).isoformat(),"parent_jobs_sha256":sha256(source/"jobs.jsonl"),
              "config":{**manifest["config"],"parser_version":PARSER_VERSION,"retrieval_contract":contract_info()},
              "files":files,"graph":{"edges":edge_count,"meaning":"保留嵌套要求组的机器解析关系；没有新增推断事实"},
              "semantic_summary":{"jobs":count,"skill_mentions":skills,"task_mentions":tasks,"groups":groups,"jobs_with_skills_or_tasks":coverage}}
    metadata["config_hash"]=stable_hash(metadata["config"])
    write_json(output/"manifest.json",metadata)
    for path in output.iterdir():path.chmod(0o600)
    return metadata["semantic_summary"]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(refresh(args.source,args.output),ensure_ascii=False))
