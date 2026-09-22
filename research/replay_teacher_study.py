"""对固定教师候选池重新回放当前线上特征，记录上游标注版本，不覆写旧回放。"""
import argparse
import json
from pathlib import Path
import numpy as np

from job_agent.corpus import Corpus,tokens
from job_agent.domain import parse_profile,supported_matches
from job_agent.workflow import Workflow
from job_agent.ranking import feature_vector,FEATURE_NAMES,FEATURE_VERSION
from job_agent.retrieval_contract import contract_info,stable_hash,render_query,render_document
from research.common import read_jsonl,write_json,write_jsonl,private_path,sha256


def replay(study,dataset,source,dense_dir,output):
    output=private_path(output)
    if output.exists():raise FileExistsError("输出已存在，请使用新回放版本")
    records=read_jsonl(dataset/"jobs.jsonl");profiles=read_jsonl(study/"profiles.jsonl")
    manifest=json.loads((dataset/"manifest.json").read_text())
    if manifest["config"].get("retrieval_contract",{}).get("hash")!=contract_info()["hash"]:
        raise ValueError("岗位语义快照与当前检索合同不一致，请先refresh_semantics")
    if sha256(dataset/"jobs.jsonl")!=manifest["files"]["jobs.jsonl"]["sha256"]:raise ValueError("岗位版本摘要不一致")
    pools=json.loads((study/"pools.json").read_text());tasks=read_jsonl(study/"tasks.jsonl")
    corpus=Corpus(source,dense_dir=dense_dir,records=records);workflow=Workflow(corpus)
    if corpus.dense is None:raise ValueError("真实编码器未就绪")
    # 修复解析器可以改变机器标签，但本次JD原文字段/版本不得悄悄替换。
    old_records={r["job_id"]:r for r in read_jsonl(Path(json.loads((study/"manifest.json").read_text())["dataset"])/"jobs.jsonl")}
    for task in tasks:
        current=corpus.research_records[task["job_id"]];old=old_records[task["job_id"]]
        if any(current[k]!=old[k] for k in ["job_version_id","snapshot","job_family_id","split","title","requirements","description","salary_raw","address"]):
            raise ValueError("岗位正文或分区已变化，旧教师标注不能沿用")
    rows=[];baselines={name:[] for name in ["bm25","bge","hybrid","rule"]}
    for query in profiles:
        profile=parse_profile(query["text"],query["preferences"])
        checks,allowed,rrf,sources,contexts=workflow.retrieval_state(profile,split=query["split"])
        if "dense" not in sources:raise ValueError("中途编码服务失败，不能生成不同引擎的回放")
        intent_words=[w for w in tokens(profile["intent"]) if w not in {"开发","工程师","方向","岗位","人员","技术"}]
        scores={name:[] for name in baselines}
        for job_id in pools[query["query_id"]]:
            i=corpus.lookup[job_id];job=corpus.jobs[i];vector=feature_vector(profile,job,contexts[i]);context=contexts[i]
            intent=sum(w in (job.title+" "+job.category).lower() for w in intent_words)/max(len(intent_words),1)
            confidence=sum(c["status"]=="pass" for c in checks[i])/max(len(checks[i]),1)
            district=1 if profile["district"] and profile["district"]==job.district else .5
            rule=(.25*float(rrf[i])/(float(rrf.max()) or 1)+.25*supported_matches(profile,job)[2]+.2*context["task_alignment"]+.2*intent+.05*confidence+.05*district)*(.55 if intent_words and intent==0 else 1)
            rows.append({"query_id":query["query_id"],"profile_family_id":query["profile_family_id"],"job_id":job_id,"job_family_id":job.job_family_id,
                "split":query["split"],"snapshot":corpus.snapshot,"feature_version":FEATURE_VERSION,
                "features":{name:None if np.isnan(v) else float(v) for name,v in zip(FEATURE_NAMES,vector)},
                "retrieval_contract_hash":contract_info()["hash"],"encoder_contract":corpus.dense.encoder_contract,
                "query_payload_sha256":stable_hash({"text":query["text"],"preferences":query["preferences"]}),"eligible":bool(allowed[i])})
            if allowed[i]:
                for name,value in [("bm25",context["bm25"]),("bge",context["dense"]),("hybrid",float(rrf[i])),("rule",rule)]:
                    scores[name].append({"job_id":job_id,"score":value})
        metadata={"query_id":query["query_id"],"query_payload_sha256":stable_hash({"text":query["text"],"preferences":query["preferences"]}),
            "candidate_universe_count":len(pools[query["query_id"]]),"candidate_universe_sha256":stable_hash(sorted(pools[query["query_id"]])),
            "query_sha256":stable_hash(render_query(query)),"document_view_sha256":stable_hash([(jid,render_document(corpus.by_id[jid])) for jid in sorted(pools[query["query_id"]])]),
            "constraint_policy_sha256":stable_hash({"hard_filters":"explicit_fail_removed;unknown_retained","parser":contract_info()["parser_version"],"code":contract_info()["hash"]})}
        for name in baselines:baselines[name].append({**metadata,"ranking":sorted(scores[name],key=lambda r:(-r["score"],r["job_id"]))})
        print(json.dumps({"已回放":query["query_id"],"候选数":len(pools[query["query_id"]])},ensure_ascii=False),flush=True)
    output.mkdir(parents=True,mode=0o700)
    write_jsonl(output/"all_replay.jsonl",rows)
    write_jsonl(output/"train_replay.jsonl",[row for row in rows if row["split"] in {"train","dev"}])
    for name,ranking in baselines.items():write_jsonl(output/f"{name}.jsonl",[r for r in ranking if next(q for q in profiles if q["query_id"]==r["query_id"])["split"]=="test"])
    result={"schema":"job-agent-ranker-replay-v2","replay_sha256":sha256(output/"train_replay.jsonl"),"jobs_sha256":sha256(dataset/"jobs.jsonl"),
        "profiles_sha256":sha256(study/"train_profiles.jsonl"),"feature_names":list(FEATURE_NAMES),"feature_version":FEATURE_VERSION,
        "retrieval_contract_hash":contract_info()["hash"],"encoder_contract":corpus.dense.encoder_contract,"upstream_study_sha256":sha256(study/"manifest.json"),
        "upstream_annotation_tasks_sha256":sha256(study/"tasks.jsonl"),"reparsed_dataset":str(dataset),"raw_fields_unchanged_verified":True,
        "scope":"共同池重排比较；池来自旧版本预注册检索并集，不是当前系统的全库召回测试。"}
    write_json(output/"replay_manifest.json",result)
    return {"pairs":len(rows),"queries":len(profiles),"output":str(output)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["study","dataset","source","dense-dir","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(replay(args.study,args.dataset,args.source,args.dense_dir,args.output),ensure_ascii=False))
