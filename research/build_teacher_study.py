"""构建可复跑的模型教师试验：真实岗位、明确合成画像、固定共同池与线上特征回放。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
from job_agent.corpus import Corpus, tokens
from job_agent.domain import parse_profile, CONTACT, supported_matches
from job_agent.workflow import Workflow
from job_agent.ranking import FEATURE_NAMES, FEATURE_VERSION, feature_vector
from job_agent.retrieval_contract import contract_info, stable_hash
from research.annotation_workflow import seal_task
from research.common import read_jsonl, sha256, write_json, write_jsonl, private_path

# 全部为本项目作者模型独立撰写的虚构材料；没有真人身份或人工来源背书。
FICTION = [
    ("dev", "Java后端", "本科。累计2年工作经验。\n\n项目：订单服务\n使用Java和Spring Boot实现订单查询接口，使用MySQL保存订单记录，为重复请求增加幂等检查；使用Redis缓存商品库存，排查过接口超时。"),
    ("dev", "嵌入式软件", "本科。累计3年工作经验。\n\n项目：工业采集设备\n使用C和C++开发STM32固件，调试UART通信，维护FreeRTOS任务；使用示波器定位串口时序问题，编写设备联调记录。尚未参与Linux驱动开发。"),
    ("dev", "数据分析", "本科。累计1年工作经验。\n\n项目：经营周报\n使用SQL查询销售订单并核对退款口径；使用Python和Pandas清洗门店数据，使用Excel制作月度报表，向运营同事解释环比变化。只了解机器学习基础，没有模型上线经历。"),
    ("dev", "软件测试", "本科。累计2年工作经验。\n\n项目：移动端测试\n负责测试用例设计、缺陷跟踪和发布回归，使用Python维护接口自动化测试，使用Postman调试接口，使用SQL核查测试数据。未做过性能容量规划。"),
    ("test", "Java后端", "本科。累计3年工作经验。\n\n项目：预约平台\n使用Java和Spring Boot负责预约取消、退款状态查询接口；使用MySQL事务保证状态更新，使用Redis做热点时段缓存，编写接口日志并定位过重复回调问题。熟悉Git协作。"),
    ("test", "数据分析", "硕士。累计2年工作经验。\n\n项目：广告投放复盘\n使用SQL和Python分析分渠道转化漏斗，使用Pandas处理缺失值，制作Tableau看板；与运营确认指标口径并解释样本量限制。未开发高并发服务。"),
    ("test", "前端开发", "本科，应届毕业生。\n\n项目：志愿活动报名页\n使用Vue、JavaScript、HTML和CSS实现报名表单、权限按钮和进度展示，使用Git管理代码，与后端对接接口，修复移动端布局和表单校验问题。"),
    ("test", "嵌入式软件", "大专。累计4年工作经验。\n\n项目：电机控制器\n使用C语言维护控制固件，开发STM32串口协议处理，调试CAN总线消息，记录异常复现步骤；负责生产测试工装联调。没有FPGA开发经验。"),
    ("test", "软件测试", "本科。累计5年工作经验。\n\n项目：电商服务质量\n负责接口测试策略和版本准入检查，使用Python和Pytest维护回归测试；分析Linux服务日志，使用JMeter执行压力测试并汇总响应时间，带两名同事设计边界用例。"),
    ("test", "项目经理", "本科。累计3年工作经验。\n\n项目：内部系统实施\n负责需求访谈、里程碑拆解、风险台账和验收材料，协调开发测试与使用部门排查上线阻塞；组织周会跟踪问题闭环，使用Excel维护排期。曾为Python开发团队协调资源，未编写Python代码。"),
    ("test", "技术支持", "大专。累计2年工作经验。\n\n项目：企业软件客户支持\n负责客户问题复现、远程部署说明和工单处理，使用SQL检查数据库连接与错误数据，阅读Linux日志定位权限问题，撰写故障处理手册并开展操作培训。未承担算法模型训练。"),
    ("test", "Python开发", "本科。累计1年工作经验。\n\n项目：营销数据整理\n使用Python、Pandas和SQL清洗线索表并输出日报，维护字段映射和重复记录检查；使用Excel核对导出结果。仅了解Django，未开发Web接口，也没有订单服务经验。"),
]


def build(dataset, source, synthesis, output, dense_dir, per_route=7):
    output = private_path(output)
    if output.exists(): raise FileExistsError("研究版本已存在，不覆盖")
    output.mkdir(parents=True, mode=0o700)
    records = read_jsonl(dataset/"jobs.jsonl"); by_id = {row["job_id"]:row for row in records}
    corpus = Corpus(source, dense_dir=dense_dir, records=records)
    if corpus.dense is None: raise ValueError("本次教师试验要求真实编码器，不接受静默回退")
    workflow = Workflow(corpus); profiles=[]
    for row in read_jsonl(synthesis):
        if not row.get("checks",{}).get("automatic_gate_pass"): continue
        job = by_id[row["seed_job_id"]]
        profiles.append({"query_id":"teacher-train-"+row["sample_id"], "profile_family_id":"synthesis-seed-"+row["seed_job_family_id"],
            "split":"train", "purpose":"ranker_training", "source_kind":"model_synthetic_from_job", "source_record_id":row["sample_id"],
            "seed_job_id":job["job_id"], "seed_job_family_id":job["job_family_id"], "text":row["decision"]["resume_text"],
            "preferences":{"city":"深圳","intent":job["category"] or job["title"]}, "is_human_gold":False})
    for i,(split,intent,text) in enumerate(FICTION):
        profiles.append({"query_id":f"teacher-authored-{i:02d}","profile_family_id":f"model-authored-v2-{i:02d}",
            "split":split,"purpose":"evaluation" if split=="test" else "ranker_training",
            "source_kind":"model_authored_fiction", "source_record_id":f"fiction-v2-{i:02d}","text":text,
            "preferences":{"city":"深圳","intent":intent},"is_human_gold":False})
    # 附加机器解析字段只供图分支使用；正文及人工确认偏好仍是匹配输入。
    for query in profiles:
        parsed=parse_profile(query["text"],query["preferences"])
        query.update(skills=parsed["skills"],tasks=parsed.get("tasks",{}),parser_version=parsed["parser_version"])
    all_tasks=[]; replay=[]; rank_inputs=[]; pools={}; rng=random.Random(42)
    for n,query in enumerate(profiles):
        profile=parse_profile(query["text"],query["preferences"])
        checks,allowed,rrf,sources,contexts=workflow.retrieval_state(profile,split=query["split"])
        if any("fallback" in name for name in sources) or "dense" not in sources:raise ValueError("试验中编码器失败，应复跑而非换引擎")
        indices=np.flatnonzero(allowed); rankable=np.flatnonzero(allowed & (rrf>0))
        intent_words=[word for word in tokens(profile["intent"]) if word not in {"开发","工程师","方向","岗位","人员","技术"}]
        def rule_score(i):
            job=corpus.jobs[i]; coverage=supported_matches(profile,job)[2]
            intent=sum(word in (job.title+" "+job.category).lower() for word in intent_words)/max(len(intent_words),1)
            confidence=sum(c["status"]=="pass" for c in checks[i])/max(len(checks[i]),1)
            district=1 if profile["district"] and profile["district"]==job.district else .5
            score=.25*float(rrf[i])/(float(rrf.max()) or 1)+.25*coverage+.2*contexts[i]["task_alignment"]+.2*intent+.05*confidence+.05*district
            return score*(.55 if intent_words and intent==0 else 1)
        selected=set()
        for name in ["bm25","dense"]:
            selected.update(sorted(indices,key=lambda i:(-contexts[i][name],corpus.jobs[i].id))[:per_route])
        selected.update(sorted(rankable,key=lambda i:(-rule_score(i),corpus.jobs[i].id))[:per_route])
        seed=query.get("seed_job_id")
        if seed: selected.add(corpus.lookup[seed])
        same_split=[i for i,j in enumerate(corpus.jobs) if by_id[j.id]["split"]==query["split"]]
        selected.update(rng.sample(same_split,2))
        # 同方向但硬条件不同的候选；是否负例仍交给接口判定，不按规则写标签。
        conflicts=[i for i in same_split if not allowed[i] and any(w in (corpus.jobs[i].title+corpus.jobs[i].category).lower() for w in intent_words)]
        selected.update(sorted(conflicts,key=lambda i:(-contexts[i]["bm25"],corpus.jobs[i].id))[:2])
        pool=[corpus.jobs[i].id for i in sorted(selected,key=lambda i:corpus.jobs[i].id)]; pools[query["query_id"]]=pool
        payload_hash=stable_hash({"text":query["text"],"preferences":query["preferences"]})
        for i in sorted(selected):
            job=corpus.jobs[i]; record=by_id[job.id]
            def safe(text):return CONTACT.sub("[联系方式已删除]",str(text)).replace(job.company,"某企业") if job.company else CONTACT.sub("[联系方式已删除]",str(text))
            material={"resume":query["text"],"preferences":query["preferences"],"job":{
                "title":safe(job.title),"category":job.category,"city":job.city,"salary":job.salary_raw,
                "requirements":safe(job.requirements),"description":safe(job.description)}}
            task=seal_task({"task_id":"pj-"+stable_hash([query["query_id"],job.id])[:24],"kind":"relevance",
                "query_id":query["query_id"],"profile_family_id":query["profile_family_id"],"job_id":job.id,
                "job_family_id":job.job_family_id,"split":query["split"],"source_snapshot":corpus.snapshot,
                "source_job_sha256":stable_hash(record),"source_query_sha256":stable_hash(query),
                "material":material,"input_hash":stable_hash(material),"stage":"teacher_relative_research"})
            all_tasks.append(task)
            vec=feature_vector(profile,job,contexts[i])
            replay.append({"query_id":query["query_id"],"profile_family_id":query["profile_family_id"],"job_id":job.id,
                "job_family_id":job.job_family_id,"split":query["split"],"snapshot":corpus.snapshot,
                "feature_version":FEATURE_VERSION,"features":{k:None if np.isnan(v) else float(v) for k,v in zip(FEATURE_NAMES,vec)},
                "retrieval_contract_hash":contract_info()["hash"],"encoder_contract":corpus.dense.encoder_contract,
                "query_payload_sha256":payload_hash,"eligible":bool(allowed[i])})
            rank_inputs.append({"query_id":query["query_id"],"job_id":job.id,"bm25":contexts[i]["bm25"],"dense":contexts[i]["dense"],
                                "rrf":contexts[i]["rrf"],"rule":rule_score(i),"eligible":bool(allowed[i])})
        print(json.dumps({"画像":n+1,"总画像":len(profiles),"共同池":len(pool)},ensure_ascii=False),flush=True)
    for name,rows in [("profiles",profiles),("train_profiles",[q for q in profiles if q["purpose"]=="ranker_training"]),
        ("evaluation_profiles",[q for q in profiles if q["purpose"]=="evaluation"]),("tasks",all_tasks),("replay",replay),("ranking_inputs",rank_inputs)]:
        write_jsonl(output/f"{name}.jsonl",rows)
    train_ids={q["query_id"] for q in profiles if q["purpose"]=="ranker_training"}
    write_jsonl(output/"train_replay.jsonl",[row for row in replay if row["query_id"] in train_ids])
    write_json(output/"replay_manifest.json",{"schema":"job-agent-ranker-replay-v2","replay_sha256":sha256(output/"train_replay.jsonl"),
        "jobs_sha256":sha256(dataset/"jobs.jsonl"),"profiles_sha256":sha256(output/"train_profiles.jsonl"),
        "feature_version":FEATURE_VERSION,"feature_names":list(FEATURE_NAMES),"retrieval_contract_hash":contract_info()["hash"],
        "encoder_contract":corpus.dense.encoder_contract})
    extraction=[]
    candidates=read_jsonl(dataset/"annotation_tasks.jsonl")
    for row in candidates:
        if row.get("split") not in {"train","dev"}:continue
        evidence=row.get("evidence",{});job=by_id.get(row.get("job_id"))
        if not job or not evidence.get("quote"):continue
        fragment=CONTACT.sub("[联系方式已删除]",evidence["quote"]).replace(job["company"],"某企业") if job["company"] else evidence["quote"]
        material={"fragment":fragment,"field":evidence["field"]}
        extraction.append({"task_id":"extract-v2-"+row["task_id"],"kind":"requirement_extraction","job_id":job["job_id"],
            "split":job["split"],"material":material,"input_hash":stable_hash(material),"source_job_sha256":stable_hash(job)})
    rng.shuffle(extraction);write_jsonl(output/"extraction_tasks.jsonl",extraction[:120])
    write_json(output/"pools.json",pools)
    report={"schema":"teacher-study-v2","dataset":str(dataset),"source_sha256":sha256(source),"profiles":len(profiles),
        "source_kinds":dict(Counter(q["source_kind"] for q in profiles)),"splits":dict(Counter(q["split"] for q in profiles)),
        "pairs":len(all_tasks),"extraction_tasks":min(120,len(extraction)),"per_route":per_route,"seed":42,
        "retrieval_contract":contract_info(),"encoder_contract":corpus.dense.encoder_contract,
        "human_gold_count":0,"scope":"明确合成画像与固定共同候选池，仅教师相对研究；不是独立真人效果。",
        "files":{p.name:sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/"manifest.json",report);return {k:report[k] for k in ["profiles","splits","pairs","extraction_tasks","human_gold_count"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["dataset","source","synthesis","output","dense-dir"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--per-route",type=int,default=7)
    args=parser.parse_args();print(json.dumps(build(args.dataset,args.source,args.synthesis,args.output,args.dense_dir,args.per_route),ensure_ascii=False))
