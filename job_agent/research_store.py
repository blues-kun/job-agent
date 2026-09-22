"""私有研究产物的只读浏览与可追溯人工标注账本。"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import re
import uuid
from collections import Counter
import json
import sqlite3
import time
import os
import hashlib

from research.common import private_path, read_jsonl, sha256
from .domain import display_text
from research.annotation_workflow import canonical_label, seal_task, validate_tasks, validate_query_binding
from research.review_contracts import (digest, file_sha, identifier, new_private_directory,
    profile_family, validate_profiles, write_json as exclusive_json, write_jsonl as exclusive_jsonl)



def public_material(value):
    if isinstance(value,str):return display_text(value)
    if isinstance(value,list):return [public_material(item) for item in value]
    if isinstance(value,dict):return {key:public_material(item) for key,item in value.items()}
    return value


class ResearchStore:
    def __init__(self,directory,private_root):
        self.directory=private_path(directory) if directory else None
        self.db=private_path(private_root)/"research_annotations.sqlite3"
        self.manifest={};self.jobs={};self.tasks={};self.queries={};self.material_hashes={}
        self.review_tasks={};self.review_blockers={};self.review_source_jobs={};self.task_source_lineage={}
        teacher=os.environ.get("JOB_AGENT_TEACHER_STUDY_DIR")
        self.teacher_directory=private_path(teacher) if teacher else None
        self.teacher_review_queue={"status":"not_configured","tasks":0}
        if self.directory and (self.directory/"manifest.json").exists():
            self.manifest=json.loads((self.directory/"manifest.json").read_text())
            if not {"jobs.jsonl","annotation_tasks.jsonl"}.issubset(self.manifest.get("files",{})):
                raise ValueError("数据清单缺少必需文件")
            for filename,details in self.manifest.get("files",{}).items():
                if Path(filename).name!=filename or sha256(self.directory/filename)!=details["sha256"]:
                    raise ValueError("研究数据清单校验失败，不加载变更后的材料")
            self.jobs={job["job_id"]:job for job in read_jsonl(self.directory/"jobs.jsonl")}
            configured=os.environ.get("JOB_AGENT_BENCHMARK_DIR")
            benchmark_dir=private_path(configured) if configured else self.directory/"benchmark"
            if not configured and not (benchmark_dir/"manifest.json").exists():
                versions=list(benchmark_dir.glob("benchmark-*/manifest.json"))
                if len(versions)==1:benchmark_dir=versions[0].parent
                elif len(versions)>1:raise ValueError("存在多个评测版本，请显式指定JOB_AGENT_BENCHMARK_DIR")
            benchmark=benchmark_dir/"manifest.json"
            if not benchmark.exists() and any((benchmark_dir/name).exists() for name in ["tasks.jsonl","queries.jsonl"]):
                raise ValueError("评测材料缺少版本清单")
            if benchmark.exists():
                contract=json.loads(benchmark.read_text())
                if not {"tasks.jsonl","queries.jsonl"}.issubset(contract.get("files",{})):raise ValueError("评测清单缺少必需文件")
                if contract.get("config",{}).get("jobs_sha256")!=sha256(self.directory/"jobs.jsonl"):
                    raise ValueError("评测任务不属于当前岗位版本")
                for filename,details in contract.get("files",{}).items():
                    expected=details.get("sha256") if isinstance(details,dict) else details
                    if Path(filename).name!=filename or sha256(benchmark.parent/filename)!=expected:
                        raise ValueError("评测任务校验失败")
            for path in [self.directory/"annotation_tasks.jsonl",benchmark_dir/"tasks.jsonl"]:
                if path.exists():
                    for item in read_jsonl(path):self.tasks[item["task_id"]]=item
            path=benchmark_dir/"queries.jsonl"
            if path.exists():self.queries={row["query_id"]:row for row in read_jsonl(path)}
            synthesis=os.environ.get("JOB_AGENT_SYNTHESIS_DIR")
            if synthesis:
                folder=private_path(synthesis)
                contract=json.loads((folder/"manifest.json").read_text())
                if contract["config"]["jobs_sha256"]!=sha256(self.directory/"jobs.jsonl") or sha256(folder/"samples.jsonl")!=contract["files"]["samples.jsonl"]["sha256"]:
                    raise ValueError("合成简历来源或内容校验失败")
                for row in read_jsonl(folder/"samples.jsonl"):
                    task={"task_id":"synthesis-"+row["sample_id"],"kind":"resume_synthesis","job_id":row["seed_job_id"],
                          "split":row["split"],"label":None,"text":json.dumps(row["decision"],ensure_ascii=False,indent=2),
                          "source_evidence":row["source_evidence"],"note":"完全虚构，需核对技能范围与表述一致性；点击正确也不会自动成为训练金标。"}
                    self.tasks[task["task_id"]]=task
            additional=os.environ.get("JOB_AGENT_RETRIEVAL_REVIEW_DIR")
            if additional:
                folder=private_path(additional);contract=json.loads((folder/"manifest.json").read_text())
                name="supplemental_top10_review_candidates.jsonl"
                if contract["source_snapshot"]["jobs_sha256"]!=sha256(self.directory/"jobs.jsonl") or contract["queries_sha256"]!=sha256(benchmark_dir/"queries.jsonl") or sha256(folder/name)!=contract["outputs"][name]["sha256"]:
                    raise ValueError("补审候选来源或内容校验失败")
                pairs={(task.get("query_id"),task.get("job_id")) for task in self.tasks.values() if task.get("query_id")}
                for row in read_jsonl(folder/name):
                    pair=(row["query_id"],row["job_id"])
                    if pair in pairs:continue
                    pairs.add(pair)
                    key="supplement-"+hashlib.sha256((pair[0]+":"+pair[1]).encode()).hexdigest()[:24]
                    self.tasks[key]={"task_id":key,"kind":"relevance","job_id":row["job_id"],"query_id":row["query_id"],"split":row["split"],"label":None}
            if self.teacher_directory:
                from research.teacher_dashboard import verified_manifest
                try:
                    teacher_manifest=verified_manifest(self.teacher_directory)
                    if teacher_manifest.get("schema")!="teacher-study-v2" or teacher_manifest.get("source_sha256")!=self.manifest.get("source_sha256"):
                        raise ValueError("教师材料与岗位来源不同")
                    teacher_profiles=read_jsonl(self.teacher_directory/"profiles.jsonl")
                    validate_profiles(teacher_profiles)
                    teacher_tasks=read_jsonl(self.teacher_directory/"tasks.jsonl")
                    validate_tasks(teacher_tasks)
                    incoming_queries={row["query_id"]:row for row in teacher_profiles}
                    original_jobs=self.jobs
                    source_manifest_sha=file_sha(self.directory/"manifest.json")
                    source_jobs_sha=file_sha(self.directory/"jobs.jsonl")
                    if teacher_manifest.get("dataset"):
                        teacher_dataset=private_path(teacher_manifest["dataset"])
                        source_manifest=verified_manifest(teacher_dataset)
                        source_manifest_sha=file_sha(teacher_dataset/"manifest.json")
                        source_jobs_sha=file_sha(teacher_dataset/"jobs.jsonl")
                        if "replay_manifest.json" not in teacher_manifest["files"]:
                            raise ValueError("教师来源缺原始回放清单，无法绑定源岗位文件")
                        replay_manifest=json.loads((self.teacher_directory/"replay_manifest.json").read_text("utf-8"))
                        if replay_manifest.get("jobs_sha256")!=source_jobs_sha or source_manifest.get("source_sha256")!=self.manifest.get("source_sha256"):
                            raise ValueError("教师原始岗位文件或原表来源SHA不同")
                        needed={task.get("job_id") for task in teacher_tasks}
                        original_jobs={}
                        with (teacher_dataset/"jobs.jsonl").open(encoding="utf-8") as stream:
                            for line in stream:
                                if line.strip():
                                    row=json.loads(line)
                                    if row.get("job_id") in needed:original_jobs[row["job_id"]]=row
                    parsed_fields={"skills","tasks","groups","requirement_ast","parser_version","experience_requirements",
                                   "education_requirements","experience_min","education_min"}
                    original_view=lambda row:{key:value for key,value in row.items() if key not in parsed_fields}
                    lineage={}
                    for query in teacher_profiles:
                        if query["query_id"] in self.queries and digest(self.queries[query["query_id"]])!=digest(query):
                            raise ValueError("教师画像ID与已有材料冲突")
                    for task in teacher_tasks:
                        job=self.jobs.get(task.get("job_id"));source_job=original_jobs.get(task.get("job_id"));query=incoming_queries.get(task.get("query_id"))
                        if job is None or source_job is None or query is None or task.get("source_job_sha256")!=digest(source_job):
                            raise ValueError("教师任务源岗位版本不同或缺画像")
                        if original_view(job)!=original_view(source_job):
                            raise ValueError("教师岗位与当前岗位存在原文、版本、家族或分区变更")
                        validate_query_binding(task,query)
                        lineage[task["task_id"]]={"teacher_source_manifest_sha256":source_manifest_sha,
                            "teacher_source_jobs_sha256":source_jobs_sha,"teacher_source_job_sha256":digest(source_job),
                            "current_dataset_manifest_sha256":file_sha(self.directory/"manifest.json"),
                            "current_job_sha256":digest(job),"immutable_fields_sha256":digest(original_view(job)),
                            "verified_fields":sorted(original_view(job)),"teacher_parser_version":source_job.get("parser_version"),
                            "current_parser_version":job.get("parser_version"),"sealed_task_unchanged":True,
                            "note":"只核对原始字段、岗位版本与分区一致；旧教师评审不能作为新解析器的抽取金标。"}
                        if task["task_id"] in self.tasks and digest(task)!=digest(self.tasks[task["task_id"]]):
                            raise ValueError("教师任务ID与已有材料冲突")
                    self.queries.update(incoming_queries)
                    self.tasks.update({task["task_id"]:task for task in teacher_tasks})
                    self.review_source_jobs.update({task["task_id"]:original_jobs[task["job_id"]] for task in teacher_tasks})
                    self.task_source_lineage.update(lineage)
                    self.teacher_review_queue={"status":"verified_blind_materials","tasks":len(teacher_tasks),
                                               "teacher_answers_exposed":False,"source_task_hash_preserved":True,
                                               "source_manifest_sha256":source_manifest_sha,
                                               "current_parser_gold_claimed":False}
                except (ValueError,KeyError,TypeError,OSError):
                    self.teacher_review_queue={"status":"validation_failed","tasks":0,
                        "notice":"教师材料尚未通过岗位版本、画像与任务SHA核验，未加入人工队列。"}
            for task_id,task in self.tasks.items():
                job=self.review_source_jobs.get(task_id,self.jobs.get(task.get("job_id",task.get("job_version_id"))))
                kind=task.get("kind",task.get("task_type"))
                if job is None and kind!="clarification":raise ValueError("标注任务引用了不存在的岗位")
                if job is not None and task.get("split") and task["split"]!=job["split"]:raise ValueError("任务与岗位分区不一致")
                query=self.queries.get(task.get("query_id"))
                if task.get("query_id") and (query is None or query["split"]!=(job["split"] if job else task.get("split"))):raise ValueError("任务画像不存在或与岗位分区不一致")
                material={"task":task,"job":job,"query":query,"dataset":self.manifest.get("config_hash",self.manifest.get("source_sha256"))}
                try:
                    canonical_task=self._canonical_task(task,job,query)
                    self.review_tasks[task_id]=canonical_task
                    self.material_hashes[task_id]=canonical_task["task_hash"]
                except ValueError as error:
                    # 历史材料可浏览，但不能补造画像家族来伪装成可冻结金标。
                    self.review_blockers[task_id]=str(error)
                    self.material_hashes[task_id]=digest(material)
        self.db.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS annotations (id INTEGER PRIMARY KEY, task_id TEXT, annotator TEXT, payload TEXT, created REAL)")
        self.db.chmod(0o600)

    def _latest(self):
        with sqlite3.connect(self.db) as connection:
            rows=connection.execute("SELECT task_id,annotator,payload FROM annotations WHERE id IN (SELECT MAX(id) FROM annotations GROUP BY task_id,annotator)").fetchall()
        result={}
        for task,annotator,raw in rows:
            payload=json.loads(raw)
            if task in self.tasks and payload.get("task_hash")==self.material_hashes.get(task):result[(task,annotator)]=payload
        return result

    def summary(self):
        latest=self._latest();counts=Counter(task for task,_ in latest)
        return {"available":bool(self.jobs),"jobs":len(self.jobs),"task_count":len(self.tasks),"queries":len(self.queries),
                "task_types":dict(Counter(task.get("kind",task.get("task_type")) for task in self.tasks.values())),
                "annotated_tasks":len(counts),"double_annotated_tasks":sum(count>=2 for count in counts.values()),
                "annotation_count":len(latest),"snapshot":self.manifest.get("snapshot",self.manifest.get("source_sha256","")),
                "status":"待人工标注与仲裁" if len(counts)<len(self.tasks) else "已有标注，仍需独立身份核实与仲裁",
                "note":"模型训练日志不等于推荐金标；评审者代号不提供多人身份认证。",
                "review_contract":{"schema":"job-agent-workbench-review-v2","ready_tasks":len(self.review_tasks),
                    "blocked_tasks":len(self.review_blockers),"eligible_for_gold":False,
                    "next_step":"导出原始记录，核实独立评审身份并显式仲裁；旧材料或旧评审须补全来源或重新评审。"},
                "teacher_study":self.teacher_summary()}

    def task(self,annotator,kind=None):
        latest=self._latest()
        candidates=[item for key,item in self.tasks.items() if (key,annotator) not in latest and (not kind or item.get("kind",item.get("task_type"))==kind)]
        if not candidates:return {"task":None,"remaining":0}
        original=candidates[0]
        task=deepcopy(self.review_tasks.get(original["task_id"],original))
        task["task_hash"]=self.material_hashes[task["task_id"]]
        task["review_contract_ready"]=task["task_id"] in self.review_tasks
        task["review_blocker"]=self.review_blockers.get(task["task_id"])
        if task["task_id"] in self.task_source_lineage:task["source_lineage"]=self.task_source_lineage[task["task_id"]]
        job_id=task.get("job_id",task.get("job_version_id"))
        if job_id in self.jobs:
            job=self.jobs[job_id]
            task["job"]={key:job[key] for key in ["job_id","title","category","requirements","description","salary_raw","address"] if key in job}
        if task.get("query_id") in self.queries:task["query"]=self.queries[task["query_id"]]
        # 盲审不下发另一评审者的答案或模型排序位置。
        task.pop("label",None);task.pop("labels",None);task.pop("proposed_label",None)
        task.pop("pool_sources",None);task.pop("retrieval_sources",None);task.pop("rank",None);task.pop("sources",None);task.pop("score",None)
        return {"task":public_material(task),"remaining":len(candidates)}

    def _canonical_task(self, task, job, query):
        kind=task.get("kind",task.get("task_type"))
        if kind not in {"relevance","requirement_extraction","clarification"}:
            raise ValueError("此任务类型暂无v2金标合同，保留原始核对记录")
        if task.get("task_hash"):
            validate_tasks([task])
            result=deepcopy(task)
            if result.get("source_job_sha256") and result["source_job_sha256"]!=digest(job):
                raise ValueError("任务岗位原文版本已变化")
            if query is not None:
                validate_query_binding(result,query)
        else:
            result={key:deepcopy(value) for key,value in task.items() if key not in
                    {"label","labels","proposed_label","pool_sources","retrieval_sources","rank","score","sources","task_type","task_hash"}}
            result.update(kind=kind,split=job.get("split") if job else task.get("split"),
                          stage=task.get("stage","workbench_unadjudicated"))
            if job is not None:
                result.update(job_id=job["job_id"],source_job_sha256=digest(job),
                              source_snapshot=job.get("snapshot",self.manifest.get("source_sha256")))
                if job.get("job_family_id"):result["job_family_id"]=job["job_family_id"]
            if kind in {"relevance","clarification"}:
                if query is None:
                    raise ValueError("缺少任务画像，不能导出相关性金标")
                result.update(profile_family_id=profile_family(query),source_query_sha256=digest(query))
            result=seal_task(result)
            validate_tasks([result])
        if kind=="requirement_extraction":
            evidence=result.get("evidence",{})
            field=evidence.get("field");text=job.get(field,"")
            start,end=evidence.get("start"),evidence.get("end")
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text) or text[start:end]!=evidence.get("quote"):
                raise ValueError("抽取任务缺可核对的原文跨度")
        return result

    def _canonical_extraction(self,task,extraction):
        if not isinstance(extraction,dict) or not isinstance(extraction.get("groups"),list):
            raise ValueError("抽取标签需要groups数组")
        evidence=task["evidence"];fragment=evidence["quote"]
        nodes=0
        def anchored(group,depth=0):
            nonlocal nodes
            nodes+=1
            if depth>8 or nodes>150 or not isinstance(group,dict):
                raise ValueError("要求树超过深度或节点上限")
            skills=group.get("skills",[]);children=group.get("children",[])
            if not isinstance(skills,list) or not isinstance(children,list):
                raise ValueError("要求组skills/children必须为数组")
            entries=[]
            for item in skills:
                if not isinstance(item,dict):raise ValueError("技能标注必须为对象")
                quote=item.get("quote","");occurrence=item.get("occurrence",0)
                positions=[match.start() for match in re.finditer(re.escape(quote),fragment)] if isinstance(quote,str) and quote else []
                if not identifier(item.get("skill")) or type(occurrence) is not int or not 0<=occurrence<len(positions):
                    raise ValueError("技能引用必须出现在本段原文中")
                start=evidence["start"]+positions[occurrence]
                entries.append({"skill":item["skill"],"quote":quote,"field":evidence["field"],"start":start,"end":start+len(quote)})
            if group.get("logic")=="single" and len(entries)+len(children)!=1:
                raise ValueError("single须恰好一个技能或子组")
            result={"logic":group.get("logic"),"modality":group.get("modality"),"skills":entries}
            if children:result["children"]=[anchored(child,depth+1) for child in children]
            return result
        canonical={"groups":[anchored(group) for group in extraction["groups"]]}
        return canonical_label(task,{"extraction":canonical})["extraction"]

    def annotate(self,task_id,annotator,grade,decision,notes,extraction=None,independent=False,task_hash=None,action=None,no_match_evidence_ref=None):
        if task_id not in self.tasks:raise KeyError("任务不存在")
        if not identifier(annotator):raise ValueError("评审代号不能为空")
        if task_hash is not None and task_hash!=self.material_hashes[task_id]:
            raise ValueError("浏览的任务版本已变化，请重新打开材料")
        if type(independent) is not bool:raise ValueError("独立评审声明必须为布尔值")
        task=self.review_tasks.get(task_id,self.tasks[task_id])
        kind=task.get("kind",task.get("task_type"))
        if kind=="relevance" and (type(grade) is not int or grade not in range(4)):raise ValueError("人岗相关性任务必须标0至3整数等级")
        if kind=="relevance" and (decision or extraction is not None or action is not None):raise ValueError("相关性任务不接受其他任务标签")
        if kind not in {"relevance","clarification"} and not decision:raise ValueError("请给出抽取核对结论")
        if kind!="relevance" and grade is not None:raise ValueError("非相关性任务不接受等级标签")
        canonical=None
        if kind=="requirement_extraction":
            if extraction is None:raise ValueError("需求任务须填写技能、逻辑和模态，不能只提交正确/错误")
            canonical=self._canonical_extraction(task,extraction)
        elif extraction is not None:raise ValueError("该任务不接受抽取标签")
        if kind=="clarification":
            canonical_label(task,{"action":action,"no_match_evidence_ref":no_match_evidence_ref})
        elif action is not None or no_match_evidence_ref is not None:
            raise ValueError("非动作任务不能填写追问标签")
        payload={"review_id":uuid.uuid4().hex,"reviewer_id":annotator,"task_id":task_id,
                 "grade":grade,"decision":decision,"notes":notes,"extraction":canonical,
                 "task_hash":self.material_hashes[task_id],"source":"human_entered_unadjudicated",
                 "independent":independent,"created_at":datetime.now(timezone.utc).isoformat(),
                 "submission_hash_checked":task_hash is not None,"review_contract_ready":task_id in self.review_tasks}
        if kind=="clarification":payload.update(action=action,no_match_evidence_ref=no_match_evidence_ref)
        with sqlite3.connect(self.db) as connection:
            connection.execute("INSERT INTO annotations(task_id,annotator,payload,created) VALUES(?,?,?,?)",(task_id,annotator,json.dumps(payload,ensure_ascii=False),time.time()))
        return {"saved":True,"status":"人工输入，尚未仲裁为金标","review_id":payload["review_id"],
                "independent_declared":independent,"identity_verified":False,
                "review_contract_ready":payload["review_contract_ready"] and payload["submission_hash_checked"]}

    def export_review_bundle(self,output,task_ids=None):
        """导出全部选定任务和其最新原始评审；不补造旧评审身份、时间或独立性。"""
        selected=set(self.tasks) if task_ids is None else set(task_ids)
        if not selected or selected-set(self.tasks):raise ValueError("请选择存在的非空任务集")
        ready=sorted(selected & set(self.review_tasks))
        if not ready:raise ValueError("所选任务缺v2身份/原文合同，须补全材料并重新评审")
        tasks=[self.review_tasks[key] for key in ready]
        profiles=[self.queries[key] for key in sorted({row["query_id"] for row in tasks if row.get("query_id")})]
        if profiles:validate_profiles(profiles)
        reviews=[];legacy=[]
        for (task_id,annotator),payload in self._latest().items():
            if task_id not in selected:continue
            if task_id not in ready or not payload.get("review_id") or not payload.get("submission_hash_checked"):
                legacy.append({"task_id":task_id,"annotator":annotator,"payload":payload,
                               "reason":"旧输入缺任务确认血缘，不转换为v2原始评审；需重新提交"})
                continue
            review={key:deepcopy(payload[key]) for key in ("review_id","reviewer_id","task_id","task_hash","source","independent","created_at","notes")}
            task=self.review_tasks[task_id]
            if task["kind"]=="relevance":review["grade"]=payload["grade"]
            elif task["kind"]=="requirement_extraction":review["extraction"]=payload["extraction"]
            else:review.update(action=payload["action"],no_match_evidence_ref=payload.get("no_match_evidence_ref"))
            canonical_label(task,review)
            reviews.append(review)
        target=new_private_directory(output)
        blocked=[{"task_id":key,"reason":self.review_blockers[key]} for key in sorted(selected & set(self.review_blockers))]
        for name,rows in [("tasks.jsonl",tasks),("queries.jsonl",profiles),("reviews.jsonl",reviews),
                          ("legacy_reviews_needing_resubmission.jsonl",legacy),("blocked_tasks.jsonl",blocked)]:
            exclusive_jsonl(target/name,rows)
        exclusive_json(target/"task_source_lineage.json",{key:self.task_source_lineage[key] for key in ready if key in self.task_source_lineage})
        aliases=sorted({review["reviewer_id"] for review in reviews})
        exclusive_json(target/"reviewer_registry.template.json",{
            "attestation":{"record_id":None,"attested_by":None,"attested_at":None,
                           "identity_check_performed":False,"independent_review_process_confirmed":False},
            "reviewers":[{"reviewer_id":alias,"person_id":None,"identity_verified":False} for alias in aliases],
            "note":"这是待真人核实填写的模板；不同代号不证明是不同人，不能将模型请求登记为真人。"})
        exclusive_json(target/"manifest.json",{"schema":"job-agent-workbench-review-export-v2",
            "files":{path.name:file_sha(path) for path in target.iterdir()},"source_dataset_sha256":digest(self.manifest),
            "selected_tasks":len(selected),"exported_tasks":len(tasks),"reviews":len(reviews),"legacy_reviews":len(legacy),
            "blocked_tasks":len(blocked),"independence_not_declared":sum(not row["independent"] for row in reviews),
            "human_gold_created":False,"identity_verified":False,
            "notice":"导出只保留输入血缘；核实registry后运行review-status，再显式freeze。未审任务仍保留，不能静默丢弃。"})
        return {"output":str(target),"tasks":len(tasks),"reviews":len(reviews),"blocked_tasks":len(blocked),
                "legacy_reviews":len(legacy),"human_gold_created":False}

    def teacher_summary(self):
        from research.teacher_dashboard import load_teacher_dashboard
        result=load_teacher_dashboard(self.teacher_directory,self.manifest.get("source_sha256"))
        result["human_review_queue"]=self.teacher_review_queue
        return result

    def graph(self,job_id):
        job=self.jobs.get(job_id)
        if not job:return None
        return public_material({"job_id":job_id,"title":job["title"],"job_family_id":job["job_family_id"],"split":job["split"],
                "skills":job["skills"],"groups":job["groups"],"source_record_ids":job["source_record_ids"],
                "notice":"关系由原文机器解析，未经人工确认；共现和模型预测不作为事实证据。"})

    def reports(self):
        if not self.directory:return []
        root=private_path(os.environ.get("JOB_AGENT_EXPERIMENT_DIR",str(self.directory.parent.parent/"runs")))
        output=[]
        for path in sorted(root.glob("*/metrics.json")):
            try:
                value=json.loads(path.read_text())
                output.append({"run":path.parent.name,"metrics":value})
            except (ValueError,OSError):continue
        return output
