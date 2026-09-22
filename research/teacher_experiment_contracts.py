"""教师研究的来源锁、不可变阶段产物及共同池回放校验。"""
from collections import defaultdict
import json
import inspect
import math
from pathlib import Path

from job_agent.domain import PARSER_VERSION
from job_agent.ranking import FEATURE_NAMES, FEATURE_VERSION
from job_agent.retrieval_contract import contract_info, render_query, render_document, stable_hash
from research.common import read_jsonl, sha256, write_json

MODEL_DIRECTORY = "/home/xukunbo/.cache/job-agent/models/qwen3-embedding-0.6b"
LORA_ARGUMENTS = {"seed":42,"batch_size":2,"epochs":3,"max_steps":30,"max_length":512,
    "positives_per_query":2,"negatives_per_query":2,"learning_rate":5e-5,"temperature":.05,
    "allow_model_labels":True,"allow_fixture":False,"fixture_encoder":False,"evaluate_test":True,
    "model_dir":MODEL_DIRECTORY}
GRAPH_ARGUMENTS = {"seeds":"17,42,73","epochs":30,"patience":5,"hidden":32,"learning_rate":.003,
    "allow_model_labels":True,"allow_fixture":False,"evaluate_test":True}


def json_file(path):
    return json.loads(Path(path).read_text())


def read_annotations(study,annotation_dir,tasks):
    """复用API快照与规则校验；派生标签还须可从当前原始批次确定性重现。"""
    from research.annotation_snapshot import read_snapshot
    from research.api_annotation import row_usable, validate_label, MODEL, REASONING_EFFORT
    manifest,by_task,hashes,_=read_snapshot(annotation_dir,tasks,require_complete=True)
    if manifest.get("model")!=MODEL or manifest.get("reasoning_effort")!=REASONING_EFFORT:
        raise ValueError("标注不是用户指定的模型与推理档")
    if manifest.get("schema")=="job-agent-quote-field-reconciliation-v1":
        from research.reconcile_annotations import reconcile_label
        _,originals,source_hashes,_=read_snapshot(study/"annotation",tasks,require_complete=True)
        if manifest.get("source_files_sha256")!=source_hashes:
            raise ValueError("派生标签所绑定的原始批次或标签已变化")
        if (manifest.get("rule_validator_module_sha256")!=sha256(inspect.getsourcefile(validate_label))
            or manifest.get("reconciler_module_sha256")!=sha256(inspect.getsourcefile(reconcile_label))):
            raise ValueError("派生标签的字段纠正规则已变化，请生成新派生版本")
        for task in tasks:
            original=originals[task["task_id"]];derived=by_task[task["task_id"]]
            expected,_=reconcile_label(task,original)
            if stable_hash(derived)!=stable_hash(expected):
                raise ValueError("派生标签不是保全原等级/引用的确定性字段纠正")
    elif manifest.get("schema")!="job-agent-api-labels-v2":
        raise ValueError("不支持的模型标注来源格式")
    labels=[by_task[task["task_id"]] for task in tasks]
    if any(not row_usable(task,row) for task,row in zip(tasks,labels)):
        raise ValueError("共同池仍有未通过当前原文/来源规则的标签，不能删掉失败项后训练")
    return labels,hashes


def check_frozen(target, upstream):
    manifest = json_file(target/"manifest.json")
    if manifest.get("upstream_files") != upstream:
        raise ValueError("已冻结材料来源变化，拒绝覆盖")
    files = manifest.get("files", {})
    if not files or any(not (target/name).is_file() or sha256(target/name) != digest for name,digest in files.items()):
        raise ValueError("已冻结标签或查询文件被修改，不能沿用旧manifest")


def check_group_bridge(study, dataset, replay_manifest):
    payload = json_file(study/"group-text-vectors-r4.json")
    contract = payload["contract"]
    bridge = contract.get("semantic_bridge", {})
    expected = {"selected_jobs_file_sha256":sha256(study/"group-jobs-r4.jsonl"),
                "runtime_profiles_file_sha256":sha256(study/"group-profiles-r4.jsonl"),
                "tasks_sha256":sha256(study/"tasks.jsonl")}
    if any(contract.get(key) != value for key,value in expected.items()):
        raise ValueError("r4图文本向量与实际岗位/画像/任务文件不一致")
    if (bridge.get("semantic_jobs_file_sha256") != sha256(dataset/"jobs.jsonl")
        or bridge.get("parent_profiles_file_sha256") != sha256(study/"profiles.jsonl")
        or bridge.get("parser_version") != PARSER_VERSION
        or any(bridge.get(key) is not True for key in ("original_task_hashes_verified","raw_fields_unchanged","runtime_reproduction_verified"))):
        raise ValueError("r2标注→r4解析桥接不完整或解析器版本已改变")
    if contract["encoder"].get("encoder_contract_hash") != replay_manifest.get("encoder_contract",{}).get("encoder_contract_hash"):
        raise ValueError("GNN冻结初始化与BGE对照的编码器不同")
    return contract


def check_replay(study, dataset, replay, profiles, tasks):
    manifest = json_file(replay/"replay_manifest.json")
    if manifest.get("jobs_sha256") != sha256(dataset/"jobs.jsonl") or manifest.get("replay_sha256") != sha256(replay/"train_replay.jsonl"):
        raise ValueError("回放与当前r4岗位或训练特征不一致")
    if manifest.get("retrieval_contract_hash") != contract_info()["hash"]:
        raise ValueError("回放解析/检索口径不是当前运行时代码")
    if manifest.get("upstream_annotation_tasks_sha256") != sha256(study/"tasks.jsonl"):
        raise ValueError("回放候选与冻结标注任务版本不同")
    queries = {row["query_id"]:row for row in profiles}
    jobs = {row["job_id"]:row for row in read_jsonl(study/"group-jobs-r4.jsonl")}
    expected = {(row["query_id"],row["job_id"]):row for row in tasks}
    if len(expected) != len(tasks):
        raise ValueError("共同池人岗对重复")
    rows = read_jsonl(replay/"all_replay.jsonl")
    by_pair = {(row["query_id"],row["job_id"]):row for row in rows}
    if len(rows) != len(by_pair) or set(by_pair) != set(expected):
        raise ValueError("全部回放缺行、重复或含共同池外人岗对")
    pools = defaultdict(list)
    for pair,row in by_pair.items():
        query,job = queries[pair[0]],jobs[pair[1]]
        identities = {"profile_family_id":query["profile_family_id"],"job_family_id":job["job_family_id"],
            "split":query["split"],"snapshot":job["snapshot"],"feature_version":FEATURE_VERSION,
            "retrieval_contract_hash":manifest["retrieval_contract_hash"],"encoder_contract":manifest["encoder_contract"],
            "query_payload_sha256":stable_hash({"text":query["text"],"preferences":query.get("preferences",{})})}
        if any(row.get(key)!=value for key,value in identities.items()) or query["split"]!=job["split"] or type(row.get("eligible")) is not bool:
            raise ValueError("全部回放身份、画像正文、硬过滤或编码来源错误")
        values=row.get("features",{})
        if set(values)!=set(FEATURE_NAMES) or any(value is not None and (type(value) not in (int,float) or not math.isfinite(value)) for value in values.values()):
            raise ValueError("全部回放特征缺项或含非法数值")
        pools[pair[0]].append(pair[1])
    expected_train = {pair:row for pair,row in by_pair.items() if row["split"] in {"train","dev"}}
    actual_train = read_jsonl(replay/"train_replay.jsonl")
    if len(actual_train)!=len(expected_train) or {(r["query_id"],r["job_id"]):r for r in actual_train} != expected_train:
        raise ValueError("训练回放与全部共同池回放不是同一批特征")
    test_ids = {key for key,row in queries.items() if row.get("purpose")=="evaluation"}
    baseline = {}
    for name in ("bm25","bge","hybrid","rule"):
        rankings=read_jsonl(replay/f"{name}.jsonl")
        if len(rankings)!=len(test_ids) or {row["query_id"] for row in rankings}!=test_ids:
            raise ValueError("基线查询与完整独立评测集不一致")
        for row in rankings:
            key=row["query_id"];query=queries[key];candidates=sorted(pools[key])
            context={"query_payload_sha256":stable_hash({"text":query["text"],"preferences":query.get("preferences",{})}),
                "candidate_universe_count":len(candidates),"candidate_universe_sha256":stable_hash(candidates),
                "query_sha256":stable_hash(render_query(query)),
                "document_view_sha256":stable_hash([(jid,render_document(jobs[jid])) for jid in candidates]),
                "constraint_policy_sha256":stable_hash({"hard_filters":"explicit_fail_removed;unknown_retained","parser":PARSER_VERSION,"code":contract_info()["hash"]})}
            if any(row.get(field)!=value for field,value in context.items()):
                raise ValueError("基线复制了不同画像、岗位正文或解析器的输入合同")
            ids=[item["job_id"] for item in row["ranking"]]
            eligible={jid for jid in candidates if by_pair[key,jid]["eligible"]}
            if len(ids)!=len(set(ids)) or set(ids)!=eligible or any(type(item.get("score")) not in (int,float) or not math.isfinite(item["score"]) for item in row["ranking"]):
                raise ValueError("基线未使用同一完整已过滤候选池或分数非法")
        baseline[name]=rankings
    return rows,baseline,manifest


def make_run_contract(study,dataset,replay,label_file,training_python,gpu,repository,runner):
    inputs={"labels":label_file,"annotation-manifest":label_file.parent/"manifest.json", "dataset-manifest":dataset/"manifest.json","dataset-jobs":dataset/"jobs.jsonl"}
    for name in ("tasks.jsonl","profiles.jsonl","train_profiles.jsonl","evaluation_profiles.jsonl","group-jobs-r4.jsonl","group-profiles-r4.jsonl","group-text-vectors-r4.json"):
        inputs["study/"+name]=study/name
    for name in ("replay_manifest.json","all_replay.jsonl","train_replay.jsonl","bm25.jsonl","bge.jsonl","hybrid.jsonl","rule.jsonl"):
        inputs["replay/"+name]=replay/name
    sources=["job_agent/domain.py","job_agent/semantics.py","job_agent/encoder.py","job_agent/retrieval_contract.py","job_agent/ranking.py",
        "research/train_ranker.py","research/prepare_ranker.py","research/train_person_job_embedding.py","research/train_group_graph.py",
        "research/group_graph.py","research/review_contracts.py","research/metrics.py","research/score_rankings.py","research/train_embedding.py",
        "research/annotation_workflow.py","research/annotation_snapshot.py","research/api_annotation.py","research/reconcile_annotations.py"]
    model_root=Path(MODEL_DIRECTORY)
    model_files=sorted(path for path in model_root.iterdir() if path.suffix in (".json",".txt",".safetensors") and path.is_file())
    payload={"schema":"teacher-experiment-run-v2","inputs":{name:sha256(path) for name,path in inputs.items()},
        "source_code":{name:sha256(repository/name) for name in sources},"runner_sha256":sha256(runner),
        "guard_sha256":sha256(__file__),"base_model_files":{path.name:sha256(path) for path in model_files},
        "training_python":str(Path(training_python).resolve()),"gpu":str(gpu),
        "lora_arguments":LORA_ARGUMENTS,"graph_arguments":GRAPH_ARGUMENTS,
        "annotation_origin":"r2原文任务；训练使用已验证仅解析字段变化的r4",
        "teacher_relative":True,"eligible_for_production":False}
    return {**payload,"contract_sha256":stable_hash(payload)}


def lock_run(output,contract):
    path=output/"run_contract.json"
    if path.exists():
        if json_file(path)!=contract:
            raise ValueError("实验输入、r4语义、代码、基座或参数变化，拒绝复用旧模型；请使用新输出版本")
    else:
        if any((output/name).exists() for name in ("ranker","person-job-lora","group-graph")):
            raise ValueError("已有模型缺完整实验来源锁，不能自动采纳为本次结果")
        write_json(path,contract)


def seal_stage(directory,run_hash):
    path=directory/"completion-seal.json"
    files={str(item.relative_to(directory)):sha256(item) for item in sorted(directory.rglob("*")) if item.is_file() and item!=path}
    value={"run_contract_sha256":run_hash,"files":files}
    if path.exists():
        if json_file(path)!=value:
            raise ValueError("已完成阶段的模型、报告或来源文件被替换")
    else:write_json(path,value)


def check_stage(directory,kind,inputs,retrieval_path=None,text_vectors=None):
    manifest=json_file(directory/"manifest.json")
    expected={key:sha256(path) for key,path in inputs.items()}
    if manifest.get("input_hashes")!=expected:
        raise ValueError("训练器来源与本次r4岗位、画像或标签不一致")
    required=LORA_ARGUMENTS if kind=="lora" else GRAPH_ARGUMENTS
    if any(manifest.get("arguments",{}).get(key)!=value for key,value in required.items()):
        raise ValueError("训练器实际参数与预注册预算不同")
    if kind=="lora":
        if manifest.get("status")!="completed" or manifest.get("fixture_only") or not manifest.get("model_weak_supervision"):
            raise ValueError("LoRA未完成或不是当前教师研究")
        if manifest.get("retrieval_contract_file_sha256")!=sha256(retrieval_path) or stable_hash(manifest.get("retrieval_contract"))!=stable_hash(contract_info()):
            raise ValueError("LoRA文本模板、解析或分块合同过期")
        if not manifest.get("reload_verification",{}).get("passed"):
            raise ValueError("LoRA未通过保存后的服务编码器重载验证")
        for name in ("baseline_metrics.json","adapted_metrics.json","adapter/adapter_model.safetensors"):
            if not (directory/name).is_file():raise ValueError("LoRA模型或共同池输出缺失")
    else:
        if manifest.get("text_vectors_sha256")!=sha256(text_vectors):
            raise ValueError("要求组图使用了旧冻结文本向量")
        summary=json_file(directory/"summary.json")
        if summary.get("status")!="completed" or summary.get("fixture_only") or not summary.get("model_weak_supervision") or not summary.get("test_evaluated"):
            raise ValueError("要求组图未完成当前教师研究与预注册测试")
        variants=summary.get("variants",[])
        expected_variants={(mode,seed) for mode in ("pool","graph","random_graph") for seed in (17,42,73)}
        if len(variants)!=len(expected_variants) or {(row.get("mode"),row.get("seed")) for row in variants}!=expected_variants:
            raise ValueError("要求组图缺少预注册模型/种子，不能挑选已完成的子集")
    return manifest


def scores_from_report(report,expected_pools):
    rows=report.get("per_query",[])
    if len(rows)!=len(expected_pools) or {row.get("query_id") for row in rows}!=set(expected_pools):
        raise ValueError("训练器报告缺查询或重复，不能改变共同评测分母")
    output={}
    for row in rows:
        ids,scores=row.get("candidate_ids",[]),row.get("candidate_scores",[])
        if len(ids)!=len(scores) or len(ids)!=len(set(ids)) or set(ids)!=set(expected_pools[row["query_id"]]):
            raise ValueError("训练器候选ID与分数不齐或不属于完整共同池")
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in scores):
            raise ValueError("训练器输出非有限数值")
        output[row["query_id"]]=dict(zip(ids,scores))
    return output
