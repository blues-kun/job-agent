"""接口标签全部完成后执行冻结、排序、LoRA、要求组图及共同池比较；任一步失败不发布成功。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from research.common import read_jsonl,write_json,write_jsonl,private_path,sha256
from research.annotation_workflow import freeze_model_dataset
from research.review_contracts import validate_qrel
from research.prepare_ranker import materialize
from research.train_ranker import train
from research.score_rankings import compare_models
from job_agent.ranking import FEATURE_NAMES
from research.teacher_experiment_contracts import (check_frozen, check_group_bridge, check_replay,
    make_run_contract, lock_run, seal_stage, check_stage, scores_from_report, read_annotations, MODEL_DIRECTORY)


def command(args,log):
    with log.open("w") as stream:
        result=subprocess.run([str(item) for item in args],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"子步骤失败（退出码{result.returncode}），检查私有日志 {log.name}")


def run(study,dataset,replay,output,training_python,gpu,wait_seconds=0,annotation_dir=None):
    output=private_path(output);output.mkdir(parents=True,exist_ok=True,mode=0o700)
    progress=output/"progress.json"
    def stage(name,**extra):
        value={"stage":name,"teacher_relative":True,"human_gold_count":0,"eligible_for_production":False,**extra}
        write_json(progress,value);print(json.dumps(value,ensure_ascii=False),flush=True)
    stage("等待全部共同池标签")
    annotation_dir=Path(annotation_dir) if annotation_dir is not None else study/"annotation"
    deadline=time.monotonic()+wait_seconds
    while True:
        label_file=annotation_dir/"labels.jsonl"
        manifest=json.loads((annotation_dir/"manifest.json").read_text())
        if manifest.get("status")=="completed" and label_file.exists():break
        if time.monotonic()>=deadline:raise ValueError("接口共同池标签尚未全部完成，未启动训练；可在完成后复跑")
        time.sleep(30)
    tasks=read_jsonl(study/"tasks.jsonl");profiles=read_jsonl(study/"profiles.jsonl")
    labels,annotation_hashes=read_annotations(study,annotation_dir,tasks)
    if len(labels)!=len(tasks) or {r["task_id"] for r in labels}!={r["task_id"] for r in tasks}:raise ValueError("标签数量或ID不齐")
    for row in labels:validate_qrel(row,allow_model_labels=True)
    repository=Path(__file__).resolve().parents[1]
    replay_rows,baseline,replay_manifest=check_replay(study,dataset,replay,profiles,tasks)
    group_contract=check_group_bridge(study,dataset,replay_manifest)
    run_contract=make_run_contract(study,dataset,replay,label_file,training_python,gpu,repository,Path(__file__))
    lock_run(output,run_contract)
    stage("冻结标签",pairs=len(labels),grades=dict(Counter(row["grade"] for row in labels)))
    sources={"tasks":sha256(study/"tasks.jsonl"),"labels":sha256(label_file),"profiles":sha256(study/"profiles.jsonl"),
             "annotation_manifest":sha256(annotation_dir/"manifest.json")}
    for name,purpose in [("training","ranker_training"),("evaluation","evaluation")]:
        subset=[q for q in profiles if q["purpose"]==purpose];ids={q["query_id"] for q in subset}
        target=output/name
        if not target.exists():freeze_model_dataset([t for t in tasks if t["query_id"] in ids],subset,[r for r in labels if r["query_id"] in ids],target,sources)
        check_frozen(target,sources)
    stage("训练LambdaRank")
    rows=materialize(dataset,study/"train_profiles.jsonl",output/"training/qrels.jsonl",replay/"train_replay.jsonl",replay/"replay_manifest.json",
                     allow_model_labels=True,evaluation_profiles=study/"evaluation_profiles.jsonl")
    row_file=output/"ranker_rows.jsonl"
    if row_file.exists() and read_jsonl(row_file)!=rows:raise ValueError("现有物化行与当前来源不同")
    if not row_file.exists():write_jsonl(row_file,rows)
    if not (output/"ranker/metrics.json").exists():train(row_file,output/"ranker",dataset=dataset,allow_model_labels=True)
    else:
        if json.loads((output/"ranker/manifest.json").read_text())["input_sha256"]!=sha256(row_file):raise ValueError("排序模型来自不同样本")
    ranker_manifest=json.loads((output/"ranker/manifest.json").read_text())
    if ranker_manifest.get("model_sha256")!=sha256(output/"ranker/model.txt"):
        raise ValueError("排序模型文件与训练manifest不一致")
    seal_stage(output/"ranker",run_contract["contract_sha256"])
    stage("训练人岗LoRA")
    lora=output/"person-job-lora"
    base=[training_python,"-B","-m","research.train_person_job_embedding","--jobs",study/"group-jobs-r4.jsonl","--profiles",study/"group-profiles-r4.jsonl",
          "--qrels",label_file,"--output",lora,"--retrieval-contract",Path(__file__).resolve().parents[1]/"job_agent/retrieval_contract.py",
          "--model-dir",MODEL_DIRECTORY,"--gpu",gpu,"--seed","42","--batch-size","2","--epochs","3","--max-steps","30",
          "--max-length","512","--positives-per-query","2","--negatives-per-query","2","--learning-rate","0.00005","--temperature","0.05",
          "--allow-model-labels","--evaluate-test"]
    if not (lora/"manifest.json").exists():command(base,output/"embedding.log")
    model_inputs={"jobs":study/"group-jobs-r4.jsonl","profiles":study/"group-profiles-r4.jsonl","qrels":label_file}
    lora_manifest=check_stage(lora,"lora",model_inputs,retrieval_path=repository/"job_agent/retrieval_contract.py")
    seal_stage(lora,run_contract["contract_sha256"])
    stage("训练三种子要求组图")
    graph=output/"group-graph"
    if not (graph/"manifest.json").exists():
        command([training_python,"-B","-m","research.train_group_graph","--jobs",study/"group-jobs-r4.jsonl","--profiles",study/"group-profiles-r4.jsonl",
                 "--qrels",label_file,"--text-vectors",study/"group-text-vectors-r4.json","--output",graph,"--allow-model-labels",
                 "--seeds","17,42,73","--epochs","30","--patience","5","--hidden","32","--learning-rate","0.003","--evaluate-test"],output/"graph.log")
    if not (graph/"summary.json").exists():raise ValueError("组图实验未完成，不能发布比较表")
    graph_manifest=check_stage(graph,"graph",model_inputs,text_vectors=study/"group-text-vectors-r4.json")
    seal_stage(graph,run_contract["contract_sha256"])
    stage("固定共同池比较")
    import lightgbm as lgb
    booster=lgb.Booster(model_file=str(output/"ranker/model.txt"))
    pair_rows={(r["query_id"],r["job_id"]):r for r in replay_rows}
    contexts={r["query_id"]:{k:v for k,v in r.items() if k!="ranking"} for r in baseline["bm25"]}
    expected_pools={key:{job for query,job in pair_rows if query==key} for key in contexts}
    def rankings(scores):
        output_rows=[]
        for query_id,context in contexts.items():
            ranked=[]
            for job_id,score in scores[query_id].items():
                if pair_rows[(query_id,job_id)]["eligible"]:ranked.append({"job_id":job_id,"score":float(score)})
            expected={job for q,job in pair_rows if q==query_id}
            if set(scores[query_id])!=expected:raise ValueError("模型结果与共同候选池不一致")
            output_rows.append({**context,"ranking":sorted(ranked,key=lambda row:(-row["score"],row["job_id"]))})
        return output_rows
    ranker_scores={}
    for query_id in contexts:
        selected=[r for r in replay_rows if r["query_id"]==query_id]
        matrix=np.array([[np.nan if r["features"][k] is None else r["features"][k] for k in FEATURE_NAMES] for r in selected])
        ranker_scores[query_id]=dict(zip([r["job_id"] for r in selected],booster.predict(matrix,num_threads=1)))
    baseline["ranker"]=rankings(ranker_scores)
    for model,file in [("qwen_frozen","baseline_metrics.json"),("qwen_lora","adapted_metrics.json")]:
        baseline[model]=rankings(scores_from_report(json.loads((lora/file).read_text())["test"],expected_pools))
    graph_result=json.loads((graph/"summary.json").read_text())
    baseline["explicit_logic"]=rankings(scores_from_report(graph_result["logic_baseline"]["test"],expected_pools))
    for variant in graph_result["variants"]:
        baseline[f"{variant['mode']}_seed{variant['seed']}"]=rankings(scores_from_report(variant["evaluation"]["test"],expected_pools))
    contract=json.loads((output/"evaluation/evaluation_contract.json").read_text())
    result=compare_models(baseline,read_jsonl(output/"evaluation/qrels.jsonl"),contract,allow_model_labels=True)
    result["scope"]=f"同一{len(contexts)}个合成测试画像、同一共同池、相同硬过滤的重排；不是全库召回或真人验证。"
    result["run_contract_sha256"]=run_contract["contract_sha256"]
    result["representation_provenance"]={
        "source_context_meaning":"共同query_sha/document_view为原文视图摘要；各模型内部表示不同，不冒充同一编码器。",
        "ranker":{"feature_names":list(FEATURE_NAMES),"manifest_sha256":sha256(output/"ranker/manifest.json")},
        "qwen":{"rendered_query_hash":lora_manifest["rendered_query_hash"],"rendered_query_views_hash":lora_manifest["rendered_query_views_hash"],
            "rendered_document_hash":lora_manifest["rendered_document_hash"],"encoder":lora_manifest["encoder_contract"],
            "adapted_encoder":lora_manifest["reload_verification"]["inference_encoder_contract"]},
        "graph":{"frozen_node_contract":group_contract,"manifest_sha256":sha256(graph/"manifest.json")}}
    result["source_files"]={"teacher_labels":sha256(label_file),"replay":sha256(replay/"all_replay.jsonl"),"evaluation_contract":sha256(output/"evaluation/evaluation_contract.json")}
    result["annotation_snapshot_files"]=annotation_hashes
    for name,rows in baseline.items():write_jsonl(output/f"ranking-{name}.jsonl",rows)
    write_json(output/"metrics.json",result)
    from research.teacher_dashboard import publish_teacher_dashboard
    runs=[annotation_dir]
    if (study/"extraction-annotation/manifest.json").exists():runs.append(study/"extraction-annotation")
    publish_teacher_dashboard(study,runs,output/"metrics.json",lora/"manifest.json",graph/"summary.json")
    stage("completed",models=len(baseline),queries=len(contexts))
    return {key:value["summary"] for key,value in result["models"].items()}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["study","dataset","replay","output","training-python"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu",default="2");parser.add_argument("--wait-seconds",type=int,default=0)
    parser.add_argument("--annotation-dir",type=Path,help="完整且来源已核验的原始或派生标注版本；默认study/annotation")
    args=parser.parse_args()
    try:print(json.dumps(run(args.study,args.dataset,args.replay,args.output,args.training_python,args.gpu,args.wait_seconds,args.annotation_dir),ensure_ascii=False))
    except Exception as error:
        if args.output.exists():write_json(args.output/"failure.json",{"error_type":type(error).__name__,"message":str(error),"eligible_for_production":False})
        raise
