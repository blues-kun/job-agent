"""故障注入验证旧模型拒用、共同池完整性与来源保全；不发请求、不训练。"""
import copy
import importlib.util
import json
import contextlib
import io
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

import job_agent.retrieval_contract as retrieval

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("audited_contracts",HERE/"research/teacher_experiment_contracts.py")
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False))


class ContractTests(unittest.TestCase):
    def test_full_pipeline_wiring_and_identical_resume_without_training(self):
        """模拟完整三训练器产物；真实训练函数/子进程绝不执行。"""
        runner_spec=importlib.util.spec_from_file_location("audit_runner",HERE/"research/run_teacher_experiments.py")
        runner=importlib.util.module_from_spec(runner_spec)
        with patch.dict(sys.modules,{"research.teacher_experiment_contracts":guard}):runner_spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);study=root/"study";dataset=root/"dataset";replay=root/"replay";output=root/"output";annotation=root/"derived"
            for path in (study,dataset,replay,annotation):path.mkdir()
            profiles=[{"query_id":"q-"+split,"profile_family_id":"pf-"+split,"split":split,"text":"虚构经历"+split,
                "preferences":{"intent":"虚构测试"},"purpose":"evaluation" if split=="test" else "ranker_training"} for split in ("train","dev","test")]
            tasks=[];labels=[];replay_rows=[]
            for query in profiles:
                for index in range(2):
                    task={"task_id":query["query_id"]+str(index),"query_id":query["query_id"],"job_id":query["split"]+str(index),"split":query["split"]}
                    tasks.append(task);grade=3 if index==0 else 0
                    labels.append({**task,"label_source":"llm_reviewed","grade":grade,"input_hash":"b"*64,
                        "model_reviews":[{"channel":channel,"model":"fixture-model","model_revision":"fixture","prompt_hash":"a"*64,"input_hash":"b"*64,"request_id":channel,"grade":grade} for channel in ("a","b")],
                        "rule_validation":{"passed":True,"input_hash":"b"*64,"validator_version":"fixture","checks":{"fictional":True}}})
                    replay_rows.append({"query_id":task["query_id"],"job_id":task["job_id"],"split":task["split"],"eligible":index==0,
                        "features":{key:0.0 for key in runner.FEATURE_NAMES}})
            def jsonl(path,rows):path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
            jsonl(study/"profiles.jsonl",profiles);jsonl(study/"tasks.jsonl",tasks);jsonl(annotation/"labels.jsonl",labels)
            jsonl(study/"train_profiles.jsonl",profiles[:2]);jsonl(study/"evaluation_profiles.jsonl",profiles[2:])
            jsonl(study/"group-profiles-r4.jsonl",profiles);jsonl(study/"group-jobs-r4.jsonl",[{"job_id":t["job_id"]} for t in tasks])
            write(study/"group-text-vectors-r4.json",{"fixture_only":True})
            write(annotation/"manifest.json",{"status":"completed","fixture_only":True})
            jsonl(replay/"all_replay.jsonl",replay_rows)
            baseline={name:[{"query_id":"q-test","ranking":[{"job_id":"test0","score":1.0}]}] for name in ("bm25","bge","hybrid","rule")}
            observed=[]
            def freeze(task_rows,query_rows,label_rows,target,sources):
                observed.append("freeze-"+target.name);target.mkdir()
                jsonl(target/"qrels.jsonl",label_rows)
                if target.name=="evaluation":write(target/"evaluation_contract.json",{"fixture_only":True,"query_ids":["q-test"]})
                write(target/"manifest.json",{"upstream_files":sources,"files":{p.name:guard.sha256(p) for p in target.iterdir()}})
            training_rows=[{"fixture_only":True,"query_id":row["query_id"],"grade":row["grade"]} for row in labels if row["split"]!="test"]
            def materialize(*args,**kwargs):
                observed.append("materialize")
                self.assertEqual(args[:3],(dataset,study/"train_profiles.jsonl",output/"training/qrels.jsonl"))
                self.assertTrue(kwargs["allow_model_labels"])
                return training_rows
            def train(row_file,target,**kwargs):
                observed.append("ranker");self.assertEqual(kwargs["dataset"],dataset);target.mkdir()
                (target/"model.txt").write_text("mock-booster-no-training")
                write(target/"metrics.json",{"fixture_only":True})
                write(target/"manifest.json",{"input_sha256":guard.sha256(row_file),"model_sha256":guard.sha256(target/"model.txt")})
            report={"per_query":[{"query_id":"q-test","candidate_ids":["test0","test1"],"candidate_scores":[.7,.2]}]}
            def command(arguments,log):
                module=arguments[3];observed.append("lora" if module.endswith("embedding") else "graph")
                params={};index=4
                while index<len(arguments):
                    key=str(arguments[index]);index+=1
                    if index<len(arguments) and not str(arguments[index]).startswith("--"):
                        params[key]=str(arguments[index]);index+=1
                    else:params[key]=True
                self.assertEqual(params["--qrels"],str(annotation/"labels.jsonl"))
                self.assertEqual(params["--profiles"],str(study/"group-profiles-r4.jsonl"))
                target=Path(params["--output"]);target.mkdir()
                lora=module.endswith("embedding");defaults=guard.LORA_ARGUMENTS if lora else guard.GRAPH_ARGUMENTS
                parsed={key:(bool(params.get("--"+key.replace("_","-"),False)) if type(value) is bool else type(value)(params["--"+key.replace("_","-")])) for key,value in defaults.items()}
                manifest={"arguments":parsed,"input_hashes":{key:guard.sha256(params["--"+key]) for key in ("jobs","profiles","qrels")}}
                if lora:
                    target.joinpath("adapter").mkdir();target.joinpath("adapter/adapter_model.safetensors").write_text("mock-adapter")
                    write(target/"baseline_metrics.json",{"test":report});write(target/"adapted_metrics.json",{"test":report})
                    manifest.update(status="completed",fixture_only=False,model_weak_supervision=True,
                        retrieval_contract_file_sha256=guard.sha256(retrieval.__file__),retrieval_contract=retrieval.contract_info(),
                        reload_verification={"passed":True,"inference_encoder_contract":{"fixture":"adapted"}},
                        rendered_query_hash="fixture-query",rendered_query_views_hash="fixture-views",rendered_document_hash="fixture-doc",encoder_contract={"fixture":"frozen"})
                else:
                    manifest["text_vectors_sha256"]=guard.sha256(params["--text-vectors"])
                    variants=[{"mode":mode,"seed":seed,"evaluation":{"test":report}} for mode in ("pool","graph","random_graph") for seed in (17,42,73)]
                    write(target/"summary.json",{"status":"completed","fixture_only":False,"model_weak_supervision":True,"test_evaluated":True,
                        "logic_baseline":{"test":report},"variants":variants})
                write(target/"manifest.json",manifest)
            def compare(models,qrels,contract,**kwargs):
                observed.append("compare");self.assertEqual(len(models),17);self.assertTrue(kwargs["allow_model_labels"])
                for rankings in models.values():self.assertEqual([row["job_id"] for row in rankings[0]["ranking"]],["test0"])
                self.assertEqual(len(qrels),2)
                return {"models":{name:{"summary":{"fixture_only":True}} for name in models}}
            class Booster:
                def __init__(self,**kwargs):self.path=kwargs["model_file"]
                def predict(self,matrix,num_threads):
                    self_outer.assertEqual(matrix.shape,(2,len(runner.FEATURE_NAMES)))
                    return [.7,.2]
            self_outer=self
            lgb=types.ModuleType("lightgbm");lgb.Booster=Booster
            dashboard=types.ModuleType("research.teacher_dashboard")
            dashboard.publish_teacher_dashboard=lambda *args:observed.append("publish")
            runner.__file__=str(Path(retrieval.__file__).resolve().parents[1]/"research/run_teacher_experiments.py")
            with contextlib.ExitStack() as stack:
                for name,value in {"read_annotations":lambda *args:(labels,{"labels.jsonl":"fixture"}),
                    "check_replay":lambda *args:(replay_rows,copy.deepcopy(baseline),{}),"check_group_bridge":lambda *args:{"fixture":True},
                    "make_run_contract":lambda *args:{"contract_sha256":"fixture-run"},"freeze_model_dataset":freeze,
                    "materialize":materialize,"train":train,"command":command,"compare_models":compare}.items():
                    stack.enter_context(patch.object(runner,name,value))
                stack.enter_context(patch.dict(sys.modules,{"lightgbm":lgb,"research.teacher_dashboard":dashboard}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runner.run(study,dataset,replay,output,Path("fixture-python"),"3",annotation_dir=annotation)
                first=list(observed)
                runner.run(study,dataset,replay,output,Path("fixture-python"),"3",annotation_dir=annotation)
            self.assertEqual(first,["freeze-training","freeze-evaluation","materialize","ranker","lora","graph","compare","publish"])
            self.assertEqual(observed[len(first):],["materialize","compare","publish"])
            self.assertEqual(json.loads((output/"progress.json").read_text())["stage"],"completed")

    def test_derived_annotation_cannot_change_grade_or_origin(self):
        tasks=[{"task_id":"t"}];original={"task_id":"t","grade":2,"job_field":"wrong"}
        def reconcile(task,row):
            value=copy.deepcopy(row);value["job_field"]="description"
            return value,{}
        def validate(task,row):return {"passed":True}
        api=types.ModuleType("research.api_annotation")
        api.MODEL="fixture-model";api.REASONING_EFFORT="max";api.validate_label=validate
        api.row_usable=lambda task,row:row.get("grade") in range(4)
        reconciler=types.ModuleType("research.reconcile_annotations");reconciler.reconcile_label=reconcile
        derived=reconcile(tasks[0],original)[0]
        manifest={"schema":"job-agent-quote-field-reconciliation-v1","model":"fixture-model","reasoning_effort":"max",
            "source_files_sha256":{"labels.jsonl":"raw-hash"},"rule_validator_module_sha256":guard.sha256(__file__),
            "reconciler_module_sha256":guard.sha256(__file__)}
        snapshot=types.ModuleType("research.annotation_snapshot")
        def read_snapshot(path,task_rows,require_complete):
            self.assertTrue(require_complete)
            if path.name=="derived":return manifest,{"t":derived},{"labels.jsonl":"derived-hash"},{}
            return {},{"t":original},{"labels.jsonl":"raw-hash"},{}
        snapshot.read_snapshot=read_snapshot
        with patch.dict(sys.modules,{"research.api_annotation":api,"research.reconcile_annotations":reconciler,"research.annotation_snapshot":snapshot}):
            rows,_=guard.read_annotations(Path("study"),Path("derived"),tasks)
            self.assertEqual(rows[0]["grade"],2)
            derived["grade"]=3
            with self.assertRaisesRegex(ValueError,"确定性字段纠正"):
                guard.read_annotations(Path("study"),Path("derived"),tasks)
            derived["grade"]=2;manifest["source_files_sha256"]={"labels.jsonl":"other-run"}
            with self.assertRaisesRegex(ValueError,"原始批次或标签已变化"):
                guard.read_annotations(Path("study"),Path("derived"),tasks)

    def test_r4_changed_with_same_labels_rejects_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            contract={"inputs":{"jobs":"r2","qrels":"same-labels"},"contract_sha256":"old"}
            guard.lock_run(root,contract)
            guard.lock_run(root,contract)
            changed=copy.deepcopy(contract);changed["inputs"]["jobs"]="r4"
            with self.assertRaisesRegex(ValueError,"拒绝复用"):
                guard.lock_run(root,changed)

    def test_legacy_model_without_source_lock_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"group-graph").mkdir()
            with self.assertRaisesRegex(ValueError,"缺完整实验来源锁"):
                guard.lock_run(root,{"contract_sha256":"new"})

    def test_finished_model_or_report_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"model.txt").write_text("fixture-weights")
            guard.seal_stage(root,"fixture-run")
            guard.seal_stage(root,"fixture-run")
            (root/"model.txt").write_text("replaced-weights")
            with self.assertRaisesRegex(ValueError,"被替换"):
                guard.seal_stage(root,"fixture-run")

    def test_frozen_qrels_changed_even_with_same_upstream_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"qrels.jsonl").write_text("original-fixture")
            write(root/"manifest.json",{"upstream_files":{"labels":"unchanged"},"files":{"qrels.jsonl":guard.sha256(root/"qrels.jsonl")}})
            guard.check_frozen(root,{"labels":"unchanged"})
            (root/"qrels.jsonl").write_text("changed-fixture")
            with self.assertRaisesRegex(ValueError,"被修改"):
                guard.check_frozen(root,{"labels":"unchanged"})

    def test_output_scores_must_be_complete_unique_and_finite(self):
        valid={"per_query":[{"query_id":"q","candidate_ids":["a","b"],"candidate_scores":[.2,.1]}]}
        self.assertEqual(guard.scores_from_report(valid,{"q":{"a","b"}}),{"q":{"a":.2,"b":.1}})
        bad=copy.deepcopy(valid);bad["per_query"][0]["candidate_scores"].pop()
        with self.assertRaisesRegex(ValueError,"分数不齐"):
            guard.scores_from_report(bad,{"q":{"a","b"}})
        bad=copy.deepcopy(valid);bad["per_query"][0]["candidate_scores"][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"非有限"):
            guard.scores_from_report(bad,{"q":{"a","b"}})
        bad=copy.deepcopy(valid);bad["per_query"]*=2
        with self.assertRaisesRegex(ValueError,"缺查询或重复"):
            guard.scores_from_report(bad,{"q":{"a","b"}})

    def test_lora_serialized_contract_and_real_cli_fields_are_compatible(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs={name:root/(name+".jsonl") for name in ("jobs","profiles","qrels")}
            for path in inputs.values():path.write_text("fictional-only\n")
            target=root/"lora";target.mkdir();(target/"adapter").mkdir()
            for name in ("baseline_metrics.json","adapted_metrics.json","adapter/adapter_model.safetensors"):
                (target/name).write_text("fixture-artifact")
            manifest={"input_hashes":{name:guard.sha256(path) for name,path in inputs.items()},
                "arguments":guard.LORA_ARGUMENTS,"status":"completed","fixture_only":False,"model_weak_supervision":True,
                "retrieval_contract_file_sha256":guard.sha256(retrieval.__file__),"retrieval_contract":retrieval.contract_info(),
                "reload_verification":{"passed":True}}
            write(target/"manifest.json",manifest)
            guard.check_stage(target,"lora",inputs,retrieval_path=Path(retrieval.__file__))
            (inputs["profiles"]).write_text("r4-changed-fixture\n")
            with self.assertRaisesRegex(ValueError,"r4岗位、画像或标签"):
                guard.check_stage(target,"lora",inputs,retrieval_path=Path(retrieval.__file__))

    def test_graph_reuse_requires_same_vectors_and_all_three_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs={name:root/(name+".jsonl") for name in ("jobs","profiles","qrels")}
            for path in inputs.values():path.write_text("fictional-only\n")
            vectors=root/"vectors.json";vectors.write_text("fixture-vectors")
            graph=root/"graph";graph.mkdir()
            manifest={"input_hashes":{name:guard.sha256(path) for name,path in inputs.items()},"arguments":guard.GRAPH_ARGUMENTS,
                      "text_vectors_sha256":guard.sha256(vectors)}
            summary={"status":"completed","fixture_only":False,"model_weak_supervision":True,"test_evaluated":True,
                "variants":[{"mode":mode,"seed":seed} for mode in ("pool","graph","random_graph") for seed in (17,42,73)]}
            write(graph/"manifest.json",manifest);write(graph/"summary.json",summary)
            guard.check_stage(graph,"graph",inputs,text_vectors=vectors)
            summary["variants"].pop();write(graph/"summary.json",summary)
            with self.assertRaisesRegex(ValueError,"缺少预注册"):
                guard.check_stage(graph,"graph",inputs,text_vectors=vectors)
            vectors.write_text("changed-fixture-vectors")
            with self.assertRaisesRegex(ValueError,"旧冻结文本向量"):
                guard.check_stage(graph,"graph",inputs,text_vectors=vectors)


if __name__=="__main__":unittest.main()
