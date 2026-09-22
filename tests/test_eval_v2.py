"""全部材料为虚构CPU测试；不代表真人审核，不调用服务或启动模型训练。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import research
research.__path__.insert(0, str(Path(__file__).resolve().parents[1]/"research"))

from research.metrics import ranking_metrics, paired_bootstrap
from research.score_rankings import score, compare_models
from research.review_contracts import build_evaluation_contract, digest, file_sha, validate_qrel
from research.annotation_workflow import (seal_task, review_status, adjudicate, freeze_dataset, freeze_model_dataset,
    import_profiles, sample_pilot)
from research.prepare_ranker import materialize, query_payload_sha256
from research.train_ranker import validate_rows
from job_agent.ranking import FEATURE_NAMES, FEATURE_VERSION


def profile(query="q", family="pf", split="test", purpose="evaluation"):
    return {"query_id": query, "profile_family_id": family, "split": split, "purpose": purpose,
            "text": query+"虚构简历：使用Python完成练习。", "preferences": {}, "scenario_group": "recommendation_case"}


def human_qrel(job, grade, query="q"):
    return {"query_id": query, "job_id": job, "grade": grade, "label_source": "human_adjudicated",
            "adjudication_id": "fixture-only-"+job, "reviewer_ids": ["fixture-only-a", "fixture-only-b"],
            "reviewer_person_ids": ["fixture-person-a", "fixture-person-b"], "review_ids": ["fixture-review-a", "fixture-review-b"],
            "review_hashes": {"fixture-review-a": "a"*64, "fixture-review-b": "b"*64},
            "task_hash": "t"*64, "adjudicator_id": "fixture-only-a", "adjudication_sha256": "c"*64, "registry_sha256": "d"*64}


def model_qrel(job="a", grade=2, query="q"):
    audit = {"model": "fixture-model", "model_revision": "provider-not-disclosed", "prompt_hash": "f"*64,
             "input_hash": "i"*64, "request_id": "fixture-batch-request", "grade": grade}
    return {"query_id": query, "job_id": job, "grade": grade, "label_source": "llm_reviewed", "input_hash": "i"*64,
        "model_reviews": [{**audit, "channel": "first"}, {**audit, "channel": "second", "prompt_hash": "e"*64}],
        "rule_validation": {"passed": True, "validator_version": "fixture-rules", "input_hash": "i"*64, "checks": {"schema": True}}}


def human_fixture():
    task = seal_task({"task_id": "t", "kind": "relevance", "split": "test", "query_id": "q", "job_id": "a",
                      "profile_family_id": "pf", "job_family_id": "jf", "source_query_sha256": digest(profile())})
    registry = {"attestation": {"record_id": "fixture-attestation", "attested_by": "fixture-coordinator",
        "attested_at": "2026-09-22T00:00:00Z", "identity_check_performed": True, "independent_review_process_confirmed": True},
        "reviewers": [{"reviewer_id": alias, "person_id": person, "identity_verified": True} for alias, person in [("fixture-a", "person-a"), ("fixture-b", "person-b")]]}
    reviews = [{"review_id": alias+"-review", "reviewer_id": alias, "task_id": "t", "task_hash": task["task_hash"],
        "source": "human_entered_unadjudicated", "independent": True, "created_at": "2026-09-22T01:00:00Z", "grade": 2}
        for alias in ["fixture-a", "fixture-b"]]
    decision = {"task_id": "t", "task_hash": task["task_hash"], "source": "explicit_human_adjudication",
        "decision": "approved", "human_confirmed": True, "review_hashes": {row["review_id"]: digest(row) for row in reviews},
        "adjudicator_id": "fixture-a", "adjudication_id": "fixture-decision", "created_at": "2026-09-22T02:00:00Z", "rationale": "仅用于测试的虚构仲裁", "grade": 2}
    return [task], reviews, registry, [decision]


def test_common_pool_fixes_actual_denominator_counterexample():
    qrels = [human_qrel("z", 0), human_qrel("a", 3), human_qrel("b", 2)]
    pools = {"q": ["z", "a", "b"]}
    contract = build_evaluation_contract([profile()], pools, qrels)
    models = {"A": [{"query_id": "q", "ranking": [{"job_id": "a"}]}], "B": [{"query_id": "q", "ranking": [{"job_id": "b"}]}]}
    result = compare_models(models, qrels, contract)
    assert result["models"]["A"]["summary"]["ndcg_at_k"] == pytest.approx(.78715460299)
    assert result["models"]["B"]["summary"]["ndcg_at_k"] == pytest.approx(.33735197271)
    assert result["models"]["A"]["summary"]["pool_recall_at_k"] == .5
    assert result["models"]["A"]["common_pool_sha256"] == result["models"]["B"]["common_pool_sha256"]
    assert result["comparisons"][0]["ndcg_delta"]["ci95"] is None  # 一家族不能伪造区间。


def test_missing_query_outside_pool_and_changed_qrels_refuse_comparison():
    qrels = [human_qrel("a", 3)]
    contract = build_evaluation_contract([profile()], {"q": ["a"]}, qrels)
    ranked = [{"query_id": "q", "ranking": [{"job_id": "a"}]}]
    with pytest.raises(ValueError, match="缺查询"):
        compare_models({"A": [], "B": []}, qrels, contract)
    with pytest.raises(ValueError, match="共同池外"):
        score([{"query_id": "q", "ranking": [{"job_id": "b"}]}], qrels, {"q": ["a"]})
    changed = deepcopy(qrels); changed[0]["grade"] = 0
    with pytest.raises(ValueError, match="qrels已变更"):
        compare_models({"A": ranked, "B": ranked}, changed, contract)
    partial = score(ranked, qrels, {"q": ["a", "unreviewed"]})
    assert partial["summary"]["ndcg_at_k"] is None
    with pytest.raises(ValueError, match="尚未全部"):
        score(ranked, qrels, {"q": ["a", "unreviewed"]}, strict=True)


def test_family_bootstrap_resamples_entire_correlated_families():
    result = paired_bootstrap([0, 0, 0, 0], [1, 1, 1, -1], clusters=["same", "same", "same", "other"])
    assert result["clusters"] == 2 and result["queries"] == 4
    assert result["mean_delta"] == .5
    assert result["ci95"] == [-1., 1.]
    assert result["resampling_unit"] == "profile_family"
    with pytest.raises(ValueError, match="profile_family"):
        paired_bootstrap([0], [1], clusters=[""])


def test_agreement_never_auto_gold_and_human_decisions_bind_all_reviews(tmp_path):
    tasks, reviews, registry, decisions = human_fixture()
    report = review_status(tasks, reviews, registry)
    assert report["counts"] == {"agreement_pending_adjudication": 1}
    assert report["human_gold_created"] is False
    assert adjudicate(tasks, reviews, registry, [])["gold"] == []
    with pytest.raises(ValueError, match="未完成"):
        freeze_dataset(tasks, [profile()], reviews, registry, [], tmp_path/"no-freeze")
    assert not (tmp_path/"no-freeze").exists()
    changed = deepcopy(reviews); changed[0]["grade"] = 1
    assert review_status(tasks, changed, registry)["counts"] == {"disagreement_pending_adjudication": 1}
    with pytest.raises(ValueError, match="绑定"):
        adjudicate(tasks, changed, registry, decisions)
    frozen = freeze_dataset(tasks, [profile()], reviews, registry, decisions, tmp_path/"frozen")
    assert frozen["gold_count"] == 1
    saved = json.loads((tmp_path/"frozen/evaluation_contract.json").read_text())
    assert saved["label_sources"] == ["human_adjudicated"]


def test_same_person_aliases_and_changed_task_cannot_pass_human_gate():
    tasks, reviews, registry, decisions = human_fixture()
    registry["reviewers"][1]["person_id"] = registry["reviewers"][0]["person_id"]
    with pytest.raises(ValueError, match="同一人"):
        review_status(tasks, reviews, registry)
    tasks[0]["job_id"] = "changed"
    with pytest.raises(ValueError, match="任务内容hash"):
        review_status(tasks, reviews, registry)


def test_teacher_gates_keep_source_explicit_and_allow_batch_request_reuse():
    first, second = model_qrel("a"), model_qrel("b")
    with pytest.raises(ValueError, match="默认禁止"):
        validate_qrel(first)
    validate_qrel(first, True); validate_qrel(second, True)
    disagree = deepcopy(first); disagree["model_reviews"][1]["grade"] = 0
    with pytest.raises(ValueError, match="模型评审缺"):
        validate_qrel(disagree, True)
    disagree["model_adjudication"] = {**first["model_reviews"][0], "rationale": "虚构分歧裁定", "request_id": "fixture-adjudicate"}
    validate_qrel(disagree, True)
    contract = build_evaluation_contract([profile()], {"q": ["a", "b"]}, [first, second], allow_model_labels=True)
    ranked = [{"query_id": "q", "ranking": [{"job_id": "a"}, {"job_id": "b"}]}]
    with pytest.raises(ValueError, match="默认禁止"):
        compare_models({"A": ranked, "B": ranked}, [first, second], contract)
    result = compare_models({"A": ranked, "B": ranked}, [first, second], contract, True)
    assert result["teacher_relative"] is True and result["eligible_for_production"] is False


def test_model_freeze_refuses_missing_task_and_never_calls_it_human(tmp_path):
    tasks, _, _, _ = human_fixture()
    label = {**model_qrel(), "task_id": "t", "task_hash": tasks[0]["task_hash"],
             "profile_family_id": "pf", "job_family_id": "jf", "split": "test"}
    with pytest.raises(ValueError, match="全部共同池"):
        freeze_model_dataset(tasks, [profile()], [], tmp_path/"missing")
    freeze_model_dataset(tasks, [profile()], [label], tmp_path/"weak")
    manifest = json.loads((tmp_path/"weak/manifest.json").read_text())
    assert manifest["human_kappa"] is None and manifest["human_review_completed"] is False
    assert manifest["label_source"] == "llm_reviewed"


def test_import_profiles_groups_variants_and_rejects_leaking_exclusions():
    rows = []
    for index, family in enumerate(["f1", "f1", "f2", "f3", "f4"]):
        row = profile("q"+str(index), family)
        row.update(source_kind="independent_fiction", source_author_id="fixture-author", source_record_id="fixture-"+str(index), authorship_attested_by="fixture-attestor")
        rows.append(row)
    imported = import_profiles(rows)
    assert len({row["split"] for row in imported if row["profile_family_id"] == "f1"}) == 1
    with pytest.raises(ValueError, match="重合"):
        import_profiles(rows, exclude=[rows[0]])
    rows[0].pop("profile_family_id")
    with pytest.raises(ValueError, match="profile_family"):
        import_profiles(rows)


def test_pilot_exact_120_300_no_test_and_clarification_separate():
    profiles = [profile("q"+str(i), "pf"+str(i), "dev") for i in range(4)]
    profiles.append({**profile("stress", "stress-family", "dev"), "scenario_group": "clarification_stress"})
    jobs, tasks = [], []
    for index in range(120):
        job = {"job_id": "e"+str(index), "job_family_id": "ef"+str(index), "split": "train", "description": "Python", "category": "技术", "snapshot": "fixture"}
        jobs.append(job)
        tasks.append({"task_id": "extract"+str(index), "task_type": "requirement_extraction", "job_id": job["job_id"], "evidence": {"field": "description", "start": 0, "end": 6, "quote": "Python"}})
    for index in range(300):
        job = {"job_id": "r"+str(index), "job_family_id": "rf"+str(index), "split": "dev", "category": "技术", "snapshot": "fixture"}
        jobs.append(job)
        tasks.append({"task_id": "rel"+str(index), "kind": "relevance", "query_id": "q"+str(index%4), "job_id": job["job_id"], "pool_sources": ["不可下发给盲审者"]})
    result = sample_pilot(tasks, profiles, jobs)
    assert sum(row["kind"] == "requirement_extraction" for row in result) == 120
    assert sum(row["kind"] == "relevance" for row in result) == 300
    assert sum(row["kind"] == "clarification" for row in result) == 1
    assert all(row["label"] is None and row["split"] != "test" and "pool_sources" not in row for row in result)
    with pytest.raises(ValueError, match="不足"):
        sample_pilot(tasks[:-1], profiles, jobs)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))


def materialization_fixture(tmp_path):
    dataset = tmp_path/"dataset"; dataset.mkdir()
    profiles = [profile("train-q", "train-pf", "train", "ranker_training"), profile("dev-q", "dev-pf", "dev", "ranker_training")]
    jobs, qrels, replay = [], [], []
    for query in profiles:
        for grade in [0, 3]:
            job = {"job_id": query["query_id"]+str(grade), "job_family_id": query["query_id"]+"family"+str(grade), "split": query["split"], "snapshot": "fixture-snapshot"}
            jobs.append(job); qrels.append(human_qrel(job["job_id"], grade, query["query_id"]))
            replay.append({**job, "query_id": query["query_id"], "profile_family_id": query["profile_family_id"], "feature_version": FEATURE_VERSION,
                "features": {key: None for key in FEATURE_NAMES}, "retrieval_contract_hash": "r"*64, "encoder_contract": {"model": "fixture-encoder"},
                "query_payload_sha256": query_payload_sha256(query)})
    paths = {name: tmp_path/(name+".jsonl") for name in ("profiles", "qrels", "replay", "evaluation")}
    write_rows(dataset/"jobs.jsonl", jobs)
    (dataset/"manifest.json").write_text(json.dumps({"files": {"jobs.jsonl": {"sha256": file_sha(dataset/"jobs.jsonl")}}}))
    for name, rows in [("profiles", profiles), ("qrels", qrels), ("replay", replay), ("evaluation", [profile("independent", "independent-family")])]:
        write_rows(paths[name], rows)
    manifest = tmp_path/"replay.manifest.json"
    contract = {"schema": "job-agent-ranker-replay-v2", "replay_sha256": file_sha(paths["replay"]), "jobs_sha256": file_sha(dataset/"jobs.jsonl"),
        "profiles_sha256": file_sha(paths["profiles"]), "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
        "retrieval_contract_hash": "r"*64, "encoder_contract": {"model": "fixture-encoder"}}
    manifest.write_text(json.dumps(contract))
    return dataset, paths, manifest, replay


def test_ranker_materialization_uses_exact_replay_and_missing_family_refuses(tmp_path):
    dataset, paths, manifest, replay = materialization_fixture(tmp_path)
    rows = materialize(dataset, paths["profiles"], paths["qrels"], paths["replay"], manifest, evaluation_profiles=paths["evaluation"])
    assert len(rows) == 4 and rows[0]["features"] == replay[0]["features"]
    assert rows[0]["retrieval_contract_hash"] == "r"*64 and len(rows[0]["features"]) == len(FEATURE_NAMES)
    with pytest.raises(ValueError, match="固定retrieval"):
        materialize(dataset, paths["profiles"], paths["qrels"])
    altered = deepcopy(rows); altered[0].pop("profile_family_id")
    with pytest.raises(ValueError, match="profile_family"):
        validate_rows(altered)
    replay[0]["query_payload_sha256"] = "tampered"
    write_rows(paths["replay"], replay)
    contract = json.loads(manifest.read_text()); contract["replay_sha256"] = file_sha(paths["replay"]); manifest.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="原文输入"):
        materialize(dataset, paths["profiles"], paths["qrels"], paths["replay"], manifest, evaluation_profiles=paths["evaluation"])


def test_teacher_abstention_needs_explicit_determinate_adjudication():
    label = model_qrel()
    label["model_reviews"][0]["grade"] = None
    with pytest.raises(ValueError, match="模型评审缺"):
        validate_qrel(label, True)
    label["model_adjudication"] = {**label["model_reviews"][1], "request_id": "fixture-adjudication", "rationale": "重新核对材料后给出确定弱标签"}
    validate_qrel(label, True)
    label["model_adjudication"]["grade"] = None
    with pytest.raises(ValueError, match="最终裁定"):
        validate_qrel(label, True)


def test_direct_human_qrel_cannot_skip_two_people_or_bound_reviews():
    label = human_qrel("a", 2)
    label["reviewer_person_ids"] = ["same-person", "same-person"]
    with pytest.raises(ValueError, match="两位"):
        validate_qrel(label)
    label = human_qrel("a", 2)
    label["review_hashes"].pop("fixture-review-a")
    with pytest.raises(ValueError, match="全部原始"):
        validate_qrel(label)
    label.pop("registry_sha256")
    with pytest.raises(ValueError, match="身份登记"):
        validate_qrel(label)


def test_freeze_binds_actual_resume_and_preferences_not_only_query_id():
    qrels = [human_qrel("a", 2)]
    query = profile()
    contract = build_evaluation_contract([query], {"q": ["a"]}, qrels)
    changed = deepcopy(query); changed["text"] += "另一条实际经历"
    changed_contract = build_evaluation_contract([changed], {"q": ["a"]}, qrels)
    assert contract["query_set_sha256"] != changed_contract["query_set_sha256"]
    ranked = [{"query_id": "q", "ranking": [{"job_id": "a"}], "query_payload_sha256": query_payload_sha256(changed)}]
    with pytest.raises(ValueError, match="冻结text"):
        compare_models({"A": ranked, "B": ranked}, qrels, contract)
    with pytest.raises(ValueError, match="K必须"):
        build_evaluation_contract([query], {"q": ["a"]}, qrels, k=0)


def test_nested_requirement_gold_retains_and_or_and_checks_every_span():
    from research.annotation_workflow import canonical_label
    text = "Python或Java，并且SQL"
    task = {"kind": "requirement_extraction", "evidence": {"field": "description", "start": 0, "end": len(text), "quote": text}}
    def leaf(skill):
        return {"skill": skill, "field": "description", "quote": skill, "start": text.index(skill), "end": text.index(skill)+len(skill)}
    tree = {"logic": "all", "modality": "required", "skills": [leaf("SQL")], "children": [
        {"logic": "any", "modality": "required", "skills": [leaf("Python"), leaf("Java")]}]}
    label = canonical_label(task, {"extraction": {"groups": [tree]}})
    assert label["extraction"]["groups"][0]["children"][0]["logic"] == "any"
    tree["children"][0]["skills"][0]["quote"] = "伪造跨度"
    with pytest.raises(ValueError, match="原文不一致"):
        canonical_label(task, {"extraction": {"groups": [tree]}})


def test_ranker_rejects_nonstandard_missing_or_text_features(tmp_path):
    dataset, paths, manifest, replay = materialization_fixture(tmp_path)
    rows = materialize(dataset, paths["profiles"], paths["qrels"], paths["replay"], manifest, evaluation_profiles=paths["evaluation"])
    for invalid in [float("nan"), float("inf"), True, "0.5"]:
        changed = deepcopy(rows); changed[0]["features"][FEATURE_NAMES[0]] = invalid
        with pytest.raises(ValueError, match="有限数值"):
            validate_rows(changed)



def test_teacher_freeze_cannot_reuse_old_reviews_for_changed_resume(tmp_path):
    tasks, _, _, _ = human_fixture()
    label = {**model_qrel(), "task_id": "t", "task_hash": tasks[0]["task_hash"],
             "profile_family_id": "pf", "job_family_id": "jf", "split": "test"}
    changed = profile(); changed["text"] += "另一份未评审经历"
    with pytest.raises(ValueError, match="原文版本"):
        freeze_model_dataset(tasks, [changed], [label], tmp_path/"no-freeze")
    assert not (tmp_path/"no-freeze").exists()
