"""独立画像导入、盲审试标、人工仲裁与冻结；一致票不会自动成为金标。"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random

from research.metrics import weighted_kappa
from research.review_contracts import (build_evaluation_contract, digest, file_sha, identifier, new_private_directory,
    profile_family, read_jsonl, timestamp, unique_rows, validate_profiles, validate_qrel, write_json, write_jsonl)

KINDS = {"relevance", "requirement_extraction", "clarification"}
ACTIONS = {"recommend", "clarify", "no_match"}


def task_hash(task):
    return digest({key: value for key, value in task.items() if key != "task_hash"})


def seal_task(task):
    result = dict(task)
    result["task_hash"] = task_hash(result)
    return result


def validate_tasks(tasks):
    lookup = unique_rows(tasks, "task_id")
    for task in tasks:
        if task.get("kind") not in KINDS or task.get("split") not in {"train", "dev", "test"}:
            raise ValueError("任务类型或分区不合法")
        if task.get("task_hash") != task_hash(task):
            raise ValueError("任务内容hash失配；旧材料须重新导出v2合同，不接受伪造沿用旧hash")
        if task["kind"] != "clarification" and not identifier(task.get("job_id")):
            raise ValueError("岗位任务缺job_id")
        if task["kind"] != "requirement_extraction":
            if not identifier(task.get("query_id")):
                raise ValueError("画像任务缺query_id")
            profile_family(task)
    return lookup


def validate_query_binding(task, query):
    """任务所评正文必须对应被冻结画像；ID相同不能掩盖材料换版。"""
    source_hash = task.get("source_query_sha256")
    material = task.get("material")
    if source_hash is not None and source_hash != digest(query):
        raise ValueError("任务所评画像原文版本与待冻结profiles不同")
    if isinstance(material, dict):
        if task.get("input_hash") != digest(material):
            raise ValueError("任务材料input_hash失配")
        if material.get("resume") != query["text"] or material.get("preferences", {}) != query.get("preferences", {}):
            raise ValueError("接口所评简历/偏好与待冻结profiles不同")
    elif source_hash is None:
        raise ValueError("画像任务须绑定source_query_sha256或带input_hash的实际评审材料")


def import_profiles(rows, purpose="evaluation", seed=42, exclude=()):
    """按输入的真实来源家族分组；不从query_id猜家族，不把模板认作独立人。"""
    if purpose not in {"evaluation", "ranker_training"}:
        raise ValueError("用途必须为evaluation或ranker_training")
    unique_rows(rows, "query_id")
    excluded = list(exclude)
    excluded_families = {profile_family(row) for row in excluded}
    excluded_texts = {digest(" ".join(row["text"].split())) for row in excluded}
    excluded_ids = {row["query_id"] for row in excluded}
    result = []
    for original in rows:
        row = dict(original); family = profile_family(row)
        if row.get("source_kind") not in {"authorized_real", "independent_fiction"}:
            raise ValueError("只导入授权脱敏或独立撰写画像；已有未审程序模板不能冒充独立来源")
        if not all(identifier(row.get(key)) for key in ("source_author_id", "source_record_id", "authorship_attested_by")):
            raise ValueError("画像缺作者、来源记录或人工来源确认记录")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("画像正文为空")
        if family in excluded_families or row["query_id"] in excluded_ids or digest(" ".join(row["text"].split())) in excluded_texts:
            raise ValueError("画像与排除的独立训练/评测集重合")
        row.update(profile_family_id=family, purpose=purpose, label=None, is_human_gold=False)
        row.pop("split", None)
        result.append(row)
    families = sorted({profile_family(row) for row in result})
    if len(families) < 2:
        raise ValueError("至少两个独立画像家族才能分区")
    random.Random(seed).shuffle(families)
    count = max(1, min(len(families)-1, round(len(families)*(0.8 if purpose == "ranker_training" else 0.25))))
    first, second = ("train", "dev") if purpose == "ranker_training" else ("dev", "test")
    assignments = {family: first if index < count else second for index, family in enumerate(families)}
    for row in result:
        row["split"] = assignments[profile_family(row)]
    validate_profiles(result)
    return sorted(result, key=lambda row: row["query_id"])


def balanced_sample(rows, count, seed, keys):
    if len(rows) < count:
        raise ValueError(f"符合试标隔离条件的任务仅{len(rows)}，不足请求的{count}；不借用冻结测试集补齐")
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "未知")) for key in keys)].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        values.sort(key=lambda row: row["task_id"]); rng.shuffle(values)
    order = sorted(buckets); rng.shuffle(order)
    result = []
    while len(result) < count:
        for key in order:
            if buckets[key] and len(result) < count:
                result.append(buckets[key].pop())
    return result


def sample_pilot(tasks, profiles, jobs, extraction_count=120, relevance_count=300, seed=42, splits=("train", "dev")):
    if not set(splits).issubset({"train", "dev"}):
        raise ValueError("试标/对齐rubric不能使用冻结test分区")
    queries = validate_profiles(profiles); catalog = unique_rows(jobs, "job_id")
    prepared = []
    for original in tasks:
        kind = original.get("kind", original.get("task_type"))
        if kind not in {"relevance", "requirement_extraction"}:
            continue
        job = catalog.get(original.get("job_id"))
        if job is None:
            raise ValueError("试标任务岗位不存在")
        if job["split"] not in splits:
            continue
        row = {key: value for key, value in original.items() if key not in {"task_hash", "pool_sources", "sources", "retrieval_sources", "rank", "score", "proposed_label", "labels", "label"}}
        row.update(kind=kind, split=job["split"], job_family_id=job["job_family_id"],
                   source_job_sha256=digest(job), source_snapshot=job.get("snapshot"), category=job.get("category"),
                   label=None, stage="pilot", is_human_gold=False)
        if kind == "relevance":
            query = queries.get(row.get("query_id"))
            if query is None:
                raise ValueError("试标相关性任务缺已导入独立画像；不能按ID补造家族")
            if query["split"] != row["split"]:
                raise ValueError("试标岗位与画像分区不同")
            if query.get("scenario_group") == "clarification_stress":
                continue
            row.update(profile_family_id=profile_family(query), source_query_sha256=digest(query))
        else:
            evidence = row.get("evidence", {})
            text = job.get(evidence.get("field"), "")
            start, end = evidence.get("start"), evidence.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != evidence.get("quote"):
                raise ValueError("抽取试标原文跨度不一致")
        prepared.append(seal_task(row))
    unique_rows(prepared, "task_id")
    extraction = balanced_sample([row for row in prepared if row["kind"] == "requirement_extraction"], extraction_count, seed, ("split", "category"))
    relevance = balanced_sample([row for row in prepared if row["kind"] == "relevance"], relevance_count, seed+1, ("profile_family_id",))
    actions = []
    for query in sorted(profiles, key=lambda row: row["query_id"]):
        if query["split"] in splits and query.get("scenario_group") == "clarification_stress":
            actions.append(seal_task({"task_id": "action-"+digest(query["query_id"])[:20], "kind": "clarification",
                "query_id": query["query_id"], "profile_family_id": profile_family(query), "split": query["split"],
                "source_query_sha256": digest(query), "stage": "pilot", "label": None, "is_human_gold": False,
                "note": "场景类型不是答案；由人判断推荐、追问或无匹配；无匹配需可核对候选依据。"}))
    return extraction+relevance+actions


def reviewer_registry(document):
    attestation = document.get("attestation", {})
    if not identifier(attestation.get("record_id")) or not identifier(attestation.get("attested_by")) or attestation.get("identity_check_performed") is not True or attestation.get("independent_review_process_confirmed") is not True:
        raise ValueError("评审身份与独立盲审需有人类确认记录；不同代号不等于不同人")
    timestamp(attestation.get("attested_at"))
    lookup = unique_rows(document.get("reviewers", []), "reviewer_id")
    for reviewer in lookup.values():
        if not identifier(reviewer.get("person_id")) or reviewer.get("identity_verified") is not True:
            raise ValueError("审核员缺已核实的匿名person_id")
    return lookup


def canonical_label(task, row):
    kind = task["kind"]
    if kind == "relevance":
        if type(row.get("grade")) is not int or row["grade"] not in range(4) or row.get("action") is not None or row.get("extraction") is not None:
            raise ValueError("相关性任务仅接受0至3等级；追问与抽取必须分开")
        return {"grade": row["grade"]}
    if row.get("grade") is not None:
        raise ValueError("非相关性任务不得填grade")
    if kind == "clarification":
        if row.get("action") not in ACTIONS or row.get("extraction") is not None:
            raise ValueError("动作任务仅接受recommend/clarify/no_match")
        if row["action"] == "no_match" and not identifier(row.get("no_match_evidence_ref")):
            raise ValueError("无匹配判断须提供候选审阅依据，不能把缺信息当无岗位")
        return {"action": row["action"], "no_match_evidence_ref": row.get("no_match_evidence_ref")}
    extraction = row.get("extraction")
    if not isinstance(extraction, dict) or not isinstance(extraction.get("groups"), list) or row.get("action") is not None:
        raise ValueError("抽取标签须有groups结构，不能只有正确/错误")
    evidence = task["evidence"]; used = set()
    def canonical_group(group):
        if not isinstance(group, dict) or group.get("logic") not in {"single", "any", "all", "unknown"} or group.get("modality") not in {"required", "preferred", "negated", "unknown"}:
            raise ValueError("抽取逻辑或模态非法")
        skills, children = group.get("skills", []), group.get("children", [])
        if not isinstance(skills, list) or not isinstance(children, list) or not skills and not children:
            raise ValueError("要求组不能为空；没有技能应提交空groups")
        checked = []
        for item in skills:
            if not isinstance(item, dict):
                raise ValueError("抽取技能必须为对象")
            start, end = item.get("start"), item.get("end")
            if not identifier(item.get("skill")) or type(start) is not int or type(end) is not int or item.get("field") != evidence["field"] or not evidence["start"] <= start < end <= evidence["end"]:
                raise ValueError("抽取技能跨度必须位于本段原文")
            if evidence["quote"][start-evidence["start"]:end-evidence["start"]] != item.get("quote"):
                raise ValueError("抽取引用与原文不一致")
            key = (item["skill"], start, end)
            if key in used:
                raise ValueError("同一技能跨度重复")
            used.add(key); checked.append({key: item[key] for key in ("skill", "quote", "field", "start", "end")})
        nested = [canonical_group(child) for child in children]
        operands = len(checked) + len(nested)
        if group["logic"] == "single" and operands != 1 or group["logic"] in {"any", "all"} and operands < 2:
            raise ValueError("逻辑运算与技能/子组数量不符")
        output = {"logic": group["logic"], "modality": group["modality"], "skills": sorted(checked, key=lambda item: (item["start"], item["skill"]))}
        if nested:
            output["children"] = sorted(nested, key=digest)
        return output
    return {"extraction": {"groups": sorted([canonical_group(group) for group in extraction["groups"]], key=digest), "source_evidence": evidence}}



def review_status(tasks, reviews, registry):
    task_map = validate_tasks(tasks); people = reviewer_registry(registry)
    unique_rows(reviews, "review_id")
    buckets = defaultdict(list); person_tasks = set()
    for review in reviews:
        task = task_map.get(review.get("task_id"))
        if task is None or review.get("task_hash") != task["task_hash"]:
            raise ValueError("评审对应任务不存在或材料版本已变化")
        if review.get("source", review.get("label_source")) != "human_entered_unadjudicated":
            raise ValueError("只接受原始人工评审，不接受模型标注或预先伪装的仲裁标签")
        reviewer = people.get(review.get("reviewer_id"))
        if reviewer is None or review.get("independent") is not True:
            raise ValueError("原始评审缺核实身份或独立盲审声明")
        timestamp(review.get("created_at"))
        key = (task["task_id"], reviewer["person_id"])
        if key in person_tasks:
            raise ValueError("同一人不能用多个代号或版本充当两份独立评审；请明确选择有效版本")
        person_tasks.add(key)
        label = canonical_label(task, review)
        buckets[task["task_id"]].append({"review_id": review["review_id"], "reviewer_id": review["reviewer_id"],
            "person_id": reviewer["person_id"], "review_sha256": digest(review), "label": label})
    output = []
    for task_id, task in task_map.items():
        values = sorted(buckets[task_id], key=lambda row: row["review_id"])
        status = "pending_independent_reviews" if len(values) < 2 else "disagreement_pending_adjudication" if len({digest(row["label"]) for row in values}) > 1 else "agreement_pending_adjudication"
        output.append({"task_id": task_id, "kind": task["kind"], "task_hash": task["task_hash"], "status": status, "reviews": values})
    pairs = defaultdict(lambda: ([], []))
    for row in output:
        if row["kind"] != "relevance":
            continue
        values = sorted(row["reviews"], key=lambda review: review["reviewer_id"])
        for i, a in enumerate(values):
            for b in values[i+1:]:
                first, second = pairs[(a["reviewer_id"], b["reviewer_id"])]
                first.append(a["label"]["grade"]); second.append(b["label"]["grade"])
    return {"tasks": output, "counts": dict(Counter(row["status"] for row in output)), "human_gold_created": False,
            "agreement": [{"reviewers": list(pair), "items": len(first), "raw_agreement": sum(a == b for a, b in zip(first, second))/len(first),
                           "quadratic_weighted_kappa": weighted_kappa(first, second)} for pair, (first, second) in pairs.items()],
            "notice": "仅验证身份确认记录与内容血缘，不认证真人身份；同意票也必须显式人工仲裁。"}


def adjudicate(tasks, reviews, registry, decisions):
    status = review_status(tasks, reviews, registry); task_map = {row["task_id"]: row for row in tasks}
    people = reviewer_registry(registry); decision_map = unique_rows(decisions, "task_id")
    if set(decision_map)-set(task_map):
        raise ValueError("仲裁引用了未知任务")
    gold, pending = [], []
    adjudication_ids = set()
    for item in status["tasks"]:
        decision = decision_map.get(item["task_id"])
        if decision is None or len(item["reviews"]) < 2:
            pending.append({"task_id": item["task_id"], "status": item["status"] if decision is None else "pending_independent_reviews"})
            continue
        task = task_map[item["task_id"]]
        if decision.get("source") != "explicit_human_adjudication" or decision.get("decision") != "approved" or decision.get("human_confirmed") is not True:
            raise ValueError("仲裁必须是有真人明确确认的approved决议")
        if decision.get("task_hash") != task["task_hash"] or decision.get("review_hashes") != {row["review_id"]: row["review_sha256"] for row in item["reviews"]}:
            raise ValueError("仲裁未绑定当前材料与全部独立评审原文")
        if decision.get("adjudicator_id") not in people or not identifier(decision.get("adjudication_id")) or decision["adjudication_id"] in adjudication_ids or not identifier(decision.get("rationale")):
            raise ValueError("仲裁缺人员、唯一记录ID或明确理由")
        timestamp(decision.get("created_at")); adjudication_ids.add(decision["adjudication_id"])
        label = canonical_label(task, decision)
        gold.append({key: task[key] for key in ("task_id", "kind", "query_id", "job_id", "profile_family_id", "job_family_id", "split", "stage") if key in task} |
                    label | {"task_hash": task["task_hash"], "label_source": "human_adjudicated", "reviewer_ids": [row["reviewer_id"] for row in item["reviews"]],
                    "review_ids": [row["review_id"] for row in item["reviews"]],
                    "reviewer_person_ids": [row["person_id"] for row in item["reviews"]], "review_hashes": decision["review_hashes"],
                    "adjudication_id": decision["adjudication_id"], "adjudicator_id": decision["adjudicator_id"],
                    "adjudication_sha256": digest(decision), "registry_sha256": digest(registry)})
    return {"gold": gold, "pending": pending, "review_status": status}


def freeze_dataset(tasks, profiles, reviews, registry, decisions, output, source_hashes=None):
    result = adjudicate(tasks, reviews, registry, decisions)
    if result["pending"]:
        raise ValueError(f"尚有{len(result['pending'])}项未完成双人审核与显式仲裁；不冻结部分金标")
    queries = validate_profiles(profiles)
    for task in tasks:
        if task["kind"] != "requirement_extraction":
            query = queries.get(task["query_id"])
            if query is None:
                raise ValueError("任务引用未知画像")
            validate_query_binding(task, query)
    qrels = [row for row in result["gold"] if row["kind"] == "relevance"]
    pools = defaultdict(list)
    for row in qrels:
        query = queries.get(row["query_id"])
        if query is None or profile_family(query) != row["profile_family_id"] or query["split"] != row["split"]:
            raise ValueError("冻结标签与独立画像身份/分区不一致")
        if query.get("scenario_group") == "clarification_stress":
            raise ValueError("追问压力画像不得作为相关性qrels")
        pools[row["query_id"]].append(row["job_id"])
    selected_queries = [queries[key] for key in sorted(pools)]
    purpose = {row.get("purpose") for row in selected_queries}
    is_pilot = any(row.get("stage") == "pilot" for row in tasks)
    if len(purpose) > 1:
        raise ValueError("不能混合训练和评测用途冻结")
    # 全部验证先完成；输出只使用新目录。
    if qrels and purpose != {"ranker_training"}:
        build_evaluation_contract(selected_queries, pools, qrels)
    target = new_private_directory(output)
    payloads = {"tasks.jsonl": tasks, "queries.jsonl": selected_queries, "qrels.jsonl": qrels,
                "extraction_gold.jsonl": [row for row in result["gold"] if row["kind"] == "requirement_extraction"],
                "action_gold.jsonl": [row for row in result["gold"] if row["kind"] == "clarification"],
                "all_profiles.jsonl": profiles, "raw_reviews.jsonl": reviews, "human_decisions.jsonl": decisions}
    for name, values in payloads.items():
        write_jsonl(target/name, values)
    write_json(target/"reviewer_registry.json", registry)
    hashes = {path.name: file_sha(path) for path in target.iterdir()}
    if qrels and purpose != {"ranker_training"}:
        contract = build_evaluation_contract(selected_queries, pools, qrels, hashes)
        contract["stage"] = "pilot_not_final_test" if is_pilot else "frozen_evaluation"
        contract["contract_sha256"] = digest({key: value for key, value in contract.items() if key != "contract_sha256"})
        write_json(target/"evaluation_contract.json", contract)
    write_json(target/"manifest.json", {"schema": "job-agent-human-freeze-v2", "files": {path.name: file_sha(path) for path in target.iterdir()},
        "upstream_files": source_hashes or {}, "purpose": sorted(purpose), "stage": "pilot" if is_pilot else "frozen",
        "counts": dict(Counter(row["kind"] for row in result["gold"])), "unresolved": 0,
        "notice": "金标来源为外部提交的显式人审决议；工具只核验记录，不认证身份。不自动训练或上线。"})
    return {"output": str(target), "gold_count": len(result["gold"]), "stage": "pilot" if is_pilot else "frozen"}


def freeze_model_dataset(tasks, profiles, labels, output, source_hashes=None):
    """显式教师弱标冻结，只处理相关性；不会产生human_adjudicated或人工κ。"""
    task_map = validate_tasks(tasks); queries = validate_profiles(profiles)
    if not tasks or any(task["kind"] != "relevance" for task in tasks):
        raise ValueError("freeze-model仅接收相关性任务；抽取和追问请保持独立任务材料，不能转成qrels")
    by_task = unique_rows(labels, "task_id")
    if set(by_task) != set(task_map):
        raise ValueError("教师冻结需要全部共同池任务完成，不能丢弃分歧或失败任务")
    qrels = []; pools = defaultdict(list)
    for task_id, task in task_map.items():
        row = by_task[task_id]
        if row.get("label_source") != "llm_reviewed":
            raise ValueError("教师冻结不能混入或改名human标签")
        validate_qrel(row, allow_model_labels=True)
        if row.get("task_hash") != task["task_hash"]:
            raise ValueError("教师弱标与冻结任务hash不一致")
        query = queries.get(task["query_id"])
        if query is None or query["split"] != task["split"] or profile_family(query) != profile_family(task):
            raise ValueError("教师任务与画像家族/分区不一致")
        validate_query_binding(task, query)
        if query.get("scenario_group") == "clarification_stress":
            raise ValueError("追问压力画像不能混入相关性弱标")
        for key in ("query_id", "job_id", "split", "profile_family_id", "job_family_id"):
            if row.get(key) != task.get(key):
                raise ValueError("教师标签身份与任务不一致")
        qrels.append(row); pools[row["query_id"]].append(row["job_id"])
    selected = [queries[key] for key in sorted(pools)]
    purposes = {query.get("purpose") for query in selected}
    if len(purposes) != 1:
        raise ValueError("冻结弱标用途不一致")
    evaluation = purposes != {"ranker_training"}
    if evaluation:
        build_evaluation_contract(selected, pools, qrels, allow_model_labels=True)
    target = new_private_directory(output)
    for name, rows in {"tasks.jsonl": tasks, "queries.jsonl": selected, "qrels.jsonl": qrels}.items():
        write_jsonl(target/name, rows)
    hashes = {path.name: file_sha(path) for path in target.iterdir()}
    if evaluation:
        contract = build_evaluation_contract(selected, pools, qrels, hashes, allow_model_labels=True)
        contract["stage"] = "teacher_relative_research"
        contract["contract_sha256"] = digest({key: value for key, value in contract.items() if key != "contract_sha256"})
        write_json(target/"evaluation_contract.json", contract)
    write_json(target/"manifest.json", {"schema": "job-agent-model-freeze-v2", "files": {path.name: file_sha(path) for path in target.iterdir()},
        "upstream_files": source_hashes or {}, "label_source": "llm_reviewed", "purpose": sorted(purposes),
        "pairs": len(qrels), "two_channel_grade_agreement": sum(len({review["grade"] for review in row["model_reviews"]}) == 1 for row in qrels)/len(qrels),
        "human_kappa": None, "human_review_completed": False, "eligible_for_production": False,
        "notice": "双通道可能同模型、同偏差；一致率不是人工一致性或标签正确率。规则通过不证明人岗语义。"})
    return {"output": str(target), "pairs": len(qrels), "label_source": "llm_reviewed", "human_gold_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imported = sub.add_parser("import-profiles")
    imported.add_argument("--input", type=Path, required=True); imported.add_argument("--exclude", type=Path, nargs="*", default=[])
    imported.add_argument("--purpose", choices=("evaluation", "ranker_training"), required=True); imported.add_argument("--seed", type=int, default=42)
    imported.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--tasks", type=Path, nargs="+", required=True); pilot.add_argument("--jobs", type=Path, required=True)
    pilot.add_argument("--profiles", type=Path, required=True); pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--extraction-count", type=int, default=120); pilot.add_argument("--relevance-count", type=int, default=300)
    pilot.add_argument("--seed", type=int, default=42)
    for name in ("review-status", "freeze"):
        command = sub.add_parser(name)
        for field in ("tasks", "reviews", "reviewers", "output"):
            command.add_argument("--"+field, type=Path, required=True)
        if name == "freeze":
            command.add_argument("--decisions", type=Path, required=True); command.add_argument("--profiles", type=Path, required=True)
    weak = sub.add_parser("freeze-model")
    for field in ("tasks", "profiles", "labels", "output"):
        weak.add_argument("--"+field, type=Path, required=True)
    weak.add_argument("--allow-model-labels", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "import-profiles":
        rows = import_profiles(read_jsonl(args.input), args.purpose, args.seed, [row for path in args.exclude for row in read_jsonl(path)])
        target = new_private_directory(args.output); write_jsonl(target/"profiles.jsonl", rows)
        write_json(target/"manifest.json", {"schema": "job-agent-independent-profiles-v2", "input_sha256": file_sha(args.input),
            "profiles_sha256": file_sha(target/"profiles.jsonl"), "exclude_hashes": {str(path): file_sha(path) for path in args.exclude},
            "purpose": args.purpose, "seed": args.seed, "families": len({profile_family(row) for row in rows}),
            "split_counts": dict(Counter(row["split"] for row in rows)), "note": "来源声明由人提供，程序不认证作者身份或排除所有语义近重复。"})
    elif args.command == "pilot":
        if args.extraction_count < 1 or args.relevance_count < 1:
            raise ValueError("试标数量必须为正")
        rows = sample_pilot([row for path in args.tasks for row in read_jsonl(path)], read_jsonl(args.profiles), read_jsonl(args.jobs), args.extraction_count, args.relevance_count, args.seed)
        target = new_private_directory(args.output); write_jsonl(target/"tasks.jsonl", rows)
        write_json(target/"manifest.json", {"schema": "job-agent-pilot-v2", "task_sha256": file_sha(target/"tasks.jsonl"),
            "source_hashes": {str(path): file_sha(path) for path in [*args.tasks, args.jobs, args.profiles]},
            "counts": dict(Counter(row["kind"] for row in rows)), "seed": args.seed, "all_labels_null": True,
            "stage": "pilot_not_final_test", "test_used": False})
    elif args.command == "freeze-model":
        print(json.dumps(freeze_model_dataset(read_jsonl(args.tasks), read_jsonl(args.profiles), read_jsonl(args.labels), args.output,
            {str(path): file_sha(path) for path in [args.tasks, args.profiles, args.labels]}), ensure_ascii=False))
    else:
        tasks, reviews = read_jsonl(args.tasks), read_jsonl(args.reviews)
        registry = json.loads(args.reviewers.read_text("utf-8"))
        if args.command == "review-status":
            result = review_status(tasks, reviews, registry)
            target = new_private_directory(args.output); write_json(target/"review_status.json", result)
            write_jsonl(target/"pending_adjudication.jsonl", [{"task_id": row["task_id"], "task_hash": row["task_hash"],
                "status": row["status"], "review_hashes": {review["review_id"]: review["review_sha256"] for review in row["reviews"]},
                "decision": None, "human_confirmed": False} for row in result["tasks"]])
        else:
            result = freeze_dataset(tasks, read_jsonl(args.profiles), reviews, registry, read_jsonl(args.decisions), args.output,
                {str(path): file_sha(path) for path in [args.tasks, args.profiles, args.reviews, args.reviewers, args.decisions]})
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
