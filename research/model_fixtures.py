"""完全虚构的结构反例；所有标签显式标为synthetic_fixture，绝不冒充人工审阅。"""
from pathlib import Path
import json


def leaf(key, kind="skill", modality="required"):
    return {"type": kind, "key": key, "text": key, "required": modality == "required", "modality": modality}


def group(op, *children, **extra):
    return {"op": op, "children": list(children), "parse_status": "known", "modality": "required", **extra}


def counterexample_asts():
    # 叶子完全相同，OR组作用域不同。
    return (group("all", group("any", leaf("Python"), leaf("Java")), group("any", leaf("SQL"), leaf("Docker"))),
            group("all", group("any", leaf("Python"), leaf("SQL")), group("any", leaf("Java"), leaf("Docker"))))


def make_fixtures():
    jobs, profiles, qrels = [], [], []
    asts = counterexample_asts()
    phrases = {"train": "项目实践记录", "dev": "独立交付经历", "test": "个人作品说明"}
    for split in ("train", "dev", "test"):
        for index, ast in enumerate(asts):
            jobs.append({"job_id": f"{split}-job-{index}", "job_family_id": f"{split}-family-{index}",
                         "split": split, "snapshot": "entirely-fictional-v2", "job_version_id": "fixture-only",
                         "title": "开发岗位", "requirements": f"仅用于{split}分区结构验证的虚构岗位{index}。",
                         "description": "这些材料不是任何真实招聘信息。", "requirement_ast": ast})
        for index, skills in enumerate((["Python", "Java"], ["Python", "SQL"])):
            query_id = f"{split}-query-{index}"
            profiles.append({"query_id": query_id, "profile_family_id": f"{split}-profile-family-{index}",
                             "split": split, "text": f"{phrases[split]}：本人使用{'和'.join(skills)}完成练习。",
                             "skills": skills, "tasks": [], "preferences": {"intent": "开发岗位"}})
            positive = 1 if index == 0 else 0
            for job_index in range(2):
                qrels.append({"query_id": query_id, "job_id": f"{split}-job-{job_index}", "split": split,
                              "grade": 3 if job_index == positive else 0,
                              "hard_negative": job_index != positive, "label_source": "synthetic_fixture",
                              "note": "为逻辑结构反例手工设定的虚构预期，不是人工人岗标签"})
    return jobs, profiles, qrels


def write_fixtures(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("fixture目录已存在，拒绝覆盖")
    directory.mkdir(parents=True)
    for name, rows in zip(("jobs", "profiles", "qrels"), make_fixtures()):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    write_fixtures(parser.parse_args().output)
