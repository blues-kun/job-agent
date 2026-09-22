"""画像字段来源、显式确认与事实摘要；不将推断写成用户确认。"""
import re
from .domain import EDUCATION, education, experience, parse_profile
from .retrieval_contract import stable_hash, experience_blocks

FIELD_LABELS = {"city":"目标城市", "intent":"求职方向", "education":"最高学历", "experience_years":"工作总年资",
                "salary_min":"最低月薪", "salary_max":"期望月薪上限", "district":"区域偏好"}


def extracted_fields(text):
    values = {"education": None, "experience_years": experience(text)}
    grade = education(text)
    if grade is not None:
        values["education"] = next((name for name, value in EDUCATION.items() if value == grade), None)
    for key, pattern in [("city",r"(?:目标城市|意向城市|求职城市)\s*[:：]\s*([^\s，,；;]+)"),
                         ("intent",r"(?:求职意向|目标职位|求职方向)\s*[:：]\s*([^\n，,；;]+)")]:
        match = re.search(pattern, text)
        values[key] = match.group(1).strip() if match else None
    return values


def profile_preview(text, preferences, origins=None):
    origins = origins or {}
    extracted = extracted_fields(text)
    fields, effective = [], dict(preferences)
    for key, label in FIELD_LABELS.items():
        value, inferred = preferences.get(key), extracted.get(key)
        source = origins.get(key, "user_input" if value not in (None, "") else "unknown")
        # 上传客户端把旧样例标为sample_stale；即使调用者忘记清空，也不采纳。
        if source == "sample_stale":
            value = None if key in {"experience_years", "salary_min", "salary_max"} else ""
        if value in (None, "") and inferred not in (None, ""):
            value, source = inferred, "text_extracted"
        effective[key] = value
        quote = ""
        if inferred is not None:
            for line in text.splitlines():
                if (key == "education" and str(inferred) in line) or (key == "experience_years" and "年" in line and "经验" in line) or (key in {"city", "intent"} and str(inferred) in line):
                    quote = line.strip(); break
        fields.append({"key":key,"label":label,"value":value,"source":source,"extracted":inferred,"quote":quote,
                       "conflict":value not in (None, "") and inferred not in (None, "") and value != inferred,
                       "status":"unknown" if value in (None, "") else "needs_confirmation"})
    profile = parse_profile(text, effective)
    return {"version":profile["version"],"preferences":effective,"fields":fields,
            "conflicts":[field["key"] for field in fields if field["conflict"]],
            "note":"请确认字段来源与冲突。技能表述来自简历，不代表已经核验实际能力。"}


def fact_fingerprint(text, preferences):
    """比较完整经历块多重集及确认画像，防止拆散任职标题/日期归属。"""
    profile = parse_profile(text, preferences)
    facts = {key:profile.get(key) for key in ["education","experience_years","city","intent","salary_min","salary_max","district"]}
    facts["skills"] = {key:item["level"] for key,item in profile["skills"].items()}
    facts["blocks"] = sorted(experience_blocks(text))
    return facts


def compare_versions(before, after, preferences):
    previous, current = parse_profile(before, preferences), parse_profile(after, preferences)
    old_blocks, new_blocks = set(experience_blocks(before)), set(experience_blocks(after))
    keys = ["education","experience_years","city","intent","salary_min","salary_max","district"]
    return {"before_version":previous["version"],"after_version":current["version"],
            "added_blocks":sorted(new_blocks-old_blocks),"removed_blocks":sorted(old_blocks-new_blocks),
            "changed_fields":{key:{"before":previous.get(key),"after":current.get(key)} for key in keys if previous.get(key)!=current.get(key)},
            "added_skill_statements":sorted(set(current["skills"])-set(previous["skills"])),
            "note":"新增表述需要用户确认；段落整理和能力提升分别记录。"}
