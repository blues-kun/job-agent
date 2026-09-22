"""证据约束的推荐、诊断与简历迭代状态机。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import time
import uuid
import os

import numpy as np

from .corpus import Corpus, tokens
from .domain import constraints, display_text, parse_profile, supported_matches, digest, skill_evidence
from .coach import Coach
from .retrieval_contract import render_query, query_segments, semantic_text, experience_blocks, contract_info
from .profiles import fact_fingerprint, compare_versions
from .ranking import feature_vector, FEATURE_NAMES, FEATURE_VERSION

SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills"


class Workflow:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.coach = Coach()
        self.ranker = None
        self.ranker_error = None
        if os.environ.get("JOB_AGENT_RANKER_DIR"):
            from .ranker_runtime import RankerRuntime
            try:
                self.ranker = RankerRuntime(Path(os.environ["JOB_AGENT_RANKER_DIR"]), corpus)
            except (ValueError, OSError, ImportError) as error:
                self.ranker_error = type(error).__name__

    @staticmethod
    def skill(name: str) -> dict:
        # 白名单路由；只在触发时读取目标技能正文，绝不执行材料中的命令。
        if name not in {"resume-diagnose", "gap-analysis", "resume-rewrite", "mock-interview"}:
            raise ValueError("未知技能")
        content = (SKILL_ROOT / name / "SKILL.md").read_text("utf-8")
        return {"name": name, "version": digest(content)[:12], "content": content}

    def retrieval_state(self, profile, method="hybrid", strict_unknown=False, query_mode="structured", excluded_companies=(), split=None):
        statuses = [constraints(profile, job) for job in self.corpus.jobs]
        excluded_companies = set(excluded_companies)
        allowed = np.array([not any(item["status"] == "fail" or (strict_unknown and item["status"] == "unknown") for item in checks)
                            and self.corpus.jobs[i].company not in excluded_companies
                            and (split is None or self.corpus.research_records.get(self.corpus.jobs[i].id, {}).get("split") == split)
                            for i, checks in enumerate(statuses)])
        query = render_query(profile, query_mode)
        rrf, sources, raw = self.corpus.retrieve_detailed(query, allowed, method, query_segments(profile, query_mode))
        if method != "bm25":
            intent_tokens = tokens(profile["intent"])
            structured = [i for i, job in enumerate(self.corpus.jobs) if allowed[i] and any(word in (job.title+" "+job.category).lower() for word in intent_tokens)]
            structured = sorted(structured,key=lambda i:(-rrf[i],self.corpus.jobs[i].id))[:80]
            sources["structured"] = structured
            rrf[structured] += 1.0 / (60 + np.arange(1,len(structured)+1))
        task_vector = self.corpus.characters.transform([semantic_text(profile["text"])])
        task_alignment = (self.corpus.char_matrix @ task_vector.T).toarray().ravel()
        retrieval = [{**{name:float(values[i]) for name,values in raw.items()},"rrf":float(rrf[i]),
                      "sources":[name for name,indices in sources.items() if i in indices],"task_alignment":float(task_alignment[i])}
                     for i in range(len(self.corpus.jobs))]
        return statuses, allowed, rrf, sources, retrieval

    def replay_features(self, text, preferences, job_ids, query_mode="structured", split=None):
        profile = parse_profile(text,preferences)
        _, allowed, _, _, contexts = self.retrieval_state(profile,query_mode=query_mode,split=split)
        rows=[]
        for job_id in job_ids:
            index=self.corpus.lookup[job_id];job=self.corpus.jobs[index]
            vector=feature_vector(profile,job,contexts[index])
            rows.append({"job_id":job_id,"job_family_id":job.job_family_id,"eligible":bool(allowed[index]),
                         "feature_version":FEATURE_VERSION,"features":{name:None if np.isnan(value) else float(value) for name,value in zip(FEATURE_NAMES,vector)},
                         "retrieval":contexts[index]})
        return rows

    def recommend(self, text: str, preferences: dict, limit: int = 10, method: str = "hybrid", strict_unknown: bool = False,
                  query_mode: str = "structured", excluded_companies=(), split=None) -> dict:
        started = time.perf_counter()
        profile = parse_profile(text, preferences)
        self.skill("resume-diagnose")
        trace = [{"step": "解析简历", "status": "done", "detail": f"识别{len(profile['skills'])}项技能表述"}]
        questions = []
        if not profile["city"]:
            questions.append("你希望在哪个城市工作？")
        if not profile["intent"]:
            questions.append("你优先考虑哪些职位方向？")
        supported = [skill for skill, item in profile["skills"].items() if item["weight"] >= .7]
        task_support = any(item.get("weight",0) >= .7 for item in profile.get("tasks",{}).values())
        described_work = len(semantic_text(text)) >= 60 and bool(re.search(r"负责|完成|实现|开展|组织|协调|维护|解决",semantic_text(text)))
        if not supported and not task_support and not described_work:
            questions.append("请补充一个实际项目：使用了什么技能，具体做了什么？")
        base = {"run_id": uuid.uuid4().hex, "snapshot": self.corpus.snapshot[:16], "profile": {key: value for key, value in profile.items() if key != "text"},
                "method": method, "trace": trace, "model_status": "证据与任务规则排序" if not self.ranker else self.ranker.status,
                "retriever": self.corpus.overview()["engine"],"query_mode":query_mode,"retrieval_contract":contract_info(),
                "encoder_contract":getattr(self.corpus.dense,"encoder_contract",{"model":"字符TF-IDF"}),"feature_version":FEATURE_VERSION}
        if questions:
            trace.append({"step": "确认必要信息", "status": "needs_input", "detail": "先补足求职方向和能力证据"})
            return {**base, "action": "clarify", "questions": questions, "jobs": [], "total_eligible": 0, "latency_ms": round((time.perf_counter()-started)*1000)}
        statuses, allowed, rrf, sources, contexts = self.retrieval_state(profile,method,strict_unknown,query_mode,excluded_companies,split)
        excluded = Counter(item["name"] for checks in statuses for item in checks if item["status"] == "fail")
        trace.append({"step": "检查硬条件", "status": "done", "detail": f"{int(allowed.sum())}个岗位未发现明确冲突"})
        if "char_tfidf_fallback" in sources:
            base["retriever"] = "BM25 + 字符TF-IDF（本次语义编码器不可用，已回退）"
        elif method == "bm25":
            base["retriever"] = "BM25词级检索"
        candidates = np.flatnonzero(rrf > 0)
        trace.append({"step": "多路召回", "status": "done", "detail": f"合并{len(candidates)}个候选岗位"})
        max_rrf = float(rrf.max()) or 1
        intent_words = [word for word in tokens(profile["intent"]) if word not in {"开发", "工程师", "方向", "岗位", "人员", "技术"}]
        ranking = []
        for index in candidates:
            job = self.corpus.jobs[index]
            matched, gaps, coverage = supported_matches(profile, job)
            title = (job.title + " " + job.category).lower()
            intent = sum(word in title for word in intent_words) / max(len(intent_words), 1)
            confidence = sum(item["status"] == "pass" for item in statuses[index]) / max(len(statuses[index]), 1)
            district = 1.0 if profile["district"] and job.district == profile["district"] else .5
            task_alignment = contexts[index]["task_alignment"]
            score = .25 * float(rrf[index])/max_rrf + .25 * coverage + .2 * task_alignment + .2 * intent + .05 * confidence + .05 * district
            if intent_words and intent == 0:
                # 仅“开发”等通用词相同不能掩盖方向不符；相邻方向仍保留候选。
                score *= .55
            if method == "bm25":
                score = float(rrf[index])/max_rrf
            elif method == "retrieval":
                score = float(rrf[index])/max_rrf
            vector = feature_vector(profile,job,contexts[index])
            learned = self.ranker.predict(vector) if self.ranker else None
            if learned is not None and self.ranker.promoted and method == "hybrid":
                score = learned / 100
            ranking.append({**job.public(), "score": round(score*100, 2), "shadow_ranker_score":learned, "skill_coverage": round(coverage*100, 1),
                            "matched": matched, "gaps": gaps, "constraints": statuses[index],
                            "sources": [name for name, indexes in sources.items() if int(index) in indexes],
                            "features": {name:None if np.isnan(value) else float(value) for name,value in zip(FEATURE_NAMES,vector)},
                            "reasons": [f"{item['skill']}：{item['note']}" for item in matched[:3]], "_index": int(index)})
        ranking.sort(key=lambda row: (-row["score"], row["id"]))
        selected, company_counts, families = [], Counter(), set()
        for row in ranking:
            job = self.corpus.jobs[row.pop("_index")]
            family = job.job_family_id or (digest(job.description.strip()) if job.description else job.id)
            if company_counts[job.company] >= 2 or family in families:
                continue
            # 最终逐条核验双侧引用，不把生成文本或引用存在当成能力事实。
            verified = all(text[item["resume_evidence"]["start"]:item["resume_evidence"]["end"]] == item["resume_evidence"]["quote"]
                           and getattr(job, item["job_evidence"]["field"])[item["job_evidence"]["start"]:item["job_evidence"]["end"]] == item["job_evidence"]["quote"] for item in row["matched"])
            if not verified:
                continue
            row["evidence_verified"] = True
            row["evidence_check"] = "原文跨度已核对；能力真实性尚未外部核验"
            selected.append(row)
            company_counts[job.company] += 1
            families.add(family)
            if len(selected) >= limit:
                break
        trace.extend([{"step": "证据精排", "status": "done", "detail": "区分技能提及、使用经历与替代技能组"},
                      {"step": "核验与去重", "status": "done", "detail": "双侧引用已定位，同公司最多2个岗位"}])
        return {**base, "action": "recommend" if selected else "no_match", "questions": [], "jobs": selected,
                "total_eligible": int(allowed.sum()), "retrieved": len(candidates), "excluded": dict(excluded),
                "source_counts": {name: len(indexes) for name, indexes in sources.items()},
                "latency_ms": round((time.perf_counter()-started)*1000),
                "message": "匹配分用于比较本次候选，不代表录用概率。" if selected else "当前条件下未找到有检索依据的岗位。可检查最低薪资或补充项目证据；系统没有自动放宽条件。"}

    def diagnose(self, text: str, preferences: dict, job_id: str) -> dict:
        skill = self.skill("gap-analysis")
        job = self.corpus.by_id[job_id]
        profile = parse_profile(text, preferences)
        matches, gaps, coverage = supported_matches(profile, job)
        same_level = [other for other in self.corpus.jobs if other.category == job.category and other.experience_min == job.experience_min and other.salary["status"] == "月薪"]
        monthly = [(other.salary["monthly_low"] + other.salary["monthly_high"])/2 for other in same_level]
        return {"job": job.public(), "profile_version": profile["version"], "snapshot": self.corpus.snapshot[:16],
                "matched": matches, "gaps": gaps, "coverage": round(coverage*100, 1), "constraints": constraints(profile, job),
                "market": {"n": len(monthly), "category": job.category, "experience_min": job.experience_min,
                           "quantiles": [round(float(value)) for value in np.quantile(monthly, [.25, .5, .75])] if len(monthly) >= 30 else None,
                           "note": "同职类、相同最低年资的广告月薪中点；样本不足30条时不报分位。"},
                "skill": {"name": skill["name"], "version": skill["version"]},
                "description": display_text(job.description), "groups": job.groups,
                "actions": [{"skill": gap["skill"], "action": "若已做过，请补充项目背景、实际使用方法与可核验结果；尚未做过则作为学习计划，不写成已有能力。"} for gap in gaps[:5]]}

    def rewrite(self, text: str, preferences: dict, job_id: str) -> dict:
        skill = self.skill("resume-rewrite")
        job = self.corpus.by_id[job_id]
        # 确定性版本只调整现有句子的顺序与分段；逐句保留，不虚构数字/经历。
        chunks = experience_blocks(text)
        evidence = set(job.skills)
        scored = [(sum(key in evidence for key, item in skill_evidence(part).items() if item["weight"] > 0), i, part) for i, part in enumerate(chunks)]
        reordered = [part for _, _, part in sorted(scored, key=lambda item: (-item[0], item[1]))]
        model = None
        if self.coach.url and 1 < len(chunks) <= 40:
            try:
                reordered, model = self.coach.reorder(chunks, job.title)
            except Exception:
                model = None
        if re.search(r"前者|后者|上述|下述|该项目|其|随后|接着|首先|其次|最后|以上|以下", text):
            reordered = chunks[:]
        revised = "\n\n".join(reordered)
        assert Counter(chunks) == Counter(reordered)
        same_facts = fact_fingerprint(text,preferences) == fact_fingerprint(revised,preferences)
        if not same_facts:
            revised = text
        return {"original": text, "revised": revised, "facts_preserved": fact_fingerprint(text,preferences)==fact_fingerprint(revised,preferences), "new_claims": 0,
                "mode": "本地LLM原段排序＋事实保全" if model else "证据保全整理", "model":model, "skill_version": skill["version"],
                "explanation": "保留完整经历块的标题、日期与职责归属；核对确认字段及原块内容。语义或事实检查不通过时保留原文。",
                "suggestions": self.diagnose(text, preferences, job_id)["actions"],
                **compare_versions(text,revised,preferences)}

    def interview(self, text: str, preferences: dict, job_id: str) -> dict:
        skill = self.skill("mock-interview")
        diagnosis = self.diagnose(text, preferences, job_id)
        questions = [{"question": f"请结合你实际做过的项目，说明{item['skill']}解决了什么问题，如何确认结果可靠？",
                      "evidence": item["job_evidence"], "rubric": ["说明真实背景与个人职责", "解释技术选择和具体操作", "提供可核验结果并承认局限"]} for item in diagnosis["matched"][:5]]
        return {"questions": questions, "skill_version": skill["version"], "note": "用于自查与面试准备，不是录用评价。"}
