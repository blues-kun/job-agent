"""只读岗位库、中文多路检索与需求统计。"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import logging
import re
import time

import jieba
import numpy as np
from openpyxl import load_workbook
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .domain import Job, digest, PARSER_VERSION
from .retrieval_contract import render_document, contract_info

jieba.setLogLevel(logging.WARNING)
TOKENIZER = jieba.Tokenizer()
STOP = set("的 了 和 与 或 及 等 有 在 对 为 是 能 负责 相关 要求 工作 具备 以上 岗位 公司 进行 使用 熟悉 掌握 优先 经验".split())


def tokens(text: str) -> list[str]:
    # 中文词级分词；保护 C++、C# 等技术词，匹配数字避免引入姓名差异。
    text = text.lower().replace("c++", "cplusplus").replace("c#", "csharp")
    return [word for word in TOKENIZER.lcut(text) if word not in STOP and re.search(r"[a-z\u4e00-\u9fff]", word) and len(word) > 1]


class Corpus:
    def __init__(self, source: Path, dense_dir: Path | None = None, records: list[dict] | None = None):
        started = time.perf_counter()
        self.source = source
        self.snapshot = hashlib.sha256(source.read_bytes()).hexdigest()
        self.jobs = []
        self.raw_count, self.duplicates, self.invalid_count = 0, 0, 0
        self.research_records = {record["job_id"]: record for record in records or []}
        self.parsed_source_versions = sorted({record.get("parser_version", "legacy-unversioned") for record in records or []})
        if records is not None:
            for record in records:
                if record["snapshot"] != self.snapshot:
                    raise ValueError("研究岗位版本不属于当前原表快照")
                if record.get("parser_version") and record["parser_version"] != PARSER_VERSION:
                    raise ValueError("岗位解析版本过期，请生成新的语义快照后加载，不能静默重解释已记录的特征")
                self.raw_count += len(record["source_record_ids"])
                if not record["title"]:
                    self.invalid_count += 1
                    continue
                job = Job(id=record["job_id"], version=record["job_version_id"], title=record["title"], company=record["company"],
                          category=record["category"], family=record["category_parent"], salary_raw=record["salary_raw"],
                          requirements=record["requirements"], description=record["description"], address=record["address"], row=0)
                job.job_family_id = record["job_family_id"]
                self.jobs.append(job)
            self.duplicates = self.raw_count - len(records)
        else:
            seen = set()
            book = load_workbook(source, read_only=True, data_only=True)
            try:
                for sheet_number, sheet in enumerate(book.worksheets, 1):
                    rows = sheet.iter_rows(values_only=True)
                    header = [str(value or "").strip().strip("\ufeff") for value in next(rows, [])]
                    for row_number, row in enumerate(rows, 2):
                        if not any(value is not None for value in row):
                            continue
                        self.raw_count += 1
                        data = {key: str(value).strip() if value is not None else "" for key, value in zip(header, row)}
                        def get(*keys):
                            return next((data[key] for key in keys if data.get(key)), "")
                        title, company = get("岗位名称", "职位名称", "jobName"), get("企业", "公司", "公司名称", "brandName")
                        if not title:
                            self.invalid_count += 1
                            continue
                        salary = get("岗位薪资", "薪资", "salaryDesc")
                        key = (title, company, salary)
                        if key in seen:
                            self.duplicates += 1
                            continue
                        seen.add(key)
                        description = get("岗位职责", "职位描述", "岗位描述")
                        self.jobs.append(Job(id=digest("\0".join(key))[:20], version=digest(str(sorted(data.items())))[:16],
                            title=title, company=company, category=get("职位类型名称", "三级分类", "职位类型"),
                            family=get("二级分类", "大类"), salary_raw=salary, requirements=get("岗位要求", "任职要求", "职位要求"),
                            description=description, address=get("岗位地址", "工作地点", "地址"), row=row_number, sheet=sheet_number))
            finally:
                book.close()
        if not self.jobs:
            raise ValueError("岗位库没有可用记录，请检查 Excel 字段映射。")
        self.by_id = {job.id: job for job in self.jobs}
        self.lookup = {job.id: index for index, job in enumerate(self.jobs)}
        self.categories = Counter(job.category for job in self.jobs if job.category)
        documents = [render_document(job) for job in self.jobs]
        self.retrieval_contract = contract_info()
        self.counts = CountVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, min_df=1, max_features=80000)
        frequency = self.counts.fit_transform(documents).tocsr().astype(np.float32)
        length = np.asarray(frequency.sum(axis=1)).ravel()
        document_frequency = np.asarray((frequency > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(self.jobs) - document_frequency + .5) / (document_frequency + .5))
        denominator = 1.2 * (1 - .75 + .75 * length / max(float(length.mean()), 1))
        repeated = np.repeat(denominator, np.diff(frequency.indptr))
        frequency.data = frequency.data * 2.2 / (frequency.data + repeated)
        self.bm25 = frequency.multiply(idf).tocsr()
        self.characters = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2, max_features=50000, dtype=np.float32)
        self.char_matrix = self.characters.fit_transform(documents)
        self.dense = None
        self.dense_status = "尚未加载预训练向量"
        if dense_dir:
            from .dense import DenseIndex
            try:
                self.dense = DenseIndex(dense_dir, self.snapshot, [job.id for job in self.jobs], documents)
                self.dense_status = "预训练向量已就绪"
            except Exception as error:
                self.dense_status = "预训练向量初始化失败，已回退字符TF-IDF"
                logging.getLogger(__name__).warning("%s（%s）", self.dense_status, type(error).__name__)
        self.build_seconds = round(time.perf_counter() - started, 2)

    def retrieve(self, query: str, allowed: np.ndarray, method: str = "hybrid") -> tuple[np.ndarray, dict]:
        rrf, sources, _ = self.retrieve_detailed(query, allowed, method)
        return rrf, sources

    def retrieve_detailed(self, query: str, allowed: np.ndarray, method: str = "hybrid", segments=None):
        word_query = self.counts.transform([query])
        bm25 = (self.bm25 @ word_query.T).toarray().ravel()
        chars = (self.char_matrix @ self.characters.transform([query]).T).toarray().ravel()
        scores = {"bm25": bm25}
        if method != "bm25":
            if self.dense:
                try:
                    scores["dense"] = self.dense.score_segments(segments or [query])
                except Exception:
                    # 本次结果显式显示字符回退，不能把词面分数声称为预训练向量。
                    scores["char_tfidf_fallback"] = chars
            else:
                scores["char_tfidf"] = chars
        rrf = np.zeros(len(self.jobs), dtype=np.float32)
        sources = {}
        for name, values in scores.items():
            indices = np.flatnonzero(allowed & (values > 0))
            limit = 100 if name == "bm25" else 120
            order = indices[np.argsort(-values[indices], kind="stable")[:limit]]
            sources[name] = order.tolist()
            rrf[order] += 1.0 / (60 + np.arange(1, len(order) + 1))
        return rrf, sources, scores

    @lru_cache(maxsize=100)
    def overview(self, category: str = "") -> dict:
        jobs = [job for job in self.jobs if not category or job.category == category]
        n = len(jobs)
        skill_counts = Counter(skill for job in jobs for skill, item in job.skills.items() if item["level"] != "否定")
        education_counts = Counter(job.requirements for job in [])
        for job in jobs:
            match = re.search(r"博士|硕士|本科|大专|学历不限|高中|中专", job.requirements)
            education_counts[match.group() if match else "其他/未知"] += 1
        experience_counts = Counter("未给年资" if job.experience_min is None else "不限/应届" if job.experience_min == 0 else "1–3年" if job.experience_min < 3 else "3–5年" if job.experience_min < 5 else "5年以上" for job in jobs)
        district_counts = Counter(job.district or "区域未知" for job in jobs)
        monthly = [(job.salary["monthly_low"] + job.salary["monthly_high"]) / 2 for job in jobs if job.salary["status"] == "月薪"]
        def series(counter, limit=12):
            return [{"name": key, "count": count, "ratio": round(count / n, 4) if n else 0} for key, count in counter.most_common(limit)]
        examples = []
        for job in jobs:
            for group in job.groups:
                if group["kind"] == "any":
                    examples.append({"job_id": job.id, "title": job.title, "label": group["label"], "quote": group["evidence"]["quote"]})
                    break
            if len(examples) >= 3:
                break
        return {"snapshot": self.snapshot[:16], "source_label": "仓库岗位快照" if self.source.name == "job_data.xlsx" else "用户指定岗位快照",
                "parser_version":PARSER_VERSION,"parsed_source_versions":self.parsed_source_versions,
                "retrieval_contract_hash":self.retrieval_contract["hash"],
                "dedup_policy":"按完整内容去重，保留职责版本" if self.research_records else "按岗位、企业、薪资业务键去重",
                "total": n, "raw_total": self.raw_count, "duplicates": self.duplicates, "invalid": self.invalid_count,
                "category_count": len(self.categories), "company_count": len({job.company for job in jobs}),
                "unknown_location": sum(job.city is None for job in jobs), "salary_count": len(monthly),
                "salary_quantiles": [round(float(value)) for value in np.quantile(monthly, [.25, .5, .75])] if len(monthly) >= 30 else None,
                "skills": series(skill_counts, 16), "experience": series(experience_counts), "education": series(education_counts),
                "districts": series(district_counts, 8), "categories": series(Counter(job.category for job in jobs), 12),
                "category_options": [{"name": key, "count": count} for key, count in self.categories.most_common()],
                "alternative_groups": sum(group["kind"] == "any" for job in jobs for group in job.groups), "alternative_examples": examples,
                "notes": ["历史岗位快照，无法确认目前仍在招聘。", "薪资为广告月薪区间中点，非到手工资；少于30条不报分位。", "技能统计来自JD正文的词典提及，必需与优先仍需核对原文。"],
                "engine": "BM25 + 预训练向量" if self.dense else f"BM25 + 字符TF-IDF（{self.dense_status}）"}
