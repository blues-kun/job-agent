#!/usr/bin/env python3
"""只读剖析招聘 XLSX，输出脱敏中文报告；不转换、复制或写回原始数据。

例：python scripts/profile_data.py --input E:\\file\\note\\profile\\merged_data.xlsx
Linux 必须显式指定实际可访问的绝对路径。输入不存在时失败，绝不偷偷换源。
依赖：pandas、openpyxl。脚本与业务模块隔离，避免导入旧项目的模型或密钥。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile


DEFAULT_INPUT = r"E:\file\note\profile\merged_data.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "DATA_PROFILE.md"
REPORT_MARKER = "<!-- job-agent:profile-data-report:4bc1eabe-969c-4f4d-a130-ec7dce304d80:v1 -->"
ALIASES = {
    "title": ["岗位名称", "职位名称", "职位名", "岗位名", "jobName", "title"],
    "company": ["企业", "公司名称", "公司", "企业名称", "brandName", "company"],
    "salary": ["岗位薪资", "薪资", "薪资待遇", "salaryDesc", "salary"],
    "requirements": ["岗位要求", "任职要求", "职位要求", "requirements"],
    "description": ["岗位职责", "职位描述", "岗位描述", "职位详情", "description"],
    "address": ["岗位地址", "工作地点", "工作地址", "地址", "address"],
    "district": ["区域", "行政区", "区县", "district"],
    "business": ["商圈", "商圈名称", "businessDistrict"],
    "education": ["学历要求", "学历", "jobDegree", "education"],
    "experience": ["经验要求", "工作经验", "经验", "jobExperience", "experience"],
    "size": ["公司规模", "企业规模", "brandScaleName", "company_size"],
    "industry": ["公司行业", "所属行业", "行业", "brandIndustry", "industry"],
    "skills": ["技能标签", "岗位技能", "技能", "标签", "skills", "skill_tags"],
    "category": ["职位类型名称", "职位类型", "岗位类别", "三级分类", "category"],
    "supercategory": ["二级分类", "一级分类", "大类"],
}
DISTRICTS = ["南山区", "福田区", "宝安区", "龙岗区", "龙华区", "罗湖区", "光明区", "坪山区", "盐田区", "大鹏新区", "深汕特别合作区"]
# 词典只用于描述覆盖率，不能把“命中关键词”当作胜任能力或招聘要求。
SKILLS = {
    "Java": r"\bjava\b", "Python": r"\bpython\b", "C++": r"c\s*\+\s*\+", "C#": r"c\s*#", "C语言": r"(?<![a-z])c语言|(?<![a-z])c(?=\s*[/、,，])",
    "JavaScript": r"\bjavascript\b|\bjs\b", "TypeScript": r"\btypescript\b", "Go": r"\bgolang\b|\bgo语言|\bgo\b", "PHP": r"\bphp\b", "Rust": r"\brust\b",
    "SQL": r"\bsql\b", "MySQL": r"\bmysql\b", "PostgreSQL": r"\bpostgresql\b", "Oracle": r"\boracle\b", "Redis": r"\bredis\b", "MongoDB": r"\bmongodb\b",
    "Linux": r"\blinux\b", "Unix": r"\bunix\b", "Shell": r"\bshell\b", "Git": r"\bgit\b", "Docker": r"\bdocker\b", "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "Spring": r"\bspring\b", "Spring Boot": r"spring\s*boot", "Spring Cloud": r"spring\s*cloud", "Django": r"\bdjango\b", "Flask": r"\bflask\b", "FastAPI": r"\bfastapi\b",
    "Vue": r"\bvue(?:\.js)?\b", "React": r"\breact\b", "Angular": r"\bangular\b", "Node.js": r"\bnode(?:\.js|js)\b", "HTML": r"\bhtml5?\b", "CSS": r"\bcss3?\b",
    "Android": r"\bandroid\b", "iOS": r"\bios\b", "Swift": r"\bswift\b", "Kotlin": r"\bkotlin\b", "Flutter": r"\bflutter\b", "Unity": r"\bunity(?:3d)?\b", "UE4/UE5": r"\bue[45]\b|unreal",
    "PyTorch": r"\bpytorch\b", "TensorFlow": r"\btensorflow\b", "Keras": r"\bkeras\b", "Scikit-learn": r"scikit[- ]learn|\bsklearn\b", "Pandas": r"\bpandas\b", "NumPy": r"\bnumpy\b",
    "机器学习": r"机器学习", "深度学习": r"深度学习", "自然语言处理": r"自然语言处理|\bnlp\b", "计算机视觉": r"计算机视觉|\bcv\b", "大语言模型": r"大语言模型|大模型|\bllm[s]?\b",
    "Transformer": r"\btransformer[s]?\b", "BERT": r"\bbert\b", "RAG": r"\brag\b|检索增强生成", "LoRA": r"\blora\b", "推荐系统": r"推荐系统|推荐算法", "强化学习": r"强化学习", "知识图谱": r"知识图谱",
    "数据分析": r"数据分析", "数据挖掘": r"数据挖掘", "数据仓库": r"数据仓库|数仓", "Hadoop": r"\bhadoop\b", "Spark": r"\bspark\b", "Flink": r"\bflink\b", "Hive": r"\bhive\b", "Kafka": r"\bkafka\b", "Elasticsearch": r"\belasticsearch\b",
    "微服务": r"微服务", "分布式": r"分布式", "高并发": r"高并发", "TCP/IP": r"tcp\s*/\s*ip|\btcp\b", "HTTP": r"\bhttps?\b", "REST": r"\brest(?:ful)?\b", "自动化测试": r"自动化测试", "性能测试": r"性能测试",
    "Selenium": r"\bselenium\b", "JMeter": r"\bjmeter\b", "Appium": r"\bappium\b", "嵌入式": r"嵌入式", "FPGA": r"\bfpga\b", "ARM": r"\barm\b", "CUDA": r"\bcuda\b", "MATLAB": r"\bmatlab\b",
    "OpenCV": r"\bopencv\b", "图像处理": r"图像处理", "语音识别": r"语音识别", "目标检测": r"目标检测", "项目管理": r"项目管理", "敏捷开发": r"敏捷|\bscrum\b", "Excel": r"\bexcel\b", "Power BI": r"power\s*bi", ".NET": r"\.net\b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(source: Path, output: Path) -> None:
    """拒绝输入别名、符号链接及不属于本工具的既有文档。"""
    if output.suffix.lower() != ".md":
        raise ValueError("输出必须为 .md 报告文件。")
    if output.is_symlink():
        raise ValueError("输出不能是符号链接。")
    if output == source or (output.exists() and os.path.samefile(source, output)):
        raise ValueError("输出与输入指向同一文件（包括硬链接），禁止覆盖原表。")
    if output.exists():
        if not output.is_file():
            raise ValueError("输出不是普通文件。")
        with output.open("rb") as stream:
            first_line = stream.readline(len(REPORT_MARKER.encode("utf-8")) + 4)
        if first_line.strip() != REPORT_MARKER.encode("utf-8"):
            raise ValueError("既有输出没有本工具的报告标记，拒绝覆盖；请指定新的 .md 文件名。")


def atomic_write_report(source: Path, output: Path, content: str, source_hash: str) -> None:
    """以新 inode 原子替换报告，不沿输出硬链接写入；写后复核源指纹。"""
    validate_output(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent, prefix=".profile-report-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        validate_output(source, output)
        if sha256(source) != source_hash:
            raise RuntimeError("写报告前源文件指纹变化，拒绝发布本次报告。")
        os.replace(temporary, output)
        temporary = None
        if sha256(source) != source_hash:
            raise RuntimeError("写报告后源文件指纹变化，原始数据完整性验证失败。")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalize_headers(df):
    """去除列头 BOM/首尾空白；同名列保留并加序号，避免隐式丢列。"""
    df = df.copy()
    counts = Counter()
    columns = []
    for index, column in enumerate(df.columns, 1):
        base = str(column).strip().strip("\ufeff").strip() or f"未命名列{index}"
        counts[base] += 1
        columns.append(base if counts[base] == 1 else f"{base}__重复{counts[base]}")
    df.columns = columns
    return df


def resolve_fields(df) -> dict:
    """依别名优先序选有非空值的列；全空同义列不遮蔽后续有效列。"""
    fields = {}
    for role, aliases in ALIASES.items():
        candidates = [column for alias in aliases for column in df.columns if re.sub(r"__重复\d+$", "", column) == alias]
        fields[role] = next((column for column in candidates if df[column].fillna("").astype(str).str.strip().ne("").any()), candidates[0] if candidates else None)
    return fields


def parse_salary(value: object) -> dict:
    """保留日薪、时薪、面议；只将明确月薪/年薪作可比口径，不用固定天数外推。"""
    raw = str(value).strip()
    if raw.lower() in ("", "nan", "none", "<na>"):
        return {"status": "缺失"}
    if "面议" in raw:
        return {"status": "面议"}
    s = re.sub(r"\s+", "", raw).replace("～", "-").replace("—", "-").replace("–", "-").replace("~", "-").replace("至", "-")
    months_match = re.search(r"[·・•](\d{1,2})薪$", s)
    months = int(months_match.group(1)) if months_match else 12
    if months_match:
        s = s[:months_match.start()]
    if not 12 <= months <= 24:
        return {"status": "未解析：薪数异常"}
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kK千万元]?)(?:-(\d+(?:\.\d+)?)([kK千万元]?))?(?:[/／](月|年|天|日|小时|时|周))?", s)
    if not m:
        return {"status": "未解析：格式不支持"}
    left, unit_a, right, unit_b, period = m.groups()
    unit_a = unit_a or unit_b
    unit_b = unit_b or unit_a
    factor = {"k": 1000, "千": 1000, "万": 10000, "元": 1, "": 1}
    low, high = float(left) * factor[unit_a.lower()], float(right or left) * factor[unit_b.lower()]
    if low <= 0 or high < low:
        return {"status": "未解析：数值异常"}
    if period in ("天", "日", "小时", "时", "周"):
        return {"status": {"天": "日薪", "日": "日薪", "小时": "时薪", "时": "时薪", "周": "周薪"}[period], "low": low, "high": high}
    if period is None and unit_a.lower() not in ("k", "千"):
        return {"status": "未解析：缺少周期"}
    if period == "年":
        if months_match:
            return {"status": "未解析：年薪附带薪数"}
        return {"status": "年薪", "annual_low": low, "annual_high": high, "months": None}
    return {"status": "月薪", "monthly_low": low, "monthly_high": high, "annual_low": low * months, "annual_high": high * months, "months": months, "months_explicit": bool(months_match)}


def education_from_text(value: str) -> str:
    if not value.strip():
        return "缺失"
    found = re.findall(r"博士|硕士|研究生|本科|大专|专科|中专|中技|高中|初中|学历不限|学历不要求", value)
    return "/".join(dict.fromkeys(found)) if found else "未识别"


def experience_from_text(value: str) -> str:
    if not value.strip():
        return "缺失"
    m = re.search(r"经验不限|不限经验|在校\s*/\s*应届|应届(?:毕业生)?|无需经验|无经验|\d+\s*[-~至]\s*\d+年|\d+年(?:以上|以下|以内)?|\d+天/周", value)
    return re.sub(r"\s+", "", m.group()) if m else "未识别"


def minimum_experience_years(value: str) -> int | None:
    """只识别明确年资下限，不把每周出勤或“以内”当作最低年资。"""
    match = re.fullmatch(r"(\d+)(?:[-~至]\d+年|年(?:以上)?)", value)
    return int(match.group(1)) if match else None


def redactor(companies: list[str]):
    names = sorted({name.strip() for name in companies if len(name.strip()) >= 2}, key=len, reverse=True)
    company_pattern = re.compile("|".join(re.escape(name) for name in names)) if names else None

    def redact(value: object) -> str:
        text = str(value)
        if text in ("地址缺失", "地址存在但未识别深圳行政区"):
            return text
        # 保留薪资聚合中的数值区间，避免 200-1000 被电话号码规则误删。
        if re.fullmatch(r"[0-9.Kk千万元/月年天日小时周薪·\-]{1,35}", text) and not re.search(r"\d{7,}", text):
            return text
        if company_pattern:
            text = company_pattern.sub("某企业", text)
        text = re.sub(r"(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.\w+", "[联系方式已隐去]", text)
        text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)|(?<!\d)\d[\d\s-]{6,}\d(?!\d)", "[号码已隐去]", text)
        text = re.sub(r"(?:微信|vx|wechat|QQ|电话|邮箱|联系|地址|坐标)[^。；;\n]{0,100}", "[联系或地址信息已隐去]", text, flags=re.I)
        text = re.sub(r"[\u4e00-\u9fffA-Za-z0-9（）()]{2,45}(?:有限公司|集团|股份公司|科技公司)", "某企业", text)
        return text.replace("\n", " ").replace("\r", " ").replace("|", "\\|").replace("`", "'")
    return redact


def table(headers: list[str], rows: list[list[object]]) -> str:
    def escape(x: object) -> str:
        return str(x).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(escape(x) for x in row) + " |" for row in rows])


def profile_sheet(df, name: str) -> tuple[str, dict]:
    import pandas as pd

    df = normalize_headers(df)
    n = len(df)
    fields = resolve_fields(df)
    texts = {c: df[c].fillna("").astype(str).str.strip() for c in df.columns}
    company_columns = [column for column in df.columns if re.sub(r"__重复\d+$", "", column) in ALIASES["company"]]
    company_values = [value for column in company_columns for value in texts[column].tolist()]
    redact = redactor(company_values)
    known_headers = {alias for aliases in ALIASES.values() for alias in aliases}
    safe_header = lambda c: c if re.sub(r"__重复\d+$", "", c) in known_headers else redact(c)
    lines = [f"## 工作表：{redact(name)}", f"有效数据记录 **{n:,} 行 × {len(df.columns)} 列**（不含表头；全空行已排除）。dtype 为本次 pandas 推断。列头已去 BOM 与首尾空白，同名列保留加序号；非标准元数据已脱敏。缺失率同时计入空单元格与纯空白字符串。", "### 字段与缺失"]
    lines.append(table(["列名", "dtype", "缺失条数", "缺失率", "非空不同值数"], [[safe_header(c), str(df[c].dtype), int((texts[c] == "").sum()), f"{(texts[c] == '').mean():.2%}" if n else "—", int(texts[c][texts[c] != ""].nunique())] for c in df.columns]))
    def distribution(title: str, series, top: int = 30) -> None:
        counts = series.replace("", "缺失").value_counts(dropna=False)
        lines.extend([f"### {title}", table(["类别", "记录数", "占全部记录"], [[redact(k), int(v), f"{v / n:.2%}" if n else "—"] for k, v in counts.head(top).items()])])
        if len(counts) > top:
            lines.append(f"共 {len(counts):,} 个取值，展示前 {top} 个。")
    keys = [fields[k] for k in ("title", "company", "salary")]
    exact_duplicate = int(df.duplicated().sum())
    lines.append("### 重复与独立样本")
    summary = {"rows": n, "columns": len(df.columns), "exact_duplicates": exact_duplicate}
    if all(keys):
        key_df = pd.DataFrame({k: texts[k] for k in keys})
        duplicate = int(key_df.duplicated().sum())
        involved = int(key_df.duplicated(keep=False).sum())
        lines.append(f"全行完全重复 **{exact_duplicate:,} 条**；按“{' + '.join(keys)}”判重，额外重复 **{duplicate:,} 条（{duplicate / n:.2%}）**，涉及 **{involved:,} 条**，去重剩 **{n - duplicate:,} 条**。判重只修剪首尾空白、将缺失归为空串，没有合并企业别名；同名同薪不同 JD 可能是真实不同岗位，必须再比较职责与发布时间。")
        summary.update(key_duplicates=duplicate, dedup_rows=n-duplicate)
    else:
        lines.append(f"全行完全重复 {exact_duplicate:,} 条；缺少岗位名、公司或薪资列，不能执行指定业务键判重。")
    if fields["company"]:
        lines.append(f"企业字段有 **{texts[fields['company']].replace('', pd.NA).nunique():,} 个非空不同值**（未经实体归一化，不公开企业明细）。")
    for role, title, limit in (("title", "岗位名称 Top 30", 30), ("category", "职位类型分布 Top 30", 30), ("supercategory", "上层分类分布", 30)):
        if fields[role]:
            distribution(title + f"（独立字段：{fields[role]}）", texts[fields[role]], limit)
            summary[role + "_distinct"] = int(texts[fields[role]].replace("", pd.NA).nunique())
    lines.append("### 薪资原文格式与可比口径")
    if fields["salary"]:
        salary = texts[fields["salary"]]
        parsed = [parse_salary(v) for v in salary]
        counts = Counter(row["status"] for row in parsed)
        lines.append(table(["解析状态", "条数", "占比", "原格式样例（最多 3 个）"], [[state, count, f"{count/n:.2%}", "；".join(redact(x) for x in list(dict.fromkeys(v for v,p in zip(salary,parsed) if p['status']==state))[:3]) or "空值"] for state,count in counts.most_common()]))
        syntax_success = sum(counts[s] for s in ("月薪", "年薪", "日薪", "时薪", "周薪"))
        syntax_failure = sum(count for status,count in counts.items() if status.startswith("未解析"))
        lines.append(f"语法识别成功（包括分池处理的日/时/周薪）**{syntax_success:,} 条（{syntax_success/n:.2%}）**，未解析 **{syntax_failure:,} 条**，显式年薪 **{counts['年薪']:,} 条**；面议、缺失单列，不计入成功或语法失败。")
        distribution("薪资原始格式 Top 20", salary, 20)
        comparable = [p for p in parsed if p["status"] in ("月薪", "年薪")]
        monthly = [p for p in parsed if p["status"] == "月薪"]
        explicit_months = sum(bool(p["months_explicit"]) for p in monthly)
        lines.append(f"可在明确口径下比较的月/年薪共 **{len(comparable):,} 条（{len(comparable)/n:.2%}）**；月薪中显式给出发薪月数 **{explicit_months:,} 条**。K 默认解读为千元/月；未写薪数按 12 月作估计，额外月薪是否保证发放无法由原串确定。日薪/时薪保留原单位，不按 21.75 天外推。面议或解析失败为 unknown，不能当作 0 元或自动判定不匹配。")
        if comparable:
            annual_mid = pd.Series([(p["annual_low"]+p["annual_high"])/2 for p in comparable])
            quantile_rows = []
            if monthly:
                monthly_mid = pd.Series([(p["monthly_low"]+p["monthly_high"])/2 for p in monthly])
                quantile_rows.append(["广告月薪区间中点（元/月）"] + [f"{monthly_mid.quantile(q):,.0f}" for q in (.1,.25,.5,.75,.9)])
                summary["monthly_mid_median"] = float(monthly_mid.median())
            quantile_rows.append(["估计年薪区间中点（元/年）"] + [f"{annual_mid.quantile(q):,.0f}" for q in (.1,.25,.5,.75,.9)])
            lines.append(table(["统计对象", "P10", "P25", "P50", "P75", "P90"], quantile_rows))
            lines.append("以上是招聘广告薪资区间中点的样本分布，不是实际到手工资、真实录用薪资或全深圳劳动力市场分位数；重复发布和技术岗位采样会造成偏差。比较个人预期应先按去重岗位、职类、经验、学历分层，并保留薪资区间。")
            summary.update(annual_mid_median=float(annual_mid.median()), salary_comparable=len(comparable))
        lines.append("解析挑战：`15-25K·14薪`应分别保存月薪 15,000–25,000、薪数 14、估计年薪 210,000–350,000；`面议`不可数值化；`200-300元/天`不与全职月薪混排；`15-25万/年`无需再乘 12。格式被解析仅代表语法成功，不保证广告可信或税前/税后口径明确。")
    else:
        lines.append("未识别独立薪资字段，无法统计或做薪资硬过滤。")
    for role, title, parser in (("education", "学历要求", education_from_text), ("experience", "经验要求", experience_from_text)):
        if fields[role]:
            values, source = texts[fields[role]], f"独立字段：{fields[role]}"
            normalized_values = values.map(parser)
        elif fields["requirements"]:
            values, source = texts[fields["requirements"]].map(parser), f"从 {fields['requirements']} 规则提取，非独立字段"
            normalized_values = values
        else:
            lines.extend([f"### {title}", "缺少独立字段与可定位的岗位要求，未知值不能作不满足处理。"])
            continue
        distribution(title + "（" + source + "）", values)
        attendance_only = int(normalized_values.str.contains(r"天/周", regex=True).sum()) if role == "experience" else 0
        unknown = int(normalized_values.isin(["", "缺失", "未识别"]).sum()) + attendance_only
        lines.append(f"缺失或未识别 **{unknown:,} 条**。规则只识别字面表达；混合学历、‘优先’、‘可放宽’与实习出勤条件需要按原文复核，不能全部当成必须门槛。")
        if fields[role]:
            lines.append("上表保留独立字段的原始取值分布；‘未知’、‘暂无’等无明确含义文本在判定通道归入 unknown，不能当作已满足门槛。")
        if attendance_only:
            lines.append(f"其中 **{attendance_only:,} 条**提取结果是每周出勤天数，不是工作年资；上表保留原表达用于发现数据类型混杂，年资过滤必须将这些记录设为 unknown，另存 days_per_week。")
        summary[role + "_unknown"] = unknown
    lines.append("### 目标人群覆盖与职类 × 经验薪资参考")
    technical_categories = {"后端开发", "人工智能", "测试", "前端/移动开发", "数据", "高端技术职位", "技术项目管理", "销售技术支持", "其他技术职位"}
    if fields["supercategory"]:
        supercategories = texts[fields["supercategory"]]
        technical_count = int(supercategories.isin(technical_categories).sum())
        lines.append(f"按上述 9 项明确的技术相关上层分类识别，本表有 **{technical_count:,}/{n:,} 条（{technical_count/n:.2%}）**技术岗位记录。分类依据是岗位职能，不是雇主行业；这份样本不能代表教育、医疗、财务等全行业招聘市场。")
    exp_column = fields["experience"] or fields["requirements"]
    if exp_column:
        experience_values = texts[exp_column].map(experience_from_text)
        senior_mask = experience_values.map(lambda value: (minimum_experience_years(value) or 0) >= 3)
        senior_count = int(senior_mask.sum())
        lines.append(f"全部原始记录中，明确要求 **至少 3 年经验**的岗位 **{senior_count:,}/{n:,} 条（{senior_count/n:.2%}）**；仅按可识别的年资下限判定，经验不限、在校/应届、年资缺失及每周出勤要求均不计入该分子。由此可见当前样本明显偏向有经验求职者，不能把全表月薪中点直接作为深圳大学应届生的目标薪资。")
        summary.update(experience_at_least_3_years=senior_count, experience_at_least_3_years_ratio=senior_count/n)
        if all(keys):
            keep_mask = ~key_df.duplicated(keep="first")
            dedup_count = int(keep_mask.sum())
            dedup_senior = int((senior_mask & keep_mask).sum())
            lines.append(f"按同一业务键去重后，至少 3 年经验 **{dedup_senior:,}/{dedup_count:,} 条（{dedup_senior/dedup_count:.2%}）**。下表先在全表按“{' + '.join(keys)}”去重、保留文件内首条，再选取明确月薪记录；不把日薪、时薪、周薪、面议换算为月薪，也不把经验不限合并到应届档。")
            if fields["salary"] and fields["category"]:
                strata = pd.DataFrame({
                    "category": texts[fields["category"]],
                    "experience": experience_values.replace({"应届": "在校/应届", "应届毕业生": "在校/应届"}),
                    "monthly_mid": [(item["monthly_low"]+item["monthly_high"])/2 if item["status"] == "月薪" else float("nan") for item in parsed],
                }, index=df.index)
                strata = strata.loc[keep_mask & strata["monthly_mid"].notna()]
                categories = [value for value in ("Python", "数据分析师", "数据分析", "Java") if value in set(texts[fields["category"]])]
                rows, metrics = [], []
                for category in categories:
                    for experience in ("在校/应届", "1-3年", "3-5年"):
                        salaries = strata.loc[(strata["category"] == category) & (strata["experience"] == experience), "monthly_mid"]
                        count = len(salaries)
                        quantiles = [float(salaries.quantile(q)) for q in (.25,.5,.75)] if count >= 30 else None
                        rows.append([redact(category), experience, count] + ([f"{q:,.0f}" for q in quantiles] if quantiles else ["样本不足", "样本不足", "样本不足"]))
                        metrics.append({"category":category, "experience":experience, "n":count, "monthly_mid_p25_p50_p75":quantiles})
                if rows:
                    lines.append(table(["实际职类", "经验档", "去重月薪样本 n", "P25（元/月）", "P50（元/月）", "P75（元/月）"], rows))
                    lines.append("分位数对象仍是广告月薪区间的中点，采用 pandas 默认线性插值；**仅 n ≥ 30 才给出 P25/P50/P75**。n < 30 只显示样本数，不能借邻近职类或高年资样本生成看似精确的应届薪资结论。该表用于同源样本参考，不是录用概率、个人薪酬承诺或现时市场价格；额外发薪月数、学历与细分技术方向仍需进一步分层。")
                    summary["salary_strata"] = metrics
                else:
                    lines.append("本表不含 Python、数据分析师/数据分析、Java 这几个指定职类，不生成对应薪资分位。")
            else:
                lines.append("缺少独立薪资或职类字段，无法生成可核验的职类 × 经验分位。")
        else:
            lines.append("缺少指定去重键，不能按要求计算去重后的薪资分位，故留空。")
    else:
        lines.append("未找到年资字段或可解析岗位要求，无法评估应届覆盖和职类 × 经验薪资分位。")
    if fields["district"]:
        distribution("行政区分布（独立字段）", texts[fields["district"]])
    elif fields["address"]:
        district_pattern = re.compile("|".join(DISTRICTS))
        def district(value):
            if not value:
                return "地址缺失"
            m = district_pattern.search(value)
            return m.group() if m else "地址存在但未识别深圳行政区"
        distribution("行政区分布（地址启发式识别；不展示详细地址）", texts[fields["address"]].map(district))
        city_hits = int(texts[fields["address"]].str.contains("深圳", regex=False).sum())
        address_nonempty = int((texts[fields["address"]] != "").sum())
        lines.append(f"非空地址 **{address_nonempty:,} 条**，其中字面包含‘深圳’ **{city_hits:,} 条**。这只是地址原文证据，不对缺失地址补城市，也不代表坐标/商圈已验证。")
        summary["address_missing"] = int((texts[fields["address"]] == "").sum())
    else:
        lines.extend(["### 行政区分布", "缺少独立行政区与地址字段，无法统计。"])
    if fields["business"]:
        distribution("商圈分布（独立字段）", texts[fields["business"]])
    else:
        lines.extend(["### 商圈分布", "没有独立商圈字段。地址中园区、楼宇、街道与商圈无法用一个可靠的切分规则互换，故不伪造商圈分布。后续接入可核验行政区/商圈词典后另报覆盖率，公开报告只保留聚合区级统计。"])
    for role, title in (("size", "公司规模"), ("industry", "公司行业")):
        if fields[role]:
            distribution(title + "分布（独立字段）", texts[fields[role]])
        else:
            lines.extend([f"### {title}分布", f"未识别独立{title}字段，无法输出有效分布；不能把职位分类推断成公司行业，不能由企业名称臆测规模。"])
    lines.append("### 技能字段、词典覆盖与 Top 50")
    source_columns = [fields[role] for role in ("skills", "title", "requirements", "description") if fields[role]]
    merged = [" ".join(texts[c].iloc[i] for c in source_columns) for i in range(n)]
    # ASCII 边界允许“熟悉Python开发”命中 Python，且避免 Java 命中 JavaScript。
    compiled = {skill: re.compile(pattern, re.I | re.ASCII) for skill, pattern in SKILLS.items()}
    counts = Counter()
    has_skill = 0
    multi_skill = 0
    for value in merged:
        found = [skill for skill, pattern in compiled.items() if pattern.search(value)]
        counts.update(found)
        has_skill += bool(found)
        multi_skill += len(found) >= 2
    lines.append((f"存在独立技能字段 `{fields['skills']}`。" if fields["skills"] else "**不存在独立技能标签字段**。") + f"本表为 {len(SKILLS)} 项固定词典在 {'、'.join(source_columns)} 上的启发式命中，每项技能每份 JD 只计一次，不是原始标签频数。至少命中 1 项的记录 **{has_skill:,}（{has_skill/n:.2%}）**，至少 2 项 **{multi_skill:,}（{multi_skill/n:.2%}）**。")
    lines.append(table(["技能词（规范名）", "命中 JD 数", "占全部 JD"], [[skill, count, f"{count/n:.2%}"] for skill,count in counts.most_common(50)]))
    lines.append("词典偏重技术职位，中文与英文同义词覆盖不全；提及、否定、‘优先’和岗位硬要求尚未区分。Spring/Spring Boot、SQL/MySQL、机器学习/深度学习有包含或上下位关系；共现统计先规范化、去重、标注证据跨度，并记录字典版本，避免把词频当成能力强弱。")
    summary.update(skill_hit_records=has_skill, skill_multi_records=multi_skill)
    lines.append("### 岗位描述长度与 3 个脱敏片段")
    if fields["description"]:
        desc = texts[fields["description"]]
        lengths = desc[desc != ""].str.len()
        lines.append(table(["描述字段", "非空数", "空值数", "平均字符数", "P50", "P90", "最大值"], [[fields["description"], len(lengths), n-len(lengths), f"{lengths.mean():.1f}" if len(lengths) else "—", f"{lengths.median():.0f}" if len(lengths) else "—", f"{lengths.quantile(.9):.0f}" if len(lengths) else "—", int(lengths.max()) if len(lengths) else "—"]]))
        lines.append("字符数按原文去首尾空白后计算，包含标点与换行。以下仅展示技能/职责短句，企业统一为‘某企业’，不展示行号、完整岗位原行或地址；仅用于观察文本，不是训练样本发布。")
        shown = 0
        for idx in desc[desc != ""].index:
            value = desc.loc[idx]
            clauses = [x.strip() for x in re.split(r"[。；;\n\r]+", value) if 15 <= len(x.strip()) <= 150 and re.search(r"熟悉|掌握|负责|开发|设计|算法|测试", x) and not re.search(r"公司|企业|集团|地址|电话|微信|联系|www\.|https?://|@|\d{7,}|[路街道楼栋座园]", x)]
            if not clauses:
                continue
            snippet = redact(clauses[0])[:140]
            shown += 1
            lines.append(f"{shown}. 某企业，职责/要求片段：{snippet}。")
            if shown == 3:
                break
        if shown < 3:
            lines.append(f"仅找到 {shown} 个满足严格脱敏条件的短句，其余省略。")
        summary["description_mean_chars"] = round(float(lengths.mean()),1) if len(lengths) else None
    else:
        lines.append("未识别岗位描述字段，无法构造描述正对或引用证据。")
    lines.extend(["### 数据能支持什么", table(["用途", "本表可用程度", "前置处理 / 限制"], [
        ["标题↔JD、技能↔JD 自监督对", "可作为领域适配弱正对，前提是对应字段非空", "先按企业和近重复 JD 分组切分；标题与描述只表达任务，不能代表真实 person-job fit 标签"],
        ["同职类 JD↔JD", "有职类字段时可挖候选正对", "同职类或同标题不自动为正：需级别、核心技能一致；不同方向/薪资不自动为真负例"],
        ["简历→JD 匹配", "当前无真实简历和投递/面试/录用标签", "真实 JD 约束的合成简历仅作弱监督；人工金标和真实反馈承担最终验证"],
        ["技能共现图谱", "可做带原文跨度的共现图", "去重后计算 PMI/支持度；共现不是技能等价或先修关系，推断边独立标记 hypothesis"],
        ["薪资硬过滤", "仅对已解析且双方同周期、有区间的记录可判定", "unknown 保留并解释；月薪和预计年薪双字段，日/时薪分池"],
        ["经验/学历硬过滤", "独立字段优先，复合要求需可追溯解析", "字面识别覆盖率不等于准确率；抽样人工验准，unknown 不误杀，优先项不作门槛"],
        ["城市/通勤过滤", "只按可核验地理字段；缺失项 unknown", "地址缺失不可默认为深圳或用户所在区；无经纬度无法承诺实际通勤时间"],
        ["市场薪资诊断", "可做同源招聘广告样本的分层区间参考", "无采集时间不能判断现时供给；偏技术样本不能冒充全深圳市场"],
    ])])
    return "\n\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="原始 xlsx 绝对路径；不存在即失败")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="中文 Markdown 报告路径")
    parser.add_argument("--source-label", default="用户指定的数据文件", help="明确数据身份，替代数据须在此说明")
    parser.add_argument("--environment-note", default="", help="可选：本次运行环境或依赖降级说明")
    args = parser.parse_args()
    if os.name != "nt" and PureWindowsPath(args.input).drive:
        parser.error(f"当前系统无法直接读取 Windows 路径 {args.input}。请用 --input 指定已经挂载或存在于当前机器的绝对路径；不会自动切换其他数据文件。")
    source = Path(args.input).expanduser()
    if not source.is_absolute() or not source.is_file():
        parser.error(f"输入必须是当前机器上存在的绝对文件路径：{source}。没有读取或替换数据源。")
    source = source.resolve()
    output = Path(os.path.abspath(Path(args.output).expanduser()))
    try:
        validate_output(source, output)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import pandas as pd
        import openpyxl
    except (ImportError, ValueError) as exc:
        print(f"读取依赖不可用：{exc}\n请在隔离环境安装兼容版本的 pandas 与 openpyxl；没有修改原始文件。", file=sys.stderr)
        return 2
    before = sha256(source)
    with pd.ExcelFile(source, engine="openpyxl") as book:
        # 原始 sheet 名可能含企业或联系方式，公开输出一律使用工作表编号。
        sheets = {f"工作表{index}": normalize_headers(book.parse(name).dropna(how="all")) for index, name in enumerate(book.sheet_names, 1)}
    after = sha256(source)
    if after != before:
        raise RuntimeError("读取前后 SHA-256 不一致：源文件可能被外部进程修改，拒绝生成报告。")
    sections, summaries = [], []
    company_values = [value for df in sheets.values() for column in df.columns if re.sub(r"__重复\d+$", "", column) in ALIASES["company"] for value in df[column].fillna("").astype(str).tolist()]
    redact_metadata = redactor(company_values)
    for name, df in sheets.items():
        if len(df):
            section, summary = profile_sheet(df, name)
            sections.append(section)
            summaries.append({"sheet":name, **summary})
        else:
            sections.append(f"## 工作表：{name}\n\n空表，无可剖析记录。")
    header = [REPORT_MARKER, "# 招聘数据只读剖析报告", f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}（UTC）。", f"**数据身份：{redact_metadata(args.source_label)}。**", f"本次实际读取（路径中的敏感信息如有则脱敏）：`{redact_metadata(source)}`。用户原请求文件是 `{DEFAULT_INPUT}`，仅当本报告的实际输入确为该文件或用户确认的映射副本时，才能将这里的统计归属于 merged_data.xlsx；不能仅凭规模相同就认定为同一数据集。", f"输入 SHA-256（读取前、后与报告写入后相同）：`{before}`。脚本只读原表，报告使用同目录临时文件原子替换；仅允许覆盖有本工具唯一标记的旧报告。", f"Python `{sys.version.split()[0]}`；pandas `{pd.__version__}`；openpyxl `{openpyxl.__version__}`。", redact_metadata(args.environment_note) if args.environment_note else "本次未记录额外环境异常。", "## 工作簿概览", f"共 **{len(sheets)} 个 sheet**，合计 **{sum(len(df) for df in sheets.values()):,} 条记录**。原始工作表名称不公开，按文件内顺序编号；各 sheet 独立剖析，未盲目拼表或跨表去重。", table(["工作表", "数据行数", "列数"], [[name,len(df),len(df.columns)] for name,df in sheets.items()])]
    footer = ["## 与原请求和仓库口径的关系", "当前报告的规模、字段、缺失率仅属于上文明确记录的实际输入。仓库 README 的 105 个职位类型、8 大类必须与原表实测分别比较；`data/job_data_meta.json` 的 316 列是派生向量表口径，不是原始 XLSX 字段数。未取得外部 merged_data.xlsx 时，不报告它的实际行数或薪资分布，也不宣称本报告替代其最终验收。", "## 复跑方法", "Windows PowerShell（在外部原文件所在机器执行，不复制原表入仓库）：", "```powershell\npython --version\npython -m venv e:\\file\\note\\profile\\.venv-profile\n& e:\\file\\note\\profile\\.venv-profile\\Scripts\\python.exe -m pip install pandas openpyxl\n& e:\\file\\note\\profile\\.venv-profile\\Scripts\\python.exe e:\\file\\note\\profile\\job-agent\\scripts\\profile_data.py --input 'E:\\file\\note\\profile\\merged_data.xlsx' --output 'E:\\file\\note\\profile\\job-agent\\docs\\DATA_PROFILE.md' --source-label '用户外部原始 merged_data.xlsx'\n```", "Linux 当前替代数据复跑（实际使用的隔离环境，不能据此声称读取 Windows 原表）：", "```bash\n/tmp/job-agent-profile-venv/bin/python /storage/xukunbo2/job-agent/scripts/profile_data.py \\\n  --input /storage/xukunbo2/job-agent/data/job_data.xlsx \\\n  --output /storage/xukunbo2/job-agent/docs/DATA_PROFILE.md \\\n  --source-label '仓库现有 job_data.xlsx：外部 merged_data.xlsx 不可达时的替代剖析，未验证两者同源'\n```", "部署到其他机器时先创建隔离环境并安装 pandas/openpyxl；脚本不绑定 /tmp 环境。去掉 --input 会尝试用户原 Windows 绝对路径，在 Linux 上明确报错，绝不悄悄换用仓库数据。", "## 提交边界", "报告仅含聚合统计与严格截短脱敏片段。原始 XLSX、完整 JD、公司明细、联络方式和真实简历不作为公开附件；本次没有移动/复制输入，没有执行 git add、commit 或 push。公开分享报告前仍需人工检查聚合维度与示例是否符合用户拥有的数据使用权限。"]
    atomic_write_report(source, output, "\n\n".join(header + sections + footer) + "\n", before)
    import json
    print(json.dumps({"input":redact_metadata(source),"output":redact_metadata(output),"sha256":before,"sheets":summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
