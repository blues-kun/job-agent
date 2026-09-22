# 深圳岗位数据集 · 公开处理版

本目录提供 **14,118 条岗位内容记录**，覆盖 **72 个职类、9 个上级分类和 6,950 个企业编号**。数据为项目已有的招聘广告历史快照，保留岗位职责、要求与薪资，用于需求分析、文本检索和人岗匹配研究。

当前已核实的规模约为 **1.4 万条岗位**。Excel、CSV 和旧向量表是同一批记录的不同表示，不能相加作为岗位总量。

## 下载

| 文件 | 格式 | 用途 |
|---|---|---|
| [job_data_public.xlsx](job_data_public.xlsx) | Excel，单工作表 | 查看岗位、直接接入平台 |
| [job_data_public.csv.gz](job_data_public.csv.gz) | UTF-8 BOM 压缩 CSV | pandas、数据分析和批处理 |
| [job_data_public.manifest.json](job_data_public.manifest.json) | JSON | 来源摘要、数据规模、处理统计、文件 SHA-256 |
| [job_data_public.validation.json](job_data_public.validation.json) | JSON | 格式一致性、独立复扫、平台加载和回归结果 |

文件页中的 **Download raw file** 可下载完整文件。两种数据格式包含相同的 14,118 行、9 列；`记录编号` 是公开版新增字段，其余列沿用岗位输入格式。

## 字段

| 字段 | 含义与处理 |
|---|---|
| 岗位名称 | 岗位标题；清理联系信息并规范字符 |
| 企业 | 本版本内稳定编号，例如 `企业00001`；同一企业共用编号，不发布映射 |
| 岗位薪资 | 保留广告原有表达，如区间、薪数、日薪或面议；不统一换算到手收入 |
| 岗位要求 | 学历、年资及岗位门槛文本，保留空值 |
| 岗位职责 | 职责及技能要求正文；清理联系信息、同名企业标识和详细地址提示句段 |
| 岗位地址 | 仅保留可识别城市与行政区；缺失或无法识别时保持空值 |
| 职位类型名称 | 原表职类，共72类 |
| 二级分类 | 原表上级分类，共9类 |
| 记录编号 | 本版本内唯一行编号，例如 `J000001`；不用于跨版本实体匹配 |

**记录数与平台去重口径：**14,118 行中有3条额外记录与其他行的“岗位名称＋企业＋薪资”相同，但职责版本不同。文件保留全部记录；当前平台直接读取 Excel 时按该业务键加载 **14,115 条**。企业编号保持一企一号，避免不同公司同名岗位被误合并。版本化研究应进一步区分内容版本与岗位家族。

## 在平台中使用

在仓库根目录安装 `requirements-platform.txt` 后，Linux / macOS 使用：

```bash
export JOB_AGENT_DATA="$PWD/data/job_data_public.xlsx"
python -m job_agent --port 8094
```

Windows PowerShell 使用：

```powershell
$env:JOB_AGENT_DATA = Join-Path (Get-Location) "data/job_data_public.xlsx"
.\.venv\Scripts\python.exe -m job_agent --port 8094
```

访问 `http://127.0.0.1:8094`。当前岗位加载器读取 XLSX，压缩 CSV 用于离线分析。未配置语义编码服务时，平台使用 BM25＋字符 TF-IDF。

**已有研究环境切换数据时：**先清除指向旧数据的 `JOB_AGENT_RESEARCH_DIR`、`JOB_AGENT_TEACHER_STUDY_DIR`、`JOB_AGENT_EXPERIMENT_DIR` 和 `JOB_AGENT_RANKER_DIR`；向量缓存使用新的 `JOB_AGENT_DENSE_DIR`，运行记录使用新的仓库外 `JOB_AGENT_PRIVATE_ROOT`。保持原有数据摘要检查，按公开版重新构建索引和研究材料。

## 在 Python 中分析

```python
import pandas as pd

jobs = pd.read_csv("data/job_data_public.csv.gz", dtype=str, keep_default_na=False)
print(jobs.shape)  # (14118, 9)
print(jobs["职位类型名称"].value_counts().head(10))
```

`pandas` 为离线分析的可选依赖。也可通过 Python 标准库 `gzip` 与 `csv.DictReader` 读取，无需解压到磁盘。

## 数据来源与处理

公开版只读导出自项目现有 `job_data.xlsx`，来源摘要见清单；该原表的统计与来源核对见 [数据剖析](../docs/DATA_PROFILE.md)。本次没有发现另一份十几万条的岗位原表，也未将外部 Windows `merged_data.xlsx` 认作已读取的数据。

导出时进行 Unicode 规范化、企业字段编码、地址降粒度，清理规则覆盖网址、邮箱及混淆写法、电话、社交账号提示、联系人句段和长号码；技术名称如 ASP.NET、Socket.IO 保留。表格以文本单元格写入，处理公式前缀，不继承原工作簿的作者、超链接、注释或隐藏内容。原文件保持不变，企业映射与用户简历不发布。

处理统计记录的是规则命中与字段变化，不是已确认的个人信息数量。去标识处理不保证无法通过岗位正文关联到公开招聘广告；请勿用于恢复联系人身份或联系方式。

公开版与原始研究快照具有不同的文件和文本摘要，清理会改变原文跨度与内容ID。旧评测标签、图关系、模型和索引需要重新校验，不能直接把旧实验指标归属于本版。数据没有可靠的采集时间与招聘有效状态字段，不代表当前在招供给。

本次发布未对第三方招聘原文另行授予许可证；使用与再分发应核对相应来源的授权条件。联系方式、个人简历、接口密钥、反馈日志和训练权重不在本数据包中。

## 校验与复跑

发布前完成两种格式的逐行一致性检查、独立联系信息复扫及平台加载验证。公开版按现有业务键加载14,115条，内置样例返回10个候选，引用规则校验通过；完整CPU回归187项通过。详细口径见验证记录。

```bash
# 校验文件摘要、行数、两种格式逐行一致性及联系信息规则。
python -m scripts.verify_public_data --directory data

# 从有权使用的原表重新导出；已有产物不会覆盖。
python -m scripts.export_public_jobs \
  --source /绝对路径/岗位原表.xlsx \
  --output-dir /绝对路径/新的公开版本目录
```

导出器仅接受经过核对的原始8列模式，新增字段须另行检查。复跑应保留新的来源和版本记录，不能沿用旧清单中的文件摘要。

本目录原有的 `job_data_meta.json` 是历史 Word2Vec 向量表元数据，`position_dictionary.txt` 是历史词典；两者不代表本次公开版的9列模式或实际职类数。
