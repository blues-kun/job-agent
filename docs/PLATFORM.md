# 本地平台：岗位需求 → 简历匹配 → 证据诊断 → 整理后再匹配

> 历史阶段记录：保留当时的参数、端口和验证状态。当前实现以 [第二轮实施记录](V2_IMPLEMENTATION.md) 与 [仓库首页](../README.md) 为准。

> 2026-09-22更新：本文保留首版本的运行背景；最新版本已接入领域LoRA研究、图实验、文件解析、本地LLM及标注工作台。请优先阅读 [实施结果](IMPLEMENTATION_RESULTS.md) 和 [最新运行手册](RESEARCH_RUNBOOK.md)。下文“没有调用生成LLM/没有向量微调”等描述仅对应旧版。

## 当前实现与产品顺序

本版本把用户提出的顺序做成三个可操作页面：

1. **岗位需求**：在真实Excel快照上按职位方向筛选；展示技能提及、经验结构、广告月薪中点、区域分布与替代技能组原句。
2. **简历匹配**：下拉选择7份完全虚构样例，或粘贴/导入UTF-8文本；修改城市、方向、学历、年资、最低月薪和区域偏好，运行推荐。信息不足时先追问。
3. **算法对比**：使用同一简历和快照实际运行BM25、多路召回、多路召回加证据精排，显示返回数、技能证据覆盖、硬违规、公司数、耗时和前三项。

岗位详情支持双侧引用、未证明技能、替代组、硬条件状态、同职类同最低年资薪资参考、原文查看、事实保全整理、审核后重新匹配及面试练习。感兴趣/跳过按真实曝光记录；跳过不直接转成负标签。

当前数据默认来自仓库既有 `data/job_data.xlsx`：14,118原始记录，按岗位名＋企业＋薪资去重后14,115。外部 `merged_data.xlsx` 仍未在服务器取得，不能把当前数据称为该文件的实测结果。原表只读，模型与向量缓存均在仓库外。

## 运行方法

### 不需要GPU的完整本地流程

创建隔离环境后，在仓库根目录执行：

```bash
python -m venv /tmp/job-agent-platform-venv
/tmp/job-agent-platform-venv/bin/python -m pip install -r requirements-platform.txt
/tmp/job-agent-platform-venv/bin/python -m job_agent --port 8090
```

打开 `http://127.0.0.1:8090`。若IDE连接远程服务器，转发服务器的8090端口后在本机打开。无需Node构建或外部API密钥。未启用向量编码器时使用 **BM25＋字符TF-IDF**，界面明确标注这一模式，不能称为预训练语义向量。

本次已可用的隔离环境是 `/tmp/job-agent-profile-venv`。`/tmp`环境不是持久部署保证，可按上述命令重建。没有修改系统Python或已有vLLM的依赖。

Windows PowerShell（Python3.11或3.12，环境放在仓库外）：

```powershell
python -m venv E:\file\note\profile\.venv-job-platform
& E:\file\note\profile\.venv-job-platform\Scripts\python.exe -m pip install -r requirements-platform.txt
$env:JOB_AGENT_DATA = 'E:\file\note\profile\merged_data.xlsx'
$env:JOB_AGENT_PRIVATE_ROOT = 'E:\file\note\profile\job-agent-private'
& E:\file\note\profile\.venv-job-platform\Scripts\python.exe -m job_agent --port 8090
```

显式设置的数据路径不存在时会启动失败，不会悄悄切换其他文件。列名别名目前覆盖现有表和常用招聘字段；新原表接入后仍须先运行 `scripts/profile_data.py` 核对schema。

### 真正的BGE向量召回

提供独立的本地编码进程，服务只绑定127.0.0.1，与已有GPU服务和训练环境分离。使用 `BAAI/bge-small-zh-v1.5` 验证完整向量链，固定模型revision为 `7999e1d3359715c523056ef9478215996d62a620`；512维，CLS池化、L2归一化、最多512token。基座选择与池化依据[官方模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)。这是未微调基线，不是原方案中的Qwen领域微调结果。

模型目录需要官方快照中的 `config.json/tokenizer_config.json/special_tokens_map.json/vocab.txt/tokenizer.json/model.safetensors`，以及内容为对应revision的 `revision.txt`。只加载safetensors和本地可信模型配置，不启用remote code。模型下载到仓库外；正式复现实验应同时记录下载文件SHA-256。

本机运行编码器（GPU0，既有GPU1服务保持独立）：

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B scripts/serve_embeddings.py \
  --model-dir /home/xukunbo/.cache/job-agent/models/bge-small-zh-v1.5 \
  --device cuda:0 --port 8091
```

另一个终端启动平台：

```bash
JOB_AGENT_DENSE_DIR=/home/xukunbo/.cache/job-agent/dense \
JOB_AGENT_EMBEDDING_URL=http://127.0.0.1:8091 \
/tmp/job-agent-profile-venv/bin/python -B -m job_agent --port 8090
```

首次启动按64条一批编码岗位并生成索引；缓存键包含岗位快照、ID顺序、文本hash、模型revision、池化和模板版本。后续加载缓存。查询端检查编码器revision，避免模型切换后混用坐标空间；编码器在启动或运行时不可用均显式回退到字符TF-IDF。启动时降级后需重启平台才能重新加载向量索引。编码器端点只允许本机地址，不把用户材料自动发给外部模型。

本版本采用一个岗位一个向量、512token截断和NumPy精确内积。长JD尾部可能被截断，分块与更大基座仍是后续对照项；不能把本版描述成已经实现REDESIGN的所有分块训练设计。

## 算法与创新边界

| 层 | 已实现 | 要验证的问题 |
|---|---|---|
| 硬条件 | 城市、最低月薪、学历、年资；pass/fail/unknown；未知可严格排除 | 解析错误和“优先/不限”局部修饰，岗位类型与经验不限的混杂 |
| 召回 | 中文jieba BM25 top100；BGE top120或字符TF-IDF；结构化意图top80；RRF常数60 | 不同路由是否增加人工确认的相关岗位 |
| 精排 | 0.35检索融合＋0.35证据技能覆盖＋0.20意图＋0.05硬字段可确认率＋0.05区域偏好；标题/职类未命中具体意图时乘0.55，不以“开发”等通用词冒充方向命中 | 这是可解释规则，不是训练过的LightGBM/XGBoost |
| 需求逻辑 | 局部“或/任选一种”的多选一组；优先项减权；任职要求与职责独立引用 | 简单共现误报缺口是否减少，AND/OR复杂嵌套仍不支持 |
| 能力证据 | 否定、提及、了解、熟悉、使用经历与精通自述分级 | 词面证据不是实际掌握；要对抗“项目：关键词清单” |
| 解释核验 | ID/快照、字段、起止位置、双侧引用；同公司≤2、同JD正文去重 | 当前理由来自受控模板；不是任意LLM主张的完整语义核验器 |
| 简历迭代 | 整段保全、指代/时序关系保守处理、用户审核、重新匹配 | 整理表达不能冒充新增能力，更不能为了排名编造经历 |

研究主张可以收敛为：**显式需求逻辑与简历证据分级能否减少人岗匹配中的假缺口，并抵抗技能堆砌。** 它目前是可测假设，不是已证明的科研创新或SOTA。先建立OR组与否定/熟练度金标，再比较去掉OR识别、去掉证据权重、仅BM25、加入冻结BGE等消融。

当前对比页的技能覆盖由排序器同一规则计算，天然存在优化自身代理指标的偏向。不能拿该表证明真实相关性提升；必须引入独立人工qrels，再计算nDCG/MRR/Recall。完整协议见 [REDESIGN.md](REDESIGN.md)。

## 智能体框架与技能

本版使用显式状态机：解析 → 必要追问 → 硬条件 → 多路召回 → 证据精排 → 引用核验。每次返回run_id、画像版本、快照、步骤状态、召回来源及特征，便于调试和评测。

四个技能包位于 `skills/`：resume-diagnose、gap-analysis、resume-rewrite、mock-interview。路由器只允许调用白名单技能，按需读取对应SKILL.md并记录版本hash；岗位/简历中的指令不能决定工具调用。技能目录和触发规范使用本次的skill-creator流程创建并验证。

页面还提供检测到WebMCP支持时才注册的三个工具：读取聚合需求、选择虚构样例、匹配当前编辑器。工具通过同一前端流程更新可见状态；没有改写自动应用或反馈提交工具。当前环境没有支持WebMCP的浏览器验证入口，这三个工具尚未完成运行验收，普通手动流程不依赖它们。

**没有调用生成式LLM，没有训练新排序器或微调向量。** 简历整理当前是确定性段落整理，面试问题是带证据的模板。后续LLM只负责有事实清单约束的措辞与更细致诊断，接入后必须用独立核验器评估，不以界面中“智能体”名称冒充模型能力。

## 数据与反馈边界

API只绑定127.0.0.1；默认拒绝跨Origin修改、限定Host；前端文本转义、无外部CDN、无简历浏览器持久化。API不打印简历正文。岗位联系方式在展示字段中隐藏。

反馈与曝光保存在 `JOB_AGENT_PRIVATE_ROOT/feedback.sqlite3`，默认 `~/.cache/job-agent`，不允许放进仓库。仅保存会话ID、运行ID、岗位ID、快照、展示位置、偏好和时间，**不保存简历正文**。签名密钥在同一私有目录，Cookie为HttpOnly和SameSite Strict。记录在启动与推荐请求时清理超过24小时的数据；停机期间不会定时执行删除，下次启动清理。界面可立即删除当前会话记录。

这是单机研发平台，不是多租户生产部署。远程共享前仍需身份认证、TLS、速率限制、任务队列、数据许可核对和运维监控；当前不把服务绑定到公网，也不发布真实数据。

## 验证和复跑

```bash
/tmp/job-agent-profile-venv/bin/python -m pip install pytest
/tmp/job-agent-profile-venv/bin/python -B -m pytest tests/test_platform.py tests/test_repository_hygiene.py -q
```

针对真实故障模式验证学历修饰、三选一、否定/熟练度、任职要求引用、薪资周期、未知地址、注入/关键词清单、排序硬过滤、改写指代、会话隔离、幂等反馈及路径保护。API真实快照的运行结果见 [PLATFORM_RESULTS.md](PLATFORM_RESULTS.md)。界面做了HTML资源和JavaScript静态检查；未执行浏览器截图或点击验收，不声称已完成视觉验收。

依赖按本次隔离环境锁在 `requirements-platform.txt`，API静态资源挂载参考[FastAPI官方文档](https://fastapi.tiangolo.com/tutorial/static-files/)。原业务入口和历史模型保留，新的主入口为 `python -m job_agent`，不要用旧 `web/unified_server.py` 启动本平台。
