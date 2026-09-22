# 48 份虚构简历画像的真实向量召回运行

2026-09-22 已在 GPU 3、BF16、seed 42 下，分别运行冻结 Qwen3-Embedding-0.6B 与已有 LoRA，生成每份画像的 top30 岗位 ID、余弦相似度及查询延迟。本次没有训练，没有调参，没有计算 nDCG、Recall 或任何需要相关性金标的指标。

这证明产物能够接受简历格式的输入并完成检索。**48 份画像均为完全虚构、尚未人工审阅的模板，不能据此声称真实人岗匹配有效或微调优于基座。**

## 输入、候选和任务指令

输入 benchmark 为 `benchmark-8b1795a66eaa1849`，共开发 12 份、测试 36 份，查询取 `text` 与 `preferences.intent`。候选仅来自同一分区，每个岗位家族取字典序最小的 `job_id`，与既有人工池使用同一代表口径。开发候选宇宙 2,091 条、测试 2,093 条。

正文构造直接调用训练产物冻结脚本中的去标题正文函数，未经重写。开发候选中有 1 条完全空正文：保留其候选宇宙记录，但不为其编造向量或填入标题，不参加余弦排名，等价于排在全部有效文档之后。因此实际可编码开发文档 2,090 条、测试文档 2,093 条；非空短正文正常编码。

训练任务指令是“根据目标职位名称和岗位类别，检索职责与技能要求相符的招聘描述。”本次改为“根据简历中的经历、技能与求职意向，检索职责和任职要求相符的招聘描述。”使用官方 `Instruct: ...\nQuery: ...` 包装、最后有效 token 池化、L2 归一化、精确余弦排序。指令变更属于未经额外训练验证的任务分布变化。

## 人工池覆盖缺口

既有人工池每查询 30 条、共 1,440 条任务，标签全部为空，qrels 文件为空。因此**池内也没有金标；池外数量只能解释为需要补充审阅的候选数量，不能将池外视作负例**。

| 模型 | 开发 top10 池外对数 | 测试 top10 池外对数 | 合计池外对数 | top10 全部未标注对数 |
|---|---:|---:|---:|---:|
| 冻结 Qwen | 78 / 120 | 254 / 360 | 332 / 480 | 480 / 480 |
| Qwen + LoRA | 91 / 120 | 241 / 360 | 332 / 480 | 480 / 480 |

两个模型池外 top10 的并集有 **541 个不同的查询—岗位对**，已写入 `supplemental_top10_review_candidates.jsonl`，标签仍为空。没有改写原有 benchmark 或人工池。后续可按固定规则合并补标，给标注者展示时应隐藏模型来源、分数和排名。

## 延迟与核验

单查询平均延迟：冻结基座 36.91 毫秒、LoRA 43.99 毫秒。包含查询编码、设备同步、CPU 余弦计算与稳定排序；不包含模型加载和预先编码候选库的时间。开发/测试候选库编码分别为：基座 15.53 / 15.11 秒，LoRA 17.23 / 17.50 秒。

这些是当前共享 GPU 环境的一次测量，不能据此作严谨的速度优劣结论。全程 PyTorch 峰值分配显存 1,371,862,016 字节，峰值保留 1,491,075,072 字节，不包含全部驱动开销。

独立结构核验通过：96 条查询—模型结果、2,880 条 top30 排名全部具有合法 ID、去重家族、有限且单调排序的余弦分数；源文件和产物 SHA 校验一致；没有补造人工标签。结果见 `verification.json`。

本程序没有应用薪资、学历等硬过滤；对信息缺失、指令注入等压力画像仍生成待审阅排名，不代表产品应直接推荐。产品决策仍需独立拒答/追问和事实核验层。

## 复跑

脚本：`/tmp/embedding/retrieve_resumes.py`，副本保存在本目录的 `frozen_retrieve_resumes.py`。SHA256：`d995888d0176feaa4939795e967bc20f51be3e570661dedb032301fc83a838f6`。

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B \
  /tmp/embedding/retrieve_resumes.py \
  --jobs-jsonl /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6/jobs.jsonl \
  --benchmark-dir /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6/benchmark/benchmark-8b1795a66eaa1849 \
  --training-run /home/xukunbo/.cache/job-agent/research/runs/embedding-lora-seed42 \
  --output-dir /home/xukunbo/.cache/job-agent/research/runs/resume-retrieval-v1-复跑新目录 \
  --gpu 3 --seed 42 --batch-size 8 --max-length 512 --top-k 30
```

输出路径必须在 Git 工作区外且为空。程序验证岗位快照、benchmark、冻结训练代码、官方基座权重和 adapter 权重后才执行，不修改输入文件。
