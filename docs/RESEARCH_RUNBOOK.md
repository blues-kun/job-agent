# 研究平台运行手册

本手册对应2026-09-22的实际Linux环境。所有命令在仓库根目录执行；输出目录必须在仓库外。不会自动提交Git、下载真人简历或发布岗位数据。已有同名实验不覆盖，复跑请更换输出目录。

## 一、已生成的私有资产

| 用途 | 路径 |
|---|---|
| 规范数据版本 | `/home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6` |
| 基础评测池 | 上述目录下 `benchmark/benchmark-8b1795a66eaa1849` |
| 私有结构化岗位库 | `/home/xukunbo/.cache/job-agent/research/catalog-v1.sqlite3` |
| BGE图输入特征 | `/home/xukunbo/.cache/job-agent/research/features-v1.npz` |
| 模型与实验根目录 | `/home/xukunbo/.cache/job-agent/research/runs` |
| Qwen领域LoRA | 实验根目录下 `embedding-lora-seed42/adapter` |
| 图模型 | 实验根目录下 `graph-{graphsage,pool_mlp,random_edges}-seed{42,43,44}` |
| 简历格式向量召回 | 实验根目录下 `resume-retrieval-v1` |
| 20份LLM合成试点 | 实验根目录下 `resume-synthesis-pilot/pilot-3208390cd57b4247` |
| 人工标注账本 | `/home/xukunbo/.cache/job-agent/research_annotations.sqlite3` |
| 产品反馈账本 | `/home/xukunbo/.cache/job-agent/feedback.sqlite3` |
| 两分钟说明视频 | `/home/xukunbo/.cache/job-agent/demo/job-agent-120s-v2.mp4` |

这些目录可能含原始岗位、企业或关联信息，不能整目录复制到公开仓库。对外发布仅用聚合报告和明确虚构的脱敏材料。

## 二、当前服务器启动

平台环境：`/tmp/job-agent-profile-venv/bin/python`。训练环境：`/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python`。后者已有torch2.9.0+cu128、transformers5.13.0、peft0.20.0，本轮没有升级其依赖。GPU0/1已有其他任务，本轮自己的服务用GPU2、向量训练用GPU3；以后启动前仍需查看实际空闲情况。

终端一：冻结BGE服务。

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B scripts/serve_embeddings.py \
  --model-dir /home/xukunbo/.cache/job-agent/models/bge-small-zh-v1.5 \
  --device cuda:2 --port 8091
```

终端二：本地指令模型。该路径是本机已有权重，另一机器需要自行准备适配的可信本地模型。

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B scripts/serve_coach.py \
  --model-dir /storage/xukunbo2/hy-agent/models/Qwen3-4B-Instruct-2507 \
  --device cuda:2 --port 8092
```

终端三：完整研究工作台。

```bash
/tmp/job-agent-profile-venv/bin/python -B -m scripts.run_research_platform \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --benchmark /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6/benchmark/benchmark-8b1795a66eaa1849 \
  --synthesis /home/xukunbo/.cache/job-agent/research/runs/resume-synthesis-pilot/pilot-3208390cd57b4247 \
  --retrieval-review /home/xukunbo/.cache/job-agent/research/runs/resume-retrieval-v1 \
  --dense-cache /home/xukunbo/.cache/job-agent/dense-v2 \
  --coach-url http://127.0.0.1:8092 --port 8090
```

没有GPU服务时，去掉 `--dense-cache` 和 `--coach-url`：平台使用BM25＋字符TF-IDF和确定性整理。界面/返回值会标注降级，不能将其称为预训练向量或LLM输出。省略标注材料参数只隐藏对应材料，不伪造标签。

这些进程没有注册为系统服务，服务器重启或终端任务被清理后需重启。当前不配置公网反向代理，也不保证跨会话永久在线。

## 三、重建数据与实验

安装新的平台/研究环境时使用 `requirements-platform.txt` / `requirements-research.txt`；GPU训练依赖请单独配置，不要直接覆盖已有模型服务环境。

```bash
# 从只读原表建立新版本，输出子目录名由数据及规则摘要生成。
/tmp/job-agent-profile-venv/bin/python -B -m research.data_pipeline \
  --repo-root /storage/xukunbo2/job-agent \
  --input /storage/xukunbo2/job-agent/data/job_data.xlsx \
  --output-root /home/xukunbo/.cache/job-agent/research/dataset-v1
```

代码或解析规则发生变化可能生成新版本目录，这是预期行为；不同版本不混用ID、特征或标注。向量与图训练命令分别见 [向量实测报告](EMBEDDING_RESULTS.md) 和下例：

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m research.train_graph \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --features /home/xukunbo/.cache/job-agent/research/features-v1.npz \
  --output /home/xukunbo/.cache/job-agent/research/runs/graph-new-seed42 \
  --mode graphsage --device cuda:2 --epochs 5 --batch-size 256 --seed 42 --eval-queries 3000
```

图指标是自身文本对齐，不能替代人工人岗指标。实际九组训练时源码保存在实验根目录的冻结快照；当前脚本的默认仓库根定位已改成相对文件路径，便于迁移，不能把当前文件hash误称为旧运行hash。

```bash
# 生成新的基础评测材料；包含48份虚构场景、空qrels与候选池。
/tmp/job-agent-profile-venv/bin/python -B -m research.build_benchmark \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --repo-root /storage/xukunbo2/job-agent \
  --features /home/xukunbo/.cache/job-agent/research/features-v1.npz
```

第一次构建的命令参数会决定版本名，已存在且一致时复用；不一致不会覆盖。具体可选参数以各脚本 `--help` 为准。生成试点入口 `python -m research.synthesize_resume_pilot` 默认固定20个训练家族，未经人审不扩成正式训练集。

私有SQLite物化复跑（不覆盖已存在库）：

```bash
/tmp/job-agent-profile-venv/bin/python -B -m research.export_catalog \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --output /home/xukunbo/.cache/job-agent/research/catalog-new.sqlite3 \
  --repo-root /storage/xukunbo2/job-agent
```

数据库含字段索引与证据视图，目前用于研究材料物化与SQL分析，平台在线检索仍从JSONL加载。

## 四、人工标注与监督训练

研究页选择评审代号和任务类型。相关性任务阅读岗位地点、薪资、门槛、正文及简历；需求任务填写多组结构标签，quote必须复制自当前片段。两个评审独立完成后由真人仲裁，记录标签版本、评审标识和依据。

导出原始人工输入：

```bash
/tmp/job-agent-profile-venv/bin/python -B -m research.export_annotations \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --private-root /home/xukunbo/.cache/job-agent \
  --output /home/xukunbo/.cache/job-agent/research/annotation-export-new
```

若需要导出额外合成与补审任务，须同时设置与平台一致的 `JOB_AGENT_SYNTHESIS_DIR`、`JOB_AGENT_RETRIEVAL_REVIEW_DIR`，否则导出器只识别基础任务。输出是未仲裁材料；κ只描述代号一致性，需核实是真正两人独立标注后才能作为人工可靠性结果。

正式排序标签行需要 `query_id/job_id/grade/label_source=human_adjudicated/reviewer_ids/adjudication_id`，画像需要 `query_id/profile_family_id/split/text/preferences`。train和dev都要有已审核画像；目前48份开发/测试场景不能拿去补训练分区。

```bash
# 以下输入目前不存在；须完成真实审核后准备，不能自动填审核人跳过人审。
/tmp/job-agent-profile-venv/bin/python -B -m research.prepare_ranker \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --profiles /home/xukunbo/.cache/job-agent/research/reviewed-profiles.jsonl \
  --qrels /home/xukunbo/.cache/job-agent/research/adjudicated-qrels.jsonl \
  --output /home/xukunbo/.cache/job-agent/research/ranker-reviewed-features.jsonl

/tmp/job-agent-profile-venv/bin/python -B -m research.train_ranker \
  --input /home/xukunbo/.cache/job-agent/research/ranker-reviewed-features.jsonl \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --output /home/xukunbo/.cache/job-agent/research/runs/ranker-human-v1
```

生成SFT的输入合同、assistant掩码及命令见 [生成训练手册](GENERATION_TRAINING.md)。默认只校验，显式`--mode train`且审核门槛通过才加载模型；当前没有可用的人审训练集。

## 五、评测与演示复跑

`research/score_rankings.py` 接受一个或多个模型排名、全部候选任务及已仲裁qrels。当前空qrels执行结果为null；不能用0标签替换缺失。`research/evaluate_evidence.py`执行虚构挑战集双通道评测；默认调用本机8092，`--rules-only`只执行规则并明确没有judge。

```bash
/tmp/job-agent-profile-venv/bin/python -B scripts/verify_platform.py \
  --dataset /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --output /home/xukunbo/.cache/job-agent/research/runs/platform-new/verification.json

# 两分钟静态中文字幕说明视频，非浏览器录屏；不带音轨。
/tmp/job-agent-profile-venv/bin/python -m pip install -r requirements-demo.txt
/tmp/job-agent-profile-venv/bin/python -B -m scripts.render_demo \
  --output /home/xukunbo/.cache/job-agent/demo/job-agent-new-120s.mp4 \
  --embedding-result /home/xukunbo/.cache/job-agent/research/runs/embedding-lora-seed42/run_analysis.json
```

针对性测试：

```bash
/tmp/job-agent-profile-venv/bin/python -B -m pytest \
  tests/test_platform.py tests/test_repository_hygiene.py \
  tests/test_research_data.py tests/test_research_benchmark.py \
  tests/test_research_synthesis.py tests/test_research_runtime.py tests/test_research_catalog.py -q

/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m unittest discover -s tests -p 'test_research_graph.py'
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m unittest discover -s tests -p 'test_research_embedding.py'
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m unittest discover -s tests -p 'test_research_generation.py'
```

两类环境分开是为避免升级共享GPU依赖。旧仓库的所有历史模型并未重新训练，旧入口仍保留；新版入口不依赖旧XGBoost合成标签。
