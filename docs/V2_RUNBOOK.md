# 第二轮平台与研究运行手册

本文用于Linux服务器当前目录 `/storage/xukunbo2/job-agent`；实际原表只读，所有运行材料在仓库外。平台Python为 `/tmp/job-agent-profile-venv/bin/python`，GPU训练Python为 `/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python`。临时虚拟环境不保证重启后存在，可分别从 `requirements-platform.txt`、`requirements-research.txt`重建。

## 1. 本地平台

先启动编码器；请先检查端口，已有进程就复用，不重复启动：

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m scripts.serve_embeddings \
  --model-dir /home/xukunbo/.cache/job-agent/models/bge-small-zh-v1.5 \
  --device cuda:2 --port 8093
```

再启动平台：

```bash
JOB_AGENT_RESEARCH_DIR=/home/xukunbo/.cache/job-agent/research/v2/dataset-r4 \
JOB_AGENT_TEACHER_STUDY_DIR=/home/xukunbo/.cache/job-agent/research/v2/teacher-study \
JOB_AGENT_EXPERIMENT_DIR=/home/xukunbo/.cache/job-agent/research/v2/runs \
JOB_AGENT_DENSE_DIR=/home/xukunbo/.cache/job-agent/research/v2/dense \
JOB_AGENT_EMBEDDING_URL=http://127.0.0.1:8093 \
JOB_AGENT_COACH_URL=http://127.0.0.1:8092 \
JOB_AGENT_PRIVATE_ROOT=/home/xukunbo/.cache/job-agent/runtime-v2 \
OPENBLAS_NUM_THREADS=2 \
/tmp/job-agent-profile-venv/bin/python -B -m job_agent --port 8094
```

访问 `http://127.0.0.1:8094`。远程IDE需转发8094端口。服务仅绑定本机；不要直接当多用户公网服务使用。

本机8092现已启用已有Qwen3-4B指令模型，负责白名单行动选择和原段顺序；接口标注仍是用户指定的GPT模型。没有本地指令模型时删去`JOB_AGENT_COACH_URL`，页面会明确使用规则降级。若需恢复本地模型，先检查端口再启动：

```bash
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m scripts.serve_coach \
  --model-dir /storage/xukunbo2/hy-agent/models/Qwen3-4B-Instruct-2507 \
  --device cuda:3 --port 8092
```

正常流程：查看需求→选择虚构样例或上传简历→解析并确认画像→推荐→目标岗位详情→记录补证行动→明确同意保存单条证据→确认新增表述→查看差异并重新确认画像→再次匹配。公司排除可在成长记录中撤销；清空记录只清理当前会话。完整简历不存库，明确保存的单条证据和研究事件有效期为30天，反馈曝光有效期为24小时。过期材料不再返回或参与授权；数据库过期行在应用启动、推荐或成长记录访问时按对应路径清理，停机期间不会自动物理删除。

API调用也须先 `/api/v2/profile/preview`，核对返回的有效偏好，再以 `user_confirmed=true` 调 `/api/v2/profile/confirm`。使用返回的 `confirmation_token` 调推荐等接口。更改正文、偏好、解析版本或会话后旧令牌失效。不要在自动化里把未知用户确认伪装成已确认；验收脚本的主动确认只适用于完全虚构样例。

## 2. 版本化数据与固定特征回放

```bash
python -m research.refresh_semantics \
  --source /home/xukunbo/.cache/job-agent/research/dataset-v1/snapshot-c88558a0d732-seed42-b647fc087de6 \
  --output /仓库外/新的语义快照

JOB_AGENT_EMBEDDING_URL=http://127.0.0.1:8093 python -m research.replay_teacher_study \
  --study /home/xukunbo/.cache/job-agent/research/v2/teacher-study \
  --dataset /仓库外/新的语义快照 --source /绝对路径/岗位.xlsx \
  --dense-dir /仓库外/向量缓存 --output /仓库外/新的回放版本
```

已有r4快照与回放不要覆盖。模板或解析源码改变后必须新建快照/索引/回放；不能只改清单中的版本名。旧教师任务只在原始JD、画像正文与偏好、岗位版本、家族和分区未变时沿用，保留原父来源hash。

## 3. 接口标注与自动执行

接口只用于本轮明确授权的私有研究任务，不自动发送平台用户上传的真实简历。配置路径以命令参数传入；代码不source环境文件、不复制密钥、不打印密钥。

后台已经有本轮运行时，先读 `teacher-study/supervisor_status.json`、`supervisor.log` 及进程PID文件，避免重复发起付费请求。恢复命令：

```bash
/tmp/job-agent-profile-venv/bin/python -B -m research.annotation_supervisor \
  --study /home/xukunbo/.cache/job-agent/research/v2/teacher-study \
  --config /home/xukunbo/code/agent/agent/.env \
  --dataset /home/xukunbo/.cache/job-agent/research/v2/dataset-r4 \
  --replay /home/xukunbo/.cache/job-agent/research/v2/replay-r4 \
  --output /home/xukunbo/.cache/job-agent/research/v2/runs/teacher-v2 \
  --training-python /storage/xukunbo2/venvs/qwen3-grpo-train/bin/python \
  --gpu 2 --max-rounds 3 --annotation-concurrency 1 --extraction-concurrency 1
```

该监督进程有单实例锁，保留每个通道的输入hash和结果，最多3轮恢复，定期发布校验后的工作台摘要，刷新进度不会移除已发布指标。所有标签齐备、引用规则通过、等级确定后，才进入独立派生引用纠正、全量抽取评测、冻结、LambdaRank、人岗LoRA、三种子图对照及共同池评分。若达到有限重试后仍有失败，状态为 `needs_review`，保留失败材料，不静默删样本、不补标签。研究训练不会自动上线。

HTTP429会使两个队列共享暂停，监督器写`rate_limited`后退出，不自动进入下一轮。遵守服务商`Retry-After`；缺失时默认300秒。先看`resume_not_before`，冷却到期后再用上述1＋1并发命令恢复；仍429应继续保留断点并等待接口恢复，不能反复启动造成请求风暴。冷却时间只是最早尝试时间，不保证第三方已恢复。

单独恢复某条标注队列时，必须先确认没有监督进程正在写同一目录：

```bash
python -m research.api_annotation \
  --tasks /仓库外/teacher-study/tasks.jsonl --config /仓库外/接口配置.env \
  --output /仓库外/teacher-study/annotation --batch-size 2 --concurrency 1
```

断点恢复不能改变任务清单、批次大小或模型。执行次序可以按画像轮转，但批号和输入摘要保持一致。原始API输出保存在私有目录；公开报告只取聚合数字和虚构反例。

## 4. 人工评审与仲裁

需求抽取可先做明确标为进度的分析；移除`--allow-partial`后必须所有任务严格通过才出全量报告。输出新目录，保留旧快照：

```bash
python -m research.evaluate_requirements \
  --study /home/xukunbo/.cache/job-agent/research/v2/teacher-study \
  --dataset /home/xukunbo/.cache/job-agent/research/v2/dataset-r4 \
  --annotation /home/xukunbo/.cache/job-agent/research/v2/teacher-study/extraction-annotation \
  --output /仓库外/新的抽取评测版本 --allow-partial
```

工作台显示原始任务材料，屏蔽模型答案。评审者主动声明独立阅读，提交当前任务hash；代号仅用于记录，不认证真人身份。

```bash
JOB_AGENT_TEACHER_STUDY_DIR=/仓库外/teacher-study python -m research.workbench_reviews export \
  --dataset /仓库外/语义快照 --private-root /仓库外/平台私有目录 --output /仓库外/新的评审导出

python -m research.workbench_reviews review-status \
  --bundle /仓库外/新的评审导出 --registry /仓库外/真实核实的评审登记.json \
  --output /仓库外/新的分歧报告

python -m research.workbench_reviews freeze \
  --bundle /仓库外/新的评审导出 --registry /仓库外/真实核实的评审登记.json \
  --decisions /仓库外/人工明确决议.jsonl --output /仓库外/新的人工冻结集
```

导出会生成未核实的registry模板，不能直接把模板中的 `false` 批量改为 `true` 冒充真实确认。两人同意仍需显式仲裁；只有实际人类完成登记和决议后才能声称人工金标。模型标注继续走单独的 `freeze-model --allow-model-labels` 路径。

## 5. 训练与服务连接

`research/run_teacher_experiments.py` 只消费完整可验证弱标和固定回放，输出模型、源hash、参数、指标及逐查询排名到私有目录。学习排序可通过 `JOB_AGENT_RANKER_DIR=/仓库外/.../ranker` 加载影子分数；服务检查特征、原表、模板与编码器一致性。默认推荐顺序仍由已验收规则给出。

Qwen服务可使用 `scripts.serve_embeddings --adapter /仓库外/.../adapter`；它会形成新的编码器合同和向量索引，不能沿用BGE或旧adapter缓存。不要给不同编码器的服务装入原BGE特征训练的排序器。

GNN默认是离线研究模块。要上线必须另有共同人岗评测、输入一致性和线上时延依据；本轮不会因为某次教师分数更高就自动切换。

## 6. 验收与故障恢复

除端到端验收外，可单独核对HTTP推荐与离线回放的35维特征。以下脚本仅使用平台明确提供的虚构样例，不读取教师答案；需要与服务相同的岗位版本及编码器：

```bash
JOB_AGENT_EMBEDDING_URL=http://127.0.0.1:8093 OPENBLAS_NUM_THREADS=2 \
/tmp/job-agent-profile-venv/bin/python -B -m scripts.verify_feature_parity \
  --url http://127.0.0.1:8094 \
  --source /storage/xukunbo2/job-agent/data/job_data.xlsx \
  --dataset /home/xukunbo/.cache/job-agent/research/v2/dataset-r4 \
  --dense-dir /home/xukunbo/.cache/job-agent/research/v2/dense \
  --output /仓库外/新的特征一致性验收.json
```

```bash
/tmp/job-agent-profile-venv/bin/python -B scripts/verify_platform.py \
  --url http://127.0.0.1:8094 \
  --source /storage/xukunbo2/job-agent/data/job_data.xlsx \
  --dataset /home/xukunbo/.cache/job-agent/research/v2/dataset-r4 \
  --output /仓库外/新的平台验收.json
```

模型服务中断时推荐显式回退字符TF-IDF，不能把回退结果称为BGE。索引或模型合同不一致时先恢复对应服务，再重启平台；不要绕过校验。接口失败检查私有日志里的错误类别和HTTP状态，禁止打印完整配置文件或第三方可能回显凭据的错误正文。

当前CPU环境没有Torch，按环境分别运行平台与模型测试；不要为测试方便升级用户已有GPU环境。全部源码验证命令和结果应与本次产物一起保留。浏览器视觉验收与公网部署不在本轮API验收结论内。
