# 有人工仲裁门槛的生成 LoRA 训练入口

`research/` 目录提供抽取、诊断、简历改写任务的 SFT 续接工具。**本次只完成代码和 CPU/tokenizer 校验，没有真实人工仲裁训练集，没有运行 SFT，没有生成可上线的生成适配器。** 不将合成数据填上虚构审核信息后开训，也不宣称训练 loss 能证明简历事实正确。

默认 `--mode validate` 仅检查数据与 assistant loss 掩码，不加载模型权重，不调用 CUDA。只有显式指定 `--mode train` 且全部门槛通过后才加载本地因果语言模型，训练 LoRA。依赖 `torch`、`transformers`、`peft`，不需要 `trl`、在线服务或任意远程模型代码。

## 输入契约

输入是仓库外、经真人审核的 UTF-8 JSONL。每行一条样本，必须同时存在独立 `train` 与 `dev`，不接受 `test` 混入训练入口。字段如下：

| 字段 | 要求 |
|---|---|
| `sample_id` | 非空、唯一、无首尾空格、非占位符 |
| `split` | `train` 或 `dev`；应在人工审核前按来源家族冻结 |
| `task` | `extract`、`diagnose`、`rewrite`，也接受中文“抽取”“诊断”“改写” |
| `source_kind` | `human_authored`、`human_reviewed_synthetic`、`authorized_real`；一律拒绝 `synthetic_unreviewed` |
| `review_status` | 必须恰好为 `human_adjudicated` |
| `reviewer_ids` | 非空、去重的审核员标识数组；使用可追溯的内部标识，无需姓名或联系方式 |
| `adjudication` | 对象，含 `record_id`、`reviewer_id`、`decision="approved"`、含时区的 ISO `reviewed_at`、`payload_sha256`；仲裁人必须出现在审核员数组中 |
| `source_job_family` | 来源岗位家族标识或标识数组；不适用时显式 `[]` |
| `profile_family` | 来源简历/画像家族标识或标识数组；不适用时显式 `[]`；每条样本两种家族中至少一种非空 |
| `messages` | 仅 `[user]` 或 `[system,user]`，每项 `{role,content}`；不接受提示中已有 assistant 轮次 |
| `label` | 经审核的非空 assistant 文本，不接受 null、空值、字符串 `null`/`none` |
| `rewrite_mode` | 改写任务必填：`sentence_preserving` 或 `human_reviewed_complex` |

同一岗位家族、同一画像家族不得跨 `train/dev`。即使换了家族标识，完全相同的提示也不得跨分区。近重复/改写/不同任务对应的同源画像必须在上游使用同一 `profile_family`，脚本不会仅凭不同 ID 就证明文本独立，也不能替上游自动发现所有近重复泄漏。

仲裁哈希由 `review_payload_sha256(record)` 计算，包含样本 ID、任务、分区、来源类型、岗位/画像家族、完整消息、label、改写模式、source_text 与 alignment_spans。审核工作流应保存被审版本并由真人完成批准后填写仲裁记录；改动这些字段后必须重新核对批准记录。该哈希绑定内容版本，**不认证审核员身份，也不能阻止伪造“真人已审核”的声明**，因此不能用自动生成假审核员或假仲裁记录越过实际人审。

单测数据有 `fixture_only=true`，正式 CLI 会拒绝。代码内部 `allow_test_fixtures=True` 仅供单元测试，无对应 CLI 开关。仓库中没有可冒充真人训练数据的完整 JSONL 样例。

## 保全改写边界

`sentence_preserving` 额外要求：

1. `source_text` 必须逐字出现在末条用户输入中。
2. 源文本与 label 按中文句号、问号、叹号、分号、换行切分后，去除句首尾空白的句子多重集相等；字符替换、“了解”升级“精通”、增加经历都会拒绝。
3. `alignment_spans` 是非空数组，每项含 `source_start/source_end/target_start/target_end`，使用 Python Unicode 码点半开区间。两侧切片必须完全一致，跨度不得重复覆盖非空白字符，且必须覆盖两侧全部非空白字符。
4. 检出常见指代或顺序词时，不允许重排句序。此规则不是完整中文语义解析，因此通过后仍只报告“文本/跨度保全”，不报告“所有事实已证明”。

缺少跨度直接拒绝，不自动猜测。其他复杂重写只能用 `human_reviewed_complex` 并逐条人审，不使用保全检查为其事实背书。任何样本声明 `programmatically_proven_facts=true` 都会拒绝。

## loss 掩码与截断

工具要求 fast tokenizer，并同时核验原生 chat template `{% generation %}` mask 和独立字符 offset。仅 assistant label 正文 token 参与 loss；system/user 提示、角色头、结束控制符、padding 都是 `-100`。任何跨越目标边界的 token、缺失 generation mask、生成前缀不一致、原生 mask 混入提示/漏标目标都直接拒绝。

完整样本超过 `--max-length` 时直接报错，不截断、不静默丢样本、不把被截断监督当作成功训练。长样本应回到数据准备阶段，在保留上下文与证据的前提下拆分并重新审核。当前 SFT 不对结束控制符计算 loss，因此也不声称训练了新的停止行为。

`qwen3_assistant_mask.jinja` 是显式的 Qwen ChatML 正文模板，没有隐式 thinking 前缀。仅在确认本地因果指令模型使用相同特殊 token/协议时传入。工具会拒绝不符合边界契约的模板，不能把这个文件随意套给其他模型。编码器/embedding 模型只可用于 tokenizer 校验，不能作为 `--mode train` 的因果语言模型。

## 人审后使用的完整命令

以下路径中的本地因果模型与审核 JSONL 必须由操作者真实准备，不会自动下载或合成人审记录。校验和训练输出目录必须事先不存在且位于仓库外；脚本以目录 0700、文件默认 0600 权限保存私有结果，并拒绝复用/覆盖已有路径。

```bash
PYTHON=/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python
GENERATION=/storage/xukunbo2/job-agent/research
REVIEWED=/home/xukunbo/.cache/job-agent/research/generation-human-reviewed.jsonl
CAUSAL_MODEL=/home/xukunbo/.cache/job-agent/models/approved-qwen-causal-instruct

"$PYTHON" "$GENERATION/train_generation_sft.py" \
  --data "$REVIEWED" --model-dir "$CAUSAL_MODEL" \
  --chat-template-file "$GENERATION/qwen3_assistant_mask.jinja" \
  --output /home/xukunbo/.cache/job-agent/research/generation-validation-v1 \
  --mode validate --max-length 2048

# 只有上一步通过且真实人工审核完成后，才执行此命令。
# CUDA设备由操作者确认当前可用，示例不会自动运行。
"$PYTHON" "$GENERATION/train_generation_sft.py" \
  --data "$REVIEWED" --model-dir "$CAUSAL_MODEL" \
  --chat-template-file "$GENERATION/qwen3_assistant_mask.jinja" \
  --output /home/xukunbo/.cache/job-agent/research/generation-sft-pilot-v1 \
  --mode train --device cuda:2 --dtype bfloat16 \
  --max-length 2048 --max-steps 20 --batch-size 1 --gradient-accumulation 4 \
  --eval-every 5 --lr 5e-5 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-targets q_proj,k_proj,v_proj,o_proj --seed 42
```

这是最小续接试跑，20 个优化步不能替代正式训练与独立评测。梯度累积按真实 assistant token 数加权；基础模型冻结，训练指定 LoRA 模块，启用 gradient checkpointing 与梯度裁剪。默认 CPU/float32，CUDA auto 使用 BF16 并检查设备能力；没有量化和多卡训练支持。

输出 `validation.json`、`manifest.json`；实际训练另有逐步 `training_log.json`、`metrics.json`、`adapter/`。清单含输入/源码/模型配置/模板哈希、训练参数与边界声明，不保存输入简历正文或虚构的人审评测分数。adapter、tokenizer 与训练日志仍可能含敏感关联信息，应留在仓库外。

训练前记录基础开发集 assistant token loss；仅按独立 dev loss 选择 checkpoint，允许最佳步数为 0，保留“训练没有改善”的结果。没有 test 选模，没有自动上线，也没有因 loss 下降就宣称事实性、人岗推荐或简历质量提升。正式采用前需另外做人审、事实核验、拒答与对抗样例评测。

## 已执行验证

```bash
cd /storage/xukunbo2/job-agent
/storage/xukunbo2/venvs/qwen3-grpo-train/bin/python -B -m unittest discover -s tests -p 'test_research_generation.py'
```

4 个 CPU 测试覆盖：assistant-only loss/padding/恶意掩码/截断拒绝；未审核与空标签/虚构 fixture/仲裁哈希失效拒绝；岗位/画像/原样提示跨分区拒绝；保全改写跨度、术语夸大与任意事实背书拒绝。另用本机 Qwen tokenizer 做真实模板冒烟检查，35 个序列 token 中仅 7 个目标正文 token 参与 loss，解码后与目标逐字一致；未加载模型权重、未启动 SFT。
