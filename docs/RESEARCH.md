# 招聘推荐重构的技术调研与选型依据

检索日期：2026-09-20。本文区分公开来源的事实、针对本项目的工程建议和仍待验证的假设。公开榜单上的成绩不能直接证明某模型更适合深圳招聘；所有候选都须在同一份、人工确认的简历—岗位留出集上比较。

## 一、“最近最火的苏神 skill”核查

使用了“苏神 skill 2026”“苏剑林 skill Claude”“苏剑林 skills”“苏神 Claude skill”“苏神 简历 skill github”等组合，并检查苏剑林个人网站、候选项目的原始仓库与 Anthropic 官方说明。

**结论：未找到唯一出处，按 Agent Skills（SKILL.md 模块化技能包）理解；同时记录一个名称相关的社区候选，不能把它写成苏剑林本人发布或认证的简历工具。** “最近最火”没有可复核的统计口径，本次不据此做作者归属判断。

| 核查对象 | 找到的证据 | 能得出的结论 |
| --- | --- | --- |
| 苏剑林个人发布 | [科学空间个人网站](https://www.kexue.fm/)与[文章归档](https://www.kexue.fm/content.html) | 本次检索没有找到能唯一对应用户表述的本人发布声明；这不等于断言不存在 |
| 名称相关社区项目 | [tianming23/SuGPT-kexue](https://github.com/tianming23/SuGPT-kexue)与[其中的 sujianlin/SKILL.md](https://github.com/tianming23/SuGPT-kexue/blob/main/.codex-skills/sujianlin/SKILL.md) | 仓库作者将苏剑林博客整理为检索资料和方法论技能；不是简历训练工具，亦不能据名字认定为本人作品；项目首发日期本次未核准 |
| Anthropic Agent Skills | [官方发布公告](https://claude.com/blog/skills)，首发 2025-10-16，公告记录 2025-12-18 开放标准更新 | 有明确、可核查的产品和格式出处，但不能据此证明它就是用户说的“苏神 skill” |

可迁移的方法是“按任务加载流程 + 将输出绑定到来源”，不是模拟某个人的口吻。简历技能可以依次执行信息抽取、缺口核验、基于已知经历的改写、重新匹配；“用户未写明”与“用户确实不会”必须分开。

Agent Skills 的标准目录入口是 `SKILL.md`，使用 YAML 元数据描述名称与适用任务，再按需读取正文、参考资料和脚本。建议将 `resume-diagnose`、`gap-analysis`、`resume-rewrite`、`mock-interview` 分包，每包定义输入、输出、证据约束、失败行为与固定回归样例。该结构来自[开放格式规范](https://agentskills.io/specification)与[Anthropic 工程说明](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)；它不是训练模型权重的方法，也不会自动保证建议正确。

## 二、中文向量模型的候选与边界

以下参数按官方模型卡、配置与论文核对。“上下文长度”以 token 计；不是中文字数，也不是推荐在所有请求中用满的长度。

| 候选 | 参数量 | 默认向量维度 | 最大输入 | 本项目定位 | 来源与发布日期 |
| --- | ---: | ---: | ---: | --- | --- |
| `BAAI/bge-large-zh-v1.5` | 约 326M | 1024 | 512 | 中文、成本较低的必做基线；JD 按职责/技能分段，避免截断尾部条件 | [模型卡](https://huggingface.co/BAAI/bge-large-zh-v1.5)，v1.5 更新 2023-09-12；[配置](https://huggingface.co/BAAI/bge-large-zh-v1.5/blob/main/config.json) |
| `BAAI/bge-m3` | 约 568M | 1024 | 8192 | 长简历与中英技能混排候选；主线先用 dense，稀疏与多向量另做消融 | [模型卡](https://huggingface.co/BAAI/bge-m3)，发布 2024-01-30；[论文](https://arxiv.org/abs/2402.03216)首投 2024-02-05 |
| `Alibaba-NLP/gte-multilingual-base` | 305M | 768 | 8192 | 长文本与较低资源消耗的另一候选；选择 multilingual 版本，而非默认拿英文模型代替中文评测 | [模型卡](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)；[mGTE 论文](https://arxiv.org/abs/2407.19669)，2024-07-29 |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 | 32768 | 当前资源下优先尝试的可微调主线，短批次能快速做完整消融 | [官方发布](https://qwenlm.github.io/blog/qwen3-embedding/)，2025-06-05 |
| `Qwen/Qwen3-Embedding-4B` | 4B | 2560 | 32768 | 未微调强基线与后续 LoRA 候选，用实测收益决定是否部署 | [官方系列表](https://qwenlm.github.io/blog/qwen3-embedding/)，2025-06-05 |
| `Qwen/Qwen3-Embedding-8B` | 8B | 4096 | 32768 | 研究对照；显存、延迟和增益均实测后再决定 | [官方系列表](https://qwenlm.github.io/blog/qwen3-embedding/)，2025-06-05 |

BGE 参数量是根据公开骨干配置得到的近似规模，加载头和具体实现会影响精确计数；应把运行时的 `sum(p.numel() for p in model.parameters())` 写入实验记录。BGE-M3 的多向量表示不能直接当作单个 1024 维向量放入同一个普通 FAISS 索引。

Qwen3-Embedding 支持可变输出维度与任务指令。0.6B 模型卡给出 32–1024 维范围；不能因此推断任意模型都能直接裁剪向量。需在训练、索引和查询时保持模型版本、池化方法、归一化和维度一致。其模型卡列出的早期兼容下限为 `transformers>=4.51.0`、`sentence-transformers>=2.7.0`，这只是兼容下限，最终环境应锁定实际通过测试的版本。[Qwen 0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

BGE 中文模型可使用官方检索指令；Qwen 应使用其模型卡约定的查询格式和最终 token 池化，不能套用 BERT 的 CLS 池化。为了保持产出中文，本项目先用中文任务指令，另做有/无指令对照；模型卡提出的英文指令建议可以作为后续受控实验。句向量余弦分数不等于录用概率，阈值必须从本地验证集校准。[BGE 使用说明](https://huggingface.co/BAAI/bge-large-zh-v1.5)、[Qwen 使用说明](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

### 2.1 本地 CPU、消费级显卡与当前服务器

以下是工程容量估算，不是本次已经测得的吞吐或显存峰值。BF16/FP16 纯权重内存近似为“参数量 × 2 字节”；不含激活、梯度、优化器、框架缓存和并行开销。

| 规模 | 半精度权重下限约值 | 推理建议 | 微调建议 |
| --- | ---: | --- | --- |
| BGE-large-zh / mGTE | 0.6–0.7 GB | CPU 小批次可做；8 GB 级 GPU 可从短序列、小批次开始 | 16–24 GB GPU，从 512 token、每卡 8–16 对起步 |
| BGE-M3 / Qwen 0.6B | 1.1–1.2 GB | CPU 能运行但需实测响应时间；GPU 从 1024 token、batch 8 开始 | 单张 24–32 GB 从 512/1024 token、每卡 4–8 对起步，梯度检查点与缓存负例池 |
| Qwen 4B | 8 GB | 16–24 GB GPU 可先做短序列小批次推理；CPU 不作为交互默认 | 单张 32 GB 先尝试 LoRA、短序列与很小微批次；全参训练需分片与额外资源 |
| Qwen 8B | 16 GB | 24–32 GB GPU 可尝试短序列小批次推理；32K 并非默认可承受 | 32 GB 上 LoRA 仍需控制长度，必要时量化或分片；不能承诺单卡全参训练 |

官方还有 [Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)，可作为 CPU 部署候选，但量化版必须重测检索指标，不能混用不同量化版本建立的索引。通用的 [Sentence Transformers 推理加速文档](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)可用于评估 ONNX/OpenVINO 路线，具体模型导出兼容性须单独验证。

当前根任务已探测到 4 张 RTX 5090、每张约 32 GB，且 1 号卡有现存 vLLM 进程。建议先在一张空闲卡完成 0.6B 模型的小实验，再把空闲卡用于独立模型对照和不同随机种子；既有进程不能被训练脚本抢占或终止。四张卡并不自动形成一个 128 GB 的单卡显存空间。

环境建议：现场驱动为 `570.133.20`，`nvidia-smi` 显示 CUDA 12.8 兼容上限；先复用已有支持 `sm_120`、与该驱动兼容的 PyTorch 环境，或在隔离环境中复用其已验证版本。正式训练前做 BF16 小张量、模型前向和反向测试。本轮不安装 PyTorch、不升级驱动，也不把 CUDA 13 安装命令套用到现场。PyTorch 2.7 官方说明提供了 Blackwell/CUDA 12.8 支持依据；检索到的 2.12 文档另有 CUDA 13 路线，这仅是该版本文档的信息，不能当作当前机器应升级的理由。依赖版本与驱动兼容性必须以实际烟雾测试为准。[PyTorch 2.7 说明](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 2.12 版本说明](https://pytorch.org/blog/pytorch-2-12-release-blog/)

### 2.2 推荐的微调工具链与实验起点

1. **通用主线：** `sentence-transformers` 负责模型、数据列映射、训练器和检索评测，PEFT 负责 LoRA。BGE 专用统一 dense/sparse/ColBERT 微调再用 [FlagEmbedding 官方工程](https://github.com/FlagOpen/FlagEmbedding)。不同时维护两套等价主训练入口。
2. **首轮对照：** 未微调 BGE-large-zh、BGE-M3、Qwen0.6B；Qwen0.6B LoRA；Qwen0.6B 全参；Qwen4B 未微调。每个配置使用相同数据划分、候选库、截断策略和检索预算，分别报告 Recall@10、MRR、nDCG@10、p95 延迟、峰值显存。
3. **参数起点（待验证）：** 最大长度 1024；每样本 1 个正例、2–4 个已核验难负例；对比 batch 128，显存微批次 4–8；温度 0.05，并比较 0.02/0.1；1–3 个 epoch；全参学习率 `1e-5`，LoRA `1e-4`，`r=16/alpha=32/dropout=0.05`；warmup 10%，每 100 步或每个 epoch 评估，按验证 nDCG@10 选检查点。低样本量时优先低学习率和早停。
4. **不要把梯度累积误写成扩大负例池：** 普通梯度累积只累加多次反向的梯度；每次 InfoNCE softmax 看到的负例仍受当前批次限制。需要大池时用 `CachedMultipleNegativesRankingLoss`，或经验证的跨卡 gather。`scale=20` 对应温度 0.05。[Sentence Transformers 损失文档](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
5. **假负例过滤：** `NO_DUPLICATES` 只解决字面重复；同一候选人的多条真正匹配岗位仍可能互相成为错误负例。需要基于岗位族、已有正标签和人工/教师复核作屏蔽或采用多正例目标。[官方采样器文档](https://www.sbert.net/docs/package_reference/base/sampler.html)、[官方难负例挖掘工具](https://sbert.net/docs/package_reference/util/hard_negatives.html)

## 三、从招聘论文可以迁移什么

| 一手来源与时间 | 论文实际问题 | 可迁移到本项目 | 不能据此声称 |
| --- | --- | --- | --- |
| [PJFNN：基于联合表示的人岗匹配](https://arxiv.org/abs/1810.04040)，2018-10 | 用职位要求与人才资格联合表示做匹配 | 简历与 JD 分别编码、显式学习两端关系，而非只训练 JD 词向量 | 复制老模型就能解决本地标签不足 |
| [人岗匹配的可迁移深度全局匹配网络](https://aclanthology.org/D19-1487/)，EMNLP 2019-11 | 标注不足时的跨领域适配和句子/全局匹配 | 职类分层评估、留出未见公司/岗位族、检查跨职类泛化 | 训练总 AUC 足以代表所有职类 |
| [职业路径在人岗匹配中的作用](https://ojs.aaai.org/index.php/AAAI/article/view/28685)，AAAI 2024-03-24，官网列出 BOSS 直聘作者 | 从职业路径提取一致性、相似性和连续性偏好 | 把经历时间线、职业转向和偏好与技能证据分开表示 | 应届生没有职业路径就无法推荐；或历史职业必定限制未来方向 |
| [JobBERT：通过技能理解职位标题](https://arxiv.org/abs/2109.09605)，2021-09-20 | 利用职位技能共现增强职位标题表示，做标题归一化 | 标题别名归一化、技能辅助弱监督、岗位名与技能关系建模 | **JobBERT 是 BOSS 直聘论文**；或标题归一化提升等于简历匹配提升 |
| [YouTube 深度推荐论文](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)，RecSys 2016 | 分离候选生成与排序 | 将高召回的多路检索和可解释特征排序分开优化 | 本项目只有万级岗位也必须照搬超大规模服务拆分 |

JobBERT 的原始论文明确列出 Ghent University–imec 与 TechWolf；其训练目标和评估任务不是完整的招聘录用预测。本项目借鉴“技能帮助理解职位名称”的思想即可，不应错误归因。[JobBERT 原文第一页](https://arxiv.org/pdf/2109.09605)

对本项目的判断：没有持续、大量、可审计的用户行为之前，先用共享文本编码器的简历塔/岗位塔，再配结构化特征排序；无需先训练两个互不共享权重的大网络。“两塔”描述的是查询侧和岗位侧可独立编码，权重可以共享。工业“召回—粗排—精排”的核心是预算分配；万级库可省略独立粗排服务，仍保留逻辑边界与观测指标。[Google 推荐系统分阶段说明](https://developers.google.com/machine-learning/recommendation/overview/types)

## 四、训练数据设计中需要特别纠正的假设

以下是结合本项目任务得出的设计建议，不是声称论文已经验证了本数据。

- **岗位名相同不自动为正例。** “运营”“工程师”“助理”等标题会跨业务甚至跨职类。JD—JD 正对需先通过职责相容、技能重合、级别相近等筛选；标题与 JD 的自监督正对只能作为领域适配，不能直接当作某份简历被人类认可的正反馈。
- **不同薪资不自动是语义负例。** 同一职责可能因公司、奖金口径和招聘日期而薪资不同。薪资不满足应由过滤/排序处理；训练向量时只把已核验的职责不相容、资格差异显著且确实不适合的样本当作负例，避免破坏语义空间。
- **基于真实 JD 合成简历仍有泄漏风险。** 需要改变叙述方式，拒绝照抄完整技能列表与职责；加入不完全匹配、转行、应届与信息缺失版本。合成器输出来源 ID、生成模板、能力证据和未满足要求；另一个评审器校验，抽样人工复核。真实用户简历必须经授权与脱敏。
- **先划分来源，再合成。** 同源 JD、近重复簇、同一简历模板的变体不得跨训练/验证/测试；本地测试集不得交给生成训练数据的脚本。以公司组或岗位近重复簇作分组留出，再补时间切分。
- **跳过不是稳定负例。** 记录是否曝光、位置、当次筛选条件和反馈原因；用户可能跳过已投递、暂不感兴趣或重复岗位。未经曝光的职位不能直接标为不喜欢。
- **先验证标签，再扩大模型。** 人工小金标集应覆盖弱监督冲突。合成标签和规则标签作为带来源/置信度的弱监督，不能再由同一教师给模型打满分来证明成功。

## 五、研究扩展的进入条件

技能共现图可以从经过规范化的 `job_id→skill` 事实边开始，每条边保留原文片段和抽取版本。共现统计表示“同时出现在招聘文本中”，不等于“具备 A 必然掌握 B”。预测出的技能关系进入单独的待验证表；经人工核验才升级为可引用事实。TransE/RotatE 的得分也不能作为岗位事实来源。先检验图特征对冷门职类 Recall@10 和证据正确率有无增益，再投入独立图嵌入训练。[TransE 原论文](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)、[RotatE 原论文](https://arxiv.org/abs/1902.10197)

GRPO 检索控制器仅在已有稳定查询动作、可执行评测器、大量独立查询与可靠奖励后再做。先记录规则控制器与简单路由的结果，动作限定为查询改写、检索路数和预算；奖励需同时包含相关性、硬条件违背、虚构事实和成本，不可只奖励更长的理由。当前样本与人工金标不足时，强化学习容易学会利用评测器漏洞。[DeepSeekMath 中的 GRPO 原始方法](https://arxiv.org/abs/2402.03300)

## 六、可复核实验记录的最小字段

每次结果必须保存：数据快照摘要、分组切分清单摘要、训练样本来源构成、模型仓库与 revision、实际依赖版本、分词/池化/归一化、序列长度、随机种子、负例池大小、LoRA 配置、硬件与批大小、训练时间、模型选择依据、每个查询的指标，以及失败样例。报告 bootstrap 查询级置信区间；若仅用了合成简历，明确称为“合成场景离线结果”，不能称为真实用户成功率。

本轮工作完成的是来源核查与方案建议，未运行这些候选模型的横向检索实验，也未据此声称任何模型已经优于旧系统。
