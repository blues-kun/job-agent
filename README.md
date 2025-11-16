# 智能职位推荐项目（job_agent）

## 项目简介
- 面向简历与岗位数据的智能推荐系统，融合本地匹配算法与大语言模型生成推荐理由与行动建议
- 前端静态站点 + 统一后端服务（同时提供网页与 API），开箱即用，支持本地数据与 LLM 智能对话
- 支持 XGBoost 机器学习模型，基于用户反馈持续优化推荐质量

## 🤖 核心架构：主-辅双 Agent 协作

### 辅 Agent：简历智能解析与用户画像生成
- **技术栈**：基于 **LangChain v1.0** 驱动的提示词版本化管理
- **核心功能**：
  - 从用户输入（自然语言简历）中智能提取结构化信息
  - 主动推理缺失字段（城市、薪资、工作经验、学历等）
  - 生成完整的 `ResumeProfile` 用户画像
- **关键模块**：`resume_extract/extractor.py`（`ResumeExtractor`）

### 主 Agent：职位匹配与推荐决策
- **技术栈**：结合规则引擎、语义相似度、XGBoost 机器学习
- **核心功能**：
  - 抽取岗位 JD 关键信息（职位类型、地点、薪资、要求）
  - 联合辅 Agent 生成的用户画像，计算多维度匹配分数
  - 生成个性化推荐理由与行动建议（岗位推荐、换岗建议、简历优化方案）
- **关键模块**：
  - `search/matcher.py`（规则匹配 + 相似度计算）
  - `search/scorer_xgb.py`（XGBoost 智能排序）
  - `agent.py`（LLM 推荐文案生成）

### 协作流程
```
用户输入 → 辅Agent(简历解析) → 用户画像 → 主Agent(JD抽取+匹配) → 推荐结果
           ↓                                    ↓
      LangChain提示词管理              规则+ML+相似度融合
```

---

## 主要能力
- **对话式交互**：粘贴简历或描述偏好，辅 Agent 智能抽取结构化简历并更新表单，可启用/禁用 LLM
- **本地匹配**（JobMatcher）：类型过滤、地点匹配、薪资计算与评分排序，返回 Top N
- **LLM 推荐**：主 Agent 将 Top N 结构化岗位传给模型，生成友好的推荐表格与行动建议（Markdown）
- **XGBoost 学习排序**：基于用户反馈（喜欢/跳过）训练模型，提升推荐准确度
- **数据向量化预处理**：支持 Word2Vec 文本向量化，提取 100 维语义特征用于深度学习训练
- **多模型对比训练**：支持 XGBoost、MLP、CNN、LSTM 等多种模型训练与性能对比
- **数据处理与清洗**：支持真实岗位数据导入（CSV/XLSX/JSONL）、岗位职责清洗、薪资解析

---

## 🚀 快速开始

### 完整工作流程（从零开始）

**步骤1：环境配置**
```bash
# 安装依赖
pip install -r requirements.txt

# 配置 LLM（如需使用智能对话）
cp env_template.txt .env
# 编辑 .env 填入你的 API_KEY
```

**步骤2：准备岗位数据**
```bash
# 如果使用真实岗位数据，确保 data/job_data.csv 已存在


**步骤3：数据向量化（可选，用于深度学习）**
```bash
# 将岗位数据向量化
python preprocess_job_data.py
# 输出: data/job_data_vectorized.csv 和 data/job_data_vectorized.parquet
```

**步骤4：生成训练数据**
```bash
# LLM智能生成训练数据
python logs/generate_realistic_samples.py
```

**步骤5：训练模型**
```bash
# 训练 XGBoost 高级版（推荐）
python training/train_xgb_advanced.py

# 或训练其他模型进行对比
python training/train_mlp.py
python training/train_cnn.py
```

**步骤6：启动服务**
```bash
cd web
python unified_server.py
# 访问 http://127.0.0.1:8002/
```

---

## 🚀 运行方式

### 方式一：网页模式（🌟 推荐）

**特点**：
- ✅ **可视化界面**：美观的卡片式布局，实时展示推荐结果
- ✅ **完整功能**：支持 LLM 对话、简历上传、权重调节、反馈收集、模型训练
- ✅ **交互友好**：拖拽滑块调整权重，按钮点击即可完成所有操作
- ✅ **数据管理**：可视化查看训练数据统计、模型评估指标（AUC、准确率）
- ✅ **反馈闭环**：点击"喜欢/跳过"按钮收集反馈，一键重训 XGBoost 模型

**启动步骤**：

**⚠️ 重要提示**：
- 启动服务器前，请先配置 `.env` 文件中的 API_KEY（见"环境配置"章节）
- 如果先启动服务器后配置密钥，需要**重启服务器并刷新浏览器页面**才能生效

```bash
# 1. 进入 web 目录
cd web

# 2. 启动服务器
python unified_server.py

# 3. 访问网页
# 浏览器打开 http://127.0.0.1:8002/
```

**⚠️ 注意**：
- 默认端口为 **8002**，如提示端口被占用，可修改 `web/unified_server.py` 中的 `PORT` 常量
- 或使用任务管理器/命令行关闭占用端口的进程：
  ```bash
  # Windows 查看端口占用
  netstat -ano | findstr :8002
  # 终止进程（替换 PID）
  taskkill /PID <进程ID> /F
  ```

**使用流程**：
1. **上传简历**：在对话框粘贴简历文本或上传 `.txt` 文件
2. **配置偏好**：填写意向城市、期望年薪、目标职位，调整权重滑块
3. **匹配职位**：点击"匹配职位"按钮，查看推荐结果
4. **收集反馈**：对推荐结果点击"👍 喜欢"或"👎 跳过"按钮
5. **训练模型**：积累足够反馈后，点击"🚀 训练模型"重训 XGBoost，查看训练指标

---

### 方式二：命令行模式

**特点**：
- ✅ **轻量快速**：纯命令行交互，无需启动浏览器
- ⚠️ **功能受限**：不支持反馈收集、模型训练、可视化展示
- ⚠️ **交互单一**：适合快速测试匹配算法，不适合日常使用

**启动步骤**：
```bash
# 在项目根目录执行
python main.py
```

**使用流程**：
1. 按提示粘贴简历文本（自动保存到 `resume.json`）
2. 输入命令与模型对话（如 `/get_resume` 查看简历）
3. 模型自动调用本地匹配并生成推荐文案（纯文本输出）

**适用场景**：
- 快速验证匹配算法逻辑
- 调试本地推荐流程
- 批量脚本化处理（结合 Python SDK）

---

## 📦 数据准备

### 岗位数据
- **默认岗位数据**：`data/job_data.xlsx` 或 `data/job_data.csv`
- **数据来源**：深圳地区真实招聘数据（5000+ 岗位）
- **数据格式要求**：CSV/XLSX/JSONL，包含岗位名称、企业、地址、薪资、要求等字段

**真实数据使用示例**（以深圳岗位数据为例）：
  ```bash
# 确保数据文件已放置在正确位置
# data/job_data.csv - 包含深圳地区真实招聘岗位信息

# 查看数据统计
python -c "import pandas as pd; df = pd.read_csv('data/job_data.csv'); print(f'总岗位数: {len(df)}'); print(f'涉及企业: {df[\"企业\"].nunique()}'); print(f'职位类型: {df[\"岗位名称\"].nunique()}')"

# 数据向量化（用于深度学习训练）
python preprocess_job_data.py

# 生成训练数据（基于真实岗位）
python logs/generate_realistic_samples.py
  ```

### 职位分类管理
- **职位分类字典**：`data/position_dictionary.txt`
  - 包含所有支持的真实职位类型（基于深圳招聘市场实际岗位）
  - 用于简历解析时的职位类型匹配和验证
  - 格式：`[一级分类]` 后跟具体职位名称列表
  
- **当前支持的职位分类**（8大类105个职位）：
  - **后端开发**（16）：Java、Python、Golang、C/C++、PHP、C#、.NET、Node.js、全栈工程师、区块链工程师等
  - **前端/移动开发**（10）：前端开发工程师、Android、iOS、鸿蒙开发工程师、U3D、UE4、Cocos等
  - **测试**（10）：测试工程师、自动化测试、测试开发、性能测试、硬件测试、渗透测试等
  - **运维/技术支持**（12）：运维工程师、IT技术支持、网络工程师、网络安全、系统工程师、DBA等
  - **人工智能**（17）：图像算法、NLP算法、大模型算法、SLAM算法、推荐算法、搜索算法、机器学习、深度学习、自动驾驶等
  - **数据**（9）：数据分析师、数据开发、数据仓库、ETL工程师、数据架构师、爬虫工程师、数据治理等
  - **销售技术支持**（4）：售前技术支持、售后技术支持、客户成功等
  - **技术项目管理**（5）：项目经理、实施工程师、需求分析工程师等
  - **高端技术职位**（6）：技术经理、架构师、技术总监、CTO/CIO等

**注意**：职位分类来源于真实岗位数据，系统会自动识别数据中的所有职位类型。

### 简历输入
- **命令行模式**：按提示粘贴简历，自动保存 `resume.json`
- **网页模式**：在顶部对话框粘贴简历或上传 `.txt` 文件

### 训练数据生成

使用 LLM 智能生成训练数据：

```bash
python logs/generate_realistic_samples.py
```

- **特点**：调用大语言模型分析职位和简历的真实匹配度
- **优势**：生成数据更接近真实用户行为，模型训练效果更好
- **要求**：需要配置 `.env` 中的 API_KEY
- **数据来源**：从 `data/job_data.csv` 或 `data/job_data_vectorized.csv` 获取真实职位数据
- **生成策略**：
  - 分层采样：高匹配、中匹配、低匹配职位均衡分布
  - LLM判断：根据城市匹配、薪资匹配、职位类型等维度智能决策
  - 正负样本平衡：确保约 30-40% 的正样本比例

**推荐做法**：
1. 首次使用运行 `generate_realistic_samples.py` 生成冷启动数据
2. 系统运行后收集真实用户反馈
3. 定期重新训练模型，融合冷启动数据和用户反馈

### 数据说明

**数据来源**：深圳地区真实岗位数据（实际招聘平台采集）✅

**数据特点**：
- 📊 **真实性**：来自实际招聘平台，准确可靠
- 🔒 **隐私保护**：已脱敏处理，不包含求职者个人隐私信息
- 📝 **数据内容**：企业名称、岗位要求、薪资范围等公开招聘信息
- 🎯 **模型效果**：基于真实数据训练，AUC 达到 0.5739
- 📈 **数据规模**：5000+ 真实岗位

---

## 📊 数据向量化与模型训练

### 数据向量化预处理

系统支持将职位数据进行向量化处理，为深度学习模型提供数值特征输入：

**向量化流程**：
```bash
python preprocess_job_data.py
```

**处理内容**：
- **文本向量化**：使用 Word2Vec 将岗位职责、岗位要求、岗位名称转换为 100 维语义向量
- **薪资标准化**：解析薪资范围（如"15-25K"）并计算中位数
- **地点处理**：提取省市区信息
- **特征工程**：生成 300+ 维特征（100×3 文本向量 + 其他特征）

**输出文件**：
- `data/job_data_vectorized.csv` - 向量化后的职位数据（CSV格式）
- `data/job_data_vectorized.parquet` - 向量化后的职位数据（Parquet格式，推荐使用）
- `data/word2vec.model` - 训练好的 Word2Vec 模型

### 模型训练与性能对比

系统支持多种机器学习模型训练，并进行全面性能对比：

#### 支持的模型

| 模型类型 | 训练脚本 | 适用场景 | 验证集AUC |
|---------|---------|---------|----------|
| **XGBoost高级** ⭐ | `training/train_xgb_advanced.py` | **表格数据推荐（最佳）** | **0.5739** |
| XGBoost优化 | `training/train_xgb_optimized.py` | 表格数据推荐 | 0.5674 |
| XGBoost原始 | `training/train_with_vectorized_data.py` | 快速baseline | 0.5439 |
| CNN | `training/train_cnn.py` | 图像/序列特征学习 | 0.5537 |
| MLP | `training/train_mlp.py` | 简单神经网络 | 0.5482 |
| LSTM | `training/train_lstm.py` | 序列数据（不推荐） | 0.5044 |

#### XGBoost 三版本演进

系统通过三个阶段优化 XGBoost 模型：

**1. 原始版本（Baseline）**
```bash
python training/train_with_vectorized_data.py
```
- 默认参数快速训练
- AUC: 0.5439
- 用于建立基准线

**2. 优化版本（Hyperparameter Tuning）**
```bash
python training/train_xgb_optimized.py
```
- 参数搜索：learning_rate、max_depth、n_estimators
- AUC: 0.5674（+4.32%）
- 特征重要性可视化

**3. 高级版本（Multi-Stage Training）** ⭐
```bash
python training/train_xgb_advanced.py
```
- 多阶段训练 + 学习率衰减
- 早停机制防止过拟合
- AUC: 0.5739（+5.52%）
- 训练曲线可视化

#### 模型性能对比结果

**核心结论**：
- 🥇 **XGBoost高级版** 在所有模型中表现最佳（AUC 0.5739）
- ✅ 优于 MLP **+4.69%**（0.5739 vs 0.5482）
- ✅ 优于 CNN **+3.65%**（0.5739 vs 0.5537）
- ✅ 优于 LSTM **+13.78%**（0.5739 vs 0.5044）
- 📈 XGBoost 在可解释性、部署成本、训练稳定性等10个维度全面领先

**为什么 XGBoost 最适合职位推荐**：
1. **表格数据天然优势** - 职位推荐是典型的结构化表格数据任务
2. **可解释性强** - 支持特征重要性分析，可指导业务决策
3. **部署成本低** - 单个200KB JSON文件，无需GPU
4. **训练稳定** - 波动极小（±0.0003），深度学习波动大（±0.0220）
5. **增量学习** - 支持继续训练，易于模型更新
6. **行业标准** - Kaggle表格数据竞赛90%+使用XGBoost

---

## ⚙️ 环境配置（LLM）
在项目根目录创建 `.env` 文件：
```env
BASE_URL=https://api.deepseek.com/v1
API_KEY=你的密钥
MODEL=deepseek-chat
TEMPERATURE=0
```

**说明**：
- 若未配置密钥，网页的"启用智能对话"开关关闭后仍可使用本地匹配
- 开启"启用智能对话"时必须有有效 API_KEY，否则会返回错误提示

---

## 🎨 网页前端使用

### 对话区域
- **与职业未来对话**：粘贴简历或描述偏好，点击"发送"即可交互
- **简历上传**：点击"选择文件"按钮上传 `.txt` 格式简历（自动解析）
- **开关控制**：
  - **启用智能对话**：控制是否调用 LLM（需配置 API_KEY）
  - **自动提取简历信息到表单**：解析到简历后自动写入表单（默认开启）

### 数据输入区
- **意向城市**：填写目标城市（如"北京"）
- **期望年薪**：填写年薪（千元），支持 +/- 按钮快速调整（步长 1000）
- **意向职位**：填写目标职位（多个用逗号分隔，如"Java,后端开发"）

### 权重设置
拖拽滑块调整各维度权重（0-10）：
- 职位类型、地点、薪资、工作年限、学历要求
- 院校层次、文本相似度、标题意向

**提示**：点击"建议权重"按钮恢复默认配置

### 匹配职位
点击"匹配职位"按钮触发推荐：
1. 本地算法生成 Top N（结果显示在"结果"面板）
2. 若启用智能对话，LLM 生成推荐表与行动建议（显示在对话区）
3. 可对每个推荐职位点击"👍 喜欢"或"👎 跳过"收集反馈

### 训练数据管理（XGBoost）
在"反馈记录"面板管理训练数据：
- **🔄 刷新列表**：重新加载训练样本和反馈事件
- **🚀 训练模型**：一键训练 XGBoost 模型，自动从 `logs/recommend_events.jsonl` 加载数据并提取特征

**数据统计显示**：
- 总样本数、正样本数（喜欢）、负样本数（跳过）、样本比例
- 验证集 AUC、准确率、精确率、召回率、F1 分数

**数据来源说明**：
- 🔵 **冷启动数据**：系统初始数据，用于模型训练
- 🟢 **用户反馈**：实际使用中收集的喜欢/跳过数据
- 两类数据统一存储在 `logs/recommend_events.jsonl`，自动合并用于训练

---

## 🔌 API 列表

- **获取岗位数据**：`GET /api/jobs`
- **对话与简历抽取**：`POST /api/chat_resume`
- **简历信息增强**：`POST /api/resume_enhance`
- **职位推荐**：`POST /api/recommend`
- **用户反馈提交**：`POST /api/feedback`
- **推荐事件管理**：`POST /api/recommend_events`
- **XGBoost 训练**：`POST /api/xgb_ops`

---

## 🔄 推荐管线架构

### 基础推荐流程
1. **本地匹配**：岗位数据（JSONL/Excel）→ JobMatcher 规则过滤与评分
2. **Top N 生成**：按多维度加权排序，返回 Top N（默认 10）
3. **LLM 增强**（可选）：将 Top N 发送给 LLM，生成推荐表与行动建议
4. **网页展示**：
   - 结构化 Top 列表 → "结果"面板（卡片展示）
   - LLM 推荐文案 → "对话区"（Markdown 渲染）

### XGBoost 混合推荐（训练后启用）
1. **特征提取**：从简历和职位提取 9 维特征（文本相似度、地点匹配、薪资比、学历等）
2. **模型预测**：XGBoost 输出 0-1 概率分数
3. **混合融合**：`blend_score = (1-α) × rule_score + α × (xgb_score × 5.0)`（α=0.5）
4. **动态优化**：收集用户反馈（喜欢/跳过）→ 重训模型 → 提升准确度

---

## 📂 关键代码位置

### 后端服务
- **统一服务入口**：`web/unified_server.py`
  - 推荐接口：处理 `/api/recommend` 请求
  - 对话接口：处理 `/api/chat_resume` 请求，调用 LLM 或简历抽取器
  - XGBoost 训练 API：`/api/xgb_ops`（导出数据、训练模型）
  - 反馈事件管理：`/api/recommend_events`（增删改查）

### 前端界面
- **页面结构**：`web/index.html`
- **样式定义**：`web/styles.css`
- **交互逻辑**：`web/app.js`
  - 匹配按钮行为：调用 `/api/recommend` 并展示结果
  - 反馈按钮：点击"喜欢/跳过"发送到 `/api/feedback`
  - Markdown 渲染：助手消息卡片展示
- **按钮特效**：`web/button_effects.js`
  - 点击涟漪、粒子爆发、脉冲动画、触觉反馈

### 推荐核心模块
- **匹配器**：`search/matcher.py`（JobMatcher 类，Top N 排序）
- **规则评分器**：`search/scorer.py`（MatchScorer 类，多维度加权）
- **XGBoost 评分器**：`search/scorer_xgb.py`（模型加载与预测）
- **相似度引擎**：`similarity/engine.py`、`similarity/text.py`（文本匹配与分词）

### XGBoost 训练模块（模块化）
- **训练工具**：`training/training_utils.py`
  - 数据集加载、指标计算、特征名提取、JSONL 读写
- **模型管理器**：`training/model_manager.py`
  - XGBoostModelManager 类：训练、评估、保存、加载模型
- **API 处理器**：`training/xgb_api.py`
  - `handle_xgb_ops`：导出训练数据、触发模型训练
- **特征提取器**：`features/extractor.py`
  - `extract` 函数：从简历和职位提取 9 维特征

### 数据处理
- **岗位数据管理**：
  - 支持导入真实岗位数据（CSV/XLSX/JSONL格式）
  - 基于深圳地区真实招聘数据（5000+ 岗位，105个职位类型）
- **训练数据生成**：
  - `logs/generate_realistic_samples.py` - LLM智能生成高质量用户反馈样本（推荐）
- **数据预处理**：
  - `data_preprocess/cleaner.py` - 数据清洗与薪资解析
  - `preprocess_job_data.py` - 数据向量化（Word2Vec）
- **简历提取器**：`resume_extract/extractor.py`（LLM 驱动，主动推断缺失信息）

### 配置文件
- **环境配置**：`.env`（API_KEY、BASE_URL、MODEL）
- **匹配配置**：`config.py`
  - MatchConfig：权重配置
  - USE_XGB_SCORER：是否启用 XGBoost
  - XGB_BLEND_ALPHA：混合融合系数（默认 0.5）

---

## 📝 完整使用流程（网页模式）

### 1. 启动服务

**⚠️ 重要**：启动前请先配置 `.env` 文件（见上方"环境配置"章节），否则智能对话功能无法使用。如果先启动后配置，需要重启服务器并刷新页面。

```bash
cd web
python unified_server.py
# 访问 http://127.0.0.1:8002/
```

### 2. 输入简历
- **方式 A**：在对话框粘贴简历文本 → 点击"发送"
- **方式 B**：点击"选择文件" → 上传 `.txt` 简历

### 3. 配置偏好
- 填写**意向城市**（如"北京"）
- 调整**期望年薪**（支持 +/- 按钮）
- 输入**目标职位**（多个用逗号分隔）
- 调整**权重滑块**（或点击"建议权重"）

### 4. 获取推荐
点击"**匹配职位**"按钮：
- ✅ 本地算法生成 Top 10（"结果"面板显示）
- ✅ LLM 生成推荐表与行动建议（对话区显示，需开启"启用智能对话"）

### 5. 收集反馈
对推荐结果点击：
- **👍 喜欢**：记录为正样本（action=like）
- **👎 跳过**：记录为负样本（action=skip）
- 反馈数据自动保存到 `logs/recommend_events.jsonl`

### 6. 训练模型（可选）
在"**反馈记录**"面板：
1. 点击"**🔄 刷新列表**" → 查看已收集的反馈和冷启动数据统计
2. 点击"**🚀 训练模型**" → 训练 XGBoost，自动提取特征并训练，查看 AUC、准确率等指标
3. 模型保存到 `models/xgb_model.json`，下次推荐自动启用

**说明**：系统支持冷启动数据和用户反馈的统一管理，首次使用即可利用预置的冷启动数据训练模型

---

## 💡 使用技巧与说明

### 简历自动同步
- **默认行为**：勾选"自动提取简历信息到表单"后，解析到的简历会实时更新表单字段
- **适用场景**：
  - ✅ 上传 `.txt` 简历文件（自动勾选开关，快速初始化）
  - ✅ 粘贴完整简历文本（一次性提取所有信息）
- **关闭开关**：仅在对话区交流，不更新表单（适合试探性咨询）

### 匹配按钮逻辑
- **点击"匹配职位"时**：
  1. 读取当前表单数据（意向城市、年薪、职位类型、权重）
  2. 执行本地多维度匹配与评分，生成 Top N 结构化列表
  3. 若启用智能对话，将 Top N 传给 LLM 生成推荐表与行动建议
- **结果展示**：
  - **结构化列表** → "结果"面板（卡片格式，含分数与匹配理由）
  - **LLM 推荐文案** → "对话区"（Markdown 表格与行动建议）

### XGBoost 模型优化
- **首次使用**：
  - 系统提供冷启动数据（`logs/recommend_events.jsonl` 中的预置样本）
  - 可直接点击"🚀 训练模型"利用冷启动数据训练基础模型
- **积累反馈后**：
  - 收集用户实际反馈（喜欢/跳过）
  - 反馈数据自动追加到 `recommend_events.jsonl`
  - 再次点击"🚀 训练模型"，系统自动合并冷启动数据和用户反馈进行训练
  - 模型训练成功后，下次推荐自动融合 XGBoost 分数（持续提升准确度）
- **查看效果**：训练完成后显示 AUC、准确率、精确率、召回率、F1 等指标，验证模型质量

---

## 🧠 核心算法说明

### 文本匹配与相似度
- **分词与规范化**：中英文统一切分（中文连续汉字块、英文单词块）
- **Jaccard 相似度**：衡量简历技能与职位要求的交集占比
- **计数余弦相似度**：基于词频向量计算，兼顾重复出现的重要词
- **综合评分**：`0.5 × Jaccard + 0.5 × 计数余弦`

### 多维度评分
- **职位类型匹配**：意向职位与岗位标题/类别的交集命中
- **地点匹配**：意向城市与岗位地址/城市字段严格匹配
- **薪资匹配**：期望年薪与岗位薪资区间的重合度（支持"面议"解析）
- **学历/经验匹配**：学历层次（本科/硕士/博士）、工作年限的达标判断
- **院校层次**：985/211 院校加分

**权重配置**：`config.py:MatchConfig`（可调整各维度权重）

### LLM 增强推荐
- **输入**：本地 Top N 的结构化字段（公司、岗位、城市、薪资、匹配理由）
- **输出**：
  - 🎯 高匹配岗位推荐表（Markdown 表格）
  - 📝 下一步行动建议（投递策略、简历优化、面试准备）
- **控制**：勾选"启用智能对话"开关（需配置有效 API_KEY）

### XGBoost 学习排序
- **特征工程**（9 维）：
  - `text_sim`：文本相似度（Jaccard + 余弦）
  - `location_match`：地点匹配度（0/1）
  - `salary_ratio`：薪资匹配比例
  - `education_match`：学历匹配度
  - `experience_ratio`：经验年限比例
  - `title_intent`：标题意向命中（0/1）
  - `salary_negotiable`：薪资可议标记
  - `jaccard`、`cosine_cnt`：细分相似度指标
- **训练目标**：二分类（label=1 喜欢 / label=0 跳过）
- **模型融合**：`blend_score = (1-0.5) × rule_score + 0.5 × (xgb_score × 5.0)`
- **评估指标**：AUC、准确率、精确率、召回率、F1 分数

---

### 模型文件说明
- **`models/xgb_model.json`**：XGBoost 模型本体（**必须**，用于预测）
- **`models/model_meta.json`**：训练元数据（可选，记录训练时间、指标、特征名）
- **建议**：都保留，便于模型版本管理和性能追踪


---

## ❓ 常见问题

### 网页无法访问
- **问题**：浏览器打开 `http://127.0.0.1:8002/` 无法连接
- **解决**：
  1. 确认服务已启动（终端显示 `Server running on port 8002`）
  2. 检查端口占用（见上方"端口占用问题"部分）
  3. 尝试更换浏览器或清除缓存（Ctrl+F5 硬刷新）

### 点击"匹配职位"无反应
- **可能原因**：
  - 表单数据为空（未填写意向城市/职位）
  - 服务器未启动或崩溃
  - 岗位数据文件缺失（`data/job_data.xlsx`）
- **解决**：
  1. 填写完整表单数据（意向城市、年薪、目标职位）
  2. 重启服务：`cd web && python unified_server.py`
  3. 刷新页面（Ctrl+F5）清除浏览器缓存
  4. 打开浏览器控制台（F12 → Network）查看 API 请求状态

### 智能对话失败（500 错误）
- **问题**：勾选"启用智能对话"后提示"LLM 调用失败"
- **解决**：
  1. 检查 `.env` 文件是否存在且配置正确：
     ```env
     BASE_URL=https://api.deepseek.com/v1  # 必须带 /v1
     API_KEY=sk-xxxxxx  # 替换为你的实际密钥
     MODEL=deepseek-chat
     ```
  2. 确认 API_KEY 有效且有余额
  3. 尝试关闭"启用智能对话"开关，仅使用本地匹配

### XGBoost 训练失败
- **问题**：点击"🚀 训练模型"提示"样本数量不足"或训练失败
- **解决**：
  1. 检查 `logs/recommend_events.jsonl` 是否存在且包含数据
  2. 系统提供冷启动数据，首次使用即可训练
  3. 如需生成更多冷启动数据：
     - **LLM智能生成**：`python logs/generate_realistic_samples.py`（推荐）
  4. 确保至少有 **20+ 条样本**（冷启动数据 + 用户反馈）才能有效训练
  5. 如果训练指标异常（如 AUC=1.0），说明数据过于简单，建议使用 LLM 生成更真实的数据

### 反馈事件无法删除
- **问题**：点击"删除"按钮无效果
- **解决**：
  1. 刷新页面后重试
  2. 检查浏览器控制台是否有错误日志
  3. 重启服务器后再试

### 数据向量化失败
- **问题**：运行 `preprocess_job_data.py` 时提示模块缺失
- **解决**：
  1. 安装必需的依赖：
     ```bash
     pip install gensim pyarrow tqdm
     ```
  2. 确保 `data/job_data.csv` 文件存在
  3. 如果内存不足，可以减小 Word2Vec 的 `vector_size` 参数（默认100）

### 深度学习模型训练失败
- **问题**：训练 MLP/CNN/LSTM 时提示 PyTorch 未安装
- **解决**：
  1. 安装 PyTorch：
     ```bash
     # CPU 版本
     pip install torch torchvision torchaudio
     
     # GPU 版本（需要CUDA）
     # 访问 https://pytorch.org/ 获取对应版本的安装命令
     ```
  2. 确保已运行数据向量化：`python preprocess_job_data.py`
  3. 检查 `data/job_data_vectorized.parquet` 是否存在

### 模型训练AUC过低
- **问题**：训练后 AUC < 0.52，接近随机猜测
- **解决**：
  1. **数据质量问题**：
     - 使用 LLM 生成更真实的训练数据：`python logs/generate_realistic_samples.py`
     - 确保训练数据样本数 ≥ 500 条
     - 检查正负样本比例是否平衡（30-40% 正样本最佳）
  2. **特征工程问题**：
     - 检查 `logs/recommend_events.jsonl` 中的职位数据是否与 `job_data_vectorized.csv` 匹配
     - 确保职位字段（岗位名称、企业、地址）完整
  3. **模型选择问题**：
     - LSTM 不适合表格数据，建议使用 XGBoost
     - MLP 训练不稳定，建议使用 XGBoost

### 为什么 LSTM 的 AUC 这么低？
- **问题**：LSTM 模型的 AUC 仅 0.5044，精确率和召回率都是 0.0000
- **原因**：
  - LSTM 是为**序列数据**设计的（如文本、时间序列）
  - 职位推荐数据是**表格数据**（特征独立，无时序关系）
  - LSTM 期望输入有时间依赖性，而表格数据的特征顺序是任意的
- **建议**：
  - ✅ 使用 **XGBoost**（表格数据的最佳选择）
  - ⚠️ 如需使用神经网络，选择 **MLP** 或 **CNN**
  - ❌ 不要使用 **LSTM**（完全不适合）

---

## 👨‍💻 开发说明

### 技术栈
- **前端**：纯静态 HTML + CSS + JavaScript（无打包工具）
  - Markdown 渲染：`marked.js`
  - 代码高亮：`highlight.js`
  - 按钮特效：`button_effects.js`
- **后端**：Python `http.server.SimpleHTTPRequestHandler`
  - LLM 集成：`langchain` + `langchain-openai`
  - 机器学习：`xgboost` + `scikit-learn` + `torch`（PyTorch）
  - 数据处理：`pandas` + `openpyxl` + `pyarrow`
  - 文本向量化：`gensim`（Word2Vec）
  - 数据可视化：`matplotlib`

### 项目结构
```
job_agent/
├── web/                     # 前端界面
│   ├── index.html          # 主页面
│   ├── app.js              # 交互逻辑
│   ├── styles.css          # 样式
│   ├── button_effects.js   # 按钮特效
│   └── unified_server.py   # 后端服务
├── search/                  # 推荐引擎
│   ├── matcher.py          # 匹配器
│   ├── scorer.py           # 规则评分器
│   └── scorer_xgb.py       # XGBoost 评分器
├── training/                # 模型训练
│   ├── training_utils.py   # 训练工具
│   ├── model_manager.py    # 模型管理器
│   ├── xgb_api.py          # 训练 API
│   ├── train_with_vectorized_data.py  # XGBoost 原始训练
│   ├── train_xgb_optimized.py         # XGBoost 优化训练
│   ├── train_xgb_advanced.py          # XGBoost 高级训练（推荐）
│   ├── train_mlp.py        # MLP 神经网络训练
│   ├── train_cnn.py        # CNN 神经网络训练
│   └── train_lstm.py       # LSTM 神经网络训练
├── features/                # 特征提取
│   ├── extractor.py        # 特征工程
│   └── vectorized_extractor.py  # 向量化特征提取
├── similarity/              # 相似度计算
│   ├── engine.py           # 相似度引擎
│   └── text.py             # 文本处理
├── resume_extract/          # 简历解析
│   └── extractor.py        # LLM 简历提取器
├── data/                    # 数据文件
│   ├── job_data.xlsx       # 岗位数据（XLSX格式）
│   ├── job_data.csv        # 岗位数据（CSV格式，深圳真实招聘数据5000+）
│   ├── job_data.jsonl      # 岗位数据（JSONL格式，优先级最高）
│   ├── position_dictionary.txt  # 职位分类字典（8大类105个职位）
│   ├── job_data_vectorized.csv      # 向量化岗位数据（CSV）
│   ├── job_data_vectorized.parquet  # 向量化岗位数据（Parquet）
│   └── word2vec.model      # Word2Vec 模型文件
├── data_preprocess/         # 数据预处理模块
│   ├── loader.py           # 数据加载器
│   ├── cleaner.py          # 数据清洗
│   ├── text_vectorizer.py  # 文本向量化（Word2Vec）
│   ├── salary_normalizer.py  # 薪资标准化
│   └── location_processor.py  # 地点处理
├── logs/                    # 日志与训练数据
│   ├── recommend_events.jsonl  # 统一的推荐事件（包含冷启动数据和用户反馈）
│   └── generate_realistic_samples.py  # 训练数据生成脚本（LLM智能生成）
├── models/                  # 模型文件
│   ├── xgb_model.json      # XGBoost 模型（原始）
│   ├── xgb_advanced_model.json   # XGBoost 高级模型
│   ├── model_meta.json     # 训练元数据（原始）
│   ├── xgb_advanced_meta.json    # XGBoost 高级元数据
│   ├── mlp_model.pth       # MLP 模型
│   ├── mlp_meta.json       # MLP 元数据
│   ├── cnn_model.pth       # CNN 模型
│   ├── cnn_meta.json       # CNN 元数据
│   ├── lstm_model.pth      # LSTM 模型
│   ├── lstm_meta.json      # LSTM 元数据
│   └── advanced_training_history.png  # 训练曲线图
├── preprocess_job_data.py  # 数据向量化脚本
├── config.py               # 配置文件
├── main.py                 # 命令行入口
├── agent.py                # Agent 核心逻辑
├── models.py               # 数据模型定义
└── requirements.txt        # 依赖列表
```

### 扩展建议
- **前端优化**：
  - 支持 Markdown 表格美化（自定义样式）
  - 添加岗位收藏功能（LocalStorage 持久化）
  - 推荐结果导出（PDF/Excel）
  - 模型性能仪表盘（实时展示 AUC、F1 等指标）
- **后端优化**：
  - 接口鉴权（JWT/API Key）
  - 数据库持久化（SQLite/PostgreSQL）
  - 异步处理（FastAPI + async/await）
  - 模型版本管理（A/B测试）
- **算法优化**：
  - 引入 Sentence-BERT 语义相似度
  - 添加协同过滤推荐
  - 多目标优化（排序 + CTR + 转化率）
  - 在线学习（实时更新模型）

---

## 📄 版权声明

**Copyright © 2025 徐琨博（深圳大学）保留所有权利**  
**All Rights Reserved**

创作日期：2025年11月

### 使用限制
本项目代码及相关文档受中华人民共和国著作权法保护，未经作者书面许可，任何单位或个人不得以任何形式复制、修改、发布、分发或用于商业用途。

### 数据说明
- **真实数据**：使用实际招聘平台的岗位信息（如深圳地区招聘数据），数据已脱敏处理，不包含求职者个人信息。

所有数据处理均符合数据安全与隐私保护相关法律法规要求。

### 学术用途
如需在学术研究中引用本项目，请注明出处：
```
徐琨博. 智能职位推荐系统（job-agent）[CP/OL]. 
https://github.com/blues-kun/job-agent, 2025.
```

### 联系方式
如需获得使用授权或有其他问题，请通过blues924@outlook.com联系作者。
