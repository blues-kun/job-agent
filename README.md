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
- **数据生成与清洗**：支持生成虚拟岗位数据（CSV/XLSX/JSONL）、岗位职责清洗、薪资解析

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
- **默认岗位数据**：`data/job_data.xlsx`（可通过生成脚本或清洗脚本更新）
- **生成虚拟数据**：
  ```bash
  python data/generate_fake_jobs.py
  # 生成 data/job_data.csv / .xlsx / .jsonl（包含10000条虚拟岗位数据）
  ```

### 职位分类管理
- **职位三级分类文件**：`data/builtin_positions.txt`
  - 包含所有支持的职位类型（必须与 `generate_fake_jobs.py` 中的 `sec_ter` 字典保持一致）
  - 用于简历解析时的职位类型匹配和验证
  - 格式：每行一个职位名称，`#` 开头为注释，空行会被忽略
  
- **当前支持的职位分类**：
  - 后端开发：Java、Python、Golang、C/C++、PHP、Node.js、全栈工程师
  - 前端/移动开发：前端开发工程师、Android、iOS、鸿蒙开发工程师
  - 测试：测试工程师、自动化测试、测试开发、性能测试
  - 运维/技术支持：运维工程师、IT技术支持、网络工程师、系统工程师、DBA
  - 人工智能：算法工程师、机器学习、深度学习、推荐算法、数据挖掘
  - 数据：数据分析师、数据开发、数据仓库、数据治理

**⚠️ 重要**：修改职位分类时，需要同步更新以下文件：
1. `data/builtin_positions.txt` - 职位分类列表
2. `data/generate_fake_jobs.py` - 数据生成脚本中的 `sec_ter` 字典
3. 重新运行 `python data/generate_fake_jobs.py` 生成新的岗位数据

### 简历输入
- **命令行模式**：按提示粘贴简历，自动保存 `resume.json`
- **网页模式**：在顶部对话框粘贴简历或上传 `.txt` 文件

### 训练数据生成

系统提供两种训练数据生成方式：

**方式一：纯算法规则生成**（快速但质量一般）
```bash
python logs/regenerate_training_data.py
```
- **特点**：使用Python规则和随机算法生成训练样本
- **优势**：生成速度快，不依赖LLM API
- **适用场景**：快速测试、初期原型验证

**方式二：LLM智能生成**（慢速但质量高）🌟 **推荐**
```bash
python logs/generate_realistic_samples.py
```
- **特点**：调用大语言模型分析职位和简历的真实匹配度
- **优势**：生成数据更接近真实用户行为，模型训练效果更好
- **要求**：需要配置 `.env` 中的 API_KEY
- **数据来源**：从 `data/job_data.csv` 和 `data/builtin_positions.txt` 获取真实职位数据
- **生成策略**：
  - 分层采样：高匹配、中匹配、低匹配职位均衡分布
  - LLM判断：根据城市匹配、薪资匹配、职位类型等维度智能决策
  - 正负样本平衡：确保约 30-40% 的正样本比例

**推荐做法**：
1. 首次使用运行 `generate_realistic_samples.py` 生成高质量冷启动数据
2. 系统运行后收集真实用户反馈
3. 定期重新训练模型，融合冷启动数据和用户反馈

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
- **🔄 刷新列表**：重新加载反馈事件和冷启动数据
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

### 基础接口
- **获取岗位数据**：`GET /api/jobs`
  - 返回：结构化岗位列表（JSON）

### 核心推荐接口
- **对话与简历抽取**：`POST /api/chat_resume`
  - 请求体：`{"messages": [...], "use_llm": true/false, "extract_resume": true/false}`
  - 返回：`{"assistant_reply": str, "resume_profile": obj|null}`
  
- **简历信息增强**：`POST /api/resume_enhance`
  - 请求体：`{"resume_text": str, "current_profile": obj, "messages": [...]}`
  - 返回：`{"enhanced_profile": obj, "is_complete": bool, "assistant_reply": str}`
  - 功能：智能补全简历信息，不询问隐私数据（姓名、电话、邮箱）
  
- **职位推荐**：`POST /api/recommend`
  - 请求体：`{"resume": obj, "limit": 10, "min_score": 0.5, "use_llm": true/false, "use_xgb": true/false}`
  - 返回：`{"jobs": [...], "assistant_reply": str|null}`

### 反馈与模型训练
- **用户反馈提交**：`POST /api/feedback`
  - 请求体：`{"action": "like"|"skip", "job": obj, "resume": obj}`
  - 返回：`{"ok": true}`
  - 功能：记录用户对职位的喜好，自动保存到 `logs/recommend_events.jsonl`

- **推荐事件管理**：`POST /api/recommend_events`
  - 操作类型：
    - `list`：获取所有推荐事件（包含冷启动数据和用户反馈）
    - `update`：更新事件（标注备注）
    - `delete`：删除指定事件
  
- **XGBoost 训练**：`POST /api/xgb_ops`
  - 操作类型：
    - `train`：训练 XGBoost 模型，自动从 `recommend_events.jsonl` 加载数据并提取特征，返回训练指标

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
- **岗位数据生成**：`data/generate_fake_jobs.py` - 生成虚拟岗位数据
- **训练数据生成**：
  - `logs/regenerate_training_data.py` - 纯算法规则生成用户反馈样本
  - `logs/generate_realistic_samples.py` - LLM智能生成高质量用户反馈样本（推荐）
- **清洗与薪资解析**：`data_preprocess/cleaner.py`
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
     - **快速生成**：`python logs/regenerate_training_data.py`（纯算法规则）
     - **高质量生成**：`python logs/generate_realistic_samples.py`（LLM智能生成，推荐）
  4. 确保至少有 **20+ 条样本**（冷启动数据 + 用户反馈）才能有效训练
  5. 如果训练指标异常（如 AUC=1.0），说明数据过于简单，建议使用 LLM 生成更真实的数据

### 反馈事件无法删除
- **问题**：点击"删除"按钮无效果
- **解决**：
  1. 刷新页面后重试
  2. 检查浏览器控制台是否有错误日志
  3. 重启服务器后再试

---

## 👨‍💻 开发说明

### 技术栈
- **前端**：纯静态 HTML + CSS + JavaScript（无打包工具）
  - Markdown 渲染：`marked.js`
  - 代码高亮：`highlight.js`
  - 按钮特效：`button_effects.js`
- **后端**：Python `http.server.SimpleHTTPRequestHandler`
  - LLM 集成：`langchain` + `langchain-openai`
  - 机器学习：`xgboost` + `scikit-learn`
  - 数据处理：`pandas` + `openpyxl`

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
│   └── xgb_api.py          # 训练 API
├── features/                # 特征提取
│   └── extractor.py        # 特征工程
├── similarity/              # 相似度计算
│   ├── engine.py           # 相似度引擎
│   └── text.py             # 文本处理
├── resume_extract/          # 简历解析
│   └── extractor.py        # LLM 简历提取器
├── data/                    # 数据文件
│   ├── job_data.xlsx       # 岗位数据（XLSX格式，默认使用）
│   ├── job_data.csv        # 岗位数据（CSV格式）
│   ├── job_data.jsonl      # 岗位数据（JSONL格式，优先级最高）
│   ├── builtin_positions.txt  # 职位三级分类列表
│   └── generate_fake_jobs.py  # 虚拟数据生成脚本
├── logs/                    # 日志与训练数据
│   ├── recommend_events.jsonl  # 统一的推荐事件（包含冷启动数据和用户反馈）
│   ├── regenerate_training_data.py  # 训练数据生成脚本（纯算法规则）
│   └── generate_realistic_samples.py  # 训练数据生成脚本（LLM智能生成）
├── models/                  # 模型文件
│   ├── xgb_model.json      # XGBoost 模型
│   └── model_meta.json     # 训练元数据
├── config.py               # 配置文件
├── main.py                 # 命令行入口
└── requirements.txt        # 依赖列表
```

### 扩展建议
- **前端优化**：
  - 支持 Markdown 表格美化（自定义样式）
  - 添加岗位收藏功能（LocalStorage 持久化）
  - 推荐结果导出（PDF/Excel）
- **后端优化**：
  - 接口鉴权（JWT/API Key）
  - 数据库持久化（SQLite/PostgreSQL）
  - 异步处理（FastAPI + async/await）
- **算法优化**：
  - 引入 Sentence-BERT 语义相似度
  - 添加协同过滤推荐
  - 多目标优化（排序 + CTR + 转化率）

---

## 📄 版权声明

**Copyright © 2025 徐琨博（深圳大学）保留所有权利**  
**All Rights Reserved**

创作日期：2025年11月

### 使用限制
本项目代码及相关文档受中华人民共和国著作权法保护，未经作者书面许可，任何单位或个人不得以任何形式复制、修改、发布、分发或用于商业用途。

### 数据说明
本项目使用的岗位数据为参考真实职位样例通过代码生成的虚拟数据，不涉及任何真实个人信息或企业敏感数据，符合数据安全与隐私保护相关法律法规要求。

### 学术用途
如需在学术研究中引用本项目，请注明出处：
```
徐琨博. 智能职位推荐系统（job-agent）[CP/OL]. 
https://github.com/blues-kun/job-agent, 2025.
```

### 联系方式
如需获得使用授权或有其他问题，请通过blues924@outlook.com联系作者。
