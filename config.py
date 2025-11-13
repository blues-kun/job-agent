"""
配置管理模块
加载和管理环境变量、API配置等
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# =========================
# API 配置
# =========================
MODEL = os.environ.get("MODEL", "deepseek-chat")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com/v1")
# 为安全起见，不在代码中设置默认密钥，请通过环境变量或 .env 文件提供
API_KEY = os.environ.get("API_KEY", "")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))

# =========================
# 文件路径配置
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# 岗位数据文件（支持 .xlsx, .csv, .jsonl）
# 优先使用 JSONL 格式（如果存在）
JOBS_FILE_JSONL = BASE_DIR / "岗位数据.jsonl"
JOBS_FILE_CSV = BASE_DIR / "岗位数据.csv"
JOBS_FILE_EXCEL = DATA_DIR / "job_data.xlsx"

# 自动选择可用的数据文件
if JOBS_FILE_JSONL.exists():
    JOBS_FILE = JOBS_FILE_JSONL
elif JOBS_FILE_CSV.exists():
    JOBS_FILE = JOBS_FILE_CSV
else:
    JOBS_FILE = JOBS_FILE_EXCEL

RESUME_FILE = BASE_DIR / "resume.txt"
POSITION_DICT_FILE = BASE_DIR / "position_dictionary.txt"

# =========================
# 匹配算法配置
# =========================
class MatchConfig:
    """匹配算法权重配置"""
    POSITION_TYPE_WEIGHT = 1.5  # 职位类型匹配权重
    LOCATION_WEIGHT = 0.8       # 地点匹配权重
    SALARY_WEIGHT = 1.0         # 薪资匹配权重
    EXPERIENCE_WEIGHT = 0.6     # 工作年限权重
    EDUCATION_WEIGHT = 0.5      # 学历权重
    SCHOOL_LEVEL_WEIGHT = 0.4   # 院校层次权重
    TEXT_SIMILARITY_WEIGHT = 0.9 # 文本相似度权重
    TITLE_INTENT_WEIGHT = 0.3    # 职位标题与意向匹配加分
    
    # 学历等级映射
    EDU_ORDER = {"不限": 0, "大专": 1, "本科": 2, "硕士": 3, "博士": 4, "其他": 1}

# XGBoost 集成
USE_XGB_SCORER = True
XGB_MODEL_PATH = "models/xgb_model.json"
XGB_BLEND_ALPHA = float(os.environ.get("XGB_BLEND_ALPHA", "0.5"))
    
# 默认推荐数量
DEFAULT_RECOMMENDATION_COUNT = 10

# =========================
# Agent 配置
# =========================
SYSTEM_PROMPT = """你是一个专业的智能求职助理。你的职责：

1. **简历管理**：帮助用户完善简历信息，确保所有必要字段（城市、薪资、学历、工作年限等）都已填写。

2. **岗位匹配**：
   - 使用 find_job 工具根据简历自动匹配合适的岗位
   - 匹配考虑：职位类型、地点、薪资、学历、工作年限等多维度
   - 按匹配分数从高到低推荐岗位

3. **智能推荐**：
   - 默认推荐10个最匹配的岗位
   - 为每个推荐提供详细的匹配理由
   - 解释为什么该岗位适合用户

4. **交互指令**：
   - 用户输入 /get_resume 可查看当前简历
   - 用户输入 /quit 结束对话

注意：position_type_name 字段在简历初始化后不可修改。
"""

# =========================
# 日志配置
# =========================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "agent.log"

