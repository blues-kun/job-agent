"""
简历提取器
使用LLM从简历文本中提取结构化信息
"""
import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from models import ResumeProfile
from config import MODEL, BASE_URL, API_KEY, TEMPERATURE


class ResumeExtractor:
    """简历信息提取器"""
    
    def __init__(self, allowed_positions: List[str]):
        """
        初始化提取器
        
        Args:
            allowed_positions: 允许的职位类型列表
        """
        self.allowed_positions = allowed_positions
        self.llm = ChatOpenAI(
            model=MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE
        )
    
    async def extract(self, resume_text: str) -> ResumeProfile:
        """
        从简历文本中提取结构化信息
        
        Args:
            resume_text: 简历文本
            
        Returns:
            ResumeProfile 对象
        """
        # 构建JSON结构示例
        json_structure = """
{
  "personal_info": {
    "current_city": "深圳",
    "willingness_to_relocate": true,
    "availability_date": "2024-08-01"
  },
  "work_preferences": {
    "position_type_name": ["Java", "后端开发"],
    "salary_expectation": {
      "min_annual_package": 300000,
      "currency": "CNY"
    }
  },
  "professional_summary": {
    "total_experience_years": 3.5,
    "education_level": "本科",
    "school_level": 0
  },
  "full_resume_text": "原始简历全文"
}
"""
        
        # 构建职位类型列表
        positions_text = "\n- ".join(self.allowed_positions[:50])  # 限制数量避免太长
        
        messages = [
            SystemMessage(content=(
                "你是专业的简历信息提取助手。请从简历文本中**一次性**提取所有能识别的信息，并以JSON格式返回。"
                "\n\n核心原则："
                "\n❗ **主动提取，智能推断，尽量避免反复询问用户**"
                "\n\n字段提取规则："
                "\n1. position_type_name: 从提供的职位列表中选择最匹配的1-3个职位"
                "\n   - 优先从工作经历、项目经验、技能栈推断职位类型"
                "\n   - 如提到Java/SpringBoot → 选择['Java', 'Java后端', '后端开发']"
                "\n   - 如提到Python/机器学习 → 选择['Python', '机器学习', '算法工程师']"
                "\n\n2. current_city: 从简历中识别当前所在城市或意向城市"
                "\n   - 提到具体城市 → 直接使用"
                "\n   - 未提及 → 从公司地址推断"
                "\n   - 仍无法确定 → 使用'北京'作为默认值"
                "\n\n3. salary_expectation.min_annual_package: 期望年薪(整数)"
                "\n   - 明确提及 → 直接使用"
                "\n   - 根据工作年限推断:"
                "\n     * 0-2年: 150000-250000"
                "\n     * 3-5年: 250000-400000"
                "\n     * 6-8年: 400000-600000"
                "\n     * 9年+: 600000-1000000"
                "\n\n4. total_experience_years: 工作年限(浮点数)"
                "\n   - 从工作经历时间段累计计算"
                "\n   - 未提及 → 根据职位级别推断(初级:1, 中级:3, 高级:5, 资深:8)"
                "\n\n5. education_level: 学历层次"
                "\n   - 明确提及 → 直接使用('大专'/'本科'/'硕士'/'博士')"
                "\n   - 未提及 → 默认'本科'"
                "\n\n6. school_level: 院校层次"
                "\n   - 0=普通本科/未知"
                "\n   - 1=211院校"
                "\n   - 2=985院校"
                "\n\n7. willingness_to_relocate: 是否愿意异地"
                "\n   - 提到'仅限XX城市' → false"
                "\n   - 提到'可接受异地'/'全国' → true"
                "\n   - 未提及 → false"
                "\n\n⚠️ 重要: 所有字段必须有值,不得为null。优先从简历提取,其次智能推断,最后使用合理默认值。"
            )),
            HumanMessage(content=(
                f"可选择的职位类型（position_type_name）：\n- {positions_text}\n\n"
                f"请按以下JSON格式返回：\n{json_structure}\n\n"
                f"=== 简历全文 ===\n{resume_text}\n\n"
                "请直接返回JSON，不要有任何其他文字。"
            ))
        ]
        
        print("正在提取简历结构化信息...")
        
        # 调用LLM（使用简单的JSON模式）
        response = await self.llm.ainvoke(
            messages,
            response_format={"type": "json_object"}
        )
        
        # 解析JSON
        data = json.loads(response.content)
        
        # 过滤并验证职位类型
        position_names = data.get("work_preferences", {}).get("position_type_name", [])
        if isinstance(position_names, list):
            filtered_positions = [
                name for name in position_names 
                if name in self.allowed_positions
            ]
            if not filtered_positions and position_names:
                # 如果没有匹配的，尝试模糊匹配
                for name in position_names:
                    for allowed in self.allowed_positions:
                        if name.lower() in allowed.lower() or allowed.lower() in name.lower():
                            filtered_positions.append(allowed)
                            break
            
            data["work_preferences"]["position_type_name"] = filtered_positions
        
        # 转换为Pydantic模型
        resume_profile = ResumeProfile.model_validate(data)
        
        print("✓ 简历提取完成")
        return resume_profile

