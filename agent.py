"""
Job Agent 核心
基于 LangChain v1 create_agent 的智能求职助理
"""
import json
from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from models import ResumeProfile
from config import MODEL, BASE_URL, API_KEY, TEMPERATURE, SYSTEM_PROMPT, MatchConfig
from resume_extract.storage import ResumeStorage
from search.matcher import JobMatcher


class JobAgent:
    """智能求职 Agent"""
    
    def __init__(
        self,
        resume_storage: ResumeStorage,
        job_matcher: JobMatcher
    ):
        """
        初始化 Agent
        
        Args:
            resume_storage: 简历存储管理器
            job_matcher: 岗位匹配器
        """
        self.resume_storage = resume_storage
        self.job_matcher = job_matcher
        
        # 创建工具
        self.tools = self._create_tools()
        
        # 初始化模型
        self.model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE
        )
        
        # 创建 agent（使用 LangChain v1）
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )
    
    def _create_tools(self) -> List:
        """创建 Agent 工具"""
        
        @tool
        def get_resume() -> str:
            """获取当前简历的完整信息"""
            resume = self.resume_storage.get()
            if not resume:
                return json.dumps({"error": "简历尚未加载"}, ensure_ascii=False)
            
            return json.dumps(resume.model_dump(), ensure_ascii=False, indent=2)
        
        @tool
        def update_resume(patch: Dict[str, Any]) -> str:
            """
            更新简历信息（部分字段更新，不能修改 position_type_name）
            
            参数说明：
            - patch: 要更新的字段，支持嵌套结构
            
            示例：更新城市
            {"personal_info": {"current_city": "上海"}}
            
            示例：更新薪资期望
            {"work_preferences": {"salary_expectation": {"min_annual_package": 400000}}}
            
            示例：更新工作年限
            {"professional_summary": {"total_experience_years": 5.5}}
            """
            # 禁止修改 position_type_name
            if "work_preferences" in patch:
                wp = patch["work_preferences"]
                if isinstance(wp, dict) and "position_type_name" in wp:
                    # 移除这个字段而不是返回错误
                    wp_copy = dict(wp)
                    wp_copy.pop("position_type_name", None)
                    patch["work_preferences"] = wp_copy
            
            try:
                updated_resume = self.resume_storage.update(patch)
                return json.dumps({
                    "success": True,
                    "message": "简历已更新",
                    "updated_fields": list(patch.keys()),
                    "current_resume": updated_resume.model_dump()
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                import traceback
                return json.dumps({
                    "error": f"更新失败: {str(e)}",
                    "traceback": traceback.format_exc()
                }, ensure_ascii=False)
        
        @tool
        def find_job(limit: Optional[int] = None) -> str:
            """
            根据简历查找匹配的岗位
            
            参数:
            - limit: 返回结果数量（默认返回所有匹配）
            """
            resume = self.resume_storage.get()
            if not resume:
                return json.dumps({
                    "error": "简历尚未加载，无法匹配岗位"
                }, ensure_ascii=False)
            
            # 如果没有指定limit，使用默认值
            if limit is None:
                limit = MatchConfig.DEFAULT_RECOMMENDATION_COUNT * 5  # 返回更多让agent筛选
            
            matches = self.job_matcher.find_matches(
                resume=resume,
                limit=limit
            )
            
            return json.dumps({
                "total_matches": len(matches),
                "jobs": matches
            }, ensure_ascii=False, indent=2)
        
        return [get_resume, update_resume, find_job]
    
    async def ainvoke(self, user_input: str, config: Dict = None) -> Dict[str, Any]:
        """
        异步调用 Agent
        
        Args:
            user_input: 用户输入
            config: 配置参数
            
        Returns:
            Agent 响应
        """
        if config is None:
            config = {}
        
        response = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": user_input}]
        }, config=config)
        
        return response
    
    async def astream(self, user_input: str, config: Dict = None):
        """
        流式输出
        
        Args:
            user_input: 用户输入
            config: 配置参数
            
        Yields:
            流式响应块
        """
        if config is None:
            config = {}
        
        async for chunk in self.agent.astream({
            "messages": [{"role": "user", "content": user_input}]
        }, config=config):
            yield chunk

