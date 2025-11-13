"""
数据模型定义
定义简历、岗位等核心数据结构
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# =========================
# 简历相关模型
# =========================
class SalaryExpectation(BaseModel):
    """薪资期望"""
    min_annual_package: int = Field(description="最低可接受年包（元）")
    currency: str = Field(default="CNY", description="币种")


class PersonalInfo(BaseModel):
    """个人信息"""
    current_city: str = Field(description="当前所在城市")
    willingness_to_relocate: bool = Field(default=False, description="是否愿意异地工作")
    availability_date: Optional[str] = Field(default=None, description="最快可入职日期 YYYY-MM-DD")


class WorkPreferences(BaseModel):
    """工作偏好"""
    position_type_name: List[str] = Field(description="期望的职位类型名称（从字典中选择）")
    salary_expectation: SalaryExpectation


class ProfessionalSummary(BaseModel):
    """专业背景"""
    total_experience_years: float = Field(description="总工作年限（年）")
    education_level: str = Field(description="最高学历：博士/硕士/本科/大专/其他")
    school_level: int = Field(default=0, description="院校层次：0=无/未知，1=211，2=985")


class ResumeProfile(BaseModel):
    """完整简历"""
    personal_info: PersonalInfo
    work_preferences: WorkPreferences
    professional_summary: ProfessionalSummary
    full_resume_text: str = Field(description="原始简历全文")


# =========================
# 岗位匹配结果模型
# =========================
class JobMatchResult(BaseModel):
    """单个岗位匹配结果"""
    company_name: Optional[str] = Field(description="公司名称")
    job_title: Optional[str] = Field(description="岗位名称")
    position_type_name: Optional[str] = Field(description="职位类型")
    secondary_category: Optional[str] = Field(description="二级分类")
    tertiary_category: Optional[str] = Field(description="三级分类")
    city: Optional[str] = Field(description="城市")
    salary: Optional[str] = Field(description="薪资范围")
    requirements: Optional[str] = Field(description="岗位要求")
    score: float = Field(description="匹配分数")
    reasons: List[str] = Field(description="匹配理由")


class JobRecommendation(BaseModel):
    """岗位推荐结果（用于结构化输出）"""
    recommended_jobs: List[JobMatchResult] = Field(description="推荐的岗位列表")
    summary: str = Field(description="推荐总结")
    total_matches: int = Field(description="总匹配数量")


# =========================
# 工具参数模型
# =========================
class UpdateResumeArgs(BaseModel):
    """更新简历的参数"""
    personal_info: Optional[PersonalInfo] = None
    salary_expectation: Optional[SalaryExpectation] = None
    total_experience_years: Optional[float] = None
    education_level: Optional[str] = None
    school_level: Optional[int] = None


class FindJobArgs(BaseModel):
    """查找岗位的参数"""
    limit: Optional[int] = Field(default=None, description="返回结果数量限制")
    filter_by_location: Optional[bool] = Field(default=True, description="是否按地点过滤")

