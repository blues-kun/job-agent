"""
匹配评分器
计算简历与岗位的匹配分数
"""
import json
from typing import Dict, Any, List, Tuple, Optional

from models import ResumeProfile
from config import MatchConfig, USE_XGB_SCORER, XGB_MODEL_PATH, XGB_BLEND_ALPHA
from data_preprocess.cleaner import DataCleaner
from similarity.engine import SimilarityEngine


class MatchScorer:
    """岗位匹配评分器"""
    
    @staticmethod
    def calculate_score(resume: ResumeProfile, job: Dict[str, Any], use_xgb: Optional[bool] = None) -> Tuple[float, List[str]]:
        """
        计算匹配分数
        
        Args:
            resume: 简历对象
            job: 岗位字典
            
        Returns:
            (分数, 匹配理由列表)
        """
        score = 0.0
        reasons = []
        
        # 1. 职位类型匹配（最重要）
        job_position = job.get('三级分类', '')
        if job_position and any(
            job_position.strip().lower() == pos.strip().lower() 
            for pos in resume.work_preferences.position_type_name
        ):
            score += MatchConfig.POSITION_TYPE_WEIGHT
            reasons.append(f"✓ 职位类型匹配: {job_position}")
        
        # 2. 地点匹配
        if not resume.personal_info.willingness_to_relocate:
            if MatchScorer._check_location(job, resume.personal_info.current_city):
                score += MatchConfig.LOCATION_WEIGHT
                reasons.append(f"✓ 地点匹配: {resume.personal_info.current_city}")
            else:
                reasons.append(f"✗ 地点不匹配")
        else:
            score += 0.2
            reasons.append("✓ 候选人可异地工作")
        
        # 3. 薪资匹配
        salary_score, salary_reason = MatchScorer._check_salary(
            job, 
            resume.work_preferences.salary_expectation.min_annual_package
        )
        score += salary_score
        reasons.append(salary_reason)
        
        # 4. 工作年限匹配
        exp_score, exp_reason = MatchScorer._check_experience(
            job,
            resume.professional_summary.total_experience_years
        )
        score += exp_score
        reasons.append(exp_reason)
        
        # 5. 学历匹配
        edu_score, edu_reason = MatchScorer._check_education(
            job,
            resume.professional_summary.education_level
        )
        score += edu_score
        reasons.append(edu_reason)
        
        # 6. 院校层次匹配
        school_score, school_reason = MatchScorer._check_school_level(
            job,
            resume.professional_summary.school_level
        )
        if school_score > 0:
            score += school_score
            reasons.append(school_reason)

        # 7. 文本相似度
        text_score, text_reason = MatchScorer._check_text_similarity(resume, job)
        if text_score > 0:
            score += text_score
            reasons.append(text_reason)

        # 8. 标题意向匹配
        intent_score, intent_reason = MatchScorer._check_title_intent(resume, job)
        if intent_score > 0:
            score += intent_score
            reasons.append(intent_reason)

        # 9. XGB综合打分融合
        flag = USE_XGB_SCORER if use_xgb is None else bool(use_xgb)
        if flag:
            xgb_score = MatchScorer._xgb_score(resume, job)
            if xgb_score > 0:
                alpha = max(0.0, min(1.0, XGB_BLEND_ALPHA))
                blend = (1 - alpha) * score + alpha * (xgb_score * 5.0)
                reasons.append(f"✓ XGB综合评分: {round(xgb_score,3)}")
                score = blend
        
        return round(score, 3), reasons
    
    @staticmethod
    def _check_location(job: Dict[str, Any], city: str) -> bool:
        """检查地点是否匹配"""
        try:
            address = str(job.get('岗位地址', '') or '')
            job_city = str(job.get('城市', '') or '')
            if city and city in address:
                return True
            if city and job_city and city in job_city:
                return True
            return False
        except Exception:
            return False
    
    @staticmethod
    def _check_salary(job: Dict[str, Any], expected_min: int) -> Tuple[float, str]:
        """检查薪资是否匹配"""
        salary_str = job.get('岗位薪资', '')
        
        if not salary_str or salary_str == '面议':
            return MatchConfig.SALARY_WEIGHT * 0.4, "⚠ 薪资面议"
        
        # 解析薪资
        salary_info = DataCleaner.parse_salary_range(salary_str)
        max_salary = salary_info.get('max')
        
        if max_salary and max_salary >= expected_min:
            return (
                MatchConfig.SALARY_WEIGHT,
                f"✓ 薪资满足: 最高{max_salary//10000}万 ≥ 期望{expected_min//10000}万"
            )
        elif max_salary:
            return (
                0,
                f"✗ 薪资不足: 最高{max_salary//10000}万 < 期望{expected_min//10000}万"
            )
        else:
            return 0, "⚠ 薪资信息不明确"
    
    @staticmethod
    def _check_experience(job: Dict[str, Any], candidate_years: float) -> Tuple[float, str]:
        """检查工作年限是否匹配"""
        requirements = job.get('岗位要求', '')
        if not requirements:
            return MatchConfig.EXPERIENCE_WEIGHT * 0.5, "⚠ 无明确年限要求"
        
        # 简单匹配（实际应该用更复杂的解析）
        import re
        exp_match = re.search(r'(\d+)[年\-~]+', str(requirements))
        
        if exp_match:
            required_years = int(exp_match.group(1))
            if candidate_years >= required_years:
                return (
                    MatchConfig.EXPERIENCE_WEIGHT,
                    f"✓ 年限满足: {candidate_years}年 ≥ {required_years}年"
                )
            else:
                return (
                    0,
                    f"✗ 年限不足: {candidate_years}年 < {required_years}年"
                )
        
        return MatchConfig.EXPERIENCE_WEIGHT * 0.3, "⚠ 年限要求不明确"
    
    @staticmethod
    def _check_education(job: Dict[str, Any], candidate_edu: str) -> Tuple[float, str]:
        """检查学历是否匹配"""
        requirements = str(job.get('岗位要求', ''))
        
        # 查找学历要求
        edu_keywords = ['博士', '硕士', '本科', '大专']
        required_edu = None
        
        for edu in edu_keywords:
            if edu in requirements:
                required_edu = edu
                break
        
        if not required_edu:
            return MatchConfig.EDUCATION_WEIGHT * 0.5, "⚠ 无明确学历要求"
        
        # 比较学历
        candidate_rank = MatchConfig.EDU_ORDER.get(candidate_edu, 0)
        required_rank = MatchConfig.EDU_ORDER.get(required_edu, 0)
        
        if candidate_rank >= required_rank:
            return (
                MatchConfig.EDUCATION_WEIGHT,
                f"✓ 学历满足: {candidate_edu} ≥ {required_edu}"
            )
        else:
            return (
                0,
                f"✗ 学历不足: {candidate_edu} < {required_edu}"
            )
    
    @staticmethod
    def _check_school_level(job: Dict[str, Any], candidate_level: int) -> Tuple[float, str]:
        """检查院校层次是否匹配"""
        requirements = str(job.get('岗位要求', ''))
        
        # 检查是否要求985/211
        if '985' in requirements:
            if candidate_level == 2:
                return MatchConfig.SCHOOL_LEVEL_WEIGHT, "✓ 院校层次: 985"
            else:
                return 0, "✗ 院校层次: 需要985"
        elif '211' in requirements:
            if candidate_level >= 1:
                level_name = '985' if candidate_level == 2 else '211'
                return MatchConfig.SCHOOL_LEVEL_WEIGHT, f"✓ 院校层次: {level_name}"
            else:
                return 0, "✗ 院校层次: 需要211"
        
        return 0, ""

    @staticmethod
    def _check_text_similarity(resume: ResumeProfile, job: Dict[str, Any]) -> Tuple[float, str]:
        engine = SimilarityEngine()
        s, j, c = engine.text_similarity(resume, job)
        if s <= 0:
            return 0.0, ""
        score = MatchConfig.TEXT_SIMILARITY_WEIGHT * s
        return score, f"✓ 文本相似度: {round(s, 3)} (Jaccard {round(j, 3)}, Cosine {round(c, 3)})"

    @staticmethod
    def _check_title_intent(resume: ResumeProfile, job: Dict[str, Any]) -> Tuple[float, str]:
        engine = SimilarityEngine()
        matched, count = engine.title_intent_match(resume, job)
        if not matched:
            return 0.0, ""
        return MatchConfig.TITLE_INTENT_WEIGHT, f"✓ 标题意向匹配: 关键词重合{count}"

    @staticmethod
    def _xgb_score(resume: ResumeProfile, job: Dict[str, Any]) -> float:
        try:
            from search.scorer_xgb import is_ready, score as xgb_score
            if not is_ready(XGB_MODEL_PATH):
                return 0.0
            return float(xgb_score(resume, job, XGB_MODEL_PATH))
        except Exception:
            return 0.0

