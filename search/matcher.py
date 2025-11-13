"""
岗位匹配器
筛选和排序匹配的岗位
"""
from typing import List, Dict, Any
from models import ResumeProfile
from search.scorer import MatchScorer


class JobMatcher:
    """岗位匹配器"""
    
    def __init__(self, jobs_db: List[Dict[str, Any]]):
        """
        初始化匹配器
        
        Args:
            jobs_db: 岗位数据库
        """
        self.jobs_db = jobs_db
        self.scorer = MatchScorer()
    
    def find_matches(
        self,
        resume: ResumeProfile,
        limit: int = None,
        min_score: float = 0.5,
        use_xgb: bool | None = None
    ) -> List[Dict[str, Any]]:
        """
        查找匹配的岗位
        
        Args:
            resume: 简历对象
            limit: 返回数量限制
            min_score: 最低分数阈值
            
        Returns:
            匹配的岗位列表（已排序）
        """
        matches = []
        
        print(f"正在从 {len(self.jobs_db)} 个岗位中筛选匹配...")
        
        for job in self.jobs_db:
            # 1. 预过滤：职位类型必须匹配
            if not self._match_position_type(job, resume):
                continue
            
            # 2. 预过滤：地点要求
            if not self._match_location(job, resume):
                continue
            
            # 3. 计算详细匹配分数
            score, reasons = self.scorer.calculate_score(resume, job, use_xgb=use_xgb)
            
            # 4. 过滤低分
            if score < min_score:
                continue
            
            # 5. 添加到结果
            company = job.get('公司') or job.get('企业')
            city = job.get('城市') or job.get('岗位地址')
            matches.append({
                'company_name': company,
                'job_title': job.get('岗位名称'),
                'position_type_name': job.get('三级分类'),
                'secondary_category': job.get('二级分类'),
                'tertiary_category': job.get('三级分类'),
                'city': city,
                'salary': job.get('岗位薪资'),
                'requirements': job.get('岗位要求'),
                'score': score,
                'reasons': reasons,
                'raw': job
            })
        
        # 按分数排序（降序）
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # 限制数量
        if limit:
            matches = matches[:limit]
        
        print(f"✓ 找到 {len(matches)} 个匹配岗位")
        
        return matches
    
    def _match_position_type(self, job: Dict[str, Any], resume: ResumeProfile) -> bool:
        """检查职位类型是否匹配"""
        job_position = job.get('三级分类', '')
        if not job_position:
            return False
        
        return any(
            job_position.strip().lower() == pos.strip().lower()
            for pos in resume.work_preferences.position_type_name
        )
    
    def _match_location(self, job: Dict[str, Any], resume: ResumeProfile) -> bool:
        """检查地点是否匹配"""
        # 如果愿意异地，直接通过
        if resume.personal_info.willingness_to_relocate:
            return True
        
        # 否则检查是否包含当前城市
        city = resume.personal_info.current_city
        job_address = str(job.get('岗位地址', '') or '')
        job_city = str(job.get('城市', '') or '')
        if city and city in job_address:
            return True
        if city and job_city and city in job_city:
            return True
        return False

