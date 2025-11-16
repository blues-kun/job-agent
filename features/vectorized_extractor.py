"""
向量化特征提取器
基于Word2Vec等向量化方法提取特征
"""
from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from models import ResumeProfile
from data_preprocess.text_vectorizer import TextVectorizer
from data_preprocess.salary_normalizer import SalaryNormalizer
from data_preprocess.location_processor import LocationProcessor


class VectorizedFeatureExtractor:
    """向量化特征提取器"""
    
    def __init__(self, vector_size: int = 100, model_dir: str = 'models/word2vec'):
        """
        初始化
        
        Args:
            vector_size: 向量维度
            model_dir: 模型目录
        """
        self.vector_size = vector_size
        self.model_dir = Path(model_dir)
        self.vectorizers = {}
        self._load_vectorizers()
    
    def _load_vectorizers(self):
        """加载预训练的向量化模型"""
        if not self.model_dir.exists():
            print(f"警告: 模型目录不存在: {self.model_dir}")
            return
        
        # 加载各个文本字段的向量化器
        text_fields = ['岗位职责', '岗位要求', '岗位名称']
        
        for field in text_fields:
            model_path = self.model_dir / f"{field}_w2v.model"
            if model_path.exists():
                try:
                    vectorizer = TextVectorizer(vector_size=self.vector_size)
                    vectorizer.load(str(model_path))
                    self.vectorizers[field] = vectorizer
                    print(f"✓ 已加载 {field} 向量化模型")
                except Exception as e:
                    print(f"警告: 加载 {field} 模型失败: {e}")
    
    def extract(self, resume: ResumeProfile, job: Dict[str, Any]) -> Dict[str, float]:
        """
        提取特征
        
        Args:
            resume: 简历对象
            job: 职位字典
            
        Returns:
            特征字典
        """
        features = {}
        
        # 1. 基础数值特征
        features.update(self._extract_basic_features(resume, job))
        
        # 2. 向量化文本特征
        features.update(self._extract_text_features(resume, job))
        
        return features
    
    def _extract_basic_features(
        self, 
        resume: ResumeProfile, 
        job: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取基础数值特征"""
        features = {}
        
        # 1. 地理位置匹配
        city = resume.personal_info.current_city
        job_addr = str(job.get('岗位地址', '') or '')
        job_city = str(job.get('城市', '') or '')
        features['location_match'] = 1.0 if (city and (city in job_addr or city in job_city)) else 0.0
        features['willingness_relocate'] = 1.0 if resume.personal_info.willingness_to_relocate else 0.0
        
        # 2. 薪资相关
        salary_info = SalaryNormalizer.normalize(job.get('岗位薪资', ''))
        user_expect = float(resume.work_preferences.salary_expectation.min_annual_package or 0)
        job_avg = float(salary_info.get('avg_annual') or 0)
        
        features['salary_ratio'] = (job_avg / user_expect) if (user_expect and job_avg) else 0.0
        features['salary_negotiable'] = 1.0 if salary_info.get('salary_type') == 'negotiable' else 0.0
        features['salary_months'] = float(salary_info.get('months', 12))
        features['user_expect_salary'] = user_expect / 100000  # 归一化到0-10范围
        features['job_avg_salary'] = job_avg / 100000  # 归一化
        
        # 3. 工作经验
        import re
        req_text = str(job.get('岗位要求', '') or '')
        exp_match = re.search(r'(\d+)[-~年]', req_text)
        req_years = float(exp_match.group(1)) if exp_match else 0.0
        user_years = float(resume.professional_summary.total_experience_years or 0.0)
        
        features['experience_ratio'] = (user_years / max(req_years, 1.0)) if user_years else 0.0
        features['user_experience'] = user_years
        features['required_experience'] = req_years
        
        # 4. 学历匹配
        edu_order = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}
        user_edu = edu_order.get(resume.professional_summary.education_level, 0)
        
        req_edu = 0
        for edu, level in edu_order.items():
            if edu in req_text:
                req_edu = max(req_edu, level)
        
        features['education_match'] = 1.0 if user_edu >= req_edu else 0.0
        features['user_education'] = float(user_edu)
        features['required_education'] = float(req_edu)
        
        # 5. 院校层次
        features['school_level'] = float(resume.professional_summary.school_level or 0)
        features['school_985'] = 1.0 if resume.professional_summary.school_level == 2 else 0.0
        features['school_211'] = 1.0 if resume.professional_summary.school_level >= 1 else 0.0
        
        return features
    
    def _extract_text_features(
        self, 
        resume: ResumeProfile, 
        job: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取文本向量化特征"""
        features = {}
        
        if not self.vectorizers:
            # 如果没有加载模型，使用简单的文本相似度
            return self._fallback_text_features(resume, job)
        
        # 1. 简历全文向量
        resume_text = resume.full_resume_text or ''
        if 'full_resume' in self.vectorizers:
            resume_vec = self._text_to_vector(resume_text, 'full_resume')
        else:
            # 合并意向作为简历表示
            intent_text = ' '.join(resume.work_preferences.position_type_name)
            resume_vec = self._simple_text_vector(intent_text)
        
        # 2. 岗位各字段向量
        job_title = str(job.get('岗位名称', '') or '')
        job_desc = str(job.get('岗位职责', '') or '')
        job_req = str(job.get('岗位要求', '') or '')
        
        title_vec = self._text_to_vector(job_title, '岗位名称')
        desc_vec = self._text_to_vector(job_desc, '岗位职责')
        req_vec = self._text_to_vector(job_req, '岗位要求')
        
        # 3. 计算相似度特征
        features['resume_title_sim'] = self._cosine_similarity(resume_vec, title_vec)
        features['resume_desc_sim'] = self._cosine_similarity(resume_vec, desc_vec)
        features['resume_req_sim'] = self._cosine_similarity(resume_vec, req_vec)
        
        # 4. 职位意向匹配
        intent_match = 0.0
        for intent in resume.work_preferences.position_type_name:
            if intent.lower() in job_title.lower():
                intent_match += 1.0
            if intent.lower() in str(job.get('三级分类', '')).lower():
                intent_match += 0.5
        features['intent_match_score'] = min(intent_match, 3.0) / 3.0  # 归一化到0-1
        
        return features
    
    def _text_to_vector(self, text: str, field: str) -> np.ndarray:
        """将文本转换为向量"""
        if field in self.vectorizers:
            vectorizer = self.vectorizers[field]
            vec = vectorizer.transform([text])[0]
            return vec
        else:
            return self._simple_text_vector(text)
    
    def _simple_text_vector(self, text: str) -> np.ndarray:
        """简单的文本向量化（后备方案）"""
        # 使用字符级统计作为简单特征
        return np.zeros(self.vector_size)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _fallback_text_features(
        self, 
        resume: ResumeProfile, 
        job: Dict[str, Any]
    ) -> Dict[str, float]:
        """后备的文本特征提取（不依赖向量化模型）"""
        features = {}
        
        # 简单的关键词匹配
        resume_keywords = set(resume.work_preferences.position_type_name)
        job_title = str(job.get('岗位名称', '') or '').lower()
        job_type = str(job.get('三级分类', '') or '').lower()
        
        match_count = sum(1 for kw in resume_keywords if kw.lower() in job_title or kw.lower() in job_type)
        
        features['keyword_match'] = min(match_count / max(len(resume_keywords), 1), 1.0)
        features['resume_title_sim'] = 0.5 if match_count > 0 else 0.0
        features['resume_desc_sim'] = 0.3
        features['resume_req_sim'] = 0.3
        features['intent_match_score'] = features['keyword_match']
        
        return features


def extract_vectorized_features(
    resume: ResumeProfile, 
    job: Dict[str, Any],
    vector_size: int = 100,
    model_dir: str = 'models/word2vec'
) -> Dict[str, float]:
    """
    便捷函数：提取向量化特征
    
    Args:
        resume: 简历对象
        job: 职位字典
        vector_size: 向量维度
        model_dir: 模型目录
        
    Returns:
        特征字典
    """
    extractor = VectorizedFeatureExtractor(vector_size=vector_size, model_dir=model_dir)
    return extractor.extract(resume, job)

