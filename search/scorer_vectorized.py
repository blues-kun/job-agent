"""
基于向量化特征的评分器
完全数据驱动，不使用规则打分
"""
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from models import ResumeProfile
from config import XGB_MODEL_PATH


class VectorizedScorer:
    """向量化评分器"""
    
    def __init__(self, model_path: str = XGB_MODEL_PATH):
        """
        初始化评分器
        
        Args:
            model_path: XGBoost模型路径
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if not Path(self.model_path).exists():
            print(f"警告: 模型文件不存在: {self.model_path}")
            return
        
        try:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            
            # 加载特征名称
            meta_path = Path(self.model_path).parent / 'model_meta.json'
            if meta_path.exists():
                import json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.feature_names = meta.get('feature_names', [])
            
            print(f"✓ 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"警告: 模型加载失败: {e}")
    
    def score(self, resume: ResumeProfile, job: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        计算匹配分数
        
        Args:
            resume: 简历对象
            job: 职位字典
            
        Returns:
            (分数[0-5], 匹配理由列表)
        """
        if self.model is None:
            return 0.0, ["模型未加载"]
        
        try:
            from features.vectorized_extractor import VectorizedFeatureExtractor
            
            # 提取特征
            extractor = VectorizedFeatureExtractor(vector_size=100, model_dir='models/word2vec')
            features = extractor.extract(resume, job)
            
            # 按照训练时的特征顺序排列
            if self.feature_names:
                X = [[features.get(k, 0.0) for k in self.feature_names]]
            else:
                X = [[features[k] for k in sorted(features.keys())]]
            
            # 预测
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names or sorted(features.keys()))
            prob = float(self.model.predict(dmatrix)[0])
            
            # 转换为0-5分制
            score = prob * 5.0
            
            # 生成理由
            reasons = self._generate_reasons(features, prob)
            
            return round(score, 3), reasons
            
        except Exception as e:
            print(f"评分失败: {e}")
            return 0.0, [f"评分失败: {str(e)}"]
    
    def _generate_reasons(self, features: Dict[str, float], prob: float) -> List[str]:
        """
        根据特征生成匹配理由
        
        Args:
            features: 特征字典
            prob: 预测概率
            
        Returns:
            理由列表
        """
        reasons = []
        
        # 1. 整体匹配度
        if prob >= 0.8:
            reasons.append(f"✓ 高度匹配 (匹配度: {prob*100:.1f}%)")
        elif prob >= 0.6:
            reasons.append(f"✓ 较好匹配 (匹配度: {prob*100:.1f}%)")
        elif prob >= 0.4:
            reasons.append(f"⚠ 一般匹配 (匹配度: {prob*100:.1f}%)")
        else:
            reasons.append(f"✗ 匹配度较低 (匹配度: {prob*100:.1f}%)")
        
        # 2. 关键特征分析
        if features.get('location_match', 0) > 0.5:
            reasons.append("✓ 地点匹配")
        
        salary_ratio = features.get('salary_ratio', 0)
        if salary_ratio >= 1.0:
            reasons.append(f"✓ 薪资满足期望 (比例: {salary_ratio:.1f})")
        elif salary_ratio >= 0.8:
            reasons.append(f"⚠ 薪资接近期望 (比例: {salary_ratio:.1f})")
        elif salary_ratio > 0:
            reasons.append(f"✗ 薪资低于期望 (比例: {salary_ratio:.1f})")
        
        exp_ratio = features.get('experience_ratio', 0)
        if exp_ratio >= 1.0:
            reasons.append(f"✓ 经验符合要求 (年限比例: {exp_ratio:.1f})")
        elif exp_ratio >= 0.8:
            reasons.append(f"⚠ 经验接近要求 (年限比例: {exp_ratio:.1f})")
        
        if features.get('education_match', 0) > 0.5:
            reasons.append("✓ 学历符合要求")
        
        if features.get('intent_match_score', 0) > 0.5:
            reasons.append("✓ 职位意向匹配")
        
        # 3. 文本相似度
        resume_title_sim = features.get('resume_title_sim', 0)
        if resume_title_sim > 0.6:
            reasons.append(f"✓ 与职位标题高度相关 (相似度: {resume_title_sim:.2f})")
        elif resume_title_sim > 0.3:
            reasons.append(f"⚠ 与职位标题部分相关 (相似度: {resume_title_sim:.2f})")
        
        return reasons


def calculate_score_vectorized(
    resume: ResumeProfile, 
    job: Dict[str, Any],
    model_path: str = XGB_MODEL_PATH
) -> Tuple[float, List[str]]:
    """
    便捷函数：计算向量化评分
    
    Args:
        resume: 简历对象
        job: 职位字典
        model_path: 模型路径
        
    Returns:
        (分数, 理由列表)
    """
    scorer = VectorizedScorer(model_path=model_path)
    return scorer.score(resume, job)

