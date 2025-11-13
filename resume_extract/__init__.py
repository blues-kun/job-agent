"""
简历提取模块
负责从文本中提取结构化简历信息
"""
from .extractor import ResumeExtractor
from .storage import ResumeStorage

__all__ = ['ResumeExtractor', 'ResumeStorage']

