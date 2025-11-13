"""
数据清洗器
提供薪资解析等工具方法
"""
import re
from typing import Dict, Any


class DataCleaner:
    """数据清洗工具类"""
    
    @staticmethod
    def parse_salary_range(salary_str: str) -> Dict[str, Any]:
        """
        解析薪资范围字符串
        
        Args:
            salary_str: 薪资字符串，如 "10-15K", "20-30K·13薪"
            
        Returns:
            {"min": 最低年薪, "max": 最高年薪, "months": 薪资月数}
        """
        if not salary_str or not isinstance(salary_str, str):
            return {"min": None, "max": None, "months": 12}
        
        result = {"min": None, "max": None, "months": 12}
        
        try:
            # 提取薪资月数
            months_match = re.search(r'[·x×]\s*(\d+)\s*薪', salary_str)
            if months_match:
                result["months"] = int(months_match.group(1))
            
            # 提取薪资范围 (10-15K 或 10K-15K)
            salary_match = re.search(r'(\d+)(?:K)?[-~](\d+)K', salary_str, re.IGNORECASE)
            if salary_match:
                min_k = int(salary_match.group(1))
                max_k = int(salary_match.group(2))
                
                # 转换为年薪
                result["min"] = min_k * 1000 * result["months"]
                result["max"] = max_k * 1000 * result["months"]
            
        except Exception:
            pass
        
        return result

