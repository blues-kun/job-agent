"""
薪资标准化模块
将各种薪资格式（月薪、日薪、年薪、多薪）统一转换为年薪
"""
import re
from typing import Dict, Optional


class SalaryNormalizer:
    """薪资标准化处理器"""
    
    @staticmethod
    def normalize_to_annual(salary_str: str) -> Dict[str, Optional[float]]:
        """
        将薪资字符串标准化为年薪
        
        Args:
            salary_str: 薪资字符串，如 "15-30K·14薪", "100-200元/天", "3000-8000元/月"
            
        Returns:
            {
                'min_annual': 最低年薪（元），
                'max_annual': 最高年薪（元），
                'avg_annual': 平均年薪（元），
                'months': 月数（薪资包含的月份数），
                'type': 薪资类型 ('annual', 'monthly', 'daily', 'negotiable', 'unknown')
            }
        """
        if not salary_str or not isinstance(salary_str, str):
            return {
                'min_annual': None,
                'max_annual': None,
                'avg_annual': None,
                'months': 12,
                'type': 'unknown'
            }
        
        salary_str = salary_str.strip()
        
        # 面议
        if '面议' in salary_str or salary_str == '':
            return {
                'min_annual': None,
                'max_annual': None,
                'avg_annual': None,
                'months': 12,
                'type': 'negotiable'
            }
        
        # 提取薪资月数（如14薪、13薪）
        months_match = re.search(r'[·\-]\s*(\d+)\s*薪', salary_str)
        months = int(months_match.group(1)) if months_match else 12
        
        # 识别薪资类型
        salary_type = 'unknown'
        if '元/天' in salary_str or '元/日' in salary_str:
            salary_type = 'daily'
        elif '元/月' in salary_str or 'K' in salary_str.upper() or '千/月' in salary_str:
            salary_type = 'monthly'
        elif '万' in salary_str or 'W' in salary_str.upper():
            salary_type = 'annual'
        elif re.search(r'\d+K', salary_str, re.IGNORECASE):
            salary_type = 'monthly'
        
        # 提取数字范围
        min_val, max_val = SalaryNormalizer._extract_range(salary_str)
        
        if min_val is None and max_val is None:
            return {
                'min_annual': None,
                'max_annual': None,
                'avg_annual': None,
                'months': months,
                'type': salary_type
            }
        
        # 转换为年薪
        min_annual, max_annual = SalaryNormalizer._convert_to_annual(
            min_val, max_val, salary_type, months
        )
        
        # 计算平均值
        avg_annual = None
        if min_annual is not None and max_annual is not None:
            avg_annual = (min_annual + max_annual) / 2
        elif min_annual is not None:
            avg_annual = min_annual
        elif max_annual is not None:
            avg_annual = max_annual
        
        return {
            'min_annual': min_annual,
            'max_annual': max_annual,
            'avg_annual': avg_annual,
            'months': months,
            'type': salary_type
        }
    
    @staticmethod
    def _extract_range(salary_str: str) -> tuple:
        """提取薪资范围的最小值和最大值"""
        # 移除常见的单位词
        clean_str = salary_str.replace('元/天', '').replace('元/日', '').replace('元/月', '')
        clean_str = clean_str.replace('千/月', 'K').replace('薪', '')
        
        # 匹配各种格式
        # 格式1: 15-30K, 15-30k, 15-30万
        pattern1 = r'(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)\s*([KkWw万千]?)'
        match1 = re.search(pattern1, clean_str)
        
        if match1:
            min_val = float(match1.group(1))
            max_val = float(match1.group(2))
            unit = match1.group(3).upper() if match1.group(3) else ''
            
            # 处理单位
            if unit in ['K', 'k', '千']:
                min_val *= 1000
                max_val *= 1000
            elif unit in ['W', 'w', '万']:
                min_val *= 10000
                max_val *= 10000
            
            return min_val, max_val
        
        # 格式2: 单个数字（如 "100元/天"）
        pattern2 = r'(\d+(?:\.\d+)?)\s*([KkWw万千]?)'
        match2 = re.search(pattern2, clean_str)
        
        if match2:
            val = float(match2.group(1))
            unit = match2.group(2).upper() if match2.group(2) else ''
            
            if unit in ['K', 'k', '千']:
                val *= 1000
            elif unit in ['W', 'w', '万']:
                val *= 10000
            
            return val, val
        
        return None, None
    
    @staticmethod
    def _convert_to_annual(min_val: Optional[float], max_val: Optional[float], 
                          salary_type: str, months: int) -> tuple:
        """将薪资转换为年薪"""
        if min_val is None and max_val is None:
            return None, None
        
        # 日薪转年薪（按工作日计算，一年约250天）
        if salary_type == 'daily':
            working_days = 250
            min_annual = min_val * working_days if min_val else None
            max_annual = max_val * working_days if max_val else None
        
        # 月薪转年薪
        elif salary_type == 'monthly':
            min_annual = min_val * months if min_val else None
            max_annual = max_val * months if max_val else None
        
        # 年薪（已经是年薪）
        elif salary_type == 'annual':
            min_annual = min_val
            max_annual = max_val
        
        # 未知类型，尝试智能判断
        else:
            # 如果数值很小（<5000），可能是日薪或月薪的K表示
            if max_val and max_val < 5000:
                # 判断是否是K（千）的形式
                if 50 <= max_val <= 1000:  # 50-1000 可能是月薪K
                    min_annual = min_val * months if min_val else None
                    max_annual = max_val * months if max_val else None
                else:  # 否则当做日薪
                    min_annual = min_val * 250 if min_val else None
                    max_annual = max_val * 250 if max_val else None
            else:
                # 数值较大，默认当月薪处理
                min_annual = min_val * months if min_val else None
                max_annual = max_val * months if max_val else None
        
        return min_annual, max_annual


def test_salary_normalizer():
    """测试薪资标准化功能"""
    test_cases = [
        "15-30K·14薪",
        "100-200元/天",
        "3000-8000元/月",
        "11-18K",
        "30-60K·15薪",
        "40-70K·16薪",
        "面议",
        "15-30K",
        "25-50K·15薪",
    ]
    
    normalizer = SalaryNormalizer()
    print("薪资标准化测试:\n")
    print(f"{'原始薪资':<20} {'类型':<10} {'最低年薪':<12} {'最高年薪':<12} {'平均年薪':<12}")
    print("-" * 70)
    
    for salary_str in test_cases:
        result = normalizer.normalize_to_annual(salary_str)
        min_annual = f"{result['min_annual']/10000:.1f}万" if result['min_annual'] else "N/A"
        max_annual = f"{result['max_annual']/10000:.1f}万" if result['max_annual'] else "N/A"
        avg_annual = f"{result['avg_annual']/10000:.1f}万" if result['avg_annual'] else "N/A"
        
        print(f"{salary_str:<20} {result['type']:<10} {min_annual:<12} {max_annual:<12} {avg_annual:<12}")


if __name__ == '__main__':
    test_salary_normalizer()

