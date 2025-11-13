"""
岗位数据加载器
支持从Excel、CSV、JSONL加载岗位数据
"""
import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path


class JobDataLoader:
    """岗位数据加载器"""
    
    def __init__(self, file_path: Path):
        """
        初始化加载器
        
        Args:
            file_path: 数据文件路径（支持 .xlsx, .csv, .jsonl）
        """
        self.file_path = Path(file_path)
        self._data = None
        self._jobs_list = None
        self.file_type = self._detect_file_type()
    
    def _detect_file_type(self) -> str:
        """检测文件类型"""
        suffix = self.file_path.suffix.lower()
        if suffix in ['.xlsx', '.xls']:
            return 'excel'
        elif suffix == '.csv':
            return 'csv'
        elif suffix == '.jsonl':
            return 'jsonl'
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def load(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            pandas DataFrame
        """
        print(f"正在加载岗位数据: {self.file_path} (格式: {self.file_type})")
        
        if self.file_type == 'excel':
            self._data = pd.read_excel(self.file_path)
        elif self.file_type == 'csv':
            self._data = pd.read_csv(self.file_path)
        elif self.file_type == 'jsonl':
            self._data = self._load_jsonl()
        
        print(f"[OK] 已加载 {len(self._data)} 条岗位数据")
        return self._data
    
    def _load_jsonl(self) -> pd.DataFrame:
        """
        从JSONL加载并转换为扁平DataFrame
        
        Returns:
            pandas DataFrame
        """
        jobs = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                job_obj = json.loads(line)
                
                # 将嵌套结构转换为扁平结构（与matcher期望的格式对齐）
                flat_job = {
                    '公司': job_obj.get('job_identity', {}).get('company_name', ''),
                    '岗位名称': job_obj.get('job_identity', {}).get('original_job_title', ''),
                    '三级分类': job_obj.get('job_identity', {}).get('position_type_name', ''),
                    '二级分类': job_obj.get('job_identity', {}).get('secondary_category', ''),
                    '岗位要求': job_obj.get('original_text', {}).get('raw_requirements', ''),
                    '岗位职责': job_obj.get('original_text', {}).get('raw_responsibilities', ''),
                    '岗位地址': job_obj.get('location', {}).get('address', ''),
                    '城市': job_obj.get('location', {}).get('city', ''),
                }
                
                # 处理薪资信息
                compensation = job_obj.get('compensation', {})
                if compensation.get('salary_negotiable'):
                    flat_job['岗位薪资'] = '面议'
                elif 'salary_range_annual' in compensation:
                    salary_range = compensation['salary_range_annual']
                    min_k = salary_range.get('min', 0) // 12000
                    max_k = salary_range.get('max', 0) // 12000
                    flat_job['岗位薪资'] = f"{min_k}-{max_k}K"
                else:
                    flat_job['岗位薪资'] = ''
                
                jobs.append(flat_job)
        
        return pd.DataFrame(jobs)
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        将DataFrame转换为字典列表
        
        Returns:
            岗位数据字典列表
        """
        if self._data is None:
            self.load()
        
        # 转换为字典列表，处理NaN值
        self._jobs_list = self._data.to_dict('records')
        
        # 清理NaN值
        for job in self._jobs_list:
            for key, value in job.items():
                if pd.isna(value):
                    job[key] = None
        
        return self._jobs_list
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        if self._data is None:
            self.load()
        
        stats = {
            'total_jobs': len(self._data),
            'columns': list(self._data.columns),
            'missing_values': self._data.isnull().sum().to_dict(),
        }
        
        # 如果有分类字段，统计分布
        if '二级分类' in self._data.columns:
            stats['secondary_category_dist'] = self._data['二级分类'].value_counts().to_dict()
        
        if '三级分类' in self._data.columns:
            stats['tertiary_category_dist'] = self._data['三级分类'].value_counts().head(20).to_dict()
        
        return stats
    
    def filter_by_category(self, secondary: str = None, tertiary: str = None) -> List[Dict[str, Any]]:
        """
        按分类筛选岗位
        
        Args:
            secondary: 二级分类
            tertiary: 三级分类
            
        Returns:
            筛选后的岗位列表
        """
        if self._data is None:
            self.load()
        
        filtered = self._data
        
        if secondary:
            filtered = filtered[filtered['二级分类'] == secondary]
        
        if tertiary:
            filtered = filtered[filtered['三级分类'] == tertiary]
        
        return filtered.to_dict('records')

