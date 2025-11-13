"""
简历存储管理
负责简历的持久化和读取
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any

from models import ResumeProfile


class ResumeStorage:
    """简历存储管理器"""
    
    def __init__(self, file_path: Path):
        """
        初始化存储管理器
        
        Args:
            file_path: 简历文件路径
        """
        self.file_path = file_path
        self._resume: Optional[ResumeProfile] = None
        
        # 如果文件存在，尝试加载
        if self.file_path.exists():
            try:
                self.load()
            except Exception as e:
                print(f"警告: 加载现有简历失败: {e}")
    
    def load(self) -> Optional[ResumeProfile]:
        """
        从文件加载简历
        
        Returns:
            ResumeProfile 对象或 None
        """
        try:
            suffix = self.file_path.suffix.lower()
            if suffix == '.json':
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._resume = ResumeProfile.model_validate(data)
                print(f"✓ 已加载简历: {self.file_path}")
                return self._resume
            elif suffix == '.txt':
                import re
                text = self.file_path.read_text('utf-8')
                city = None
                for c in ['上海','深圳','北京','南京','杭州']:
                    if c in text:
                        city = c
                        break
                edu = None
                for e in ['博士','硕士','本科','大专']:
                    if e in text:
                        edu = e
                        break
                exp = None
                m = re.search(r'(\d+(?:\.\d+)?)\s*年', text)
                if m:
                    try:
                        exp = float(m.group(1))
                    except:
                        exp = None
                sal = None
                m2 = re.search(r'(\d+)\s*万', text)
                if m2:
                    sal = int(m2.group(1)) * 10000
                m3 = re.search(r'(\d+)\s*[Kk]', text)
                if m3 and not sal:
                    sal = int(m3.group(1)) * 1000 * 12
                intent = []
                for kw in ['Java','后端开发','全栈工程师','Python','Golang']:
                    if kw.lower() in text.lower():
                        intent.append(kw)
                if not intent:
                    intent = ['Java']
                fb = {
                    'personal_info': {
                        'current_city': city or '上海',
                        'willingness_to_relocate': True,
                        'availability_date': None
                    },
                    'work_preferences': {
                        'position_type_name': intent,
                        'salary_expectation': {
                            'min_annual_package': sal or 300000,
                            'currency': 'CNY'
                        }
                    },
                    'professional_summary': {
                        'total_experience_years': exp or 3.0,
                        'education_level': edu or '本科',
                        'school_level': 0
                    },
                    'full_resume_text': text
                }
                self._resume = ResumeProfile.model_validate(fb)
                print(f"✓ 已加载简历文本并解析: {self.file_path}")
                return self._resume
            else:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._resume = ResumeProfile.model_validate(data)
                print(f"✓ 已加载简历: {self.file_path}")
                return self._resume
        except Exception as e:
            print(f"加载简历失败: {e}")
            return None
    
    def save(self, resume: ResumeProfile) -> None:
        """
        保存简历到文件
        
        Args:
            resume: ResumeProfile 对象
        """
        self._resume = resume
        
        # 确保目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(
                resume.model_dump(),
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"✓ 简历已保存: {self.file_path}")
    
    def get(self) -> Optional[ResumeProfile]:
        """
        获取当前简历
        
        Returns:
            ResumeProfile 对象或 None
        """
        return self._resume
    
    def update(self, patch: Dict[str, Any]) -> ResumeProfile:
        """
        部分更新简历（深度合并）
        
        Args:
            patch: 要更新的字段（支持嵌套结构）
            
        Returns:
            更新后的ResumeProfile
            
        Example:
            storage.update({"personal_info": {"current_city": "上海"}})
        """
        if not self._resume:
            raise ValueError("简历尚未加载，无法更新")
        
        # 深度合并更新
        current_data = self._resume.model_dump()
        merged_data = self._deep_merge(current_data, patch)
        
        # 重新验证并保存
        self._resume = ResumeProfile.model_validate(merged_data)
        self.save(self._resume)
        
        return self._resume
    
    def patch(self, patch_obj: Dict[str, Any]) -> ResumeProfile:
        """
        部分更新简历的别名方法（与 hello-uv 兼容）
        
        Args:
            patch_obj: 要更新的字段
            
        Returns:
            更新后的ResumeProfile
        """
        return self.update(patch_obj)
    
    @staticmethod
    def _deep_merge(base: Dict, updates: Dict) -> Dict:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            updates: 更新字典
            
        Returns:
            合并后的字典
        """
        result = dict(base)
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ResumeStorage._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def exists(self) -> bool:
        """
        检查简历是否存在
        
        Returns:
            是否存在简历
        """
        return self._resume is not None

