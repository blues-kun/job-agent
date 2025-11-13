"""
XGBoost模型管理
负责：模型保存、加载、元数据管理
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ModelManager:
    """XGBoost模型管理器"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / 'xgb_model.json'
        self.meta_path = self.model_dir / 'model_meta.json'
    
    def save_model(self, bst, metrics: Dict[str, Any], feature_names: list):
        """
        保存模型及元数据
        
        Args:
            bst: XGBoost模型
            metrics: 训练指标字典
            feature_names: 特征名称列表
        """
        # 保存模型文件
        bst.save_model(str(self.model_path))
        
        # 保存元数据
        meta = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'feature_names': feature_names,
            'metrics': metrics
        }
        
        with self.meta_path.open('w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f'\n✅ 模型已保存: {self.model_path}')
        print(f'✅ 元数据已保存: {self.meta_path}')
    
    def load_meta(self) -> Dict[str, Any]:
        """加载模型元数据"""
        if not self.meta_path.exists():
            return {}
        
        with self.meta_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    
    def model_exists(self) -> bool:
        """检查模型是否存在"""
        return self.model_path.exists()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model_exists():
            return {'exists': False}
        
        meta = self.load_meta()
        
        return {
            'exists': True,
            'model_path': str(self.model_path),
            'timestamp': meta.get('timestamp'),
            'feature_names': meta.get('feature_names', []),
            'metrics': meta.get('metrics', {})
        }
    
    def export_training_report(self, output_path: Path = None) -> str:
        """导出训练报告"""
        if output_path is None:
            output_path = self.model_dir / 'training_report.txt'
        
        info = self.get_model_info()
        
        if not info['exists']:
            return '模型不存在，无法生成报告'
        
        lines = []
        lines.append('='*60)
        lines.append('XGBoost模型训练报告')
        lines.append('='*60)
        lines.append(f"训练时间: {info.get('timestamp', 'Unknown')}")
        lines.append(f"模型路径: {info.get('model_path', 'Unknown')}")
        lines.append('')
        
        metrics = info.get('metrics', {})
        if metrics:
            lines.append('训练指标')
            lines.append('-'*60)
            
            # 数据统计
            if 'data_stats' in metrics:
                stats = metrics['data_stats']
                lines.append(f"  总样本数: {stats.get('total_samples', 0)}")
                lines.append(f"  正样本数: {stats.get('positive_samples', 0)}")
                lines.append(f"  负样本数: {stats.get('negative_samples', 0)}")
                lines.append('')
            
            # 训练集指标
            if 'train' in metrics:
                train = metrics['train']
                lines.append('  训练集:')
                if 'auc' in train:
                    lines.append(f"    AUC: {train['auc']:.4f}")
                if 'accuracy' in train:
                    lines.append(f"    准确率: {train['accuracy']:.4f}")
                lines.append('')
            
            # 验证集指标
            if 'val' in metrics:
                val = metrics['val']
                lines.append('  验证集:')
                if 'auc' in val:
                    lines.append(f"    AUC: {val['auc']:.4f}")
                if 'accuracy' in val:
                    lines.append(f"    准确率: {val['accuracy']:.4f}")
                if 'precision' in val:
                    lines.append(f"    精确率: {val['precision']:.4f}")
                if 'recall' in val:
                    lines.append(f"    召回率: {val['recall']:.4f}")
                if 'f1' in val:
                    lines.append(f"    F1分数: {val['f1']:.4f}")
        
        lines.append('='*60)
        
        report = '\n'.join(lines)
        
        # 保存到文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding='utf-8')
        
        return report


