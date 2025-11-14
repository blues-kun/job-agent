"""
XGBoost API处理逻辑
负责：训练模型
"""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import sys

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from training.training_utils import (
    load_training_data,
    calculate_data_stats,
    split_train_val,
    calculate_metrics,
    print_training_summary
)
from training.model_manager import ModelManager


class XGBoostAPI:
    """XGBoost API处理器"""
    
    def __init__(
        self,
        feedback_path: str = None,
        dataset_path: str = None,
        model_dir: str = None
    ):
        # 使用项目根目录的绝对路径
        self.feedback_path = Path(feedback_path) if feedback_path else PROJECT_ROOT / 'logs' / 'recommend_events.jsonl'
        self.dataset_path = Path(dataset_path) if dataset_path else PROJECT_ROOT / 'logs' / 'training_dataset.csv'
        model_dir = model_dir or str(PROJECT_ROOT / 'models')
        self.model_manager = ModelManager(model_dir)
    
    def train_model(
        self,
        max_depth: int = 5,
        eta: float = 0.2,
        num_boost_round: int = 100,
        eval_metric: str = 'auc'
    ) -> Dict[str, Any]:
        """
        训练XGBoost模型
        
        Args:
            max_depth: 树的最大深度
            eta: 学习率
            num_boost_round: 迭代轮数
            eval_metric: 评估指标
        
        Returns:
            包含训练结果和指标的字典
        """
        if not XGB_AVAILABLE:
            return {
                'success': False,
                'error': 'xgboost未安装'
            }
        
        try:
            # 加载训练数据
            X, y, feature_names = load_training_data(self.feedback_path)
            
            # 数据统计
            data_stats = calculate_data_stats(y)
            
            # 切分训练集和验证集
            X_train, X_val, y_train, y_val = split_train_val(X, y)
            
            # 创建DMatrix
            dtrain = xgb.DMatrix(np.array(X_train), label=np.array(y_train), feature_names=feature_names)
            
            # 训练参数
            params = {
                'max_depth': max_depth,
                'eta': eta,
                'objective': 'binary:logistic',
                'eval_metric': eval_metric
            }
            
            # 训练模型
            evals_result = {}
            if X_val is not None:
                dval = xgb.DMatrix(np.array(X_val), label=np.array(y_val), feature_names=feature_names)
                watchlist = [(dtrain, 'train'), (dval, 'val')]
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=watchlist,
                    evals_result=evals_result,
                    verbose_eval=False
                )
            else:
                watchlist = [(dtrain, 'train')]
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=watchlist,
                    evals_result=evals_result,
                    verbose_eval=False
                )
            
            # 计算指标
            train_pred = bst.predict(dtrain)
            train_metrics = calculate_metrics(np.array(y_train), train_pred)
            
            val_metrics = None
            if X_val is not None:
                val_pred = bst.predict(dval)
                val_metrics = calculate_metrics(np.array(y_val), val_pred)
                # 添加AUC
                if 'val' in evals_result and eval_metric in evals_result['val']:
                    val_metrics['auc'] = float(evals_result['val'][eval_metric][-1])
            
            # 添加AUC到训练指标
            if 'train' in evals_result and eval_metric in evals_result['train']:
                train_metrics['auc'] = float(evals_result['train'][eval_metric][-1])
            
            # 保存模型和元数据
            all_metrics = {
                'data_stats': data_stats,
                'train': train_metrics,
            }
            if val_metrics:
                all_metrics['val'] = val_metrics
            
            self.model_manager.save_model(bst, all_metrics, feature_names)
            
            # 打印训练摘要
            print_training_summary(data_stats, train_metrics, val_metrics)
            
            return {
                'success': True,
                'model_path': str(self.model_manager.model_path),
                'samples': len(X),
                'metrics': all_metrics
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# 创建全局实例
xgb_api = XGBoostAPI()


def handle_xgb_ops(op: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    统一的XGBoost操作处理接口
    
    Args:
        op: 操作类型 ('train')
        params: 操作参数
    
    Returns:
        操作结果字典
    """
    params = params or {}
    
    if op == 'train':
        # 训练模型
        return xgb_api.train_model(
            max_depth=params.get('max_depth', 5),
            eta=params.get('eta', 0.2),
            num_boost_round=params.get('num_boost_round', 100),
            eval_metric=params.get('eval_metric', 'auc')
        )
    else:
        return {
            'success': False,
            'error': f'不支持的操作: {op}，当前仅支持 train'
        }


