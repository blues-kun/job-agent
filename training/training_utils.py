"""
XGBoost训练工具函数
负责：数据加载、特征提取、指标计算
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

from models import ResumeProfile
from features.extractor import extract


def load_training_data(jsonl_path: Path) -> Tuple[List[List[float]], List[int], List[str]]:
    """
    从JSONL文件加载训练数据
    
    Returns:
        X: 特征矩阵
        y: 标签列表
        feature_names: 特征名称列表
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f'训练数据不存在: {jsonl_path}')
    
    X = []
    y = []
    feature_names = None
    
    for line in jsonl_path.read_text('utf-8').splitlines():
        if not line.strip():
            continue
        
        try:
            obj = json.loads(line)
            resume_dict = obj.get('resume') or {}
            job = obj.get('job') or {}
            
            # 确定标签
            if 'label' in obj:
                label = int(obj['label'])
            elif 'action' in obj:
                # 从action推断label
                action = obj['action']
                if action == 'like':
                    label = 1
                elif action == 'skip':
                    label = 0
                else:
                    continue
            else:
                continue
            
            # 提取特征
            resume = ResumeProfile.model_validate(resume_dict)
            feats = extract(resume, job)
            
            # 获取特征名称
            if feature_names is None:
                feature_names = sorted(feats.keys())
            
            # 添加样本
            X.append([feats[k] for k in feature_names])
            y.append(label)
            
        except Exception as e:
            # 跳过无效样本
            continue
    
    if not X:
        raise ValueError('没有有效的训练样本')
    
    return X, y, feature_names


def calculate_data_stats(y: List[int]) -> Dict[str, Any]:
    """计算数据集统计信息"""
    total = len(y)
    positive = sum(y)
    negative = total - positive
    
    return {
        'total_samples': total,
        'positive_samples': positive,
        'negative_samples': negative,
        'positive_ratio': positive / total if total > 0 else 0,
        'negative_ratio': negative / total if total > 0 else 0
    }


def split_train_val(X: List, y: List, test_size: float = 0.2, min_samples_for_split: int = 20):
    """
    切分训练集和验证集
    
    Returns:
        (X_train, X_val, y_train, y_val) or (X, None, y, None) if too few samples
    """
    if len(X) < min_samples_for_split:
        return X, None, y, None
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def calculate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred_prob: 预测概率
        threshold: 分类阈值
    
    Returns:
        包含accuracy, precision, recall, f1的字典
    """
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # 基础指标
    accuracy = np.mean(y_pred == y_true)
    
    # 混淆矩阵元素
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    # 精确率、召回率、F1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def print_training_summary(
    data_stats: Dict[str, Any],
    train_metrics: Dict[str, Any] = None,
    val_metrics: Dict[str, Any] = None
):
    """打印训练摘要"""
    print('\n' + '='*50)
    print('训练数据统计')
    print('='*50)
    print(f"总样本数: {data_stats['total_samples']}")
    print(f"正样本(label=1): {data_stats['positive_samples']} ({data_stats['positive_ratio']*100:.1f}%)")
    print(f"负样本(label=0): {data_stats['negative_samples']} ({data_stats['negative_ratio']*100:.1f}%)")
    
    if train_metrics:
        print('\n' + '='*50)
        print('训练集指标')
        print('='*50)
        if 'auc' in train_metrics:
            print(f"AUC: {train_metrics['auc']:.4f}")
        for k, v in train_metrics.items():
            if k != 'auc':
                print(f"{k}: {v}")
    
    if val_metrics:
        print('\n' + '='*50)
        print('验证集指标')
        print('='*50)
        if 'auc' in val_metrics:
            print(f"AUC: {val_metrics['auc']:.4f}")
        if 'accuracy' in val_metrics:
            print(f"准确率: {val_metrics['accuracy']:.4f}")
        if 'precision' in val_metrics:
            print(f"精确率: {val_metrics['precision']:.4f}")
        if 'recall' in val_metrics:
            print(f"召回率: {val_metrics['recall']:.4f}")
        if 'f1' in val_metrics:
            print(f"F1分数: {val_metrics['f1']:.4f}")
        
        if 'tp' in val_metrics:
            print(f"\n混淆矩阵:")
            print(f"  TP={val_metrics['tp']}, FP={val_metrics['fp']}")
            print(f"  FN={val_metrics['fn']}, TN={val_metrics['tn']}")
    
    print('='*50)


