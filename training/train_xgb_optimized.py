"""
XGBoost 超参数优化训练脚本
包含：
1. 超参数网格搜索
2. 增加训练轮数
3. 训练过程可视化（损失曲线、AUC曲线）
4. 特征重要性分析
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from models import ResumeProfile
from features.vectorized_extractor import VectorizedFeatureExtractor

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据"""
    print("\n[1/7] 加载向量化职位数据...")
    vectorized_file = Path(__file__).parent.parent / 'data' / 'job_data_vectorized.parquet'
    vectorized_jobs_df = pd.read_parquet(vectorized_file)
    print(f"  >>> 加载了 {len(vectorized_jobs_df)} 个职位")
    print(f"  >>> 特征数: {len(vectorized_jobs_df.columns)}")
    
    # 创建职位索引
    job_dict = {}
    for idx, row in vectorized_jobs_df.iterrows():
        name = str(row.get('岗位名称', '')).strip()
        company = str(row.get('企业', '')).strip()
        if name and name != 'nan' and company and company != 'nan':
            key = f"{idx}_{name}_{company}"
            job_dict[key] = row.to_dict()
    
    print(f"  >>> 建立职位索引 ({len(job_dict)} 个职位)")
    
    print("\n[2/7] 加载交互数据...")
    events_file = Path(__file__).parent.parent / 'logs' / 'recommend_events.jsonl'
    samples = []
    with events_file.open('r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  >>> 加载了 {len(samples)} 个交互样本")
    
    return vectorized_jobs_df, job_dict, samples


def extract_features(samples, job_dict):
    """提取特征"""
    print("\n[3/7] 提取特征...")
    extractor = VectorizedFeatureExtractor(vector_size=100, model_dir='models/word2vec')
    
    X_list = []
    y_list = []
    feature_names = None
    skipped = 0
    
    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(samples)}")
        
        try:
            resume_dict = sample.get('resume', {})
            job = sample.get('job', {})
            action = sample.get('action', 'skip')
            label = 1 if action == 'like' else 0
            
            resume = ResumeProfile.model_validate(resume_dict)
            
            job_name = job.get('岗位名称', '').strip()
            job_company = job.get('企业', '').strip()
            
            vectorized_job = None
            for key, value in job_dict.items():
                if job_name == value.get('岗位名称', '').strip() and \
                   job_company == value.get('企业', '').strip():
                    vectorized_job = value
                    break
            
            if vectorized_job:
                job_with_vectors = {**job, **vectorized_job}
            else:
                job_with_vectors = job
            
            features = extractor.extract(resume, job_with_vectors)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            X_list.append([features.get(k, 0.0) for k in feature_names])
            y_list.append(label)
            
        except Exception as e:
            skipped += 1
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  >>> 特征提取完成")
    print(f"  >>> 有效样本: {len(y)}")
    print(f"  >>> 跳过样本: {skipped}")
    print(f"  >>> 特征数: {len(feature_names)}")
    print(f"  >>> 正样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"  >>> 负样本: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    return X, y, feature_names


def hyperparameter_search(X_train, y_train, X_val, y_val, mode='quick'):
    """
    超参数搜索
    mode: 'quick' (快速) 或 'thorough' (全面)
    """
    print(f"\n[4/7] 超参数搜索 (模式: {mode})...")
    
    if mode == 'quick':
        # 快速搜索：测试几组关键参数
        param_configs = [
            {
                'name': 'baseline',
                'params': {
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1
                }
            },
            {
                'name': 'deeper_tree',
                'params': {
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1
                }
            },
            {
                'name': 'regularized',
                'params': {
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'n_estimators': 300,
                    'min_child_weight': 5,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'gamma': 0.2,
                    'reg_alpha': 1,
                    'reg_lambda': 2
                }
            },
            {
                'name': 'high_capacity',
                'params': {
                    'max_depth': 6,
                    'learning_rate': 0.03,
                    'n_estimators': 500,
                    'min_child_weight': 2,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.05,
                    'reg_alpha': 0.5,
                    'reg_lambda': 1.5
                }
            }
        ]
    else:
        # 全面搜索：网格搜索
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 500],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.05, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        # 这里使用 GridSearchCV 会非常耗时，实际项目中建议使用 RandomizedSearchCV
        print("  警告：全面搜索将非常耗时，建议使用 'quick' 模式")
    
    # 快速搜索
    best_config = None
    best_auc = 0
    results = []
    
    for config in param_configs:
        print(f"\n  测试配置: {config['name']}")
        params = {
            **config['params'],
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'tree_method': 'hist'  # 使用更快的直方图算法
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 训练
        evals = [(dtrain, 'train'), (dval, 'val')]
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=False
        )
        
        # 评估
        val_pred = bst.predict(dval)
        val_auc = roc_auc_score(y_val, val_pred)
        
        print(f"    验证集 AUC: {val_auc:.4f}")
        
        results.append({
            'name': config['name'],
            'params': params,
            'auc': val_auc,
            'best_iteration': bst.best_iteration
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_config = config
    
    print(f"\n  >>> 最佳配置: {best_config['name']} (AUC: {best_auc:.4f})")
    
    return best_config['params'], results


def train_with_visualization(X_train, y_train, X_val, y_val, params, feature_names):
    """训练模型并生成可视化"""
    print("\n[5/7] 训练最优模型（含可视化）...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # 存储训练过程
    evals_result = {}
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # 训练
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 500),
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=evals_result
    )
    
    print(f"\n  >>> 最佳迭代: {bst.best_iteration}")
    print(f"  >>> 最佳验证 AUC: {bst.best_score:.4f}")
    
    # 生成可视化
    plot_training_curves(evals_result, bst.best_iteration)
    plot_feature_importance(bst, feature_names)
    
    return bst, evals_result


def plot_training_curves(evals_result, best_iteration):
    """绘制训练曲线"""
    print("\n[6/7] 生成训练曲线图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 获取评估指标键（可能是 'auc' 或 'rmse'）
    metric_key = 'auc' if 'auc' in evals_result['train'] else list(evals_result['train'].keys())[0]
    metric_name = 'AUC' if metric_key == 'auc' else metric_key.upper()
    
    # 曲线
    epochs = range(len(evals_result['train'][metric_key]))
    
    ax1.plot(epochs, evals_result['train'][metric_key], label='训练集', linewidth=2)
    ax1.plot(epochs, evals_result['val'][metric_key], label='验证集', linewidth=2)
    ax1.axvline(x=best_iteration, color='r', linestyle='--', label=f'最佳迭代 ({best_iteration})')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel(metric_name, fontsize=12)
    ax1.set_title(f'训练过程 - {metric_name} 变化', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 过拟合分析
    train_metric = np.array(evals_result['train'][metric_key])
    val_metric = np.array(evals_result['val'][metric_key])
    # 对于 RMSE 等越小越好的指标，gap 符号相反
    if metric_key in ['rmse', 'logloss', 'error']:
        gap = val_metric - train_metric  # 验证集更大表示过拟合
    else:
        gap = train_metric - val_metric  # 训练集更大表示过拟合
    
    ax2.plot(epochs, gap, label='训练-验证 差距', color='orange', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=best_iteration, color='r', linestyle='--', label=f'最佳迭代 ({best_iteration})')
    ax2.fill_between(epochs, 0, gap, where=(gap >= 0), alpha=0.3, color='red', label='过拟合区域')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel(f'{metric_name} 差距', fontsize=12)
    ax2.set_title('过拟合分析', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"  >>> 训练曲线已保存: models/training_curves.png")
    plt.close()


def plot_feature_importance(bst, feature_names, top_n=20):
    """绘制特征重要性"""
    importance = bst.get_score(importance_type='gain')
    
    # 转换为 DataFrame 并排序
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False).head(top_n)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('特征重要性 (Gain)', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.title(f'Top {top_n} 特征重要性排名', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"  >>> 特征重要性图已保存: models/feature_importance.png")
    plt.close()


def evaluate_model(bst, X_train, y_train, X_val, y_val, feature_names):
    """评估模型"""
    print("\n[7/7] 模型评估...")
    
    dtrain = xgb.DMatrix(X_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, feature_names=feature_names)
    
    # 预测
    train_pred = bst.predict(dtrain)
    val_pred = bst.predict(dval)
    
    train_pred_binary = (train_pred > 0.5).astype(int)
    val_pred_binary = (val_pred > 0.5).astype(int)
    
    # 计算指标
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    val_accuracy = accuracy_score(y_val, val_pred_binary)
    val_precision = precision_score(y_val, val_pred_binary, zero_division=0)
    val_recall = recall_score(y_val, val_pred_binary)
    val_f1 = f1_score(y_val, val_pred_binary)
    
    print(f"  >>> 训练集 AUC: {train_auc:.4f}")
    print(f"  >>> 验证集 AUC: {val_auc:.4f}")
    print(f"  >>> 验证集准确率: {val_accuracy:.4f}")
    print(f"  >>> 验证集精确率: {val_precision:.4f}")
    print(f"  >>> 验证集召回率: {val_recall:.4f}")
    print(f"  >>> 验证集 F1: {val_f1:.4f}")
    
    # 特征重要性
    importance = bst.get_score(importance_type='gain')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  特征重要性 Top 15:")
    for i, (feat, score) in enumerate(importance_sorted[:15], 1):
        print(f"  {i:2d}. {feat:30s}: {score:.2f}")
    
    return {
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'val_accuracy': float(val_accuracy),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        'best_iteration': int(bst.best_iteration),
        'feature_importance': {k: float(v) for k, v in importance_sorted[:15]}
    }


def main():
    print("="*80)
    print("XGBoost 超参数优化训练")
    print("="*80)
    
    # 加载数据
    vectorized_jobs_df, job_dict, samples = load_data()
    
    # 提取特征
    X, y, feature_names = extract_features(samples, job_dict)
    
    # 切分数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  >>> 训练集: {len(y_train)} 条")
    print(f"  >>> 验证集: {len(y_val)} 条")
    
    # 超参数搜索
    best_params, search_results = hyperparameter_search(X_train, y_train, X_val, y_val, mode='quick')
    
    # 训练最优模型
    bst, evals_result = train_with_visualization(X_train, y_train, X_val, y_val, best_params, feature_names)
    
    # 评估模型
    metrics = evaluate_model(bst, X_train, y_train, X_val, y_val, feature_names)
    
    # 保存模型
    model_path = Path('models/xgb_model_optimized.json')
    bst.save_model(str(model_path))
    print(f"\n  >>> 模型已保存: {model_path}")
    
    # 保存元数据
    meta = {
        'model_type': 'XGBoost_Optimized',
        'model_path': str(model_path),
        'best_params': best_params,
        'feature_names': feature_names,
        'metrics': metrics,
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'search_results': search_results
    }
    
    meta_path = Path('models/xgb_optimized_meta.json')
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  >>> 元数据已保存: {meta_path}")
    
    print("\n" + "="*80)
    print("训练完成！")
    print(f"  - 验证集 AUC: {metrics['val_auc']:.4f}")
    print(f"  - 验证集 F1: {metrics['val_f1']:.4f}")
    print(f"  - 最佳迭代: {metrics['best_iteration']}")
    print("="*80)
    
    return metrics


if __name__ == '__main__':
    main()

