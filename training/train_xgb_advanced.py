"""
XGBoost 高级训练脚本 - 支持人为干预
功能：
1. 分阶段训练（可调整学习率）
2. 手动学习率衰减
3. 实时监控和可视化
4. 自定义损失权重
5. 交互式训练控制
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from models import ResumeProfile
from features.vectorized_extractor import VectorizedFeatureExtractor

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


class XGBoostAdvancedTrainer:
    """XGBoost 高级训练器"""
    
    def __init__(self, X_train, y_train, X_val, y_val, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.feature_names = feature_names
        
        self.dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        self.dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        self.model = None
        self.training_history = {
            'train_auc': [],
            'val_auc': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'iterations': []
        }
        
    def train_stage(self, params, num_rounds, stage_name="Stage"):
        """
        训练一个阶段
        
        参数:
            params: XGBoost 参数字典
            num_rounds: 训练轮数
            stage_name: 阶段名称
        """
        print(f"\n{'='*80}")
        print(f"{stage_name}")
        print(f"{'='*80}")
        print(f"学习率: {params['learning_rate']}")
        print(f"轮数: {num_rounds}")
        print(f"max_depth: {params.get('max_depth', 3)}")
        print(f"{'='*80}")
        
        # 如果已有模型，继续训练
        xgb_model = self.model if self.model is not None else None
        
        evals = [(self.dtrain, 'train'), (self.dval, 'val')]
        evals_result = {}
        
        # 训练（使用 verbose_eval 代替自定义回调）
        self.model = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            xgb_model=xgb_model,  # 继续训练
            verbose_eval=10  # 每10轮打印一次
        )
        
        # 记录历史
        for i in range(len(evals_result['train']['auc'])):
            self.training_history['iterations'].append(
                len(self.training_history['iterations']) + 1
            )
            self.training_history['train_auc'].append(evals_result['train']['auc'][i])
            self.training_history['val_auc'].append(evals_result['val']['auc'][i])
            self.training_history['learning_rates'].append(params['learning_rate'])
        
        # 保存当前阶段的最佳结果
        best_val_auc = max(evals_result['val']['auc'])
        best_iteration = evals_result['val']['auc'].index(best_val_auc)
        
        print(f"\n{stage_name} 完成:")
        print(f"  最佳迭代: {best_iteration}")
        print(f"  最佳验证AUC: {best_val_auc:.4f}")
        
        return evals_result
    
    def plot_training_history(self):
        """绘制完整的训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        iterations = self.training_history['iterations']
        train_auc = self.training_history['train_auc']
        val_auc = self.training_history['val_auc']
        learning_rates = self.training_history['learning_rates']
        
        # 图1: AUC 变化
        ax1 = axes[0, 0]
        ax1.plot(iterations, train_auc, label='训练集', linewidth=2, color='blue')
        ax1.plot(iterations, val_auc, label='验证集', linewidth=2, color='red')
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('AUC', fontsize=12)
        ax1.set_title('AUC 变化曲线', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 标记学习率变化点
        unique_lr = sorted(set(learning_rates))
        if len(unique_lr) > 1:
            for i in range(1, len(iterations)):
                if learning_rates[i] != learning_rates[i-1]:
                    ax1.axvline(x=iterations[i], color='green', linestyle='--', alpha=0.5)
        
        # 图2: 过拟合分析
        ax2 = axes[0, 1]
        gap = np.array(train_auc) - np.array(val_auc)
        ax2.plot(iterations, gap, color='orange', linewidth=2)
        ax2.fill_between(iterations, 0, gap, where=(gap >= 0), alpha=0.3, color='red')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('AUC 差距 (训练 - 验证)', fontsize=12)
        ax2.set_title('过拟合分析', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 图3: 学习率变化
        ax3 = axes[1, 0]
        ax3.plot(iterations, learning_rates, linewidth=2, color='green', marker='o', markersize=3)
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_ylabel('学习率', fontsize=12)
        ax3.set_title('学习率调度', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 图4: 验证集 AUC 提升速度
        ax4 = axes[1, 1]
        if len(val_auc) > 1:
            improvement = np.diff(val_auc)
            ax4.bar(iterations[1:], improvement, color='steelblue', alpha=0.7)
            ax4.axhline(y=0, color='red', linestyle='-', linewidth=1)
            ax4.set_xlabel('迭代次数', fontsize=12)
            ax4.set_ylabel('AUC 提升量', fontsize=12)
            ax4.set_title('每轮AUC提升量（验证集）', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('models/advanced_training_history.png', dpi=300, bbox_inches='tight')
        print(f"\n>>> 训练历史图已保存: models/advanced_training_history.png")
        plt.close()
    
    def evaluate(self):
        """评估最终模型"""
        dtrain = xgb.DMatrix(self.X_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(self.X_val, feature_names=self.feature_names)
        
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        train_pred_binary = (train_pred > 0.5).astype(int)
        val_pred_binary = (val_pred > 0.5).astype(int)
        
        metrics = {
            'train_auc': float(roc_auc_score(self.y_train, train_pred)),
            'val_auc': float(roc_auc_score(self.y_val, val_pred)),
            'val_accuracy': float(accuracy_score(self.y_val, val_pred_binary)),
            'val_precision': float(precision_score(self.y_val, val_pred_binary, zero_division=0)),
            'val_recall': float(recall_score(self.y_val, val_pred_binary)),
            'val_f1': float(f1_score(self.y_val, val_pred_binary))
        }
        
        return metrics
    
    def save_model(self, path='models/xgb_model_advanced.json'):
        """保存模型"""
        self.model.save_model(path)
        print(f"\n>>> 模型已保存: {path}")


def load_data():
    """加载数据"""
    print("\n[1/5] 加载数据...")
    vectorized_file = Path(__file__).parent.parent / 'data' / 'job_data_vectorized.parquet'
    vectorized_jobs_df = pd.read_parquet(vectorized_file)
    
    job_dict = {}
    for idx, row in vectorized_jobs_df.iterrows():
        name = str(row.get('岗位名称', '')).strip()
        company = str(row.get('企业', '')).strip()
        if name and name != 'nan' and company and company != 'nan':
            key = f"{idx}_{name}_{company}"
            job_dict[key] = row.to_dict()
    
    events_file = Path(__file__).parent.parent / 'logs' / 'recommend_events.jsonl'
    samples = []
    with events_file.open('r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"  >>> 职位数: {len(job_dict)}")
    print(f"  >>> 样本数: {len(samples)}")
    
    return vectorized_jobs_df, job_dict, samples


def extract_features(samples, job_dict):
    """提取特征"""
    print("\n[2/5] 提取特征...")
    extractor = VectorizedFeatureExtractor(vector_size=100, model_dir='models/word2vec')
    
    X_list = []
    y_list = []
    feature_names = None
    
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
            
        except Exception:
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  >>> 特征数: {len(feature_names)}")
    print(f"  >>> 有效样本: {len(y)}")
    print(f"  >>> 正样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    return X, y, feature_names


def multi_stage_training():
    """多阶段训练策略"""
    print("="*80)
    print("XGBoost 高级训练 - 多阶段学习率衰减")
    print("="*80)
    
    # 加载数据
    vectorized_jobs_df, job_dict, samples = load_data()
    X, y, feature_names = extract_features(samples, job_dict)
    
    # 切分数据
    print("\n[3/5] 切分数据...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  >>> 训练集: {len(y_train)}")
    print(f"  >>> 验证集: {len(y_val)}")
    
    # 创建训练器
    print("\n[4/5] 多阶段训练...")
    trainer = XGBoostAdvancedTrainer(X_train, y_train, X_val, y_val, feature_names)
    
    # 基础参数
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    # 阶段1: 快速学习（高学习率）
    stage1_params = {**base_params, 'learning_rate': 0.1}
    trainer.train_stage(stage1_params, num_rounds=100, stage_name="阶段1: 快速学习（lr=0.1）")
    
    # 阶段2: 精细调整（中学习率）
    stage2_params = {**base_params, 'learning_rate': 0.05}
    trainer.train_stage(stage2_params, num_rounds=150, stage_name="阶段2: 精细调整（lr=0.05）")
    
    # 阶段3: 微调优化（低学习率）
    stage3_params = {**base_params, 'learning_rate': 0.01}
    trainer.train_stage(stage3_params, num_rounds=100, stage_name="阶段3: 微调优化（lr=0.01）")
    
    # 可选：阶段4: 极致优化（极低学习率）
    print("\n" + "="*80)
    print("是否继续阶段4（极低学习率）？")
    print("当前最佳验证AUC:", max(trainer.training_history['val_auc']))
    print("="*80)
    
    # 自动判断是否需要阶段4
    recent_auc = trainer.training_history['val_auc'][-20:]  # 最近20轮
    if len(recent_auc) > 1:
        improvement = max(recent_auc) - min(recent_auc)
        if improvement > 0.001:  # 如果还在提升
            print("检测到模型仍在提升，继续阶段4...")
            stage4_params = {**base_params, 'learning_rate': 0.005}
            trainer.train_stage(stage4_params, num_rounds=100, stage_name="阶段4: 极致优化（lr=0.005）")
        else:
            print("模型已收敛，跳过阶段4")
    
    # 绘制训练历史
    print("\n[5/5] 生成可视化...")
    trainer.plot_training_history()
    
    # 评估
    print("\n最终评估:")
    metrics = trainer.evaluate()
    print(f"  训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"  验证集 AUC: {metrics['val_auc']:.4f}")
    print(f"  验证集 F1: {metrics['val_f1']:.4f}")
    print(f"  验证集召回率: {metrics['val_recall']:.4f}")
    
    # 保存模型
    trainer.save_model()
    
    # 保存元数据
    meta = {
        'model_type': 'XGBoost_Advanced_MultiStage',
        'training_strategy': 'Multi-stage learning rate decay',
        'total_iterations': len(trainer.training_history['iterations']),
        'stages': [
            {'stage': 1, 'lr': 0.1, 'rounds': 100},
            {'stage': 2, 'lr': 0.05, 'rounds': 150},
            {'stage': 3, 'lr': 0.01, 'rounds': 100}
        ],
        'metrics': metrics,
        'feature_names': feature_names,
        'best_val_auc': max(trainer.training_history['val_auc']),
        'training_history': {
            'total_iterations': len(trainer.training_history['iterations']),
            'final_train_auc': trainer.training_history['train_auc'][-1],
            'final_val_auc': trainer.training_history['val_auc'][-1]
        }
    }
    
    with open('models/xgb_advanced_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("训练完成！")
    print(f"总迭代次数: {len(trainer.training_history['iterations'])}")
    print(f"最佳验证AUC: {max(trainer.training_history['val_auc']):.4f}")
    print(f"最终验证AUC: {metrics['val_auc']:.4f}")
    print("="*80)
    
    return trainer, metrics


if __name__ == '__main__':
    trainer, metrics = multi_stage_training()

