"""
基于向量化特征训练XGBoost模型
删除规则打分，纯数据驱动
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.vectorized_extractor import VectorizedFeatureExtractor
from models import ResumeProfile


def main():
    """主训练流程"""
    print("="*80)
    print("基于向量化特征训练XGBoost模型")
    print("="*80)
    
    # 1. 加载训练数据
    print("\n[1/5] 加载训练数据...")
    logs_path = Path('logs/recommend_events.jsonl')
    
    if not logs_path.exists():
        print(f"错误: 训练数据不存在: {logs_path}")
        print("请先运行: python -m training.generate_training_data")
        return
    
    samples = []
    for line in logs_path.read_text('utf-8').splitlines():
        if line.strip():
            samples.append(json.loads(line))
    
    print(f"  ✓ 加载了 {len(samples)} 个样本")
    
    # 2. 提取特征
    print("\n[2/5] 提取向量化特征...")
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
            label = int(sample.get('label', 0))
            
            # 验证并转换resume
            resume = ResumeProfile.model_validate(resume_dict)
            
            # 提取特征
            features = extractor.extract(resume, job)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            X_list.append([features.get(k, 0.0) for k in feature_names])
            y_list.append(label)
            
        except Exception as e:
            print(f"  警告: 样本 {i} 处理失败: {e}")
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  ✓ 特征提取完成")
    print(f"  ✓ 样本数: {len(y)}")
    print(f"  ✓ 特征数: {len(feature_names)}")
    print(f"  ✓ 正样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"  ✓ 负样本: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # 3. 数据切分
    print("\n[3/5] 切分训练集和验证集...")
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"错误: 缺少依赖库: {e}")
        print("请安装: pip install xgboost scikit-learn")
        return
    
    if len(y) < 20:
        print(f"错误: 样本数太少 ({len(y)} 条)，至少需要20条")
        return
    
    # 切分数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  ✓ 训练集: {len(y_train)} 条")
    print(f"  ✓ 验证集: {len(y_val)} 条")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # 4. 训练模型
    print("\n[4/5] 训练XGBoost模型...")
    
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42
    }
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    bst = xgb.train(
        params, 
        dtrain,
        num_boost_round=200,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=20
    )
    
    # 5. 评估模型
    print("\n[5/5] 模型评估...")
    
    # 训练集指标
    train_auc = evals_result['train']['auc'][-1]
    val_auc = evals_result['val']['auc'][-1]
    
    print(f"  ✓ 训练集AUC: {train_auc:.4f}")
    print(f"  ✓ 验证集AUC: {val_auc:.4f}")
    
    # 计算准确率、精确率、召回率
    y_pred_prob = bst.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = np.mean(y_pred == y_val)
    precision = np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_pred == 1), 1)
    recall = np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_val == 1), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  ✓ 验证集准确率: {accuracy:.4f}")
    print(f"  ✓ 验证集精确率: {precision:.4f}")
    print(f"  ✓ 验证集召回率: {recall:.4f}")
    print(f"  ✓ 验证集F1分数: {f1:.4f}")
    
    # 特征重要性
    print("\n特征重要性 Top 10:")
    importance = bst.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_importance[:10], 1):
        print(f"  {i}. {feat}: {score:.2f}")
    
    # 6. 保存模型
    print("\n保存模型...")
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'xgb_model.json'
    bst.save_model(str(model_path))
    print(f"  ✓ 模型已保存: {model_path}")
    
    # 保存元数据
    meta = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_samples': len(y),
        'n_positive': int(sum(y)),
        'n_negative': int(len(y) - sum(y)),
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'val_accuracy': float(accuracy),
        'val_precision': float(precision),
        'val_recall': float(recall),
        'val_f1': float(f1),
        'feature_importance': {k: float(v) for k, v in sorted_importance[:20]}
    }
    
    meta_path = model_dir / 'model_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 元数据已保存: {meta_path}")
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print("\n模型性能总结:")
    print(f"  - 验证集AUC: {val_auc:.4f}")
    print(f"  - 验证集准确率: {accuracy:.4f}")
    print(f"  - 验证集F1: {f1:.4f}")
    print(f"\n模型已保存到: {model_path}")


if __name__ == '__main__':
    main()

