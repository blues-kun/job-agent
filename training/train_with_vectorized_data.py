"""
使用向量化后的job_data训练XGBoost
配合recommend_events.jsonl中的交互数据
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models import ResumeProfile
from features.vectorized_extractor import VectorizedFeatureExtractor


def load_vectorized_jobs(file_path: str = 'data/job_data_vectorized.parquet') -> pd.DataFrame:
    """加载向量化后的职位数据"""
    path = Path(file_path)
    
    if not path.exists():
        # 尝试CSV格式
        csv_path = path.with_suffix('.csv')
        if csv_path.exists():
            print(f"  使用CSV格式: {csv_path}")
            return pd.read_csv(csv_path, encoding='utf-8-sig')
        else:
            raise FileNotFoundError(f"向量化数据不存在: {file_path}\n请先运行: python preprocess_job_data.py")
    
    if path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    return df


def extract_job_key(job: dict, index: int = 0) -> str:
    """
    生成职位的唯一键
    
    Args:
        job: 职位字典
        index: 行索引（用于区分重复职位）
        
    Returns:
        职位键
    """
    # 尝试多种匹配方式
    base_key = f"{job.get('岗位名称', '')}_{job.get('企业', '')}_{job.get('岗位地址', '')}"
    return base_key


def main():
    """主训练流程"""
    print("="*80)
    print("使用向量化数据训练XGBoost")
    print("="*80)
    
    # 1. 加载向量化的职位数据
    print("\n[1/6] 加载向量化职位数据...")
    try:
        vectorized_jobs_df = load_vectorized_jobs('data/job_data_vectorized.parquet')
        print(f"  >>> 加载了 {len(vectorized_jobs_df)} 个职位")
        print(f"  >>> 特征数: {len(vectorized_jobs_df.columns)}")
        
        # 创建职位索引（用于快速查找）
        # 添加行号以确保唯一性
        vectorized_jobs_df['_job_key'] = vectorized_jobs_df.apply(
            lambda row: f"{row.name}_{row.get('岗位名称', '')}_{row.get('企业', '')}_{row.get('岗位地址', '')}", 
            axis=1
        )
        
        # 检查并处理重复
        if vectorized_jobs_df['_job_key'].duplicated().any():
            print(f"  注意: 发现 {vectorized_jobs_df['_job_key'].duplicated().sum()} 个重复职位键，已添加行号区分")
        
        job_dict = vectorized_jobs_df.set_index('_job_key').to_dict('index')
        print(f"  >>> 建立职位索引 ({len(job_dict)} 个职位)")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请先运行以下命令生成向量化数据:")
        print("  python preprocess_job_data.py")
        return
    
    # 2. 加载训练数据
    print("\n[2/6] 加载交互数据...")
    logs_path = Path('logs/recommend_events.jsonl')
    
    if not logs_path.exists():
        print(f"错误: 训练数据不存在: {logs_path}")
        print("请先运行: python -m training.generate_training_data")
        return
    
    samples = []
    for line in logs_path.read_text('utf-8').splitlines():
        if line.strip():
            samples.append(json.loads(line))
    
    print(f"  >>> 加载了 {len(samples)} 个交互样本")
    
    # 3. 提取特征
    print("\n[3/6] 提取特征...")
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
            # 从action字段提取label：like=1, skip=0
            action = sample.get('action', 'skip')
            label = 1 if action == 'like' else 0
            
            # 验证并转换resume
            resume = ResumeProfile.model_validate(resume_dict)
            
            # 查找向量化的职位数据
            job_key = extract_job_key(job, i)
            
            # 尝试匹配：先精确匹配（含地址），再忽略地址匹配
            vectorized_job = None
            job_name = job.get('岗位名称', '').strip()
            job_company = job.get('企业', '').strip()
            job_addr = job.get('岗位地址', '').strip()
            
            # 策略1: 完全匹配（名称+企业+地址）
            for key, value in job_dict.items():
                if job_name == value.get('岗位名称', '').strip() and \
                   job_company == value.get('企业', '').strip() and \
                   job_addr and job_addr == value.get('岗位地址', '').strip():
                    vectorized_job = value
                    break
            
            # 策略2: 如果地址为空或未匹配，只匹配名称+企业
            if not vectorized_job and job_name and job_company:
                for key, value in job_dict.items():
                    if job_name == value.get('岗位名称', '').strip() and \
                       job_company == value.get('企业', '').strip():
                        vectorized_job = value
                        break
            
            if vectorized_job:
                # 合并原始job和向量化特征
                job_with_vectors = {**job, **vectorized_job}
            else:
                # 如果找不到，使用原始job（可能缺少向量特征）
                job_with_vectors = job
                if skipped <= 5:
                    print(f"  警告: 样本 {i} 未找到匹配的向量化数据")
            
            # 提取特征
            features = extractor.extract(resume, job_with_vectors)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            X_list.append([features.get(k, 0.0) for k in feature_names])
            y_list.append(label)
            
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  警告: 样本 {i} 处理失败: {e}")
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  >>> 特征提取完成")
    print(f"  >>> 有效样本: {len(y)}")
    print(f"  >>> 跳过样本: {skipped}")
    print(f"  >>> 特征数: {len(feature_names)}")
    print(f"  >>> 正样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"  >>> 负样本: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # 4. 数据切分
    print("\n[4/6] 切分训练集和验证集...")
    
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
    
    print(f"  >>> 训练集: {len(y_train)} 条")
    print(f"  >>> 验证集: {len(y_val)} 条")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # 5. 训练模型
    print("\n[5/6] 训练XGBoost模型...")
    
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
    
    # 6. 评估模型
    print("\n[6/6] 模型评估...")
    
    # 训练集指标
    train_auc = evals_result['train']['auc'][-1]
    val_auc = evals_result['val']['auc'][-1]
    
    print(f"  >>> 训练集AUC: {train_auc:.4f}")
    print(f"  >>> 验证集AUC: {val_auc:.4f}")
    
    # 计算准确率、精确率、召回率
    y_pred_prob = bst.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = np.mean(y_pred == y_val)
    precision = np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_pred == 1), 1)
    recall = np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_val == 1), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  >>> 验证集准确率: {accuracy:.4f}")
    print(f"  >>> 验证集精确率: {precision:.4f}")
    print(f"  >>> 验证集召回率: {recall:.4f}")
    print(f"  >>> 验证集F1分数: {f1:.4f}")
    
    # 特征重要性
    print("\n特征重要性 Top 15:")
    importance = bst.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_importance[:15], 1):
        print(f"  {i:2d}. {feat:30s}: {score:.2f}")
    
    # 7. 保存模型
    print("\n保存模型...")
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'xgb_model.json'
    bst.save_model(str(model_path))
    print(f"  >>> 模型已保存: {model_path}")
    
    # 保存元数据
    meta = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_samples': len(y),
        'n_samples_used': len(y) - skipped,
        'n_samples_skipped': skipped,
        'n_positive': int(sum(y)),
        'n_negative': int(len(y) - sum(y)),
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'val_accuracy': float(accuracy),
        'val_precision': float(precision),
        'val_recall': float(recall),
        'val_f1': float(f1),
        'feature_importance': {k: float(v) for k, v in sorted_importance[:20]},
        'used_vectorized_data': True
    }
    
    meta_path = model_dir / 'model_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  >>> 元数据已保存: {meta_path}")
    
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

