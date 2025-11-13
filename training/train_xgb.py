import json
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.extractor import extract
from config import JOBS_FILE
from data_preprocess import JobDataLoader
from models import ResumeProfile

def main():
    logs_path = Path('logs/recommend_events.jsonl')
    if not logs_path.exists():
        print('缺少训练样本 logs/recommend_events.jsonl')
        print('格式: {"resume": {...}, "job": {...}, "label": 0/1}')
        return
    X = []
    y = []
    for line in logs_path.read_text('utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        resume_dict = obj.get('resume') or {}
        job = obj.get('job') or {}
        label = int(obj.get('label') or 0)
        try:
            resume = ResumeProfile.model_validate(resume_dict)
        except Exception:
            continue
        feats = extract(resume, job)
        X.append([feats[k] for k in sorted(feats.keys())])
        y.append(label)
    try:
        import xgboost as xgb
    except Exception:
        print('未安装 xgboost，无法训练')
        return
    dtrain = xgb.DMatrix(
        X,
        label=y,
        feature_names=sorted(
            extract(
                ResumeProfile.model_validate({
                    'personal_info': {'current_city': ''},
                    'work_preferences': {
                        'position_type_name': [],
                        'salary_expectation': {'min_annual_package': 0}
                    },
                    'professional_summary': {
                        'total_experience_years': 0.0,
                        'education_level': ''
                    },
                    'full_resume_text': ''
                }),
                {}
            ).keys()
        )
    )
    # 数据集切分
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    if len(X) < 10:
        print(f'样本数太少({len(X)}条)，无法训练')
        return
    
    # 统计数据
    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(f'\n数据统计:')
    print(f'  总样本: {len(y)}')
    print(f'  正样本(label=1): {pos_count} ({pos_count/len(y)*100:.1f}%)')
    print(f'  负样本(label=0): {neg_count} ({neg_count/len(y)*100:.1f}%)')
    
    # 切分训练集和验证集
    if len(X) >= 20:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=sorted(
            extract(
                ResumeProfile.model_validate({
                    'personal_info': {'current_city': ''},
                    'work_preferences': {
                        'position_type_name': [],
                        'salary_expectation': {'min_annual_package': 0}
                    },
                    'professional_summary': {
                        'total_experience_years': 0.0,
                        'education_level': ''
                    },
                    'full_resume_text': ''
                }),
                {}
            ).keys()
        ))
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=sorted(
            extract(
                ResumeProfile.model_validate({
                    'personal_info': {'current_city': ''},
                    'work_preferences': {
                        'position_type_name': [],
                        'salary_expectation': {'min_annual_package': 0}
                    },
                    'professional_summary': {
                        'total_experience_years': 0.0,
                        'education_level': ''
                    },
                    'full_resume_text': ''
                }),
                {}
            ).keys()
        ))
    else:
        # 样本太少，不切分
        dtrain = xgb.DMatrix(X, label=y, feature_names=sorted(
            extract(
                ResumeProfile.model_validate({
                    'personal_info': {'current_city': ''},
                    'work_preferences': {
                        'position_type_name': [],
                        'salary_expectation': {'min_annual_package': 0}
                    },
                    'professional_summary': {
                        'total_experience_years': 0.0,
                        'education_level': ''
                    },
                    'full_resume_text': ''
                }),
                {}
            ).keys()
        ))
        dval = None
    
    # 训练
    params = {'max_depth': 5, 'eta': 0.2, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    evals = [(dtrain, 'train')]
    if dval is not None:
        evals.append((dval, 'val'))
    
    evals_result = {}
    bst = xgb.train(
        params, dtrain, 
        num_boost_round=100,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=10
    )
    
    # 计算指标
    print('\n训练指标:')
    if 'train' in evals_result and 'auc' in evals_result['train']:
        train_auc = evals_result['train']['auc'][-1]
        print(f'  训练集AUC: {train_auc:.4f}')
    
    if dval is not None and 'val' in evals_result and 'auc' in evals_result['val']:
        val_auc = evals_result['val']['auc'][-1]
        print(f'  验证集AUC: {val_auc:.4f}')
        
        # 计算验证集准确率
        y_pred_prob = bst.predict(dval)
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_val)
        precision = np.sum((y_pred == 1) & (np.array(y_val) == 1)) / max(np.sum(y_pred == 1), 1)
        recall = np.sum((y_pred == 1) & (np.array(y_val) == 1)) / max(np.sum(np.array(y_val) == 1), 1)
        print(f'  验证集准确率: {accuracy:.4f}')
        print(f'  验证集精确率: {precision:.4f}')
        print(f'  验证集召回率: {recall:.4f}')
    
    # 保存模型
    out = Path('models'); out.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(out / 'xgb_model.json'))
    
    # 保存训练统计
    stats = {
        'total_samples': len(y),
        'positive_samples': int(pos_count),
        'negative_samples': int(neg_count),
        'train_auc': float(evals_result['train']['auc'][-1]) if 'train' in evals_result else None,
        'val_auc': float(evals_result['val']['auc'][-1]) if dval and 'val' in evals_result else None,
        'timestamp': json.dumps({'now': 'placeholder'})  # 简化
    }
    with open(out / 'xgb_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f'\n✓ 训练完成: models/xgb_model.json')
    print(f'✓ 统计信息: models/xgb_stats.json')

if __name__ == '__main__':
    main()
