"""
LSTM (长短期记忆网络) 模型训练脚本
用于捕捉特征之间的序列依赖关系
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from models import ResumeProfile
from features.vectorized_extractor import VectorizedFeatureExtractor


class JobRecommendDataset(Dataset):
    """职位推荐数据集"""
    def __init__(self, X, y):
        # LSTM 需要 3D 输入: (batch, seq_len, features)
        # 将特征reshape为序列，每个特征作为一个时间步
        self.X = torch.FloatTensor(X).unsqueeze(2)  # (N, features, 1) -> 视为 features 个时间步
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM 模型"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间步的特征维度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向 LSTM
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 因为是双向
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len=features, input_size=1)
        
        # LSTM 处理
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # 对于双向 LSTM，拼接前向和后向的最后隐藏状态
        # hidden shape: (num_layers*2, batch, hidden_dim)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        lstm_output = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # 全连接层
        x = self.relu(self.fc1(lstm_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return self.sigmoid(x).squeeze()


def load_data():
    """加载数据"""
    print("\n[1/6] 加载向量化职位数据...")
    vectorized_file = Path(__file__).parent.parent / 'data' / 'job_data_vectorized.parquet'
    vectorized_jobs_df = pd.read_parquet(vectorized_file)
    print(f"  >>> 加载了 {len(vectorized_jobs_df)} 个职位")
    
    job_dict = {}
    for idx, row in vectorized_jobs_df.iterrows():
        name = str(row.get('岗位名称', '')).strip()
        company = str(row.get('企业', '')).strip()
        if name and name != 'nan' and company and company != 'nan':
            key = f"{idx}_{name}_{company}"
            job_dict[key] = row.to_dict()
    
    print("\n[2/6] 加载交互数据...")
    events_file = Path(__file__).parent.parent / 'logs' / 'recommend_events.jsonl'
    samples = []
    with events_file.open('r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  >>> 加载了 {len(samples)} 个交互样本")
    
    return vectorized_jobs_df, job_dict, samples


def extract_features(samples, job_dict):
    """提取特征"""
    print("\n[3/6] 提取特征...")
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
    
    print(f"  >>> 特征提取完成")
    print(f"  >>> 有效样本: {len(y)}")
    print(f"  >>> 特征数: {len(feature_names)}")
    print(f"  >>> 正样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    return X, y, feature_names


def train_lstm(X_train, y_train, X_val, y_val, input_dim):
    """训练 LSTM 模型"""
    print("\n[5/6] 训练 LSTM 模型...")
    
    train_dataset = JobRecommendDataset(X_train, y_train)
    val_dataset = JobRecommendDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_dim, hidden_dim=64, num_layers=2, dropout=0.3).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_y.numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/lstm_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停于第 {epoch+1} 轮")
                break
    
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """评估模型"""
    print("\n[6/6] 模型评估...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2).to(device)
        train_probs = model(X_train_tensor).cpu().numpy()
    
    train_auc = roc_auc_score(y_train, train_probs)
    
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(2).to(device)
        val_probs = model(X_val_tensor).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
    
    val_auc = roc_auc_score(y_val, val_probs)
    val_acc = accuracy_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    
    print(f"  >>> 训练集 AUC: {train_auc:.4f}")
    print(f"  >>> 验证集 AUC: {val_auc:.4f}")
    print(f"  >>> 验证集准确率: {val_acc:.4f}")
    print(f"  >>> 验证集精确率: {val_precision:.4f}")
    print(f"  >>> 验证集召回率: {val_recall:.4f}")
    print(f"  >>> 验证集 F1: {val_f1:.4f}")
    
    return {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'val_accuracy': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    }


def main():
    print("="*80)
    print("LSTM 模型训练")
    print("="*80)
    
    vectorized_jobs_df, job_dict, samples = load_data()
    X, y, feature_names = extract_features(samples, job_dict)
    
    print("\n[4/6] 切分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  >>> 训练集: {len(y_train)} 条")
    print(f"  >>> 验证集: {len(y_val)} 条")
    
    model = train_lstm(X_train, y_train, X_val, y_val, X.shape[1])
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
    
    meta = {
        'model_type': 'LSTM',
        'model_path': 'models/lstm_model.pth',
        'feature_names': feature_names,
        'metrics': metrics,
        'train_samples': len(y_train),
        'val_samples': len(y_val)
    }
    
    with open('models/lstm_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("训练完成！")
    print(f"  - 验证集 AUC: {metrics['val_auc']:.4f}")
    print(f"  - 验证集 F1: {metrics['val_f1']:.4f}")
    print("="*80)
    
    return metrics


if __name__ == '__main__':
    main()

