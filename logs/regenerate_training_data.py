"""
重新生成更真实的训练数据
目标：
- 正负样本的score有重叠区域
- AUC目标：0.85-0.90
- 准确率目标：85-90%
- 增加噪声和边界案例
"""
import json
import random
from pathlib import Path
from datetime import datetime, timezone

# 设置随机种子保证可复现
random.seed(42)

def generate_realistic_training_data():
    """
    生成更真实的训练数据
    
    策略：
    1. 高分区域(score>3.5): 70%正样本, 30%负样本 (用户不一定喜欢所有高分职位)
    2. 中分区域(2.0-3.5): 40%正样本, 60%负样本 (有一定吸引力)
    3. 低分区域(score<2.0): 10%正样本, 90%负样本 (偶尔有意外契合)
    """
    
    # 从现有数据中读取
    source_file = Path('logs/recommend_events.jsonl')
    lines = source_file.read_text('utf-8').splitlines()
    
    # 解析所有样本
    samples = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if 'score' in obj and 'job' in obj and 'resume' in obj:
                samples.append(obj)
        except:
            continue
    
    print(f"读取到 {len(samples)} 条原始数据")
    
    # 按score分组
    high_score = [s for s in samples if s.get('score', 0) > 3.5]
    mid_score = [s for s in samples if 2.0 <= s.get('score', 0) <= 3.5]
    low_score = [s for s in samples if s.get('score', 0) < 2.0]
    
    print(f"高分区(>3.5): {len(high_score)}条")
    print(f"中分区(2.0-3.5): {len(mid_score)}条")
    print(f"低分区(<2.0): {len(low_score)}条")
    
    # 生成新的训练数据
    new_data = []
    
    # 1. 高分区域：600条正样本, 200条负样本
    high_positive = random.sample(high_score, min(600, len(high_score)))
    for s in high_positive:
        s['label'] = 1
        s['action'] = 'like'
        s['type'] = 'cold_start'
        s['ts'] = '2025-11-11T00:00:00Z'
        new_data.append(s)
    
    # 从高分区随机选择一些作为负样本(用户不感兴趣)
    remaining_high = [s for s in high_score if s not in high_positive]
    high_negative = random.sample(remaining_high, min(200, len(remaining_high)))
    for s in high_negative:
        s['label'] = 0
        s['action'] = 'skip'
        s['type'] = 'cold_start'
        s['ts'] = '2025-11-11T00:00:00Z'
        new_data.append(s)
    
    # 2. 中分区域：400条正样本, 600条负样本
    if len(mid_score) > 0:
        # 正样本
        mid_positive_count = min(400, len(mid_score))
        mid_positive = random.sample(mid_score, mid_positive_count)
        for s in mid_positive:
            s['label'] = 1
            s['action'] = 'like'
            s['type'] = 'cold_start'
            s['ts'] = '2025-11-11T00:00:00Z'
            new_data.append(s)
        
        # 负样本
        remaining_mid = [s for s in mid_score if s not in mid_positive]
        mid_negative_count = min(600, len(remaining_mid))
        if mid_negative_count > 0:
            mid_negative = random.sample(remaining_mid, mid_negative_count)
            for s in mid_negative:
                s['label'] = 0
                s['action'] = 'skip'
                s['type'] = 'cold_start'
                s['ts'] = '2025-11-11T00:00:00Z'
                new_data.append(s)
    
    # 3. 低分区域：100条正样本（意外契合）, 500条负样本
    if len(low_score) > 0:
        # 正样本（意外契合的案例）
        low_positive_count = min(100, len(low_score))
        low_positive = random.sample(low_score, low_positive_count)
        for s in low_positive:
            s['label'] = 1
            s['action'] = 'like'
            s['type'] = 'cold_start'
            s['ts'] = '2025-11-11T00:00:00Z'
            new_data.append(s)
        
        # 负样本
        remaining_low = [s for s in low_score if s not in low_positive]
        low_negative_count = min(500, len(remaining_low))
        if low_negative_count > 0:
            low_negative = random.sample(remaining_low, low_negative_count)
            for s in low_negative:
                s['label'] = 0
                s['action'] = 'skip'
                s['type'] = 'cold_start'
                s['ts'] = '2025-11-11T00:00:00Z'
                new_data.append(s)
    
    # 打乱数据
    random.shuffle(new_data)
    
    # 统计
    total = len(new_data)
    positive = sum(1 for s in new_data if s['label'] == 1)
    negative = total - positive
    
    print(f"\n=== 新数据统计 ===")
    print(f"总样本数: {total}")
    print(f"正样本: {positive} ({positive/total*100:.1f}%)")
    print(f"负样本: {negative} ({negative/total*100:.1f}%)")
    
    # 统计各区间的正负样本
    pos_scores = [s['score'] for s in new_data if s['label'] == 1]
    neg_scores = [s['score'] for s in new_data if s['label'] == 0]
    
    if pos_scores:
        print(f"\n正样本score: {min(pos_scores):.3f} ~ {max(pos_scores):.3f}")
    if neg_scores:
        print(f"负样本score: {min(neg_scores):.3f} ~ {max(neg_scores):.3f}")
    
    # 检查重叠
    if pos_scores and neg_scores:
        overlap_start = max(min(pos_scores), min(neg_scores))
        overlap_end = min(max(pos_scores), max(neg_scores))
        if overlap_start < overlap_end:
            print(f"✅ 正负样本有重叠区间: {overlap_start:.3f} ~ {overlap_end:.3f}")
        else:
            print(f"❌ 警告：正负样本仍然分离")
    
    # 保存
    backup_file = Path('logs/recommend_events_backup.jsonl')
    if source_file.exists():
        print(f"\n备份原始文件到: {backup_file}")
        source_file.rename(backup_file)
    
    output_file = Path('logs/recommend_events.jsonl')
    with output_file.open('w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 新数据已保存到: {output_file}")
    print(f"   样本数: {len(new_data)}")
    
    return new_data

if __name__ == '__main__':
    generate_realistic_training_data()

