"""
将job_data.csv向量化并保存到新表
用于XGBoost训练时直接加载特征
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from data_preprocess import JobDataLoader, JobDataProcessor
from data_preprocess.salary_normalizer import SalaryNormalizer
from data_preprocess.location_processor import LocationProcessor
from data_preprocess.text_vectorizer import TextVectorizer


def preprocess_and_vectorize(
    input_file: str = 'data/job_data.csv',
    output_csv: str = 'data/job_data_vectorized.csv',
    output_parquet: str = 'data/job_data_vectorized.parquet',
    vector_size: int = 100,
    vectorize_text: bool = True
):
    """
    向量化职位数据并保存
    
    Args:
        input_file: 输入的CSV文件
        output_csv: 输出的CSV文件（可选，体积大）
        output_parquet: 输出的Parquet文件（推荐，体积小）
        vector_size: 向量维度
        vectorize_text: 是否进行文本向量化
    """
    print("="*80)
    print("职位数据向量化处理")
    print("="*80)
    
    # 1. 加载原始数据
    print("\n[1/4] 加载原始数据...")
    loader = JobDataLoader(input_file)
    df = loader.load()
    print(f"  ✓ 加载了 {len(df)} 条职位数据")
    print(f"  ✓ 原始列数: {len(df.columns)}")
    
    # 2. 基础预处理（不含文本向量化）
    print("\n[2/4] 基础预处理...")
    result_df = df.copy()
    
    # 2.1 薪资标准化
    print("  - 薪资标准化...")
    salary_data = df['岗位薪资'].apply(SalaryNormalizer.normalize)
    result_df['薪资_最低年薪'] = salary_data.apply(lambda x: x['min_annual'])
    result_df['薪资_最高年薪'] = salary_data.apply(lambda x: x['max_annual'])
    result_df['薪资_平均年薪'] = salary_data.apply(lambda x: x['avg_annual'])
    result_df['薪资_类型'] = salary_data.apply(lambda x: x['salary_type'])
    result_df['薪资_月数'] = salary_data.apply(lambda x: x['months'])
    
    valid_count = result_df['薪资_平均年薪'].notna().sum()
    print(f"    ✓ 有效薪资: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    # 2.2 地理位置处理
    print("  - 地理位置处理...")
    location_data = df['岗位地址'].apply(LocationProcessor.normalize_address)
    result_df['城市_提取'] = location_data.apply(lambda x: x['city'])
    result_df['区域'] = location_data.apply(lambda x: x['district'])
    result_df['地址_是否有效'] = location_data.apply(lambda x: x['is_valid'])
    
    valid_addr = result_df['地址_是否有效'].sum()
    print(f"    ✓ 有效地址: {valid_addr}/{len(df)} ({valid_addr/len(df)*100:.1f}%)")
    
    # 3. 文本向量化（可选）
    if vectorize_text:
        print("\n[3/4] 文本向量化...")
        text_columns = ['岗位职责', '岗位要求', '岗位名称']
        model_dir = Path('models/word2vec')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for col in text_columns:
            if col not in df.columns:
                print(f"    警告: 列 '{col}' 不存在，跳过")
                continue
            
            print(f"    - 处理 {col}...")
            texts = df[col].fillna('').astype(str).tolist()
            
            # 检查是否已有模型
            model_path = model_dir / f"{col}_w2v.model"
            vectorizer = TextVectorizer(vector_size=vector_size)
            
            if model_path.exists():
                print(f"      加载已有模型...")
                vectorizer.load(str(model_path))
            else:
                print(f"      训练新模型...")
                vectorizer.train(texts, model_path=str(model_path))
            
            # 转换为向量
            print(f"      向量化...")
            vec_df = vectorizer.to_dataframe(texts, prefix=f"{col}_vec")
            result_df = pd.concat([result_df, vec_df], axis=1)
            print(f"      ✓ 添加了 {len(vec_df.columns)} 个向量特征")
    else:
        print("\n[3/4] 跳过文本向量化")
    
    # 4. 保存结果
    print("\n[4/4] 保存向量化数据...")
    
    # 创建输出目录
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为CSV（可读性好，但文件大）
    if output_csv:
        print(f"  - 保存CSV: {output_csv}")
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        file_size = Path(output_csv).stat().st_size / (1024 * 1024)
        print(f"    ✓ 文件大小: {file_size:.1f} MB")
    
    # 保存为Parquet（压缩好，读取快）
    if output_parquet:
        print(f"  - 保存Parquet: {output_parquet}")
        result_df.to_parquet(output_parquet, index=False, compression='snappy')
        file_size = Path(output_parquet).stat().st_size / (1024 * 1024)
        print(f"    ✓ 文件大小: {file_size:.1f} MB")
    
    # 保存元数据
    meta = {
        'original_file': input_file,
        'n_rows': len(result_df),
        'n_columns': len(result_df.columns),
        'original_columns': len(df.columns),
        'added_columns': len(result_df.columns) - len(df.columns),
        'vector_size': vector_size,
        'vectorized': vectorize_text,
        'columns': list(result_df.columns)
    }
    
    meta_path = Path(output_parquet).parent / 'job_data_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  - 保存元数据: {meta_path}")
    
    # 统计信息
    print("\n" + "="*80)
    print("向量化完成！")
    print("="*80)
    print(f"\n数据统计:")
    print(f"  原始行数: {len(df)}")
    print(f"  原始列数: {len(df.columns)}")
    print(f"  处理后列数: {len(result_df.columns)}")
    print(f"  新增列数: {len(result_df.columns) - len(df.columns)}")
    
    print(f"\n新增的列类型:")
    new_cols = set(result_df.columns) - set(df.columns)
    
    # 分类统计
    salary_cols = [c for c in new_cols if '薪资' in c]
    location_cols = [c for c in new_cols if '城市' in c or '区域' in c or '地址' in c]
    vector_cols = [c for c in new_cols if '_vec' in c]
    
    if salary_cols:
        print(f"  - 薪资特征: {len(salary_cols)} 个")
    if location_cols:
        print(f"  - 地理特征: {len(location_cols)} 个")
    if vector_cols:
        print(f"  - 向量特征: {len(vector_cols)} 个")
    
    print(f"\n输出文件:")
    if output_csv and Path(output_csv).exists():
        print(f"  - CSV: {output_csv}")
    if output_parquet and Path(output_parquet).exists():
        print(f"  - Parquet: {output_parquet} (推荐使用)")
    print(f"  - 元数据: {meta_path}")
    
    return result_df


def load_vectorized_jobs(file_path: str = 'data/job_data_vectorized.parquet') -> pd.DataFrame:
    """
    加载向量化后的职位数据
    
    Args:
        file_path: 文件路径（支持.csv或.parquet）
        
    Returns:
        DataFrame
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif path.suffix == '.csv':
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    print(f"✓ 加载向量化数据: {len(df)} 行, {len(df.columns)} 列")
    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='向量化职位数据')
    parser.add_argument('--input', default='data/job_data.csv', help='输入CSV文件')
    parser.add_argument('--output-csv', default='data/job_data_vectorized.csv', help='输出CSV文件')
    parser.add_argument('--output-parquet', default='data/job_data_vectorized.parquet', help='输出Parquet文件')
    parser.add_argument('--vector-size', type=int, default=100, help='向量维度')
    parser.add_argument('--no-vectorize', action='store_true', help='不进行文本向量化（只做基础预处理）')
    
    args = parser.parse_args()
    
    preprocess_and_vectorize(
        input_file=args.input,
        output_csv=args.output_csv,
        output_parquet=args.output_parquet,
        vector_size=args.vector_size,
        vectorize_text=not args.no_vectorize
    )


if __name__ == '__main__':
    main()

