"""
数据预处理主模块
整合薪资标准化、文本向量化、地理位置处理等功能
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
from .salary_normalizer import SalaryNormalizer
from .text_vectorizer import TextVectorizer
from .location_processor import LocationProcessor


class JobDataProcessor:
    """职位数据预处理器"""
    
    def __init__(self, vector_size: int = 100):
        """
        初始化处理器
        
        Args:
            vector_size: 文本向量维度
        """
        self.vector_size = vector_size
        self.text_vectorizers = {}  # 存储各个文本列的向量化器
    
    def process(
        self,
        df: pd.DataFrame,
        vectorize_text: bool = True,
        text_columns: Optional[List[str]] = None,
        model_dir: str = 'models/word2vec'
    ) -> pd.DataFrame:
        """
        完整的数据预处理流程
        
        Args:
            df: 原始数据DataFrame
            vectorize_text: 是否进行文本向量化
            text_columns: 需要向量化的文本列
            model_dir: 模型保存目录
            
        Returns:
            处理后的DataFrame
        """
        print("="*80)
        print("开始数据预处理")
        print("="*80)
        
        result_df = df.copy()
        
        # 1. 薪资标准化
        print("\n[1/3] 薪资标准化...")
        result_df = self._process_salary(result_df)
        
        # 2. 地理位置处理
        print("\n[2/3] 地理位置处理...")
        result_df = self._process_location(result_df)
        
        # 3. 文本向量化（可选）
        if vectorize_text:
            print("\n[3/3] 文本向量化...")
            result_df = self._process_text_vectorization(
                result_df, 
                text_columns or ['岗位职责', '岗位要求', '岗位名称'],
                model_dir
            )
        else:
            print("\n[3/3] 跳过文本向量化")
        
        print("\n" + "="*80)
        print("数据预处理完成！")
        print(f"原始特征数: {len(df.columns)}")
        print(f"处理后特征数: {len(result_df.columns)}")
        print(f"新增特征数: {len(result_df.columns) - len(df.columns)}")
        print("="*80)
        
        return result_df
    
    def _process_salary(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理薪资数据"""
        if '岗位薪资' not in df.columns:
            print("  警告: 未找到'岗位薪资'列，跳过薪资处理")
            return df
        
        result_df = df.copy()
        
        # 标准化薪资
        salary_data = df['岗位薪资'].apply(SalaryNormalizer.normalize)
        
        # 提取各个字段
        result_df['薪资_最低年薪'] = salary_data.apply(lambda x: x['min_annual'])
        result_df['薪资_最高年薪'] = salary_data.apply(lambda x: x['max_annual'])
        result_df['薪资_平均年薪'] = salary_data.apply(lambda x: x['avg_annual'])
        result_df['薪资_类型'] = salary_data.apply(lambda x: x['salary_type'])
        result_df['薪资_月数'] = salary_data.apply(lambda x: x['months'])
        
        # 统计
        valid_count = result_df['薪资_平均年薪'].notna().sum()
        print(f"  ✓ 已处理 {len(df)} 条记录")
        print(f"  ✓ 有效薪资: {valid_count} 条 ({valid_count/len(df)*100:.1f}%)")
        print(f"  ✓ 平均年薪: {result_df['薪资_平均年薪'].mean()/10000:.1f}万元")
        
        return result_df
    
    def _process_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理地理位置数据"""
        if '岗位地址' not in df.columns:
            print("  警告: 未找到'岗位地址'列，跳过位置处理")
            return df
        
        result_df = df.copy()
        
        # 标准化地址
        location_data = df['岗位地址'].apply(LocationProcessor.normalize_address)
        
        # 提取各个字段
        result_df['城市'] = location_data.apply(lambda x: x['city'])
        result_df['区域'] = location_data.apply(lambda x: x['district'])
        result_df['地址_是否有效'] = location_data.apply(lambda x: x['is_valid'])
        
        # 如果原来有'城市'列，用处理后的覆盖
        if '城市' in df.columns:
            print("  注意: 原有'城市'列已被覆盖")
        
        # 统计
        valid_count = result_df['地址_是否有效'].sum()
        city_counts = result_df['城市'].value_counts()
        
        print(f"  ✓ 已处理 {len(df)} 条记录")
        print(f"  ✓ 有效地址: {valid_count} 条 ({valid_count/len(df)*100:.1f}%)")
        print(f"  ✓ 城市分布: {dict(city_counts.head(5))}")
        
        return result_df
    
    def _process_text_vectorization(
        self, 
        df: pd.DataFrame, 
        text_columns: List[str],
        model_dir: str
    ) -> pd.DataFrame:
        """处理文本向量化"""
        result_df = df.copy()
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        
        for col in text_columns:
            if col not in df.columns:
                print(f"  警告: 列 '{col}' 不存在，跳过")
                continue
            
            print(f"\n  处理列: {col}")
            
            # 填充空值
            texts = df[col].fillna('').astype(str).tolist()
            
            # 创建向量化器
            vectorizer = TextVectorizer(vector_size=self.vector_size)
            
            # 训练并转换
            model_path = model_dir_path / f"{col}_w2v.model"
            vectors = vectorizer.fit_transform(texts, model_path=str(model_path))
            
            # 转换为DataFrame并添加到结果中
            vec_df = vectorizer.to_dataframe(texts, prefix=f"{col}_vec")
            result_df = pd.concat([result_df, vec_df], axis=1)
            
            # 保存向量化器
            self.text_vectorizers[col] = vectorizer
            
            print(f"  ✓ {col} 向量化完成，添加了 {len(vec_df.columns)} 个特征")
        
        return result_df
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        format: str = 'csv'
    ):
        """
        保存处理后的数据
        
        Args:
            df: 处理后的DataFrame
            output_path: 输出路径
            format: 保存格式 ('csv'|'jsonl'|'xlsx')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif format == 'jsonl':
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        elif format == 'xlsx':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"\n✓ 数据已保存到: {output_path}")
        print(f"  格式: {format}")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")


def main():
    """测试主流程"""
    from ..data_preprocess.loader import JobDataLoader
    from ..config import JOBS_FILE
    
    print("加载原始数据...")
    loader = JobDataLoader(JOBS_FILE)
    df = loader.load()
    
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 创建处理器
    processor = JobDataProcessor(vector_size=50)  # 测试用小维度
    
    # 处理数据（先不向量化，速度快）
    processed_df = processor.process(
        df.head(100),  # 测试用前100条
        vectorize_text=False  # 不向量化
    )
    
    print(f"\n处理后数据: {len(processed_df)} 行, {len(processed_df.columns)} 列")
    print(f"\n新增列:")
    new_cols = set(processed_df.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    # 保存样例
    processor.save_processed_data(
        processed_df,
        'data/processed_sample.csv',
        format='csv'
    )


if __name__ == '__main__':
    main()

