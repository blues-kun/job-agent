"""
文本向量化模块
使用Word2Vec对职位描述、职责等文本字段进行向量化
参考客户贷款类别划分的实现方式
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from gensim.models import Word2Vec
import jieba


class TextVectorizer:
    """文本向量化器"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        """
        初始化向量化器
        
        Args:
            vector_size: 词向量维度（维度越大，特征越丰富，但训练时间越长）
            window: 上下文窗口大小
            min_count: 词频阈值，低于此频率的词会被忽略
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model: Optional[Word2Vec] = None
    
    @staticmethod
    def tokenize_text(text: str, use_jieba: bool = True) -> List[str]:
        """
        文本分词
        
        Args:
            text: 原始文本
            use_jieba: 是否使用jieba分词（针对中文）
            
        Returns:
            分词后的token列表
        """
        if not text or not isinstance(text, str):
            return []
        
        if use_jieba:
            # 使用jieba进行中文分词
            tokens = list(jieba.cut(text))
            # 过滤停用词和标点
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
        else:
            # 使用正则表达式提取中文和数字
            tokens = re.findall(r'[\u4e00-\u9fff]+|[0-9]+(?:\.[0-9]+)?', str(text))
        
        return tokens
    
    def train(self, texts: List[str], model_path: Optional[str] = None) -> 'TextVectorizer':
        """
        训练Word2Vec模型
        
        Args:
            texts: 文本列表
            model_path: 模型保存路径（可选）
            
        Returns:
            self
        """
        # 分词
        print("开始分词...")
        sentences = [self.tokenize_text(t) for t in texts]
        
        # 过滤空句子
        sentences = [s for s in sentences if s]
        
        if not sentences:
            raise ValueError("没有有效的文本数据用于训练")
        
        print(f"分词完成，共{len(sentences)}条文本")
        
        # 训练Word2Vec模型
        print("开始训练Word2Vec模型...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=10,
            sg=0,  # CBOW模型
            seed=42
        )
        
        vocab_size = len(self.model.wv)
        print(f"Word2Vec模型训练完成！词汇量: {vocab_size}, 向量维度: {self.vector_size}")
        
        # 保存模型
        if model_path:
            self.save(model_path)
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            形状为 (n_texts, vector_size) 的numpy数组
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        vectors = []
        for text in texts:
            tokens = self.tokenize_text(text)
            if tokens:
                # 获取每个词的词向量，然后求平均
                word_vecs = [
                    self.model.wv[token] 
                    for token in tokens 
                    if token in self.model.wv
                ]
                if word_vecs:
                    avg_vec = np.mean(word_vecs, axis=0)
                else:
                    avg_vec = np.zeros(self.vector_size)
            else:
                avg_vec = np.zeros(self.vector_size)
            
            vectors.append(avg_vec)
        
        return np.array(vectors)
    
    def fit_transform(
        self, 
        texts: List[str], 
        model_path: Optional[str] = None
    ) -> np.ndarray:
        """
        训练并转换
        
        Args:
            texts: 文本列表
            model_path: 模型保存路径（可选）
            
        Returns:
            形状为 (n_texts, vector_size) 的numpy数组
        """
        self.train(texts, model_path)
        return self.transform(texts)
    
    def save(self, model_path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"模型已保存到: {model_path}")
    
    def load(self, model_path: str) -> 'TextVectorizer':
        """加载模型"""
        self.model = Word2Vec.load(model_path)
        self.vector_size = self.model.vector_size
        print(f"模型已加载: {model_path}")
        return self
    
    def to_dataframe(
        self, 
        texts: List[str], 
        prefix: str = 'text_vec'
    ) -> pd.DataFrame:
        """
        转换为DataFrame格式
        
        Args:
            texts: 文本列表
            prefix: 列名前缀
            
        Returns:
            DataFrame，每一列对应一个向量维度
        """
        vectors = self.transform(texts)
        col_names = [f"{prefix}_{i}" for i in range(self.vector_size)]
        return pd.DataFrame(vectors, columns=col_names)


def vectorize_job_data(
    df: pd.DataFrame,
    text_columns: List[str] = ['岗位职责', '岗位要求', '岗位名称'],
    vector_size: int = 100,
    model_dir: str = 'models/word2vec'
) -> pd.DataFrame:
    """
    对职位数据进行向量化
    
    Args:
        df: 原始数据DataFrame
        text_columns: 需要向量化的文本列
        vector_size: 向量维度
        model_dir: 模型保存目录
        
    Returns:
        添加了向量化特征的DataFrame
    """
    result_df = df.copy()
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    for col in text_columns:
        if col not in df.columns:
            print(f"警告: 列 '{col}' 不存在，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理列: {col}")
        print(f"{'='*60}")
        
        # 填充空值
        texts = df[col].fillna('').astype(str).tolist()
        
        # 创建向量化器
        vectorizer = TextVectorizer(vector_size=vector_size)
        
        # 训练并转换
        model_path = model_dir_path / f"{col}_w2v.model"
        vec_df = vectorizer.fit_transform(texts, model_path=str(model_path))
        
        # 转换为DataFrame并添加到结果中
        vec_df = vectorizer.to_dataframe(texts, prefix=f"{col}")
        result_df = pd.concat([result_df, vec_df], axis=1)
        
        print(f"✓ {col} 向量化完成，添加了 {len(vec_df.columns)} 个特征")
    
    return result_df


def test_text_vectorizer():
    """测试用例"""
    test_texts = [
        "负责Java后端开发，熟悉Spring框架和微服务架构",
        "参与系统分析和设计，编写高质量代码",
        "3年以上Python开发经验，熟悉Django和Flask",
        "熟练使用MySQL、Redis等数据库",
        ""
    ]
    
    print("="*80)
    print("文本向量化测试")
    print("="*80)
    
    # 创建向量化器
    vectorizer = TextVectorizer(vector_size=50)
    
    # 训练
    vectorizer.train(test_texts)
    
    # 转换
    vectors = vectorizer.transform(test_texts)
    print(f"\n向量形状: {vectors.shape}")
    print(f"第一个文本的向量（前10维）:\n{vectors[0][:10]}")
    
    # 转换为DataFrame
    df = vectorizer.to_dataframe(test_texts, prefix='test')
    print(f"\nDataFrame形状: {df.shape}")
    print(f"列名示例: {df.columns[:5].tolist()}")


if __name__ == '__main__':
    test_text_vectorizer()

