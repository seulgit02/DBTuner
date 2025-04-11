# src/workload_encoder/encoder.py

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 전역적으로 하나의 벡터라이저 유지
_vectorizer = TfidfVectorizer()

def fit_encoder(corpus: list[str]):
    """
    historical workload로 encoder 학습
    -> 워크로드 임베딩할때 인코더 하나 사용해서 해버릴까??
    """
    _vectorizer.fit(corpus)

def encode_sql(sql: str) -> np.ndarray:
    """
    SQL 문장을 벡터로 인코딩
    """
    vec = _vectorizer.transform([sql])
    return vec.toarray()[0]
