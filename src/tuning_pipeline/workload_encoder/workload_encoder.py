import numpy as np
from sql_encoder import extract_sql_features, vectorize_sql_features, QUERY_TYPES
from collections import Counter

'''
    [Workload 인코딩] (SQL 단인 인코딩의 평균 벡터) + (Query Type 통계)로 구성.
    Query Type 통계는 R/W 비율처럼 쿼리 특성 반영하는데 사용.
'''
def extract_workload_vector(sql_list: list[str]):
    '''
    여러개의 Query 집합으로 이루어진 하나의 workload를 받아서
    고유 feature vector 생성
    -> np.mean(sql_vectors) + query type count
    '''
    vectors = []
    query_type_counts = Counter()

    for sql in sql_list:
        # 단일 Query에 대한 feature vector 생성
        features = extract_sql_features(sql)
        vector = vectorize_sql_features(features)
        vectors.append(vector)

        # query_type 카운트 저장
        qtype = features['query_type']
        if qtype:
            query_type_counts[qtype]+=1

    # (1) 모든 쿼리 vector 평균
    mean_vector = np.mean(vectors, axis = 0)

    # (2) query_type count 벡터화 (선택적)
    type_count_vector = np.array([query_type_counts[q] for q in QUERY_TYPES])

    # (3) 최종 워크로드 벡터 = 평균 벡터 + 쿼리타입 분포
    final_vector = np.concatenate([mean_vector, type_count_vector])

    return final_vector



