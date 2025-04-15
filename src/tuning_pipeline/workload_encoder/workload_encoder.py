import numpy as np
from sql_encoder import extract_sql_features, vectorize_sql_features, QUERY_TYPES
from collections import Counter

'''
    [Workload 인코딩] (SQL 단인 인코딩의 평균 벡터) + (Query Type 통계)로 구성.
    Query Type 통계는 R/W 비율처럼 쿼리 특성 반영하는데 사용.
    
    sample) 
    Workload vector shape: (19,)
    ➤ Workload Structure Avg: [0.375 0.125 0.125 0.125 0.125 0.    0.125 0.5   0.    0.5   0.125 0.   ]
    ➤ Query Type Ratio: [0.375 0.125 0.125 0.125 0.125 0.    0.125]
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

    # (2) query_type count 벡터화 + 정규화(worklaod 크기 다를 수 있으니까)
    type_count_vector = np.array([query_type_counts[q] for q in QUERY_TYPES])
    type_sum = sum(type_count_vector)
    type_normalized_vector = type_count_vector / type_sum if type_sum != 0 else type_count_vector

    # (3) 최종 워크로드 벡터 = 평균 벡터 + 쿼리타입 분포
    final_vector = np.concatenate([mean_vector, type_normalized_vector])

    return final_vector

if __name__ == "__main__":
    sql_list = [
        "SELECT name FROM users WHERE age > 30;",
        "SELECT id FROM orders WHERE status = 'delivered';",
        "INSERT INTO logs (user_id, action) VALUES (1, 'login');",
        "UPDATE users SET last_login = NOW() WHERE id = 1;",
        "DELETE FROM sessions WHERE expired = true;",
        "SELECT COUNT(*) FROM orders GROUP BY status;",
        "CREATE TABLE archived_users (id INT, name VARCHAR(100));",
        "DROP TABLE temp_sessions;"
    ]

    vector = extract_workload_vector(sql_list)

    print("Workload vector shape:", vector.shape)
    print("Workload Structure Mean Vector:\n", vector[:12])
    print("Query Type Ratio:\n", vector[12:])



