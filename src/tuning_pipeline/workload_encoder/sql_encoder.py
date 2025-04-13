import re
import sqlparse
import numpy as np
from collections import Counter

'''
    [SQL 단일 인코딩] 쿼리 유형, JOIN 수, 조건 수 등 query instruction의 특성을 추출하여 인코딩
    * Query Type은 one-hot encoding 방식 -> vector index [0]~[6]까지가 Query Type.
    
    sample)
    Features: {'query_type': 'SELECT', 'num_tables': 2, 'num_joins': 1, 'num_conditions': 2, 'has_group_by': 0, 'has_order_by': 1}
    Vector: [1 0 0 0 0 0 0 2 1 2 0 1]
'''

QUERY_TYPES = ['SELECT', 'UPDATE', 'DELETE', 'INSERT', 'CREATE', 'ALTER', 'DROP']

def extract_sql_features(sql:str):
    parsed = sqlparse.parse(sql)[0]
    tokens = parsed.tokens

    features = {
        'query_type': None,
        'num_tables': 0,
        'num_joins': 0,
        'num_conditions': 0,
        'has_group_by': 0,
        'has_order_by': 0
    }
    # 대문자 처리
    sql_upper = sql.upper()

    # 1. 쿼리 유형 판별
    for qtype in QUERY_TYPES:
        if sql_upper.strip().startswith(qtype):
            features['query_type'] = qtype
            break;

    # 2. 테이블 수 추정(FROM ~ JOIN 혹은 , 기준 split)
    # (1) FROM 뒤에 있는 테이블 1개 추출
    from_match = re.search(r'FROM\s+([A-Z_][A-Z0-9_]*)', sql_upper)
    num_from_tables = 1 if from_match else 0
    # (2) JOIN 뒤에 서브쿼리 등등에 나오는 테이블 수 세기 (INNER, LEFT, RIGHT, FULL 다 포함)
    join_matches = re.findall(r'(?:LEFT\s+|RIGHT\s+|FULL\s+|INNER\s+|OUTER\s+)?JOIN\s+([A-Z_][A-Z0-9_]*)', sql_upper)
    num_join_tables = len(join_matches)
    features['num_tables'] = num_from_tables + num_join_tables

    # 3. JOIN 수
    features['num_joins'] = len(re.findall(r'\sJOIN\s', sql_upper))

    # 4. WHERE 조건 수(AND 기준 분리)
    where_match = re.search(r'\bWHERE\b\s+(.*)', sql_upper, flags=re.DOTALL)
    if not where_match:
        features['num_conditions'] = 0
    else:
        where_body = where_match.group(1)
        stop_patterns = ['GROUP BY', 'ORDER BY', ';']
        stop_idx = len(where_body)

        for pat in stop_patterns:
            idx = where_body.find(pat)
            if idx != -1:
                stop_idx = min(stop_idx, idx)

        trimmed_where = where_body[:stop_idx].strip()

        # AND 또는 OR 기준으로 나누되, 빈 문자열은 제외
        conditions = re.split(r'\s+(?:AND|OR)\s+', trimmed_where, flags=re.IGNORECASE)
        conditions = [c for c in conditions if c.strip()]  # 빈 조건 제거
        features['num_conditions'] = len(conditions)

    # 5. GROUP BY / ORDER 여부
    if 'GROUP BY' in sql_upper:
        features['has_group_by'] = 1
    if 'ORDER BY' in sql_upper:
        features['has_order_by'] = 1
    return features

def vectorize_sql_features(feature_dict):
    query_type_vector = [0] * len(QUERY_TYPES)
    if feature_dict['query_type'] in QUERY_TYPES:
        idx = QUERY_TYPES.index(feature_dict['query_type'])
        query_type_vector[idx] = 1

    numeric_vector = [
        feature_dict['num_tables'],
        feature_dict['num_joins'],
        feature_dict['num_conditions'],
        feature_dict['has_group_by'],
        feature_dict['has_order_by']
    ]

    return np.array(query_type_vector + numeric_vector)

if __name__ == "__main__":
    example_sql = """
    SELECT name, age FROM users
    JOIN purchases ON users.id = purchases.user_id
    WHERE age > 30 AND country = 'KR'
    ORDER BY name
    """

    features = extract_sql_features(example_sql)
    vector = vectorize_sql_features(features)

    print("Features:", features)
    print("Vector:", vector)



