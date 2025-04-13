import re
import sqlparse
import numpy as np
from collections import Counter

QUERY_TYPES = ['SELECT', 'UPDATE', 'DELETE', 'INSERT', 'CREATE', 'ALTER', 'DROP']

'''
    [SQL 단일 인코딩] 쿼리 유형, JOIN 수, 조건 수 등 query instruction의 특성을 추출하여 인코딩
'''

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
    # 뭐 하는 애지?
    sql_upper = sql.upper()

    # 1. 쿼리 유형
    for qtype in QUERY_TYPES:
        if sql_upper.strip().startswith(qtype):
            features['query_type'] = qtype
            break;

    # 2. 테이블 수 추정(FROM ~ JOIN 혹은 , 기준 split)
    from_clause = re.findall(r'FROM\s+([\w, ]+)', sql_upper)
    if from_clause:
        tables = re.split(r',\s*', from_clause[0])
        features['num_tables'] = len(tables)

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



