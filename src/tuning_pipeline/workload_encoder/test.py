from src.tuning_pipeline.workload_encoder.workload_io import save_incoming_workload
from src.tuning_pipeline.workload_encoder.encoder import fit_encoder, encode_sql

if __name__ == "__main__":
    # 예시 SQL workload
    sql = "SELECT * FROM users WHERE age > 30 ORDER BY created_at DESC"
    dbms = "mysql"

    # 수신 및 저장
    workload_id = save_incoming_workload(sql, dbms)
    print(f"📥 저장된 workload ID: {workload_id}")

    # 임시 학습용 corpus (실제는 historical data 사용)
    # benchmarking tool은 연구실 원격 서버 접속하여 사용.
    corpus = [
        "SELECT * FROM users",
        "SELECT name FROM customers WHERE age > 20",
        "INSERT INTO logs VALUES (...)",
        sql
    ]
    fit_encoder(corpus)

    # 인코딩
    encoded = encode_sql(sql)
    print(f"📊 인코딩 벡터 길이: {len(encoded)}")
    print(encoded)
