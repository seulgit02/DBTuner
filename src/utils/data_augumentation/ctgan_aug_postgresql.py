import os
import time
import argparse
import pandas as pd
from ctgan import CTGAN
from util_functions import create_postgres_workload_df, extract_discrete_columns

workload_list = ["a", "b"]
def main(args):

    for workload in workload_list:
        # 1. 데이터 경로 설정
        config_path = f"../../../data/workloads/postgresql/original_data/ycsb-{workload}"
        print(config_path)
        # 2. [PostgreSQL] Workload data를 Dataframe으로 변환하여 불러오기
        combined_df = create_postgres_workload_df(config_path)

        # data info 출력
        print("Columns: ", combined_df.columns)
        print("Preview: ", combined_df.head())

        # 3. discrete column 처리
        # PostgreSQL은 knob type이 안나와있는 관계로 수동 필터링
        discrete_columns = [
            "autovacuum",
            "data_sync_retry",
            "enable_bitmapscan",
            "enable_gathermerge",
            "enable_hashagg",
            "enable_hashjoin",
            "enable_incremental_sort",
            "enable_indexonlyscan",
            "enable_indexscan",
            "enable_material",
            "enable_mergejoin",
            "enable_nestloop",
            "enable_parallel_append",
            "enable_parallel_hash",
            "enable_partition_pruning",
            "enable_partitionwise_aggregate",
            "enable_partitionwise_join",
            "enable_seqscan",
            "enable_sort",
            "enable_tidscan",
            "full_page_writes",
            "geqo",
            "parallel_leader_participation",
            "quote_all_identifiers",
            "wal_compression",
            "wal_init_zero",
            "wal_log_hints",
            "wal_recycle"
        ]

        # 4. CTGAN 모델 학습
        ctgan = CTGAN(epochs=10) # epoch 늘리면 어케되는거지
        ctgan.fit(combined_df, discrete_columns=discrete_columns)

        # 5. 샘플 수 지정(multiplier default 값 = 5)
        n_samples = len(combined_df) * args.multiplier
        synthetic_data = ctgan.sample(n_samples)

        # 6. 데이터 증강 및 저장
        output_path = f"../../../data/workloads/postgresql/ctgan_data/ycsb-{workload}_result.csv"
        synthetic_data.to_csv(output_path, index=False)
        print(f"[✔] Generated {len(synthetic_data)} synthetic rows.")
        print(f"[💾] Saved to {output_path}")

if __name__ == "__main__":
    print("[Start] CTGAN augmentation...")
    start_time = time.time()
    parser = argparse.ArgumentParser(description=f"CTGAN-based data augmentation for ycsb workload")
    parser.add_argument("--multiplier", type=int, default=5, help="How many times to augment the data (e.g., 5 means 5x)")

    args = parser.parse_args()
    main(args)
    finish_time = time.time()
    consumed_time = finish_time - start_time
    print(f"[Finish] CTGAN augmentation...: [{consumed_time}sec]")
