import os
import time
import argparse
import pandas as pd
from ctgan import CTGAN
from util_functions import create_workload_df, extract_discrete_columns

workload_list = ["AA", "BB", "EE", "FF"]
def main(args):
    knob_info_path = f"../../../data/workloads/mysql/Knob_Information_MySQL_v5.7.csv"

    for workload in workload_list:
        # 1. 데이터 경로 설정
        config_path = f"../../../data/workloads/mysql/original_data/ycsb_{workload}/configs"
        print(config_path)
        result_path = f"../../../data/workloads/mysql/original_data/ycsb_{workload}/results/external_metrics_{workload}.csv"
        print(result_path)
        # 2. [MySQL] Workload data를 Dataframe으로 변환하여 불러오기
        combined_df = create_workload_df(config_path, result_path)

        # data info 출력
        print("Columns: ", combined_df.columns)
        print("Preview: ", combined_df.head())

        # 3. discrete column 처리
        discrete_columns = extract_discrete_columns(knob_info_path)
        filtered_discrete_columns = [col for col in discrete_columns if col in combined_df.columns]
        print("filtered_discrete_columns: ", filtered_discrete_columns)

        # 4. CTGAN 모델 학습
        ctgan = CTGAN(epochs=10) # epoch 늘리면 어케되는거지
        ctgan.fit(combined_df, discrete_columns=filtered_discrete_columns)

        # 5. 샘플 수 지정(multiplier default 값 = 5)
        n_samples = len(combined_df) * args.multiplier
        synthetic_data = ctgan.sample(n_samples)

        # 6. 데이터 증강 및 저장
        output_path = f"../../../data/workloads/mysql/ctgan_data/ycsb_{workload}_result.csv"
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
