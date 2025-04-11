import os
import time
import argparse
import pandas as pd
from ctgan import CTGAN
from util_functions import extract_boolean_like_columns

def main(args):
    # 데이터 경로 설정
    target_dir = "../../../data/workloads/mysql/ycsb_AA"
    knob_info_path = "../../../data/workloads/mysql/Knob_Information_MySQL_v5.7.csv"
    external_metrics_path = "../../../data/workloads/mysql/ycsb_AA/results/external_metrics_AA.csv"
    # CSV 로드
    real_data = pd.read_csv(external_metrics_path)
    # data info 출력
    print("Columns: ", real_data.columns)
    print("Preview: ", real_data.head())
    # discrete column 추출
    discrete_columns = extract_boolean_like_columns(knob_info_path)
    filtered_discrete_columns = [col for col in discrete_columns if col in real_data.columns]
    print("filtered_discrete_columns: ", filtered_discrete_columns)

    # CTGAN 모델 학습
    ctgan = CTGAN(epochs=10) # epoch 늘리면 어케되는거지
    ctgan.fit(real_data, discrete_columns=filtered_discrete_columns)

    # 샘플 수 계산 및 생성
    n_samples = len(real_data) * args.multiplier
    synthetic_data = ctgan.sample(n_samples)

    # 6. 저장
    output_path = "../../../data/workloads/mysql/ctgan_data/result.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"[✔] Generated {n_samples} synthetic rows.")
    print(f"[💾] Saved to {output_path}")

if __name__ == "__main__":
    print("[Start] CTGAN augmentation...")
    start_time = time.time()
    parser = argparse.ArgumentParser(description="CTGAN-based data augmentation for YCSB-A workload")
    parser.add_argument("--multiplier", type=int, default=5, help="How many times to augment the data (e.g., 5 means 5x)")

    args = parser.parse_args()
    main(args)
    finish_time = time.time()
    consumed_time = finish_time - start_time
    print(f"[Finish] CTGAN augmentation...: [{consumed_time}sec]")
