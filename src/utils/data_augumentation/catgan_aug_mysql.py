import os
import argparse
import pandas as pd
from ctgan import CTGAN

def main(args):
    # 데이터 경로 설정
    target_dir = "DBTune/workloads/mysql/ycsb_AA"
    external_metrics_path = os.path.join(target_dir, "results", "externel_metrics_AA.csv")
    # CSV 로드
    real_data = pd.read_csv(external_metrics_path)
    # data info 출력
    print("Columns: ", real_data.columns)
    print("Preview: ", real_data.head())

    # CTGAN 모델 학습
    ctgan = CTGAN(epochs=10) # epoch 늘리면 어케되는거지
    ctgan.fit(real_data)

    # 샘플 수 계산 및 생성
    n_samples = len(real_data) * args.multiplier
    synthetic_data = ctgan.sample(n_samples)

    # 6. 저장
    output_path = os.path.join(target_dir, "results", f"augmented_externel_metrics_AA_{args.multiplier}x.csv")
    synthetic_data.to_csv(output_path, index=False)
    print(f"[✔] Generated {n_samples} synthetic rows.")
    print(f"[💾] Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTGAN-based data augmentation for YCSB-A workload")
    parser.add_argument("--multiplier", type=int, default=5, help="How many times to augment the data (e.g., 5 means 5x)")

    args = parser.parse_args()
    main(args)
