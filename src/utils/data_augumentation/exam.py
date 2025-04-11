import os
import time
import argparse
import pandas as pd
from ctgan import CTGAN
from util_functions import extract_boolean_like_columns

def main(args):
    # 0. 경로 설정
    workload = args.workload  # 예: ycsb_AA
    multiplier = args.multiplier
    base_dir = f"../../../data/workloads/mysql/{workload}"
    knob_info_path = "../../../data/workloads/mysql/Knob_Information_MySQL_v5.7.csv"
    external_metrics_path = os.path.join(base_dir, "results", f"external_metrics_{workload.split('_')[1]}.csv")

    # 1. 데이터 로드
    real_data = pd.read_csv(external_metrics_path)
    if 'Unnamed: 0' in real_data.columns:
        real_data.drop(columns=['Unnamed: 0'], inplace=True)

    # 2. Boolean knob 추출 (type == boolean)
    discrete_columns = extract_boolean_like_columns(knob_info_path)

    # 3. CTGAN 학습
    print("[CTGAN] 모델 학습 시작...")
    ctgan = CTGAN(epochs=10)
    ctgan.fit(real_data, discrete_columns=discrete_columns)

    # 4. 증강 데이터 생성
    n_samples = len(real_data) * multiplier
    synthetic_data = ctgan.sample(n_samples)

    # 5. config vs metric 컬럼 분리
    knob_info = pd.read_csv(knob_info_path)
    knob_columns = [col for col in knob_info['name'].tolist() if col in synthetic_data.columns]
    metric_columns = ["tps", "latency"]

    # 6. 저장 경로 설정
    config_save_dir = os.path.join(base_dir, "configs_augmented")
    result_save_path = os.path.join(base_dir, "results", f"augmented_external_metrics_{workload.split('_')[1]}_{multiplier}x.csv")
    os.makedirs(config_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    # 7. 파일 저장
    print("[SAVE] 증강된 데이터 저장 중...")
    performance_rows = []
    for idx, row in synthetic_data.iterrows():
        config_filename = f"my_{idx}.cnf"
        config_path = os.path.join(config_save_dir, config_filename)

        # .cnf 파일 저장
        with open(config_path, "w") as f:
            f.write("[mysqld]\n")
            for col in knob_columns:
                value = row[col]
                value = round(value, 6) if isinstance(value, float) else value
                f.write(f"{col} = {value}\n")

        # metrics 저장용
        perf_data = {col: row[col] for col in metric_columns}
        perf_data["config_id"] = config_filename
        performance_rows.append(perf_data)

    # 8. 성능 결과 저장
    perf_df = pd.DataFrame(performance_rows)
    perf_df = perf_df[["config_id"] + metric_columns]
    perf_df.to_csv(result_save_path, index=False)

    # 9. 로그 출력
    print(f"[✔] {n_samples}개의 증강 샘플 저장 완료")
    print(f"[💾] 설정 파일 경로: {config_save_dir}")
    print(f"[💾] 성능 결과 경로: {result_save_path}")

if __name__ == "__main__":
    print("[Start] CTGAN augmentation...")
    start_time = time.time()
    parser = argparse.ArgumentParser(description="CTGAN-based data augmentation for MySQL YCSB workload")
    parser.add_argument("--workload", type=str, default="ycsb_AA", help="Workload folder name (e.g., ycsb_AA)")
    parser.add_argument("--multiplier", type=int, default=5, help="How many times to augment the data (e.g., 5x)")
    args = parser.parse_args()
    main(args)
    print(f"[Finish] 소요 시간: {time.time() - start_time:.2f}초")
