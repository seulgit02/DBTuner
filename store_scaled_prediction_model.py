from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import joblib

def keep_xgboost_for_bo(input_dir, output_dir, knob_info_path):
    # 0. Knob Info 로딩
    try:
        knob_info_df = pd.read_csv(knob_info_path)
        knob_info_df = knob_info_df.set_index('name')  # Knob 이름으로 인덱싱
        print(f"✅ Knob 정보 로드 완료: {knob_info_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Knob 정보 파일({knob_info_path})을 찾을 수 없습니다. 스크립트를 중단합니다.")
        return
    except Exception as e:
        print(f"❌ ERROR: Knob 정보 파일 로드 중 오류 발생: {e}. 스크립트를 중단합니다.")
        return

    # 1. 데이터 로딩
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue

        file_path = f"{input_dir}/{file_name}"
        print("💾file_path: ", file_path)
        df = pd.read_csv(file_path)

        # Knob 이름 순서 유지 중요
        knob_names = df.drop(columns=["tps", "latency", "Unnamed: 0", "conf_id"], errors='ignore').columns.tolist()
        print(f"  - 사용될 Knobs ({len(knob_names)}개): {knob_names}")
        A_config = df[knob_names] # Knob 순서 보장
        ex_metrics = df[["tps", "latency"]]

        X_all  = np.array(A_config)
        Y_all = np.array(ex_metrics)

        # Xgboost 모델 학습
        n = 1
        while (True):
            X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.3, shuffle=True)
            X_scaler = MinMaxScaler().fit(X_train)
            y_scaler = MinMaxScaler().fit(y_train)

            print("  - X_scaler 설정 시작 (Knob 정보 기준)...")
            X_scaler = MinMaxScaler(feature_range=(0, 1))

            num_knobs = len(knob_names)
            data_min_ = np.zeros(num_knobs)
            data_max_ = np.zeros(num_knobs)
            valid_knobs_mask = np.ones(num_knobs, dtype=bool) # Knob 정보가 있는 Knob만 표시

            for i, knob_name in enumerate(knob_names):
                if knob_name in knob_info_df.index:
                    print(f"knob_name: {knob_name}")
                    data_min_[i] = knob_info_df.loc[knob_name, 'raw_min']
                    data_max_[i] = knob_info_df.loc[knob_name, 'raw_max']
                    print(f"{knob_name}의 min: {data_min_[i]}")
                    print(f"{knob_name}의 max: {data_max_[i]}")
                else:
                    print(f"    ⚠️ WARNING: Knob '{knob_name}' 정보가 CSV에 없습니다. 이 Knob은 스케일링에서 제외됩니다 (원본 값 사용).")
                    valid_knobs_mask[i] = False

            # MinMaxScaler 내부 속성 직접 설정(fit 호출 안 함.)
            X_scaler.n_features_in_ = num_knobs
            if hasattr(X_scaler, 'feature_names_in_'):  # 최신 버전 호환성
                X_scaler.feature_names_in_ = np.array(knob_names, dtype=object)
            X_scaler.data_min_ = data_min_
            X_scaler.data_max_ = data_max_

            # scale_과 min_ 계산 (MinMaxScaler 내부 로직 참고)
            feature_range = X_scaler.feature_range
            data_range = data_max_ - data_min_
            # 0으로 나누기 방지: min==max인 경우 scale=1, min=0 (또는 다른 합리적 값)으로 처리
            scale_ = (feature_range[1] - feature_range[0]) / np.where(data_range == 0, 1, data_range)
            min_ = feature_range[0] - data_min_ * scale_

            # Knob 정보가 없는 경우 scale=1, min=0으로 설정하여 원본 값 유지
            scale_[~valid_knobs_mask] = 1.0
            min_[~valid_knobs_mask] = 0.0

            X_scaler.scale_ = scale_
            X_scaler.min_ = min_
            X_scaler.n_samples_seen_ = -1  # fit을 호출하지 않았음을 의미

            print("  - X_scaler 설정 완료.")
            # --- X_scaler 수동 설정 끝 ---

            # --- y_scaler는 기존 방식 유지 ---
            y_scaler = MinMaxScaler().fit(y_train)
            # --- y_scaler 끝 ---

            # 설정된 스케일러로 변환
            scaled_X_train = X_scaler.transform(X_train)
            scaled_X_test = X_scaler.transform(X_test)
            scaled_y_train = y_scaler.transform(y_train)
            scaled_y_test = y_scaler.transform(y_test)

            # --- XGBoost 모델 학습 (scaled_X 사용) ---
            print("  - XGBoost 모델 학습 시작...")
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                random_state=2, n_estimators=100, max_depth=6, learning_rate=0.1)

            xgb_model.fit(scaled_X_train, scaled_y_train)
            print("  - XGBoost 모델 학습 완료.")

            pred = xgb_model.predict(scaled_X_test)

            accuracy = r2_score(scaled_y_test, pred)
            print('RMSE : ', np.sqrt(mean_squared_error(scaled_y_test, pred)))
            print('R2_SCORE : ', r2_score(scaled_y_test, pred))

            if accuracy > 0.9:
                # 후에 모델 로드를 위해 저장
                base_name = file_name.replace("csv","")
                model_path = os.path.join(output_dir, f"{base_name}_model.pkl")
                x_scaler_path = os.path.join(output_dir, f"{base_name}_x_scaler.pkl")
                y_scaler_path = os.path.join(output_dir, f"{base_name}_y_scaler.pkl")
                knob_names_path = os.path.join(output_dir, f"{base_name}_knobs.pkl")

                # 저장
                joblib.dump(xgb_model, model_path)
                joblib.dump(X_scaler, x_scaler_path)
                joblib.dump(y_scaler, y_scaler_path)
                joblib.dump(A_config.columns.tolist(), knob_names_path)
                break

            # rmse
        print('RMSE : ', np.sqrt(mean_squared_error(scaled_y_test, pred)))
        print('R2_SCORE : ', r2_score(scaled_y_test, pred))

        print("scaled_X: ", scaled_X_test.shape)


if __name__ == "__main__":
    input_dir = "../../../data/workloads/mysql/original_data/preprocess/knob_filter"
    output_dir = "../../../data/models/xgboost_mysql/scale"
    knob_info_path = "../../../data/knob_info/Knob_Information_MySQL_v5.7.csv"
    keep_xgboost_for_bo(input_dir, output_dir, knob_info_path)

