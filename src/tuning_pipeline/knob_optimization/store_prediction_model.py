from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import shap
import os
import joblib

def keep_xgboost_for_bo(input_dir, output_dir):
    # 1. 데이터 로딩
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue

        file_path = f"{input_dir}/{file_name}"
        print("💾file_path: ", file_path)
        df = pd.read_csv(file_path)

        # A_config = df.drop(columns = ["Unnamed: 0", "tps", "latency", "conf_id"])
        A_config = df.drop(columns=["tps", "latency", "Unnamed: 0", "conf_id"], errors='ignore')
        ex_metrics = df[["tps", "latency"]]

        X_all  = np.array(A_config)
        Y_all = np.array(ex_metrics)

        # Xgboost 모델 학습
        n = 1
        while (True):
            X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.3, shuffle=True)
            X_scaler = MinMaxScaler().fit(X_train)
            y_scaler = MinMaxScaler().fit(y_train)

            scaled_X_train = X_scaler.transform(X_train)
            scaled_X_test = X_scaler.transform(X_test)
            scaled_y_train = y_scaler.transform(y_train)
            scaled_y_test = y_scaler.transform(y_test)

            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                random_state=2, n_estimators=100, max_depth=6, learning_rate=0.1)

            xgb_model.fit(scaled_X_train, scaled_y_train)

            pred = xgb_model.predict(scaled_X_test)
            # print("scaled_X_test: ", scaled_X_test)

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
    output_dir = "../../../data/models/xgboost_mysql"
    keep_xgboost_for_bo(input_dir, output_dir)

