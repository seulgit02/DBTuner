import os
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def knob_selection_with_shap(dbms, input_dir, output_dir, target_col):
    # 1. 데이터 로딩
    for file_name in os.listdir(input_dir):
        file_path = f"{input_dir}/{file_name}"
        print("💾file_path: ", file_path)
        df = pd.read_csv(file_path)
        # 2. target 설정 -> TPS로 선택
        target_col = target_col  # MySQL: "tps" or "latency" / PostgreSQL: "result"
        X = df.drop(columns=["tps", "latency", "result"], errors = "ignore")  # knob 값들
        y = df[target_col]

        # 3. 모델 학습 (트리 기반 모델로 -> XGBoost 사용, RF나 다른 옵션들도 있음.)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)

        # 4. SHAP 값 계산
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        # 5. SHAP summary plot
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary_plot.png")

        # 7. 중요도 순 정렬된 knob 추출
        mean_abs_shap = pd.DataFrame({
            "knob": X.columns,
            "importance": shap_values.abs.mean(0).values
        }).sort_values(by="importance", ascending=False)

        # top-K knob selection
        top_k = 5
        selected_knobs = mean_abs_shap.head(top_k)
        workload = file_name.split("_result"[0])
        # knob가 선택되었으면 1, 아니면 0
        knob_selection = {
            "dbms": [dbms],
            "workload": [workload],
            "knob_list": [selected_knobs]  # 리스트 형태로 묶어서 저장
        }
        workload = file_name.split("_result")[0]
        df_knob_selection = pd.DataFrame(knob_selection)
        df_knob_selection.to_csv(f"{output_dir}/{workload}_knob_selection.csv", index=False)


        print("[저장 완료]Top-K 중요한 knob들:")
        print(selected_knobs)

if __name__ == "__main__":
    # MySQL Knob Selection
    input_dir = "../../../data/workloads/mysql/ctgan_data"
    output_dir = "../../../data/knob_info/mysql/shap"
    target_col = "tps"
    knob_selection_with_shap("MySQL", input_dir, output_dir, target_col)


    # PostgreSQL knob Selection
    # MySQL Knob Selection
    input_dir = "../../../data/workloads/postgresql/ctgan_data"
    output_dir = "../../../data/knob_info/postgresql/shap"
    target_col = "result"
    knob_selection_with_shap("PostgreSQL", input_dir, output_dir, target_col)
