import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

def knob_selection_with_lasso(dbms, input_dir, output_dir, target_col, top_k = 5):
    # 1. 데이터 로딩
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue

        file_path = f"{input_dir}/{file_name}"
        print("💾file_path: ", file_path)
        df = pd.read_csv(file_path)

        # 2. target 설정 -> TPS로 선택
        target_col = target_col  # MySQL: "tps" or "latency" / PostgreSQL: "result"
        X = df.drop(columns=["tps", "latency", "result"], errors = "ignore")  # knob 값들
        y = df[target_col]

        # 3. 모델 학습을 위한 데이터셋 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Lasso 값 계산
        model = LassoCV(cv = 5, max_iter = 10000, random_state=42)
        model.fit(X_train, y_train)

        # 5. 중요도 계산
        importance = pd.Series(model.coef_, index = X.columns).abs()
        mean_abs_lasso = importance.sort_values(ascending=False).reset_index()
        mean_abs_lasso.columns = ["knob", "importance"]

        selected_knobs = mean_abs_lasso.head(top_k)

        workload = file_name.split("_result")[0]

        knob_selection = {
            "dbms": [dbms],
            "workload": [workload],
            "knob_list": [selected_knobs]
        }

        df_knob_selection = pd.DataFrame(knob_selection)
        df_knob_selection.to_csv(f"{output_dir}/{workload}_knob_selection.csv", index=False)

        print(f"[저장 완료]Top-{top_k} 중요한 knob들:")
        print(selected_knobs)

if __name__ == "__main__":
    # MySQL Knob Selection
    input_dir = "../../../data/workloads/mysql/ctgan_data"
    output_dir = "../../../data/knob_info/mysql/lasso"
    target_col = "tps"
    knob_selection_with_lasso("MySQL", input_dir, output_dir, target_col)

    # PostgreSQL knob Selection
    input_dir = "../../../data/workloads/postgresql/ctgan_data"
    output_dir = "../../../data/knob_info/postgresql/lasso"
    target_col = "result"
    knob_selection_with_lasso("PostgreSQL", input_dir, output_dir, target_col)
