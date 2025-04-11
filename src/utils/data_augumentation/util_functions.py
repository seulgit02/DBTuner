import pandas as pd
import os

# cnf file 전처리
def parse_cnf_file(file_path):
    # dataframe 형태로 만들어야 함.
    config = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()[5:] # knob만 dataframe column으로 수집

    for line in lines:
        line = line.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            value = float(value)
        except ValueError:
            pass

        config[key] = value
    return config

def load_cnf_to_dataframe(config_dir):
    data = []
    for filename in sorted(os.listdir(config_dir)):
        if filename.endswith(".cnf"):
            file_path = os.path.join(config_dir, filename)
            parsed = parse_cnf_file(file_path)
            # parsed["config_id"] = filename
            data.append(parsed)
    df = pd.DataFrame(data)
    return df

def load_metric_to_dataframe(config_dir_path):
    df = pd.read_csv(config_dir_path)
    metrics_df = df = df.drop(df.columns[0], axis=1)
    return metrics_df

def create_workload_df(config_dir_path, metric_dir_path):
    # configuration data를 dataframe으로 변환
    config_df = load_cnf_to_dataframe(config_dir_path)
    # metric data를 dataframe으로 변환
    metrics_df = load_metric_to_dataframe(metric_dir_path)
    combined_df = pd.concat([config_df, metrics_df], axis = 1) # df index 번호를 기준으로 concat
    return combined_df

def extract_boolean_like_columns(csv_path: object) -> object:
    df = pd.read_csv(csv_path)

    if 'type' not in df.columns or 'name' not in df.columns:
        raise ValueError("Knob Info file must contain 'type' and 'name' columns. ")

    discrete_knobs = df[df['type'].str.lower() == 'boolean']['name'].dropna().tolist()
    return discrete_knobs


if __name__ == "__main__":
    config_path = "../../../data/workloads/mysql/ycsb_AA/configs"
    result_path = "../../../data/workloads/mysql/ycsb_AA/results/external_metrics_AA.csv"
    config_df = load_cnf_to_dataframe(config_path)
    combined_df = create_workload_df(config_path, result_path)

    print("config_df: ", len(config_df))
    print("combined_df: ", len(combined_df))
    print(combined_df.columns)
    print(len(combined_df))
    print(combined_df)



