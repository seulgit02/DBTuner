import pandas as pd
import re
import os
import json
# 파일 경로 설정
mysql_dir_path = "../../../data/knob_info/mysql/shap"
postgresql_dir_path = "../../../data/knob_info/postgresql/shap"

# clustering = 70 vs clustering = 100
coarsed_knowledge = "../../../data/knowledge/role_based_mapping_knob/role_based_knob_mapping.json"
fined_knowledge = "../../../data/knowledge/role_based_mapping_knob/role_based_knob_mapping_v2_finer.json"
output_path = "../../../data/knowledge/role_based_mapping_knob/mapping_result"


def parse_knob_select_result(mysql_dir_path):
    # knob selection 데이터 하나씩 뽑아오기
    for file in os.listdir(mysql_dir_path):
        file_path = f"{mysql_dir_path}/{file}"
        # MySQL SHAP 기반 key knob 불러오기
        df = pd.read_csv(file_path)
        # knob list 파싱
        raw_text = df.loc[0, 'knob_list']
        lines = raw_text.strip().split('\n')

        # 헤더 제거 및 파싱
        parsed_data = []
        for line in lines:
            if "knob" in line and "importance" in line:
                continue  # 헤더는 건너뜀
            parts = line.strip().split()
            # 중요도는 마지막, knob 이름은 그 앞까지 모두 join
            importance = float(parts[-1])
            knob_name = ' '.join(parts[:-1])
            knob_name = re.sub(r'^\d+\s*', '', knob_name)
            parsed_data.append((knob_name, importance))

        # DataFrame으로 변환
        knob_df = pd.DataFrame(parsed_data, columns=['knob', 'importance'])
        return knob_df

def role_based_knob_mapping(mysql_dir_path, knowledge_path):
    df = parse_knob_select_result(mysql_dir_path)
    selected_knobs = df["knob"].tolist()
    print("selected_knobs_df: ", selected_knobs)
    with open(knowledge_path, "r") as f:
        role_map = json.load(f)
    mapping_results = []
    cnt = 0
    for knob in selected_knobs:
        for role, content in role_map.items():
            mysql_knobs = content.get("MySQL", [])
            postgres_knobs = content.get("PostgreSQL", [])

            if mysql_knobs and knob in mysql_knobs:
                if postgres_knobs: #PostgreSQL쪽에도 맵핑된 knob 있는 경우
                    cnt = cnt + 1
                    mapping_results.append({
                        "mysql_knob": knob,
                        "role": role,
                        "postgresql_knobs": postgres_knobs
                    })
                break
    print(f"[Role Based Mapping] 총 {cnt}개의 knob가 매칭 가능합니다.")
    df = pd.DataFrame(mapping_results)
    print("knob 매칭 결과: ", df)
    df.to_csv("knob_role_mapping_result.csv", index =False)










if __name__ == "__main__":
    role_based_knob_mapping(mysql_dir_path, coarsed_knowledge)