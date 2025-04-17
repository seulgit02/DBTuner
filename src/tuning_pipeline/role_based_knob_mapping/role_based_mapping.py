import pandas as pd
import re
import os
import json
# 파일 경로 설정
dir_path = "../../../data/knob_info"
mysql_dir_path = "../../../data/knob_info/mysql/shap"
postgresql_dir_path = "../../../data/knob_info/postgresql/shap"

# clustering = 70 vs clustering = 100
role_mapping_knowledge = "../../../data/knowledge/role_based_mapping/role_based_knob_mapping_expert_guided.json" # 약 33개 매칭 가능
output_path = "../../../data/knowledge/role_based_mapping/result"

def parse_knob_select_result(dir_path):
    for dbms in os.listdir(dir_path):# DBMS 기종별 구분: [mysql, postgresql]
        dbms_path = f"{dir_path}/{dbms}"
        for method in os.listdir(dbms_path): # Algorithm 분류: [lasso, shap]
            method_path = f"{dbms_path}/{method}"
            for file_name in os.listdir(method_path):
                file_path = f"{method_path}/{file_name}" # file 불러오기
                print("file_path: ", file_path)
                # MySQL SHAP 기반 key knob 불러오기
                df = pd.read_csv(file_path)
                # knob list 파싱
                raw_text = df.loc[0, 'knob_list']
                lines = raw_text.strip().split('\n')

                # 헤더 제거 및 파싱
                parsed_data = []
                for line in lines:
                    if "knob" in line and "importance" in line:
                        continue  # 헤더 처리
                    parts = line.strip().split()
                    # 중요도는 마지막, knob 이름은 그 앞까지 모두 join
                    importance = float(parts[-1])
                    knob_name = ' '.join(parts[:-1])
                    knob_name = re.sub(r'^\d+\s*', '', knob_name) # knob id값 제거
                    parsed_data.append((knob_name, importance))

                # DataFrame으로 변환
                knob_df = pd.DataFrame(parsed_data, columns=['knob', 'importance'])
                role_based_knob_mapping(dbms, file_name, knob_df, role_mapping_knowledge, output_path)


def role_based_knob_mapping(dbms, file_name, knob_df, knowledge_path, output_path):
    with open(knowledge_path, "r") as f:
        role_map = json.load(f)

    selected_knobs = knob_df["knob"].tolist()

    mapping_results = []
    cnt = 0
    for knob in selected_knobs:
        for role, content in role_map.items():
            mysql_knobs = content.get("MySQL", [])
            postgres_knobs = content.get("PostgreSQL", [])

            if dbms == "mysql" and knob in mysql_knobs:
                if postgres_knobs:
                    cnt += 1
                    mapping_results.append({
                        "mysql_knob": knob,
                        "role": role,
                        "postgresql_knobs": postgres_knobs
                    })
                break

    print(f"[{dbms}] {file_name} 매핑된 knob 수: {cnt}")
    df_result = pd.DataFrame(mapping_results)

    # 파일 이름마다 저장 (확장자 제거)
    file_name = f"[{dbms}] {file_name}"
    base_filename = os.path.splitext(file_name)[0]
    save_path =f"{output_path}/{file_name}"
    df_result.to_csv(save_path, index=False)




if __name__ == "__main__":
    parse_knob_select_result(dir_path)
    # role_based_knob_mapping(mysql_dir_path, role_mapping_knowledge, output_path)