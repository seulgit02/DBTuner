import re
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

# logfile 이름
parser.add_argument("--logfile", type=str, required=True)

args = parser.parse_args()

file_name = "MYSQL_YCSB_AA_ORIGIN.csv_knob_selection_10.csv"
knob_selection_input_path = f"../../../data/bo_result/{args.logfile}.log"
workload = args.logfile[:-4]
output_path = f"../../../data/bo_result/{workload}_conf.json"
with open(knob_selection_input_path, 'r', encoding='utf-8') as f:
    log_line = f.read()

# 정규 표현식을 사용하여 'Knobs:' 뒤의 딕셔너리 형태 문자열 추출
match = re.search(r"- Knobs: (\{.*\})", log_line)

knob_info_path = "../../../data/knob_info/Knob_Information_MySQL_v5.7.csv"

knob_info_df = pd.read_csv(knob_info_path)
# type별로 저장(int, float etc ...)
type_map = dict(zip(knob_info_df['name'], knob_info_df['type']))


if match:
    knobs_str = match.group(1)

    # JSON 형식으로 만들기 위해 키 부분에 큰따옴표 추가
    knobs_str = re.sub(r'(\w+):', r'"\1":', knobs_str)
    # 숫자 뒤의 불필요한 '.0000' 제거
    knobs_str = re.sub(r'\.0000', '', knobs_str)

    try:
        # 문자열을 Python 딕셔너리로 변환
        knobs_dict = json.loads(knobs_str)
        print(f"knobs_dict: {knobs_dict}")

        updated_knobs_dict = {}

        # type 대조 후 변환
        for key, value in knobs_dict.items():
            print(f"key, value: {key}, {value}")
            if key in type_map:  # 해당 knob에 대한 타입 정보가 있는지 확인
                target_type_str = type_map[key]
                print(f"target_type_str: {target_type_str}")
                updated_value = None
                try:
                    # 타입 문자열에 따라 실제 타입으로 변환 시도
                    if target_type_str == 'integer' or 'boolean':
                        updated_value = int(value)
                    elif target_type_str == 'float':
                        updated_value = float(value)
                    elif target_type_str == 'str':
                        updated_value = str(value)
                    # 필요한 다른 타입 변환 로직 추가 (예: boolean 등)
                    # elif target_type_str == 'bool'

                    updated_knobs_dict[key] = updated_value
                    print(f"updated_value: {updated_value}")


                except ValueError as e:
                    # 타입 변환 중 오류 발생 시 (예: 'abc'를 int로 바꾸려 할 때)
                    print(
                        f"Error converting key '{key}' with value '{value}' to type '{target_type_str}': {e}. Keeping original value.")


        print(f"updated_knobs_dict: {updated_knobs_dict}")
        # 딕셔너리를 JSON 파일로 저장 (utf-8 인코딩 사용, 보기 좋게 indent 적용)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_knobs_dict, f, ensure_ascii=False, indent=4)

        print(f"설정값이 '{output_path}'에 저장되었습니다.")
        # print(knobs_dict) # 저장된 딕셔너리 내용 확인 (선택 사항)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"파싱 시도 문자열: {knobs_str}")
else:
    print("로그 라인에서 'Knobs:' 패턴을 찾을 수 없습니다.")

