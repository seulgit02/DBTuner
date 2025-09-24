import openai
import os
import re
import ast
from dotenv import load_dotenv  # pip install python-dotenv
from typing import Dict, Any, Optional, Tuple
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# 쿼리 호출 함수
openai.api_key = api_key


def query_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


# def parse_llm_response(response: str) -> Dict[str, Any]:
#     """LLM 응답 파싱 (이전과 동일)."""
#     # 이전 답변 함수 내용과 동일
#     relation_type = None;
#     knob_names = None;
#     reasoning = None
#     pattern_map = {"POSITIVE_COUPLING": "positive", "INVERSE_RELATION": "inverse", "THRESHOLD_DEPENDENCY": "threshold",
#                    "NOTHING": "nothing"}
#     try:
#         pattern_match = re.search(r"-\s*pattern:\s*(\S+)", response, re.IGNORECASE)
#         if pattern_match:
#             raw_pattern = pattern_match.group(1).upper();
#             relation_type = pattern_map.get(raw_pattern)
#             if relation_type is None and raw_pattern != "NOTHING": print(
#                 f"WARN: 알 수 없는 LLM 패턴 '{pattern_match.group(1)}'")
#             if raw_pattern == "NOTHING": relation_type = "nothing"
#         knobs_match = re.search(r"-\s*knobs:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
#         if knobs_match:
#             try:
#                 parsed_list = ast.literal_eval(knobs_match.group(1))
#                 if isinstance(parsed_list, list):
#                     knob_names = [str(item).strip() for item in parsed_list]
#                 else:
#                     print(f"WARN: 파싱된 knobs가 리스트 아님: {knobs_match.group(1)}")
#             except Exception as e:
#                 print(f"WARN: knobs 문자열 파싱 실패: {knobs_match.group(1)} - {e}")
#         reasoning_match = re.search(r"-\s*reasoning:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
#         if reasoning_match: reasoning = reasoning_match.group(1).strip()
#     except Exception as e:
#         print(f"❌ 오류: LLM 응답 파싱 중 문제 - {e}")
#     return {"relation_type": relation_type, "knob_names": knob_names, "reasoning": reasoning, "threshold_value":threshold_value}


# 프롬프트 생성 함수
# def build_dependency_prompt(assistant_id, prev_conf, prev_perf, curr_conf, curr_perf):
#     return f"""
#     You are a DB tuning expert.
#
#     Analyze the following configuration change and performance improvement. Determine if any meaningful dependency exists among the knobs based on their change pattern and effect.
#     if any knob  meaningful knob dependency is not founded, write 'pattern' = "NOTHING"
#     Please identify and return one or more of:
#     Knob dependency should be between two knobs
#     - [POSITIVE_COUPLING]
#     - [INVERSE_RELATION]
#     - [THRESHOLD_DEPENDENCY]
#
#     config & prev data is scaled by range of [0, 1].
#     ### configuration & performance before improvement:
#     [configuration]
#     {prev_conf}
#
#     [performance]
#     {prev_perf}
#
#     ### configuration & performance after improvement:
#     [configuration]
#     {curr_conf}
#
#     [performance]
#     {curr_perf}
#
#     Format output as:
#     [TYPE]
#     - knobs: [knob_name_1, knob_name_2]
#     - pattern: POSITIVE_COUPLING or INVERSE_RELATION or THRESHOLD_DEPENDENCY or NOTHING
#     - reasoning: ...
#     """
import re
import ast
from typing import Dict, Any, Optional # Optional 추가

# --- LLM 파싱 함수 (threshold_value 처리 추가) ---
# --- LLM 파싱 함수 (knobs 파싱 로직 개선) ---
def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    LLM 응답을 파싱하여 관련 정보를 추출합니다.
    다양한 'knobs' 형식('- knobs: [...]' 및 '- knobs='[...]')과
    리스트 내부의 따옴표 유무에 관계없이 파싱하도록 개선되었습니다.
    """
    relation_type: Optional[str] = None
    knob_names: Optional[list] = None
    reasoning: Optional[str] = None
    threshold_value: Optional[float] = None

    pattern_map = {
        "POSITIVE_COUPLING": "positive",
        "INVERSE_RELATION": "inverse",
        "THRESHOLD_DEPENDENCY": "threshold",
        "NOTHING": "nothing"
    }

    try:
        # 1. Pattern 파싱 (기존과 동일)
        pattern_match = re.search(r"-\s*pattern:\s*(\S+)", response, re.IGNORECASE)
        if pattern_match:
            raw_pattern = pattern_match.group(1).upper()
            relation_type = pattern_map.get(raw_pattern)
            if raw_pattern == "NOTHING": relation_type = "nothing"
            elif relation_type is None: print(f"WARN: 알 수 없는 LLM 패턴 '{pattern_match.group(1)}'")
        else:
             print("WARN: LLM 응답에서 'pattern' 필드를 찾을 수 없습니다.")

        # 2. Knobs 파싱 (<<< 수정된 로직 >>>)
        #    - knobs 다음의 구분자(= 또는 :)와 선택적인 작은따옴표 처리
        #    - 리스트 괄호 '[', ']' 사이의 내용만 캡처
        knobs_content_match = re.search(
            r"-\s*knobs\s*(?:=|:)\s*'?\[(.*?)\]'?", # 캡처 그룹이 괄호 안의 내용 (.*?)
            response,
            re.IGNORECASE | re.DOTALL
        )
        if knobs_content_match:
            inner_content = knobs_content_match.group(1).strip() # 괄호 안 내용 추출 및 좌우 공백 제거
            parsed_knobs = []
            if inner_content: # 내용이 비어있지 않으면 (예: "[]"가 아닐 때)
                items = inner_content.split(',') # 쉼표로 분리
                for item in items:
                    cleaned_item = item.strip() # 각 항목의 좌우 공백 제거
                    # 항목 시작/끝의 따옴표(작은따옴표 또는 큰따옴표) 제거
                    if (cleaned_item.startswith("'") and cleaned_item.endswith("'")) or \
                       (cleaned_item.startswith('"') and cleaned_item.endswith('"')):
                        cleaned_item = cleaned_item[1:-1]

                    # 최종적으로 비어있지 않은 문자열만 추가
                    if cleaned_item:
                        parsed_knobs.append(cleaned_item)
            knob_names = parsed_knobs # 최종 파싱된 리스트 할당
            print(f"INFO: Knobs 파싱 성공: {knob_names}")

            # --- 파싱된 Knob 리스트 유효성 검사 (기존 로직 유지 및 약간 수정) ---
            if relation_type == "nothing":
                if knob_names:
                     print(f"INFO: 'NOTHING' 패턴이지만 knobs 리스트가 비어있지 않음: {knob_names}. 빈 리스트로 강제 조정.")
                knob_names = [] # NOTHING은 항상 빈 리스트
            elif relation_type in ["positive", "inverse", "threshold"]:
                if len(knob_names) != 2:
                    print(f"WARN: '{relation_type}' 패턴인데 knob 개수가 2개가 아님: {knob_names}. 파싱 결과를 그대로 사용하나 예상과 다를 수 있음.")
                    # 필요시 여기서 knob_names = None 처리하여 오류로 간주 가능
            # -------------------------------------------------------------

        elif relation_type != "nothing": # NOTHING 아닌데 knobs 필드 못 찾으면 경고
             print(f"WARN: '{relation_type}' 패턴 응답에서 'knobs' 필드를 찾거나 파싱할 수 없습니다.")
             knob_names = None

        # 3. Threshold Value 파싱 (기존과 동일)
        if relation_type == "threshold" and knob_names is not None:
            threshold_match = re.search(r"-\s*threshold_value:\s*([0-9.]+)", response, re.IGNORECASE)
            if threshold_match:
                try:
                    value = float(threshold_match.group(1))
                    if 0.0 <= value <= 1.0:
                        threshold_value = value
                        print(f"INFO: Threshold value 파싱 성공: {threshold_value}")
                    else:
                        print(f"WARN: 파싱된 threshold_value '{value}'가 [0, 1] 범위를 벗어남. 무시됨.")
                except ValueError:
                    print(f"WARN: threshold_value를 숫자로 변환 실패: {threshold_match.group(1)}. 무시됨.")
            else:
                print(f"WARN: 'threshold' 패턴 응답에서 유효한 'threshold_value' 필드를 찾을 수 없습니다.")

        # 4. Reasoning 파싱 (기존과 동일)
        reasoning_match = re.search(r"-\s*reasoning:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

    except Exception as e:
        print(f"❌ 오류: LLM 응답 파싱 중 예외 발생 - {e}")
        return {"relation_type": None, "knob_names": None, "threshold_value": None, "reasoning": None}

    # 최종 파싱 결과 반환
    return {
        "relation_type": relation_type,
        "knob_names": knob_names,
        "threshold_value": threshold_value,
        "reasoning": reasoning
    }



def build_dependency_prompt(prev_conf: Dict[str, float], prev_perf: Dict[str, float], curr_conf: Dict[str, float], curr_perf: Dict[str, float]) -> str:
    """
    LLM에게 의존성 분석을 요청, THRESHOLD_DEPENDENCY 시 임계값(T) 추정을 포함하도록
    """
    # 입력 딕셔너리를 JSON 문자열로 변환하여 가독성 향상 (들여쓰기 2칸)
    try:
        prev_conf_str = json.dumps(prev_conf, indent=2)
        prev_perf_str = json.dumps(prev_perf, indent=2)
        curr_conf_str = json.dumps(curr_conf, indent=2)
        curr_perf_str = json.dumps(curr_perf, indent=2)
    except Exception: # JSON 변환 실패 시 간단한 문자열 변환 사용
        prev_conf_str = str(prev_conf)
        prev_perf_str = str(prev_perf)
        curr_conf_str = str(curr_conf)
        curr_perf_str = str(curr_perf)

    return f"""
    Act as an expert Database Administrator specializing in MySQL performance tuning.

    Your task is to analyze the following experimental data (scaled configuration values are in the [0, 1] range) showing a change in configuration and the resulting performance shift.

    **Critically evaluate this data using your extensive domain knowledge about MySQL knob interactions.** Based on BOTH the provided data AND your domain expertise, identify the single, most likely dependency pattern between knobs that explains the performance improvement.

    ** You should inference knob dependency between only two knobs, not three or more.
    Please identify ONLY ONE pattern type from the list below. If no meaningful dependency is found, use "NOTHING".

    Dependency Types:
    - [POSITIVE_COUPLING]: Two knobs tend to increase or decrease together for better performance.
    - [INVERSE_RELATION]: One knob increases while the other decreases (or vice-versa) for better performance.
    - [THRESHOLD_DEPENDENCY]: The effect of changing 'affected_knob_B' on performance significantly changes ONLY AFTER 'threshold_knob_A' crosses a certain **threshold value (T)**.

    ### Experimental Data Before Improvement:
    [Configuration]
    {prev_conf_str}
    [Performance]
    {prev_perf_str}

    ### Experimental Data After Improvement:
    [Configuration]
    {curr_conf_str}
    [Performance]
    {curr_perf_str}

    Format your output strictly as follows:
    [TYPE]
    - knobs: [list of relevant knob names]
    - pattern: POSITIVE_COUPLING | INVERSE_RELATION | THRESHOLD_DEPENDENCY | NOTHING
    - threshold_value: (Required ONLY for THRESHOLD_DEPENDENCY, your estimated value T for threshold_knob_A, as a number between 0 and 1)
    - reasoning: (Explain your reasoning, referencing data and/or domain knowledge shortly, not too long. If THRESHOLD_DEPENDENCY, justify the estimated threshold_value T.)

    **Important Formatting Rules:**
    1.  **'knobs' list:**
        [Important Thing]: 'scaled_tps' and 'scaled_latency' are not knob names, so they must never be included when selecting the list of relevant knob names. They are performance metrics, not knob names.
        - For POSITIVE_COUPLING / INVERSE_RELATION: Exactly **two** knob names. Example: `knobs: [knob_X, knob_Y]`
        - For THRESHOLD_DEPENDENCY: Exactly **two** knob names in the order: `[threshold_knob_A, affected_knob_B]`. Example: `knobs: [innodb_buffer_pool_size, innodb_log_file_size]`
        - For NOTHING: Empty list. Example: `knobs: []`
    2.  **'pattern'**: Must be one of the exact names listed above.
    3.  **'threshold_value'**:
        - **MUST be included ONLY when 'pattern' is THRESHOLD_DEPENDENCY.**
        - Provide your best estimate for the threshold value T for `threshold_knob_A`.
        - The value **must be a number between 0 and 1** (inclusive), reflecting the scaled data range. Example: `threshold_value: 0.75`(scaling value based on original min, max value)
        - For all other patterns (POSITIVE, INVERSE, NOTHING), this line **must be omitted**.
    4.  **'reasoning'**: Justify your choice, including the estimated T if applicable.
    """



if __name__ == "__main__":
    # 임의의 scaled configuration 및 performance 값 설정 (예: 5개의 knob)
    prev_conf = {
        "innodb_buffer_pool_size": 0.4,
        "innodb_log_file_size": 0.3,
        "max_connections": 0.5,
        "table_open_cache": 0.6,
        "query_cache_size": 0.2
    }

    curr_conf = {
        "innodb_buffer_pool_size": 0.8,
        "innodb_log_file_size": 0.3,
        "max_connections": 0.8,
        "table_open_cache": 0.9,
        "query_cache_size": 0.2
    }

    prev_perf = {
        "tps": 0.52,
        "latency": 0.38
    }

    curr_perf = {
        "tps": 0.67,
        "latency": 0.25
    }

    # 프롬프트 생성
    prompt = build_dependency_prompt(prev_conf, prev_perf, curr_conf, curr_perf)

    # OpenAI 쿼리
    result = query_openai(prompt)

    # 결과 출력
    print("===== LLM Output =====")
    print(result)

    print(parse_llm_response(result))

