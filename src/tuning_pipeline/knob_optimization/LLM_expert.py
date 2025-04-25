import openai
import osimport

openai
import os
import re
import ast
from dotenv import load_dotenv  # pip install python-dotenv
from typing import Dict, Any, Optional, Tuple
import json

load_dotenv()
api_key = os.getenv("API_KEY")
# 쿼리 호출 함수
openai.api_key = api_key


def query_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 또는 "gpt-4o"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(response: str) -> Dict[str, Any]:
    """LLM 응답 파싱 (이전과 동일)."""
    # 이전 답변 함수 내용과 동일
    relation_type = None;
    knob_names = None;
    reasoning = None
    pattern_map = {"POSITIVE_COUPLING": "positive", "INVERSE_RELATION": "inverse", "THRESHOLD_DEPENDENCY": "threshold",
                   "NOTHING": "nothing"}
    try:
        pattern_match = re.search(r"-\s*pattern:\s*(\S+)", response, re.IGNORECASE)
        if pattern_match:
            raw_pattern = pattern_match.group(1).upper();
            relation_type = pattern_map.get(raw_pattern)
            if relation_type is None and raw_pattern != "NOTHING": print(
                f"WARN: 알 수 없는 LLM 패턴 '{pattern_match.group(1)}'")
            if raw_pattern == "NOTHING": relation_type = "nothing"
        knobs_match = re.search(r"-\s*knobs:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
        if knobs_match:
            try:
                parsed_list = ast.literal_eval(knobs_match.group(1))
                if isinstance(parsed_list, list):
                    knob_names = [str(item).strip() for item in parsed_list]
                else:
                    print(f"WARN: 파싱된 knobs가 리스트 아님: {knobs_match.group(1)}")
            except Exception as e:
                print(f"WARN: knobs 문자열 파싱 실패: {knobs_match.group(1)} - {e}")
        reasoning_match = re.search(r"-\s*reasoning:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        if reasoning_match: reasoning = reasoning_match.group(1).strip()
    except Exception as e:
        print(f"❌ 오류: LLM 응답 파싱 중 문제 - {e}")
    return {"relation_type": relation_type, "knob_names": knob_names, "reasoning": reasoning}


# 프롬프트 생성 함수
def build_dependency_prompt(prev_conf, prev_perf, curr_conf, curr_perf):
    return f"""
    You are a DB tuning expert.
    
    Analyze the following configuration change and performance improvement. Determine if any meaningful dependency exists among the knobs based on their change pattern and effect.
    if any knob  meaningful knob dependency is not founded, write 'pattern' = "NOTHING"
    Please identify and return one or more of:
    Knob dependency should be between two knobs
    - [POSITIVE_COUPLING]
    - [INVERSE_RELATION]
    - [THRESHOLD_DEPENDENCY]
    
    config & prev data is scaled by range of [0, 1].
    ### configuration & performance before improvement:
    [configuration] 
    {prev_conf}
    
    [performance]
    {prev_perf}

    ### configuration & performance after improvement:
    [configuration] 
    {curr_conf}
    
    [performance]
    {curr_perf}
    
    Format output as:
    [TYPE]
    - knobs: [knob_name_1, knob_name_2]
    - pattern: POSITIVE_COUPLING or INVERSE_RELATION or THRESHOLD_DEPENDENCY or NOTHING
    - reasoning: ...
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

cy
": 0.38
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
