import os
import json
from datetime import datetime

WORKLOAD_DIR = "data/knowledge"

def save_incoming_workload(sql:str, dbms_type: str) -> str:
    '''
    워크로드 저장 및 ID 반환
    '''
    os.makedirs(WORKLOAD_DIR, exist_ok = True)
    workload_id = datetime.now().strftime("%Y%m%d%H%M%S")
    data = {
        "id": workload_id,
        "sql": sql,
        "dbms": dbms_type,
        "timestamp": workload_id
    }
    with open(f"{WORKLOAD_DIR}/{workload_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return workload_id
