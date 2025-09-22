### Knob_Optimization 모듈 실행 방법

Python IDE 실행 터미널에서 아래와 같이 실행.
실행완료된 log는 data/bo_result에서 저장한 log 파일명으로 확인 가능.
```
cd DBTuner\src\tuning_pipeline\knob_optimization
python run_bo.py --logfile {logfile_name} --workload {workload_name} --alpha {int} --beta {int} -gamma {int} --iter {bo_iteration_num}
```

**run_bo.py 실행시 argparse**
- logfile_name: 저장할 로그 파일 이름
- workload: knob optimization 실행할 workload 정보(MYSQL_YCSB_AA_FILTERED orMYSQL_YCSB_BB_FILTERED or MYSQL_YCSB_EE_FILTERED or MYSQL_YCSB_FF_FILTERED)
- alpha: knob dependency score 가중치 파라미터(alpha값 높을 수록 positive coupling 함수에서 민감한 차이에도 score 가중치 gap이 커짐. 10/30/50/100)
- beta: knob dependency score 가중치 파라미터(alpha값 높을 수록 inverse relation 함수에서 민감한 차이에도 score 가중치 gap이 커짐. 10/30/50/100)
- gamma: knob dependency score 가중치 파라미터(alpha값 높을 수록 threshold trigger 함수에서 민감한 차이에도 score 가중치 gap이 커짐. 10/30/50/100)
- iter: Bayesian Optimization 몇번 반복할지 횟수 입력



**SRC DIR 구조**
```
src/
└── tuning_pipeline/
    ├── knob_optimization/
    │   ├── calculate_dependency_scorre.py
    │   ├── example.csv
    │   ├── knob_dependency_score.py
    │   ├── LLM_expert.py
    │   ├── run_bo.py ⭐
    │   ├── store_prediction_model.py
    │   └── store_scaled_prediction_model.py
    │
    ├── knob_selection/
    │   ├── calculate_lasso.py
    │   ├── calculate_shap.py
    │   ├── serach_top_n_similarities.py
    │  
    │
    └── role_based_knob_mapping/
    │   ├── preprocess_knobinfo.py
    │   └── role_based_mapping.py
    │
    └── workload_encoder/
            ├── workload_io.py
            ├── sql_encoder.py
            └── workload_encoder.py


```

**DATA DIR 구조**

```
data/
├── bo_result/ ⭐
│
├── knob_info/
│   ├── mysql/
│   ├── postgresql/
│   └── Knob_Information_MySQL_v5.7.csv
│
├── knowledge/
│
├── models/
│   └── xgboost_mysql/
│       └── .gitkeep
│
├── workloads/
│   ├── ctgan_visualizations/
│   │
│   ├── mysql/
│   │   ├── ctgan_data/
│   │   └── original_data/
│   │
│   └── postgresql/
│       ├── ctgan_data/
│       ├── original_data/
│       └── Knob_Information_PostgreSQL.csv

```

