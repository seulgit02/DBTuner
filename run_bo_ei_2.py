import os
import torch
import re
import pandas as pd
import numpy as np
import ast
import joblib
from typing import Dict, Any, Optional, Tuple
import logging
import argparse
from dotenv import load_dotenv  # pip install python-dotenv
import openai
from tabulate import tabulate # For table logging
import csv

# BoTorch
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, AcquisitionFunction, ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import BadInitialCandidatesWarning
import warnings

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler

# 사용자 정의 모듈
from knob_dependency_score import DependencyScore
from LLM_expert import query_openai, parse_llm_response, build_dependency_prompt


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
# 쿼리 호출 함수
openai.api_key = api_key


# --- 명령줄인자 ---
parser = argparse.ArgumentParser()

# logfile 이름
parser.add_argument("--logfile", type=str, required=True)
# worklaod 종류 이름
parser.add_argument("--workload", type=str, required=True)
# dependency score parameter(민감도 파라미터, 값이 작을수록 완만해지는 그래프, 더 민감한 차이도 잘 반영함)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--beta", type=float, required=True)
parser.add_argument("--gamma", type=float, required=True)
# bayesian optimization iteration 횟수
parser.add_argument("--iter", type=int, required=True)

args = parser.parse_args()

# --- 로깅 설정 ---
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('BO_Logger')
logger.setLevel(logging.INFO) # 로그 레벨 설정

# 파일 핸들러 설정 (실시간 기록)
log_file_path = f'../../../data/bo_result/{args.logfile}.log'
file_handler = logging.FileHandler(log_file_path, mode='a', encoding = 'utf-8') # 'a' 모드로 이어쓰기
file_handler.setFormatter(log_formatter)
# 파일 핸들러 추가 시 즉시 파일에 쓰도록 설정 (버퍼링 최소화)
# Python 3.7+ 에서는 FileHandler가 기본적으로 버퍼링하지 않음, 명시적 flush 불필요
logger.addHandler(file_handler)

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 3. Log Parsed Arguments as a Table (수정된 방식) ---
logger.info("") # 로그 시작 전 빈 줄 추가 (선택 사항)
logger.info("=" * 60)
logger.info("🚀 Script Execution Started with Arguments:")
logger.info("=" * 60)

# Convert args namespace to dictionary
args_dict = vars(args)

# Create table string using tabulate
args_table = tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="grid")

# <<< 핵심 수정: 테이블을 한 줄씩 로깅 >>>
for line in args_table.splitlines():
    logger.info(line)
# <<< 수정 끝 >>>

logger.info("-" * 60)
logger.info("") # 로그 끝난 후 빈 줄 추가 (선택 사항)

# --- 초기 설정 및 경로 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# --- BO 설정 ---
PERFORMANCE_THRESHOLD = 1.1 # 성능 10프로 이상 향상(1.1~1.3)
INITIAL_DEPENDENCY_WEIGHT = 1.0
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

# --- 목적 함수 관련 설정 ---
TPS_METRIC_INDEX = 0 # 0: tps, 1: latency
LTC_METRIC_INDEX = 1
IS_MAXIMIZATION = True
N_ITERATIONS = args.iter

'''
# 함수 정의
    ￮ load_data_and_components: df, xgb_model, x_scaler, y_scaler, knob_names 불러오는 함수
    ￮ prepare_initial_data:  X(conf), Y(perf) 데이터 MinMaxScaling하는 함수
    ￮ predict_performance_scaled: performance 예측값 반환하는 함수(학습된 xgboost 로드)
    ￮ calculate_dependency_weight: knob간의 dependency_score 반환하는 함수
    ￮ parse_llm_response: LLM output 파싱하는 함수(dependency_score 함수에 넣을 매개변수 파싱)
    ￮ get_fitted_model: GP(surrogate model) 업데이트 함수
    ￮ find_best_initial_point_scaled: 초기 데이터셋중 가장 성능 좋은 데이터 반환하는 함수(탐색 시작점 설정)
    ￮ select_random_initial_point_scaled: 초기 데이터셋중 랜덤 데이터 반환하는 함수(탐색 시작점 설정)
    ￮ select_random_initial_point_scaled: 초기 데이터셋중 랜덤 데이터 반환하는 함수(탐색 시작점 설정)
    ◆ DependencyWeightedAcquisitionFunction: 커스텀 Acquisition 함수 정의 클래스
        ￮
        ￮
'''
global tps_importance
tps_importance = 2.0

def load_data_and_components(csv_path: str, model_path: str, x_scaler_path: str, y_scaler_path: str, knob_names_path: str) \
        -> Tuple[pd.DataFrame, Any, MinMaxScaler, MinMaxScaler, list]:
    """데이터, 모델, 스케일러, Knob 이름을 로드"""
    # 이전 답변과 동일
    logger.info("--- 데이터 및 구성 요소 로딩 시작 ---")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ CSV 데이터 로드 완료: {csv_path} (Shape: {df.shape})")
        xgb_model = joblib.load(model_path)
        logger.info(f"✅ XGBoost 모델 로드 완료: {model_path}")
        x_scaler = joblib.load(x_scaler_path)
        logger.info(f"✅ X Scaler 로드 완료: {x_scaler_path}")
        y_scaler = joblib.load(y_scaler_path)
        logger.info(f"✅ Y Scaler 로드 완료: {y_scaler_path}")
        knob_names = joblib.load(knob_names_path)
        logger.info(f"✅ Knob 이름 로드 완료: {knob_names_path} ({len(knob_names)}개)")
        logger.info("--- 모든 구성 요소 로딩 성공 ---\n")
        return df, xgb_model, x_scaler, y_scaler, knob_names
    except FileNotFoundError as e: print(f"❌ 오류: 필수 파일 없음 - {e}"); exit()
    except Exception as e: print(f"❌ 오류: 파일 로딩 중 문제 - {e}"); exit()

def prepare_initial_data(df: pd.DataFrame, knob_names: list, x_scaler: MinMaxScaler, y_scaler: MinMaxScaler) \
        -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """로드된 Scaler로 초기 데이터를 스케일링하고 텐서/원본 배열 반환."""
    # 이전 답변과 동일
    logger.info("--- 초기 데이터 준비 시작 ---")
    try:
        X_all = df[knob_names].values
        Y_all = df[['tps', 'latency']].values
        logger.info(f"원본 데이터 Shape: X={X_all.shape}, Y={Y_all.shape}")
        scaled_X_all = x_scaler.transform(X_all)
        scaled_Y_all = y_scaler.transform(Y_all)
        logger.info("✅ 데이터 스케일링 완료 ([0, 1] 범위)")
        logger.info(f"✅ scaled_X_all: {scaled_X_all}")
        logger.info(f"✅ scaled_Y_all: {scaled_Y_all}")

        epsilon = 1e-9
        scaled_scores = (tps_importance*scaled_Y_all[:, 0])/(scaled_Y_all[:, 1]+epsilon)  #shape: (N, )
        train_X = torch.tensor(scaled_X_all, device=DEVICE, dtype=DTYPE)
        train_Y = torch.tensor(scaled_scores, device=DEVICE, dtype=DTYPE).unsqueeze(-1)
        logger.info(f"초기 학습 텐서 생성 완료: train_X={train_X.shape}, train_Y={train_Y.shape}")
        logger.info("--- 초기 데이터 준비 완료 ---\n")
        return train_X, train_Y, X_all, Y_all # 원본 배열도 반환
    except KeyError as e: logger.info(f"❌ 오류: CSV 컬럼 없음 - {e}"); exit()
    except Exception as e: logger.info(f"❌ 오류: 초기 데이터 준비 중 문제 - {e}"); exit()

# --- XGBoost 예측 함수 (스케일링된 입력/출력 버전) ---
def predict_performance_scaled(scaled_input_np: np.ndarray, xgb_model: Any) \
        -> Optional[Dict[str, float]]:
    """
    스케일링된 knob 설정(scaled_input_np)을 받아 XGBoost 모델로
    스케일링된 성능 {'scaled_tps': ..., 'scaled_latency': ...} 딕셔너리 반환
    """
    # 입력 shape 확인 (1, DIM)
    if scaled_input_np.shape != (1, len(knob_names)):
         logger.info(f"❌ 오류: predict_performance_scaled 입력 shape 오류 - 기대: (1, {len(knob_names)}), 실제: {scaled_input_np.shape}")
         return None

    # logger.info(f"--- 스케일링된 성능 예측 시작 ---")
    try:
        # XGBoost 모델 예측 (스케일링된 X 입력 -> 스케일링된 Y 출력 가정)
        scaled_predictions = xgb_model.predict(scaled_input_np) # shape (1, 2) 가정

        scaled_tps = scaled_predictions[0, 0]
        scaled_latency = scaled_predictions[0, 1]

        return {'scaled_tps': scaled_tps, 'scaled_latency': scaled_latency}

    except Exception as e:
        logger.info(f"❌ 오류: XGBoost 예측 중 문제 발생 - {e}")
        return None

# --- 의존성 가중치 계산 함수 ---
# 입력: 원본 스케일 값들, 반환: 가중치 (float)
def calculate_dependency_weight(
    prev_config_scaled: Optional[Dict[str, float]],
    prev_perf_scaled: Optional[Dict[str, float]],
    curr_config_scaled: Dict[str, float],
    curr_perf_scaled: Dict[str, float],
    knob_names: list
) -> float:

    weight = INITIAL_DEPENDENCY_WEIGHT
    if prev_perf_scaled is None or prev_config_scaled is None:
        logger.info("INFO: 첫 반복, 기본 가중치 1.0 사용.")
        return weight

    # 성능 변화 계산 (스케일링된 값 기준)
    # 키 이름이 'scaled_tps', 'scaled_latency' 임에 주의
    prev_latency_scaled = prev_perf_scaled.get('scaled_latency', 0)
    curr_latency_scaled = curr_perf_scaled.get('scaled_latency', 0)
    prev_tps_scaled = prev_perf_scaled.get('scaled_tps', 0)
    curr_tps_scaled = curr_perf_scaled.get('scaled_tps', 0)

    # 스케일링된 값으로 성능 비율 계산 (이 비교 방식이 유효한지 확인 필요)
    # 예를 들어, latency가 0에 가까워지면 비율이 불안정해질 수 있음
    prev_metric = prev_tps_scaled / (prev_latency_scaled + 1e-9)
    curr_metric = curr_tps_scaled / (curr_latency_scaled + 1e-9)

    # 스케일링된 값 기준의 성능 향상 임계값 비교
    perf_improvement = curr_metric / prev_metric
    logger.info(f"📈 성능 향상값: {perf_improvement}")
    if perf_improvement >= PERFORMANCE_THRESHOLD: # PERFORMANCE_THRESHOLD 값의 의미가 달라짐
        logger.info("🔥 INFO: 유의미한 성능 향상 감지 (스케일링 값 기준), LLM 호출...")
        try:
            prompt = build_dependency_prompt(prev_config_scaled, prev_perf_scaled, curr_config_scaled, curr_perf_scaled)
            response = query_openai(prompt)
            logger.info(f"💬LLM Resposne: {response}")
            parsed_data = parse_llm_response(response)
            relation_type = parsed_data.get("relation_type"); knob_names_from_llm = parsed_data.get("knob_names")
            logger.info(f"📊LLM 결과: 관계='{relation_type}', Knobs='{knob_names_from_llm}'")
            # POSITIVE, INVERSE
            if relation_type in ["positive", "inverse"] and knob_names_from_llm and len(knob_names_from_llm) >= 2:
                knob1_name, knob2_name = knob_names_from_llm[0], knob_names_from_llm[1]

                A_prev_scaled, A_curr_scaled = prev_config_scaled[knob1_name], curr_config_scaled[knob1_name]
                B_prev_scaled, B_curr_scaled = prev_config_scaled[knob2_name], curr_config_scaled[knob2_name]

                if relation_type == "positive": weight = DependencyScore("positive", alpha=args.alpha).dependency_score_func_ver2(A_prev_scaled, A_curr_scaled, B_prev_scaled, B_curr_scaled)
                else: weight = DependencyScore("inverse", beta=args.beta).dependency_score_func_ver2(A_prev_scaled, A_curr_scaled, B_prev_scaled, B_curr_scaled)
            # THRESHOLD
            elif relation_type == "threshold":
                 threshold_knob_name, affected_knob_name = knob_names_from_llm[0], knob_names_from_llm[1]
                 threshold_value = parsed_data.get("threshold_value")

                 A_prev_scaled, A_curr_scaled = prev_config_scaled[threshold_knob_name], curr_config_scaled[threshold_knob_name]
                 B_prev_scaled, B_curr_scaled = prev_config_scaled[affected_knob_name], curr_config_scaled[affected_knob_name]
                 weight = DependencyScore("threshold", gamma=args.gamma).dependency_score_func_ver2(A_prev_scaled, A_curr_scaled, B_prev_scaled, B_curr_scaled, threshold_value)

            # NOTHING
            elif relation_type == "nothing": print("⛔ INFO: LLM 분석 결과: 의존성 없음."); weight = INITIAL_DEPENDENCY_WEIGHT
            else: logger.info("WARN: LLM 응답에서 유효 정보 추출 실패."); weight = INITIAL_DEPENDENCY_WEIGHT
        except KeyError as e: logger.info(f"❌ 오류: LLM 반환 knob 이름 '{e}' 없음."); weight = INITIAL_DEPENDENCY_WEIGHT
        except Exception as e: logger.info(f"❌ 오류: LLM/의존성 계산 중 문제 - {e}"); weight = INITIAL_DEPENDENCY_WEIGHT
    else: logger.info("💀 INFO: 성능 변화 미미 (스케일링 값 기준), LLM 호출 건너뜀."); weight = INITIAL_DEPENDENCY_WEIGHT
    weight = max(0.1, float(weight))
    logger.info(f"INFO: 다음 반복 가중치: {weight:.4f}")
    return weight



# --- 커스텀 Acquisition 함수 ---
class DependencyWeightedAcquisitionFunction(AcquisitionFunction):
    """커스텀 Acquisition 함수"""
    # 이전 답변 클래스 내용과 동일
    def __init__(self, model: Model, base_acquisition_function: AcquisitionFunction, dependency_weight: float):
        super().__init__(model=model); self.base_acqf = base_acquisition_function; self.dependency_weight = dependency_weight
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        base_acqf_value = self.base_acqf(X); weight_tensor = torch.tensor(self.dependency_weight, device=X.device, dtype=X.dtype)
        return base_acqf_value * weight_tensor

# --- GP 모델 학습 함수 ---
def get_fitted_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """GP 모델 학습 (Y 내부 표준화 제거됨, [0, 1] 스케일 입력 사용)."""
    # 이전 답변 함수 내용과 동일 (내부 표준화 제거 버전)
    logger.info("INFO: GP 모델 학습 시작..."); model = SingleTaskGP(train_X, train_Y); mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try: fit_gpytorch_mll(mll); logger.info("INFO: GP 모델 학습 완료."); return model
    except Exception as e: logger.info(f"❌ 오류: GP 모델 학습 중 문제 - {e}"); raise e

# --- 초기 최고점 탐색 함수 (스케일링된 값 기준) ---
def find_best_initial_point_scaled(scaled_score_Y_all_np: np.ndarray) -> Tuple[float, int]:
    """초기 데이터셋에서 최고 성능 지점의 인덱스와 스케일링된 Y값 찾기"""
    target_y_scaled = scaled_score_Y_all_np
    # 비교용 값 (최소화 문제 시 부호 반전)
    y_for_comparison = target_y_scaled if IS_MAXIMIZATION else -target_y_scaled

    best_idx = y_for_comparison.argmax()
    best_y_scaled = target_y_scaled[best_idx] # 실제 스케일링된 목표값

    logger.info(f"INFO: 초기 데이터 최고 성능 (TPS/Latency, 스케일링 값): {best_y_scaled:.4f} at index {best_idx}")
    return best_y_scaled, best_idx # 스케일링된 최고 Y값과 해당 인덱스 반환

# --- 초기 랜덤 지점 선택 함수 (스케일링된 값 기준) ---
def select_random_initial_point_scaled(scaled_score_Y_all_np: np.ndarray) -> Tuple[float, int]:
    """초기 데이터셋에서 랜덤 지점의 인덱스와 스케일링된 목표 Y값을 선택합니다."""
    num_initial_points = scaled_score_Y_all_np.shape[0]
    if num_initial_points == 0:
        raise ValueError("초기 데이터셋이 비어 있습니다.")

    # 0부터 (데이터 개수 - 1) 사이의 정수 인덱스를 랜덤하게 선택
    random_idx = np.random.randint(0, num_initial_points)

    # 선택된 인덱스에 해당하는 스케일링된 목표 Y 값
    selected_y_score_scaled = scaled_score_Y_all_np[random_idx]
    # print(f"selected_y_score_scaled: {selected_y_score_scaled}")
    # print(f"selected_y_score_scaled.shape: {selected_y_score_scaled.shape}")

    logger.info(f"INFO: 초기 지점으로 랜덤 선택 (TPS/Latency, 스케일링 값): {selected_y_score_scaled} at index {random_idx}")
    return selected_y_score_scaled, random_idx # 선택된 스케일링된 Y값과 해당 인덱스 반환


# =============================================================================
# 메인 실행 로직
# =============================================================================

if __name__ == "__main__":

    BASE_DIR = "../../../data"
    CSV_FILE_PATH = os.path.join(BASE_DIR, f"workloads/mysql/original_data/preprocess/knob_filter/{args.workload}.csv") # MYSQL_YCSB_{AA/BB/EE/FF}_ORIGIN -> Historical Tuning Data
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models/xgboost_mysql/scale") # pre-trained X, Y Scaler
    BASE_FILENAME = args.workload

    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{BASE_FILENAME}._model.pkl") # performance prediction model
    X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, f"{BASE_FILENAME}._x_scaler.pkl")
    Y_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, f"{BASE_FILENAME}._y_scaler.pkl")
    KNOB_NAMES_PATH = os.path.join(MODEL_SAVE_DIR, f"{BASE_FILENAME}._knobs.pkl")
    # 1. 데이터 및 구성 요소 로드
    df, xgb_model, x_scaler, y_scaler, knob_names = load_data_and_components(
        CSV_FILE_PATH, MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, KNOB_NAMES_PATH
    )
    DIM = len(knob_names) # input dimension
    bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], device=DEVICE, dtype=DTYPE)

    # 2. 초기 데이터 준비 ([0, 1] 스케일 텐서 및 원본 배열), train_Y = scaled_score(TPS/Latency)
    train_X, train_Y, X_all_orig, Y_all_orig = prepare_initial_data(
        df, knob_names, x_scaler, y_scaler
    )
    scaled_Y_all_np = y_scaler.transform(Y_all_orig) # 역변환 위해 원본 Y도 스케일링 -> 이게 먼솔? 무슨 역변환?
    # print(f"scaled_Y_all_np: {scaled_Y_all_np.shape}")
    epsilon = 1e-9
    scaled_score_Y_all_np = (
        scaled_Y_all_np[:, 0] / scaled_Y_all_np[:, 1] + epsilon #(1000,)
    )
    # print(f"scaled_score_Y_all_np: {scaled_score_Y_all_np.shape}")

    # 3. 초기 상태 및 최고 성능 추적 변수 초기화
    # <<< 수정 시작: 상태 변수를 스케일링된 값으로 저장 >>>
    prev_config_scaled: Optional[Dict[str, float]]
    prev_perf_scaled: Optional[Dict[str, float]]
    # <<< 수정 끝 >>>
    current_dependency_weight: float = INITIAL_DEPENDENCY_WEIGHT

    # 최고 성능 추적 (스케일링된 값 기준), 처음 시작점은 랜덤으로 설정
    best_y_scaled, best_idx_init = select_random_initial_point_scaled(scaled_score_Y_all_np)
    #best_y_scaled, best_idx_init = find_best_initial_point_scaled(scaled_Y_all_np)
    best_x_scaled_tensor = train_X[best_idx_init].unsqueeze(0)

    ## ((추가)) 결과 저장을 위한 리스트
    history_weights = []
    history_best_y_scaled = [best_y_scaled]

    # 4. 베이즈 최적화 루프
    logger.info(f"\n=== 베이즈 최적화 시작 ({N_ITERATIONS}회 반복) ===")
    for iteration in range(1, N_ITERATIONS + 1):
        logger.info(f"\n--- 반복 {iteration}/{N_ITERATIONS} ---")

        # --- GP 모델(surrogate model) 학습 ---
        try:
            gp_model = get_fitted_model(train_X, train_Y)
        except Exception:
            logger.info("WARN: GP 모델 학습 실패, 이번 반복 건너뜀."); continue
            history_weights.append(current_dependency_weight)
            history_best_y_scaled.append(best_y_scaled)
            continue
        # history_best_y_scaled.append(best_y_scaled)

        # --- Acquisition Function 준비 ---
        # ucb = UpperConfidenceBound(model=gp_model, beta=6.25) # exploration(beta가 커질수록 탐색 범위 넓어짐)
        base_ei = ExpectedImprovement(model=gp_model, best_f=best_y_scaled, maximize=True)
        # Acquisition Function: GP(surrogate model)를 입력으로 받음
        custom_acqf = DependencyWeightedAcquisitionFunction(gp_model, base_ei, current_dependency_weight)
        logger.info(f"INFO: 획득 함수 현재 적용 가중치: {current_dependency_weight:.4f}")

        # --- 다음 후보 지점 탐색 ([0, 1] 스케일) ---
        logger.info("INFO: 다음 후보 지점 탐색 중...")
        try:
            candidate_normalized, acqf_value = optimize_acqf(
                custom_acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )
            logger.info("INFO: 후보 지점 탐색 완료.")
        except Exception as e:
            logger.info(f"❌ 오류: Acq Func 최적화 중 - {e}\nWARN: 이번 반복 건너뜀."); continue
            history_weights.append(current_dependency_weight)
            history_best_y_scaled.append(best_y_scaled)
            continue

        # --- 후보 지점 성능 예측 (스케일링된 값 사용) ---
        scaled_candidate_np = candidate_normalized.cpu().numpy()
        curr_perf_dict_scaled: Optional[Dict[str, float]] = predict_performance_scaled(scaled_candidate_np, xgb_model)

        if curr_perf_dict_scaled is None:
            logger.info("WARN: 후보 지점 성능 예측 실패, 이번 반복 건너뜀.")
            history_weights.append(current_dependency_weight)
            history_best_y_scaled.append(best_y_scaled)
            continue

        epsilon = 1e-9
        new_objective_value_scaled: float = (tps_importance*curr_perf_dict_scaled['scaled_tps']) / (curr_perf_dict_scaled['scaled_latency']+epsilon)
        logger.info(f"  - 예측된 성능 (TPS/Latency, 스케일링 값): {new_objective_value_scaled:.4f}")

        ############################  save tuning history data   ############################       
        # (참조)
        info_df = pd.read_csv("Knob_Information_MySQL_v5.7.csv")
        info_df.columns = info_df.columns.str.strip().str.replace("\ufeff", "", regex=False)
        # 실제 csv 컬럼명 맞춰주기(ex) Name, Default Value)
        default_values = dict(zip(info_df['name'], info_df['d_f_default']))
        # print(f"🐢🐢🐢default values: {default_values}")
        full_knob_list = list(default_values.keys())

        # tuning data 저장하는 코드 
        HISTORICAL_TUNING_DATA_FILE_PATH = os.path.join(BASE_DIR, f"workloads/mysql/original_data/preprocess/historical_data_{args.workload}.csv")
        ORIGINAL_TUNING_DATA_FILE_PATH = os.path.join(BASE_DIR, f"workloads/mysql/original_data/preprocess/MYSQL_YCSB_AA_ORIGIN.csv")
        # iteration == 1이면 csv header 초기화
        # (참조) (바꿔야 할 부분)
        if os.path.exists(ORIGINAL_TUNING_DATA_FILE_PATH):
            hist_df = pd.read_csv(ORIGINAL_TUNING_DATA_FILE_PATH)
        else: # teration == 1 and not os.path.exists(HISTORICAL_TUNING_DATA_FILE_PATH):
            with open(HISTORICAL_TUNING_DATA_FILE_PATH, mode = 'w', newline = '') as f:
                writer = csv.writer(f)
                # header = knob_names + ['tps', 'latency']
                header = full_knob_list + ['tps', 'latency'] # (참조)
                writer.writerow(header)


        # 1) 후보 config 역스케일링
        candidate_unscaled = x_scaler.inverse_transform(candidate_normalized.cpu().numpy())

        # 2) 예측된 성능 역스케일링
        perf_unscaled = y_scaler.inverse_transform(
            [[curr_perf_dict_scaled['scaled_tps'], curr_perf_dict_scaled['scaled_latency']]]
        )

        # tuning된 knob 값 매핑 -> 현재 튜닝한 knob들의 실제 값
        tuned_knob_dict = dict(zip(knob_names, candidate_unscaled[0]))


        # 3) CSV 행 구성
        # row = list(candidate_unscaled[0]) + list(perf_unscaled[0])
        row = []
        for knob in full_knob_list:
            value = tuned_knob_dict.get(knob, default_values[knob]) # tuning 안 된 knob은 default 사용
            row.append(value)

        row += list(perf_unscaled[0]) # [tps, latency]

        # 4) csv file에 append
        # (참조) (바꿔야 할 부분)
        with open(ORIGINAL_TUNING_DATA_FILE_PATH, mode='a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        #####################################################################################


        # 의존성 가중치 계산
        curr_config_scaled_dict = {knob_names[i]: scaled_candidate_np[0, i] for i in range(DIM)}
        logger.info("INFO: 다음 반복을 위한 의존성 가중치 계산")
        try:
            # (참조) 의존성 가중치 계산
            next_dependency_weight = calculate_dependency_weight(prev_config_scaled, prev_perf_scaled, curr_config_scaled_dict, curr_perf_dict_scaled, knob_names)
        except Exception as e:
            logger.info(f"❌ {e}. 기본 가중치 사용.")
            next_dependency_weight = INITIAL_DEPENDENCY_WEIGHT


        # --- GP 데이터 업데이트(새로운 config 추가) ---
        new_objective_value_scaled_tensor = torch.tensor(
            [[new_objective_value_scaled]], device=DEVICE, dtype=DTYPE
        )
        train_X = torch.cat([train_X, candidate_normalized], dim=0)
        train_Y = torch.cat([train_Y, new_objective_value_scaled_tensor], dim=0)
        logger.info(f"INFO: GP 학습 데이터 업데이트 완료. 현재 데이터 수: {train_X.shape[0]}")

        # --- 최고 성능 업데이트 (스케일링된 값 기준) ---
        # print(f"best_y_scaled: {best_y_scaled}")
        objective_value_for_comparison = new_objective_value_scaled if IS_MAXIMIZATION else -new_objective_value_scaled
        best_y_scaled_comparison = best_y_scaled if IS_MAXIMIZATION else -best_y_scaled

        # logger.info(f"objective_value_for_comparison: {objective_value_for_comparison}")
        # logger.info(f"best_y_scaled_comparison: {best_y_scaled_comparison}")
        # print(f"objective_value_for_comparison: {objective_value_for_comparison}")
        # print(f"best_y_scaled_comparison: {best_y_scaled_comparison}")
        if objective_value_for_comparison > best_y_scaled_comparison:
            best_y_scaled = new_objective_value_scaled
            best_x_scaled_tensor = candidate_normalized # (참조) 새 후보지점 선택
            logger.info(f" M_")
            logger.info(f"| ✨ 새로운 최고 성능 발견! (TPS/Latency, 스케일링 값: {best_y_scaled:.4f}) ✨ |")
            logger.info(f" L_")
        else:
            logger.info(f"INFO: 최고 성능 유지 (TPS/Latency, 스케일링 값: {best_y_scaled:.4f})")

        prev_config_scaled = curr_config_scaled_dict
        prev_perf_scaled = curr_perf_dict_scaled

        current_dependency_weight = next_dependency_weight

        history_weights.append(current_dependency_weight)
        history_best_y_scaled.append(best_y_scaled)


    # 5. 최종 결과 출력 (역스케일링)

    logger.info("\n=== 최적화 완료 ===")
    if best_x_scaled_tensor is not None:

        best_x_scaled_np = best_x_scaled_tensor.cpu().numpy()
        final_perf_scaled_dict = predict_performance_scaled(best_x_scaled_np, xgb_model) # (참조) 성능 예측
        # ... 정규화 데이터 역변환 ...
        if final_perf_scaled_dict:
             final_tps_scaled = final_perf_scaled_dict['scaled_tps']; final_latency_scaled = final_perf_scaled_dict['scaled_latency']
             final_perf_unscaled = y_scaler.inverse_transform([[final_tps_scaled, final_latency_scaled]])
             final_tps_unscaled = final_perf_unscaled[0, 0]; final_latency_unscaled = final_perf_unscaled[0, 1]
        else:
             logger.info("WARN: 최종 성능 예측 실패. 저장된 목표값만 역변환 시도.")
             temp_y_scaled = np.zeros((1, 2))
             final_perf_unscaled = y_scaler.inverse_transform(temp_y_scaled)
             final_tps_unscaled = final_perf_unscaled[0, 0] if TARGET_METRIC_INDEX == 0 else np.nan
             final_latency_unscaled = final_perf_unscaled[0, 1] if TARGET_METRIC_INDEX == 1 else np.nan
        final_x_unscaled_np = x_scaler.inverse_transform(best_x_scaled_np)
        final_x_unscaled_config = {knob_names[j]: final_x_unscaled_np[0, j] for j in range(DIM)}
        logger.info("🏆 최종 최적 설정 (원래 스케일):")
        config_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in final_x_unscaled_config.items()])
        logger.info(f"  - Knobs: {{{config_str}}}")
        logger.info(f"  - 성능 (TPS): {final_tps_unscaled:.4f}")
        logger.info(f"  - 성능 (Latency): {final_latency_unscaled:.4f}")
        final_target_metric_value = final_tps_unscaled/final_latency_unscaled
    else:
        logger.info("❌ 최적 설정을 찾지 못했습니다.")
