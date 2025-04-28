import numpy as np
import pandas as pd

# knob_dependency_score.py (DependencyScore 클래스 내부 수정 예시)
class DependencyScore:
    def __init__(self, relation_type, **params):
        self.relation_type = relation_type
        self.params = params

        # 기본 파라미터 값 조정 (새로운 함수에 맞게 더 작은 값 사용)
        # self.default_alpha = 50.0
        # self.default_beta = 50.0
        # self.default_gamma = 50.0
        # self.default_theta = -0.1

    def dependency_score_func_ver1(self, A_prev, A_curr, B_prev, B_curr, threshold_value=None):
        # 입력값이 스케일링된 값(0~1)임을 가정
        delta_A = A_curr - A_prev
        delta_B = B_curr - B_prev

        alpha = self.params.get("alpha")
        beta = self.params.get("beta")
        gamma = self.params.get("gamma")
        theta = self.params.get("theta", 0)  # Threshold 파라미터

        if self.relation_type == "positive":
            # alpha 기본값을 50 대신 더 작은 값(예: 10)으로 변경
            # alpha = self.params.get("alpha", 10) # <--- 기본값 수정
            score = np.exp(-alpha * (delta_A - delta_B)**2)
            # 값의 범위를 명시적으로 [0, 1]로 제한 (혹시 모를 부동소수점 오류 대비)
            weight = 1.0 + score
            return weight

        elif self.relation_type == "inverse":
            # beta 기본값을 50 대신 더 작은 값(예: 10)으로 변경
            # beta = self.params.get("beta", 10) # <--- 기본값 수정
            score = np.exp(-beta * (delta_A + delta_B)**2)
            weight = 1.0 + score
            return weight

        elif self.relation_type == "threshold":
            T = self.params["T"]
            # theta 기본값도 필요시 조정 (예: 0)
            # theta = self.params.get("theta", 0.0)
            # gamma 기본값도 필요시 조정 (예: 10)
            # gamma = self.params.get("gamma", 10)

            if A_curr > T:
                # A가 임계값을 넘었을 때만 B의 변화를 평가
                score = np.exp(-gamma * (delta_B - theta)**2)
                weight = 1.0 + score
                return weight
            else:
                # A가 임계값을 넘지 않으면 이 의존성은 발현되지 않은 것으로 간주
                return 1.0
        else:
            raise ValueError(f"존재하지 않는 Relation Type입니다: {self.relation_type}")

    def dependency_score_func_ver2(self, A_prev, A_curr, B_prev, B_curr, threshold_value=None):
        # delta 계산은 동일
        delta_A = A_curr - A_prev
        delta_B = B_curr - B_prev

        # 파라미터 가져오기 (기본값 사용)
        alpha = self.params.get("alpha")
        beta = self.params.get("beta")
        gamma = self.params.get("gamma")
        theta = self.params.get("theta")  # Threshold 파라미터
        score = 0.0  # 기본 점수

        if self.relation_type == "positive":
            # delta_A와 delta_B가 비슷할수록 1에 가까워짐
            difference_sq = (delta_A - delta_B) ** 2
            # 점수 계산: 1 / (1 + alpha * 차이제곱)
            score = 1.0 / (1.0 + alpha * difference_sq)
            weight = 1.0 + score

        elif self.relation_type == "inverse":
            # delta_A와 delta_B의 합이 0에 가까울수록 1에 가까워짐
            sum_sq = (delta_A + delta_B) ** 2
            # 점수 계산: 1 / (1 + beta * 합제곱)
            score = 1.0 / (1.0 + beta * sum_sq)
            weight = 1.0 + score

        elif self.relation_type == "threshold":

            # --- 수정 시작: 외부 threshold_value 사용 ---

            # threshold 패턴인데 threshold_value가 제공되지 않으면 오류 발생

            if threshold_value is None:
                raise ValueError("relation_type이 'threshold'일 경우 반드시 threshold_value를 전달해야 합니다.")

            T = threshold_value

            if A_curr > T:
                diff_from_theta_sq = (delta_B - theta) ** 2
                score = 1.0 / (1.0 + gamma * diff_from_theta_sq)
                weight = 1.0 + score

            else:
                # A_curr가 임계값 T를 넘지 않으면 점수는 0
                score = 0.0
                weight = 1.0 + score
        else:
                raise ValueError(f"존재하지 않는 Relation Type입니다: {self.relation_type}")

        return max(0.0, weight)


if __name__ == "__main__":
    A_prev=0.4
    A_curr=0.6
    B_prev=0.3
    B_curr=0.6

    # alpha 커질수록 더 날카로운 피크모양(작은 차이에도 더 크게 차이)
    exam = pd.read_csv("example.csv")
    print(exam)


    result = pd.DataFrame()
    for idx, data in exam.iterrows():
        A_prev = data['A_prev']
        A_curr = data['A_curr']
        B_prev = data['B_prev']
        B_curr = data['B_curr']

        # Positive Coupling
        pscorer = DependencyScore("positive", alpha=50)
        positive_score = pscorer.dependency_score_func_ver2(A_prev, A_curr, B_prev, B_curr)

        # Inverse Relation
        iscorer = DependencyScore("inverse", beta=50)
        inverse_score = iscorer.dependency_score_func_ver2(A_prev, A_curr, B_prev, B_curr)

        # Threshold Relation
        tscorer = DependencyScore("threshold", T=0.7, theta=-0.2, gamma=50)
        threshold_score = tscorer.dependency_score_func_ver2(A_prev, A_curr, B_prev, B_curr,  0.5)
        data = data.copy()
        data['positive'] = positive_score
        data['inverse'] = inverse_score
        data['threshold'] = threshold_score

        result = result._append(data, ignore_index = True)

    result.to_csv("score_result_50")



