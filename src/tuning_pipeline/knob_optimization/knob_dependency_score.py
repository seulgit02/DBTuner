import numpy as np
import pandas as pd

# knob_dependency_score.py (DependencyScore 클래스 내부 수정 예시)
class DependencyScore:
    def __init__(self, relation_type, **params):
        self.relation_type = relation_type
        self.params = params

    def dependency_score_func(self, A_prev, A_curr, B_prev, B_curr):
        # 입력값이 스케일링된 값(0~1)임을 가정
        delta_A = A_curr - A_prev
        delta_B = B_curr - B_prev

        if self.relation_type == "positive":
            # alpha 기본값을 50 대신 더 작은 값(예: 10)으로 변경
            alpha = self.params.get("alpha", 10) # <--- 기본값 수정
            score = np.exp(-alpha * (delta_A - delta_B)**2)
            # 값의 범위를 명시적으로 [0, 1]로 제한 (혹시 모를 부동소수점 오류 대비)
            return np.clip(score, 0.0, 1.0)

        elif self.relation_type == "inverse":
            # beta 기본값을 50 대신 더 작은 값(예: 10)으로 변경
            beta = self.params.get("beta", 10) # <--- 기본값 수정
            score = np.exp(-beta * (delta_A + delta_B)**2)
            return np.clip(score, 0.0, 1.0)

        elif self.relation_type == "threshold":
            T = self.params["T"]
            # theta 기본값도 필요시 조정 (예: 0)
            theta = self.params.get("theta", 0.0) # <--- 기본값 수정 (예시)
            # gamma 기본값도 필요시 조정 (예: 10)
            gamma = self.params.get("gamma", 10) # <--- 기본값 수정 (예시)

            if A_curr > T:
                # A가 임계값을 넘었을 때만 B의 변화를 평가
                score = np.exp(-gamma * (delta_B - theta)**2)
                return np.clip(score, 0.0, 1.0)
            else:
                # A가 임계값을 넘지 않으면 이 의존성은 발현되지 않은 것으로 간주
                # 중립적인 점수 1.0 반환 (이 의존성으로 인해 가중치를 낮추지 않음)
                # 또는 상황에 따라 0.0 이나 다른 값을 반환할 수도 있음 (현재 로직 유지)
                return 1.0
        else:
            raise ValueError(f"존재하지 않는 Relation Type입니다: {self.relation_type}")

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
        pscorer = DependencyScore("positive", alpha=100)
        positive_score = pscorer.dependency_score_func(A_prev, A_curr, B_prev, B_curr)

        # Inverse Relation
        iscorer = DependencyScore("inverse", beta=100)
        inverse_score = iscorer.dependency_score_func(A_prev, A_curr, B_prev, B_curr)

        # Threshold Relation
        tscorer = DependencyScore("threshold", T=0.7, theta=-0.2, gamma=100)
        threshold_score = tscorer.dependency_score_func(A_prev, A_curr, B_prev, B_curr)
        data = data.copy()
        data['positive'] = positive_score
        data['inverse'] = inverse_score
        data['threshold'] = threshold_score

        result = result._append(data, ignore_index = True)

    result.to_csv("score_result")



