import numpy as np
import pandas as pd
class DependencyScore:
    def __init__(self, relation_type, **params):
        self.relation_type = relation_type
        self.params = params
        self.default_alpha = 40
        self.default_beta = 40
        self.default_gamma = 40
        self.default_theta = 40

    def dependency_score_func(self, A_prev, A_curr, B_prev, B_curr):
        # delta 계산은 동일
        delta_A = A_curr - A_prev
        delta_B = B_curr - B_prev

        # 파라미터 가져오기 (기본값 사용)
        alpha = self.params.get("alpha", self.default_alpha)
        beta = self.params.get("beta", self.default_beta)
        gamma = self.params.get("gamma", self.default_gamma)
        theta = self.params.get("theta", self.default_theta)  # Threshold 파라미터

        score = 0.0  # 기본 점수

        if self.relation_type == "positive":
            # delta_A와 delta_B가 비슷할수록 1에 가까워짐
            difference_sq = (delta_A - delta_B) ** 2
            # 점수 계산: 1 / (1 + alpha * 차이제곱)
            score = 1.0 / (1.0 + alpha * difference_sq)

        elif self.relation_type == "inverse":
            # delta_A와 delta_B의 합이 0에 가까울수록 1에 가까워짐
            sum_sq = (delta_A + delta_B) ** 2
            # 점수 계산: 1 / (1 + beta * 합제곱)
            score = 1.0 / (1.0 + beta * sum_sq)

        elif self.relation_type == "threshold":
            # T 값은 필수 파라미터로 가정
            if "T" not in self.params:
                raise ValueError("Threshold 관계 타입('threshold')에는 'T' 파라미터가 필수입니다.")
            T = self.params["T"]

            if A_curr > T:
                # B의 변화량(delta_B)이 특정 값(theta)에 가까울수록 1에 가까워짐
                diff_from_theta_sq = (delta_B - theta) ** 2
                # 점수 계산: 1 / (1 + gamma * 차이제곱)
                score = 1.0 / (1.0 + gamma * diff_from_theta_sq)
            else:
                # A_curr가 임계값 T를 넘지 않으면 점수는 0
                score = 0.0
        else:
            raise ValueError(f"존재하지 않는 Relation Type입니다: {self.relation_type}")

        # 1 / (1 + 양수) 형태는 자연스럽게 (0, 1] 범위에 속하므로 별도 clipping 불필요
        # 다만, 계산 오류 등으로 음수가 나올 경우 대비하여 max(0.0, ...)는 유지 가능
        return max(0.0, score*20)

    def dependency_score_func_ver2(self, A_prev, A_curr, B_prev, B_curr):
        delta_A = A_curr - A_prev
        delta_B = B_curr - B_prev

        if self.relation_type == "positive":
            alpha = self.params.get("alpha", 40)
            score = np.exp(-alpha * (delta_A - delta_B)**2)

        elif self.relation_type == "inverse":
            beta = self.params.get("beta", 40)
            score = np.exp(-beta * (delta_A + delta_B)**2)

        elif self.relation_type == "threshold":
            T = self.params["T"]
            theta = self.params.get("theta", -0.1)
            gamma = self.params.get("gamma", 100)

            if A_curr > T:
                score = np.exp(-gamma * (delta_B - theta)**2)
            else:
                score = 0.0
        else:
            raise ValueError(f"존재하지 않는 Relation Type입니다: {self.relation_type}")

        # 항상 0 ~ 1 범위로 clipping
        return max(0.0, min(score, 1.0))






