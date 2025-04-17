import numpy as np
import pandas as pd

class DependencyScore:
    def __init__(self, relation_type, **params):
        self.relation_type = relation_type
        self.params = params
    def dependency_score_func(self, A_prev, A_curr, B_prev, B_curr):
        delta_A = A_curr - A_prev
        delta_B = B_curr - B_prev

        if self.relation_type == "positive":
            alpha = self.params.get("alpha", 50)
            return np.exp(-alpha * (delta_A-delta_B)**2)

        elif self.relation_type == "inverse":
            beta = self.params.get("beta", 50)
            return np.exp(-beta * (delta_A+delta_B)**2)

        elif self.relation_type == "threshold":
            T = self.params["T"]
            theta = self.params.get("theta", -0.1)
            gamma = self.params.get("gamma", 100)

            if A_curr > T:
                return np.exp(-gamma * (delta_B - theta)**2)
            else:
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



