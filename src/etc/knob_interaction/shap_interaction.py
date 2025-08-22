'''
Compute SHAP Interaction Values with xgboost
'''
import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns

import shap

# example
data_path = "../../../data/workloads/mysql/original_data/preprocess/knob_filter/MYSQL_YCSB_AA_FILTERED.csv"

df = pd.read_csv(data_path)

X = df.drop(columns=["tps", "latency"]).values
y = df["tps"].values / df["latency"].values


# train a model with single tree
Xd = xgboost.DMatrix(X, label=y) # (참조) y -> tps/latency로 수정할 예정
model = xgboost.train({"eta": 1, "max_depth": 4, "base_score": 0, "lambda": 0}, Xd, 1)
print("Model error =", np.linalg.norm(y - model.predict(Xd)))
print(model.get_dump(with_stats=True)[0])


# shap values
pred = model.predict(Xd, output_margin = True)
explainer = shap.TreeExplainer(model)
explanation = explainer(Xd)

shap_values = explanation.values
np.abs(shap_values.sum(axis=1)+explanation.base_values -pred).max()

# result visualization
shap.plots.beeswarm(explanation, show=False)
plt.tight_layout()
plt.savefig("results/beeswarm_interactions.png", dpi=200)



# shap interaction values
shap_interaction_values = explainer.shap_interaction_values(Xd)
shap_interaction_values[0].round(2)

np.abs(shap_interaction_values.sum((1, 2)) +explainer.expected_value - pred).max()


# shap_interaction_values: (n_samples, n_features, n_features)
interaction_strength = np.abs(shap_interaction_values).mean(axis=0)

# feature 이름 가져오기(Dataframe에서)
feature_names = df.drop(columns=["tps", "latency"]).columns

plt.figure(figsize=(8, 6))
sns.heatmap(interaction_strength, xticklabels=feature_names, yticklabels=feature_names, cmap="viridis", annot=False)
plt.title("Mean |SHAP interaction|")
plt.tight_layout()
plt.savefig("results/shap_interaction_heatmap.png", dpi=200)
plt.close()

interaction_df = pd.Dataframe(interaction_strength, index=feature_names, columns=feature_names)
print(interaction_df.round(4))
interaction_df.to_csv("results/shap_interaction_matrix.csv")