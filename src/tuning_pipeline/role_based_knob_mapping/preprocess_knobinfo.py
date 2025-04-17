import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 수집한 knob info 데이터 불러오기
mysql_df = pd.read_csv("../../../data/knowledge/role_based_mapping/mysql_all_knobs.csv")[["knob_name", "type", "description"]]
postgresql_df = pd.read_csv("../../../data/knowledge/role_based_mapping/postgresql_all_knobs.csv")[["knob_name", "type", "description"]]

mysql_df["DBMS"] = "MySQL"
postgresql_df["DBMS"] = "PostgreSQL"

combined_df = pd.concat([mysql_df, postgresql_df], ignore_index=True)

# knob info의 description 기준으로 임베딩
model = SentenceTransformer("all-MiniLM-L6-v2")
descriptions = combined_df["description"].fillna("").tolist()
embeddings = model.encode(descriptions, show_progress_bar = True)

# 정규화 + 클러스터링
X_scaled = StandardScaler().fit_transform(embeddings)
dbscan = DBSCAN(eps=1.0, min_samples=5, metric='cosine')
combined_df["cluster"]=dbscan.fit_predict(X_scaled)

# 저장
combined_df.to_csv("knob_clustered_roles.csv", index = False)
print("✔ 클러스터링 완료! → 'knob_clustered_roles.csv' 파일로 저장됨.")