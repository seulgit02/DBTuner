from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
'''
통계적 안정성을 위해 top_k개의 유사 workload를 기반으로 knob_selection을 진행할 예정
similarity <0.7일 경우, top-k 안에 들어도 knob selection에 반영 안함.
'''

def find_similarity_workloads(current_vector, historical_vectors, historical_ids = None, top_k = 5, threshold=0.8):
    current_vector = np.array(current_vector).reshape(1, -1)
    historical_vectors = np.array(historical_vectors)

    similarities = cosine_similarity(current_vector, historical_vectors)[0]
    print("[코사인 유사도]: ", similarities)
    top_indices = np.argsort(similarities)[::-1] # 유사도 내림차순 정렬
    max_sim = similarities[top_indices[0]]

    selected = [(i, similarities[i]) for i in top_indices[:top_k] if similarities[i]>=threshold]

    if not selected:
        print(f"⚠️ 유사도 {threshold} 이상의 historical workload가 없음. 가장 높은 유사도 = {max_sim:.3f}")
        print("top-1 workload만을 Knob Selection에 이용합니다.(정확도가 낮을 수 있습니다.)\n")
        selected = [(top_indices[0], similarities[top_indices[0]])]
    # 사용할 historical workloads 지정해서 쓸 거면 사용
    if historical_ids:
        return [(historical_ids[i], sim) for i, sim in selected]
    else:
        return selected

def majority_vote_knobs(current_vector, historical_vectors, knob_lists, knob_nums=3, top_k=5, threshold=0.7):
    """
       current_vector와 유사한 top_k historical workload 중
       similarity ≥ threshold(=default는 0.7)인 것만 고려해서 knob selection 진행.
       similarity >= threshold인 historical workloads 없으면 top-1만 가져와서 knob selection 진행.
    """
    similar_workloads = find_similarity_workloads(current_vector, historical_vectors, top_k=5, threshold=0.7)
    top_indiecs = [idx for idx, _ in similar_workloads]

    key_knobs = []
    for i in top_indiecs:
        key_knobs.extend(knob_lists[i])

    knob_counts = Counter(key_knobs)
    sorted_knobs = [knob for knob, _ in knob_counts.most_common()]
    return sorted_knobs[:knob_nums]


if __name__ == "__main__":
    # test용 knoblist 랜덤생성
    current_vector = [0.1] * 19
    historical_vectors = np.random.rand(10, 19)
    print("historical_vectors: ", historical_vectors)
    knob_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    knob_lists = [random.sample(knob_names, 3) for _ in range(10)]
    print("knob_lists: ", knob_lists)


    major_knobs = majority_vote_knobs(
        current_vector=current_vector,
        historical_vectors=historical_vectors,
        knob_lists=knob_lists,
        top_k=5,
        threshold=0.7
    )
    print("최종 선택된 knob 리스트:", major_knobs)



