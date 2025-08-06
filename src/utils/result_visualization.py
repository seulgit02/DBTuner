import matplotlib.pyplot as plt
import numpy as np

# 데이터
models = ['DBTune', 'SMAC', 'Vanilla BO']
tps = [4967.81, 4241.92, 2921.47]
latency = [825, 990, 1127]

x = np.arange(len(models))*1.5

# 색상 설정
colors_tps = ['lightcoral', 'skyblue', 'lightsteelblue']  # VBO를 더 연한 하늘색
colors_latency = ['lightcoral', 'skyblue', 'lightsteelblue']

# ---------- TPS 그래프 ----------
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, tps, color=colors_tps,width=1)

ax.set_ylabel('TPS (↑)')
ax.set_title('TPS Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)

# y축 gridline
ax.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

# 각 막대 위에 수치 표시
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig('TPS_comparison.png', dpi=300, bbox_inches='tight')

# ---------- Latency 그래프 ----------
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, latency, color=colors_latency, width=1)

ax.set_ylabel('Latency (↓)')
ax.set_title('Latency Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)

# y축 gridline
ax.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

# 각 막대 위에 수치 표시
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# 이미지 저장 (dpi=300 → 고해상도)
plt.savefig('Latency_comparison.png', dpi=300, bbox_inches='tight')


