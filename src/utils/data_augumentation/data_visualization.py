import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def visualize_knob_effects(df, dbms_name, workload_name, output_dir, metric1, metric2=None):
    metric_cols = [metric1]
    if metric2:
        metric_cols.apend(metric2)
    knob_cols = [col for col in df.columns if col not in metric_cols and not col.startswith("Unnamed")]
    knob_df = df[knob_cols]

    # 스케일링
    scaler = StandardScaler()
    scaled_knob = scaler.fit_transform(knob_df)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_knob)

    # 시각화
    n_plots = 1 if metric2 is None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"{dbms_name.upper()} - {workload_name.upper()} (CTGAN PCA Projection)", fontsize=14)

    for idx, metric in enumerate([metric1] if metric2 is None else [metric1, metric2]):
        axes[idx].scatter(pca_result[:, 0], pca_result[:, 1],
                          c=df[metric], cmap='viridis', alpha=0.6)
        axes[idx].set_title(f"{metric} color")
        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")
        cbar = plt.colorbar(axes[idx].collections[0], ax=axes[idx])
        cbar.set_label(metric)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 저장 경로 생성
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"[{dbms_name}] {workload_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    mysql_root_dir = "../../../data/workloads/mysql/ctgan_data"
    postgresql_root_dir = "../../../data/workloads/postgresql/ctgan_data"
    output_root = "../../data/workloads/ctgan_visualizations"
    root_dirs = [postgresql_root_dir]
    for target_dir in root_dirs:
        for file_name in os.listdir(target_dir):
            file_path = f"{target_dir}/{file_name}"
            if not file_name.endswith(".csv"):
                continue

            print(f"\n📦 Processing file: {file_name}")

            workload_name = os.path.splitext(file_name)[0].replace("_result", "")


            try:
                df = pd.read_csv(file_path)
                if target_dir == mysql_root_dir:
                    visualize_knob_effects(
                        df,
                        dbms_name="MySQL",
                        workload_name=workload_name,
                        output_dir=output_root,
                        metric1 = "tps",
                        metric2 = "Latency"
                    )
                else:
                    visualize_knob_effects(
                        df,
                        dbms_name="PostgreSQL",
                        workload_name=workload_name,
                        output_dir=output_root,
                        metric1="result"
                    )

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
