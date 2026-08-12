import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入你之前写好的哈密顿量构建和精确求解函数
from physics.ising import transverse_field_ising
from benchmark.exact_solver import exact_ground_state


def run_experiment():
    # 1. 定义实验参数
    N_list = [4, 6, 8, 10]
    J = 1.0
    h = 1.0

    results = []

    print("开始运行有限尺寸能级实验...")
    for N in N_list:
        # 构建哈密顿量
        H = transverse_field_ising(N=N, J=J, h=h)
        # 求解基态能量
        E_0, _ = exact_ground_state(H)
        # 计算每格点平均能量 E_0 / N
        E_per_site = E_0 / N

        print(f"N = {N:2d} | 基态总能量 = {E_0:.6f} | 每格点平均能量 = {E_per_site:.6f}")

        results.append({
            "N": N,
            "Ground_Energy": E_0,
            "Energy_per_Site": E_per_site
        })

    # 2. 转换成 Pandas DataFrame 方便处理和保存
    df = pd.DataFrame(results)

    # 确保 results 文件夹存在（根据你项目根目录的相对路径）
    # 这里用 os.path 确保无论在哪个目录下运行都能准确找到 results 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 保存数据到 results/ising_fss_data.csv
    csv_path = os.path.join(results_dir, "ising_fss_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[成功] 数据已保存至: {csv_path}")

    # 3. 开始画科研级图表
    plt.figure(figsize=(7, 5), dpi=300)  # 高清分辨率，适合放进论文或PPT

    # 画带有点和线的折线图
    plt.plot(df["N"], df["Energy_per_Site"],
             marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=8, label='Exact ED')

    # 润色图表
    plt.title("Finite-Size Scaling of 1D TFIM ($J=1.0, h=1.0$)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("System Size $N$", fontsize=12)
    plt.ylabel("Ground State Energy per Site $E_0 / N$", fontsize=12)

    # 设置横坐标刻度正好对应我们的 N
    plt.xticks(N_list)

    # 开启美观的细网格线
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # 保存图片到 results/ising_fss_plot.png
    plot_path = os.path.join(results_dir, "ising_fss_plot.png")
    plt.savefig(plot_path)
    print(f"[成功] 图表已保存至: {plot_path}")

    # 如果想在屏幕上直接弹出来看，可以加上这行：
    # plt.show()


if __name__ == "__main__":
    run_experiment()