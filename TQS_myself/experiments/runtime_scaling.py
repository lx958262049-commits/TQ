import itertools
import time
import matplotlib.pyplot as plt
import numpy as np

from models.rbm import RBM
from physics.ising import transverse_field_ising
from vmc.energy import expectation_energy, expectation_energy_tfim
from vmc.metropolis import metropolis_sample

N_list = [4, 6, 8, 10, 12, 14, 16]

dense_time = []
sparse_time = []

for N in N_list:
    print("N =", N)

    H = transverse_field_ising(N, 1, 1)

    basis = list(itertools.product([-1, 1], repeat=N))

    rbm = RBM(N, 8)

    samples = metropolis_sample(rbm.psi, N, 5000)

    # ----------------
    # dense
    # ----------------
    t0 = time.time()

    E1 = expectation_energy(samples, H, rbm.psi, basis)

    t1 = time.time()

    dense_time.append(t1 - t0)

    # ----------------
    # sparse
    # ----------------
    t0 = time.time()

    E2 = expectation_energy_tfim(samples, rbm.psi, 1, 1)

    t1 = time.time()

    sparse_time.append(t1 - t0)

    print("dense:", dense_time[-1], "sparse:", sparse_time[-1])


# ---------------------------------------------------------
# 1. 保存数据到 ../results/ 路径
# ---------------------------------------------------------
csv_path = "../results/runtime_scaling.csv"

np.savetxt(
    csv_path,
    np.column_stack((N_list, dense_time, sparse_time)),
    delimiter=",",
    header="N,dense,sparse",
    comments="",  # 移除默认的 # 号，方便后续读取
)
print(f"\n数据已保存至: {csv_path}")


# ---------------------------------------------------------
# 2. 绘制并保存计算成本对比图 (Computational cost comparison)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# 根据要求：蓝色代表 Dense，橙色代表 Sparse
plt.plot(
    N_list,
    dense_time,
    "o-",
    color="blue",
    label="Dense Hamiltonian",
    linewidth=2,
    markersize=6,
)
plt.plot(
    N_list,
    sparse_time,
    "s-",
    color="orange",
    label="Sparse TFIM evaluation",
    linewidth=2,
    markersize=6,
)

# 设置坐标轴与标题
plt.xlabel("System size N", fontsize=12)
plt.ylabel("Runtime(log(s))", fontsize=12)
plt.title("Computational cost comparison", fontsize=14)

# 开启对数坐标轴，直观展现指数增长 (Dense) 与近乎线性增长 (Sparse) 的巨大差异
plt.yscale("log")

# 细节美化
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

# 保存图片到 ../results/ 路径
img_path = "../results/computational_cost_comparison.png"
plt.savefig(img_path, dpi=300)
plt.show()

print(f"图片已保存至: {img_path}")