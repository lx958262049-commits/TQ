import numpy as np
import matplotlib.pyplot as plt
import itertools


from physics.ising import transverse_field_ising

from benchmark.exact_solver import exact_ground_state

from models.random_state import RandomWaveFunction

from vmc.metropolis import metropolis_sample

from vmc.energy import expectation_energy
from models.rbm import RBM



# =====================
# system
# =====================

N = 4

J = 1
h = 1


H = transverse_field_ising(
    N,
    J,
    h
)


E_exact,_ = exact_ground_state(H)



print(
    "Exact energy:",
    E_exact
)



# =====================
# basis
# =====================

basis=list(
    itertools.product(
        [1,-1],
        repeat=N
    )
)



# =====================
# wavefunction
# =====================

psi_model=RBM(
    N_visible=N,
    N_hidden=4
)


psi=psi_model.psi


# =====================
# Monte Carlo samples
# =====================


sample_list=[
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    100000
]


energies=[]



for ns in sample_list:


    samples=metropolis_sample(
        psi,
        N,
        ns,
        burn_in=500
    )


    E=expectation_energy(
        samples,
        H,
        psi,
        basis
    )


    energies.append(
        E.real
    )


    print(
        "Samples:",
        ns,
        "Energy:",
        E.real
    )



# =====================
# Save Data & Setup Path
# =====================
import os

# 动态获取当前脚本所在目录的上一级目录（即 TQS_myself 根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
results_dir = os.path.join(root_dir, "results")

# 确保 results 文件夹存在，如果不存在则自动创建
os.makedirs(results_dir, exist_ok=True)

# 1. 保存数据为 CSV 文件
csv_path = os.path.join(results_dir, "mc_convergence_data.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("num_samples,energy\n")  # 写入表头
    for ns, eng in zip(sample_list, energies):
        f.write(f"{ns},{eng}\n")

print(f"\n[Data Saved]: {csv_path}")

# =====================
# plot
# =====================

plt.figure(figsize=(7,5))

plt.plot(
    sample_list,
    energies,
    marker="o",
    label="VMC"
)

plt.axhline(
    E_exact,
    linestyle="--",
    color="red",  # 显式加个颜色区分
    label="Exact"
)

plt.xscale("log")

plt.xlabel(
    "Number of Monte Carlo samples"
)

plt.ylabel(
    "Energy"
)

plt.title(
    "Monte Carlo convergence of VMC"
)

plt.legend()
plt.grid()

# 2. 保存图片到 results 目录下
img_path = os.path.join(results_dir, "mc_convergence_plot.png")
plt.savefig(img_path, dpi=300, bbox_inches='tight')
print(f"[Plot Saved]: {img_path}")

plt.show()
