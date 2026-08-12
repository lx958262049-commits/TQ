from models.rbm import RBM

from vmc.metropolis import metropolis_sample

from physics.ising import transverse_field_ising


from benchmark.exact_solver import exact_ground_state

from vmc.gradient import vmc_gradient
from vmc.optimizer import SGD


# 稀疏版局域能量：不再需要 H 矩阵 / basis 全枚举，
# 每个样本只需 O(N) 次 psi 求值（推导见 vmc/energy.py 注释）
from vmc.energy import get_local_energies_tfim, expectation_energy_tfim

import numpy as np

N=10
J=1.0
h=1.0

# H 只用于精确对角化基准对比（一次性开销，不在训练循环里）
H=transverse_field_ising(
    N,
    J,
    h
)


E_exact,_=exact_ground_state(H)


print(
    "Exact:",
    E_exact
)



rbm=RBM(
    N_visible=N,
    N_hidden=20
)

optimizer = SGD(
    rbm,
    lr=0.01
)

samples=metropolis_sample(
    rbm.psi,
    N,
    50000
)


E=expectation_energy_tfim(
    samples,
    rbm.psi,
    J=J,
    h=h
)


print(
    "RBM initial energy:",
    E
)
energy_history = []

best_energy = 999
best_epoch = 0


for epoch in range(500):


    # sample configurations

    samples=metropolis_sample(
        rbm.psi,
        N,
        50000
    )


    # local energy (稀疏版，O(N) 每样本)

    local_E=get_local_energies_tfim(
        samples,
        rbm.psi,
        J=J,
        h=h
    )


    E = np.mean(local_E.real)

    energy_history.append(E)

    if E < best_energy:
        best_energy = E
        best_epoch = epoch

    print(
        epoch,
        E
    )


    # gradient

    gradients = vmc_gradient(
        rbm,
        samples,
        local_E
    )

    optimizer.step(
        gradients
    )

print(
    "Best energy:",
    best_energy,
    "at epoch:",
    best_epoch
)

import os
import matplotlib.pyplot as plt

# 1. 自动定位到当前项目根目录下的 results 文件夹
save_dir = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(save_dir, exist_ok=True) # 如果 results 文件夹存在就不报错，不存在就自动创建

# 2. 保存数值数据 (数据持久化，导师超喜欢看这个)
data = np.column_stack(
    (
        np.arange(len(energy_history)),
        energy_history
    )
)


np.savetxt(
    os.path.join(save_dir,"rbm_energy_data_2.csv"),
    data,
    delimiter=",",
    header="epoch,energy"
)

# 3. 画图并保存图片到 results 文件夹
plt.figure(figsize=(6, 4))
plt.plot(energy_history, label="RBM-VMC Energy")
plt.axhline(E_exact, color="r", linestyle="--", label=f"Exact ({E_exact:.4f})")

plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("RBM-VMC Energy Convergence")
plt.grid(True, alpha=0.3)
plt.legend()

# 🎯 保存图片到 ../results 目录
save_path = os.path.join(save_dir, "rbm_energy_convergence_2.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"数据与图片已成功保存至: {save_path}")