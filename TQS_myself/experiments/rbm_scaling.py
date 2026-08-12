import numpy as np

from models.rbm import RBM
from physics.ising import transverse_field_ising
from vmc.energy import get_local_energies_tfim, expectation_energy_tfim
from vmc.metropolis import metropolis_sample
from vmc.gradient import vmc_gradient
from vmc.optimizer import Adam

# 1. 系统与哈密顿量设置
N = 10
J_coupling = 1.0
h_field = 1.0  # 横场强度

# H 只用于以后需要精确对角化对比时（本脚本目前没用到，
# 稀疏局域能量走的是解析展开，不依赖 H 矩阵/basis 全枚举）
H = transverse_field_ising(N, J_coupling, h_field)

hidden_sizes = [4, 8, 16, 32, 64]
epochs = 300
results = []

# 2. 遍历不同的隐层神经元数量 (Nh)
for Nh in hidden_sizes:
    print(f"\n--- Training RBM with hidden_size (Nh): {Nh} ---")

    # 实例化模型和优化器
    np.random.seed(42)
    rbm = RBM(N_visible=N, N_hidden=Nh)
    optimizer = Adam(
        rbm,
        lr=0.001
    )

    # 3. VMC 优化循环（必须缩进在 Nh 循环内部）
    for epoch in range(epochs):
        # Metropolis 采样
        samples = metropolis_sample(rbm.psi, N, 5000)

        # 计算局部能量（稀疏版，O(N) 每样本，而不是 O(2^N)）
        local_E = get_local_energies_tfim(
            samples, rbm.psi, J=J_coupling, h=h_field
        )

        # 计算变分梯度
        grads = vmc_gradient(rbm, samples, local_E)

        # 参数更新
        optimizer.step(grads)

        # 打印训练进度：直接复用上面已经算好的 local_E，
        # 不再对同一批 samples 重复调用一次昂贵的能量计算
        if epoch % 50 == 0:
            E = np.mean(local_E.real)
            print(f"Epoch {epoch:3d} | Energy: {E:.6f}")

    # 最终计算并记录该 Nh 下的最终能量
    final_samples = metropolis_sample(
        rbm.psi,
        N,
        20000
    )

    final_E = expectation_energy_tfim(
        final_samples,
        rbm.psi,
        J=J_coupling,
        h=h_field
    )
    results.append([Nh, final_E])
    print(f"Finished Nh={Nh}, Final Energy: {final_E:.6f}")

np.savetxt("../results/rbm_scaling_Adam.txt", results, header="Nh Energy")
print("\nResults successfully saved to rbm_scaling_Adam.txt!")