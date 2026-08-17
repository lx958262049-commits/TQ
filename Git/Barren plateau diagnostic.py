"""
贫瘠高原 (Barren Plateau) 诊断脚本
====================================
目的: 检验 HVA 卡在初始 Neel 态附近这个现象, 是否是贫瘠高原
(梯度方差随 N 指数衰减) 造成的, 并与 HEA 做对照。

标准诊断方法 (McClean et al. 2018):
  固定电路深度(层数), 对每个 N, 随机采样 M_SAMPLES 组参数,
  每次只算一次梯度(不优化), 取同一个固定参数分量的梯度值,
  统计这些梯度值在多次随机采样下的方差 Var(∂E/∂θ_0)。
  如果 Var 随 N 指数衰减(log-linear下降), 就是贫瘠高原的标志。

用法: 直接跑, 会输出:
  - barren_plateau_raw.csv   : 每次采样的梯度原始值
  - barren_plateau_summary.csv: 每个(ansatz, N, J2)的梯度方差汇总
  - barren_plateau.png       : log(方差) vs N 的对比图

依赖:
  pip install tensorcircuit jax jaxlib matplotlib pandas --break-system-packages
"""

import csv
import time
import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

# =========================================================
# 0. 实验参数配置
# =========================================================
ANSATZ_LIST  = ["hea", "hva"]
N_LIST       = [4, 6, 8, 10, 12]   # 不需要精确对角化, 可以跑得比VQE实验更大
LAYERS_FIXED = 4                    # 固定层数(贫瘠高原诊断标准做法: 固定深度看N效应)
J2_LIST      = [0.0, 0.5]           # 无阻挫 vs MG点
M_SAMPLES    = 200                  # 每个配置采样多少组随机参数
RAW_CSV      = "barren_plateau_raw.csv"
SUMMARY_CSV  = "barren_plateau_summary.csv"
PLOT_PNG     = "barren_plateau.png"


# =========================================================
# 1. Hamiltonian bonds (跟主实验一致)
# =========================================================
def build_J1J2_bonds(n_qubits, J1=1.0, J2=0.0):
    bonds = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        bonds.append((i, j, 0.25 * J1))
    if abs(J2) > 1e-12:
        for i in range(n_qubits):
            j = (i + 2) % n_qubits
            bonds.append((i, j, 0.25 * J2))
    return bonds


# =========================================================
# 2. Ansatz (跟主实验完全一致, 复用同样的电路结构)
# =========================================================
def build_circuit_hea(params, n_qubits, n_layers, bonds=None):
    c = tc.Circuit(n_qubits)
    for l in range(n_layers):
        for q in range(n_qubits):
            c.ry(q, theta=params[l, q, 0])
        for q in range(n_qubits):
            c.cnot(q, (q + 1) % n_qubits)
        for q in range(n_qubits):
            c.ry(q, theta=params[l, q, 1])
    return c


def build_circuit_hva(params, n_qubits, n_layers, bonds):
    c = tc.Circuit(n_qubits)
    for q in range(n_qubits):
        if q % 2 == 1:
            c.x(q)
    for l in range(n_layers):
        for bond_idx, (i, j, _) in enumerate(bonds):
            theta = params[l, bond_idx]
            c.exp1(i, j, theta=theta, unitary=tc.gates._xx_matrix)
            c.exp1(i, j, theta=theta, unitary=tc.gates._yy_matrix)
            c.exp1(i, j, theta=theta, unitary=tc.gates._zz_matrix)
    return c


ANSATZ_REGISTRY = {
    "hea": {
        "build_circuit": build_circuit_hea,
        "param_shape": lambda n_qubits, n_layers, bonds: (n_layers, n_qubits, 2),
        # 取第一个参数分量作为"固定参考分量"的索引
        "ref_index": (0, 0, 0),
    },
    "hva": {
        "build_circuit": build_circuit_hva,
        "param_shape": lambda n_qubits, n_layers, bonds: (n_layers, len(bonds)),
        "ref_index": (0, 0),
    },
}


def make_energy_fn(ansatz, n_qubits, n_layers, bonds):
    build_circuit = ANSATZ_REGISTRY[ansatz]["build_circuit"]

    def energy(params):
        c = build_circuit(params, n_qubits, n_layers, bonds)
        e = 0.0
        for (i, j, coeff) in bonds:
            e = e + coeff * c.expectation_ps(x=[i, j])
            e = e + coeff * c.expectation_ps(y=[i, j])
            e = e + coeff * c.expectation_ps(z=[i, j])
        return tc.backend.real(e)

    return energy


# =========================================================
# 3. 单点梯度采样
# =========================================================
def sample_gradients(ansatz, n_qubits, n_layers, bonds, n_samples, seed0=0):
    """
    对给定配置, 随机采样 n_samples 组参数, 每组只算一次梯度(不优化)。
    返回: 固定参考分量的梯度值列表, 以及整体梯度范数列表(附加信息)。
    """
    energy_fn = make_energy_fn(ansatz, n_qubits, n_layers, bonds)
    grad_fn = tc.backend.jit(tc.backend.grad(energy_fn))

    param_shape = ANSATZ_REGISTRY[ansatz]["param_shape"](n_qubits, n_layers, bonds)
    ref_index = ANSATZ_REGISTRY[ansatz]["ref_index"]

    ref_grads = []
    grad_norms = []

    for s in range(n_samples):
        key = jax.random.PRNGKey(seed0 * 100000 + s)
        params = jax.random.uniform(
            key, shape=param_shape, minval=0.0, maxval=2 * jnp.pi
        )
        grads = grad_fn(params)
        g = np.array(grads)

        ref_grads.append(float(g[ref_index]))
        grad_norms.append(float(np.linalg.norm(g)))

    return np.array(ref_grads), np.array(grad_norms)


# =========================================================
# 4. 主循环
# =========================================================
def main():
    raw_fields = ["ansatz", "n_qubits", "J2_over_J1", "sample_idx", "ref_grad"]
    summary_fields = ["ansatz", "n_qubits", "J2_over_J1",
                       "var_ref_grad", "mean_abs_ref_grad", "mean_grad_norm"]

    raw_file = open(RAW_CSV, "w", newline="")
    summary_file = open(SUMMARY_CSV, "w", newline="")
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
    raw_writer.writeheader()
    summary_writer.writeheader()

    total = len(ANSATZ_LIST) * len(N_LIST) * len(J2_LIST)
    done = 0
    t0 = time.time()

    for ansatz in ANSATZ_LIST:
        for n_qubits in N_LIST:
            for J2_over_J1 in J2_LIST:
                done += 1
                bonds = build_J1J2_bonds(n_qubits, J2=J2_over_J1)

                print(f"[{done}/{total}] ansatz={ansatz}, N={n_qubits}, "
                      f"J2/J1={J2_over_J1}, layers={LAYERS_FIXED} ...", end=" ")
                t1 = time.time()

                ref_grads, grad_norms = sample_gradients(
                    ansatz, n_qubits, LAYERS_FIXED, bonds, M_SAMPLES
                )

                for idx, g in enumerate(ref_grads):
                    raw_writer.writerow({
                        "ansatz": ansatz,
                        "n_qubits": n_qubits,
                        "J2_over_J1": J2_over_J1,
                        "sample_idx": idx,
                        "ref_grad": g,
                    })
                raw_file.flush()

                var_ref = float(np.var(ref_grads))
                mean_abs_ref = float(np.mean(np.abs(ref_grads)))
                mean_norm = float(np.mean(grad_norms))

                summary_writer.writerow({
                    "ansatz": ansatz,
                    "n_qubits": n_qubits,
                    "J2_over_J1": J2_over_J1,
                    "var_ref_grad": var_ref,
                    "mean_abs_ref_grad": mean_abs_ref,
                    "mean_grad_norm": mean_norm,
                })
                summary_file.flush()

                print(f"Var(∂E/∂θ_0)={var_ref:.3e}  "
                      f"mean|grad_norm|={mean_norm:.3e}  ({time.time()-t1:.1f}s)")

    raw_file.close()
    summary_file.close()
    print(f"\n完成! 总用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"原始梯度数据: {RAW_CSV}")
    print(f"方差汇总: {SUMMARY_CSV}")


# =========================================================
# 5. 画图: log(Var梯度) vs N, 按 ansatz/J2 分组
#    贫瘠高原的标志: 曲线随N近似直线下降(半对数图上)
# =========================================================
def plot_barren_plateau(summary_csv=SUMMARY_CSV, out_png=PLOT_PNG):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    styles = {
        ("hea", 0.0): dict(color="#1f77b4", ls="-", marker="o", label="HEA, J2/J1=0.0"),
        ("hea", 0.5): dict(color="#d62728", ls="-", marker="o", label="HEA, J2/J1=0.5 (MG)"),
        ("hva", 0.0): dict(color="#1f77b4", ls="--", marker="s", label="HVA, J2/J1=0.0"),
        ("hva", 0.5): dict(color="#d62728", ls="--", marker="s", label="HVA, J2/J1=0.5 (MG)"),
    }

    for (ansatz, j2), grp in df.groupby(["ansatz", "J2_over_J1"]):
        grp = grp.sort_values("n_qubits")
        style = styles.get((ansatz, j2), {})
        ax.plot(grp["n_qubits"], grp["var_ref_grad"], **style)

    ax.set_yscale("log")
    ax.set_xlabel("N (qubits)")
    ax.set_ylabel(r"Var($\partial E/\partial\theta_0$) over random inits")
    ax.set_title(f"Barren plateau diagnostic (fixed layers={LAYERS_FIXED})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"图已保存: {out_png}")


if __name__ == "__main__":
    main()
    plot_barren_plateau()