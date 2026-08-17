"""
VQE Scaling Law 扫描脚本
========================
实验设计：
  - N (qubit数):   4, 6, 8
  - n_layers:      2, 4, 6, 8, 10, 12
  - J2/J1:         0.0, 0.25
  - seeds:         每个配置跑 N_SEEDS 个, 取最好2个的均值作为该点的V-score

计算量f的定义 (类比论文的FLOPs):
  f = n_gates × n_steps
  n_gates = n_layers × n_qubits × 3   (每层: n_qubits个RY + n_qubits个CNOT + n_qubits个RY)

输出：
  - raw_results.csv   : 每一个 (N, layers, J2/J1, seed) 的原始结果
  - agg_results.csv   : 每个配置取 best-2 均值后的聚合结果 (含f), 用于画图
"""

import csv
import os
import time
import pennylane as qml
from pennylane import numpy as np

# =========================================================
# 0. 实验参数配置 (只需改这里)
# =========================================================
N_LIST       = [4, 6]        # qubit数 (N=8需要更深的电路才能收敛, 单独作为阶段2实验)
LAYERS_LIST  = [2, 4, 6, 8, 10, 12]  # 层数
J2_LIST      = [0.0, 0.25]      # 阻挫程度
N_SEEDS      = 6                # 每个配置跑几个seed (10个太耗时, 6个仍能可靠取best-2均值)
N_STEPS      = None              # 每次VQE优化步数，由 n_layers 决定
STEPSIZE     = 0.1              # Adam学习率
RAW_CSV      = "raw_results3.csv"
AGG_CSV      = "agg_results3.csv"

# =========================================================
# 1. Hamiltonian
# =========================================================
def build_J1J2_hamiltonian(n_qubits, J1=1.0, J2=0.0):
    coeffs, ops = [], []
    def add_bond(i, j, J):
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            coeffs.append(0.25 * J)
            ops.append(P(i) @ P(j))
    for i in range(n_qubits):
        add_bond(i, (i + 1) % n_qubits, J1)
    if abs(J2) > 1e-12:
        for i in range(n_qubits):
            add_bond(i, (i + 2) % n_qubits, J2)
    return qml.Hamiltonian(coeffs, ops)


# =========================================================
# 2. Ansatz: Hardware-efficient (RY + CNOT环)
# =========================================================
def hea_ansatz(params, n_qubits, n_layers):
    """
    每层结构: RY(每qubit) -> CNOT环 -> RY(每qubit)
    params shape: (n_layers, n_qubits, 2)
    """
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(params[l, q, 0], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
        for q in range(n_qubits):
            qml.RY(params[l, q, 1], wires=q)


# =========================================================
# 3. 计算量 f 的定义
# =========================================================
def compute_flops(n_qubits, n_layers, n_steps):
    """
    f = n_gates_per_step × n_steps
    n_gates_per_step = n_layers × (n_qubits RY + n_qubits CNOT + n_qubits RY)
                     = n_layers × 3 × n_qubits
    """
    n_gates = n_layers * 3 * n_qubits
    return n_gates * n_steps


# =========================================================
# 4. 单次 VQE
# =========================================================
def run_vqe(n_qubits, n_layers, J2_over_J1, seed, n_steps, H, H_matrix, E_exact):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit_energy(params):
        hea_ansatz(params, n_qubits, n_layers)
        return qml.expval(H)

    @qml.qnode(dev)
    def circuit_state(params):
        hea_ansatz(params, n_qubits, n_layers)
        return qml.state()

    np.random.seed(seed)
    params = np.random.uniform(
        0, 2 * np.pi,
        size=(n_layers, n_qubits, 2),
        requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=STEPSIZE)
    for _ in range(n_steps):
        params, _ = opt.step_and_cost(circuit_energy, params)

    # 用态向量精确计算 <H> 和 Var(H) -> V-score
    psi  = circuit_state(params)
    H_psi   = H_matrix @ psi
    E_final = float(np.real(np.vdot(psi, H_psi)))
    H2      = float(np.real(np.vdot(H_psi, H_psi)))
    var_H   = H2 - E_final ** 2
    V_score = n_qubits * var_H / (E_final ** 2)

    return {
        "E_final"  : E_final,
        "abs_error": abs(E_final - E_exact),
        "V_score"  : V_score,
    }


# =========================================================
# 5. 主扫描循环
# =========================================================
def main():
    # --- 初始化 CSV ---
    raw_fields = ["n_qubits", "n_layers", "J2_over_J1", "seed",
                  "f", "E_exact", "E_final", "abs_error", "V_score"]
    agg_fields = ["n_qubits", "n_layers", "J2_over_J1",
                  "f", "E_exact", "V_score_best2mean", "V_score_min"]

    raw_file = open(RAW_CSV, "w", newline="")
    agg_file = open(AGG_CSV, "w", newline="")
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
    agg_writer = csv.DictWriter(agg_file, fieldnames=agg_fields)
    raw_writer.writeheader()
    agg_writer.writeheader()

    total = len(N_LIST) * len(LAYERS_LIST) * len(J2_LIST)
    done  = 0
    t0    = time.time()

    for n_qubits in N_LIST:
        for J2_over_J1 in J2_LIST:

            # exact diag (每个 (N, J2) 只算一次)
            H        = build_J1J2_hamiltonian(n_qubits, J2=J2_over_J1)
            H_matrix = qml.matrix(H, wire_order=range(n_qubits))
            E_exact  = float(np.linalg.eigvalsh(H_matrix)[0])

            for n_layers in LAYERS_LIST:
                done += 1
                n_steps = 200 + 50 * n_layers
                f = compute_flops(n_qubits, n_layers, n_steps)

                print(f"\n[{done}/{total}] N={n_qubits}, layers={n_layers}, "
                      f"J2/J1={J2_over_J1}, f={f:.2e}, E_exact={E_exact:.4f}")

                v_scores = []
                for seed in range(N_SEEDS):
                    t1 = time.time()
                    res = run_vqe(n_qubits, n_layers, J2_over_J1,
                                  seed, n_steps, H, H_matrix, E_exact)
                    elapsed = time.time() - t1

                    v_scores.append(res["V_score"])

                    # 写原始结果
                    raw_writer.writerow({
                        "n_qubits"   : n_qubits,
                        "n_layers"   : n_layers,
                        "J2_over_J1" : J2_over_J1,
                        "seed"       : seed,
                        "f"          : f,
                        "E_exact"    : E_exact,
                        "E_final"    : res["E_final"],
                        "abs_error"  : res["abs_error"],
                        "V_score"    : res["V_score"],
                    })
                    raw_file.flush()

                    print(f"  seed={seed}: V-score={res['V_score']:.3e}  "
                          f"abs_err={res['abs_error']:.3e}  ({elapsed:.1f}s)")

                # 聚合: best-2 均值
                v_sorted       = sorted(v_scores)
                best2_mean     = (v_sorted[0] + v_sorted[1]) / 2
                best1          = v_sorted[0]

                agg_writer.writerow({
                    "n_qubits"        : n_qubits,
                    "n_layers"        : n_layers,
                    "J2_over_J1"      : J2_over_J1,
                    "f"               : f,
                    "E_exact"         : E_exact,
                    "V_score_best2mean": best2_mean,
                    "V_score_min"     : best1,
                })
                agg_file.flush()

                print(f"  --> best-2 mean V-score = {best2_mean:.3e}  "
                      f"min = {best1:.3e}")

    raw_file.close()
    agg_file.close()
    print(f"\n完成! 总用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"原始数据: {RAW_CSV}")
    print(f"聚合数据: {AGG_CSV}")


if __name__ == "__main__":
    main()