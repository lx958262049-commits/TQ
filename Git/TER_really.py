"""
VQE Scaling Law 扫描脚本 (TensorCircuit + JAX 版本)
====================================================
与 PennyLane 版本逻辑完全对齐, 只是后端换成 TensorCircuit + JAX,
利用 JIT 编译大幅提速 (尤其在重复跑大量小电路优化任务时).

实验设计：
  - N (qubit数):   4, 6   (8留作阶段2, 需要更深电路)
  - n_layers:      2, 4, 6, 8, 10, 12
  - J2/J1:         0.0, 0.25
  - seeds:         每个配置跑 N_SEEDS 个, 取最好2个的均值作为该点的V-score

计算量f的定义 (类比论文FLOPs):
  f = n_gates × n_steps
  n_gates = n_layers × n_qubits × 3   (每层: n_qubits个RY + n_qubits个CNOT + n_qubits个RY)

V-score 定义:
  V-score = N * Var(H) / <H>^2

输出：
  - raw_results.csv   : 每一个 (N, layers, J2/J1, seed) 的原始结果
  - agg_results.csv   : 每个配置取 best-2 均值后的聚合结果 (含f), 用于画图

依赖安装:
  pip install tensorcircuit jax jaxlib optax --break-system-packages
"""

import csv
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

# =========================================================
# 0. 实验参数配置 (只需改这里)
# =========================================================
N_LIST       = [4, 6,8]                  # qubit数
LAYERS_LIST  = [2, 4, 6, 8, 10, 12]    # 层数
J2_LIST      = [0.0, 0.25]             # 阻挫程度
N_SEEDS      = 10                       # 每个配置跑几个seed, 取best-2均值
STEPSIZE     = 0.1                     # Adam学习率
RAW_CSV      = "raw_results3.csv"
AGG_CSV      = "agg_results3.csv"


def n_steps_for_layers(n_layers):
    """优化步数随层数自适应增长, 避免深层电路优化不充分"""
    return 200 + 50 * n_layers


# =========================================================
# 1. Hamiltonian: 1D J1-J2 Heisenberg链 (周期边界条件)
#    S_i . S_j = 1/4 (X_i X_j + Y_i Y_j + Z_i Z_j)
# =========================================================
def build_J1J2_bonds(n_qubits, J1=1.0, J2=0.0):
    """返回 bond 列表: [(i, j, coeff_per_pauli), ...]"""
    bonds = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        bonds.append((i, j, 0.25 * J1))
    if abs(J2) > 1e-12:
        for i in range(n_qubits):
            j = (i + 2) % n_qubits
            bonds.append((i, j, 0.25 * J2))
    return bonds


def exact_ground_energy(n_qubits, bonds):
    """用稠密矩阵做exact diagonalization, 验证/对照用"""
    dim = 2 ** n_qubits
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def op_on(qubit, P):
        mats = [I] * n_qubits
        mats[qubit] = P
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((dim, dim), dtype=complex)
    for (i, j, coeff) in bonds:
        for P in (X, Y, Z):
            H += coeff * (op_on(i, P) @ op_on(j, P))

    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0]), H


# =========================================================
# 2. Ansatz: Hardware-efficient (RY + CNOT环)
#    params shape: (n_layers, n_qubits, 2)
# =========================================================
def build_circuit(params, n_qubits, n_layers):
    c = tc.Circuit(n_qubits)
    for l in range(n_layers):
        for q in range(n_qubits):
            c.ry(q, theta=params[l, q, 0])
        for q in range(n_qubits):
            c.cnot(q, (q + 1) % n_qubits)
        for q in range(n_qubits):
            c.ry(q, theta=params[l, q, 1])
    return c


def make_energy_fn(n_qubits, n_layers, bonds):
    """
    返回 energy(params) -> 标量, 通过对每个 bond 的
    XX/YY/ZZ 期望值加权求和实现 <H>。
    """
    def energy(params):
        c = build_circuit(params, n_qubits, n_layers)
        e = 0.0
        for (i, j, coeff) in bonds:
            e = e + coeff * c.expectation_ps(x=[i, j])
            e = e + coeff * c.expectation_ps(y=[i, j])
            e = e + coeff * c.expectation_ps(z=[i, j])
        return tc.backend.real(e)

    return energy


def make_state_fn(n_qubits, n_layers):
    def state_fn(params):
        c = build_circuit(params, n_qubits, n_layers)
        return c.state()
    return state_fn


# =========================================================
# 3. 计算量 f 的定义
# =========================================================
def compute_flops(n_qubits, n_layers, n_steps):
    n_gates = n_layers * 3 * n_qubits
    return n_gates * n_steps


# =========================================================
# 4. 单次 VQE (JIT编译 + JAX优化)
# =========================================================
def run_vqe(n_qubits, n_layers, seed, n_steps, bonds, H_matrix, E_exact):
    energy_fn = make_energy_fn(n_qubits, n_layers, bonds)
    state_fn = make_state_fn(n_qubits, n_layers)

    energy_and_grad = tc.backend.jit(tc.backend.value_and_grad(energy_fn))

    key = jax.random.PRNGKey(seed)
    params = jax.random.uniform(
        key, shape=(n_layers, n_qubits, 2), minval=0.0, maxval=2 * jnp.pi
    )

    optimizer = optax.adam(learning_rate=STEPSIZE)
    opt_state = optimizer.init(params)

    @tc.backend.jit
    def step_fn(params, opt_state):
        loss, grads = energy_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        params, opt_state, loss = step_fn(params, opt_state)

    psi = np.array(state_fn(params))
    H_psi = H_matrix @ psi
    E_final = float(np.real(np.vdot(psi, H_psi)))
    H2 = float(np.real(np.vdot(H_psi, H_psi)))
    var_H = H2 - E_final ** 2
    V_score = n_qubits * var_H / (E_final ** 2)

    return {
        "E_final": E_final,
        "abs_error": abs(E_final - E_exact),
        "V_score": V_score,
    }


# =========================================================
# 5. 主扫描循环
# =========================================================
def main():
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
    done = 0
    t0 = time.time()

    for n_qubits in N_LIST:
        for J2_over_J1 in J2_LIST:

            bonds = build_J1J2_bonds(n_qubits, J2=J2_over_J1)
            E_exact, H_matrix = exact_ground_energy(n_qubits, bonds)

            for n_layers in LAYERS_LIST:
                done += 1
                n_steps = n_steps_for_layers(n_layers)
                f = compute_flops(n_qubits, n_layers, n_steps)

                print(f"\n[{done}/{total}] N={n_qubits}, layers={n_layers}, "
                      f"J2/J1={J2_over_J1}, f={f:.2e}, E_exact={E_exact:.4f}")

                v_scores = []
                for seed in range(N_SEEDS):
                    t1 = time.time()
                    res = run_vqe(n_qubits, n_layers, seed, n_steps,
                                  bonds, H_matrix, E_exact)
                    elapsed = time.time() - t1

                    v_scores.append(res["V_score"])

                    raw_writer.writerow({
                        "n_qubits": n_qubits,
                        "n_layers": n_layers,
                        "J2_over_J1": J2_over_J1,
                        "seed": seed,
                        "f": f,
                        "E_exact": E_exact,
                        "E_final": res["E_final"],
                        "abs_error": res["abs_error"],
                        "V_score": res["V_score"],
                    })
                    raw_file.flush()

                    print(f"  seed={seed}: V-score={res['V_score']:.3e}  "
                          f"abs_err={res['abs_error']:.3e}  ({elapsed:.1f}s)")

                v_sorted = sorted(v_scores)
                best2_mean = (v_sorted[0] + v_sorted[1]) / 2
                best1 = v_sorted[0]

                agg_writer.writerow({
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "J2_over_J1": J2_over_J1,
                    "f": f,
                    "E_exact": E_exact,
                    "V_score_best2mean": best2_mean,
                    "V_score_min": best1,
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