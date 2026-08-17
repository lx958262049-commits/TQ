"""
VQE Scaling Law 扫描脚本 (TensorCircuit + JAX 版本) —— HEA vs HVA 对照实验 + 收敛诊断
==========================================================================
在原有对照实验基础上, 新增"训练结束后梯度范数"记录, 用于区分:
  (a) 真的卡在驻点(局部极小值/鞍点): 训练结束时梯度范数很小, 但能量依然很差
  (b) 只是没训练够: 训练结束时梯度范数依然较大, 说明还在下降路上, 只是步数不够

用法：只需修改 ANSATZ_LIST：
  ANSATZ_LIST = ["hea"]          -> 只跑 HEA
  ANSATZ_LIST = ["hva"]          -> 只跑 HVA
  ANSATZ_LIST = ["hea", "hva"]   -> 两个都跑

新增输出列: final_grad_norm (训练结束时的梯度范数)
新增图: grad_norm_vs_vscore.png (梯度范数 vs V-score 散点图, 用于诊断)

依赖安装:
  pip install tensorcircuit jax jaxlib optax matplotlib pandas --break-system-packages
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
ANSATZ_LIST    = ["hea", "hva"]
N_LIST         = [4, 6, 8]
LAYERS_LIST    = [2, 4, 6, 8, 10, 12]
J2_LIST        = [0.0, 0.25, 0.5]
N_SEEDS        = 10
STEPSIZE       = 0.1
RAW_CSV        = "raw_results_cmp.csv"
AGG_CSV        = "agg_results_cmp.csv"
PLOT_AFTER_RUN = True
PLOT_PNG       = "hea_vs_hva_comparison.png"
GRAD_PLOT_PNG  = "grad_norm_vs_vscore.png"   # <-- 新增: 收敛诊断图


def n_steps_for_layers(n_layers):
    return 200 + 50 * n_layers


# =========================================================
# 1. Hamiltonian
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


def exact_ground_energy(n_qubits, bonds):
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
# 2a. Ansatz: HEA
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


def hea_param_shape(n_qubits, n_layers, bonds):
    return (n_layers, n_qubits, 2)


# =========================================================
# 2b. Ansatz: HVA
# =========================================================
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


def hva_param_shape(n_qubits, n_layers, bonds):
    return (n_layers, len(bonds))


ANSATZ_REGISTRY = {
    "hea": {
        "build_circuit": build_circuit_hea,
        "param_shape": hea_param_shape,
        "n_gates_per_layer": lambda n_qubits, bonds: n_qubits * 3,
    },
    "hva": {
        "build_circuit": build_circuit_hva,
        "param_shape": hva_param_shape,
        "n_gates_per_layer": lambda n_qubits, bonds: len(bonds) * 3,
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


def make_state_fn(ansatz, n_qubits, n_layers, bonds):
    build_circuit = ANSATZ_REGISTRY[ansatz]["build_circuit"]

    def state_fn(params):
        c = build_circuit(params, n_qubits, n_layers, bonds)
        return c.state()

    return state_fn


def compute_flops(ansatz, n_qubits, n_layers, n_steps, bonds):
    n_gates_per_layer = ANSATZ_REGISTRY[ansatz]["n_gates_per_layer"](n_qubits, bonds)
    n_gates = n_layers * n_gates_per_layer
    return n_gates * n_steps


# =========================================================
# 4. 单次 VQE (新增: 训练结束后计算最终梯度范数)
# =========================================================
def run_vqe(ansatz, n_qubits, n_layers, seed, n_steps, bonds, H_matrix, E_exact):
    energy_fn = make_energy_fn(ansatz, n_qubits, n_layers, bonds)
    state_fn = make_state_fn(ansatz, n_qubits, n_layers, bonds)

    energy_and_grad = tc.backend.jit(tc.backend.value_and_grad(energy_fn))

    param_shape = ANSATZ_REGISTRY[ansatz]["param_shape"](n_qubits, n_layers, bonds)

    key = jax.random.PRNGKey(seed)
    params = jax.random.uniform(
        key, shape=param_shape, minval=0.0, maxval=2 * jnp.pi
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

    # ---- 新增: 训练结束后, 在最终参数点上再算一次梯度, 记录范数 ----
    # 用来区分: "真的停在了驻点(梯度≈0)" vs "只是没训练够(梯度依然不小)"
    _, final_grads = energy_and_grad(params)
    final_grad_norm = float(np.linalg.norm(np.array(final_grads)))

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
        "final_grad_norm": final_grad_norm,   # <-- 新增字段
    }


# =========================================================
# 5. 主扫描循环 (新增: final_grad_norm 写入CSV)
# =========================================================
def main():
    raw_fields = ["ansatz", "n_qubits", "n_layers", "J2_over_J1", "seed",
                  "f", "E_exact", "E_final", "abs_error", "V_score", "final_grad_norm"]
    agg_fields = ["ansatz", "n_qubits", "n_layers", "J2_over_J1",
                  "f", "E_exact", "V_score_best2mean", "V_score_min",
                  "final_grad_norm_mean"]   # <-- 新增: 聚合表也带上梯度范数均值

    raw_file = open(RAW_CSV, "w", newline="")
    agg_file = open(AGG_CSV, "w", newline="")
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
    agg_writer = csv.DictWriter(agg_file, fieldnames=agg_fields)
    raw_writer.writeheader()
    agg_writer.writeheader()

    total = len(ANSATZ_LIST) * len(N_LIST) * len(LAYERS_LIST) * len(J2_LIST)
    done = 0
    t0 = time.time()

    for ansatz in ANSATZ_LIST:
        for n_qubits in N_LIST:
            for J2_over_J1 in J2_LIST:

                bonds = build_J1J2_bonds(n_qubits, J2=J2_over_J1)
                E_exact, H_matrix = exact_ground_energy(n_qubits, bonds)

                for n_layers in LAYERS_LIST:
                    done += 1
                    n_steps = n_steps_for_layers(n_layers)
                    f = compute_flops(ansatz, n_qubits, n_layers, n_steps, bonds)

                    print(f"\n[{done}/{total}] ansatz={ansatz}, N={n_qubits}, "
                          f"layers={n_layers}, J2/J1={J2_over_J1}, "
                          f"f={f:.2e}, E_exact={E_exact:.4f}")

                    v_scores = []
                    grad_norms = []
                    for seed in range(N_SEEDS):
                        t1 = time.time()
                        res = run_vqe(ansatz, n_qubits, n_layers, seed, n_steps,
                                      bonds, H_matrix, E_exact)
                        elapsed = time.time() - t1

                        v_scores.append(res["V_score"])
                        grad_norms.append(res["final_grad_norm"])

                        raw_writer.writerow({
                            "ansatz": ansatz,
                            "n_qubits": n_qubits,
                            "n_layers": n_layers,
                            "J2_over_J1": J2_over_J1,
                            "seed": seed,
                            "f": f,
                            "E_exact": E_exact,
                            "E_final": res["E_final"],
                            "abs_error": res["abs_error"],
                            "V_score": res["V_score"],
                            "final_grad_norm": res["final_grad_norm"],
                        })
                        raw_file.flush()

                        print(f"  seed={seed}: V-score={res['V_score']:.3e}  "
                              f"abs_err={res['abs_error']:.3e}  "
                              f"final_grad_norm={res['final_grad_norm']:.3e}  "
                              f"({elapsed:.1f}s)")

                    v_sorted = sorted(v_scores)
                    best2_mean = (v_sorted[0] + v_sorted[1]) / 2
                    best1 = v_sorted[0]
                    grad_norm_mean = float(np.mean(grad_norms))

                    agg_writer.writerow({
                        "ansatz": ansatz,
                        "n_qubits": n_qubits,
                        "n_layers": n_layers,
                        "J2_over_J1": J2_over_J1,
                        "f": f,
                        "E_exact": E_exact,
                        "V_score_best2mean": best2_mean,
                        "V_score_min": best1,
                        "final_grad_norm_mean": grad_norm_mean,
                    })
                    agg_file.flush()

                    print(f"  --> best-2 mean V-score = {best2_mean:.3e}  "
                          f"min = {best1:.3e}  "
                          f"mean final_grad_norm = {grad_norm_mean:.3e}")

    raw_file.close()
    agg_file.close()
    print(f"\n完成! 总用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"原始数据: {RAW_CSV}")
    print(f"聚合数据: {AGG_CSV}")


# =========================================================
# 6. 画图 6a: HEA vs HVA, V-score vs f (跟之前一样)
# =========================================================
def plot_results(agg_csv=AGG_CSV, out_png=PLOT_PNG):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(agg_csv)
    n_list = sorted(df["n_qubits"].unique())
    j2_list = sorted(df["J2_over_J1"].unique())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, len(n_list), figsize=(5.3 * len(n_list), 5), sharey=True)
    if len(n_list) == 1:
        axes = [axes]

    for idx, n in enumerate(n_list):
        ax = axes[idx]
        sub_n = df[df.n_qubits == n]
        for j2_idx, j2 in enumerate(j2_list):
            color = colors[j2_idx % len(colors)]
            for ansatz, ls, marker, alpha in [("hea", "-", "o", 0.9), ("hva", "--", "s", 0.6)]:
                s = sub_n[(sub_n.J2_over_J1 == j2) & (sub_n.ansatz == ansatz)].sort_values("f")
                if len(s) == 0:
                    continue
                ax.plot(s["f"], s["V_score_best2mean"], ls, marker=marker,
                         color=color, alpha=alpha, markersize=5,
                         label=f"J2/J1={j2} ({ansatz.upper()})")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"N={n}")
        ax.set_xlabel("f (compute)")
        if idx == 0:
            ax.set_ylabel("V-score (best-2 mean)")
        ax.grid(alpha=0.3)

    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.suptitle("HEA (solid) vs HVA (dashed): V-score vs compute f, colored by frustration ratio",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"图已保存: {out_png}")


# =========================================================
# 6b. 新增画图: 最终梯度范数 vs V-score 散点图 (收敛诊断)
#     用于区分"卡在驻点"(左下角: 梯度小但V-score大) 还是
#     "没训练够"(右上角: 梯度大且V-score大, 说明还在下降路上)
# =========================================================
def plot_grad_diagnosis(agg_csv=AGG_CSV, out_png=GRAD_PLOT_PNG):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(agg_csv)
    # 只看训练最充分的一组(最大layers), 最能代表"最终收敛状态"
    max_layers = df["n_layers"].max()
    sub = df[df.n_layers == max_layers]

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"hea": "o", "hva": "s"}
    colors = {0.0: "#1f77b4", 0.25: "#ff7f0e", 0.5: "#d62728"}

    for ansatz in sub["ansatz"].unique():
        for j2 in sub["J2_over_J1"].unique():
            s = sub[(sub.ansatz == ansatz) & (sub.J2_over_J1 == j2)]
            if len(s) == 0:
                continue
            ax.scatter(s["final_grad_norm_mean"], s["V_score_best2mean"],
                       marker=markers[ansatz], color=colors[j2], s=80,
                       label=f"{ansatz.upper()}, J2/J1={j2}", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final gradient norm (after training)")
    ax.set_ylabel("V-score (best-2 mean)")
    ax.set_title(f"Convergence diagnosis (layers={max_layers})\n"
                 "bottom-left = stuck at stationary point | top-right = under-trained")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"收敛诊断图已保存: {out_png}")


if __name__ == "__main__":
    main()
    if PLOT_AFTER_RUN:
        plot_results()
        plot_grad_diagnosis()