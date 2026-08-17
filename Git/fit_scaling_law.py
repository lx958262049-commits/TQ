"""
VQE Scaling Law 拟合与画图脚本
================================
功能:
  1. 读取 agg_results.csv (包含 n_qubits, n_layers, J2_over_J1, f, V_score_best2mean)
  2. 对每个 (N, J2/J1) 组合, 自动截断"触底反弹"之后的数据点
     (截断逻辑: 找到V-score的全局最小值所在位置, 只保留该位置(含)之前的点用于拟合,
      不依赖N的具体数值, 纯粹基于数据形状判断)
  3. 对截断后的数据做幂律拟合: V-score = A * f^(-alpha)  (固定N, 单独拟合每条曲线)
  4. 画出: (a) 原始曲线 + 拟合直线 (log-log)
           (b) 不同N的拟合alpha对比 (按J2/J1分组)
           (c) alpha 随 J2/J1 变化的趋势图

用法:
  python fit_scaling_law.py agg_results.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC",
                                     "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"


# =========================================================
# 1. 自动截断: 找到"触底反弹"之前的有效数据段
# =========================================================
def auto_truncate(f, v, floor_threshold=1e-12, min_points=2):
    """
    输入: f, v 已按 f 从小到大排序的数组 (v = V_score, 可能含机器精度噪声/负值)
    逻辑:
      - 把 v 中绝对值小于 floor_threshold 的点视为"已经触及机器精度地板"
        (这些点本身是有效的好结果, 不应被当作噪声丢弃, 但其后的所有点都不再可信,
         因为继续加深电路只会让优化更不稳定, 不会有真实的物理改善)
      - 找到第一个触底点的位置 idx_floor (如果存在)
      - 同时, 在触底点之前的范围内, 找到 v 的最小值位置 idx_min (排除触底点本身的极端小值
        造成的干扰, 这里 idx_min 是指"最后一个仍在下降趋势上"的点)
      - 截断规则:
          如果存在触底点 -> 保留到第一个触底点(含)为止
          如果不存在触底点 -> 保留到全局最小值点(含)为止 (原始逻辑)
    返回: (f_trunc, v_trunc, f_dropped, v_dropped)
    """
    v_abs = np.abs(v)
    floor_mask = v_abs < floor_threshold

    if floor_mask.any():
        # 第一个触底点的位置, 截断到这里(含)为止
        idx_cut = int(np.argmax(floor_mask))  # 第一个 True 的位置
    else:
        # 没有触底, 退回到"全局最小值之前"的逻辑
        idx_cut = int(np.argmin(v_abs))

    f_trunc = f[: idx_cut + 1]
    v_trunc = v[: idx_cut + 1]
    f_dropped = f[idx_cut + 1:]
    v_dropped = v[idx_cut + 1:]

    if len(f_trunc) < min_points:
        return f, v, np.array([]), np.array([])

    return f_trunc, v_trunc, f_dropped, v_dropped


# =========================================================
# 2. 幂律拟合: V-score = A * f^(-alpha)
#    在 log-log 空间做线性拟合更稳健: log(V) = log(A) - alpha*log(f)
# =========================================================
def fit_power_law(f, v):
    """
    返回: alpha, A, alpha_err (拟合标准误差)
    要求 v 中所有值 > 0 (机器精度噪声/负值需要提前过滤)

    点数处理:
      - < 2 个有效点: 无法定义直线, 返回 None
      - = 2 个有效点: 可以唯一确定一条直线(自由度为0), 但无法估计误差,
                      alpha_err 返回 nan 而不是报错
      - >= 3 个有效点: 正常用 polyfit + cov 估计 alpha 和误差
    """
    mask = v > 0
    f, v = f[mask], v[mask]

    if len(f) < 2:
        return None, None, None

    log_f = np.log(f)
    log_v = np.log(v)

    if len(f) == 2:
        # 两点确定一条直线, 解析计算斜率, 无法估计误差
        neg_alpha = (log_v[1] - log_v[0]) / (log_f[1] - log_f[0])
        log_A = log_v[0] - neg_alpha * log_f[0]
        alpha = -neg_alpha
        A = np.exp(log_A)
        return alpha, A, np.nan

    # 线性拟合 log_v = log_A - alpha * log_f
    coeffs, cov = np.polyfit(log_f, log_v, deg=1, cov=True)
    neg_alpha, log_A = coeffs
    alpha = -neg_alpha
    A = np.exp(log_A)
    alpha_err = np.sqrt(cov[0, 0])

    return alpha, A, alpha_err


# =========================================================
# 3. 主流程
# =========================================================
def main(csv_path):
    df = pd.read_csv(csv_path)

    N_list = sorted(df["n_qubits"].unique())
    J2_list = sorted(df["J2_over_J1"].unique())

    results = []  # 存放每个 (N, J2) 的拟合结果

    # ---- 图1: 原始曲线 + 拟合直线 ----
    fig1, axes1 = plt.subplots(1, len(J2_list), figsize=(7 * len(J2_list), 5))
    if len(J2_list) == 1:
        axes1 = [axes1]

    colors = plt.cm.tab10(np.linspace(0, 1, len(N_list)))

    for ax, J2 in zip(axes1, J2_list):
        sub_J2 = df[df["J2_over_J1"] == J2]

        for N, color in zip(N_list, colors):
            sub = sub_J2[sub_J2["n_qubits"] == N].sort_values("f")
            f = sub["f"].values.astype(float)
            v_raw = sub["V_score_best2mean"].values.astype(float)

            if len(f) < 2:
                print(f"[警告] N={N}, J2/J1={J2}: 数据点不足, 跳过")
                continue

            # 数据健康度检查: 如果整组数据的V-score始终在同一数量级附近浮动,
            # 没有任何明显下降趋势, 说明这组配置 (layers范围/ansatz/步数) 根本没有
            # 让VQE收敛, 所有点都是噪声, 不应该被拟合成一个"看似有效"的alpha。
            # 判据: 最大值/最小值比 < HEALTH_RATIO_THRESHOLD 时判定为未收敛。
            v_abs_raw = np.abs(v_raw)
            v_abs_raw_nonzero = v_abs_raw[v_abs_raw > 1e-300]
            if len(v_abs_raw_nonzero) >= 2:
                health_ratio = v_abs_raw_nonzero.max() / v_abs_raw_nonzero.min()
            else:
                health_ratio = 1.0
            HEALTH_RATIO_THRESHOLD = 10.0  # V-score至少要变化1个数量级以上才算有下降趋势

            if health_ratio < HEALTH_RATIO_THRESHOLD:
                print(f"[跳过] N={N}, J2/J1={J2}: V-score在整个layers范围内几乎不变 "
                      f"(max/min={health_ratio:.2f}x < {HEALTH_RATIO_THRESHOLD}x), "
                      f"判定为该配置下电路未收敛 (可能需要更大的layers范围), 不参与拟合")
                # 仍然把原始点画在图上(全部标为excluded), 方便看出"这组数据是平的"
                v_plot = np.where(v_abs_raw < 1e-15, 1e-15, v_abs_raw)
                ax.plot(f, v_plot, "o--", color=color, markersize=6, alpha=0.4,
                         markerfacecolor="white", markeredgecolor=color,
                         label=f"N={N} (未收敛, 不参与拟合)")
                continue

            # 第一步: 用原始数据(含可能的负值/机器精度噪声)做触底检测和截断
            # 触底点本身被保留在 trunc 段内(它是"已成功收敛"的标志),
            # 但截断点之后的所有点(包括反弹的)被丢弃
            f_trunc_raw, v_trunc_raw, f_drop, v_drop = auto_truncate(f, v_raw)

            # 第二步: 对截断后的数据再过滤非正值(这些点本身仍然代表"收敛成功",
            # 只是数值受限于浮点精度无法直接用于log拟合, 不参与回归但仍画在图上)
            fit_mask = v_trunc_raw > 1e-14
            f_for_fit = f_trunc_raw[fit_mask]
            v_for_fit = v_trunc_raw[fit_mask]

            alpha, A, alpha_err = fit_power_law(f_for_fit, v_for_fit)

            # 画图: 截断段内的点(无论是否参与拟合)都标为实心
            v_trunc_plot = np.abs(v_trunc_raw)
            v_trunc_plot = np.where(v_trunc_plot < 1e-15, 1e-15, v_trunc_plot)
            ax.plot(f_trunc_raw, v_trunc_plot, "o", color=color, markersize=7,
                     label=f"N={N} (kept)")
            if len(f_drop) > 0:
                v_drop_plot = np.abs(v_drop)
                v_drop_plot = np.where(v_drop_plot < 1e-15, 1e-15, v_drop_plot)
                ax.plot(f_drop, v_drop_plot, "o", color=color, markersize=7,
                         markerfacecolor="white", markeredgecolor=color,
                         label=f"N={N} (excluded)")

            # 连线 (全部点, 方便看趋势)
            f_all_sorted = np.concatenate([f_trunc_raw, f_drop])
            v_all_plot = np.concatenate([v_trunc_plot, v_drop_plot]) if len(f_drop) > 0 else v_trunc_plot
            ax.plot(f_all_sorted, v_all_plot, "-", color=color, alpha=0.3)

            # 画拟合直线 (只用 f_for_fit 范围)
            if alpha is not None and len(f_for_fit) >= 2:
                f_fit_line = np.linspace(f_for_fit.min(), f_for_fit.max(), 50)
                v_fit_line = A * f_fit_line ** (-alpha)
                ax.plot(f_fit_line, v_fit_line, "--", color=color, linewidth=1.5)

                results.append({
                    "n_qubits": N,
                    "J2_over_J1": J2,
                    "alpha": alpha,
                    "alpha_err": alpha_err,
                    "A": A,
                    "log_A": np.log(A),
                    "n_points_used": len(f_for_fit),
                    "n_points_excluded": len(f_drop) + (len(f_trunc_raw) - len(f_for_fit)),
                })

                err_str = f"{alpha_err:.3f}" if not np.isnan(alpha_err) else "N/A (仅2点,无法估计误差)"
                confidence_note = "  [低置信度: 仅2个点]" if len(f_for_fit) == 2 else ""
                print(f"N={N}, J2/J1={J2}: alpha = {alpha:.3f} +/- {err_str}  "
                      f"(用{len(f_for_fit)}个点拟合, 截断{len(f_drop)}个反弹点, "
                      f"{len(f_trunc_raw)-len(f_for_fit)}个点已到机器精度地板未参与回归)"
                      f"{confidence_note}")
            else:
                print(f"[警告] N={N}, J2/J1={J2}: 有效拟合点不足(<2), 无法拟合alpha")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f (compute)")
        ax.set_ylabel("V-score (best-2 mean)")
        ax.set_title(f"J2/J1 = {J2}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig1.savefig("fit_powerlaw.png", dpi=130)
    print("\n已保存: fit_powerlaw.png")

    # ---- 保存拟合结果表 ----
    results_df = pd.DataFrame(results)
    results_df.to_csv("fit_results.csv", index=False)
    print("已保存: fit_results.csv")
    print("\n=== 拟合结果汇总 ===")
    print(results_df.to_string(index=False))

    # ---- 图2: alpha 随 J2/J1 变化 (核心结论图) ----
    if len(results_df) > 0:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for N, color in zip(N_list, colors):
            sub = results_df[results_df["n_qubits"] == N].sort_values("J2_over_J1")
            if len(sub) == 0:
                continue
            ax2.errorbar(sub["J2_over_J1"], sub["alpha"], yerr=sub["alpha_err"],
                         marker="o", capsize=4, color=color, label=f"N={N}")
        ax2.set_xlabel("J2 / J1 (阻挫程度)")
        ax2.set_ylabel("拟合得到的 alpha")
        ax2.set_title("Scaling exponent alpha vs 阻挫程度")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.savefig("alpha_vs_frustration.png", dpi=130)
        print("已保存: alpha_vs_frustration.png")

    plt.show()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "agg_results.csv"
    main(csv_path)