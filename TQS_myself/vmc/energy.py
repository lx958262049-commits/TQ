import numpy as np



def local_energy(
        s,
        H,
        psi,
        basis
):

    """
    Calculate local energy:

    Eloc(s)=
    sum_s' Hss' psi(s')/psi(s)

    """


    index = basis.index(
        tuple(s)
    )


    psi_s = psi(s)


    result=0


    for j,sp in enumerate(basis):

        H_element = H[index,j]


        if abs(H_element)>1e-12:

            result += (
                H_element
                *
                psi(np.array(sp))
                /
                psi_s
            )


    return result



def expectation_energy(
        samples,
        H,
        psi,
        basis
):

    energies=[]


    for s in samples:

        e=local_energy(
            s,
            H,
            psi,
            basis
        )

        energies.append(e)


    return np.mean(
        energies
    )
def get_local_energies(
        samples,
        H,
        psi,
        basis
):

    energies=[]


    for s in samples:

        e=local_energy(
            s,
            H,
            psi,
            basis
        )

        energies.append(e)


    return np.array(energies)


# ======================================================================
# 稀疏版本（针对 1D TFIM 解析展开，不再需要 H 矩阵或 basis 全枚举）
#
# 对 1D open-chain TFIM:
#   H = -J sum_i Z_i Z_{i+1}  -h sum_i X_i
#
# 给定组态 s，能与它耦合的态只有：
#   - s 本身（对角项，来自 ZZ）
#   - N 个单点翻转态 s^(i)（来自每个位点的 X）
# 即每个样本只需要 O(N) 次 psi 求值，而不是 O(2^N)。
# ======================================================================

def local_energy_tfim(
        s,
        psi,
        J=1.0,
        h=1.0
):
    """
    解析计算 1D open-chain TFIM 的局域能量：

    Eloc(s) = -J * sum_i s_i*s_{i+1}
              -h * sum_i psi(s^(i)) / psi(s)

    s: 长度为 N 的 numpy 数组，自旋取值 +1/-1
    psi: 波函数振幅函数 psi(s) -> 标量（可为复数）

    注意：这里假设的是开边界链（与 physics/ising.py 一致，
    即 sum_i 只到 i=N-2，不包含首尾环绕项）。
    """

    N = len(s)

    psi_s = psi(s)

    # ---- 对角项 (ZZ 相互作用) ----
    e_diag = -J * np.sum(s[:-1] * s[1:])

    # ---- 非对角项 (X 横场，逐点翻转) ----
    e_offdiag = 0.0

    for i in range(N):

        s_flip = s.copy()
        s_flip[i] = -s_flip[i]

        e_offdiag += psi(s_flip) / psi_s

    e_offdiag = -h * e_offdiag

    return e_diag + e_offdiag


def get_local_energies_tfim(
        samples,
        psi,
        J=1.0,
        h=1.0
):
    """
    对一批样本批量计算局域能量（稀疏版本）。
    返回 numpy 数组，用法和 get_local_energies 完全一致，
    只是不再需要传 H 和 basis。
    """

    energies = []

    for s in samples:

        e = local_energy_tfim(
            s,
            psi,
            J=J,
            h=h
        )

        energies.append(e)

    return np.array(energies)


def expectation_energy_tfim(
        samples,
        psi,
        J=1.0,
        h=1.0
):

    return np.mean(
        get_local_energies_tfim(
            samples,
            psi,
            J=J,
            h=h
        ).real
    )