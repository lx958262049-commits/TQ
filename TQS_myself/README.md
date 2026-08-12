# TQS_myself — 论文复现进度报告

复现论文：**Yuan-Hang Zhang & Massimiliano Di Ventra**,
*"Transformer Quantum State: A Multi-Purpose Model for Quantum Many-Body Problems"*,
Phys. Rev. B **107**, 075147 (2023).

论文提出 **Transformer Quantum State (TQS)**：用 encoder-only Transformer 自回归地表示量子多体波函数 $\psi(s,J)=A(s,J)e^{i\phi(s,J)}$，其中 $J$（哈密顿量参数）作为条件输入。核心卖点是**单个模型**可以：
1. 直接生成整个相图（对不同 $J$ 通用，无需重新训练）；
2. 从实验测量数据反推物理参数；
3. 把在旧系统上学到的知识**迁移**到没见过的新系统。
---

## 一、对照论文的总体进度表

| 论文中的组件 | 本项目状态 | 说明 |
|---|---|---|
| 横场伊辛模型 (TFIM) 精确对角化基准 | ✅ 已完成 | `physics/ising.py` + `benchmark/exact_solver.py` |
| VMC 采样与能量估计框架 | ✅ 已完成 | `vmc/metropolis.py` + `vmc/energy.py` |
| 局域能量计算（稀疏化，$O(N)$ 而非 $O(2^N)$） | ✅ 已完成 | `local_energy_tfim` 系列，见下文 |
| 统一模型接口（为多 ansatz 铺路） | ✅ 已完成 | `models/base.py` 抽象类，`RBM` 已实现 |
| 模型无关的梯度 / 优化器 | ✅ 已完成 | `vmc/gradient.py` + `vmc/optimizer.py`（`SGD` 类） |
| RBM 变分波函数 + 训练闭环 | ✅ 已完成 | `models/rbm.py` + `experiments/train_rbm.py`，已跑到 N=10 |
| **Transformer ansatz（论文核心）** | 🟡 **刚起了骨架，逻辑未实现** | `models/transformer.py`，`psi`/`log_derivatives`/`parameters` 目前都是空的 `pass` |
| 自回归直接采样（论文方法，替代 Metropolis） | ❌ 未开始 | 依赖 Transformer 的自回归结构，得先把上一项做出来 |
| 参数 $J$ 作为条件输入 / 生成整个相图 | ❌ 未开始 | 目前所有实验都固定 $J=1,h=1$ |
| 从实验测量反推参数 | ❌ 未开始 | 论文核心亮点之一 |
| 迁移学习到新系统 | ❌ 未开始 | 论文核心亮点之一 |
| 随机重整化 (SR) / Adam 等更优优化器 | ❌ 未开始 | `optimizer.py` 目前只有 `SGD` 一种，不过接口已经留好了，加新优化器很容易 |

**一句话总结现状**： Transformer 本身——论文标题的主角——目前还是一个空壳，还没有任何实际的网络结构或前向计算。

---

## 二、已完成的工作（详细）

### 1. 物理模型与精确对角化基准
- `physics/ising.py`：Pauli 矩阵 Kronecker 积构造 1D 横场伊辛哈密顿量 $H=-J\sum Z_iZ_{i+1}-h\sum X_i$。
- `benchmark/exact_solver.py`：精确对角化，作为所有变分结果的对照标准。
- `experiments/run_fss_experiment.py`：$N=4,6,8,10$ 有限尺寸标度（`results/ising_fss_plot.png`）。

### 2. VMC 基础设施 + 局域能量稀疏化
- `vmc/metropolis.py`：Metropolis-Hastings 采样，自旋约定统一为 $s_i\in\{+1,-1\}$（`vmc/sampler.py` 里遗留的旧接口也已经跟进这个约定）。
- `vmc/energy.py`：
  - 保留了原来基于 `H` 矩阵 + 全基枚举的 `local_energy`/`get_local_energies`（$O(2^N)$，仅适合小系统对拍验证用）；
  - **新增稀疏解析版本** `local_energy_tfim` / `get_local_energies_tfim` / `expectation_energy_tfim`：直接用 TFIM 的局域连接结构展开
    $$E_{loc}(s) = -J\sum_i s_is_{i+1} - h\sum_i \frac{\psi(s^{(i)})}{\psi(s)}$$
    每个样本从 $O(2^N)$ 降到 $O(N)$，不再需要 `H` 矩阵或 `basis` 全枚举。
- `experiments/test_sparse.py`：逐样本对比稀疏版 vs 全枚举版，验证两者数值一致。
- `experiments/runtime_scaling.py`：系统性对比 dense/sparse 随 $N$ 增长的耗时，结果存在 `results/runtime_scaling.csv` + `results/computational_cost_comparison.png`（目前跑到 $N=12$，脚本里设的是到 $N=16$，见下文"待办"）。
- `experiments/mc_convergence.py`：统计误差随采样数按 $\sigma\propto N_s^{-1/2}$ 收敛的验证。

### 3. 统一模型接口 + 模型无关的训练框架
- `models/base.py`：新增抽象基类 `NeuralQuantumState`，规定任何 ansatz 都要实现 `psi`/`log_psi`/`log_derivatives`/`parameters` 四个接口。
- `models/rbm.py`：`RBM` 类正式继承并实现这个接口（新增了 `parameters()` 方法）。
- `vmc/gradient.py`：把原来 RBM 专用的梯度公式（原 `rbm_optimizer.py`）重写成 `vmc_gradient(model, samples, local_energies)`，内部只调用 `model.log_derivatives(s)`，不关心 `model` 具体是什么结构——这样以后换成 Transformer 也能直接用。
- `vmc/optimizer.py`：**这是对上次讨论的死代码问题的修复**——原来这里是一段调用未定义函数、且没有任何地方引用的废代码，现在重写成了标准的 `SGD` 优化器类（`step(gradients)` 更新 `model.parameters()`），真正接入了训练循环。
- `experiments/check_rbm_derivative.py` / `experiments/gradient_check.py`：分别验证 `log_derivatives` 解析导数、`vmc_gradient` 解析梯度和数值梯度的一致性。
- `experiments/test_interface.py`：验证 `RBM` 是否正确实现了 `NeuralQuantumState` 接口（`psi`/`log_psi`/`log_derivatives` 能正常调用）。

### 4. RBM-VMC 训练规模升级
- `experiments/train_rbm.py` 现在用的是新的 `vmc_gradient` + `SGD` 组合，系统规模也从 $N=4$ 升级到了 **$N=10$**（隐藏单元数 20，采样数 5万/epoch）。我用缩小规模（3000 样本、8 个 epoch）跑了一遍确认脚本本身没问题：初始能量 $\approx-9.92$，几步内已经在往精确解 $E_{exact}=-12.3815$ 靠近，训练逻辑是通的。

---

## 三、还差的工作

### B. 通向论文核心结论的部分（按优先级）
1. **把 `models/transformer.py` 填起来**——这是当前最核心的缺口。需要实现：
   - 自回归结构 $P(s,J)=\prod_iP(s_i|s_{<i},J)$ 的带掩码自注意力；
   - 振幅头 $A$ 和相位头 $\phi$ 两个输出分支；
   - 因为已经有 `models/base.py` 这个统一接口，只要 `TransformerNQS` 把 `psi`/`log_derivatives`/`parameters` 实现出来，现有的 `vmc_gradient`、`SGD`、`metropolis_sample` 理论上都能直接复用，不用重写训练脚本。
2. **自回归精确采样**：Transformer 版本一旦有了自回归结构，可以直接从 $P(s|J)$ 采样，摆脱 Metropolis（论文的效率优势之一），这个可以和第1点一起做。
3. **优化器升级**：`vmc/optimizer.py` 现在的接口设计已经支持扩展，可以加一个 `SR`（随机重整化）或 `Adam` 类，复用同样的 `step(gradients)` 接口。
4. **$J$ 条件输入 / 相图生成、参数反演、迁移学习**：这三个是论文"多用途"的核心卖点，建议放在 Transformer ansatz 跑通单点训练之后再依次做。

---

## 四、项目结构

```
TQS_myself/
├── physics/
│   └── ising.py                # 横场伊辛哈密顿量构建
├── models/
│   ├── base.py                  # NeuralQuantumState 抽象接口（新增）
│   ├── rbm.py                    # RBM 变分波函数（已实现接口）
│   ├── random_state.py            # 随机波函数 baseline
│   └── transformer.py              # Transformer ansatz 骨架（待实现）
├── vmc/
│   ├── metropolis.py               # Metropolis-Hastings 采样
│   ├── sampler.py                    # 精确枚举采样等辅助工具
│   ├── energy.py                       # 局域能量：dense（全枚举）+ sparse（TFIM解析，新增）
│   ├── gradient.py                       # 模型无关的 VMC 梯度（新增，替代原 rbm_optimizer.py）
│   └── optimizer.py                        # SGD 优化器类（已修复，原为无用死代码）
├── benchmark/
│   └── exact_solver.py                       # 精确对角化
├── experiments/
│   ├── run_fss_experiment.py                   # 有限尺寸标度
│   ├── mc_convergence.py                         # MC 采样收敛性
│   ├── check_rbm_derivative.py                     # RBM 对数导数验证
│   ├── gradient_check.py                             # VMC 梯度验证
│   ├── test_interface.py                               # 模型接口一致性检验（新增）
│   ├── test_sparse.py                                    # 稀疏 vs 全枚举局域能量对拍（新增）
│   ├── runtime_scaling.py                                  # 稀疏化耗时对比基准（新增）
│   └── train_rbm.py                                          # RBM-VMC 训练主程序（已升级到 N=10）
├── results/                                                     # 各实验的 CSV + 图
└── 报告/最小模版.tex                                              # 阶段汇报
```

## 五、运行方法

```bash
# 精确对角化基准 + 有限尺寸标度
python experiments/run_fss_experiment.py

# Monte Carlo 采样收敛性
python experiments/mc_convergence.py

# 模型接口 / 导数 / 梯度正确性验证
python experiments/test_interface.py
python experiments/check_rbm_derivative.py
python experiments/gradient_check.py

# 稀疏局域能量正确性 + 耗时对比
python experiments/test_sparse.py
python experiments/runtime_scaling.py

# RBM-VMC 训练（当前项目主线成果，N=10）
python experiments/train_rbm.py
```

依赖：`numpy`、`matplotlib`、`pandas`。