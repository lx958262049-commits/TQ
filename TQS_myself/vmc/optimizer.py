import numpy as np


class SGD:


    def __init__(
        self,
        model,
        lr=0.01
    ):

        self.model=model
        self.lr=lr



    def step(
        self,
        gradients
    ):
        params = self.model.parameters()

        for p, g in zip(params, gradients):
            p -= self.lr * g.real


class Adam:
    """
    标准 Adam 优化器，接口和 SGD 完全一致（model.parameters() + step(gradients)），
    所以在 train_rbm.py 里可以直接把 SGD(...) 换成 Adam(...)，不用改别的代码。

    之后接 Transformer 时（参数量比 RBM 大好几个量级），Adam 的自适应学习率
    通常比朴素 SGD 稳得多，这也是为什么现在先把它准备好。
    """

    def __init__(
        self,
        model,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        # 一阶矩 m 和二阶矩 v，形状和每个参数一一对应
        params = model.parameters()
        self.m = [np.zeros_like(p, dtype=float) for p in params]
        self.v = [np.zeros_like(p, dtype=float) for p in params]

    def step(
        self,
        gradients
    ):
        self.t += 1

        params = self.model.parameters()

        for i, (p, g) in enumerate(zip(params, gradients)):

            g = g.real

            # 一阶矩、二阶矩的指数滑动平均
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # 偏置修正（训练初期 m、v 都从 0 开始，需要修正）
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 原地更新（p 是 model.parameters() 里的引用，直接改会同步到模型上）
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)