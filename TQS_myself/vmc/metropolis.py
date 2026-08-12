import numpy as np



def metropolis_sample(
        psi,
        N,
        num_samples,
        burn_in=100
):

    """
    Metropolis-Hastings sampler

    Samples according to |psi(s)|^2

    """


    # 初始状态

    s=2*np.random.randint(
        0,
        2,
        size=N
    ) - 1


    samples=[]


    for step in range(
        num_samples+burn_in
    ):


        # copy current state

        s_new=s.copy()


        # 随机选择一个spin翻转

        flip=np.random.randint(N)

        s_new[flip]=-s_new[flip]


        # acceptance probability

        p_old=abs(
            psi(s)
        )**2


        p_new=abs(
            psi(s_new)
        )**2

        ratio = min(
            1,
            p_new / p_old
        )



        if np.random.rand()<min(1,ratio):

            s=s_new



        # burn-in结束后保存

        if step>=burn_in:

            samples.append(
                s.copy()
            )



    return np.array(samples)
