import numpy as np


class RandomWaveFunction:


    def __init__(self,N):

        self.N=N

        self.states={}


        # 为所有configuration分配固定随机值
        import itertools

        for s in itertools.product([1,-1], repeat=N):

            self.states[s]=np.random.rand()+0.1



    def __call__(self,s):

        return self.states[tuple(s)]
