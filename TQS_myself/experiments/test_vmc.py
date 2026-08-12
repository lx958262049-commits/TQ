import itertools
import numpy as np


from physics.ising import transverse_field_ising

from benchmark.exact_solver import exact_ground_state

from vmc.sampler import wavefunction_sampling

from models.random_state import RandomWaveFunction
from vmc.metropolis import metropolis_sample


N=4


H=transverse_field_ising(
    N,
    J=1,
    h=1
)


E_exact,_=exact_ground_state(H)


print(
    "Exact:",
    E_exact
)



basis=list(
    itertools.product(
        [1,-1],
        repeat=N
    )
)


psi=RandomWaveFunction(N)


samples=metropolis_sample(
    psi,
    N,
    1000
)



print(
    "Samples:",
    samples.shape
)
# 补上最后一步：计算当前随机波函数的 VMC 能量期望值
from vmc.energy import expectation_energy  # 假设你把第二段代码存放在这里

E_vmc = expectation_energy(samples, H, psi, basis)
print("VMC Energy Estimator:", E_vmc)