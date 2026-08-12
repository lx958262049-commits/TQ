import numpy as np
import itertools

from models.rbm import RBM

from physics.ising import transverse_field_ising

from vmc.energy import (
    expectation_energy,
    expectation_energy_tfim
)

from vmc.metropolis import metropolis_sample



N=6

J=1
h=1


H=transverse_field_ising(
    N,J,h
)


basis=list(
    itertools.product(
        [-1,1],
        repeat=N
    )
)



rbm=RBM(
    N,
    8
)


samples=metropolis_sample(
    rbm.psi,
    N,
    10000
)



E_dense=expectation_energy(
    samples,
    H,
    rbm.psi,
    basis
)



E_sparse=expectation_energy_tfim(
    samples,
    rbm.psi,
    J,
    h
)



print("Dense:",E_dense)

print("Sparse:",E_sparse)

print(
"Difference:",
abs(E_dense-E_sparse)
)