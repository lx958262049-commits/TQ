import itertools
import numpy as np

from models.rbm import RBM

from physics.ising import transverse_field_ising

from vmc.energy import get_local_energies, expectation_energy

from vmc.metropolis import metropolis_sample

from vmc.gradient import vmc_gradient



N = 4


# ===============================
# Hamiltonian
# ===============================

H = transverse_field_ising(
    N,
    1,
    1
)


basis = list(
    itertools.product(
        [1,-1],
        repeat=N
    )
)



# ===============================
# RBM
# ===============================

rbm = RBM(
    N_visible=N,
    N_hidden=8
)



# ===============================
# analytic gradient
# ===============================

samples = metropolis_sample(
    rbm.psi,
    N,
    200000
)


local_E = get_local_energies(
    samples,
    H,
    rbm.psi,
    basis
)



grad_a,grad_b,grad_W = vmc_gradient(
    rbm,
    samples,
    local_E
)



epsilon = 1e-2



print("================================")
print("      VMC Gradient Check")
print("================================")



# ==================================================
# helper function
# ==================================================

def numerical_energy():

    E = expectation_energy(
        samples,
        H,
        rbm.psi,
        basis
    )

    return E.real



# ==================================================
# 1. check a[0]
# ==================================================

old = rbm.a[0]


rbm.a[0] = old + epsilon

E_plus = numerical_energy()



rbm.a[0] = old - epsilon

E_minus = numerical_energy()



rbm.a[0] = old



num_grad = (
    E_plus-E_minus
)/(2*epsilon)



print("\n[a0]")
print(
    "Analytic:",
    grad_a[0].real
)

print(
    "Numerical:",
    num_grad
)

print(
    "Difference:",
    abs(
        grad_a[0].real-num_grad
    )
)



# ==================================================
# 2. check b[0]
# ==================================================

old = rbm.b[0]


rbm.b[0] = old + epsilon

E_plus = numerical_energy()



rbm.b[0] = old - epsilon

E_minus = numerical_energy()



rbm.b[0] = old



num_grad = (
    E_plus-E_minus
)/(2*epsilon)



print("\n[b0]")
print(
    "Analytic:",
    grad_b[0].real
)

print(
    "Numerical:",
    num_grad
)

print(
    "Difference:",
    abs(
        grad_b[0].real-num_grad
    )
)



# ==================================================
# 3. check W[0,0]
# ==================================================

old = rbm.W[0,0]


rbm.W[0,0] = old + epsilon

E_plus = numerical_energy()



rbm.W[0,0] = old - epsilon

E_minus = numerical_energy()



rbm.W[0,0] = old



num_grad = (
    E_plus-E_minus
)/(2*epsilon)



print("\n[W00]")
print(
    "Analytic:",
    grad_W[0,0].real
)

print(
    "Numerical:",
    num_grad
)

print(
    "Difference:",
    abs(
        grad_W[0,0].real-num_grad
    )
)


print("\n================================")