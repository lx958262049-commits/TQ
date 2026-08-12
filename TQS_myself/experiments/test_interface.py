from models.rbm import RBM


rbm=RBM(
    N_visible=4,
    N_hidden=8
)


s=[1,-1,1,-1]


print(
    rbm.psi(s)
)


print(
    rbm.log_psi(s)
)


print(
    rbm.log_derivatives(s)
)