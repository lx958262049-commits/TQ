import numpy as np
from models.rbm import RBM


N=4

rbm=RBM(
    N_visible=N,
    N_hidden=8
)


v=np.array(
    [1,-1,1,-1]
)


eps=1e-6


print("==============================")
print("RBM derivative check")
print("==============================")


# =========================
# check a
# =========================

Oa,Ob,OW=rbm.log_derivatives(v)


old=rbm.a[0]


rbm.a[0]=old+eps
lp=rbm.log_psi(v)


rbm.a[0]=old-eps
lm=rbm.log_psi(v)


rbm.a[0]=old


num=(lp-lm)/(2*eps)


print("\na0")
print("analytic:",Oa[0])
print("numerical:",num)



# =========================
# check b
# =========================


old=rbm.b[0]


rbm.b[0]=old+eps
lp=rbm.log_psi(v)


rbm.b[0]=old-eps
lm=rbm.log_psi(v)


rbm.b[0]=old


num=(lp-lm)/(2*eps)


print("\nb0")
print("analytic:",Ob[0])
print("numerical:",num)



# =========================
# check W
# =========================


old=rbm.W[0,0]


rbm.W[0,0]=old+eps
lp=rbm.log_psi(v)


rbm.W[0,0]=old-eps
lm=rbm.log_psi(v)


rbm.W[0,0]=old


num=(lp-lm)/(2*eps)



print("\nW00")
print("analytic:",OW[0,0])
print("numerical:",num)
