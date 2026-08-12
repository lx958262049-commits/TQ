import numpy as np


# Pauli matrices

I = np.eye(2)

X = np.array([
    [0,1],
    [1,0]
],dtype=complex)


Z = np.array([
    [1,0],
    [0,-1]
],dtype=complex)



def kron_all(operators):
    """
    Kronecker product for many operators
    """

    result = operators[0]

    for op in operators[1:]:
        result=np.kron(result,op)

    return result



def transverse_field_ising(
        N,
        J=1.0,
        h=1.0
):
    """
    1D transverse field Ising model

    H=
    -J sum Zi Zi+1
    -h sum Xi

    """

    dim=2**N

    H=np.zeros(
        (dim,dim),
        dtype=complex
    )


    # interaction term
    for i in range(N-1):

        ops=[]

        for j in range(N):

            if j==i or j==i+1:
                ops.append(Z)
            else:
                ops.append(I)


        H += -J*kron_all(ops)



    # transverse field term

    for i in range(N):

        ops=[]

        for j in range(N):

            if j==i:
                ops.append(X)
            else:
                ops.append(I)


        H += -h*kron_all(ops)



    return H
