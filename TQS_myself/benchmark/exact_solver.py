import numpy as np


def exact_ground_state(H):

    """
    Exact diagonalization

    H|psi>=E|psi|

    """

    energies, states = np.linalg.eigh(H)


    ground_energy = energies[0]

    ground_state = states[:,0]


    return ground_energy, ground_state



if __name__=="__main__":


    from physics.ising import transverse_field_ising


    N=8

    hamiltonian = transverse_field_ising(
        N=N,
        J=1,
        h=1
    )


    E,psi = exact_ground_state(
        hamiltonian
    )


    print("System size:",N)

    print(
        "Ground energy:",
        E
    )


    print(
        "Norm:",
        np.linalg.norm(psi)
    )
