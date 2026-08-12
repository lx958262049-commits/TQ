import numpy as np


def random_spin_config(N):

    """
    Generate random spin configuration

    spin:
    0 or 1

    """

    return np.random.choice([-1, 1], size=N)

def wavefunction_sampling(
        psi,
        basis,
        num_samples
):


    weights=[]


    for s in basis:

        w=abs(psi(np.array(s)))**2

        weights.append(w)



    weights=np.array(weights)

    probabilities=weights/weights.sum()



    indices=np.random.choice(
        len(basis),
        size=num_samples,
        p=probabilities
    )


    samples=[]

    for i in indices:

        samples.append(
            basis[i]
        )


    return np.array(samples)



def generate_samples(
        N,
        num_samples
):

    samples=[]


    for _ in range(num_samples):

        s=random_spin_config(N)

        samples.append(s)


    return np.array(samples)


if __name__=="__main__":

    samples=generate_samples(
        N=8,
        num_samples=5
    )


    print(samples)
