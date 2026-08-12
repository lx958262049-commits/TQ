import numpy as np


from models.base import NeuralQuantumState
class RBM(NeuralQuantumState):


    def __init__(
        self,
        N_visible,
        N_hidden
    ):

        self.Nv=N_visible
        self.Nh=N_hidden


        # parameters

        self.a = 0.01 * np.random.randn(
            N_visible
        )

        self.b = 0.01 * np.random.randn(
            N_hidden
        )


        self.W=0.01*np.random.randn(
            N_visible,
            N_hidden
        )


    def log_psi(self,v):


        """
        log ψ(v)

        """


        hidden_input = (
            self.b
            +
            np.dot(v,self.W)
        )


        result = (
            np.dot(self.a,v)
            +
            np.sum(
                np.logaddexp(
                    0,
                    hidden_input
                )
            )
        )


        return result



    def psi(self,v):


        return np.exp(
            self.log_psi(v)
        )

    def parameters(self):
        return [
            self.a,
            self.b,
            self.W
        ]

    def log_derivatives(self, v):
        """
        Calculate O_theta

        O_theta =
        d log psi / d theta

        """

        hidden_input = (
                self.b
                +
                np.dot(v, self.W)
        )

        sigmoid = 1 / (1 + np.exp(-hidden_input))

        O_a = v

        O_b = sigmoid

        O_W = np.outer(
            v,
            sigmoid
        )

        return O_a, O_b, O_W

