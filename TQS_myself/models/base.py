from abc import ABC, abstractmethod


class NeuralQuantumState(ABC):

    """
    Neural Quantum State interface

    Any wavefunction model should implement:

    psi(s)

    log_psi(s)

    log_derivatives(s)

    """



    @abstractmethod
    def psi(self,s):
        pass



    @abstractmethod
    def log_psi(self,s):
        pass



    @abstractmethod
    def log_derivatives(self,s):
        pass

    @abstractmethod
    def parameters(self):
        pass