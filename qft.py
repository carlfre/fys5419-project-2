from time import time

import numpy as np

from quantum_ops import  apply_operator, hadamard
from quantum_ops import do_the_swap
from quantum_ops import apply_CU1
from linear_algebra import dft



# TODO: make qft efficient as well. Use ideas from iqft.
def qft(psi: np.ndarray) -> np.ndarray:
    """Computes the Quantum Fourier Transform (QFT) of a state vector psi.

    Args:
        psi (np.ndarray): vector to compute the Fourier transform of.

    Returns:
        np.ndarray: the fourier transformed vector.
    """

    n_qubits = round(np.log2(psi.shape[0]))

    if len(psi) != 2**n_qubits:
        raise ValueError("Input state must have length  a power of two.")

    H = hadamard()
    for i in range(n_qubits):
        psi = apply_operator(psi, H, i)
        for j in range(i + 1, n_qubits):
            theta = 2 * np.pi / (2 ** (j - i + 1))
            psi = apply_CU1(psi, theta, j, i)

    for i in range(n_qubits // 2):
        psi = do_the_swap(psi, i, n_qubits - i - 1)
    return psi


def inverse_qft(psi: np.ndarray) -> np.ndarray:
    """Computes the Inverse Quantum Fourier Transform (IQFT) of a state vector psi.

    Args:
        psi (np.ndarray): vector to compute the inverse Fourier transform of.

    Returns:
        np.ndarray: the inverse fourier transformed vector.
    """

    n_qubits = round(np.log2(psi.shape[0]))

    if len(psi) != 2**n_qubits:
        raise ValueError("Input state must have length a power of two.")

    H = hadamard()

    for i in range(n_qubits // 2 - 1, -1, -1):
        psi = do_the_swap(psi, i, n_qubits - i - 1)

    for i in range(n_qubits - 1, -1, -1):
        psi = apply_operator(psi, H, i)
        for j in range(i - 1, -1, -1):
            theta = -2 * np.pi / (2 ** (i - j + 1))
            psi = apply_CU1(psi, theta, j, i)

    return psi


def example_run():

    N = 5
    psi = np.random.rand(2**N) + 1j * np.random.rand(2**N)
    psi /= np.linalg.norm(psi) 

    dft_mat = dft(2**N)

    print("Verify that QFT acts the same as DFT matrix:")
    print("||qft(psi) - dft(psi)|| =", np.linalg.norm(qft(psi.copy()) - dft_mat @ psi.copy()) )
    print("||inverse_qft(psi) - dft^(-1)(psi)|| =", np.linalg.norm(inverse_qft(psi.copy()) - np.linalg.pinv(dft_mat) @ psi.copy()))


if __name__ == "__main__":
    example_run()
