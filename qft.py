from time import time

import numpy as np

from gates import CU1, hadamard, multi_qubit_gate, swap
from tensor_prod_stuff import apply_operator, do_the_swap
from tensor_prod_stuff import apply_CU1


# TODO: make qft efficient as well. Use ideas from iqft.
def qft(psi: np.ndarray) -> np.ndarray:

    n_qubits = round(np.log2(psi.shape[0]))

    if len(psi) != 2**n_qubits:
        raise ValueError("Input state must have length  a power of two.")

    H = hadamard()
    for i in range(n_qubits):
        psi = apply_operator(psi, H, i)
        for j in range(i + 1, n_qubits):
            theta = 2 * np.pi / (2 ** (j - i + 1))
            U = CU1(theta, j, i, n_qubits)
            psi = U @ psi

    for i in range(n_qubits // 2):
        psi = swap(i, n_qubits - i - 1, n_qubits) @ psi
    return psi


def inverse_qft(psi: np.ndarray) -> np.ndarray:

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


if __name__ == "__main__":
    from linear_algebra import quantum_fourier_transform

    N = 3

    qft_linalg = quantum_fourier_transform(2**N)

    ket0 = np.zeros(2**N, dtype=complex)
    ket0[7] = 2

    # print("ket0")
    # print(ket0)

    # print("QFT")
    ket0_F = qft(ket0)
    # print(ket0_F)

    # print("QFT old")
    QFT_mat = qft_old(N)
    ket0_F_old = qft_linalg @ ket0
    # print(ket0_F_old)

    # print("Inverse QFT")
    ket0_F_inv = inverse_qft(ket0_F)
    # print(ket0_F_inv)

    # print("Inverse QFT old")
    iQFT_mat = inverse_qft_old(N)
    ket0_F_inv_old = iQFT_mat @ ket0_F_old
    # print(ket0_F_inv_old)

    print(np.linalg.norm(ket0_F - ket0_F_old))
    print(np.linalg.norm(ket0_F_inv - ket0_F_inv_old))
    print(
        np.linalg.norm(ket0_F_inv - qft_linalg.conj().T @ ket0)
    )  # TODO: this turns out nonzero??  what is going on??

    # QFT = qft_old(N)

    # # iQFT = inverse_qft(N)

    # # print(np.linalg.norm(QFT @ iQFT - np.eye(16)))
    # print(type(QFT))
