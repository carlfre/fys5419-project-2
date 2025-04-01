from gates import CU1, hadamard, multi_qubit_gate, swap


import numpy as np


def qft(n_qubits: int) -> np.ndarray:
    H = hadamard()

    QFT = np.eye(2**n_qubits, dtype=complex)

    for i in range(n_qubits):
        Hi = multi_qubit_gate(H, i, n_qubits)
        QFT = Hi @ QFT
        for j in range(i+1, n_qubits):
            theta = 2 * np.pi / (2**(j - i +1))
            U = CU1(theta, j, i, n_qubits)
            QFT = U @ QFT

    # Reverse bit order
    for i in range(n_qubits // 2):
        QFT = swap(i, n_qubits - i - 1, n_qubits) @ QFT

    return QFT


if __name__ == "__main__":
    from linear_algebra import quantum_fourier_transform

    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    n_qubits = 5

    qft_linalg = quantum_fourier_transform(2**n_qubits)
    qft_matrix = qft(n_qubits)
    print(np.linalg.norm(qft_linalg - qft_matrix))
