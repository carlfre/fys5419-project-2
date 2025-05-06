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


def inverse_qft(n_qubits: int) -> np.ndarray:
    H = hadamard()

    iQFT = np.eye(2**n_qubits, dtype=complex)

    for i in range(n_qubits//2 -1, -1, -1):
        iQFT = swap(i, n_qubits - i - 1, n_qubits) @ iQFT
        
    
    for i in range(n_qubits - 1, -1, -1):
        Hi = multi_qubit_gate(H, i, n_qubits)
        iQFT = Hi @ iQFT
        for j in range(i - 1, -1, -1):
            theta = -2 * np.pi / (2**(i - j + 1))
            U = CU1(theta, j, i, n_qubits)
            iQFT = U @ iQFT

    return iQFT


if __name__ == "__main__":
    from linear_algebra import quantum_fourier_transform


    N = 2
    QFT = qft(N)



    # iQFT = inverse_qft(N)

    # print(np.linalg.norm(QFT @ iQFT - np.eye(16)))
    print(QFT)

