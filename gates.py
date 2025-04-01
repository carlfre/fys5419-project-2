import numpy as np


def multi_kron(*args):
    """
    Kronecker product of multiple matrices
    """
    result = args[0]
    for matrix in args[1:]:
        result = np.kron(result, matrix)
    return result


def identity(n: int = 2) -> np.ndarray:
    """
    Identity matrix
    """
    return np.eye(n, dtype=complex)


def multi_qubit_gate(single_qubit_gate: np.ndarray, index: int, n_qubits: int) -> np.ndarray:
    """
    Create a multi-qubit gate from a single qubit gate
    """
    I = identity(2)

    gates = [I for _ in range(n_qubits)]
    gates[index] = single_qubit_gate
    return multi_kron(*gates)
    

def hadamard():
    """
    Hadamard gate
    """
    return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)


def U1(theta):
    """
    U1 gate
    """
    return np.array([[1, 0], [0, np.exp(1j * theta)]], type=complex)


def swap(index1: int, index2: int, n_qubits: int) -> np.ndarray:
    """
    Swap gate
    """

    SWAP = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for i in range(2**n_qubits):
        # Convert i to binary representation
        bin_i = np.binary_repr(i, width=n_qubits).zfill(n_qubits)

        if (bin_i[index1] == bin_i[index2]):
            SWAP[i, i] = 1
        else:
            # Swap the bits!
            bin_copy = list(bin_i)

            bin_copy[index1] = bin_i[index2]
            bin_copy[index2] = bin_i[index1]
            # Convert to binary string
            bin_copy = ''.join(bin_copy)
            # Convert back to decimal
            swap_i = int(bin_copy, 2)
            SWAP[swap_i, i] = 1
            SWAP[i, swap_i] = 1

    return SWAP


def CU1(theta: float, control_index: int, target_index: int, n_qubits: int) -> np.ndarray:

    diag = np.ones((2,)* n_qubits, dtype=complex)

    index_slice = tuple(1 if i in (control_index, target_index) else slice(None) for i in range(n_qubits))
    diag[index_slice] *= np.exp(1j * theta)

    diag = diag.flatten()
    return np.diag(diag)



def qft(n_qubits: int) -> np.ndarray:
    H = hadamard()

    QFT = np.eye(2**n_qubits, dtype=complex)

    for i in range(n_qubits):
        Hi = multi_qubit_gate(H, i, n_qubits)
        QFT = Hi @ QFT
        # print("applying Hadamard to", i, "!")
        # print(QFT)
        for j in range(i+1, n_qubits):
            theta = 2 * np.pi / (2**(j - i +1))
            # print("applying rotation", theta/np.pi, "to:", i, j)
            U = CU1(theta, j, i, n_qubits)
            QFT = U @ QFT

    # Reverse bit order
    for i in range(n_qubits // 2):
        QFT = swap(i, n_qubits - i - 1, n_qubits) @ QFT
    
    return QFT







if __name__ == "__main__":
    from linear_algebra import quantum_fourier_transform

    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    n_qubits = 10

    qft_linalg = quantum_fourier_transform(2**n_qubits)
    qft_matrix = qft(n_qubits)
    print(np.linalg.norm(qft_linalg - qft_matrix))



