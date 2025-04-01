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
        print("applying Hadamard to", i, "!")
        for j in range(i+1, n_qubits):
            theta = 2 * np.pi / (2**(j - i ))
            print("applying rotation", theta/np.pi, "to:", i, j)
            U = CU1(theta, j, i, n_qubits)
            QFT = U @ QFT
    return QFT




# TEMPORARY - should probably be removed
def bit_reversal_permutation(n_qubits: int) -> np.ndarray:
    """
    Bit reversal permutation matrix
    """
    n = 2**n_qubits
    perm = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        # Reverse the bits of i
        reversed_i = int(bin(i)[2:].zfill(n_qubits)[::-1], 2)
        print(i, reversed_i)
        perm[i, reversed_i] = 1
    
    return perm


if __name__ == "__main__":
    from linear_algebra import quantum_fourier_transform

    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    qft_linalg = quantum_fourier_transform(8)

    n_qubits = 3
    qft_matrix = qft(n_qubits)

    P = bit_reversal_permutation(3)

    print( qft_linalg )
    print()
    print(qft_matrix)



A = CU1(np.pi/2, 0, 2, 4)
# print(A)
# B = alt_CU1(np.pi/2, 0, 1, 3)
# print(B)
# print(np.linalg.norm(A - B))




print(P)