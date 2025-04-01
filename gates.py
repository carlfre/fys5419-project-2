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
        bin_i = np.binary_repr(i, width=n_qubits).zfill(n_qubits)

        if (bin_i[index1] == bin_i[index2]):
            SWAP[i, i] = 1
        else:
            # We will now swap the bits! 
            # We here find the index of the vector to swap with: 
            bin_copy = list(bin_i)
            bin_copy[index1] = bin_i[index2]
            bin_copy[index2] = bin_i[index1]

            # Convert to string
            bin_copy = ''.join(bin_copy)

            # Convert back to decimal
            swap_i = int(bin_copy, 2)

            # Perform the swap
            SWAP[swap_i, i] = 1
            SWAP[i, swap_i] = 1

    return SWAP


def CU1(theta: float, control_index: int, target_index: int, n_qubits: int) -> np.ndarray:

    diag = np.ones((2,)* n_qubits, dtype=complex)

    index_slice = tuple(1 if i in (control_index, target_index) else slice(None) for i in range(n_qubits))
    diag[index_slice] *= np.exp(1j * theta)

    diag = diag.flatten()
    return np.diag(diag)






