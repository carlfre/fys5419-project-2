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
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


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


def controlled_U_gate(t: int, U: np.ndarray, control_index: int) -> np.ndarray:


    n = U.shape[0]
    m = round(np.log2(n))
    


    n_qubits = t + m


    mat = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)



    for i in range(2**t):
        binary = np.binary_repr(i, width=t)

        start_of_block = binary + '0' * m
        end_of_block = binary + '1' * m

        start_of_block = int(start_of_block, 2)
        end_of_block = int(end_of_block, 2)
        # print(start_of_block, end_of_block)
        if binary[control_index] == '1':
            mat[start_of_block:end_of_block + 1, start_of_block:end_of_block + 1] = U

        elif binary[control_index] == '0':
            mat[start_of_block:end_of_block + 1, start_of_block:end_of_block + 1] = identity(n)
        else: 
            raise ValueError("Control index must be either 0 or 1.")

    
    return mat



def U_mult_a(a: int, N: int, n_qubits: int) -> np.ndarray:

    if N >= 2**n_qubits:
        raise ValueError("N must be strictly less than 2^n_qubits.")

    U = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for k in range(N):
        U[(a*k) % N, k] = 1

    for k in range(N+1, 2**n_qubits):
        U[k, k] = 1

    return U





# np.set_printoptions(precision=3, suppress=True, linewidth=200)

# t = 2

# U = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
# control_index = 1

# print(controlled_U_gate(t, U, control_index))