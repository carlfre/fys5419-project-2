from typing import Literal

import numpy as np
import scipy.sparse as sp


def multi_kron(*args, type: Literal['sparse', 'numpy'] = 'sparse'):
    """
    Kronecker product of multiple sparse matrices
    """
    result = args[0]
    for matrix in args[1:]:
        if type == 'numpy':
            result = np.kron(result, matrix)
        elif type == 'sparse':
            result = sp.kron(result, matrix, format='csr')
        else:
            raise ValueError("Invalid type. Use 'sparse' or 'numpy'.")
    return result


def identity(n: int = 2) -> sp.spmatrix:
    """
    Identity matrix as sparse matrix
    """
    return sp.eye(n, dtype=complex, format='csr')


def multi_qubit_gate(single_qubit_gate: sp.spmatrix, index: int, n_qubits: int) -> sp.spmatrix:
    """
    Create a multi-qubit gate from a single qubit gate using sparse matrices
    """
    I = identity(2)

    gates = [I for _ in range(n_qubits)]
    gates[index] = single_qubit_gate
    return multi_kron(*gates)
    

def hadamard() -> sp.spmatrix:
    """
    Hadamard gate as sparse matrix
    """
    return sp.csr_matrix(1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex))


def U1(theta) -> sp.spmatrix:
    """
    U1 gate as sparse matrix
    """
    return sp.csr_matrix(np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex))


def swap(index1: int, index2: int, n_qubits: int) -> sp.spmatrix:
    """
    Swap gate as sparse matrix
    """
    data = []
    rows = []
    cols = []

    for i in range(2**n_qubits):
        bin_i = np.binary_repr(i, width=n_qubits).zfill(n_qubits)

        if (bin_i[index1] == bin_i[index2]):
            rows.append(i)
            cols.append(i)
            data.append(1)
        else:
            # Swap the bits
            bin_copy = list(bin_i)
            bin_copy[index1] = bin_i[index2]
            bin_copy[index2] = bin_i[index1]
            bin_copy = ''.join(bin_copy)
            swap_i = int(bin_copy, 2)
            
            rows.append(i)
            cols.append(swap_i)
            data.append(1)

    return sp.csr_matrix((data, (rows, cols)), shape=(2**n_qubits, 2**n_qubits), dtype=complex)


def CU1(theta: float, control_index: int, target_index: int, n_qubits: int) -> sp.spmatrix:
    diag = np.ones(2**n_qubits, dtype=complex)
    
    for i in range(2**n_qubits):
        bin_i = np.binary_repr(i, width=n_qubits)
        if bin_i[control_index] == '1' and bin_i[target_index] == '1':
            diag[i] = np.exp(1j * theta)
    
    return sp.diags(diag, format='csr')


def controlled_U_gate(t: int, U: sp.spmatrix, control_index: int) -> sp.spmatrix:
    n = U.shape[0]
    m = round(np.log2(n))
    n_qubits = t + m
    

    data = []
    rows = []
    cols = []
    
    for i in range(2**t):
        binary = np.binary_repr(i, width=t)
        start_of_block = int(binary + '0' * m, 2)
        
        if binary[control_index] == '1':
            # Add U block
            for i_row, i_col in zip(*U.nonzero()):
                rows.append(start_of_block + i_row)
                cols.append(start_of_block + i_col)
                data.append(U[i_row, i_col])
        else:
            # Add identity block
            for i_row in range(n):
                rows.append(start_of_block + i_row)
                cols.append(start_of_block + i_row)
                data.append(1)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(2**n_qubits, 2**n_qubits), dtype=complex)


def U_mult_a(a: int, N: int, n_qubits: int) -> sp.spmatrix:
    if N >= 2**n_qubits:
        raise ValueError("N must be strictly less than 2^n_qubits.")

    data = []
    rows = []
    cols = []
    
    # Handle multiplication modulo N
    for k in range(N):
        print("k, mod:", k, (a*k) % N)
        rows.append((a*k) % N)
        cols.append(k)
        data.append(1)
    
    # Handle identity for remaining indices
    for k in range(N, 2**n_qubits):
        rows.append(k)
        cols.append(k)
        data.append(1)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(2**n_qubits, 2**n_qubits), dtype=complex)



if __name__ == "__main__":
    # Test identity and hadamard
    I = identity()
    H = hadamard()
    print("Identity matrix:\n", I.toarray())
    print("Hadamard matrix:\n", H.toarray())

    # Test multi-qubit gate
    H_qubit1 = multi_qubit_gate(H, 1, 3)  # Hadamard on qubit 1 in 3-qubit system
    print("Hadamard on middle qubit in 3-qubit system, shape:", H_qubit1.shape)

    # Test U1 gate
    u1 = U1(np.pi/4)
    print("U1(π/4) matrix:\n", u1.toarray())

    # Test SWAP gate
    swap_01 = swap(0, 1, 2)  # Swap qubits 0 and 1 in 2-qubit system
    print("SWAP gate for qubits 0 and 1:\n", swap_01.toarray())

    # Test CU1 gate
    cu1 = CU1(np.pi/2, 0, 1, 2)  # Control qubit 0, target qubit 1 in 2-qubit system
    print("CU1(π/2) gate:\n", cu1.toarray())

    # Test controlled-U gate
    control_H = controlled_U_gate(2, H, 0)  # Control qubit 0, apply H on remaining qubit
    print("Controlled-H gate shape:", control_H.shape)

    # Test U_mult_a gate
    N = 7
    u_mult = U_mult_a(2, N, 3)  # Multiply by 2 modulo 7 in 3-qubit system
    print("U_a=2 gate (modulo 7) in 3-qubit system, shape:", u_mult.shape)
    print("First few entries of U_a=2 matrix:")
    print(u_mult.toarray()[:8, :8])