from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse as sparse


def multi_kron(*args, type: Literal["sparse", "numpy"] = "sparse"):
    """
    Kronecker product of multiple sparse matrices
    """
    result = args[0]
    for matrix in args[1:]:
        if type == "numpy":
            result = np.kron(result, matrix)
        elif type == "sparse":
            result = sp.kron(result, matrix, format="csr")
        else:
            raise ValueError("Invalid type. Use 'sparse' or 'numpy'.")
    return result


def hadamard() -> sp.spmatrix:
    """
    Hadamard gate as sparse matrix
    """
    return sp.csr_matrix(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex))


def U1(theta) -> sp.spmatrix:
    """
    U1 gate as sparse matrix
    """
    return sp.csr_matrix(np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex))


def U_mult_a(a: int, N: int, n_qubits: int) -> sp.spmatrix:
    """Matrix for multiplication by a mod N.

    Specifically, for states |k> with k ∈ [0, N), U_mult_a |k> = |(a*k) mod N>.
    Matrix acts as identity on the rest of the computational basis states."""
    if N >= 2**n_qubits:
        raise ValueError("N must be strictly less than 2^n_qubits.")

    data = []
    rows = []
    cols = []

    # Handle multiplication modulo N
    for k in range(N):
        rows.append((a * k) % N)
        cols.append(k)
        data.append(1)

    # Handle identity for remaining indices
    for k in range(N, 2**n_qubits):
        rows.append(k)
        cols.append(k)
        data.append(1)

    return sp.csr_matrix(
        (data, (rows, cols)), shape=(2**n_qubits, 2**n_qubits), dtype=complex
    )


def apply_operator(
    psi: np.ndarray, operator: np.ndarray | sparse.spmatrix, qubit_index: int
) -> np.ndarray:
    """Applies a single-qubit operator to the specified qubit index."""

    if isinstance(operator, sparse.spmatrix):
        operator = operator.toarray()

    qubits = round(np.log2(psi.shape[0]))

    n_qubits_to_the_left = qubit_index
    n_qubits_operator = round(np.log2(operator.shape[0]))
    n_qubits_to_the_right = qubits - n_qubits_to_the_left - n_qubits_operator

    psi = psi.reshape(
        (2**n_qubits_to_the_left, 2**n_qubits_operator, 2**n_qubits_to_the_right)
    )

    Upsi = np.einsum("ij,kjl->kil", operator, psi)
    return Upsi.ravel()


def apply_controlled_operator(
    psi: np.ndarray,
    operator: np.ndarray | sparse.spmatrix,
    control_index: int,
):
    """Applies a controlled operator the the last qubits of the state vector psi.
    
    The operator is always applied to the last log2(operator.shape[0]) qubits. The operator
    is controlled on qubit control_index."""
    n = psi.shape[0]
    n_qubits_total = round(np.log2(n))
    n_qubits_operator = round(np.log2(operator.shape[0]))
    t = n_qubits_total - n_qubits_operator

    if control_index >= t:
        raise ValueError(
            "Control index must be less than the number of qubits in the state."
        )

    psi = psi.reshape((2**t, 2**n_qubits_operator))

    for i in range(2**t):
        if i >> (t - 1 - control_index) & 1:
            psi[i] = operator @ psi[i]

    return psi.ravel()


def apply_CU1(
    psi: np.ndarray, theta: float, control_index: int, target_index: int
) -> np.ndarray:
    """Applies a controlled U1 gate to psi.

    Note: Control and target indices can be interchanged - this follows from the definition of CU1. """
    # This implementation might take too much memory. But it's much faster. Consider changing back
    n_qubits = round(np.log2(psi.shape[0]))
    control_bit = 1 << (n_qubits - control_index - 1)
    target_bit = 1 << (n_qubits - target_index - 1)

    indices = np.arange(len(psi))
    mask = ((indices & control_bit) != 0) & ((indices & target_bit) != 0)
    psi[mask] *= np.exp(1j * theta)
    return psi


def do_the_swap(psi: np.ndarray, qubit_index1: int, qubit_index2: int) -> np.ndarray:
    """Applies a swap gate to state psi, which swaps qubit_index1 and qubit_index2"""
    n = psi.shape[0]
    n_qubits = round(np.log2(n))

    if qubit_index1 == qubit_index2:
        return psi

    if qubit_index1 >= n_qubits or qubit_index2 >= n_qubits:
        raise ValueError(
            "Qubit indices must be less than the number of qubits in the state."
        )

    bit1 = 1 << (n_qubits - qubit_index1 - 1)
    bit2 = 1 << (n_qubits - qubit_index2 - 1)

    for i in range(n):
        b1 = (i & bit1) >> (n_qubits - qubit_index1 - 1)
        b2 = (i & bit2) >> (n_qubits - qubit_index2 - 1)

        if b1 != b2:
            swapped_i = i ^ bit1 ^ bit2  # Flip both bits
            if i < swapped_i:  # Prevent double swap
                psi[i], psi[swapped_i] = psi[swapped_i], psi[i]

    return psi


if __name__ == "__main__":
    # Test identity and hadamard
    H = hadamard()
    print("Hadamard matrix:\n", H.toarray())

    # Test U1 gate
    u1 = U1(np.pi / 4)
    print("U1(π/4) matrix:\n", u1.toarray())

    # Test U_mult_a gate
    N = 7
    u_mult = U_mult_a(2, N, 3)  # Multiply by 2 modulo 7 in 3-qubit system
    print("U_a=2 gate (modulo 7) in 3-qubit system, shape:", u_mult.shape)
    print("First few entries of U_2 matrix:")
    print(u_mult.toarray()[:8, :8])
