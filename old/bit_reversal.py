import numpy as np

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
        perm[i, reversed_i] = 1
    
    return perm


