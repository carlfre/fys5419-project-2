# print(fractions.Fraction(np.pi).limit_denominator(50))


# num = 0.001011

# print(bin(0.333333))


# # print(float(num, 2))



import numpy as np
import scipy.sparse as sparse


def apply_operator(psi: np.ndarray, operator: np.ndarray | sparse.spmatrix, qubit_index: int) -> np.ndarray:

    if isinstance(operator, sparse.spmatrix):
        operator = operator.toarray()



    qubits = round(np.log2(psi.shape[0]))

    n_qubits_to_the_left = qubit_index
    n_qubits_operator = round(np.log2(operator.shape[0]))
    n_qubits_to_the_right = qubits - n_qubits_to_the_left - n_qubits_operator

    psi = psi.reshape((2**n_qubits_to_the_left, 2**n_qubits_operator, 2**n_qubits_to_the_right))

    Upsi = np.einsum('ij,kjl->kil', operator, psi)
    return Upsi.ravel()


def apply_controlled_operator(psi: np.ndarray, operator: np.ndarray | sparse.spmatrix, control_index: int, ):
    n = psi.shape[0]
    n_qubits_total = round(np.log2(n))
    n_qubits_operator = round(np.log2(operator.shape[0]))
    t = n_qubits_total - n_qubits_operator
    
    if control_index >= t:
        raise ValueError("Control index must be less than the number of qubits in the state.")
    
    psi = psi.reshape((2**t, 2**n_qubits_operator))
    # psi_res = np.zeros_like(psi)

    for i in range(2**t):
        binary = np.binary_repr(i, width=t)
        if binary[control_index] == '1':
            psi[i] = operator @ psi[i]
        else:
            psi[i] = psi[i]
    
    return psi.ravel() #psi_res.ravel()


    raise NotImplementedError("This function is not implemented yet.")



# def apply_CU1(psi: np.ndarray, theta: float, control_index: int, target_index: int) -> np.ndarray:
#     n = psi.shape[0]
#     n_qubits = round(np.log2(n))
    
#     control_bit = 1 << (n_qubits - control_index - 1)
#     target_bit = 1 << (n_qubits - target_index - 1)
    
#     phase = np.exp(1j * theta)
#     for i in range(n):
#         if (i & control_bit) and (i & target_bit):
#             psi[i] *= phase
#     return psi

def apply_CU1(psi: np.ndarray, theta: float, control_index: int, target_index: int) -> np.ndarray:
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
        raise ValueError("Qubit indices must be less than the number of qubits in the state.")

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
    from gates import swap

    n_qubits = 7


    ind1 = 5
    ind2 = 2
    psi = np.random.rand(2**n_qubits)

    swap1 = swap(ind1, ind2, n_qubits) @ psi
    swap2 = do_the_swap(psi, ind1, ind2)

    print("swap1", swap1)
    print("swap2", swap2)
    print("swap1 == swap2", np.allclose(swap1, swap2))
    print(np.linalg.norm(swap1 - swap2))