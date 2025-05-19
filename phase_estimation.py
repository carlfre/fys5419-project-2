import numpy as np
import scipy.sparse as sp

from gates import multi_kron, identity, hadamard, controlled_U_gate, U1, multi_qubit_gate
from qft import qft_old, inverse_qft_old, inverse_qft

from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from time import time





from tensor_prod_stuff import apply_operator, apply_controlled_operator

def mat_pow(U: sp.spmatrix, n: int) -> sp.spmatrix:
    """Computes U^n efficiently."""


    if n == 0:
        return identity(U.shape[0])
    elif n == 1:
        return U
    
    half = mat_pow(U, n // 2)

    if n % 2 == 0:
        return half @ half
    else:
        return half @ half @ U
    

def phase_estimation_new(U: sp.spmatrix, u: np.ndarray, t: int, n_shots: int, display_progress: bool = False) -> sp.spmatrix:
    I = identity(2)
    H = hadamard()
    

    n = round(np.log2(U.shape[0]))

    ket0 = np.array([1, 0])
    ket0_tensor = multi_kron(*([ket0] * t), type='numpy')
    psi = np.kron(ket0_tensor, u)



    for i in range(t):
        if display_progress:
            print(f"applying H: {i} of {t}")
        psi = apply_operator(psi, H, i)
    Upow = U
    for i in range(t):
        if display_progress:
            print(f"applying U^({2**i}). {i} of {t}")
        psi = apply_controlled_operator(psi, Upow, t-i-1)
        Upow = Upow @ Upow
    psi = psi.reshape((2**t, -1))

    start_time = time()
    for i in range(2**n):
        if display_progress:
            print(f"applying iQFT {i} of {2**n}")
        psi[:, i] = inverse_qft(psi[:, i])
    if display_progress:
        print("iQFT time", time() - start_time)

    
    probability_vector = np.abs(psi)**2

    probability_vector = probability_vector.reshape((-1, u.shape[0]))
    probability_vector = probability_vector.sum(axis=1)

    samples = np.random.choice(np.arange(probability_vector.shape[0]), p=probability_vector, size=n_shots, replace=True)

    binary_representations = ["0." + np.binary_repr(num, width = t) for num in samples]
    if display_progress:
        print("phase estimation DONE.!")
    return binary_representations




#### old implementation. Computes the whole matrix  - too memory intensive. 

def phase_estimation_gates(U: np.ndarray, t: int) -> np.ndarray:
    I = identity(2).toarray()
    H = hadamard().toarray()
    

    n = round(np.log2(U.shape[0]))
    # n_qubits = t + n

    gates = [H] * t + [I] * n
    circuit = multi_kron(*gates, type='numpy')

    Upow = U
    for i in range(t):
        CU = controlled_U_gate(t, Upow, t - i - 1).toarray()
        circuit = CU @ circuit
        if i < t - 1:
            Upow = Upow @ Upow


    iQFT = inverse_qft_old(t)
    gates = [iQFT] + [I] * n
    circuit = multi_kron(*gates, type='numpy') @ circuit

    return circuit

    # vec = circuit @ u

    # print(vec)




def phase_estimation_old(U: np.ndarray, u: np.ndarray, t: int) -> str:
    gate = phase_estimation_gates(U, t)
    ket0 = np.array([1, 0])
    ket0_tensor = multi_kron(*([ket0] * t), type='numpy')
    initial_state = np.kron(ket0_tensor, u)
    phase_estimation_vector = gate @ initial_state
    probability_vector = np.abs(phase_estimation_vector)**2

    probability_vector = probability_vector.reshape((-1, u.shape[0]))
    probability_vector = probability_vector.sum(axis=1)

    num = np.argmax(np.abs(probability_vector))

    binary = np.binary_repr(num, width = t )
    return "0." + binary

def binary_to_phase(binary: str, num_counting_qubits: int) -> float:
    """Converts a binary fraction string to a phase in radians."""
    if not binary.startswith("0."):
        raise ValueError("Binary string must start with '0.'")
    
    binary_part = binary[2:]
    if len(binary_part) != num_counting_qubits:
        raise ValueError(f"Binary string length must match num_counting_qubits ({num_counting_qubits})")
    
    b = int(binary_part, 2)
    phase = 2 * np.pi * b / (2 ** num_counting_qubits)
    return phase

if __name__ == "__main__":
    # https://en.wikipedia.org/wiki/Binary_number
    t = 14
    ratio = 1/7
    U = U1(2 * np.pi * ratio)
    u = np.array([0, 1])

    from time import time
    start = time()
    # phase = phase_estimation_new(U, u, t)
    phase_binary = phase_estimation_new(U, u, t, n_shots=1)[0]
    print(f"Estimated Phase (binary fraction): {phase_binary}")
    phase_radians = binary_to_phase(phase_binary, t)
    print(f"Estimated Phase (radians): {phase_radians}")
    print("time", time() - start)
    # start = time()
    # phase = phase_estimation_old(U, u, t)
    # print(phase, "time", time() - start)




# print(probability_vector)
# probability_vector = 
# np.sum(probability_vector, axis=)

# num = np.argmax(np.abs(phase_estimation_vector))

# print(np.binary_repr(num, width = t + u.shape[0] ))


# print(phase_estimation_vector)

