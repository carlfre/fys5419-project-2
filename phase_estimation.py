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
    

def phase_estimation_new(U: sp.spmatrix, u: np.ndarray, t: int) -> sp.spmatrix:
    I = identity(2)
    H = hadamard()
    

    n = round(np.log2(U.shape[0]))
    n_qubits_tot = t + n

    ket0 = np.array([1, 0])
    ket0_tensor = multi_kron(*([ket0] * t), type='numpy')
    psi = np.kron(ket0_tensor, u)


    # print("t", t)

    for i in range(t):
        print(f"applying H: {i} of {t}")
        psi = apply_operator(psi, H, i)
        # Hi = multi_qubit_gate(H, i, n_qubits_tot)
        # psi = Hi @ psi
    
    # print("done w H")
    Upow = U
    for i in range(t):
        print(f"applying U^({2**i}). {i} of {t}")
        # print("computing Upow")
        # Upow = mat_pow(U, 2**i)
        # print(i, "applying controlled U")
        psi = apply_controlled_operator(psi, Upow, t-i-1)
        Upow = Upow @ Upow
    # print("done W CU")
    psi = psi.reshape((2**t, -1))

    # print("reshaped. Now doing iQFT")


    start_time = time()
    # This implementation maybe faster but uses more memory.
    # results = Parallel(n_jobs=-1)(
    #     delayed(inverse_qft)(psi[:, i]) for i in range(psi.shape[1])
    # )
    # for i, res in enumerate(results):
    #     psi[:, i] = res

    # This implementation uses less memory but is slower.
    for i in range(2**n):
        print(f"applying iQFT {i} of {2**n}")
        psi[:, i] = inverse_qft(psi[:, i])
    print("iQFT time", time() - start_time)
    
    # print("done w iQFT")


    
    probability_vector = np.abs(psi)**2

    probability_vector = probability_vector.reshape((-1, u.shape[0]))
    probability_vector = probability_vector.sum(axis=1)

    # cum_prob = np.cumsum(probability_vector)
    # uniform = np.random.uniform()
    num = np.random.choice(np.arange(probability_vector.shape[0]), p=probability_vector)

    # num = np.argmax(np.abs(probability_vector))

    binary = np.binary_repr(num, width = t )
    print("phase estimation DONE.!")
    return "0." + binary




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


if __name__ == "__main__":
    # https://en.wikipedia.org/wiki/Binary_number
    t = 14
    ratio = 1/7
    U = U1(2 * np.pi * ratio)
    u = np.array([0, 1])

    from time import time
    start = time()
    phase = phase_estimation_new(U, u, t)
    print(phase, "time", time() - start)
    # start = time()
    # phase = phase_estimation_old(U, u, t)
    # print(phase, "time", time() - start)




# print(probability_vector)
# probability_vector = 
# np.sum(probability_vector, axis=)

# num = np.argmax(np.abs(phase_estimation_vector))

# print(np.binary_repr(num, width = t + u.shape[0] ))


# print(phase_estimation_vector)

