import numpy as np

from gates import multi_kron, identity, hadamard, controlled_U_gate, U1
from qft import qft, inverse_qft

def mat_pow(U: np.ndarray, n: int) -> np.ndarray:


    if n == 0:
        return identity(U.shape[0])
    elif n == 1:
        return U
    
    half = mat_pow(U, n // 2)

    if n % 2 == 0:
        return half @ half
    else:
        return half @ half @ U
    


def phase_estimation_gate(U: np.ndarray, t: int) -> np.ndarray:
    I = identity(2)
    H = hadamard()
    

    n = round(np.log2(U.shape[0]))
    # n_qubits = t + n

    gates = [H] * t + [I] * n
    circuit = multi_kron(*gates)


    for i in range(t):
        Upow = mat_pow(U, 2**i)
        CU = controlled_U_gate(t, Upow, t - i - 1)
        # print(CU.shape)
        # print(circuit.shape, "circuit")
        circuit = CU @ circuit

    iQFT = inverse_qft(t)
    gates = [iQFT] + [I] * n
    circuit = multi_kron(*gates) @ circuit

    return circuit

    # vec = circuit @ u

    # print(vec)




def phase_estimation(U: np.ndarray, u: np.ndarray, t: int) -> str:
    gate = phase_estimation_gate(U, t)
    ket0 = np.array([1, 0])
    ket0_tensor = multi_kron(*([ket0] * t))
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
    t = 8
    ratio = 1/5
    U = U1(2 * np.pi * ratio)
    u = np.array([0, 1])
    phase = phase_estimation(U, u, t)
    print(phase)




# print(probability_vector)
# probability_vector = 
# np.sum(probability_vector, axis=)

# num = np.argmax(np.abs(phase_estimation_vector))

# print(np.binary_repr(num, width = t + u.shape[0] ))


# print(phase_estimation_vector)

