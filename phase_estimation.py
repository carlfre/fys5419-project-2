import numpy as np
import scipy.sparse as sp

from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from time import time

from qft import inverse_qft
from quantum_ops import (
    multi_kron,
    apply_operator,
    apply_controlled_operator,
    hadamard,
    U1,
)


def phase_estimation(
    U: sp.spmatrix, u: np.ndarray, t: int, n_shots: int, display_progress: bool = False
) -> list[str]:
    """Phase estimation algorithm.

    For a unitary operator U and an eigenvector u, this algorithm estimates the phase of the corresponding eigenvalue.

    Args:
        U (sp.spmatrix): Matrix
        u (np.ndarray): Eigenvector of U
        t (int): Number of qubits for phase estimation (ie. you only get t bits of precision).
        n_shots (int): Number of measurements to perform
        display_progress (bool, optional): If True, prints progress while running. Defaults to False.

    Returns:
        list[str]: List of binary representations of the estimated phases. Has n_shots elements.
    """
    H = hadamard()

    n = round(np.log2(U.shape[0]))

    ket0 = np.array([1, 0])
    ket0_tensor = multi_kron(*([ket0] * t), type="numpy")
    psi = np.kron(ket0_tensor, u)

    for i in range(t):
        if display_progress:
            print(f"applying H: {i} of {t}")
        psi = apply_operator(psi, H, i)
    Upow = U
    for i in range(t):
        if display_progress:
            print(f"applying U^({2**i}). {i} of {t}")
        psi = apply_controlled_operator(psi, Upow, t - i - 1)
        Upow = Upow @ Upow
    psi = psi.reshape((2**t, -1))

    start_time = time()
    for i in range(2**n):
        if display_progress:
            print(f"applying iQFT {i} of {2**n}")
        psi[:, i] = inverse_qft(psi[:, i])
    if display_progress:
        print("iQFT time", time() - start_time)

    probability_vector = np.abs(psi) ** 2

    probability_vector = probability_vector.reshape((-1, u.shape[0]))
    probability_vector = probability_vector.sum(axis=1)

    samples = np.random.choice(
        np.arange(probability_vector.shape[0]),
        p=probability_vector,
        size=n_shots,
        replace=True,
    )

    binary_representations = ["0." + np.binary_repr(num, width=t) for num in samples]
    if display_progress:
        print("phase estimation DONE.!")
    return binary_representations


def binary_to_phase(binary: str, num_counting_qubits: int) -> float:
    """Converts a binary fraction string to a phase in radians."""
    if not binary.startswith("0."):
        raise ValueError("Binary string must start with '0.'")

    binary_part = binary[2:]
    if len(binary_part) != num_counting_qubits:
        raise ValueError(
            f"Binary string length must match num_counting_qubits ({num_counting_qubits})"
        )

    b = int(binary_part, 2)
    phase = 2 * np.pi * b / (2**num_counting_qubits)
    return phase


if __name__ == "__main__":
    # https://en.wikipedia.org/wiki/Binary_number
    t = 14
    ratio = 1 / 7
    U = U1(2 * np.pi * ratio)
    u = np.array([0, 1])

    from time import time

    start = time()
    # phase = phase_estimation_new(U, u, t)
    phase_binary = phase_estimation(U, u, t, n_shots=1)[0]
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
