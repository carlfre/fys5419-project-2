from math import ceil
import random
from fractions import Fraction
from number_theory import gcd
from phase_estimation import phase_estimation
from quantum_ops import U_mult_a, multi_kron
import numpy as np
import time
from collections import Counter
from qiskit_implementations import phase_estimation_qiskit

from number_theory import find_order_classical, gcd


def phi_bin_to_order(phi_bin: str, N: int) -> int:
    """Converts a binary representation of a phase to an order using continued fractions.

    Args:
        phi_bin (str): phase in binary representation.
        N (int): Maximum denominator - the number we are factorizing is a suitable value.

    Returns:
        int: an estimated order.
    """
    phi = 0
    for i, digit in enumerate(phi_bin[2:]):
        phi += int(digit) * 2 ** (-(i + 1))

    phi_frac = Fraction(phi).limit_denominator(N)
    return phi_frac.denominator


def find_order_qm(
    a: int, N: int, n_shots: int, use_qiskit: bool = False
) -> dict[int, int]:
    """Quantum-mechanical order finding algorithm. Subroutine of Shor's algorithm.

    Finds the smallest integer r such that a^r = 1 mod N.

    Args:
        a (int): integer to find the order of
        N (int): Number we are factorizing
        n_shots (int): number of measurements/estimates to make.
        use_qiskit (bool, optional): If True, uses Qiskit implementation of phase estimation. Otherwise, uses our implementation.


    Returns:
        dict[int, int]: A dictionary mapping estimated orders to their counts.
    """

    if gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")

    L = ceil(np.log2(N))

    t = 4 * L + 2
    Ua = U_mult_a(a, N, L)

    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    kets = (L - 1) * [ket0] + [ket1]
    u = multi_kron(*kets, type="numpy")

    if use_qiskit:
        phi_binary_representations = phase_estimation_qiskit(Ua, u, t, n_shots)
        print("out of function")
    else:
        phi_binary_representations = phase_estimation(Ua, u, t, n_shots)
    order_estimates = {}
    for phi_bin, count in phi_binary_representations.items():
        order = phi_bin_to_order(phi_bin, N)
        if order not in order_estimates:
            order_estimates[order] = 0
        order_estimates[order] += count
    return order_estimates


def shors(
    N: int,
    max_iterations: int = 1000,
    n_shots_phase_estimation: int = 1,
    use_qiskit: bool = False,
) -> int:
    """Shor's algorithm for finding a factor of N.

    Args:
        N (int): Number to factorize.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
        n_shots_phase_estimation (int, optional): Number of measurements to make for each phase estimation. Defaults to 1.
        use_qiskit (bool, optional): If True, uses Qiskit implementation of phase estimation. Otherwise, uses our implementation

    Returns:
        int: A factor of N.
    """
    for _ in range(max_iterations):
        a = random.randint(2, N - 1)
        print(f"Candidate a={a}")
        if gcd(a, N) > 1:
            print(
                f"a={a} and N={N} have a common factor. Trivially return that factor."
            )
            print(f"GCD(a, N)={gcd(a, N)}")
            return gcd(a, N)

        r_estimates = find_order_qm(a, N, n_shots_phase_estimation, use_qiskit)
        r = max(r_estimates.keys())
        print(f"a={a} has estimated order r={r}.")
        if r % 2 == 0:
            print(f"r is even! Proceed to check for factors:")
            gcd1 = gcd(a ** (r // 2) - 1, N)
            gcd2 = gcd(a ** (r // 2) + 1, N)
            print(f"GCD(a^(r/2) - 1, N) = {gcd1}")
            print(f"GCD(a^(r/2) + 1, N) = {gcd2}")

            if gcd1 != 1 and gcd1 != N:
                print(f"Found non-trivial factor: {gcd1}")
                return gcd1
            if gcd2 != 1 and gcd2 != N:
                print(f"Found non-trivial factor: {gcd2}")
                return gcd2
            print("No non-trivial factors found.")

    raise ValueError(
        "Failed to find non-trivial factors of N within the set iterations."
    )


def compare_qm_classical():

    N = 15

    coprime = []
    clasicals = []
    qms = []
    for a in range(1, 15):

        if gcd(a, 15) != 1:
            continue

        classical = find_order_classical(a, N)
        qm = find_order_qm(a, N)
        print(a)
        print("classical", classical)
        print("qm", qm)
        coprime.append(a)
        clasicals.append(classical)
        qms.append(qm)

    for co, cl, qm in zip(coprime, clasicals, qms):
        print(f"i: {co}, classical: {cl}, qm: {qm}")


