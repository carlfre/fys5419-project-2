from math import ceil
import random
from fractions import Fraction
from number_theory import  gcd
from phase_estimation import phase_estimation_old, phase_estimation
from gates import U_mult_a, multi_kron
import numpy as np
import time
from collections import Counter

from utils import write_csv

def phi_bin_to_order(phi_bin: str, N: int) -> int:
    phi = 0
    for i, digit in enumerate(phi_bin[2:]):
        phi += int(digit) * 2**(-(i+1))

    phi_frac = Fraction(phi).limit_denominator(N)
    return phi_frac.denominator


def find_order_qm(a: int, N: int, n_shots: int) -> int:

    if gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")

    L = ceil(np.log2(N))

    t = 4 * L + 2
    Ua = U_mult_a(a, N, L)


    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    # kets = L * [ket1]
    kets = (L-1) * [ket0] + [ket1]
    # kets = [ket1] + (L - 1) * [ket0]
    u = multi_kron(*kets, type='numpy')

    phi_binary_representations = phase_estimation(Ua, u, t, n_shots)
    r_estimates = [phi_bin_to_order(phi_bin, N) for phi_bin in phi_binary_representations]
    return r_estimates


def shors(N: int, max_iterations: int = 1000, n_shots_phase_estimation: int=1) -> int: 

    for _ in range(max_iterations):
        a = random.randint(2, N - 1)
        print("a", a)
        if gcd(a, N) > 1:
            return gcd(a, N)

        r_estimates = find_order_qm(a, N, n_shots=n_shots_phase_estimation)
        r = max(r_estimates)
        print("r", r)
        if r%2 == 0:
            gcd1 = gcd(a**(r//2) - 1, N)
            gcd2 = gcd(a**(r//2) + 1, N)

            if gcd1 != 1 and gcd1 != N:
                return gcd1
            if gcd2 != 1 and gcd2 != N:
                return gcd2
            
    raise ValueError("Failed to find non-trivial factors of N within the set iterations.")



def estimated_order_distribution(N: int, a: int, n_shots: int) -> Counter:
    """
    Compute the distribution of estimated orders for a given number of runs.
    """

    orders = find_order_qm(a, N, n_shots=n_shots)

    order_counts = Counter(orders)
    
    filename = f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}.csv"
    write_csv(order_counts, filename)
    print(f"Estimated order distribution saved to {filename}")
    return order_counts







def compare_qm_classical():
    from number_theory import find_order_classical, gcd

    N = 15
    a = 4

    coprime = []
    clasicals = []
    qms = []
    for i in range(1, 15):

        if gcd(i, 15) != 1:
            continue

        classical =  find_order_classical(i, N)
        qm = find_order_qm(i, N)
        print(i)
        print("classical", classical)
        print("qm", qm)
        coprime.append(i)
        clasicals.append(classical)
        qms.append(qm)

    for co, cl, qm in zip(coprime, clasicals, qms):
        print(f"i: {co}, classical: {cl}, qm: {qm}")


    # Ua = U_mult_a(a, N, 5)
    # print("Ua")
    # print(Ua)

    # Ua_np = Ua.toarray()

    # print(np.allclose(Ua_np.T @ Ua_np, np.eye(2**5)))

    # classical = find_order_classical(a, N)
    # qm = find_order_qm(a, N)
    # print("classical", classical)
    # print("qm", qm)


# main()

# print(f"\nSuccessfully factored {N} into: {factor, int(N/factor)} in {end - start:.8f} seconds.")

# r = 1
# while r % 2 != 0:
#     a = randint(2, N-1)
#     # a = 17
#     print("a", a)
#     if gcd(a, N) > 1:
#         return gcd(a, N)
    
#     r = find_order_classical(a, N)
#     # print(r)
#     # print(a, r)



# gcd1 = gcd(a**(r//2) - 1, N)
# gcd2 = gcd(a**(r//2) + 1, N)
# print("gcd1", gcd1)
# print("gcd2", gcd2)
# if gcd1 != 1:
#     return gcd1
# if gcd2 != 1:
#     return gcd2
# else:
#     raise ValueError("Failed to find non-trivial factors of N.")

if __name__ == "__main__":
    n_shots = 10_000
    N = 15
    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        order_dist = estimated_order_distribution(N, a, n_shots)
        print(order_dist)