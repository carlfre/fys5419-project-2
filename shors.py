from math import ceil
import random
from fractions import Fraction
from number_theory import  gcd
from phase_estimation import phase_estimation_old, phase_estimation_new
from gates import U_mult_a, multi_kron
import numpy as np

def find_order_qm(a: int, N: int) -> int:

    if gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")

    L = ceil(np.log2(N))

    t = 4 * L + 2
    print("L, t", L, t)
    Ua = U_mult_a(a, N, L)
    print("Ua")
    print(Ua)


    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    # kets = L * [ket1]
    kets = (L-1) * [ket0] + [ket1]
    # kets = [ket1] + (L - 1) * [ket0]
    u = multi_kron(*kets, type='numpy')
    print("u")
    print(u)

    phi_bin = phase_estimation_new(Ua, u, t)
    print(phi_bin)

    phi = 0
    for i, digit in enumerate(phi_bin[2:]):
        phi += int(digit) * 2**(-(i+1))

    print("phi", phi)
    phi_frac = Fraction(phi).limit_denominator(N)
    r = phi_frac.denominator
    return r


def shors(N: int, max_iterations: int = 1000) -> int: 

    for _ in range(max_iterations):
        a = random.randint(2, N - 1)
        print("a", a)
        if gcd(a, N) > 1:
            return gcd(a, N)
        
        r = find_order_qm(a, N)
        print("r", r)
        if r%2 == 0:
            gcd1 = gcd(a**(r//2) - 1, N)
            gcd2 = gcd(a**(r//2) + 1, N)

            if gcd1 != 1 and gcd1 != N:
                return gcd1
            if gcd2 != 1 and gcd2 != N:
                return gcd2
            
    raise ValueError("Failed to find non-trivial factors of N within the set iterations.")
            



def compute_estimated_order_distribution(N: int, a: int, n_runs: int):
    """
    Compute the distribution of estimated orders for a given number of runs.
    """

    from collections import Counter
    import matplotlib.pyplot as plt

    orders = []
    for _ in range(n_runs):
        r = find_order_qm(a, N)
        orders.append(r)

    order_counts = Counter(orders)
    order_values = list(order_counts.keys())
    order_frequencies = list(order_counts.values())

    plt.bar(order_values, order_frequencies)
    plt.xlabel('Estimated Order')
    plt.ylabel('Frequency')
    plt.title(f'Estimated Order Distribution for a={a}, N={N}')
    plt.savefig(f'plots/est_order_dist_a={a}_N={N}.png')
    # plt.show()






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
    plot_estimated_order_distribution(15, 8, n_runs=200)