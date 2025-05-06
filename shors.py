from math import ceil
from random import randint
from fractions import Fraction
from number_theory import  gcd
from phase_estimation import phase_estimation
from gates import U_mult_a, multi_kron
import numpy as np
import time

def find_order_qm(a: int, N: int) -> int:

    L = ceil(np.log2(N))

    t = 4 * L + 2
    Ua = U_mult_a(a, N, L)


    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    kets = [ket1] + (L - 1) * [ket0]
    u = multi_kron(*kets)

    phi_bin = phase_estimation(Ua, u, t)
    print(phi_bin)

    phi = 0
    for i, digit in enumerate(phi_bin[2:]):
        phi += int(digit) * 2**(-(i+1))

    print(phi)
    phi_frac = Fraction(phi).limit_denominator(N)
    r = phi_frac.denominator
    return r


def shors(N: int, max_iterations: int = 1000) -> int: 

    for _ in range(max_iterations):
        a = randint(2, N - 1)
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


N = 6
start = time.time()
factor = shors(N)
end = time.time()

print(f"\nSuccessfully factored {N} into: {factor, int(N/factor)} in {end - start:.8f} seconds.")

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