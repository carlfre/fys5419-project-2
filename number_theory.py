from random import randint


def gcd(n: int, m: int) -> int:
    """Returns greatest common divisor of n and m.

    Uses Euclidean algorithm.

    Args:
        n (int): Positive integer
        m (int): Positive integer

    Returns:
        int: greatest common divisor.
    """
    if n < 0 or m < 0:
        raise ValueError("gcd() only accepts non-negative integers")

    m, n = max(n, m), min(n, m)
    if n == 0:
        return m
    if m % n == 0:
        return n
    return gcd(n, m % n)


def find_order_classical(a: int, N: int) -> int:
    """Classical algorithm for finding the order of a mod N."""
    

    if gcd(a, N) != 1:
        raise ValueError(f"a must be coprime to N. a: {a}, N: {N}")

    
    r = 1
    apow = a % N
    while apow != 1:
        apow = (apow * a) % N
        r += 1
        print(apow, r)
    return r


def factorize_classical(N: int, max_iterations: int = 1000) -> int:
    """Classical algorithm for factorizing N, using order finding. 
    
    Shor's algorithm is based on this one, but uses a quantum subroutine for finding the order instead."""

    for _ in range(max_iterations):
        a = randint(2, N - 1)
        if gcd(a, N) > 1:
            return gcd(a, N)
        
        r = find_order_classical(a, N)
        if r%2 == 0:
            gcd1 = gcd(a**(r//2) - 1, N)
            gcd2 = gcd(a**(r//2) + 1, N)

            if gcd1 != 1 and gcd1 != N:
                return gcd1
            if gcd2 != 1 and gcd2 != N:
                return gcd2
            
    raise ValueError("Failed to find non-trivial factors of N within the set iterations.")

def is_prime(n: int) -> bool:
    """Checks if a number is prime.
    
    n is assumed to be a positive integer."""

    if n < 0:
        raise ValueError("is_prime() only accepts non-negative integers")

    if n < 2:
        return False
    
    for i in range(2, int(n**0.5) + 1):
        if n%i == 0:
            return False
    return True

def simplify(numerator: int, denominator: int) -> tuple[int, int]:
    """Simplifies a fraction as much as possible.
    
    eg. 
    * 4/8 -> 1/2
    * 6/9 -> 2/3
    * 0/8 -> 0/1
    """
    gcd_val = gcd(numerator, denominator)
    return numerator // gcd_val, denominator // gcd_val


if __name__ == "__main__":
    num = factorize_classical(17**2)
    print(num)

