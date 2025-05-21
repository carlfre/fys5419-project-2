import numpy as np




def dft(N: int) -> np.ndarray:
    """N x N Discrete Fourier Transform matrix."""
    qft = np.zeros((N, N), dtype=complex)

    omega = np.exp(2 * np.pi * 1j / N)
    for j in range(N):
        for k in range(N):
            qft[j, k] = omega**(k * j)

    return qft / np.sqrt(N)


if __name__ == "__main__":
    qft = dft(4)

    print(qft)