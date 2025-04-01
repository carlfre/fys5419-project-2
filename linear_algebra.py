import numpy as np




def quantum_fourier_transform(N: int) -> np.ndarray:
    qft = np.zeros((N, N), dtype=complex)

    omega = np.exp(2 * np.pi * 1j / N)
    for j in range(N):
        for k in range(N):
            qft[j, k] = omega**(k * j)

    return qft / np.sqrt(N)


if __name__ == "__main__":
    qft = quantum_fourier_transform(4)

    print(qft)