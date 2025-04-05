import numpy as np

def identity_gate() -> np.ndarray:
    return np.array([[1, 0], [0, 1]])


def pauli_x_gate() -> np.ndarray:
    return np.array([[0, 1], [1, 0]])


def pauli_y_gate() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]])


def pauli_z_gate() -> np.ndarray:
    return np.array([[1, 0], [0, -1]])


def hadamard_gate() -> np.ndarray:
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


def phase_gate() -> np.ndarray:
    return np.array([[1, 0], [0, 1j]])


def cnot_gate(first_is_control: bool = True) -> np.ndarray:
    if first_is_control:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    else:
        return np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   


def RY_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
        ])


def RX_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])

def RZ_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ])


def CX_10_gate() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])

def SWAP_gate() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])



def multi_kron(*mats: np.ndarray) -> np.ndarray:
    """Tensor product of multiple matrices."""
    result = mats[0]
    for mat in mats[1:]:
        result = np.kron(result, mat)
    return result


if __name__ == "__main__":
    
    X = pauli_x_gate()
    I = np.eye(2)
    Z = pauli_z_gate()
    Y = pauli_y_gate()
    H = hadamard_gate()



    CX_10 = CX_10_gate()
    SWAP = SWAP_gate()
    S = phase_gate()


    # print(SWAP.T.conj() @ np.kron(I, Z) @ SWAP)

    #  print(np.kron(X, Z))
    

    np.set_printoptions(precision=1)
    np.set_printoptions(linewidth=np.inf)
    mat = multi_kron(I, I, Y, Y)

    U = multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I) @ multi_kron(I, I, CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()))
            


    # print( U @ mat @ U.T.conj())

    print(np.linalg.norm(U @ mat @ U.T.conj() - multi_kron(Z, I, I, I)))