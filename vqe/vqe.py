import multiprocessing
from time import time
from typing import Literal
import numpy as np
from scipy.optimize import minimize
from .gatesvqe import multi_kron, identity_gate, pauli_x_gate, pauli_y_gate, pauli_z_gate, hadamard_gate, SWAP_gate, CX_10_gate, phase_gate
from .ansatzes import one_qubit_ansatz, hardware_efficient_2_qubit_ansatz, hardware_efficient_4_qubit_ansatz, repeated_hae_gate_4_qubit_ansatz, complicated_2_qubit_ansatz
from .utilsvqe import write_to_csv

I, X, Y, Z, H, SWAP, CX_10, S = identity_gate(), pauli_x_gate(), pauli_y_gate(), pauli_z_gate(), hadamard_gate(), SWAP_gate(), CX_10_gate(), phase_gate()

class VQE:
    @staticmethod
    def measure_first_qubit(ket: np.ndarray, n_shots: int) -> np.ndarray:
        """Measures the first qubit of a state ket, n_shots times.

        Args:
            ket (np.ndarray): State to measure
            n_shots (int): Number of measurements to make

        Returns:
            np.ndarray: Array of 0s and 1s, representing the measurement outcomes
        """
        dim = ket.shape[0]
        probs = np.abs(ket) ** 2
        first_qubit_eq_0_prob = np.sum(probs[:dim // 2])
        uniform_samples = np.random.rand(n_shots)
        return np.where(uniform_samples < first_qubit_eq_0_prob, 0, 1)

    @staticmethod
    def measurement_to_energy(measurements: np.ndarray) -> float:
        """Converts an array of measurements to a <Z> expectation value

        Args:
            measurements (np.ndarray): array of measurements

        Returns:
            float: energy estimate
        """
        return -2 * measurements.mean() + 1

    @staticmethod
    def estimate_pauli_expval(psi_initial: np.ndarray, U: np.ndarray, n_shots: int) -> float:
        """Takes in a state psi_initial, and a change of basis matrix U, outputs expectation. 
        
        (U is determined by theparticular Pauli string P we are evaluating expectation of)

        Args:
            psi_initial (np.ndarray): State vector
            U (np.ndarray): Unitary change of basis matrix
            n_shots (int): number of measurements to make

        Returns:
            float: ground state energy estimate
        """
        measurements = VQE.measure_first_qubit(U @ psi_initial, n_shots)
        return VQE.measurement_to_energy(measurements)

    @staticmethod
    def vqe_for_2x2_hamiltonian(H: np.ndarray, n_shots: int = 10_000) -> float:
        """VQE for a general 2x2 Hamiltonian."""

        #Express the Hamiltonian in terms of Pauli matrices
        I = identity_gate()
        X = pauli_x_gate()
        Y = pauli_y_gate()
        Z = pauli_z_gate()

        a = H[0, 0] + H[1, 1]
        b = H[0, 0] - H[1, 1]
        c = H[0, 1] + H[1, 0]
        d = H[0, 1] - H[1, 0]

        coeffs = [a/2, b/2, c/2, d/2j]
        paulis = [I, Z, X, Y]
        U_matrices = [I, I, H, np.array([[0, 1], [1, 0]])]

        def expected_value(theta: np.ndarray, n_shots: int) -> float:
            energy = 0
            ket = one_qubit_ansatz(*theta)
            for coeff, pauli, U_matrix in zip(coeffs, paulis, U_matrices):
              energy += coeff * VQE.estimate_pauli_expval(ket, U_matrix, n_shots)
            return energy

        theta0 = np.random.rand(2) * (2 * np.pi)  # Random initial guess
        res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
        theta_opt = res.x
        return expected_value(theta_opt, n_shots)

    @staticmethod
    def vqe_for_4x4_hamiltonian(H: np.ndarray, n_shots: int = 10_000) -> float:

        I = identity_gate()
        X = pauli_x_gate()
        Y = pauli_y_gate()
        Z = pauli_z_gate()

        pauli_matrices = [
            multi_kron(I, I), multi_kron(I, X), multi_kron(I, Y), multi_kron(I, Z),
            multi_kron(X, I), multi_kron(X, X), multi_kron(X, Y), multi_kron(X, Z),
            multi_kron(Y, I), multi_kron(Y, X), multi_kron(Y, Y), multi_kron(Y, Z),
            multi_kron(Z, I), multi_kron(Z, X), multi_kron(Z, Y), multi_kron(Z, Z)
        ]

        coeffs = []
        for pauli in pauli_matrices:
            coeffs.append(np.trace(H @ pauli) / 4)  # Calculate Pauli coefficients

        def expected_value(theta: np.ndarray, n_shots: int) -> float:
            energy = 0
            ket = hardware_efficient_2_qubit_ansatz(*theta)
            for coeff, pauli in zip(coeffs, pauli_matrices):
                energy += coeff * VQE.estimate_pauli_expval(ket, pauli, n_shots)
            return energy

        theta0 = np.random.rand(4) * (2 * np.pi)  # Random initial guess
        res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
        theta_opt = res.x

        return expected_value(theta_opt, n_shots)



