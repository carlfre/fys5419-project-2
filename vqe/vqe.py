from.gatesvqe import (identity_gate, pauli_x_gate, pauli_y_gate, pauli_z_gate,hadamard_gate, phase_gate, multi_kron)
from.ansatzes import one_qubit_ansatz, hardware_efficient_2_qubit_ansatz
import numpy as np
from scipy.optimize import minimize

# Define gates locally for clarity if not already done in gatesvqe
I = identity_gate()
X = pauli_x_gate()
Y = pauli_y_gate()
Z = pauli_z_gate()
H_gate = hadamard_gate()
S_gate = phase_gate()
S_dagger_gate = S_gate.conj().T # S dagger

class VQE:
    @staticmethod
    def measure_first_qubit(ket: np.ndarray, n_shots: int) -> np.ndarray:
        """Measures the first qubit of a state ket, n_shots times."""
        dim = ket.shape[0]
        if dim < 2: # Handle case of non-qubit state if necessary
            return np.array([])
        probs = np.abs(ket) ** 2
        # Ensure probabilities sum to ~1, handle potential floating point errors
        probs = probs / np.sum(probs)
        first_qubit_eq_0_prob = np.sum(probs[:dim // 2])
        # Clamp probability to  to avoid issues with small numerical errors
        first_qubit_eq_0_prob = np.clip(first_qubit_eq_0_prob, 0.0, 1.0)
        uniform_samples = np.random.rand(n_shots)
        return np.where(uniform_samples < first_qubit_eq_0_prob, 0, 1)

    @staticmethod
    def measurement_to_energy(measurements: np.ndarray) -> float:
        """Converts an array of 0/1 measurements to a <Z> expectation value."""
        if measurements.size == 0:
            return 0.0 # Or handle error appropriately
        # <Z> = P(0) - P(1) = (N0/N) - (N1/N)
        # mean = (0*N0 + 1*N1)/N = N1/N = P(1)
        # <Z> = (1 - P(1)) - P(1) = 1 - 2*P(1) = 1 - 2*mean
        return 1.0 - 2 * measurements.mean()

    @staticmethod
    def estimate_pauli_expval(psi_initial: np.ndarray, U_basis_change: np.ndarray, n_shots: int) -> float:
        """Estimates <psi|P|psi> by measuring <psi'|Z|psi'> where |psi'> = U|psi>."""
        # Ensure psi_initial is a column vector for matrix multiplication
        if psi_initial.ndim == 1:
            psi_initial = psi_initial[:, np.newaxis]

        # Apply basis change: U|psi>
        psi_rotated = U_basis_change @ psi_initial
        measurements = VQE.measure_first_qubit(psi_rotated.flatten(), n_shots) # Flatten for measure function
        return VQE.measurement_to_energy(measurements)

    @staticmethod
    def estimate_pauli_string_expectation(ket: np.ndarray, pauli_op: np.ndarray) -> float:
        """Calculates exact expectation value <ket|pauli_op|ket>."""
        if ket.ndim == 1:
            ket_col = ket[:, np.newaxis] # Column vector
            ket_bra = ket[np.newaxis, :].conj() # Bra vector (conjugate transpose)
        else: # Assume ket is already a column vector
            ket_col = ket
            ket_bra = ket.conj().T

        expectation_value = ket_bra @ pauli_op @ ket_col
        # Expectation value of a Hermitian operator must be real
        return float(np.real(expectation_value))


    @staticmethod
    def vqe_for_2x2_hamiltonian(H: np.ndarray, n_shots: int = 10_000) -> float:
        """VQE for a general 2x2 Hamiltonian (Corrected)."""

        # Calculate Pauli coefficients using trace formula: c_P = Tr(H @ P) / d, d=2
        coeff_I = np.trace(H @ I) / 2.0
        coeff_Z = np.trace(H @ Z) / 2.0
        coeff_X = np.trace(H @ X) / 2.0
        coeff_Y_complex = np.trace(H @ Y) / 2.0 # Note: Y is Hermitian, trace is real

        # Store non-zero terms and their basis changes
        pauli_terms = []
        # Term Z: Basis change = I
        if abs(coeff_Z) > 1e-9:
            pauli_terms.append({'coeff': coeff_Z, 'basis_change': I})
        # Term X: Basis change = H
        if abs(coeff_X) > 1e-9:
            pauli_terms.append({'coeff': coeff_X, 'basis_change': H_gate})
        # Term Y: Basis change = S^dagger H
        if abs(coeff_Y_complex) > 1e-9:
            U_Y = S_dagger_gate @ H_gate
            pauli_terms.append({'coeff': coeff_Y_complex, 'basis_change': U_Y})

        def expected_value(theta: np.ndarray, n_shots: int) -> float:
            # Prepare ansatz state |psi(theta)>
            # Ensure one_qubit_ansatz returns a flattened array or column vector
            ket = one_qubit_ansatz(*theta).flatten()

            # Start with identity term contribution <I> = 1
            total_energy = coeff_I

            # Add contributions from other Pauli terms
            for term in pauli_terms:
                # Estimate <psi|P|psi> using shots
                exp_val_P = VQE.estimate_pauli_expval(ket, term['basis_change'], n_shots)
                # Add contribution: coeff * <P>
                # Ensure result is real, as coeffs can be complex but <H> must be real
                total_energy += (term['coeff'] * exp_val_P).real

            return float(total_energy) # Return final real energy

        # Initial guess and optimization
        # Determine number of parameters from the ansatz function
        num_params = one_qubit_ansatz.__code__.co_argcount
        theta0 = np.random.rand(num_params) * (2 * np.pi)
        res = minimize(expected_value, theta0, args=(n_shots), method="Powell",
                       options={'disp': False}) # Suppress verbose output

        # Return optimized energy found by the optimizer
        final_energy = res.fun # The minimum value found by minimize
        return final_energy


    @staticmethod
    def vqe_for_4x4_hamiltonian(H: np.ndarray, n_shots: int = 10_000) -> float:
        """VQE for a general 4x4 Hamiltonian using exact expectation."""
        
        # Pauli basis for 2 qubits
        pauli_basis_ops = {
            'II': multi_kron(I, I), 'IX': multi_kron(I, X), 'IY': multi_kron(I, Y), 'IZ': multi_kron(I, Z),
            'XI': multi_kron(X, I), 'XX': multi_kron(X, X), 'XY': multi_kron(X, Y), 'XZ': multi_kron(X, Z),
            'YI': multi_kron(Y, I), 'YX': multi_kron(Y, X), 'YY': multi_kron(Y, Y), 'YZ': multi_kron(Y, Z),
            'ZI': multi_kron(Z, I), 'ZX': multi_kron(Z, X), 'ZY': multi_kron(Z, Y), 'ZZ': multi_kron(Z, Z)
        }

        # Calculate coefficients c_P = Tr(H @ P) / d, where d=4 for 2 qubits
        hamiltonian_terms = []
        for name, op in pauli_basis_ops.items():
            coeff = np.trace(H @ op) / 4.0
            # Store terms with non-negligible coefficients
            if abs(coeff) > 1e-9:
                 # We need the coefficient and the operator matrix itself
                 hamiltonian_terms.append({'coeff': coeff, 'op': op})

        def expected_value_exact(theta: np.ndarray) -> float:
            # Prepare ansatz state |psi(theta)>
            ket = hardware_efficient_2_qubit_ansatz(*theta).flatten()

            total_energy = 0.0
            for term in hamiltonian_terms:
                # Calculate exact expectation <psi|P|psi>
                exp_val_P = VQE.estimate_pauli_string_expectation(ket, term['op'])
                # Add contribution: coeff * <P> (ensure result is real)
                total_energy += (term['coeff'] * exp_val_P).real

            return float(total_energy)

        # Initial guess and optimization
        num_params = hardware_efficient_2_qubit_ansatz.__code__.co_argcount
        theta0 = np.random.rand(num_params) * (2 * np.pi)
        # Using exact expectation, so n_shots is irrelevant here
        res = minimize(expected_value_exact, theta0, method="Powell",
                       options={'disp': False})

        final_energy = res.fun
        return final_energy

