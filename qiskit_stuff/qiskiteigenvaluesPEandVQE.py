from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseEstimation, UnitaryGate
from scipy.linalg import expm, eigh # Use eigh for Hermitian matrices
from vqe.vqe import VQE
import numpy as np
import cmath



def run_and_get_counts(circuit: QuantumCircuit, shots=1024) -> dict:
    """Simulates the circuit and returns the measurement counts."""
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator, optimization_level=0)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts

def print_formatted_statevector(statevector: Statevector, total_qubits: int):
    """Prints the statevector correctly, iterating over all basis states."""
    print(f"Statevector (total_qubits={total_qubits}):")
    found_significant = False
    for i in range(2**total_qubits):
        binary_state = bin(i)[2:].zfill(total_qubits)
        amplitude = statevector[i]
        probability = np.abs(amplitude)**2
        phase = cmath.phase(amplitude) if probability > 1e-9 else 0.0
        if probability > 1e-6:
            print(
                f"|{binary_state}> ({i}): ampl: {amplitude:.3f} prob: {probability:.3f} Phase: {phase:.3f}"
            )
            found_significant = True
    if not found_significant:
         print("  (All state amplitudes have probability < 1e-6)")

def inverse_qft(qc: QuantumCircuit, n: int):
    """Performs the inverse Quantum Fourier Transform on the first n qubits."""
    if n == 0:
        return # No QFT for 0 qubits
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            # Use negative phase for inverse QFT
            qc.cp(-np.pi / float(2**(j - k)), k, j)

# Using Qiskit's PhaseEstimation is generally recommended
def phase_estimation_qiskit(num_counting_qubits: int, unitary_gate: Gate, input_eigenstate_circuit: QuantumCircuit = None) -> QuantumCircuit:
    """Builds a QPE circuit using Qiskit's PhaseEstimation library function.

    Args:
        num_counting_qubits: Number of qubits for the phase register.
        unitary_gate: The unitary operator whose phases are to be estimated.
        input_eigenstate_circuit: A QuantumCircuit that prepares the desired input eigenstate
                                   on the target qubits. If None, assumes |0...0> input.

    Returns:
        The complete QPE QuantumCircuit with measurements.
    """
    num_target_qubits = unitary_gate.num_qubits
    total_qubits = num_counting_qubits + num_target_qubits

    pe_circuit_component = PhaseEstimation(
        num_evaluation_qubits=num_counting_qubits,
        unitary=unitary_gate,
        name=f"QPE_{num_counting_qubits}"
    )

    qc = QuantumCircuit(total_qubits, num_counting_qubits, name="Full QPE")

    target_qubits = list(range(num_counting_qubits, total_qubits))
    if input_eigenstate_circuit:
        if input_eigenstate_circuit.num_qubits!= num_target_qubits:
             raise ValueError(f"Input state circuit has {input_eigenstate_circuit.num_qubits} qubits, expected {num_target_qubits}")
        qc.compose(input_eigenstate_circuit, qubits=target_qubits, inplace=True)

    qc.compose(pe_circuit_component, qubits=list(range(total_qubits)), inplace=True)
    qc.measure(range(num_counting_qubits), range(num_counting_qubits))

    return qc


def estimate_phase_from_counts_arbitrary(counts: dict, num_counting_qubits: int) -> float:
    """Estimates the phase from measurement counts using the most probable outcome."""
    estimated_phase_phi = 0.0
    total_shots = sum(counts.values())

    if total_shots == 0:
        print("Warning: No counts received.")
        return 0.0

    most_probable_state = None
    max_count = 0

    for state_str, count in counts.items():
        if len(state_str)!= num_counting_qubits:
             print(f"Warning: Measured state '{state_str}' has unexpected length. Expected {num_counting_qubits}. Skipping.")
             continue
        if count > max_count:
            max_count = count
            most_probable_state = state_str

    if most_probable_state is None:
        print("Warning: Could not determine most probable state.")
        return 0.0

    reversed_state_str = most_probable_state[::-1]
    k = int(reversed_state_str, 2)
    estimated_phase_phi = k / (2**num_counting_qubits)
    estimated_phase_theta = estimated_phase_phi * 2 * np.pi

    return estimated_phase_theta

def calculate_and_adjust_energy(estimated_phase: float, t: float) -> float:
    #Calculates energy from phase and adjusts for 2pi wrapping.
    # Assumes the actual phase E*t is within (-pi, pi] for simple adjustment
    if estimated_phase > np.pi:
         energy_adjusted = -(estimated_phase - 2*np.pi) / t
    else:
         #energy_adjusted = -estimated_phase / t
         energy_adjusted = -(estimated_phase - 2*np.pi) / t
    return energy_adjusted

if __name__ == "__main__":

    t = 1.0  # Evolution time
    shots = 10000 # Number of simulation shots

    # H_2x2
    #H11, H12, H22 = 1.0, 0.5, -1.0
    #H_2x2 = np.array([[H11, H12], [H12, H22]])
    # 1. 2x2 Hamiltonian
    E1 = 1.0
    E2 = 2.0
    V11 = 0.1
    V12 = 0.2
    V21 = 0.2
    V22 = 0.3

    H_2x2 = np.array([[E1 + V11, V12], [V21, E2 + V22]])
    print("--- H_2x2 ---")
    print(H_2x2)
    eigvals_2x2, eigvecs_2x2 = eigh(H_2x2)
    print(f"Classical Eigenvalues (H_2x2): {eigvals_2x2}")

    # H_4x4
    #Hx, Hz = 0.5, 1.0
    #eps = 0.0
    #H_4x4 = np.array(([eps + Hz, 0, 0, Hx],
    #    [0, eps - Hz, Hx, 0],
    #    [0, Hx, eps - Hz, 0],
    #    [Hx, 0, 0, eps + Hz]))
    
        # 2. 4x4 Hamiltonian
    epsilon_00 = 1.5
    epsilon_10 = 2.5
    epsilon_01 = 3.5
    epsilon_11 = 4.5
    Hx_val = 0.4
    Hz_val = 0.6

    H_4x4 = np.array([
        [epsilon_00 + Hz_val, 0, 0, Hx_val],
        [0, epsilon_10 - Hz_val, Hx_val, 0],
        [0, Hx_val, epsilon_01 - Hz_val, 0],
        [Hx_val, 0, 0, epsilon_11 + Hz_val]
    ])
    print("\n--- H_4x4 ---")
    print(H_4x4)
    eigvals_4x4, eigvecs_4x4 = eigh(H_4x4)
    print(f"Classical Eigenvalues (H_4x4): {eigvals_4x4}")


    # === Unitary Gate Creation ===
    U_matrix_2x2 = expm(-1j * H_2x2 * t)
    U_gate_2x2 = UnitaryGate(U_matrix_2x2, label=f"U_2x2(t={t})")

    U_matrix_4x4 = expm(-1j * H_4x4 * t)
    U_gate_4x4 = UnitaryGate(U_matrix_4x4, label=f"U_4x4(t={t})")

    # === QPE for ALL Eigenvalues of H_2x2 ===
    print("\n--- QPE for H_2x2 (All Eigenvalues) ---")
    num_counting_2x2 = 8 # Number of counting qubits
    num_target_2x2 = H_2x2.shape # Should be 2
    qpe_energies_2x2 = []

    for i in range(num_target_2x2[0]):
        target_eigenvalue_index_2x2 = i
        target_eigenvector_2x2 = eigvecs_2x2[:, target_eigenvalue_index_2x2]
        exact_eigenvalue_2x2 = eigvals_2x2[target_eigenvalue_index_2x2]

        print(f"\nTargeting Eigenvalue Index {target_eigenvalue_index_2x2}: Exact E = {exact_eigenvalue_2x2:.5f}")

        # Prepare input state circuit
        prep_circuit_2x2 = QuantumCircuit(U_gate_2x2.num_qubits, name=f"PrepPsi_{i}")
        prep_circuit_2x2.initialize(target_eigenvector_2x2,)

        # Build QPE circuit
        qpe_circuit_2x2 = phase_estimation_qiskit(
            num_counting_qubits=num_counting_2x2,
            unitary_gate=U_gate_2x2,
            input_eigenstate_circuit=prep_circuit_2x2
        )

        # Simulate
        print(f"Running simulation with {shots} shots...")
        counts_2x2 = run_and_get_counts(qpe_circuit_2x2, shots=shots)
        # print(f"Measurement Counts: {counts_2x2}") # Optional: print counts

        # Estimate Phase and Energy
        estimated_phase_2x2 = estimate_phase_from_counts_arbitrary(counts_2x2, num_counting_2x2)
        estimated_energy_2x2 = calculate_and_adjust_energy(estimated_phase_2x2, t)
        qpe_energies_2x2.append(estimated_energy_2x2)

        print(f"Estimated Phase: {estimated_phase_2x2:.5f} radians")
        print(f"Estimated Energy: {estimated_energy_2x2:.5f}")
        print(f"Exact Energy:     {exact_eigenvalue_2x2:.5f}")
        print(f"Absolute Error:   {abs(estimated_energy_2x2 - exact_eigenvalue_2x2):.5f}")


    # === QPE for ALL Eigenvalues of H_4x4 ===
    print("\n--- QPE for H_4x4 (All Eigenvalues) ---")
    num_counting_4x4 = 8 # More qubits for potentially better precision
    num_target_4x4 = H_4x4.shape # Should be 4
    qpe_energies_4x4 = []

    for i in range(num_target_4x4[0]):
        target_eigenvalue_index_4x4 = i
        target_eigenvector_4x4 = eigvecs_4x4[:, target_eigenvalue_index_4x4]
        exact_eigenvalue_4x4 = eigvals_4x4[target_eigenvalue_index_4x4]

        print(f"\nTargeting Eigenvalue Index {target_eigenvalue_index_4x4}: Exact E = {exact_eigenvalue_4x4:.5f}")

        # Prepare input state circuit
        prep_circuit_4x4 = QuantumCircuit(U_gate_4x4.num_qubits, name=f"PrepPsi_{i}")
        prep_circuit_4x4.initialize(target_eigenvector_4x4,)

        # Build QPE circuit
        qpe_circuit_4x4 = phase_estimation_qiskit(
            num_counting_qubits=num_counting_4x4,
            unitary_gate=U_gate_4x4,
            input_eigenstate_circuit=prep_circuit_4x4
        )

        # Simulate
        print(f"Running simulation with {shots} shots...")
        counts_4x4 = run_and_get_counts(qpe_circuit_4x4, shots=shots)
        # print(f"Measurement Counts: {counts_4x4}") # Optional: print counts

        # Estimate Phase and Energy
        estimated_phase_4x4 = estimate_phase_from_counts_arbitrary(counts_4x4, num_counting_4x4)
        estimated_energy_4x4 = calculate_and_adjust_energy(estimated_phase_4x4, t)
        qpe_energies_4x4.append(estimated_energy_4x4)

        print(f"Estimated Phase: {estimated_phase_4x4:.5f} radians")
        print(f"Estimated Energy: {estimated_energy_4x4:.5f}")
        print(f"Exact Energy:     {exact_eigenvalue_4x4:.5f}")
        print(f"Absolute Error:   {abs(estimated_energy_4x4 - exact_eigenvalue_4x4):.5f}")



        # === VQE Calculations ===
    print("\n" + "="*30)
    print(" VQE Ground State Calculations")
    print("="*30)

    # --- VQE for H_2x2 ---
    print("\n--- VQE for H_2x2 ---")
    vqe_energy_2x2 = None # Initialize in case of error
    try:
        # Assuming VQE class and methods are defined earlier in your script
        # The vqe_for_2x2_hamiltonian uses shot-based estimation internally
        shots_vqe_2x2 = 10000 # Match the default in the function or set as desired
        print(f"Running VQE with {shots_vqe_2x2} shots per expectation value...")
        # Make sure the VQE class and its dependencies (gates, ansatz) are defined
        vqe_energy_2x2 = VQE.vqe_for_2x2_hamiltonian(H_2x2, n_shots=shots_vqe_2x2)
        exact_gs_energy_2x2 = eigvals_2x2[0] # Ground state is the first eigenvalue
        print(f"VQE Estimated Ground State Energy (H_2x2): {vqe_energy_2x2:.5f}")
        print(f"Exact Ground State Energy (H_2x2):         {exact_gs_energy_2x2:.5f}")
        print(f"Absolute Error (VQE H_2x2):                {abs(vqe_energy_2x2 - exact_gs_energy_2x2):.5f}")
    except NameError as ne:
        print(f"VQE class or related function not found. Skipping VQE for H_2x2. Error: {ne}")
    except ImportError as ie:
         print(f"Could not import VQE dependencies (.gatesvqe,.ansatzes?). Skipping VQE for H_2x2. Error: {ie}")
    except Exception as e:
        print(f"An error occurred during VQE for H_2x2: {e}")


    # --- VQE for H_4x4 ---
    print("\n--- VQE for H_4x4 ---")
    vqe_energy_4x4 = None # Initialize in case of error
    try:
        # Assuming VQE class and methods are defined earlier
        # Note: Your provided vqe_for_4x4_hamiltonian uses exact statevector expectation values
        shots_vqe_4x4 = 10000 # Match the default in the function or set as desired
        print("Running VQE using exact statevector expectation values...")
        # Make sure the VQE class and its dependencies (gates, ansatz) are defined
        vqe_energy_4x4 = VQE.vqe_for_4x4_hamiltonian(H_4x4, n_shots=shots_vqe_4x4)
        exact_gs_energy_4x4 = eigvals_4x4[0] # Ground state is the first eigenvalue
        print(f"VQE Estimated Ground State Energy (H_4x4): {vqe_energy_4x4:.5f}")
        print(f"Exact Ground State Energy (H_4x4):         {exact_gs_energy_4x4:.5f}")
        print(f"Absolute Error (VQE H_4x4):                {abs(vqe_energy_4x4 - exact_gs_energy_4x4):.5f}")
    except NameError as ne:
        print(f"VQE class or related function not found. Skipping VQE for H_4x4. Error: {ne}")
    except ImportError as ie:
         print(f"Could not import VQE dependencies (.gatesvqe,.ansatzes?). Skipping VQE for H_4x4. Error: {ie}")
    except Exception as e:
        print(f"An error occurred during VQE for H_4x4: {e}")


    # --- Comparison Summary ---
    print("\n" + "="*55)
    print(" Ground State Energy Comparison Summary")
    print("="*55)
    print(f"{'Hamiltonian':<12} | {'Exact':<10} | {'QPE (Est.)':<12} | {'VQE (Est.)':<12}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

    # H_2x2 Summary
    # Check if QPE results list is not empty before accessing index 0
    qpe_gs_2x2_str = f"{qpe_energies_2x2[0]:.5f}" if qpe_energies_2x2 else "N/A"
    vqe_gs_2x2_str = f"{vqe_energy_2x2:.5f}" if vqe_energy_2x2 is not None else "N/A"
    print(f"{'H_2x2':<12} | {exact_gs_energy_2x2:<10.5f} | {qpe_gs_2x2_str:<12} | {vqe_gs_2x2_str:<12}")

    # H_4x4 Summary
    # Check if QPE results list is not empty before accessing index 0
    qpe_gs_4x4_str = f"{qpe_energies_4x4[0]:.5f}" if qpe_energies_4x4 else "N/A"
    vqe_gs_4x4_str = f"{vqe_energy_4x4:.5f}" if vqe_energy_4x4 is not None else "N/A"
    print(f"{'H_4x4':<12} | {exact_gs_energy_4x4:<10.5f} | {qpe_gs_4x4_str:<12} | {vqe_gs_4x4_str:<12}")