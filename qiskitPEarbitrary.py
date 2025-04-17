from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseEstimation
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
        # Ensure binary representation has length total_qubits
        binary_state = bin(i)[2:].zfill(total_qubits)
        amplitude = statevector[i]
        probability = np.abs(amplitude)**2
        # Use cmath.phase for complex numbers, handle near-zero probability
        phase = cmath.phase(amplitude) if probability > 1e-9 else 0.0

        # Only print states with significant probability to avoid clutter
        if probability > 1e-6:
            print(
                f"|{binary_state}> ({i}): ampl: {amplitude:.3f} prob: {probability:.3f} Phase: {phase:.3f}"
            )
            found_significant = True
    if not found_significant:
         print("  (All state amplitudes have probability < 1e-6)")


def get_u(theta):
    """Returns a sample unitary gate."""
    qc = QuantumCircuit(1, name="U")
    qc.rz(theta, 0)
    return qc.to_gate(label="U")


def inverse_qft(qc: QuantumCircuit, n: int):
    """Performs the inverse Quantum Fourier Transform on the first n qubits."""
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(-np.pi / float(2**(j - k)), k, j)


def phase_estimation(unitary_gate: Gate, num_counting_qubits: int) -> QuantumCircuit:
    """Custom phase estimation for arbitrary counting qubits."""
    qc = QuantumCircuit(num_counting_qubits + 1, num_counting_qubits)
    qc.h(range(num_counting_qubits))  # Superposition
    qc.x(num_counting_qubits)  # Eigenstate |1>
    for j in range(num_counting_qubits):
        repetitions = 2**j
        for _ in range(repetitions):
            qc.append(unitary_gate.control(1), [j, num_counting_qubits])
    inverse_qft(qc, num_counting_qubits)
    qc.measure(range(num_counting_qubits), range(num_counting_qubits))
    return qc


def phase_estimation_qiskit(num_counting_qubits: int, unitary_gate: Gate, theta: float) -> QuantumCircuit:
    """Qiskit PhaseEstimation for arbitrary counting qubits."""
    total_qubits = num_counting_qubits + 1
    qc = QuantumCircuit(total_qubits, num_counting_qubits)
    qc.x(num_counting_qubits)  # Eigenstate |1>
    pe_circuit = PhaseEstimation(
        num_evaluation_qubits=num_counting_qubits,
        unitary=unitary_gate,
        iqft=None,  # Use default inverse QFT
        name=f"QPE_{num_counting_qubits}"
    )
    qc.append(pe_circuit, range(total_qubits))
    qc.measure(range(num_counting_qubits), range(num_counting_qubits))
    return qc


def estimate_phase_from_counts_arbitrary(counts: dict, num_counting_qubits: int) -> float:
    """Estimates the phase from measurement counts, accounting for reversed qubit order from iQFT.

    Args:
        counts: The measurement counts dictionary (e.g., {'01': 3000}).
                Keys are bit strings representing the measured state of the counting qubits.
        num_counting_qubits: The number of counting qubits used in the phase estimation.

    Returns:
        The estimated phase theta in radians (between 0 and 2*pi).
    """
    estimated_phase_phi = 0.0  # Accumulator for the fractional phase phi = theta / (2*pi)
    total_shots = sum(counts.values())

    if total_shots == 0:
        return 0.0 # Avoid division by zero

    for state_str, count in counts.items():
        # Ensure measured string length matches counting qubits, handle potential padding issues if any
        if len(state_str)!= num_counting_qubits:
             print(f"Warning: Measured state '{state_str}' has unexpected length. Expected {num_counting_qubits}. Skipping.")
             continue

        # --- Core Correction Logic ---
        # Reverse the measured string because standard iQFT outputs bits in reverse order.
        reversed_state_str = state_str[::-1]
        # ---------------------------

        # Convert the *reversed* binary string to its integer representation (k)
        k = int(reversed_state_str, 2)

        # Calculate the fractional phase contribution: phi_k = k / 2^n
        phase_contribution = k / (2**num_counting_qubits)

        # Calculate the probability of this measurement outcome
        probability = count / total_shots

        # Add the weighted contribution to the average fractional phase
        estimated_phase_phi += phase_contribution * probability

    # Convert the final average fractional phase (phi) back to the angle theta = 2 * pi * phi
    estimated_phase_theta = estimated_phase_phi * 2 * np.pi

    return estimated_phase_theta


if __name__ == "__main__":
    # 1. DEFINE THE PHASE YOU WANT TO ESTIMATE
    actual_theta = np.pi / 4  # Example: Set the target phase to pi

    # 2. CALCULATE THE REQUIRED Rz PARAMETER
    # Since Rz(lambda)|1> = exp(i*lambda/2)|1>, we need lambda = 2 * actual_theta
    rz_param = 2 * actual_theta

    # 3. CREATE THE UNITARY GATE USING THE CORRECT PARAMETER
    U = get_u(rz_param)

    # Calculate the fractional phase for reference
    actual_phi = actual_theta / (2 * np.pi)

    # Test multiple counting qubit counts
    qubit_counts = [1,2,3]  # Example: test with 2, 3, and 4 counting qubits

    # --- Custom Phase Estimation ---
    print("--- Custom Phase Estimation ---")
    for num_counting_qubits in qubit_counts:
        total_qubits = num_counting_qubits + 1
        print(f"\n{num_counting_qubits}-Counting Qubit Phase Estimation Circuit:")
        # Pass U created with the correct rz_param
        qc_pe = phase_estimation(U, num_counting_qubits)
        print(qc_pe.draw(output='text'))

        # --- Statevector Simulation ---
        qc_state = phase_estimation(U, num_counting_qubits) # Use the same U
        qc_state.remove_final_measurements()
        qc_state.save_statevector()
        simulator_sv = AerSimulator(method='statevector')
        compiled_circuit_sv = transpile(qc_state, simulator_sv)
        job_sv = simulator_sv.run(compiled_circuit_sv)
        result_sv = job_sv.result()
        statevector = result_sv.get_statevector()
        print(f"\n{num_counting_qubits}-Counting Qubit Statevector (Total Qubits: {total_qubits}):")
        print_formatted_statevector(statevector, total_qubits) # Use corrected printer

        # --- Counts Simulation ---
        counts = run_and_get_counts(qc_pe, shots=1000) # Use the circuit with measurements
        print(f"\n{num_counting_qubits}-Counting Qubit Measurement Counts:", counts)

        # --- Phase Estimation ---
        # Use the corrected estimation function
        estimated_phase = estimate_phase_from_counts_arbitrary(counts, num_counting_qubits)
        print(f"Estimated Phase ({num_counting_qubits}-counting qubit): {estimated_phase:.4f} radians")
        # Print the actual_theta you defined at the start
        print(f"Actual Phase ({num_counting_qubits}-counting qubit):    {actual_theta:.4f} radians")
        print(f"(Actual Phase as fraction phi = theta/2pi: {actual_phi:.4f})")
        print()

    # --- Qiskit's inbuilt Phase Estimation ---
    print("\n--- Qiskit inbuilt PhaseEstimation ---")
    for num_counting_qubits in qubit_counts:
        total_qubits = num_counting_qubits + 1
        print(f"\n{num_counting_qubits}-Counting Qubit Phase Estimation Circuit (Qiskit):")
        # Pass the same U (created with correct rz_param) and actual_theta for reference
        qc_pe_qiskit = phase_estimation_qiskit(num_counting_qubits, U, actual_theta)
        print(qc_pe_qiskit.draw(output='text'))

        # --- Statevector Simulation (Qiskit) ---
        qc_state_qiskit = phase_estimation_qiskit(num_counting_qubits, U, actual_theta) # Use same U
        qc_state_qiskit.remove_final_measurements()
        qc_state_qiskit.save_statevector()
        # simulator_sv defined above
        compiled_circuit_sv_q = transpile(qc_state_qiskit, simulator_sv)
        job_sv_q = simulator_sv.run(compiled_circuit_sv_q)
        result_sv_q = job_sv_q.result()
        statevector_q = result_sv_q.get_statevector()
        print(f"\n{num_counting_qubits}-Counting Qubit Statevector (Qiskit, Total Qubits: {total_qubits}):")
        print_formatted_statevector(statevector_q, total_qubits) # Use corrected printer

        # --- Counts Simulation (Qiskit) ---
        counts_qiskit = run_and_get_counts(qc_pe_qiskit, shots=1000) # Use circuit with measurements
        print(f"\n{num_counting_qubits}-Counting Qubit Measurement Counts (Qiskit):", counts_qiskit)

        # --- Phase Estimation (Qiskit) ---
        # Use the corrected estimation function
        estimated_phase_qiskit = estimate_phase_from_counts_arbitrary(counts_qiskit, num_counting_qubits)
        print(f"Estimated Phase ({num_counting_qubits}-counting qubit, Qiskit): {estimated_phase_qiskit:.4f} radians")
        # Print the actual_theta defined at the start
        print(f"Actual Phase ({num_counting_qubits}-counting qubit, Qiskit):    {actual_theta:.4f} radians")
        print(f"(Actual Phase as fraction phi = theta/2pi: {actual_phi:.4f})")
        print()