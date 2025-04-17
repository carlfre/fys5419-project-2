from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Gate  
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseEstimation
import numpy as np
import cmath

def run_and_get_counts(circuit: QuantumCircuit, shots=1024) -> dict:
    """Simulates the circuit and returns the measurement counts."""
    simulator = AerSimulator(method='statevector')  # Use 'aer_simulator'
    compiled_circuit = transpile(circuit, simulator)
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
        phase = cmath.phase(amplitude)

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


def phase_estimation_1q(unitary_gate: Gate, theta: float) -> QuantumCircuit:
    
    #Phase estimation for a 1-qubit unitary.

    #Args:
    #    unitary_gate: The unitary gate (U) for which to estimate the phase.
    #    theta: The known phase of the unitary gate's eigenvalue.

    #Returns:
    #    The quantum circuit implementing phase estimation.
    

    qc = QuantumCircuit(2, 1)  # 1 counting qubit, 1 eigenstate qubit
    qc.x(1)  # Prepare eigenstate |1⟩ 
    qc.h(0)  # Prepare counting qubit in superposition
    # Apply controlled-U
    qc.append(unitary_gate.control(1), [0, 1])  # Use append for controlled Gate
    qc.h(0)  # Inverse QFT (1-qubit QFT is just H)
    qc.measure([0], [0])  # Measure counting qubit
    return qc


def phase_estimation_2q(unitary_gate: Gate, theta: float) -> QuantumCircuit:
 
    qc = QuantumCircuit(3, 2)  # 2 counting qubits, 1 eigenstate qubit
    qc.x(2)       # Prepare eigenstate |1⟩     
    qc.h([0, 1])  # Prepare counting qubits in superposition
    # Apply controlled-U operations
    qc.append(unitary_gate.control(1), [0, 2])  # Use append for controlled Gate
    qc.append(unitary_gate.repeat(2).control(1), [1, 2])  # Use append for repeated and controlled
    # Apply inverse QFT
    qc.h(1)
    qc.cp(-np.pi/2, 0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])  # Measure counting qubits
    return qc

def phase_estimation_qiskit(num_counting_qubits: int, unitary_gate: Gate, theta: float) -> QuantumCircuit:
    
    #Phase estimation using Qiskit's PhaseEstimation for n counting qubits.
    #Args:
    #    num_counting_qubits: Number of counting (evaluation) qubits.
    #    unitary_gate: The unitary gate (U) for which to estimate the phase.
    #    theta: The known phase of the unitary gate's eigenvalue (for reference).
    #Returns:
    #    The quantum circuit implementing phase estimation.
    
    # Total qubits: num_counting_qubits + 1 (for eigenstate)
    total_qubits = num_counting_qubits + 1
    qc = QuantumCircuit(total_qubits, num_counting_qubits)
    # Prepare eigenstate |1> for Rz(theta), since Rz(theta)|1> = e^{i theta}|1>
    qc.x(num_counting_qubits)  # Eigenstate qubit is the last one
    # Apply PhaseEstimation
    pe_circuit = PhaseEstimation(num_evaluation_qubits=num_counting_qubits, unitary=unitary_gate, iqft=None, name=f"QPE_{num_counting_qubits}")
    # Append the phase estimation circuit to the qubits
    qc.append(pe_circuit, range(total_qubits))
    # Measure counting qubits
    qc.measure(range(num_counting_qubits), range(num_counting_qubits))
    return qc


def estimate_phase_from_counts_specific(counts: dict, num_counting_qubits: int) -> float:
    """Estimates the phase from measurement counts, accounting for reversed qubit order from iQFT.

    Args:
        counts: The measurement counts dictionary (e.g., {'10': 3000}).
                Keys are bit strings representing the measured state of the counting qubits.
        num_counting_qubits: The number of counting qubits used in the phase estimation.

    Returns:
        The estimated phase theta in radians (between 0 and 2*pi).
    """
    estimated_phase_phi = 0.0  # Accumulator for the fractional phase phi = theta / (2*pi)
    total_shots = sum(counts.values())

    if total_shots == 0:
        return 0.0 # Avoid division by zero if counts dict is empty or all counts are 0

    for state_str, count in counts.items():
        # Ensure the measured string has the correct length (sometimes Qiskit might omit leading zeros)
        # Although typically the keys directly correspond to the measured qubits.
        # If keys might be shorter than num_counting_qubits, padding might be needed,
        # but usually AerSimulator provides full-length keys.
        if len(state_str)!= num_counting_qubits:
             # This case should ideally not happen with standard QPE measurement if all counting qubits are measured.
             # Handle potential errors or unexpected formats if necessary.
             print(f"Warning: Measured state '{state_str}' has unexpected length. Expected {num_counting_qubits}. Skipping.")
             continue

        # --- Core Correction Logic ---
        # The standard iQFT outputs the phase bits in reverse order.
        # So, if the phase is phi = 0.b1 b2... bn, the measured state is |bn... b2 b1>.
        # We need to reverse the measured string to get b1 b2... bn.
        reversed_state_str = state_str[::-1]
        # ---------------------------

        # Convert the *reversed* binary string to its integer representation (k)
        k = int(reversed_state_str, 2)

        # Calculate the fractional phase contribution for this measurement outcome: phi_k = k / 2^n
        phase_contribution = k / (2**num_counting_qubits)

        # Calculate the probability of this measurement outcome
        probability = count / total_shots

        # Add the weighted contribution to the average fractional phase
        estimated_phase_phi += phase_contribution * probability

    # Convert the final average fractional phase (phi) back to the angle theta = 2 * pi * phi
    estimated_phase_theta = estimated_phase_phi * 2 * np.pi

    # Ensure the result is within [0, 2*pi) range, although weighted average should maintain this.
    # Using fmod can handle potential floating point nuances if needed, but usually not required here.
    # estimated_phase_theta = np.fmod(estimated_phase_theta, 2 * np.pi)
    # if estimated_phase_theta < 0:
    #     estimated_phase_theta += 2 * np.pi

    return estimated_phase_theta


if __name__ == "__main__":
    # Define the unitary gate and its *actual* phase for eigenstate |1>
    actual_theta = np.pi / 4 # Actual phase for |1> eigenstate: Rz(pi)|1> = exp(i*pi/2)|1>
    rz_param = 2 * actual_theta # Parameter for Rz
    actual_phi = actual_theta / (2 * np.pi) # Fractional phase phi = 0.125

    U = get_u(rz_param)

    phase_estimation_funcs = {
        1: phase_estimation_1q,
        2: phase_estimation_2q
    }

    # Loop over 1 and 2 qubits for custom implementation
    print("Custom Phase Estimation:")
    for num_qubits in [1, 2]:
        total_qubits = num_qubits + 1 # Correct total number of qubits

        # Create phase estimation circuit
        phase_estimation_func = phase_estimation_funcs[num_qubits]
        qc_pe = phase_estimation_func(U, actual_theta) # Pass actual theta for reference if needed
        print(f"\n{num_qubits}-Counting Qubit Phase Estimation Circuit:")
        print(qc_pe.draw(output='text')) # Use text output for clarity

        # --- Simulate Statevector (Requires separate circuit without measurement) ---
        qc_state = phase_estimation_func(U, actual_theta)
        qc_state.remove_final_measurements() # Remove measurements for statevector sim
        qc_state.save_statevector()
        simulator_sv = AerSimulator(method='statevector')
        compiled_circuit_sv = transpile(qc_state, simulator_sv)
        job_sv = simulator_sv.run(compiled_circuit_sv) # No shots needed for statevector
        result_sv = job_sv.result()
        statevector = result_sv.get_statevector()
        print(f"\n{num_qubits}-Counting Qubit Statevector (Total Qubits: {total_qubits}):")
        # Use the corrected printing function with the total qubit count
        print_formatted_statevector(statevector, total_qubits)
        # --- End Statevector Simulation ---

        # Simulate and get measurement counts (using original circuit with measurements)
        counts = run_and_get_counts(qc_pe, shots=3000)
        print(f"\n{num_qubits}-Counting Qubit Measurement Counts:", counts)

        # Extract estimated phase using the corrected function
        estimated_phase = estimate_phase_from_counts_specific(counts, num_qubits)
        print(f"Estimated Phase ({num_qubits}-counting qubit): {estimated_phase:.4f} radians")
        print(f"Actual Phase ({num_qubits}-counting qubit):    {actual_theta:.4f} radians")
        print(f"(Actual Phase as fraction phi = theta/2pi: {actual_theta / (2 * np.pi):.4f})")
        print()


#----------Qiskit's inbuilt Phase Estimation-----------#
print("\nQiskit inbuilt PhaseEstimation:")
print()
for num_qubits in [1, 2]:
    total_qubits = num_qubits + 1 # Correct total number of qubits

    # Create phase estimation circuit (for counts)
    qc_pe_qiskit = phase_estimation_qiskit(num_qubits, U, actual_theta)
    print(f"{num_qubits}-Counting Qubit Phase Estimation Circuit (Qiskit):")
    print(qc_pe_qiskit.draw(output='text')) # Use text output

    # --- Simulate Statevector (Qiskit) ---
    qc_state_qiskit = phase_estimation_qiskit(num_qubits, U, actual_theta)
    qc_state_qiskit.remove_final_measurements()
    qc_state_qiskit.save_statevector()
    # simulator_sv defined above
    compiled_circuit_sv_q = transpile(qc_state_qiskit, simulator_sv)
    job_sv_q = simulator_sv.run(compiled_circuit_sv_q)
    result_sv_q = job_sv_q.result()
    statevector_q = result_sv_q.get_statevector()
    print(f"\n{num_qubits}-Counting Qubit Statevector (Qiskit, Total Qubits: {total_qubits}):")
    # Use the corrected printing function
    print_formatted_statevector(statevector_q, total_qubits)
    # --- End Statevector Simulation ---

    # Simulate counts (using original circuit with measurements)
    counts_qiskit = run_and_get_counts(qc_pe_qiskit, shots=3000)
    print(f"\n{num_qubits}-Counting Qubit Measurement Counts (Qiskit):", counts_qiskit)

    # Extract estimated phase using the corrected function
    estimated_phase_qiskit = estimate_phase_from_counts_specific(counts_qiskit, num_qubits)
    print(f"Estimated Phase ({num_qubits}-counting qubit, Qiskit): {estimated_phase_qiskit:.4f} radians")
    print(f"Actual Phase ({num_qubits}-counting qubit, Qiskit):    {actual_theta:.4f} radians")
    print(f"(Actual Phase as fraction phi = theta/2pi: {actual_theta / (2 * np.pi):.4f})")
    print()

