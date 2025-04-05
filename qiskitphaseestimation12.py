from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from numpy import pi
from qiskit.quantum_info import Statevector
import cmath
from qiskit.circuit import Gate  # Import Gate
from typing import List


def run_and_get_counts(circuit: QuantumCircuit, shots=1024) -> dict:
    """Simulates the circuit and returns the measurement counts."""
    simulator = AerSimulator(method='statevector')  # Use 'aer_simulator'
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts


def get_u(theta):
    """Returns a sample unitary gate."""
    qc = QuantumCircuit(1, name="U")
    qc.rz(theta, 0)
    return qc.to_gate(label="U")


def phase_estimation_1q(unitary_gate: Gate, theta: float) -> QuantumCircuit:
    """
    Phase estimation for a 1-qubit unitary.

    Args:
        unitary_gate: The unitary gate (U) for which to estimate the phase.
        theta: The known phase of the unitary gate's eigenvalue.

    Returns:
        The quantum circuit implementing phase estimation.
    """

    qc = QuantumCircuit(2, 1)  # 1 counting qubit, 1 eigenstate qubit
    qc.h(0)  # Prepare counting qubit in superposition

    # Apply controlled-U
    qc.append(unitary_gate.control(1), [0, 1])  # Use append for controlled Gate

    qc.h(0)  # Inverse QFT (1-qubit QFT is just H)

    qc.measure([0], [0])  # Measure counting qubit
    return qc


def phase_estimation_2q(unitary_gate: Gate, theta: float) -> QuantumCircuit:
    """
    Phase estimation for a 1-qubit unitary.

    Args:
        unitary_gate: The unitary gate (U) for which to estimate the phase.
        theta: The known phase of the unitary gate's eigenvalue.

    Returns:
        The quantum circuit implementing phase estimation.
    """
    qc = QuantumCircuit(3, 2)  # 2 counting qubits, 1 eigenstate qubit

    # Prepare counting qubits in superposition
    qc.h([0, 1])

    # Apply controlled-U operations
    qc.append(unitary_gate.control(1), [0, 2])  # Use append for controlled Gate
    qc.append(unitary_gate.repeat(2).control(1), [1, 2])  # Use append for repeated and controlled

    # Apply inverse QFT
    qc.h(1)
    qc.cz(1, 0)
    qc.h(0)

    qc.measure([0, 1], [0, 1])  # Measure counting qubits
    return qc


def print_formatted_statevector(statevector: Statevector, num_qubits: int):
    """Prints the statevector in the desired format."""

    for i in range(2**num_qubits):
        binary_state = bin(i)[2:].zfill(num_qubits)  # Binary representation
        amplitude = statevector[i]  # Access amplitude using Statevector indexing
        probability = np.abs(amplitude) ** 2
        phase = cmath.phase(amplitude)

        print(
            f"|{binary_state}> ({i}): ampl: {amplitude:.3f} prob: {probability:.3f} Phase: {phase:.3f}"
        )


def estimate_phase_from_counts_specific(counts: dict, num_counting_qubits: int) -> float:
    """Estimates the phase from measurement counts for specific qubit numbers (1 or 2).

    Args:
        counts: The measurement counts from the simulation.
        num_counting_qubits: The number of counting qubits used in the phase estimation (1 or 2).

    Returns:
        The estimated phase.
    """
    estimated_phase = 0
    total_shots = sum(counts.values())  # Get the total number of shots

    for key, count in counts.items():
        decimal_representation = int(key, 2)  # Convert binary to decimal
        phase_contribution = (decimal_representation / (2**num_counting_qubits)) * 2 * pi
        probability = count / total_shots
        estimated_phase += phase_contribution * probability  # Weighted average

    return estimated_phase


if __name__ == "__main__":
    # Define the unitary gate and its known phase
    theta = pi / 4  # Example phase (replace with your actual phase)
    U = get_u(theta)  # Create the unitary gate

    # Phase estimation for 1 qubit
    qc_pe_1q = phase_estimation_1q(U, theta)
    print("1-Qubit Phase Estimation Circuit:")
    print(qc_pe_1q.draw())

    # Simulate and get statevector
    simulator = AerSimulator(method='statevector')
    qc_pe_1q.save_statevector()
    compiled_circuit = transpile(qc_pe_1q, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    statevector_1q = result.get_statevector()
    print("1-Qubit Statevector:")
    print_formatted_statevector(statevector_1q, 1)

    # Simulate and get measurement counts
    counts_1q = run_and_get_counts(qc_pe_1q)
    print("\n1-Qubit Measurement Counts:", counts_1q)

    # Extract estimated phase (using the improved method)
    estimated_phase_1q = estimate_phase_from_counts_specific(counts_1q, 1)
    print("Estimated Phase (1-qubit):", estimated_phase_1q)
    print("Actual Phase (1-qubit):", theta)

    # Phase estimation for 2 qubits
    qc_pe_2q = phase_estimation_2q(U, theta)
    print("\n2-Qubit Phase Estimation Circuit:")
    print(qc_pe_2q.draw())

    # Simulate and get statevector
    simulator = AerSimulator(method='statevector')
    qc_pe_2q.save_statevector()
    compiled_circuit = transpile(qc_pe_2q, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    statevector_2q = result.get_statevector()
    print("\n2-Qubit Statevector:")
    print_formatted_statevector(statevector_2q, 2)

    # Simulate and get measurement counts
    counts_2q = run_and_get_counts(qc_pe_2q)
    print("\n2-Qubit Measurement Counts:", counts_2q)

    # Extract estimated phase (more accurate with 2 qubits)
    estimated_phase_2q = estimate_phase_from_counts_specific(counts_2q, 2)
    print("Estimated Phase (2-qubit):", estimated_phase_2q)
    print("Actual Phase (2-qubit):", theta)