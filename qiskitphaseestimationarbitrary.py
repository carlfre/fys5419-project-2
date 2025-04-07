from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from numpy import pi
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


def inverse_qft(qc: QuantumCircuit, n: int):
    """Performs the inverse Quantum Fourier Transform on the first n qubits of a circuit.

    Args:
        qc: The QuantumCircuit.
        n: The number of qubits to apply the inverse QFT to.
    """
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(-pi / float(2**(j - k)), k, j)


def phase_estimation(unitary_gate: Gate, num_counting_qubits: int) -> QuantumCircuit:
    """
    Phase estimation for a 1-qubit unitary, generalized for an arbitrary number of counting qubits.

    Args:
        unitary_gate: The unitary gate (U) for which to estimate the phase.
        num_counting_qubits: The number of qubits to use for phase estimation.

    Returns:
        The quantum circuit implementing phase estimation.
    """

    qc = QuantumCircuit(num_counting_qubits + 1, num_counting_qubits)  # Counting qubits + 1 eigenstate qubit

    # Prepare counting qubits in superposition
    qc.h(range(num_counting_qubits))

    # Prepare eigenstate qubit in superposition
    qc.h(num_counting_qubits)  # <--- ADD THIS LINE

    # Apply controlled-U operations
    for j in range(num_counting_qubits):
        repetitions = 2**j
        qc.append(unitary_gate.control(1), [j, num_counting_qubits])  # Use append for controlled Gate
        for _ in range(repetitions - 1):
            qc.append(unitary_gate.control(1), [j, num_counting_qubits])

    # Apply inverse QFT
    inverse_qft(qc, num_counting_qubits)

    # Measure counting qubits
    qc.measure(range(num_counting_qubits), range(num_counting_qubits))
    return qc


def estimate_phase_from_counts_arbitrary(counts: dict, num_counting_qubits: int) -> float:
    """Estimates the phase from measurement counts for an arbitrary number of qubits.

    Args:
        counts: The measurement counts from the simulation.
        num_counting_qubits: The number of counting qubits used in the phase estimation.

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

    # Phase estimation for arbitrary qubits
    num_counting_qubits = 4
    qc_pe_arb = phase_estimation(U, num_counting_qubits)
    print(f"\n{num_counting_qubits}-Qubit Phase Estimation Circuit:")
    print(qc_pe_arb.draw())

    # Simulate and get measurement counts
    counts_arb = run_and_get_counts(qc_pe_arb)
    print("\nArbitrary Qubit Measurement Counts:", counts_arb)

    # Extract estimated phase
    estimated_phase_arb = estimate_phase_from_counts_arbitrary(counts_arb, num_counting_qubits)
    print(f"Estimated Phase ({num_counting_qubits}-qubit):", estimated_phase_arb)
    print("Actual Phase:", theta)