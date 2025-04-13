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


def print_formatted_statevector(statevector: Statevector, num_qubits: int):
    """Prints the statevector in the desired format."""
    for i in range(2**num_qubits):
        binary_state = bin(i)[2:].zfill(num_qubits)
        amplitude = statevector[i]
        probability = np.abs(amplitude) ** 2
        phase = cmath.phase(amplitude) if probability > 1e-10 else 0.0
        print(
            f"|{binary_state}> ({i}): ampl: {amplitude:.3f} prob: {probability:.3f} Phase: {phase:.3f}"
        )


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
    """Estimates the phase from measurement counts."""
    estimated_phase = 0
    total_shots = sum(counts.values())
    for key, count in counts.items():
        decimal_representation = int(key, 2)
        phase_contribution = (decimal_representation / (2**num_counting_qubits)) * 2 * np.pi
        probability = count / total_shots
        estimated_phase += phase_contribution * probability
    return estimated_phase


if __name__ == "__main__":
    # Define the unitary gate and phase
    theta = np.pi / 2  # phi = 0.5
    U = get_u(theta)

    # Test multiple counting qubit counts
    qubit_counts = [4]

    # Custom Phase Estimation
    for num_counting_qubits in qubit_counts:
        print(f"\n{num_counting_qubits}-Qubit Phase Estimation Circuit:")
        qc_pe = phase_estimation(U, num_counting_qubits)
        print(qc_pe.draw())
        # Statevector
        initial_state = Statevector.from_label('0' * num_counting_qubits + '1')
        statevector = initial_state.evolve(qc_pe.remove_final_measurements(inplace=False))
        print(f"\n{num_counting_qubits}-Qubit Statevector:")
        print_formatted_statevector(statevector, num_counting_qubits)
        # Counts
        counts = run_and_get_counts(qc_pe, shots=3000)
        print(f"\n{num_counting_qubits}-Qubit Measurement Counts:", counts)
        # Phase
        estimated_phase = estimate_phase_from_counts_arbitrary(counts, num_counting_qubits)
        print(f"Estimated Phase ({num_counting_qubits}-qubit):", estimated_phase)
        print(f"Actual Phase ({num_counting_qubits}-qubit):", theta)
        print()

#----------Qiskit's inbuilt Phase Estimation-----------#
    print("\nQiskit inbuilt PhaseEstimation:")
    for num_counting_qubits in qubit_counts:
        print(f"\n{num_counting_qubits}-Qubit Phase Estimation Circuit:")
        qc_pe = phase_estimation_qiskit(num_counting_qubits, U, theta)
        print(qc_pe.draw())
        # Statevector
        initial_state = Statevector.from_label('0' * num_counting_qubits + '1')
        statevector = initial_state.evolve(qc_pe.remove_final_measurements(inplace=False))
        print(f"\n{num_counting_qubits}-Qubit Statevector:")
        print_formatted_statevector(statevector, num_counting_qubits)
        # Counts
        counts = run_and_get_counts(qc_pe, shots=3000)
        print(f"\n{num_counting_qubits}-Qubit Measurement Counts:", counts)
        # Phase
        estimated_phase = estimate_phase_from_counts_arbitrary(counts, num_counting_qubits)
        print(f"Estimated Phase ({num_counting_qubits}-qubit):", estimated_phase)
        print(f"Actual Phase ({num_counting_qubits}-qubit):", theta)
        print()