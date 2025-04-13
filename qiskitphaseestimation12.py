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
        phase_contribution = (decimal_representation / (2**num_counting_qubits)) # Phase in [0, 1)
        probability = count / total_shots
        estimated_phase += phase_contribution * probability  # Weighted average

    return estimated_phase * 2 * np.pi   #Convert to radians 


if __name__ == "__main__":
    # Define the unitary gate and its known phase
    theta = np.pi  # Example phase (replace with your actual phase)
    U = get_u(theta)  # Create the unitary gate

    # Dictionary to map num_qubits to phase estimation functions
    phase_estimation_funcs = {
        1: phase_estimation_1q,
        2: phase_estimation_2q
    }

    # Loop over 1 and 2 qubits
    for num_qubits in [1, 2]:
        
        # Create phase estimation circuit
        phase_estimation_func = phase_estimation_funcs[num_qubits]
        qc_pe = phase_estimation_func(U, theta)
        print(f"{num_qubits}-Qubit Phase Estimation Circuit:")
        print(qc_pe.draw())
        
        # Simulate and get statevector
        simulator = AerSimulator(method='statevector')
        qc_pe.save_statevector()
        compiled_circuit = transpile(qc_pe, simulator)
        job = simulator.run(compiled_circuit, shots=3000)
        result = job.result()
        statevector = result.get_statevector()
        print(f"\n{num_qubits}-Qubit Statevector:")
        print_formatted_statevector(statevector, num_qubits)
        
        # Simulate and get measurement counts
        counts = run_and_get_counts(qc_pe, shots=3000)
        print(f"\n{num_qubits}-Qubit Measurement Counts:", counts)
        
        # Extract estimated phase
        estimated_phase = estimate_phase_from_counts_specific(counts, num_qubits)
        print(f"Estimated Phase ({num_qubits}-qubit):", estimated_phase)
        print(f"Actual Phase ({num_qubits}-qubit):", theta)
        print()


#----------Qiskit's inbuilt Phase Estimation-----------#
    print("\nQiskit inbuilt PhaseEstimation:")
    for num_qubits in [1, 2]:
        # Create phase estimation circuit
        qc_pe = phase_estimation_qiskit(num_qubits, U, theta)
        print(f"{num_qubits}-Qubit Phase Estimation Circuit:")
        print(qc_pe.draw())
        # Simulate statevector (without measurement)
        qc_state = phase_estimation_qiskit(num_qubits, U, theta)
        qc_state.remove_final_measurements()  # Remove measurements
        simulator = AerSimulator(method='statevector')
        qc_state.save_statevector()
        compiled_circuit = transpile(qc_state, simulator)
        job = simulator.run(compiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        print(f"\n{num_qubits}-Qubit Statevector:")
        print_formatted_statevector(statevector, num_qubits)
        # Simulate counts
        counts = run_and_get_counts(qc_pe, shots=3000)
        print(f"\n{num_qubits}-Qubit Measurement Counts:", counts)
        # Extract estimated phase
        estimated_phase = estimate_phase_from_counts_specific(counts, num_qubits)
        print(f"Estimated Phase ({num_qubits}-qubit):", estimated_phase)
        print(f"Actual Phase ({num_qubits}-qubit):", theta)
        print()