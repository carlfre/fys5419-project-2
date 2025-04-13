from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.linalg import expm
from qiskit.circuit import Gate, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
import numpy as np
from vqe.vqe import VQE

def run_and_get_counts(circuit: QuantumCircuit, shots=1024) -> dict:
    """Simulates the circuit."""
    simulator = AerSimulator(method='statevector')
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts

def get_unitary_gate_from_hamiltonian(hamiltonian: np.ndarray, time: float, label: str) -> Gate:
    """
    Converts a Hamiltonian to a unitary gate using time evolution: U = exp(-1j * H * t).
    """
    unitary_matrix = expm(-1j * hamiltonian * time)
    return UnitaryGate(unitary_matrix, label=label)

def inverse_qft(qc: QuantumCircuit, n: int):
    """Performs the inverse QFT."""
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(-np.pi / float(2**(j - k)), k, j)

def phase_estimation(unitary_gate: Gate, num_counting_qubits: int, eigenstate: QuantumRegister, main_circuit: QuantumCircuit):
    """Phase estimation."""
    counting_qubits = QuantumRegister(num_counting_qubits, name="counting")
    main_circuit.add_register(counting_qubits, ClassicalRegister(num_counting_qubits))

    main_circuit.h(counting_qubits)

    for j in range(num_counting_qubits):
        repetitions = 2**j
        for _ in range(repetitions):
            if len(eigenstate) == 1:  # 2x2 case
                main_circuit.append(unitary_gate.control(1), [counting_qubits[j], eigenstate[0]])
            elif len(eigenstate) == 2:  # 4x4 case
                main_circuit.append(unitary_gate.control(1), [counting_qubits[j], eigenstate[0], eigenstate[1]])
            else:
                raise ValueError("Eigenstate register must have 1 or 2 qubits.")

    inverse_qft(main_circuit, num_counting_qubits)

    main_circuit.measure(counting_qubits, range(num_counting_qubits))

def estimate_phase_from_counts_arbitrary(counts: dict, num_counting_qubits: int) -> float:
    """Estimates the phase."""
    estimated_phase = 0
    total_shots = sum(counts.values())

    for key, count in counts.items():
        decimal_representation = int(key, 2)
        phase_contribution = (decimal_representation / (2**num_counting_qubits)) * 2 * np.pi
        probability = count / total_shots
        estimated_phase += phase_contribution * probability

    return estimated_phase

if __name__ == "__main__":
    num_counting_qubits = 8 # Increased precision
    time = 1

    # 1. 2x2 Hamiltonian
    E1 = 1.0
    E2 = 2.0
    V11 = 0.1
    V12 = 0.2
    V21 = 0.2
    V22 = 0.3

    H_2x2 = np.array([[E1 + V11, V12], [V21, E2 + V22]])
    U_2x2 = get_unitary_gate_from_hamiltonian(H_2x2, time, "U_2x2")

    # Calculate actual eigenvalues
    actual_eigenvalues_2x2 = np.linalg.eigvals(H_2x2)
    print("Actual 2x2 Eigenvalues:", actual_eigenvalues_2x2)

    for i in range(2):
        eigenstate = QuantumRegister(1, name=f"eigenstate_2x2_{i}")
        main_circuit = QuantumCircuit(eigenstate)
        # Initialize eigenstate
        phase_estimation(U_2x2, num_counting_qubits, eigenstate, main_circuit)
        counts = run_and_get_counts(main_circuit)
        estimated_phase = estimate_phase_from_counts_arbitrary(counts, num_counting_qubits)
        estimated_eigenvalue = estimated_phase / time
        print(f"Estimated 2x2 Eigenvalue {i+1}:", estimated_eigenvalue)

    # VQE Calculation
    vqe_energy_2x2 = VQE.vqe_for_2x2_hamiltonian(H_2x2)
    print("VQE Energy 2x2:", vqe_energy_2x2)

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
    U_4x4 = get_unitary_gate_from_hamiltonian(H_4x4, time, "U_4x4")

    # Calculate actual eigenvalues
    actual_eigenvalues_4x4 = np.linalg.eigvals(H_4x4)
    print("\nActual 4x4 Eigenvalues:", actual_eigenvalues_4x4)

    for i in range(4):
        eigenstate = QuantumRegister(2, name=f"eigenstate_4x4_{i}")
        main_circuit = QuantumCircuit(eigenstate)
        # Initialize eigenstate
        phase_estimation(U_4x4, num_counting_qubits, eigenstate, main_circuit)
        counts = run_and_get_counts(main_circuit)
        estimated_phase = estimate_phase_from_counts_arbitrary(counts, num_counting_qubits)
        estimated_eigenvalue = estimated_phase / time
        print(f"Estimated 4x4 Eigenvalue {i+1}:", estimated_eigenvalue)

    # VQE Calculation
    vqe_energy_4x4 = VQE.vqe_for_4x4_hamiltonian(H_4x4)
    print("VQE Energy 4x4:", vqe_energy_4x4)